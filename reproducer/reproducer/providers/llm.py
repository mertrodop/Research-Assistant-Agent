"""Provider-agnostic LLM wrapper (the only module LLM nodes may call).

Backends:
  OllamaOpenAILLM  local Ollama through its OpenAI-compatible {OLLAMA_HOST}/v1 (default for
                   'ollama:'; telemetry backend "ollama"; unload() on the native endpoint)
  OllamaLLM        native Ollama /api/chat (fallback: REPRODUCER_OLLAMA_API=native)
  OpenAICompatLLM  any OpenAI-compatible /chat/completions (OpenAI, OpenRouter, ...)
  FakeLLM          deterministic, for tests

Selection is config-only: REPRODUCER_LLM_<NODE> or REPRODUCER_LLM = "<backend>:<model>",
backend one of ollama | openai | openrouter | fake, e.g.
  REPRODUCER_LLM_CRITIC=openrouter:meta-llama/llama-3.1-8b-instruct:free
Env per backend:
  ollama      OLLAMA_HOST, OLLAMA_NUM_CTX (default 8192; a call whose estimated size exceeds it
              raises ContextTooLong before any request), REPRODUCER_OLLAMA_API (openai|native,
              default openai), OLLAMA_JSON_MODE (schema|object|none, openai API only).
              Via /v1 the server context is OLLAMA_CONTEXT_LENGTH (set it on the Ollama
              service to match OLLAMA_NUM_CTX): /v1 cannot send num_ctx, and the server
              silently truncates longer prompts, so the client-side guard is what catches them.
  openai      OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_JSON_MODE (schema|object|none)
  openrouter  OPENROUTER_BASE_URL, OPENROUTER_API_KEY (required), OPENROUTER_JSON_MODE
MeteredLLM enforces a shared token / wall-clock Budget and records one CallRecord per call
(prompt stored only as sha256).

Function calling (chat(messages, tools); OpenAICompatLLM, OllamaOpenAILLM, MeteredLLM, FakeLLM
with a tool_script) uses the OpenAI format, verified live 2026-11-27:
  request   tools=[{"type": "function", "function": {name, description, parameters}}],
            tool_choice "auto"; follow-ups append the assistant turn with its tool_calls and one
            {"role": "tool", "tool_call_id", "content"} per call
  OpenRouter anthropic/claude-haiku-5.5: message.tool_calls [{id "toolu_...", type
            "function", function {name, arguments: JSON string}}], content null,
            finish_reason "tool_calls"
  Ollama 0.18.0 /v1 qwen2.5-coder:7b: no tool_calls; the call arrives as bare JSON
            '{"name", "arguments"}' in content (finish_reason "stop"): recovered only when it
            is exactly that shape with a declared tool name (ChatResponse.from_content)
Backends without chat (native OllamaLLM) or a provider 400/404 naming tools -> ToolsUnsupported;
an uninterpretable reply -> ToolProtocolError.

HTTP is stdlib only. Transient failures (429, 5xx, timeouts, connection errors) are retried
with exponential backoff, max 3 attempts. Content/validation retries are NOT done here.
API keys are read from the environment at call time; they are never stored, logged,
recorded, or put in exception messages.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol, runtime_checkable
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict

from reproducer.domain.telemetry import CallRecord

BACKENDS = ("ollama", "openai", "openrouter", "fake")
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Ollama accepts a JSON schema object in `format` since 0.5.0; older versions only "json".
_OLLAMA_SCHEMA_MIN_VERSION = (0, 5, 0)
_EXCERPT_CHARS = 500
_MAX_ATTEMPTS = 3


class LLMError(RuntimeError):
    """Non-recoverable backend error (after transient retries are exhausted)."""

    def __init__(self, message: str, *, status: int | None = None, body: str = "") -> None:
        self.status = status
        self.body = body[:_EXCERPT_CHARS]
        detail = f" (HTTP {status})" if status is not None else ""
        excerpt = f": {self.body}" if self.body else ""
        super().__init__(f"{message}{detail}{excerpt}")


class ContextTooLong(LLMError):
    """Estimated prompt + max_tokens would not fit the model context (no call was made)."""


# Conservative chars-per-token estimate used for the pre-call context check.
CHARS_PER_TOKEN_ESTIMATE = 3.5


def check_context(system: str, prompt: str, max_tokens: int, num_ctx: int, var: str) -> None:
    """Raise ContextTooLong (no request made) when the estimated size exceeds num_ctx."""
    estimate = len(system + prompt) / CHARS_PER_TOKEN_ESTIMATE + max_tokens
    if estimate > num_ctx:
        raise ContextTooLong(
            f"estimated {estimate:.0f} tokens (prompt ~{len(system + prompt)} chars / "
            f"{CHARS_PER_TOKEN_ESTIMATE} + max_tokens {max_tokens}) > num_ctx {num_ctx}; "
            f"raise {var} or shrink the prompt"
        )


class BudgetExceeded(RuntimeError):
    """Token or wall-clock budget would be exceeded by this call."""


class LLMResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    input_tokens: int
    output_tokens: int
    model: str
    backend: str
    latency_s: float


class ToolsUnsupported(LLMError):
    """The backend / model does not do function calling (no request made, or the provider
    refused the `tools` parameter)."""


class ToolProtocolError(LLMError):
    """A function-calling response the client cannot interpret (shape, tool_calls)."""


class ToolCall(BaseModel):
    """One OpenAI function call. `arguments` is the raw JSON string the model sent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str
    arguments: str


class ChatResponse(BaseModel):
    """One chat/completions turn with tools. from_content: the tool call was recovered from
    a bare '{"name": <declared tool>, "arguments": {...}}' message content (models such as
    qwen2.5-coder on Ollama answer that way instead of emitting tool_calls)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    input_tokens: int
    output_tokens: int
    model: str
    backend: str
    latency_s: float
    finish_reason: str | None = None
    from_content: bool = False


def assistant_message(resp: ChatResponse) -> dict[str, Any]:
    """The assistant turn to append to `messages` (OpenAI format) before the tool results."""
    msg: dict[str, Any] = {"role": "assistant", "content": resp.text or ""}
    if resp.tool_calls:
        msg["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in resp.tool_calls
        ]
    return msg


def tool_message(call_id: str, content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _tool_names(tools: list[dict[str, Any]]) -> set[str]:
    return {str(t.get("function", {}).get("name")) for t in tools if isinstance(t, dict)}


_CONTENT_FENCE_RE = re.compile(r"^```[ \t]*(?:json)?[ \t]*\n?(.*?)\n?```$", re.DOTALL | re.I)


def tool_call_from_content(text: str | None, tools: list[dict[str, Any]]) -> ToolCall | None:
    """A bare '{"name": <declared tool>, "arguments": {...}}' content (optionally fenced) as a
    ToolCall; None for anything else (narrow on purpose: no free-text parsing)."""
    if not text:
        return None
    s = text.strip()
    m = _CONTENT_FENCE_RE.match(s)
    s = m.group(1).strip() if m else s
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or set(obj) != {"name", "arguments"}:
        return None
    name, args = obj["name"], obj["arguments"]
    if not isinstance(name, str) or name not in _tool_names(tools):
        return None
    if isinstance(args, str):
        args_json = args
    elif isinstance(args, dict):
        args_json = json.dumps(args)
    else:
        return None
    return ToolCall(id="content_0", name=name, arguments=args_json)


def _parse_tool_calls(raw: Any) -> tuple[ToolCall, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ToolProtocolError("tool_calls is not a list", body=json.dumps(raw)[:500])
    out = []
    for i, c in enumerate(raw):
        fn = c.get("function") if isinstance(c, dict) else None
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            raise ToolProtocolError("malformed tool call", body=json.dumps(c)[:500])
        args = fn.get("arguments", "")
        if isinstance(args, dict):  # some servers send an object instead of a JSON string
            args = json.dumps(args)
        if not isinstance(args, str):
            raise ToolProtocolError("tool call arguments not a string", body=json.dumps(c)[:500])
        cid = c.get("id") if isinstance(c.get("id"), str) and c.get("id") else f"call_{i}"
        out.append(ToolCall(id=cid, name=fn["name"], arguments=args))
    return tuple(out)


_TOOLS_REFUSED_RE = re.compile(r"tool", re.IGNORECASE)


def chat_text(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    """The serialized request body parts a chat call sends (context estimate, prompt hash)."""
    return json.dumps(messages, sort_keys=True) + json.dumps(tools, sort_keys=True)


@runtime_checkable
class LLM(Protocol):
    def complete(
        self,
        system: str,
        prompt: str,
        *,
        max_tokens: int,
        json_schema: dict | None = None,
    ) -> LLMResponse: ...


# --------------------------------------------------------------------------- HTTP


def _backoff_sleep(seconds: float) -> None:
    time.sleep(seconds)  # module-level indirection so tests can patch backoff only


def _is_timeout(exc: BaseException) -> bool:
    # socket.timeout is an alias of TimeoutError on Python >= 3.10
    return isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError)


def _request_json(
    url: str,
    payload: Any | None,
    headers: Mapping[str, str],
    timeout_s: float,
    *,
    max_attempts: int = _MAX_ATTEMPTS,
    base_delay_s: float = 0.5,
    sleep: Callable[[float], None] | None = None,
) -> Any:
    """POST (payload given) or GET (payload None) JSON with transient retries.

    Retries: HTTP 429/5xx, timeouts, connection errors. Other 4xx: raise immediately.
    Error messages contain the URL, status and a response-body excerpt, never headers.
    """
    sleep = _backoff_sleep if sleep is None else sleep
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    all_headers = {"Accept": "application/json", **headers}
    if data is not None:
        all_headers["Content-Type"] = "application/json"
    last: LLMError | None = None
    for attempt in range(max_attempts):
        req = urllib.request.Request(
            url, data=data, headers=all_headers, method="GET" if data is None else "POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read()
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                raise LLMError(
                    f"invalid JSON from {url}", body=raw.decode("utf-8", "replace")
                ) from e
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace") if e.fp is not None else ""
            err = LLMError(f"request to {url} failed", status=e.code, body=body)
            if e.code != 429 and e.code < 500:
                raise err from None
            last = err
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            kind = "timed out" if _is_timeout(e) else "connection error"
            last = LLMError(f"request to {url} {kind}: {type(e).__name__}")
        if attempt + 1 < max_attempts:
            sleep(base_delay_s * 2**attempt)
    assert last is not None
    raise LLMError(f"giving up after {max_attempts} attempts; last error: {last}") from None


def _parse_version(text: str) -> tuple[int, ...] | None:
    m = re.match(r"^\s*v?(\d+)\.(\d+)\.(\d+)", text)
    return tuple(int(x) for x in m.groups()) if m else None


def _as_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


# --------------------------------------------------------------------------- backends


class OllamaLLM:
    """Native Ollama chat API (POST /api/chat, stream=false)."""

    backend = "ollama"

    def __init__(
        self,
        host: str,
        model: str,
        num_ctx: int = 8192,
        keep_alive: str | int = "5m",
        temperature: float = 0.0,
        timeout_s: float = 300,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.temperature = temperature
        self.timeout_s = timeout_s
        self._schema_ok: bool | None = None
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"OllamaLLM(host={self.host!r}, model={self.model!r})"

    def supports_schema_format(self) -> bool:
        """True if the server's version accepts a JSON-schema object in `format` (cached)."""
        with self._lock:
            if self._schema_ok is None:
                try:
                    info = _request_json(
                        f"{self.host}/api/version", None, {}, min(self.timeout_s, 10)
                    )
                    ver = _parse_version(str(info.get("version", "")))
                except LLMError:
                    ver = None
                self._schema_ok = ver is not None and ver >= _OLLAMA_SCHEMA_MIN_VERSION
            return self._schema_ok

    def complete(
        self,
        system: str,
        prompt: str,
        *,
        max_tokens: int,
        json_schema: dict | None = None,
    ) -> LLMResponse:
        check_context(system, prompt, max_tokens, self.num_ctx, "OLLAMA_NUM_CTX")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
                "temperature": self.temperature,
                "seed": 0,
            },
        }
        if json_schema is not None:
            payload["format"] = json_schema if self.supports_schema_format() else "json"
        t0 = time.monotonic()
        data = _request_json(f"{self.host}/api/chat", payload, {}, self.timeout_s)
        latency = time.monotonic() - t0
        message = data.get("message") if isinstance(data, dict) else None
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise LLMError("unexpected Ollama response shape", body=json.dumps(data)[:500])
        return LLMResponse(
            text=message["content"],
            input_tokens=_as_int(data.get("prompt_eval_count")),
            output_tokens=_as_int(data.get("eval_count")),
            model=self.model,
            backend=self.backend,
            latency_s=latency,
        )

    def unload(self) -> bool:
        """Unload the model from VRAM (empty messages + keep_alive=0, per Ollama docs).

        Call only before GPU work (smoke/execute). Reloading costs ~30 s, so never unload
        between LLM calls.
        """
        payload = {"model": self.model, "messages": [], "keep_alive": 0, "stream": False}
        data = _request_json(f"{self.host}/api/chat", payload, {}, self.timeout_s)
        return isinstance(data, dict) and data.get("done_reason") == "unload"


JsonMode = Literal["schema", "object", "none"]


class OpenAICompatLLM:
    """OpenAI-compatible POST {base_url}/chat/completions.

    `backend` only labels responses/CallRecords (e.g. "openrouter"); the protocol is the same.
    """

    backend = "openai"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        json_mode: JsonMode = "schema",
        temperature: float = 0.0,
        timeout_s: float = 300,
        env: Mapping[str, str] | None = None,
        backend: str = "openai",
        num_ctx: int | None = None,
        num_ctx_var: str = "OLLAMA_NUM_CTX",
    ) -> None:
        if json_mode not in ("schema", "object", "none"):
            raise ValueError(f"json_mode must be schema|object|none, got {json_mode!r}")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.json_mode = json_mode
        self.temperature = temperature
        self.timeout_s = timeout_s
        self._env = env  # None -> os.environ at call time; the key itself is never stored
        self.backend = backend
        self.num_ctx = num_ctx  # None: no client-side context guard (the server decides)
        self._num_ctx_var = num_ctx_var

    def __repr__(self) -> str:
        return (
            f"OpenAICompatLLM(backend={self.backend!r}, base_url={self.base_url!r}, "
            f"model={self.model!r})"
        )

    def _headers(self) -> dict[str, str]:
        env = os.environ if self._env is None else self._env
        key = env.get(self.api_key_env)
        return {"Authorization": f"Bearer {key}"} if key else {}

    def complete(
        self,
        system: str,
        prompt: str,
        *,
        max_tokens: int,
        json_schema: dict | None = None,
    ) -> LLMResponse:
        if self.num_ctx is not None:
            check_context(system, prompt, max_tokens, self.num_ctx, self._num_ctx_var)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "seed": 0,
        }
        if json_schema is not None:
            if self.json_mode == "schema":
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "response", "schema": json_schema},
                }
            elif self.json_mode == "object":
                payload["response_format"] = {"type": "json_object"}
        t0 = time.monotonic()
        data = _request_json(
            f"{self.base_url}/chat/completions", payload, self._headers(), self.timeout_s
        )
        latency = time.monotonic() - t0
        try:
            text = data["choices"][0]["message"]["content"]
            if not isinstance(text, str):
                raise TypeError
        except (KeyError, IndexError, TypeError):
            raise LLMError(
                "unexpected chat/completions response shape", body=json.dumps(data)[:500]
            ) from None
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            input_tokens=_as_int(usage.get("prompt_tokens")),
            output_tokens=_as_int(usage.get("completion_tokens")),
            model=self.model,
            backend=self.backend,
            latency_s=latency,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        max_tokens: int,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> ChatResponse:
        """One OpenAI function-calling turn: `messages` in OpenAI format (system / user /
        assistant with tool_calls / tool with tool_call_id), `tools` as
        [{"type": "function", "function": {name, description, parameters}}]. A provider 4xx
        naming tools -> ToolsUnsupported; an uninterpretable reply -> ToolProtocolError."""
        if self.num_ctx is not None:
            check_context("", chat_text(messages, tools), max_tokens, self.num_ctx,
                          self._num_ctx_var)  # fmt: skip
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "seed": 0,
        }
        t0 = time.monotonic()
        try:
            data = _request_json(
                f"{self.base_url}/chat/completions", payload, self._headers(), self.timeout_s
            )
        except LLMError as e:
            if (
                e.status in (400, 404)
                and _TOOLS_REFUSED_RE.search(e.body)
                and "context" not in (e.body.lower())
            ):
                raise ToolsUnsupported(
                    f"{self.backend}:{self.model} refused tools", status=e.status, body=e.body
                ) from None
            raise
        latency = time.monotonic() - t0
        try:
            choice = data["choices"][0]
            message = choice["message"]
            if not isinstance(message, dict):
                raise TypeError
        except (KeyError, IndexError, TypeError):
            raise ToolProtocolError(
                "unexpected chat/completions response shape", body=json.dumps(data)[:500]
            ) from None
        text = message.get("content")
        text = text if isinstance(text, str) else None
        calls = _parse_tool_calls(message.get("tool_calls"))
        from_content = False
        if not calls:
            recovered = tool_call_from_content(text, tools)
            if recovered is not None:
                calls, text, from_content = (recovered,), None, True
        usage = data.get("usage") or {}
        return ChatResponse(
            text=text,
            tool_calls=calls,
            input_tokens=_as_int(usage.get("prompt_tokens")),
            output_tokens=_as_int(usage.get("completion_tokens")),
            model=self.model,
            backend=self.backend,
            latency_s=latency,
            finish_reason=choice.get("finish_reason")
            if isinstance(choice.get("finish_reason"), str)
            else None,
            from_content=from_content,
        )


# (host, model) -> server context length from /api/ps; None = not known (checked, absent)
_SERVER_CTX: dict[tuple[str, str], int | None] = {}
_SERVER_CTX_LOCK = threading.Lock()


def _ps_context_length(host: str, model: str, timeout_s: float) -> int | None:
    """context_length of `model` in GET /api/ps (Ollama 0.18.0: listed only while loaded).
    None when not loaded, absent or the endpoint fails (one attempt, no retries)."""
    try:
        data = _request_json(f"{host}/api/ps", None, {}, timeout_s, max_attempts=1)
    except LLMError:
        return None
    models = data.get("models") if isinstance(data, dict) else None
    for m in models if isinstance(models, list) else ():
        if not isinstance(m, dict):
            continue
        names = {m.get("name"), m.get("model")}
        if model in names or f"{model}:latest" in names:
            n = m.get("context_length")
            return n if isinstance(n, int) and not isinstance(n, bool) and n > 0 else None
    return None


class OllamaOpenAILLM(OpenAICompatLLM):
    """Local Ollama through its OpenAI-compatible API ({host}/v1/chat/completions).

    Telemetry backend stays "ollama" (local tokens). The context window is the server's
    OLLAMA_CONTEXT_LENGTH; the client-side guard uses num_ctx (OLLAMA_NUM_CTX) because the
    server truncates an oversized prompt silently. On the first call per (host, model) the
    server's context_length is read from /api/ps (before the call; if the model is not loaded
    yet, once more right after it): when it is below num_ctx the guard uses the server value
    and one warning is printed. unload() uses the native endpoint.
    """

    def __init__(
        self,
        host: str,
        model: str,
        num_ctx: int = 8192,
        json_mode: JsonMode = "schema",
        temperature: float = 0.0,
        timeout_s: float = 300,
    ) -> None:
        self.host = host.rstrip("/")
        super().__init__(
            base_url=f"{self.host}/v1",
            model=model,
            api_key_env="OLLAMA_API_KEY",  # unset locally: no Authorization header
            json_mode=json_mode,
            temperature=temperature,
            timeout_s=timeout_s,
            backend="ollama",
            num_ctx=num_ctx,
        )

    def __repr__(self) -> str:
        return f"OllamaOpenAILLM(host={self.host!r}, model={self.model!r})"

    def _sync_server_ctx(self, *, final: bool) -> bool:
        """Apply the cached / probed server context length. Returns True once it is settled
        (known, or `final` and still unknown: no further probes)."""
        key = (self.host, self.model)
        with _SERVER_CTX_LOCK:
            if key not in _SERVER_CTX:
                n = _ps_context_length(self.host, self.model, min(self.timeout_s, 10))
                if n is None and not final:
                    return False  # not loaded yet: probe again after the first call
                _SERVER_CTX[key] = n
                if n is not None and self.num_ctx is not None and n < self.num_ctx:
                    print(
                        f"warning: ollama server context_length {n} < OLLAMA_NUM_CTX "
                        f"{self.num_ctx} for {self.model}: the context guard uses {n} "
                        "(raise OLLAMA_CONTEXT_LENGTH on the Ollama service)",
                        file=sys.stderr,
                    )
            n = _SERVER_CTX[key]
        if n is not None and self.num_ctx is not None and n < self.num_ctx:
            self.num_ctx = n
        return True

    def complete(
        self,
        system: str,
        prompt: str,
        *,
        max_tokens: int,
        json_schema: dict | None = None,
    ) -> LLMResponse:
        settled = self._sync_server_ctx(final=False)
        resp = super().complete(system, prompt, max_tokens=max_tokens, json_schema=json_schema)
        if not settled:
            self._sync_server_ctx(final=True)
        return resp

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        max_tokens: int,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> ChatResponse:
        settled = self._sync_server_ctx(final=False)
        resp = super().chat(messages, tools, max_tokens=max_tokens, tool_choice=tool_choice)
        if not settled:
            self._sync_server_ctx(final=True)
        return resp

    def unload(self) -> bool:
        """Native /api/chat keep_alive=0 (the OpenAI API has no unload)."""
        return OllamaLLM(self.host, self.model, timeout_s=self.timeout_s).unload()


FakeTurn = Any  # str (text) | (name, args) | list of (name, args) | ChatResponse
ChatScript = list[FakeTurn] | Callable[[list[dict[str, Any]], list[dict[str, Any]]], FakeTurn]


def _fake_chat_response(turn: FakeTurn, n_in: int) -> ChatResponse:
    if isinstance(turn, ChatResponse):
        return turn
    text: str | None = None
    calls: list[ToolCall] = []
    if isinstance(turn, str):
        text = turn
    else:
        pairs = [turn] if isinstance(turn, tuple) else list(turn)
        for i, (name, args) in enumerate(pairs):
            raw = args if isinstance(args, str) else json.dumps(args)
            calls.append(ToolCall(id=f"fake_{i}", name=name, arguments=raw))
    out = len((text or "").split()) + sum(len(c.arguments.split()) for c in calls)
    return ChatResponse(
        text=text, tool_calls=tuple(calls), input_tokens=n_in, output_tokens=out,
        model=FakeLLM.model, backend=FakeLLM.backend, latency_s=0.0,
        finish_reason="tool_calls" if calls else "stop",
    )  # fmt: skip


class FakeLLM:
    """Deterministic test double. Tokens = whitespace word counts.

    tool_script: scripted chat() turns (list, or fn(messages, tools) -> turn), each a str
    (text answer), a (tool_name, args) pair, a list of pairs (parallel calls) or a
    ChatResponse. Without a tool_script, chat() raises ToolsUnsupported."""

    backend = "fake"
    model = "fake"

    def __init__(
        self,
        responses: list[str] | Callable[[str, str], str],
        tool_script: ChatScript | None = None,
    ) -> None:
        self._responses: list[str] | None = None if callable(responses) else list(responses)
        self._fn = responses if callable(responses) else None
        self.prompts: list[tuple[str, str]] = []
        self._script: list[FakeTurn] | None = (
            None if tool_script is None or callable(tool_script) else list(tool_script)
        )
        self._script_fn = tool_script if callable(tool_script) else None
        self.chats: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
        self._lock = threading.Lock()

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        max_tokens: int,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> ChatResponse:
        with self._lock:
            if self._script is None and self._script_fn is None:
                raise ToolsUnsupported("FakeLLM: no tool_script (function calling unsupported)")
            snapshot = json.loads(json.dumps(messages))
            self.chats.append((snapshot, tools))
            if self._script_fn is not None:
                turn = self._script_fn(snapshot, tools)
            else:
                assert self._script is not None
                if not self._script:
                    raise LLMError("FakeLLM: scripted tool turns exhausted")
                turn = self._script.pop(0)
        return _fake_chat_response(turn, len(chat_text(messages, tools).split()))

    def complete(
        self,
        system: str,
        prompt: str,
        *,
        max_tokens: int,
        json_schema: dict | None = None,
    ) -> LLMResponse:
        with self._lock:
            self.prompts.append((system, prompt))
            if self._fn is not None:
                text = self._fn(system, prompt)
            else:
                assert self._responses is not None
                if not self._responses:
                    raise LLMError("FakeLLM: scripted responses exhausted")
                text = self._responses.pop(0)
        return LLMResponse(
            text=text,
            input_tokens=len(system.split()) + len(prompt.split()),
            output_tokens=len(text.split()),
            model=self.model,
            backend=self.backend,
            latency_s=0.0,
        )


# --------------------------------------------------------------------------- metering


class Budget:
    """Shared, thread-safe token + wall-clock budget.

    The pre-check reserves `max_tokens` so concurrent calls can never jointly exceed the
    cap; after the call the reservation is replaced by the actual input+output tokens.
    """

    def __init__(
        self,
        max_tokens_total: int = 200_000,
        max_wall_s: float = 3600,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_tokens_total = max_tokens_total
        self.max_wall_s = max_wall_s
        self._clock = clock
        self._start = clock()
        self._used = 0
        self._reserved = 0
        self._lock = threading.Lock()

    @property
    def used_tokens(self) -> int:
        with self._lock:
            return self._used

    @property
    def elapsed_s(self) -> float:
        return self._clock() - self._start

    @property
    def remaining_tokens(self) -> int:
        with self._lock:
            return max(0, self.max_tokens_total - self._used)

    def reserve(self, max_tokens: int) -> None:
        with self._lock:
            elapsed = self._clock() - self._start
            if elapsed > self.max_wall_s:
                raise BudgetExceeded(
                    f"wall-clock budget exceeded: {elapsed:.1f}s > {self.max_wall_s}s"
                )
            if self._used + self._reserved + max_tokens > self.max_tokens_total:
                raise BudgetExceeded(
                    f"token budget exceeded: used={self._used} reserved={self._reserved} "
                    f"+ max_tokens={max_tokens} > cap={self.max_tokens_total}"
                )
            self._reserved += max_tokens

    def commit(self, reserved: int, actual_tokens: int) -> None:
        with self._lock:
            self._reserved -= reserved
            self._used += actual_tokens

    def release(self, reserved: int) -> None:
        with self._lock:
            self._reserved -= reserved


def prompt_sha256(system: str, prompt: str) -> str:
    return hashlib.sha256(f"{system}\x00{prompt}".encode()).hexdigest()


class MeteredLLM:
    """Wraps an LLM: enforces the Budget and records one CallRecord per successful call."""

    def __init__(
        self,
        inner: LLM,
        budget: Budget,
        node: str,
        on_record: Callable[[CallRecord], None] | None = None,
    ) -> None:
        self.inner = inner
        self.budget = budget
        self.node = node
        self.on_record = on_record  # e.g. RunStore.append_call_record; errors propagate
        # section titles context eviction dropped from the prompts now being sent (set by
        # graph.eviction.call_with_eviction; copied into each CallRecord)
        self.evicted: tuple[str, ...] = ()
        self._records: list[CallRecord] = []
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"MeteredLLM(node={self.node!r}, inner={self.inner!r})"

    def complete(
        self,
        system: str,
        prompt: str,
        *,
        max_tokens: int,
        json_schema: dict | None = None,
    ) -> LLMResponse:
        self.budget.reserve(max_tokens)
        try:
            resp = self.inner.complete(
                system, prompt, max_tokens=max_tokens, json_schema=json_schema
            )
        except BaseException:
            self.budget.release(max_tokens)
            raise
        self.budget.commit(max_tokens, resp.input_tokens + resp.output_tokens)
        record = CallRecord(
            node=self.node,
            backend=resp.backend,
            model=resp.model,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            latency_s=resp.latency_s,
            prompt_sha256=prompt_sha256(system, prompt),
            evicted=self.evicted,
        )
        self._record(record)
        return resp

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        max_tokens: int,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> ChatResponse:
        """Metered function-calling turn. An inner LLM without chat() -> ToolsUnsupported
        (no budget reserved). The CallRecord hashes the serialized messages + tools."""
        chat = getattr(self.inner, "chat", None)
        if chat is None:
            raise ToolsUnsupported(f"{self.inner!r} has no function-calling support")
        self.budget.reserve(max_tokens)
        try:
            resp: ChatResponse = chat(
                messages, tools, max_tokens=max_tokens, tool_choice=tool_choice
            )
        except BaseException:
            self.budget.release(max_tokens)
            raise
        self.budget.commit(max_tokens, resp.input_tokens + resp.output_tokens)
        self._record(
            CallRecord(
                node=self.node,
                backend=resp.backend,
                model=resp.model,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                latency_s=resp.latency_s,
                prompt_sha256=prompt_sha256("", chat_text(messages, tools)),
                evicted=self.evicted,
            )
        )
        return resp

    def _record(self, record: CallRecord) -> None:
        with self._lock:
            self._records.append(record)
        if self.on_record is not None:
            self.on_record(record)

    def records(self) -> tuple[CallRecord, ...]:
        with self._lock:
            return tuple(self._records)


# --------------------------------------------------------------------------- config


def parse_spec(spec: str) -> tuple[str, str]:
    """'ollama:qwen2.5-coder:7b' -> ('ollama', 'qwen2.5-coder:7b'). Splits on the FIRST colon,
    so 'openrouter:meta-llama/llama-3.1-8b-instruct:free' keeps '/' and ':free' in the model."""
    backend, sep, model = spec.strip().partition(":")
    if not sep or not model:
        raise ValueError(f"LLM spec must be '<backend>:<model>', got {spec!r}")
    if backend not in BACKENDS:
        raise ValueError(f"unknown LLM backend {backend!r} in {spec!r}; expected one of {BACKENDS}")
    return backend, model


def _normalize_host(host: str) -> str:
    host = host.strip()
    return host if "://" in host else f"http://{host}"


OLLAMA_DEFAULT_PORT = 11434
ALLOW_OLLAMA_COMPAT_ENV = "REPRODUCER_ALLOW_OLLAMA_COMPAT"


def _host_port(url: str) -> tuple[str, int | None]:
    parts = urlsplit(_normalize_host(url))
    port = parts.port or {"http": 80, "https": 443}.get(parts.scheme)
    return (parts.hostname or "", port)


def _points_to_ollama(base_url: str, env: Mapping[str, str]) -> bool:
    """D30: True if an OpenAI-compatible base URL is actually an Ollama server."""
    target = _host_port(base_url)
    ollama = _host_port(env.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST)
    return target[1] == OLLAMA_DEFAULT_PORT or target == ollama


def _json_mode(env: Mapping[str, str], var: str) -> JsonMode:
    value = (env.get(var) or "schema").strip()
    if value not in ("schema", "object", "none"):
        raise ValueError(f"{var} must be schema|object|none, got {value!r}")
    return value  # type: ignore[return-value]


OLLAMA_API_ENV = "REPRODUCER_OLLAMA_API"
OLLAMA_APIS = ("openai", "native")


def make_llm(spec: str, env: Mapping[str, str]) -> LLM:
    backend, model = parse_spec(spec)
    if backend == "ollama":
        kwargs: dict[str, Any] = {}
        if env.get("OLLAMA_NUM_CTX"):
            try:
                kwargs["num_ctx"] = int(env["OLLAMA_NUM_CTX"])
            except ValueError:
                raise ValueError(
                    f"OLLAMA_NUM_CTX must be an integer, got {env['OLLAMA_NUM_CTX']!r}"
                ) from None
        host = _normalize_host(env.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST)
        api = (env.get(OLLAMA_API_ENV) or "openai").strip()
        if api not in OLLAMA_APIS:
            raise ValueError(f"{OLLAMA_API_ENV} must be openai|native, got {api!r}")
        if api == "native":
            return OllamaLLM(host=host, model=model, **kwargs)
        return OllamaOpenAILLM(
            host=host, model=model, json_mode=_json_mode(env, "OLLAMA_JSON_MODE"), **kwargs
        )
    if backend == "openai":
        base_url = env.get("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL
        if _points_to_ollama(base_url, env) and env.get(ALLOW_OLLAMA_COMPAT_ENV) != "1":
            raise ValueError(
                f"OPENAI_BASE_URL {base_url!r} points to Ollama: use the native 'ollama:' "
                "backend (num_ctx cannot be set via the OpenAI-compatible endpoint)"
            )
        return OpenAICompatLLM(
            base_url=base_url,
            model=model,
            env=env,
            json_mode=_json_mode(env, "OPENAI_JSON_MODE"),
        )
    if backend == "openrouter":
        if not env.get("OPENROUTER_API_KEY"):
            raise ValueError(
                "OPENROUTER_API_KEY is not set: export it to use the 'openrouter:' backend"
            )
        return OpenAICompatLLM(
            base_url=env.get("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL,
            model=model,
            api_key_env="OPENROUTER_API_KEY",
            json_mode=_json_mode(env, "OPENROUTER_JSON_MODE"),
            env=env,
            backend="openrouter",
        )
    return FakeLLM(lambda system, prompt: "{}")


def llm_for_node(node: str, budget: Budget, env: Mapping[str, str] | None = None) -> MeteredLLM:
    """Per-node LLM: REPRODUCER_LLM_<NODE> overrides REPRODUCER_LLM. Budget is shared."""
    env = os.environ if env is None else env
    node_var = f"REPRODUCER_LLM_{node.upper()}"
    spec = env.get(node_var) or env.get("REPRODUCER_LLM")
    if not spec:
        raise ValueError(
            f"no LLM configured for node {node!r}: set {node_var} or REPRODUCER_LLM "
            f"as '<backend>:<model>' with backend in {BACKENDS} (e.g. 'ollama:qwen2.5-coder:7b', "
            "'openai:gpt-4o-mini' or 'openrouter:meta-llama/llama-3.1-8b-instruct:free')"
        )
    return MeteredLLM(make_llm(spec, env), budget, node)
