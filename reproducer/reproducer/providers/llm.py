"""Provider-agnostic LLM wrapper (the only module LLM nodes may call).

Backends:
  OllamaLLM        native Ollama /api/chat (local GPU)
  OpenAICompatLLM  any OpenAI-compatible /chat/completions (OpenAI, Ollama's /v1, ...)
  FakeLLM          deterministic, for tests

Selection is config-only: REPRODUCER_LLM_<NODE> or REPRODUCER_LLM = "<backend>:<model>".
MeteredLLM enforces a shared token / wall-clock Budget and records one CallRecord per call
(prompt stored only as sha256).

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
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol, runtime_checkable
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict

BACKENDS = ("ollama", "openai", "fake")
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
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
    """OpenAI-compatible POST {base_url}/chat/completions."""

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

    def __repr__(self) -> str:
        return f"OpenAICompatLLM(base_url={self.base_url!r}, model={self.model!r})"

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


class FakeLLM:
    """Deterministic test double. Tokens = whitespace word counts."""

    backend = "fake"
    model = "fake"

    def __init__(self, responses: list[str] | Callable[[str, str], str]) -> None:
        self._responses: list[str] | None = None if callable(responses) else list(responses)
        self._fn = responses if callable(responses) else None
        self.prompts: list[tuple[str, str]] = []
        self._lock = threading.Lock()

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


class CallRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    node: str
    backend: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    prompt_sha256: str


def prompt_sha256(system: str, prompt: str) -> str:
    return hashlib.sha256(f"{system}\x00{prompt}".encode()).hexdigest()


class MeteredLLM:
    """Wraps an LLM: enforces the Budget and records one CallRecord per successful call."""

    def __init__(self, inner: LLM, budget: Budget, node: str) -> None:
        self.inner = inner
        self.budget = budget
        self.node = node
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
        )
        with self._lock:
            self._records.append(record)
        return resp

    def records(self) -> tuple[CallRecord, ...]:
        with self._lock:
            return tuple(self._records)


# --------------------------------------------------------------------------- config


def parse_spec(spec: str) -> tuple[str, str]:
    """'ollama:qwen2.5-coder:7b' -> ('ollama', 'qwen2.5-coder:7b'). Splits on the FIRST colon."""
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
        return OllamaLLM(host=host, model=model, **kwargs)
    if backend == "openai":
        base_url = env.get("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL
        if _points_to_ollama(base_url, env) and env.get(ALLOW_OLLAMA_COMPAT_ENV) != "1":
            raise ValueError(
                f"OPENAI_BASE_URL {base_url!r} points to Ollama: use the native 'ollama:' "
                "backend (num_ctx cannot be set via the OpenAI-compatible endpoint)"
            )
        return OpenAICompatLLM(base_url=base_url, model=model, env=env)
    return FakeLLM(lambda system, prompt: "{}")


def llm_for_node(node: str, budget: Budget, env: Mapping[str, str] | None = None) -> MeteredLLM:
    """Per-node LLM: REPRODUCER_LLM_<NODE> overrides REPRODUCER_LLM. Budget is shared."""
    env = os.environ if env is None else env
    node_var = f"REPRODUCER_LLM_{node.upper()}"
    spec = env.get(node_var) or env.get("REPRODUCER_LLM")
    if not spec:
        raise ValueError(
            f"no LLM configured for node {node!r}: set {node_var} or REPRODUCER_LLM "
            "(e.g. 'ollama:qwen2.5-coder:7b' or 'openai:gpt-4o-mini')"
        )
    return MeteredLLM(make_llm(spec, env), budget, node)
