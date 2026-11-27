from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Protocol

import httpx

from ..config import ModelSettings


USAGE_FIELDS = ("calls", "prompt_tokens", "completion_tokens", "total_tokens", "cost")
USAGE_LOG = Path(os.environ.get("RA_USAGE_LOG", ".cache/llm_usage.jsonl"))


class VisionError(RuntimeError):
    """Raised when the model endpoint cannot answer."""


class LlmClient(Protocol):
    model: str

    def describe(self, image_path: str | Path, prompt: str) -> str: ...

    def complete(self, prompt: str) -> str: ...

    def search(self, prompt: str) -> tuple[str, list[str]]: ...


class NullVisionClient:
    """Used when figure interpretation is disabled."""

    model = "none"

    def describe(self, image_path: str | Path, prompt: str) -> str:
        raise VisionError("vision client disabled")

    def complete(self, prompt: str) -> str:
        raise VisionError("llm client disabled")

    def search(self, prompt: str) -> tuple[str, list[str]]:
        raise VisionError("llm client disabled")


def _data_uri(image_path: Path) -> str:
    payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


class OpenAICompatibleVisionClient:
    """Any /chat/completions endpoint that accepts image_url content parts.

    Works with OpenRouter, OpenAI, vLLM, Ollama and LM Studio; only base_url,
    model and the API key environment variable change.
    """

    def __init__(self, settings: ModelSettings, api_key: str | None = None) -> None:
        self.settings = settings
        self.model = settings.model
        self._api_key = api_key if api_key is not None else os.environ.get(
            settings.api_key_env, ""
        )
        self.usage = dict.fromkeys(USAGE_FIELDS, 0)  # running totals, one run of one process
        self._lock = threading.Lock()

    def _log_usage(self, usage: dict | None) -> None:
        """One JSON line per call in USAGE_LOG; never fails the call. Cache hits make no call, so no line."""
        if not isinstance(usage, dict):
            return
        with self._lock:
            self.usage["calls"] += 1
            for k in USAGE_FIELDS[1:]:
                self.usage[k] += usage.get(k) or 0
        line = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": self.model}
        line.update({k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cost")})
        try:
            USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
            with USAGE_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line) + "\n")
        except OSError:
            pass

    def _request(self, content: list[dict] | str) -> dict:
        if not self._api_key:
            raise VisionError(
                f"missing API key: set {self.settings.api_key_env} in the environment or .env"
            )
        body = {
            "model": self.model,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = httpx.post(
                f"{self.settings.base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=body,
                timeout=self.settings.timeout_s,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            raise VisionError(
                f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise VisionError(f"{type(exc).__name__}: {exc}") from exc

        self._log_usage(payload.get("usage"))
        return payload

    def _post(self, content: list[dict] | str) -> str:
        payload = self._request(content)
        try:
            return payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise VisionError(f"unexpected response shape: {str(payload)[:200]}") from exc

    def describe(self, image_path: str | Path, prompt: str) -> str:
        return self._post(
            [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": _data_uri(Path(image_path))},
                },
            ]
        )

    def complete(self, prompt: str) -> str:
        return self._post(prompt)

    def search(self, prompt: str) -> tuple[str, list[str]]:
        """Answer plus the source URLs the web search behind the model returned.

        Needs a model that searches the web: an OpenRouter ':online' slug (google/gemini-2.5-flash:online)
        or a search model such as perplexity/sonar. Without any citation the URL list is empty.
        """
        payload = self._request(prompt)
        try:
            message = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise VisionError(f"unexpected response shape: {str(payload)[:200]}") from exc
        urls = [a["url_citation"]["url"] for a in message.get("annotations") or [] if a.get("url_citation", {}).get("url")]
        urls += [u for u in payload.get("citations") or [] if isinstance(u, str)]
        return message.get("content") or "", list(dict.fromkeys(urls))


def resolve_api_key(settings: ModelSettings) -> str:
    """The key from the environment; a local endpoint (not OpenRouter/OpenAI) needs none."""
    key = os.environ.get(settings.api_key_env, "")
    if not key and not re.search(r"openrouter\.ai|api\.openai\.com", settings.base_url):
        return "not-needed"
    return key


def build_llm(settings: ModelSettings) -> OpenAICompatibleVisionClient | NullVisionClient:
    """Real client when a key is available (or not needed), otherwise the null client."""
    key = resolve_api_key(settings)
    return OpenAICompatibleVisionClient(settings, key) if key else NullVisionClient()
