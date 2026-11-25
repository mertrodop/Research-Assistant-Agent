from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Protocol

import httpx

from ..config import ModelSettings


class VisionError(RuntimeError):
    """Raised when the model endpoint cannot answer."""


class LlmClient(Protocol):
    model: str

    def describe(self, image_path: str | Path, prompt: str) -> str: ...

    def complete(self, prompt: str) -> str: ...


class NullVisionClient:
    """Used when figure interpretation is disabled."""

    model = "none"

    def describe(self, image_path: str | Path, prompt: str) -> str:
        raise VisionError("vision client disabled")

    def complete(self, prompt: str) -> str:
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

    def _post(self, content: list[dict] | str) -> str:
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


def build_llm(settings: ModelSettings) -> OpenAICompatibleVisionClient | NullVisionClient:
    """Real client when the API key is set, otherwise the null client."""
    if not os.environ.get(settings.api_key_env):
        return NullVisionClient()
    return OpenAICompatibleVisionClient(settings)
