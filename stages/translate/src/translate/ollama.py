"""Minimal Ollama HTTP client (stdlib only)."""

import json
import time
import urllib.error
import urllib.request
from typing import Any


class OllamaError(Exception):
    pass


class Ollama:
    def __init__(self, url: str, model: str, *, timeout: float = 600) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.url}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            raise OllamaError(f"ollama {path} failed ({e.code}): {detail}") from e
        except (urllib.error.URLError, OSError) as e:
            raise OllamaError(f"ollama not reachable at {self.url}: {e}") from e

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any],
        options: dict[str, Any],
        keep_alive: str | int = "5m",
    ) -> str:
        """One non-streaming chat turn with structured (JSON Schema) output."""
        body = self._post(
            "/api/chat",
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "format": schema,
                "think": False,
                "options": options,
                "keep_alive": keep_alive,
            },
        )
        return body.get("message", {}).get("content", "")

    def loaded_models(self) -> list[str]:
        request = urllib.request.Request(f"{self.url}/api/ps")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return [m.get("name", "") for m in json.load(response).get("models", [])]
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise OllamaError(f"ollama not reachable at {self.url}: {e}") from e

    def unload(self, *, wait: float = 15.0) -> None:
        """Drop the model from VRAM now (the next stage needs the GPU) and wait
        until Ollama no longer lists it."""
        self._post("/api/generate", {"model": self.model, "keep_alive": 0})
        deadline = time.monotonic() + wait
        while self.model in self.loaded_models():
            if time.monotonic() > deadline:
                raise OllamaError(f"{self.model} still loaded {wait:g}s after unloading")
            time.sleep(0.2)
