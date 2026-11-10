"""Chat client for a local Ollama server."""

import json
import os
from collections.abc import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_DEFAULT_MODEL = "qwen2.5-coder:14b"
_OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"


class OllamaClient:
    """Send prepared messages to local Ollama and return reply text."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
    ) -> None:
        self.model = model or os.environ.get("OLLAMA_MODEL") or _DEFAULT_MODEL
        self.temperature = temperature

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Return the assistant message content from Ollama's chat API."""
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "stream": False,
            "format": "json",
        }
        if self.temperature is not None:
            payload["options"] = {"temperature": self.temperature}

        request = Request(
            _OLLAMA_CHAT_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                body = response.read()
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama returned HTTP {error.code}: {detail or error.reason}"
            ) from error
        except URLError as error:
            raise RuntimeError(f"Could not connect to local Ollama: {error.reason}") from error

        try:
            response_data = json.loads(body)
            content = response_data["message"]["content"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise RuntimeError("Ollama returned an invalid chat response") from error
        if not isinstance(content, str):
            raise RuntimeError("Ollama chat response content must be text")
        return content
