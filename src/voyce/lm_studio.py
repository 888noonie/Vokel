from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from types import TracebackType
from typing import Any
from typing import TYPE_CHECKING

from .config import LmStudioConfig

if TYPE_CHECKING:
    import httpx

ChatMessage = dict[str, str]


class LmStudioError(RuntimeError):
    pass


def parse_sse_token(line: str) -> str | None:
    """Extract a content token from one OpenAI-compatible SSE line."""

    if not line.startswith("data:"):
        return None

    payload = line.removeprefix("data:").strip()
    if not payload or payload == "[DONE]":
        return None

    data = json.loads(payload)
    choices = data.get("choices") or []
    if not choices:
        return None

    delta = choices[0].get("delta") or {}
    token = delta.get("content")
    return token if isinstance(token, str) else None


class LmStudioClient:
    def __init__(self, config: LmStudioConfig, client: "httpx.AsyncClient | None" = None):
        self.config = config
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> "LmStudioClient":
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self.config.timeout_seconds)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
        self._client = None

    async def stream_chat(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        if self._client is None:
            raise LmStudioError("LmStudioClient must be used as an async context manager")

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": list(messages),
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "stream": True,
        }

        async with self._client.stream("POST", self.config.url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.strip() == "data: [DONE]":
                    break
                token = parse_sse_token(line)
                if token:
                    yield token
