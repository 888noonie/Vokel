from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from types import TracebackType
from typing import Any
from typing import TYPE_CHECKING

from .config import LmStudioConfig
from .events import Event, TextDeltaEvent, ToolCallEvent

if TYPE_CHECKING:
    import httpx

ChatMessage = dict[str, Any]


class InferenceError(RuntimeError):
    pass


def parse_sse_delta(line: str) -> dict[str, Any] | None:
    """Extract a content token or tool call from one OpenAI-compatible SSE line."""

    if not line.startswith("data:"):
        return None

    payload = line.removeprefix("data:").strip()
    if not payload or payload == "[DONE]":
        return None

    data = json.loads(payload)
    choices = data.get("choices") or []
    if not choices:
        return None

    return choices[0].get("delta")


class LocalInferenceClient:
    def __init__(self, config: LmStudioConfig, client: "httpx.AsyncClient | None" = None):
        self.config = config
        self._client = client
        self._owns_client = client is None
        self._active_response: Any = None

    async def __aenter__(self) -> "LocalInferenceClient":
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
        await self.cancel_active()
        if self._owns_client and self._client is not None:
            await self._client.aclose()
        self._client = None

    async def cancel_active(self) -> None:
        response = self._active_response
        self._active_response = None
        if response is not None:
            await response.aclose()

    async def stream_chat(
        self, messages: Sequence[ChatMessage], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[Event]:
        if self._client is None:
            raise InferenceError("LocalInferenceClient must be used as an async context manager")

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": list(messages),
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        active_tool_calls: dict[int, dict[str, Any]] = {}

        try:
            async with self._client.stream("POST", self.config.url, json=payload) as response:
                self._active_response = response
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip() == "data: [DONE]":
                        break
                    delta = parse_sse_delta(line)
                    if not delta:
                        continue

                    if "content" in delta and delta["content"]:
                        yield TextDeltaEvent(content=delta["content"])

                    if "tool_calls" in delta:
                        for tc in delta["tool_calls"]:
                            index = tc.get("index")
                            if index is None:
                                continue

                            if index not in active_tool_calls:
                                active_tool_calls[index] = {
                                    "id": tc.get("id", ""),
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }

                            if "id" in tc and tc["id"]:
                                active_tool_calls[index]["id"] = tc["id"]

                            fn = tc.get("function", {})
                            if "name" in fn and fn["name"]:
                                active_tool_calls[index]["function"]["name"] += fn["name"]
                            if "arguments" in fn and fn["arguments"]:
                                active_tool_calls[index]["function"]["arguments"] += fn["arguments"]
        finally:
            self._active_response = None

        # After streaming is done, yield completed tool calls
        for tc in active_tool_calls.values():
            name = tc["function"]["name"]
            arguments_str = tc["function"]["arguments"]
            try:
                arguments = json.loads(arguments_str) if arguments_str else {}
            except json.JSONDecodeError:
                arguments = {}
            yield ToolCallEvent(call_id=tc["id"], name=name, arguments=arguments)
