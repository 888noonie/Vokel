from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Protocol, runtime_checkable

from .events import Event
from .inference import ChatMessage


@runtime_checkable
class AgentBackend(Protocol):
    """Streaming agent contract shared by built-in LM Studio and Hermes gateway clients."""

    async def __aenter__(self) -> AgentBackend:
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        ...

    async def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[Event]:
        ...

    async def cancel_active(self) -> None:
        """Stop an in-flight streamed turn (barge-in)."""
        ...
