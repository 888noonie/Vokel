from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TYPE_CHECKING, runtime_checkable

from .events import Event

if TYPE_CHECKING:
    from .inference import ChatMessage


@dataclass(frozen=True)
class AgentBackendCapabilities:
    streams_text: bool = True
    supports_cancellation: bool = True
    supports_session_reset: bool = False
    owns_tools: bool = False
    emits_tool_activity: bool = False
    emits_media_artifacts: bool = False


TOOL_ACTIVITY_REPORTING_CONTRACT = (
    "Report every tool call lifecycle event to Vokel before execution. "
    "At minimum, emit the tool name when a tool starts so Vokel can provide "
    "user-facing reassurance, audio cues, interruption, consent, and audit. "
    "Do not print serialized tool-call syntax in assistant text. "
    "Only use a tool when the current user turn clearly asks for that capability; "
    "do not infer a new image, GIF, or web search from brief acknowledgement, "
    "thanks, praise, or feedback on the previous result."
)


@runtime_checkable
class AgentBackend(Protocol):
    """Streaming agent contract shared by built-in LM Studio and Hermes gateway clients."""

    capabilities: AgentBackendCapabilities

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
        messages: Sequence["ChatMessage"],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[Event]:
        ...

    async def cancel_active(self) -> None:
        """Stop an in-flight streamed turn (barge-in)."""
        ...
