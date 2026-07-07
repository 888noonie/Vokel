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


HERMES_CAMERA_FRAME_CONTRACT = (
    "EXPLICIT CAMERA FRAME PAYLOAD CONTRACT (v1) for Hermes gateway.\n"
    "Vokel sends ONE fresh, explicitly user-approved webcam frame ONLY for the current turn "
    "when Camera Questions + Hermes backend are both armed. The frame is a data: URL (jpeg base64). "
    "Frame is captured locally, consent+audit recorded BEFORE send. Uses VisualContext (source, captured_at, consent, contract).\n"
    "\n"
    "HTTP (HermesAgentClient):\n"
    "  Add to the /v1/responses (or fallback /v1/chat/completions) payload:\n"
    "    \"camera_frame\": {\n"
    "      \"data_url\": \"...\",\n"
    "      \"source\": \"/dev/video4\",\n"
    "      \"captured_at\": \"...\",\n"
    "      \"consent\": \"...\",\n"
    "      \"contract\": \"hermes_camera_frame_v1\"\n"
    "    }\n"
    "\n"
    "WebSocket (HermesWebSocketClient):\n"
    "  Inside the \"start_turn\" object: \"camera_frame\": { same }\n"
    "\n"
    "Barge-in / interrupt during capture or generation prevents the frame from reaching any backend "
    "(GStreamer is terminated, the frame is discarded, and the capture lock is held until process exit).\n"
    "\n"
    "Hermes HTTP gateway consumes images via OpenAI multimodal input (image_url + text). "
    "camera_frame metadata is also attached for audit/forward-compat.\n"
    "Android: CameraX produces equivalent VisualContext; same extract/payload path."
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
