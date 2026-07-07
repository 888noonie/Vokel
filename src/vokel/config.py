from __future__ import annotations

import os
from dataclasses import dataclass, field

from .agent_backend import TOOL_ACTIVITY_REPORTING_CONTRACT

_DEFAULT_JAN_MODEL = "HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive"
_DEFAULT_JAN_URL = "http://127.0.0.1:6767/v1/chat/completions"


@dataclass(frozen=True)
class LmStudioConfig:
    url: str = os.environ.get("VOKEL_LLM_URL", os.environ.get("LM_STUDIO_URL", _DEFAULT_JAN_URL))
    model: str = os.environ.get("VOKEL_LLM_MODEL", os.environ.get("LM_STUDIO_MODEL", _DEFAULT_JAN_MODEL))
    api_key: str = os.environ.get("VOKEL_LLM_API_KEY", os.environ.get("JAN_API_KEY", ""))
    temperature: float = 0.8
    top_p: float = 0.95
    timeout_seconds: float = 120.0
    # Connect phase is bounded separately so an unreachable Jan/LM Studio socket
    # fails fast instead of hanging on the long read timeout above.
    connect_timeout_seconds: float = 5.0
    # Native /api/v1/chat MCP path (when True, client derives /api/v1/chat from base and
    # passes integrations; LM Studio executes its own configured MCP servers from mcp.json)
    use_native_chat: bool = False
    mcp_integrations: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class VoiceLoopConfig:
    system_prompt: str = (
        "You are Vokel, a concise voice assistant running entirely on the user's "
        "own hardware. Give brief, natural conversational answers suited for "
        "immediate text-to-speech readout. Never use bullet points, asterisks, "
        "markdown formatting, or numbered lists. "
        "The user can interrupt you by pressing a button or speaking if they are "
        "using a headset. Never claim you can browse the web yourself; if a web "
        "search or image search was performed, the results will be provided to "
        "you directly. "
        "The app controls a camera for you; you cannot take photos, zoom, switch "
        "lenses, or start video on your own, and there are no portrait, panorama, or "
        "other photo modes — do not offer or imply any. When the user wants a still "
        "photo, tell them to say 'shoot'; for a live video loop tell them to say "
        "'action', and 'cut' to stop. The app takes the picture the instant they say "
        "those words. When a camera frame is attached to a message, describe only what "
        "is actually visible in it and never invent camera features. "
        "You have no weather tool, no image-generation tool, and no way to display "
        "images yourself; never claim to start, run, or possess a tool or capability "
        "you were not explicitly given, and never narrate tool usage in your spoken "
        "reply (for example, never say you are 'starting a tool call'). If the result "
        "of a web search, image search, or any tool is not actually provided to you, "
        "say plainly that you could not retrieve it — never invent headlines, dates, "
        "weather, prices, or facts, and never describe an image you were not given. "
        f"{TOOL_ACTIVITY_REPORTING_CONTRACT}"
    )
    max_history_messages: int = 20
