from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LmStudioConfig:
    url: str = "http://localhost:1234/v1/chat/completions"
    model: str = "gemma-4-e4b-it-ultra-uncensored-heretic"
    temperature: float = 0.8
    top_p: float = 0.95
    timeout_seconds: float = 60.0


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
        "you directly."
    )
    max_history_messages: int = 20
