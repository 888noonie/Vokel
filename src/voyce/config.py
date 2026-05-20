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
        "You are a concise voice assistant. Give brief, natural conversational "
        "answers suited for immediate text-to-speech readout. Never use bullet "
        "points or formatting labels."
    )
    max_history_messages: int = 20
