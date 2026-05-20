from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class AudioTurn:
    """One completed user speech turn, ready for ASR or already transcribed."""

    text: str
    sample_rate: int | None = None
    audio_samples: tuple[float, ...] = ()

    @property
    def has_audio(self) -> bool:
        return bool(self.audio_samples)


class TurnProducer(Protocol):
    async def next_turn(self) -> AudioTurn:
        pass


class AsrEngine(Protocol):
    async def transcribe(self, turn: AudioTurn) -> str:
        pass


class PassthroughAsr:
    """Development ASR that treats AudioTurn.text as the transcript."""

    async def transcribe(self, turn: AudioTurn) -> str:
        return turn.text


class TextTurnProducer:
    """Async producer for deterministic tests and CLI-driven development."""

    def __init__(self, turns: list[str]):
        self._turns = list(turns)

    async def next_turn(self) -> AudioTurn:
        if not self._turns:
            raise StopAsyncIteration
        return AudioTurn(text=self._turns.pop(0))
