from __future__ import annotations

from dataclasses import dataclass, field


CLAUSE_BOUNDARIES = frozenset({".", "!", "?", ",", ";", ":", "\n"})


@dataclass
class PhraseChunker:
    """Incrementally turns streamed text into speakable phrases."""

    min_chars: int = 12
    max_chars: int = 220
    _buffer: list[str] = field(default_factory=list)

    def push(self, text: str) -> list[str]:
        if not text:
            return []

        phrases: list[str] = []
        for char in text:
            self._buffer.append(char)
            if self._should_flush(char):
                phrase = self._take_buffer()
                if phrase:
                    phrases.append(phrase)

        return phrases

    def flush(self) -> str | None:
        return self._take_buffer()

    def reset(self) -> None:
        self._buffer.clear()

    def _should_flush(self, char: str) -> bool:
        buffered_len = len("".join(self._buffer).strip())
        if buffered_len >= self.max_chars:
            return True
        return buffered_len >= self.min_chars and char in CLAUSE_BOUNDARIES

    def _take_buffer(self) -> str | None:
        phrase = "".join(self._buffer).strip()
        self._buffer.clear()
        return phrase or None
