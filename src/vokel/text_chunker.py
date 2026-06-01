from __future__ import annotations

import re
from dataclasses import dataclass, field


CLAUSE_BOUNDARIES = frozenset({".", "!", "?", ",", ";", "\n"})
_IMAGE_PATH_RE = re.compile(r"\.(?:png|jpe?g|webp|gif|svg)(?:[?#][^\s)]*)?$", re.I)


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
        buffer = "".join(self._buffer)
        buffered_len = len(buffer.strip())
        if self._has_incomplete_link_or_url(buffer):
            return False
        if buffered_len >= self.max_chars:
            return True
        return buffered_len >= self.min_chars and char in CLAUSE_BOUNDARIES

    def _take_buffer(self) -> str | None:
        phrase = "".join(self._buffer).strip()
        self._buffer.clear()
        return phrase or None

    @staticmethod
    def _has_incomplete_link_or_url(buffer: str) -> bool:
        if not buffer.strip():
            return False
        tail = buffer.rsplit(maxsplit=1)[-1]
        if not tail:
            return False
        if tail.rfind("[") > tail.rfind("]"):
            return True
        if tail.count("](") > tail.count(")"):
            return True
        lower_tail = tail.lower()
        if "http://" in lower_tail or "https://" in lower_tail or "sandbox:" in lower_tail:
            return True
        return bool(_IMAGE_PATH_RE.search(tail))
