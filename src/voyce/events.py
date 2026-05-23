from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class Event:
    pass


@dataclass(frozen=True)
class TextDeltaEvent(Event):
    content: str


@dataclass(frozen=True)
class ToolCallEvent(Event):
    call_id: str
    name: str
    arguments: dict[str, Any]
