from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


Clock = Callable[[], int]


@dataclass(frozen=True)
class TraceEvent:
    name: str
    timestamp_ns: int
    fields: dict[str, Any] = field(default_factory=dict)


class LatencyTrace:
    """Small monotonic event recorder for voice-loop latency measurements."""

    def __init__(self, clock: Clock | None = None):
        self._clock = clock or time.perf_counter_ns
        self._events: list[TraceEvent] = []

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    def mark(self, name: str, **fields: Any) -> TraceEvent:
        event = TraceEvent(name=name, timestamp_ns=self._clock(), fields=fields)
        self._events.append(event)
        return event

    def first(self, name: str) -> TraceEvent | None:
        return next((event for event in self._events if event.name == name), None)

    def duration_ms(self, start: str, end: str) -> float | None:
        start_event = self.first(start)
        end_event = self.first(end)
        if start_event is None or end_event is None:
            return None
        return (end_event.timestamp_ns - start_event.timestamp_ns) / 1_000_000

    def summary_ms(self) -> dict[str, float]:
        pairs = {
            "asr_duration": ("asr_started", "asr_finished"),
            "asr_to_first_token": ("asr_finished", "first_token"),
            "turn_to_first_token": ("turn_submitted", "first_token"),
            "turn_to_first_phrase": ("turn_submitted", "first_phrase_queued"),
            "turn_to_playback_start": ("turn_submitted", "playback_started"),
            "generation_duration": ("generation_started", "generation_finished"),
            "turn_duration": ("turn_submitted", "generation_finished"),
        }
        summary: dict[str, float] = {}
        for label, (start, end) in pairs.items():
            duration = self.duration_ms(start, end)
            if duration is not None:
                summary[label] = duration
        return summary

    def reset(self) -> None:
        self._events.clear()
