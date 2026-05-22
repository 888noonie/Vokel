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
        self._observers: list[Any] = []

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    def mark(self, name: str, **fields: Any) -> TraceEvent:
        event = TraceEvent(name=name, timestamp_ns=self._clock(), fields=fields)
        self._events.append(event)
        for observer in tuple(self._observers):
            observer.on_event(event)
        return event

    def add_observer(self, observer: Any) -> None:
        self._observers.append(observer)

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
            "capture_duration": ("capture_started", "capture_finished"),
            "capture_to_first_partial": ("capture_started", "partial_transcript"),
            "capture_to_stable": ("capture_started", "stable_transcript"),
            "capture_to_first_token": ("capture_started", "first_token"),
            "capture_to_playback_start": ("capture_started", "playback_started"),
            "asr_duration": ("asr_started", "asr_finished"),
            "asr_to_first_token": ("asr_finished", "first_token"),
            "memory_retrieval": ("memory_retrieval_started", "memory_retrieval_finished"),
            "memory_write": ("memory_write_started", "memory_write_finished"),
            "memory_to_first_token": ("memory_retrieval_finished", "first_token"),
            "turn_to_first_token": ("turn_submitted", "first_token"),
            "turn_to_first_phrase": ("turn_submitted", "first_phrase_queued"),
            "turn_to_playback_start": ("turn_submitted", "playback_started"),
            "generation_duration": ("generation_started", "generation_finished"),
            "turn_duration": ("turn_submitted", "generation_finished"),
            "turn_to_interruption": ("turn_submitted", "interruption_requested"),
            "generation_to_interruption": ("generation_started", "interruption_requested"),
            "interruption_to_playback_stop": ("interruption_requested", "playback_stop_requested"),
            "turn_to_playback_stop": ("turn_submitted", "playback_stop_requested"),
        }
        summary: dict[str, float] = {}
        for label, (start, end) in pairs.items():
            duration = self.duration_ms(start, end)
            if duration is not None:
                summary[label] = duration
        return summary

    def reset(self) -> None:
        self._events.clear()
