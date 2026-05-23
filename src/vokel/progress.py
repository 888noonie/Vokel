from __future__ import annotations

from typing import Protocol

from .telemetry import TraceEvent


class ProgressObserver(Protocol):
    def on_event(self, event: TraceEvent) -> None:
        pass


class NullProgressObserver:
    def on_event(self, event: TraceEvent) -> None:
        pass


class ConsoleProgressObserver:
    """Human-readable progress cues for live benchmark runs."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def on_event(self, event: TraceEvent) -> None:
        if event.name == "capture_started":
            self._print_once("capture_started", "[listen] Start speaking or play test clip now...")
        elif event.name == "capture_finished":
            audio_seconds = float(event.fields.get("audio_seconds") or 0)
            samples = int(event.fields.get("samples") or 0)
            print(
                f"[capture] turn closed: {audio_seconds:.2f}s audio, {samples} samples",
                flush=True,
            )
        elif event.name == "asr_started":
            print("[asr] transcribing...", flush=True)
        elif event.name == "asr_finished":
            chars = int(event.fields.get("chars") or 0)
            print(f"[asr] transcript ready: {chars} chars", flush=True)
        elif event.name == "generation_started":
            print("[llm] streaming...", flush=True)
        elif event.name == "first_token":
            print("[llm] first token", flush=True)
        elif event.name == "first_phrase_queued":
            chars = int(event.fields.get("chars") or 0)
            print(f"[playback] first phrase queued: {chars} chars", flush=True)
        elif event.name == "playback_started":
            self._print_once("playback_started", "[playback] speaking...")
        elif event.name == "generation_finished":
            print("[llm] generation complete", flush=True)

    def _print_once(self, key: str, message: str) -> None:
        if key in self._seen:
            return
        self._seen.add(key)
        print(message, flush=True)
