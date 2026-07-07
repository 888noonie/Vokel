from __future__ import annotations

import asyncio
from typing import Literal

from vokel.audio.beatclock import BeatClock, ClockStopped
from vokel.playback import PlaybackSink

Quantum = Literal["beat", "bar"]


class QuantizedPlaybackSink:
    """PlaybackSink wrapper that releases each phrase on the next beat or bar."""

    def __init__(
        self,
        inner: PlaybackSink,
        clock: BeatClock,
        quantum: Quantum = "beat",
    ) -> None:
        self.inner = inner
        self.clock = clock
        self.quantum = quantum
        self._stop_requested = asyncio.Event()

    async def speak(self, phrase: str) -> None:
        self._stop_requested.clear()
        wait_gate = (
            self.clock.wait_for_downbeat()
            if self.quantum == "bar"
            else self.clock.wait_for_beat()
        )
        beat_task = asyncio.ensure_future(wait_gate)
        stop_task = asyncio.ensure_future(self._stop_requested.wait())
        try:
            done, _pending = await asyncio.wait(
                {beat_task, stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for task in (beat_task, stop_task):
                if not task.done():
                    task.cancel()
        if stop_task in done or self._stop_requested.is_set():
            if beat_task.done():
                beat_task.exception()
            return
        try:
            beat_task.result()
        except ClockStopped:
            await self.inner.speak(phrase)
            return
        await self.inner.speak(phrase)

    async def stop(self) -> None:
        self._stop_requested.set()
        await self.inner.stop()