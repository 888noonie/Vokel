from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24_000


class BeatTrackPlayer:
    """Looping synthesized bar (kick + hat) via a dedicated PortAudio output stream."""

    def __init__(
        self,
        bpm: float,
        beats_per_bar: int = 4,
        gain: float = 0.35,
        level: float = 1.0,
    ) -> None:
        if bpm <= 0:
            raise ValueError("bpm must be positive")
        self.bpm = bpm
        self.beats_per_bar = beats_per_bar
        self.gain = gain
        self.sample_rate = SAMPLE_RATE
        self._buffer: Any = None
        self._cursor = 0
        self._stream: Any = None
        self._level = 1.0
        self.set_level(level)

    def set_level(self, level: float) -> None:
        self._level = min(1.0, max(0.0, float(level)))

    def _render_bar(self) -> Any:
        import numpy as np

        beat_samples = max(1, int(self.sample_rate * 60.0 / self.bpm))
        bar_samples = beat_samples * self.beats_per_bar
        buffer = np.zeros(bar_samples, dtype=np.float32)

        kick_len = min(int(0.12 * self.sample_rate), beat_samples)
        hat_len = min(int(0.03 * self.sample_rate), beat_samples)

        kick_t = np.arange(kick_len, dtype=np.float32) / self.sample_rate
        kick = np.sin(2.0 * np.pi * 60.0 * kick_t) * np.exp(-kick_t / 0.04)

        hat_noise = np.random.randn(hat_len).astype(np.float32)
        hat_env = np.exp(-np.arange(hat_len, dtype=np.float32) / max(hat_len * 0.15, 1.0))
        hat = hat_noise * hat_env * 0.35

        for beat in range(self.beats_per_bar):
            start = beat * beat_samples
            kick_end = min(start + kick_len, bar_samples)
            buffer[start:kick_end] += kick[: kick_end - start] * self.gain
            if beat > 0:
                hat_end = min(start + hat_len, bar_samples)
                buffer[start:hat_end] += hat[: hat_end - start] * self.gain

        peak = float(np.max(np.abs(buffer)))
        if peak > 0.95:
            buffer *= 0.95 / peak
        return buffer

    async def start(self) -> None:
        if self._stream is not None:
            return
        import sounddevice as sd

        self._buffer = self._render_bar()
        self._cursor = 0
        player = self

        def callback(outdata: Any, frames: int, _time_info: Any, status: Any) -> None:
            if status:
                logger.debug("BeatTrackPlayer callback status: %s", status)
            buffer = player._buffer
            if buffer is None:
                outdata.fill(0)
                return
            for frame in range(frames):
                outdata[frame, 0] = (
                    buffer[player._cursor % len(buffer)] * player._level
                )
                player._cursor += 1

        # Phase 1: hardware stream timing and asyncio BeatClock drift slowly apart.
        # Phase 2 should derive BeatClock from this stream's frame counter.
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=callback,
        )
        self._stream.start()
        await asyncio.sleep(0)

    async def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None
        self._buffer = None
        await asyncio.sleep(0)