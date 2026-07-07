from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24_000
MusicalStyle = Literal["beat", "metronome"]
MAX_TRACK_DECODE_SECONDS = 120.0


class MusicalTrackDecodeError(RuntimeError):
    """Raised when ffmpeg cannot decode an uploaded backing track."""


def decode_audio_file(
    path: Path,
    *,
    sample_rate: int = SAMPLE_RATE,
    timeout_seconds: float = MAX_TRACK_DECODE_SECONDS,
) -> Any:
    """Decode any ffmpeg-supported audio file to mono float32 PCM."""
    import numpy as np

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise MusicalTrackDecodeError(
            "ffmpeg is required to decode backing tracks but was not found on PATH"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise MusicalTrackDecodeError(
            "ffmpeg timed out while decoding the backing track"
        ) from exc

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise MusicalTrackDecodeError(stderr or "ffmpeg failed to decode the backing track")
    if not proc.stdout:
        raise MusicalTrackDecodeError("decoded track is empty")

    samples = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if samples.size == 0:
        raise MusicalTrackDecodeError("decoded track is empty")
    return samples


class BeatTrackPlayer:
    """Looping synthesized bar or external buffer via a dedicated PortAudio output stream."""

    def __init__(
        self,
        bpm: float,
        beats_per_bar: int = 4,
        gain: float = 0.35,
        level: float = 1.0,
        style: MusicalStyle = "beat",
    ) -> None:
        if bpm <= 0:
            raise ValueError("bpm must be positive")
        self.bpm = bpm
        self.beats_per_bar = beats_per_bar
        self.gain = gain
        self.sample_rate = SAMPLE_RATE
        self.style: MusicalStyle = style
        self._buffer: Any = None
        self._cursor = 0
        self._stream: Any = None
        self._level = 1.0
        self._external_buffer = False
        self.set_level(level)

    def set_level(self, level: float) -> None:
        self._level = min(1.0, max(0.0, float(level)))

    def load_buffer(self, samples: Any) -> None:
        import numpy as np

        buffer = np.asarray(samples, dtype=np.float32).reshape(-1)
        if buffer.size == 0:
            raise ValueError("backing track buffer must not be empty")
        self._buffer = buffer
        self._cursor = 0
        self._external_buffer = True

    def clear_buffer(self) -> None:
        self._external_buffer = False
        self._cursor = 0
        self._buffer = self._render_bar() if self._stream is not None else None

    def nudge_cursor(self, ms: float) -> None:
        buffer = self._buffer
        if buffer is None or len(buffer) == 0:
            return
        shift = int(round(float(ms) * self.sample_rate / 1000.0))
        self._cursor = (self._cursor + shift) % len(buffer)

    def _render_click(
        self,
        *,
        frequency_hz: float,
        duration_seconds: float = 0.015,
    ) -> Any:
        import numpy as np

        click_len = max(1, int(duration_seconds * self.sample_rate))
        click_t = np.arange(click_len, dtype=np.float32) / self.sample_rate
        return np.sin(2.0 * np.pi * frequency_hz * click_t) * np.exp(-click_t / 0.004)

    def _normalize_bar(self, buffer: Any) -> Any:
        import numpy as np

        peak = float(np.max(np.abs(buffer)))
        if peak > 0.95:
            buffer = buffer * (0.95 / peak)
        return buffer

    def _render_metronome_bar(self) -> Any:
        import numpy as np

        beat_samples = max(1, int(self.sample_rate * 60.0 / self.bpm))
        bar_samples = beat_samples * self.beats_per_bar
        buffer = np.zeros(bar_samples, dtype=np.float32)
        click_len = min(int(0.015 * self.sample_rate), beat_samples)

        for beat in range(self.beats_per_bar):
            start = beat * beat_samples
            click_end = min(start + click_len, bar_samples)
            frequency = 1500.0 if beat == 0 else 1000.0
            click = self._render_click(frequency_hz=frequency)[: click_end - start]
            buffer[start:click_end] += click * self.gain

        return self._normalize_bar(buffer)

    def _render_beat_bar(self) -> Any:
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

        return self._normalize_bar(buffer)

    def _render_bar(self) -> Any:
        if self.style == "metronome":
            return self._render_metronome_bar()
        return self._render_beat_bar()

    async def start(self) -> None:
        if self._stream is not None:
            return
        import sounddevice as sd

        if not self._external_buffer:
            self._buffer = self._render_bar()
            self._cursor = 0
        elif self._buffer is None:
            self._buffer = self._render_bar()
            self._cursor = 0
            self._external_buffer = False

        player = self

        def callback(outdata: Any, frames: int, _time_info: Any, status: Any) -> None:
            if status:
                logger.debug("BeatTrackPlayer callback status: %s", status)
            buffer = player._buffer
            if buffer is None:
                outdata.fill(0)
                return
            buflen = len(buffer)
            for frame in range(frames):
                outdata[frame, 0] = (
                    buffer[player._cursor % buflen] * player._level
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
            self._buffer = None
            self._external_buffer = False
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None
        self._buffer = None
        self._external_buffer = False
        await asyncio.sleep(0)