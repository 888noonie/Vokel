from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from typing import Protocol, Any


class PlaybackSink(Protocol):
    async def speak(self, phrase: str) -> None:
        pass

    async def stop(self) -> None:
        pass


class ConsolePlaybackSink:
    """Development sink that shows phrase boundaries without producing audio."""

    async def speak(self, phrase: str) -> None:
        print(f"\n[TTS phrase] {phrase}", flush=True)
        await asyncio.sleep(0)

    async def stop(self) -> None:
        await asyncio.sleep(0)


@dataclass(frozen=True)
class SubprocessPlaybackConfig:
    command: tuple[str, ...]
    stop_command: tuple[str, ...] = ()
    terminate_timeout_seconds: float = 0.5


class SubprocessPlaybackSink:
    """Playback sink backed by one cancellable subprocess at a time."""

    def __init__(self, config: SubprocessPlaybackConfig):
        self.config = config
        self._process: asyncio.subprocess.Process | None = None
        self._process_lock = asyncio.Lock()

    async def speak(self, phrase: str) -> None:
        await self.stop()
        command = self._render_command(phrase)
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        async with self._process_lock:
            self._process = process
        try:
            await process.wait()
        finally:
            async with self._process_lock:
                if self._process is process:
                    self._process = None

    async def stop(self) -> None:
        async with self._process_lock:
            process = self._process

        if self.config.stop_command:
            await self._run_stop_command()

        if process is None or process.returncode is not None:
            return

        process.terminate()
        try:
            await asyncio.wait_for(
                process.wait(),
                timeout=self.config.terminate_timeout_seconds,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

        async with self._process_lock:
            if self._process is process:
                self._process = None

    def _render_command(self, phrase: str) -> tuple[str, ...]:
        rendered = tuple(part.format(text=phrase) for part in self.config.command)
        if any("{text}" in part for part in self.config.command):
            return rendered
        return (*rendered, phrase)

    async def _run_stop_command(self) -> None:
        process = await asyncio.create_subprocess_exec(
            *self.config.stop_command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await process.wait()


class SpdSayPlaybackSink(SubprocessPlaybackSink):
    def __init__(self, rate: int = 20, voice_type: str | None = None):
        command = ["spd-say", "--wait", "--rate", str(rate)]
        if voice_type:
            command.extend(["--voice-type", voice_type])
        command.append("{text}")
        super().__init__(
            SubprocessPlaybackConfig(
                command=tuple(command),
                stop_command=("spd-say", "--cancel"),
            )
        )


class KokoroPlaybackSink:
    """Playback sink that synthesizes phrases using Kokoro ONNX and plays via sounddevice."""

    def __init__(self, voice: str = "af_heart"):
        import sounddevice as sd
        from kokoro_onnx import Kokoro
        from pathlib import Path

        models_dir = Path("models")
        model_path = models_dir / "kokoro-v1.0.onnx"
        voices_path = models_dir / "voices-v1.0.bin"

        if not model_path.exists() or not voices_path.exists():
            raise RuntimeError(
                f"Kokoro model files not found in {models_dir}. "
                "Run `python3 scripts/download_models.py kokoro-v1.0 kokoro-voices`"
            )

        self.kokoro = Kokoro(str(model_path), str(voices_path))
        self.voice = voice
        self.sd = sd
        self._playback_task: asyncio.Task[Any] | None = None
        self._synthesis_task: asyncio.Task[Any] | None = None
        self._stop_event = asyncio.Event()

    async def speak(self, phrase: str) -> None:
        await self.stop()
        self._stop_event.clear()
        
        queue: asyncio.Queue[Any] = asyncio.Queue()
        
        # Start synthesis in the background
        self._synthesis_task = asyncio.create_task(self._synthesize_loop(phrase, queue))
        # Start playback in the foreground (blocks until finished or stopped)
        self._playback_task = asyncio.create_task(self._playback_loop(queue))
        
        try:
            await self._playback_task
        except asyncio.CancelledError:
            pass

    async def _synthesize_loop(self, phrase: str, queue: asyncio.Queue[Any]) -> None:
        try:
            async for samples, sample_rate in self.kokoro.create_stream(
                phrase, self.voice, 1.0, "en-us"
            ):
                if self._stop_event.is_set():
                    break
                await queue.put((samples, sample_rate))
        except Exception as e:
            print(f"Kokoro synthesis failed: {e}")
        finally:
            await queue.put(None)

    async def _playback_loop(self, queue: asyncio.Queue[Any]) -> None:
        while not self._stop_event.is_set():
            item = await queue.get()
            if item is None:
                break
            
            samples, sample_rate = item
            self.sd.play(samples, sample_rate)
            
            while self.sd.get_stream() and self.sd.get_stream().active:
                if self._stop_event.is_set():
                    self.sd.stop()
                    return
                await asyncio.sleep(0.05)

    async def stop(self) -> None:
        self._stop_event.set()
        self.sd.stop()
        if self._synthesis_task and not self._synthesis_task.done():
            self._synthesis_task.cancel()
        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()
        self._synthesis_task = None
        self._playback_task = None


def available_playback_backends() -> list[str]:
    backends = ["console", "kokoro"]
    if shutil.which("spd-say"):
        backends.append("spd-say")
    return backends


def build_playback_sink(name: str) -> PlaybackSink:
    if name == "console":
        return ConsolePlaybackSink()
    if name == "kokoro":
        return KokoroPlaybackSink()
    if name == "spd-say":
        if not shutil.which("spd-say"):
            raise RuntimeError("spd-say is not installed")
        return SpdSayPlaybackSink()
    raise ValueError(f"Unknown playback backend: {name}")
