from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from typing import Protocol


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


def available_playback_backends() -> list[str]:
    backends = ["console"]
    if shutil.which("spd-say"):
        backends.append("spd-say")
    return backends


def build_playback_sink(name: str) -> PlaybackSink:
    if name == "console":
        return ConsolePlaybackSink()
    if name == "spd-say":
        if not shutil.which("spd-say"):
            raise RuntimeError("spd-say is not installed")
        return SpdSayPlaybackSink()
    raise ValueError(f"Unknown playback backend: {name}")
