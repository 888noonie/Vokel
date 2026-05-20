from __future__ import annotations

import asyncio
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
