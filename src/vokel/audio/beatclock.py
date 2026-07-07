from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass(frozen=True)
class BeatInfo:
    bar: int
    beat: int
    at: float


class ClockStopped(Exception):
    """Raised to waiters when the clock stops while they are waiting."""


class BeatClock:
    """Drift-corrected musical clock. Ticks are scheduled against a single
    monotonic origin (loop.time()), so error never accumulates."""

    def __init__(self, bpm: float = 90.0, beats_per_bar: int = 4):
        if bpm <= 0:
            raise ValueError("bpm must be positive")
        self.bpm = bpm
        self.beats_per_bar = beats_per_bar
        self.beat_interval = 60.0 / bpm
        self._task: asyncio.Task[None] | None = None
        self._running = False
        # Fresh-event-per-tick pattern: set-then-clear on a shared Event is racy
        # (a waiter arriving between set() and clear() waits a full extra tick,
        # and one arriving after clear() saw nothing). Each tick swaps in a new
        # Event and sets the old one exactly once.
        self._beat_gate = asyncio.Event()
        self._downbeat_gate = asyncio.Event()
        self._last_info: BeatInfo | None = None

    @property
    def running(self) -> bool:
        return self._running

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        # Wake every pending waiter so nothing hangs forever; they observe
        # running=False and raise ClockStopped.
        self._beat_gate.set()
        self._downbeat_gate.set()

    async def wait_for_beat(self) -> BeatInfo:
        return await self._wait(self._beat_gate_ref)

    async def wait_for_downbeat(self) -> BeatInfo:
        return await self._wait(self._downbeat_gate_ref)

    def _beat_gate_ref(self) -> asyncio.Event:
        return self._beat_gate

    def _downbeat_gate_ref(self) -> asyncio.Event:
        return self._downbeat_gate

    async def _wait(self, gate_ref) -> BeatInfo:
        if not self._running:
            raise ClockStopped
        gate = gate_ref()
        await gate.wait()
        if not self._running or self._last_info is None:
            raise ClockStopped
        return self._last_info

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        origin = loop.time()
        tick = 0
        bar = 0
        while self._running:
            beat = tick % self.beats_per_bar
            if beat == 0 and tick > 0:
                bar += 1
            self._last_info = BeatInfo(bar=bar, beat=beat, at=loop.time())

            old_beat, self._beat_gate = self._beat_gate, asyncio.Event()
            old_beat.set()
            if beat == 0:
                old_down, self._downbeat_gate = self._downbeat_gate, asyncio.Event()
                old_down.set()

            tick += 1
            next_deadline = origin + tick * self.beat_interval
            delay = next_deadline - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                await asyncio.sleep(0)