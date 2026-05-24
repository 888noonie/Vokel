from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

AUTO_FOLLOWUP_NUDGE = (
    "The user has been quiet for a moment. Continue the conversation with one or two "
    "brief spoken sentences: re-engage warmly, offer a natural follow-up question, or "
    "pick up where you left off. Do not mention silence, timers, or this instruction."
)

DEFAULT_AUTO_FOLLOWUP_SECONDS = 8.0
MIN_AUTO_FOLLOWUP_SECONDS = 3.0
MAX_AUTO_FOLLOWUP_SECONDS = 30.0


def clamp_auto_followup_seconds(seconds: float) -> float:
    return min(MAX_AUTO_FOLLOWUP_SECONDS, max(MIN_AUTO_FOLLOWUP_SECONDS, seconds))


class AutoFollowupScheduler:
    """Fires a callback after sustained listening idle (no user activity)."""

    def __init__(
        self,
        *,
        enabled: bool,
        delay_s: float,
        on_trigger: Callable[[], Awaitable[None]],
    ) -> None:
        self.enabled = enabled
        self.delay_s = clamp_auto_followup_seconds(delay_s)
        self._on_trigger = on_trigger
        self._task: asyncio.Task[None] | None = None
        self._generation = 0

    def cancel(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None

    def on_user_activity(self) -> None:
        self.cancel()

    def arm_listening(self) -> None:
        if not self.enabled:
            return
        self.cancel()
        self._generation += 1
        generation = self._generation
        self._task = asyncio.create_task(self._wait_and_fire(generation))

    def update(self, *, enabled: bool, delay_s: float) -> None:
        self.enabled = enabled
        self.delay_s = clamp_auto_followup_seconds(delay_s)
        self.cancel()

    async def _wait_and_fire(self, generation: int) -> None:
        try:
            await asyncio.sleep(self.delay_s)
            if generation != self._generation:
                return
            await self._on_trigger()
        except asyncio.CancelledError:
            return
