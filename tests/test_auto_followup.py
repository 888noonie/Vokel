from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from vokel.auto_followup import AutoFollowupScheduler, clamp_auto_followup_seconds
from vokel.engine import ConversationEngine


class AutoFollowupSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_fires_after_delay(self) -> None:
        fired = asyncio.Event()

        async def on_trigger() -> None:
            fired.set()

        scheduler = AutoFollowupScheduler(enabled=True, delay_s=8, on_trigger=on_trigger)
        with patch("vokel.auto_followup.asyncio.sleep", new_callable=AsyncMock):
            scheduler.arm_listening()
            await asyncio.wait_for(fired.wait(), timeout=0.5)
        scheduler.cancel()

    async def test_user_activity_cancels_pending_fire(self) -> None:
        fired = asyncio.Event()

        async def on_trigger() -> None:
            fired.set()

        scheduler = AutoFollowupScheduler(enabled=True, delay_s=8, on_trigger=on_trigger)
        with patch("vokel.auto_followup.asyncio.sleep", new_callable=AsyncMock):
            scheduler.arm_listening()
            await asyncio.sleep(0.05)
            scheduler.on_user_activity()
        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(fired.wait(), timeout=0.1)
        scheduler.cancel()

    def test_clamp_seconds(self) -> None:
        self.assertEqual(clamp_auto_followup_seconds(1), 3.0)
        self.assertEqual(clamp_auto_followup_seconds(8), 8.0)
        self.assertEqual(clamp_auto_followup_seconds(99), 30.0)


class ConversationEngineAutoFollowupTests(unittest.IsolatedAsyncioTestCase):
    async def test_submit_auto_followup_skips_when_user_spoke_last(self) -> None:
        llm = MagicMock()
        llm.cancel_active = AsyncMock()
        playback = MagicMock()
        playback.stop = AsyncMock()
        engine = ConversationEngine(llm=llm, playback=playback, echo_tokens=False)
        engine.history.append({"role": "user", "content": "Hello"})
        await engine.submit_auto_followup()
        llm.stream_chat.assert_not_called()

    async def test_submit_auto_followup_runs_when_assistant_spoke_last(self) -> None:
        async def dummy_stream(*_args: object, **_kwargs: object):
            yield "Still here?"

        llm = MagicMock()
        llm.cancel_active = AsyncMock()
        llm.stream_chat = dummy_stream
        playback = MagicMock()
        playback.stop = AsyncMock()
        playback.speak = AsyncMock()
        engine = ConversationEngine(llm=llm, playback=playback, echo_tokens=False)
        await engine.start()
        try:
            engine.history.append({"role": "assistant", "content": "Hi there."})
            await engine.submit_auto_followup()
        finally:
            await engine.close()

        self.assertTrue(any(event.name == "auto_followup_triggered" for event in engine.trace.events))


if __name__ == "__main__":
    unittest.main()
