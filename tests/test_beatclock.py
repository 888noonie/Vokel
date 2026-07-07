import asyncio
import unittest

from vokel.audio.beatclock import BeatClock, BeatInfo, ClockStopped


class BeatClockTests(unittest.IsolatedAsyncioTestCase):
    async def test_ticks_land_on_grid(self) -> None:
        bpm = 6000.0
        interval = 60.0 / bpm
        clock = BeatClock(bpm=bpm)
        await clock.start()

        timestamps: list[float] = []
        for _ in range(8):
            info = await clock.wait_for_beat()
            timestamps.append(info.at)

        await clock.stop()

        for i in range(1, len(timestamps)):
            delta = timestamps[i] - timestamps[i - 1]
            self.assertGreaterEqual(delta, interval * 0.7)
            self.assertLessEqual(delta, interval * 1.3)

        cumulative_drift = timestamps[-1] - timestamps[0] - (len(timestamps) - 1) * interval
        self.assertLess(abs(cumulative_drift), interval)

    async def test_wait_for_downbeat(self) -> None:
        beats_per_bar = 4
        clock = BeatClock(bpm=6000.0, beats_per_bar=beats_per_bar)
        await clock.start()

        bars: list[int] = []
        for _ in range(3):
            info = await clock.wait_for_downbeat()
            self.assertEqual(info.beat, 0)
            bars.append(info.bar)

        await clock.stop()

        self.assertEqual(bars, [0, 1, 2])

    async def test_stop_wakes_pending_waiter(self) -> None:
        clock = BeatClock(bpm=6000.0)
        await clock.start()

        async def waiter() -> None:
            await clock.wait_for_beat()

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.005)
        await clock.stop()

        with self.assertRaises(ClockStopped):
            await asyncio.wait_for(task, timeout=1.0)

    async def test_start_is_idempotent(self) -> None:
        clock = BeatClock(bpm=6000.0)
        await clock.start()
        first_task = clock._task
        await clock.start()
        self.assertIs(clock._task, first_task)
        await clock.stop()

    async def test_never_started_raises_clock_stopped(self) -> None:
        clock = BeatClock(bpm=6000.0)
        with self.assertRaises(ClockStopped):
            await clock.wait_for_beat()

    async def test_invalid_bpm_raises(self) -> None:
        with self.assertRaises(ValueError):
            BeatClock(bpm=0)
        with self.assertRaises(ValueError):
            BeatClock(bpm=-1)

    async def test_beat_info_fields(self) -> None:
        clock = BeatClock(bpm=6000.0, beats_per_bar=4)
        await clock.start()

        beats: list[BeatInfo] = []
        for _ in range(4):
            beats.append(await clock.wait_for_beat())

        await clock.stop()

        self.assertEqual(beats[0].bar, 0)
        self.assertEqual(beats[0].beat, 0)
        self.assertEqual(beats[1].beat, 1)
        self.assertEqual(beats[2].beat, 2)
        self.assertEqual(beats[3].beat, 3)