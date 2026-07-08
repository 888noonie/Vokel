import asyncio
import unittest
from collections.abc import AsyncIterator, Sequence
from typing import Any

from vokel.audio.beatclock import BeatClock
from vokel.audio.quantized_sink import QuantizedPlaybackSink
from vokel.engine import ConversationEngine
from vokel.telemetry import LatencyTrace
from vokel.events import Event, TextDeltaEvent
from vokel.inference import ChatMessage
from vokel.playback import PlaybackSink


class FakeSink:
    def __init__(self) -> None:
        self.spoken: list[tuple[str, float]] = []
        self.stop_count = 0

    async def speak(self, phrase: str) -> None:
        loop = asyncio.get_running_loop()
        self.spoken.append((phrase, loop.time()))

    async def stop(self) -> None:
        self.stop_count += 1


class FakeLlm:
    def __init__(self, events: list[Event], delay: float = 0) -> None:
        self.events = events
        self.delay = delay

    async def stream_chat(
        self, messages: Sequence[ChatMessage], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[Event]:
        for event in self.events:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield event

    async def cancel_active(self) -> None:
        return None

    async def __aenter__(self) -> "FakeLlm":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


class QuantizedPlaybackSinkTests(unittest.IsolatedAsyncioTestCase):
    async def test_phrases_release_on_grid_points(self) -> None:
        bpm = 6000.0
        interval = 60.0 / bpm
        clock = BeatClock(bpm=bpm)
        await clock.start()

        inner = FakeSink()
        sink = QuantizedPlaybackSink(inner, clock)

        beat_info = await clock.wait_for_beat()
        loop = asyncio.get_running_loop()
        mid_beat_at = beat_info.at + interval * 0.4
        delay = mid_beat_at - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)

        parked_at = loop.time()
        await sink.speak("on grid")

        await clock.stop()

        self.assertEqual(len(inner.spoken), 1)
        _phrase, spoken_at = inner.spoken[0]
        wait_duration = spoken_at - parked_at
        self.assertGreater(wait_duration, interval * 0.4)
        self.assertLess(wait_duration, interval * 1.1)

    async def test_stop_while_parked_prevents_speak(self) -> None:
        clock = BeatClock(bpm=6000.0)
        await clock.start()

        inner = FakeSink()
        sink = QuantizedPlaybackSink(inner, clock)

        speak_task = asyncio.create_task(sink.speak("parked"))
        await asyncio.sleep(0.002)
        await sink.stop()
        await speak_task

        await clock.stop()

        self.assertEqual(inner.spoken, [])
        self.assertEqual(inner.stop_count, 1)

    async def test_clock_dead_still_speaks(self) -> None:
        clock = BeatClock(bpm=6000.0)
        inner = FakeSink()
        sink = QuantizedPlaybackSink(inner, clock)

        loop = asyncio.get_running_loop()
        before = loop.time()
        await sink.speak("unquantized")
        after = loop.time()

        self.assertEqual(inner.spoken, [("unquantized", inner.spoken[0][1])])
        self.assertLess(after - before, 0.05)

    async def test_sequential_speaks_wait_for_own_grid_points(self) -> None:
        bpm = 6000.0
        interval = 60.0 / bpm
        clock = BeatClock(bpm=bpm)
        await clock.start()

        inner = FakeSink()
        sink = QuantizedPlaybackSink(inner, clock)

        await sink.speak("one")
        await sink.speak("two")
        await sink.speak("three")

        await clock.stop()

        self.assertEqual([phrase for phrase, _at in inner.spoken], ["one", "two", "three"])
        timestamps = [at for _phrase, at in inner.spoken]
        for i in range(1, len(timestamps)):
            delta = timestamps[i] - timestamps[i - 1]
            self.assertGreaterEqual(delta, interval * 0.7)
            self.assertLessEqual(delta, interval * 1.3)

    async def test_bar_quantum_waits_for_downbeat(self) -> None:
        clock = BeatClock(bpm=6000.0, beats_per_bar=4)
        await clock.start()

        inner = FakeSink()
        sink = QuantizedPlaybackSink(inner, clock, quantum="bar")

        await clock.wait_for_beat()
        await clock.wait_for_beat()

        speak_task = asyncio.create_task(sink.speak("bar aligned"))
        await speak_task

        await clock.stop()

        self.assertEqual(len(inner.spoken), 1)

    def test_protocol_conformance(self) -> None:
        clock = BeatClock(bpm=90.0)
        inner = FakeSink()
        sink = QuantizedPlaybackSink(inner, clock)

        def accepts_playback_sink(candidate: PlaybackSink) -> bool:
            return callable(getattr(candidate, "speak", None)) and callable(
                getattr(candidate, "stop", None)
            )

        self.assertTrue(accepts_playback_sink(sink))

    async def test_trace_marks_gate_on_release_path(self) -> None:
        clock = BeatClock(bpm=6000.0)
        await clock.start()

        inner = FakeSink()
        trace = LatencyTrace()
        sink = QuantizedPlaybackSink(inner, clock, trace=trace)

        await sink.speak("on grid")
        await clock.stop()

        self.assertIsNotNone(trace.first("musical_gate_entered"))
        self.assertIsNotNone(trace.first("musical_gate_opened"))
        self.assertNotIn("reason", trace.first("musical_gate_opened").fields)

    async def test_trace_no_open_mark_on_stop_path(self) -> None:
        clock = BeatClock(bpm=6000.0)
        await clock.start()

        inner = FakeSink()
        trace = LatencyTrace()
        sink = QuantizedPlaybackSink(inner, clock, trace=trace)

        speak_task = asyncio.create_task(sink.speak("parked"))
        await asyncio.sleep(0.002)
        await sink.stop()
        await speak_task
        await clock.stop()

        self.assertIsNotNone(trace.first("musical_gate_entered"))
        self.assertIsNone(trace.first("musical_gate_opened"))

    async def test_trace_clock_stopped_marks_open_with_reason(self) -> None:
        clock = BeatClock(bpm=6000.0)
        inner = FakeSink()
        trace = LatencyTrace()
        sink = QuantizedPlaybackSink(inner, clock, trace=trace)

        await sink.speak("unquantized")

        self.assertIsNotNone(trace.first("musical_gate_entered"))
        opened = trace.first("musical_gate_opened")
        self.assertIsNotNone(opened)
        self.assertEqual(opened.fields.get("reason"), "clock_stopped")

    async def test_trace_none_leaves_byte_identical_behavior(self) -> None:
        clock = BeatClock(bpm=6000.0)
        await clock.start()

        inner = FakeSink()
        sink = QuantizedPlaybackSink(inner, clock, trace=None)

        await sink.speak("on grid")
        await clock.stop()

        self.assertEqual(len(inner.spoken), 1)

    async def test_engine_interrupt_with_quantized_sink_no_hang(self) -> None:
        clock = BeatClock(bpm=30.0)
        await clock.start()

        inner = FakeSink()
        sink = QuantizedPlaybackSink(inner, clock)
        llm: Any = FakeLlm([TextDeltaEvent("Queued on the grid.")])
        engine = ConversationEngine(llm=llm, playback=sink)

        await engine.start()
        try:
            turn = asyncio.create_task(engine.submit_turn("Go"))
            await asyncio.sleep(0.05)
            await engine.interrupt()
            await asyncio.wait_for(turn, timeout=2.0)
        finally:
            await engine.close()
            await clock.stop()

        self.assertEqual(inner.spoken, [])
        self.assertGreaterEqual(inner.stop_count, 1)