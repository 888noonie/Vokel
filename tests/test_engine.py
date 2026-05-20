import asyncio
import unittest

from voyce.engine import ConversationEngine
from voyce.turns import PassthroughAsr, TextTurnProducer


class FakeLlm:
    def __init__(self, tokens, delay=0):
        self.tokens = tokens
        self.delay = delay

    async def stream_chat(self, messages):
        for token in self.tokens:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield token


class RecordingPlayback:
    def __init__(self):
        self.spoken = []
        self.stop_count = 0

    async def speak(self, phrase):
        self.spoken.append(phrase)

    async def stop(self):
        self.stop_count += 1


class ConversationEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_completed_turn_records_reply_and_speaks_phrases(self):
        llm = FakeLlm(["Hello, ", "Richard."])
        playback = RecordingPlayback()
        engine = ConversationEngine(llm=llm, playback=playback)

        await engine.start()
        try:
            await engine.submit_turn("Hello")
            await engine.wait_for_playback()
        finally:
            await engine.close()

        self.assertEqual(playback.spoken, ["Hello, Richard."])
        self.assertEqual(engine.history[-1], {"role": "assistant", "content": "Hello, Richard."})

    async def test_interrupt_cancels_generation_and_stops_playback(self):
        llm = FakeLlm(["This response will take a while. "], delay=1)
        playback = RecordingPlayback()
        engine = ConversationEngine(llm=llm, playback=playback)

        await engine.start()
        turn = asyncio.create_task(engine.submit_turn("Start"))
        await asyncio.sleep(0.01)
        await engine.interrupt()
        await engine.close()

        self.assertTrue(turn.cancelled() or turn.done())
        self.assertGreaterEqual(playback.stop_count, 1)

    async def test_run_turns_transcribes_and_submits_turn(self):
        llm = FakeLlm(["Ready."])
        playback = RecordingPlayback()
        engine = ConversationEngine(llm=llm, playback=playback)

        await engine.start()
        try:
            await engine.run_turns(
                producer=TextTurnProducer(["Start"]),
                asr=PassthroughAsr(),
                max_turns=1,
            )
        finally:
            await engine.close()

        self.assertIn({"role": "user", "content": "Start"}, engine.history)
        self.assertEqual(engine.history[-1], {"role": "assistant", "content": "Ready."})


if __name__ == "__main__":
    unittest.main()
