import unittest

from voyce.turns import AudioTurn, PassthroughAsr, TextTurnProducer


class TurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_turn_producer_yields_audio_turn(self):
        producer = TextTurnProducer(["hello"])

        turn = await producer.next_turn()

        self.assertEqual(turn, AudioTurn(text="hello"))
        with self.assertRaises(StopAsyncIteration):
            await producer.next_turn()

    async def test_passthrough_asr_returns_text(self):
        asr = PassthroughAsr()

        self.assertEqual(await asr.transcribe(AudioTurn(text="hello")), "hello")


if __name__ == "__main__":
    unittest.main()
