import unittest

from benchmarks.stst_latency import (
    BenchmarkPlaybackSink,
    DelayedAsr,
    ScriptedLlm,
    run_engine_benchmark,
)
from voyce.turns import TextTurnProducer


class StstBenchmarkTests(unittest.IsolatedAsyncioTestCase):
    async def test_synthetic_benchmark_records_core_metrics(self):
        result = await run_engine_benchmark(
            name="test",
            llm=ScriptedLlm(["Hello, ", "measured world."], first_token_ms=0, token_gap_ms=0),
            asr=DelayedAsr(delay_ms=0),
            playback=BenchmarkPlaybackSink(first_audio_ms=0),
            producer=TextTurnProducer(["Hello"]),
        )

        self.assertEqual(result.name, "test")
        self.assertIn("asr_duration", result.summary_ms)
        self.assertIn("turn_to_first_token", result.summary_ms)
        self.assertIn("turn_to_playback_start", result.summary_ms)
        self.assertTrue(result.events)


if __name__ == "__main__":
    unittest.main()
