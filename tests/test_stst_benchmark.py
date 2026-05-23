import sys
import unittest
from unittest.mock import Mock, patch

from benchmarks.stst_latency import (
    BenchmarkPlaybackSink,
    DelayedAsr,
    ScriptedLlm,
    build_parser,
    run_engine_benchmark,
)
from vokel.telemetry import LatencyTrace
from vokel.turns import TextTurnProducer


class StstBenchmarkTests(unittest.IsolatedAsyncioTestCase):
    async def test_synthetic_benchmark_records_core_metrics(self) -> None:
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

    def test_stst_latency_parser_supports_streaming_mic_mode(self) -> None:
        args = build_parser().parse_args(
            ["--mode", "mic-streaming-lm-studio", "--no-progress", "--streaming-asr-dir", "models/z"]
        )
        self.assertEqual(args.mode, "mic-streaming-lm-studio")
        self.assertEqual(args.streaming_asr_dir, "models/z")

    async def test_run_engine_benchmark_reuses_shared_trace(self) -> None:
        trace = LatencyTrace()
        trace.mark("capture_started")
        result = await run_engine_benchmark(
            name="trace-reuse",
            llm=ScriptedLlm(["Hi"], first_token_ms=0, token_gap_ms=0),
            asr=DelayedAsr(delay_ms=0),
            playback=BenchmarkPlaybackSink(first_audio_ms=0),
            producer=TextTurnProducer(["Hello"]),
            progress=False,
            trace=trace,
        )
        names = [e["name"] for e in result.events]
        self.assertIn("capture_started", names)
        self.assertEqual(result.name, "trace-reuse")

    def test_stst_latency_parser_supports_no_progress(self) -> None:
        args = build_parser().parse_args(["--mode", "synthetic", "--no-progress"])
        self.assertFalse(args.progress)

    @patch("benchmarks.stst_latency.async_main")
    def test_main_json_disables_progress(self, mock_async_main: Mock) -> None:
        mock_result = Mock()
        mock_result.to_json.return_value = '{"name":"test"}'
        mock_async_main.return_value = mock_result

        with patch.object(sys, "argv", ["stst_latency.py", "--mode", "synthetic", "--json"]):
            with patch("builtins.print") as mock_print:
                from benchmarks.stst_latency import main

                main()

        self.assertFalse(mock_async_main.call_args.args[0].progress)
        mock_print.assert_called_with('{"name":"test"}')


if __name__ == "__main__":
    unittest.main()
