import unittest

from voyce.telemetry import LatencyTrace


class LatencyTraceTests(unittest.TestCase):
    def test_summary_reports_known_durations(self):
        ticks = iter(
            [
                1_000_000_000,
                1_050_000_000,
                1_060_000_000,
                1_110_000_000,
                1_120_000_000,
                1_130_000_000,
                1_190_000_000,
                1_250_000_000,
                1_260_000_000,
                1_300_000_000,
                1_320_000_000,
                1_360_000_000,
            ]
        )
        trace = LatencyTrace(clock=lambda: next(ticks))

        trace.mark("capture_started")
        trace.mark("capture_finished")
        trace.mark("asr_started")
        trace.mark("asr_finished")
        trace.mark("turn_submitted")
        trace.mark("generation_started")
        trace.mark("first_token")
        trace.mark("first_phrase_queued")
        trace.mark("playback_started")
        trace.mark("generation_finished")
        trace.mark("interruption_requested")
        trace.mark("playback_stop_requested")

        self.assertEqual(
            trace.summary_ms(),
            {
                "capture_duration": 50.0,
                "capture_to_first_token": 190.0,
                "capture_to_playback_start": 260.0,
                "asr_duration": 50.0,
                "asr_to_first_token": 80.0,
                "turn_to_first_token": 70.0,
                "turn_to_first_phrase": 130.0,
                "turn_to_playback_start": 140.0,
                "generation_duration": 170.0,
                "turn_duration": 180.0,
                "turn_to_interruption": 200.0,
                "generation_to_interruption": 190.0,
                "interruption_to_playback_stop": 40.0,
                "turn_to_playback_stop": 240.0,
            },
        )

    def test_reset_removes_events(self):
        trace = LatencyTrace(clock=lambda: 1)

        trace.mark("turn_submitted")
        trace.reset()

        self.assertEqual(trace.events, ())


if __name__ == "__main__":
    unittest.main()
