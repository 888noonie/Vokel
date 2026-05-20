import unittest

from voyce.telemetry import LatencyTrace


class LatencyTraceTests(unittest.TestCase):
    def test_summary_reports_known_durations(self):
        ticks = iter(
            [
                1_000_000_000,
                1_050_000_000,
                1_060_000_000,
                1_180_000_000,
                1_240_000_000,
                1_300_000_000,
            ]
        )
        trace = LatencyTrace(clock=lambda: next(ticks))

        trace.mark("asr_started")
        trace.mark("asr_finished")
        trace.mark("turn_submitted")
        trace.mark("first_token")
        trace.mark("first_phrase_queued")
        trace.mark("generation_finished")

        self.assertEqual(
            trace.summary_ms(),
            {
                "asr_duration": 50.0,
                "asr_to_first_token": 130.0,
                "turn_to_first_token": 120.0,
                "turn_to_first_phrase": 180.0,
                "turn_duration": 240.0,
            },
        )

    def test_reset_removes_events(self):
        trace = LatencyTrace(clock=lambda: 1)

        trace.mark("turn_submitted")
        trace.reset()

        self.assertEqual(trace.events, ())


if __name__ == "__main__":
    unittest.main()
