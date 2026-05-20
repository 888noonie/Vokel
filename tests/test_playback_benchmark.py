import unittest

from benchmarks.playback_latency import run_benchmark


class PlaybackBenchmarkTests(unittest.IsolatedAsyncioTestCase):
    async def test_fake_backend_reports_stop_latency(self):
        result = await run_benchmark("fake", interrupt_after_ms=20)

        self.assertEqual(result.backend, "fake")
        self.assertLess(result.stop_latency_ms, 1000)


if __name__ == "__main__":
    unittest.main()
