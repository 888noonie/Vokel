import unittest

from scripts.audio_probe import rms_dbfs


class AudioProbeTests(unittest.TestCase):
    def test_silence_reports_floor(self):
        self.assertEqual(rms_dbfs([]), -120.0)

    def test_nonzero_signal_reports_below_zero_dbfs(self):
        value = rms_dbfs([0.5, -0.5])

        self.assertLess(value, 0)
        self.assertGreater(value, -10)


if __name__ == "__main__":
    unittest.main()
