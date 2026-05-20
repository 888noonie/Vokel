import unittest

from scripts.vad_probe import rms_dbfs


class VadProbeTests(unittest.TestCase):
    def test_empty_signal_reports_floor(self):
        self.assertEqual(rms_dbfs([]), -120.0)

    def test_signal_reports_dbfs(self):
        self.assertGreater(rms_dbfs([0.25, -0.25]), -20)


if __name__ == "__main__":
    unittest.main()
