import io
import unittest
from contextlib import redirect_stdout

from voyce.progress import ConsoleProgressObserver
from voyce.telemetry import LatencyTrace


class ProgressObserverTests(unittest.TestCase):
    def test_console_observer_prints_listen_cue(self):
        trace = LatencyTrace(clock=lambda: 1)
        trace.add_observer(ConsoleProgressObserver())
        output = io.StringIO()

        with redirect_stdout(output):
            trace.mark("capture_started")

        self.assertIn("Start speaking", output.getvalue())

    def test_json_style_without_observer_stays_quiet(self):
        trace = LatencyTrace(clock=lambda: 1)
        output = io.StringIO()

        with redirect_stdout(output):
            trace.mark("capture_started")

        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
