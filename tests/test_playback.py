import sys
import unittest
import asyncio

from voyce.playback import (
    SubprocessPlaybackConfig,
    SubprocessPlaybackSink,
    available_playback_backends,
    build_playback_sink,
)


class PlaybackTests(unittest.IsolatedAsyncioTestCase):
    async def test_subprocess_sink_can_stop_active_process(self):
        sink = SubprocessPlaybackSink(
            SubprocessPlaybackConfig(command=(sys.executable, "-c", "import time; time.sleep(10)"))
        )

        task = asyncio.create_task(sink.speak("ignored"))
        await asyncio.sleep(0.05)
        await sink.stop()
        await asyncio.wait_for(task, timeout=2)


class PlaybackFactoryTests(unittest.TestCase):
    def test_console_backend_is_always_available(self):
        self.assertIn("console", available_playback_backends())
        self.assertIsNotNone(build_playback_sink("console"))


if __name__ == "__main__":
    unittest.main()
