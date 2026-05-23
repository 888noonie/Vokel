import sys
import unittest
import asyncio

from voyce.playback import (
    SubprocessPlaybackConfig,
    SubprocessPlaybackSink,
    available_playback_backends,
    build_playback_sink,
    sanitize_for_speech,
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


class SpeechSanitizerTests(unittest.TestCase):
    def test_removes_markdown_symbols_without_losing_words(self):
        self.assertEqual(
            sanitize_for_speech("* **Important** `tool_call` result"),
            "Important tool call result",
        )

    def test_replaces_urls_with_transcript_hint(self):
        self.assertEqual(
            sanitize_for_speech("1. BBC headline\n   https://www.bbc.com/news/example"),
            "1. BBC headline Link available in transcript.",
        )

    def test_converts_markdown_links_to_speakable_text(self):
        self.assertEqual(
            sanitize_for_speech("Read [BBC News](https://www.bbc.com/news)."),
            "Read BBC News. Link available in transcript.",
        )


if __name__ == "__main__":
    unittest.main()
