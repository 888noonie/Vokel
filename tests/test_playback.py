import sys
import unittest
import asyncio

from vokel.playback import (
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

    def test_converts_markdown_image_file_links_to_speakable_hint(self):
        self.assertEqual(
            sanitize_for_speech("Here is [generated_image_20260528.png](sandbox:/mnt/data/generated_image_20260528.png)."),
            "Here is Image shown in transcript.",
        )

    def test_converts_bare_image_urls_to_speakable_hint(self):
        self.assertEqual(
            sanitize_for_speech("Here: https://example.com/generated_image_20260528.png"),
            "Here: Image shown in transcript.",
        )

    def test_converts_bare_gif_urls_to_speakable_hint(self):
        self.assertEqual(
            sanitize_for_speech("Here: https://media.giphy.com/media/abc/giphy.gif"),
            "Here: Image shown in transcript.",
        )

    def test_converts_markdown_images_to_speakable_caption(self):
        self.assertEqual(
            sanitize_for_speech("![sunset over mountains](https://images.unsplash.com/photo-abc)\nA beautiful view."),
            "sunset over mountains. Image shown in transcript. A beautiful view.",
        )

    def test_converts_gif_markdown_to_speakable_hint(self):
        self.assertEqual(
            sanitize_for_speech("![gif:mind blown](https://media.giphy.com/abc.gif)\nPowered by GIPHY"),
            "GIF shown in transcript. Powered by GIPHY",
        )

    def test_removes_serialized_tool_calls(self):
        self.assertEqual(
            sanitize_for_speech('Done. <|tool_call>call:search_image{query:<|"|>cat<|"|>}<tool_call|>'),
            "Done.",
        )

    def test_removes_analyze_image_tool_calls_with_spaces(self):
        self.assertEqual(
            sanitize_for_speech(
                'Looking. <|tool_call>call: analyze_image{image_description:<|"|>steaming mug<|"|>}<tool_call|> You are holding a mug.'
            ),
            "Looking. You are holding a mug.",
        )

    def test_removes_unclosed_parenthesised_tool_call(self):
        # A 4B model emitted exactly this (no closing token, () args) and it was spoken aloud.
        self.assertEqual(sanitize_for_speech("<|tool_call>call: camera.capture()"), "")
        self.assertEqual(
            sanitize_for_speech("Sure. <|tool_call>call: camera.capture() Let me look."),
            "Sure. Let me look.",
        )

    def test_strips_orphaned_tool_call_delimiters(self):
        self.assertEqual(sanitize_for_speech("Hold on <tool_call|> there."), "Hold on there.")
        self.assertEqual(
            sanitize_for_speech("I will call you back later."),
            "I will call you back later.",
        )

    def test_removes_no_opener_describe_image_tool_call(self):
        # A 4B model emitted exactly this for "what's in my hand" — no opener, {} args,
        # nested <|"|> quote tokens, and a trailing closer. The opener-required pattern
        # used to let the whole thing be spoken aloud.
        leak = 'call:describe_image{image_base64:<|"|>[Image data provided]<|"|>}<tool_call|>'
        self.assertEqual(sanitize_for_speech(leak), "")
        self.assertEqual(sanitize_for_speech(f"What am I holding. {leak}"), "What am I holding.")

    def test_keeps_natural_call_colon_phrasing(self):
        # "call:" without an opener or args is ordinary speech, not a tool call.
        self.assertEqual(
            sanitize_for_speech("Give me a call: tomorrow works best."),
            "Give me a call: tomorrow works best.",
        )


if __name__ == "__main__":
    unittest.main()
