import unittest

from vokel.text_chunker import PhraseChunker


class PhraseChunkerTests(unittest.TestCase):
    def test_flushes_on_clause_boundary_after_minimum_length(self):
        chunker = PhraseChunker(min_chars=5)

        self.assertEqual(chunker.push("Hello"), [])
        self.assertEqual(chunker.push(", friend"), ["Hello,"])
        self.assertEqual(chunker.flush(), "friend")

    def test_flushes_on_max_chars_without_punctuation(self):
        chunker = PhraseChunker(min_chars=50, max_chars=10)

        self.assertEqual(chunker.push("abcdefghij"), ["abcdefghij"])
        self.assertIsNone(chunker.flush())

    def test_reset_discards_partial_phrase(self):
        chunker = PhraseChunker(min_chars=5)

        self.assertEqual(chunker.push("partial"), [])
        chunker.reset()
        self.assertIsNone(chunker.flush())

    def test_does_not_flush_inside_streamed_https_image_url(self):
        chunker = PhraseChunker(min_chars=5)

        self.assertEqual(chunker.push("Image: https://imgen.x.ai/xai-imgen/file.jpeg"), [])
        self.assertEqual(chunker.push(". Nice."), ["Image: https://imgen.x.ai/xai-imgen/file.jpeg. Nice."])
        self.assertIsNone(chunker.flush())

    def test_does_not_flush_inside_streamed_gif_url(self):
        chunker = PhraseChunker(min_chars=5)

        self.assertEqual(chunker.push("GIF: https://media.giphy.com/media/abc/giphy.gif"), [])
        self.assertEqual(chunker.push(". Perfect."), ["GIF: https://media.giphy.com/media/abc/giphy.gif. Perfect."])
        self.assertIsNone(chunker.flush())

    def test_does_not_flush_inside_streamed_markdown_image_link(self):
        chunker = PhraseChunker(min_chars=5)

        self.assertEqual(chunker.push("Here is [image.png](sandbox:/mnt/data/image.png"), [])
        self.assertEqual(chunker.push("). Done."), ["Here is [image.png](sandbox:/mnt/data/image.png). Done."])
        self.assertIsNone(chunker.flush())


if __name__ == "__main__":
    unittest.main()
