import unittest

from voyce.text_chunker import PhraseChunker


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


if __name__ == "__main__":
    unittest.main()
