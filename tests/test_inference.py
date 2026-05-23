import json
import unittest

from vokel.inference import parse_sse_delta


class InferenceParsingTests(unittest.TestCase):
    def test_parses_content_token(self):
        payload = {"choices": [{"delta": {"content": "hello"}}]}

        self.assertEqual(parse_sse_delta(f"data: {json.dumps(payload)}"), {"content": "hello"})

    def test_ignores_done_line(self):
        self.assertIsNone(parse_sse_delta("data: [DONE]"))

    def test_ignores_non_data_lines(self):
        self.assertIsNone(parse_sse_delta(": keepalive"))


if __name__ == "__main__":
    unittest.main()
