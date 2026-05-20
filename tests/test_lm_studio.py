import json
import unittest

from voyce.lm_studio import parse_sse_token


class LmStudioParsingTests(unittest.TestCase):
    def test_parses_content_token(self):
        payload = {"choices": [{"delta": {"content": "hello"}}]}

        self.assertEqual(parse_sse_token(f"data: {json.dumps(payload)}"), "hello")

    def test_ignores_done_line(self):
        self.assertIsNone(parse_sse_token("data: [DONE]"))

    def test_ignores_non_data_lines(self):
        self.assertIsNone(parse_sse_token(": keepalive"))


if __name__ == "__main__":
    unittest.main()
