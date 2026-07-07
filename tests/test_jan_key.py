import unittest
from unittest.mock import patch

from vokel.jan_key import discover_jan_api_key, jan_port_from_url, resolve_llm_api_key


class JanKeyTests(unittest.TestCase):
    def test_jan_port_from_url(self) -> None:
        self.assertEqual(
            jan_port_from_url("http://127.0.0.1:6767/v1/chat/completions"),
            6767,
        )
        self.assertIsNone(jan_port_from_url("http://localhost:1234/v1/chat/completions"))

    @patch("vokel.jan_key.subprocess.check_output")
    def test_discover_jan_api_key(self, mock_output) -> None:
        mock_output.return_value = (
            "123 llama-server --host 127.0.0.1 --port 6767 --api-key abc-def-123\n"
        )
        self.assertEqual(discover_jan_api_key(6767), "abc-def-123")

    def test_resolve_prefers_explicit_key(self) -> None:
        self.assertEqual(
            resolve_llm_api_key(
                "http://127.0.0.1:6767/v1/chat/completions",
                explicit_key="from-ui",
                env_key="from-env",
            ),
            "from-ui",
        )


if __name__ == "__main__":
    unittest.main()
