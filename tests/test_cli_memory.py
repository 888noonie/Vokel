from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from voyce.cli import build_parser, run


class CliMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_remember_and_list_use_local_memory_without_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memory.sqlite3"

            remember_args = build_parser().parse_args(
                ["--memory-db", str(db_path), "--remember", "Avoid Bluetooth for benchmarks."]
            )
            list_args = build_parser().parse_args(["--memory-db", str(db_path), "--memory-list"])

            with patch("builtins.print"):
                await run(remember_args)
            with patch("builtins.print") as mock_print:
                await run(list_args)

        printed = [call.args[0] for call in mock_print.call_args_list]
        self.assertIn("1. Avoid Bluetooth for benchmarks.", printed)


if __name__ == "__main__":
    unittest.main()
