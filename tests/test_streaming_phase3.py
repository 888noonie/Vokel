"""Phase 3 (streaming ASR) regression tests — no live models required."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from vokel.audio import SherpaOnlineAsr, SherpaOnlineAsrConfig, create_streaming_asr
from vokel.turns import AudioTurn


class StreamingPhase3Tests(unittest.IsolatedAsyncioTestCase):
    def test_create_streaming_asr_missing_dir_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            create_streaming_asr(Path("/nonexistent/zipformer-dir-xyz"))

    @patch("vokel.audio._require_audio_dependencies")
    @patch.object(Path, "is_file", return_value=True)
    async def test_sherpa_online_transcribe_returns_prefilled_text(
        self, _mock_is_file: MagicMock, mock_deps: MagicMock
    ) -> None:
        np = MagicMock()
        sd = MagicMock()
        sherpa = MagicMock()
        mock_deps.return_value = (np, sd, sherpa)

        recognizer = MagicMock()
        stream = MagicMock()
        stream.result.text = "from offline decode"
        recognizer.create_stream.return_value = stream
        sherpa.OnlineRecognizer.from_transducer.return_value = recognizer

        asr = SherpaOnlineAsr(
            SherpaOnlineAsrConfig(
                tokens="t.txt",
                encoder="e.onnx",
                decoder="d.onnx",
                joiner="j.onnx",
            )
        )

        out = await asr.transcribe(
            AudioTurn(text="hello from stream", sample_rate=16_000, audio_samples=(0.0, 0.1))
        )
        self.assertEqual(out, "hello from stream")


if __name__ == "__main__":
    unittest.main()
