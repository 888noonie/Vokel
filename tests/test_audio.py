import unittest

from vokel.audio import (
    AudioDependencyError,
    MicVadConfig,
    SherpaOfflineAsrConfig,
    SileroVadTurnProducer,
    coerce_audio_device,
    _require_audio_dependencies,
)


class AudioTests(unittest.TestCase):
    def test_missing_vad_file_fails_before_audio_imports(self):
        with self.assertRaises(FileNotFoundError):
            SileroVadTurnProducer(MicVadConfig(vad_model_path="missing-silero-vad.onnx"))

    def test_audio_dependencies_are_optional(self):
        try:
            _require_audio_dependencies()
        except AudioDependencyError as exc:
            self.assertIn("optional audio dependencies", str(exc))

    def test_asr_config_requires_a_model_family_later(self):
        config = SherpaOfflineAsrConfig(tokens="tokens.txt")

        self.assertEqual(config.tokens, "tokens.txt")
        self.assertEqual(config.provider, "cpu")

    def test_coerce_audio_device_accepts_numeric_strings(self):
        self.assertEqual(coerce_audio_device("8"), 8)
        self.assertEqual(coerce_audio_device("pulse"), "pulse")
        self.assertIsNone(coerce_audio_device(None))


if __name__ == "__main__":
    unittest.main()
