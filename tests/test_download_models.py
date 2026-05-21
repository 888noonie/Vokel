import unittest

from scripts.download_models import ASSETS, build_parser


class DownloadModelsTests(unittest.TestCase):
    def test_list_flag_parses_without_positional_models(self):
        args = build_parser().parse_args(["--list"])

        self.assertTrue(args.list)
        self.assertEqual(args.models, [])

    def test_streaming_zipformer_asset_registered(self) -> None:
        self.assertIn("streaming-zipformer-en", ASSETS)

    def test_known_assets_are_registered(self) -> None:
        self.assertIn("silero-vad", ASSETS)
        self.assertIn("sense-voice-int8", ASSETS)


if __name__ == "__main__":
    unittest.main()
