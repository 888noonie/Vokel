import unittest

from voyce.audio_profiles import get_audio_profile, list_audio_profiles


class AudioProfileTests(unittest.TestCase):
    def test_profiles_are_listed(self):
        names = [profile.name for profile in list_audio_profiles()]

        self.assertIn("laptop-mic-headphones", names)
        self.assertIn("headset-wired", names)
        self.assertIn("laptop-open", names)

    def test_unknown_profile_raises_useful_error(self):
        with self.assertRaisesRegex(ValueError, "Available profiles"):
            get_audio_profile("missing")


if __name__ == "__main__":
    unittest.main()
