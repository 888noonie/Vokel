import unittest

import numpy as np

from vokel.audio.beattrack import BeatTrackPlayer


class BeatTrackPlayerTests(unittest.TestCase):
    def test_set_level_clamps(self) -> None:
        player = BeatTrackPlayer(bpm=120.0)
        player.set_level(1.5)
        self.assertEqual(player._level, 1.0)
        player.set_level(-0.25)
        self.assertEqual(player._level, 0.0)
        player.set_level(0.42)
        self.assertEqual(player._level, 0.42)

    def test_init_level_is_applied(self) -> None:
        player = BeatTrackPlayer(bpm=120.0, level=0.35)
        self.assertEqual(player._level, 0.35)

    def test_callback_scaling_halves_amplitude_at_half_level(self) -> None:
        player = BeatTrackPlayer(bpm=120.0)
        player._buffer = player._render_bar()
        reference = float(np.max(np.abs(player._buffer)))
        self.assertGreater(reference, 0.0)

        player.set_level(0.5)
        scaled_peak = float(np.max(np.abs(player._buffer * player._level)))
        self.assertAlmostEqual(scaled_peak, reference * 0.5, places=5)

    def test_callback_scaling_is_silent_at_zero_level(self) -> None:
        player = BeatTrackPlayer(bpm=120.0)
        player._buffer = player._render_bar()
        player.set_level(0.0)
        scaled_peak = float(np.max(np.abs(player._buffer * player._level)))
        self.assertEqual(scaled_peak, 0.0)