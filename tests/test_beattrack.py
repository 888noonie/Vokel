import unittest

import numpy as np

from vokel.audio.beattrack import BeatTrackPlayer, SAMPLE_RATE


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

    def test_beat_bar_has_energy_on_each_beat_offset(self) -> None:
        player = BeatTrackPlayer(bpm=120.0, style="beat")
        buffer = player._render_bar()
        beat_samples = max(1, int(SAMPLE_RATE * 60.0 / player.bpm))
        for beat in range(player.beats_per_bar):
            start = beat * beat_samples
            window = buffer[start : start + min(64, beat_samples)]
            self.assertGreater(float(np.max(np.abs(window))), 0.01)

    def test_metronome_bar_has_energy_on_each_beat_offset(self) -> None:
        player = BeatTrackPlayer(bpm=120.0, style="metronome")
        buffer = player._render_bar()
        beat_samples = max(1, int(SAMPLE_RATE * 60.0 / player.bpm))
        for beat in range(player.beats_per_bar):
            start = beat * beat_samples
            window = buffer[start : start + min(64, beat_samples)]
            self.assertGreater(float(np.max(np.abs(window))), 0.01)

    def test_metronome_bar_differs_from_beat_bar(self) -> None:
        beat_player = BeatTrackPlayer(bpm=120.0, style="beat")
        metronome_player = BeatTrackPlayer(bpm=120.0, style="metronome")
        beat_bar = beat_player._render_bar()
        metronome_bar = metronome_player._render_bar()
        self.assertFalse(np.allclose(beat_bar, metronome_bar))

    def test_load_buffer_loops_external_track(self) -> None:
        player = BeatTrackPlayer(bpm=120.0)
        samples = np.linspace(-0.5, 0.5, 2400, dtype=np.float32)
        player.load_buffer(samples)
        self.assertTrue(player._external_buffer)
        self.assertEqual(player._cursor, 0)
        self.assertEqual(len(player._buffer), len(samples))

    def test_clear_buffer_falls_back_to_synthesized_bar(self) -> None:
        player = BeatTrackPlayer(bpm=120.0, style="beat")
        samples = np.ones(2400, dtype=np.float32) * 0.5
        player.load_buffer(samples)
        player._stream = object()
        self.assertTrue(player._external_buffer)

        player.clear_buffer()
        self.assertFalse(player._external_buffer)
        self.assertEqual(player._cursor, 0)
        self.assertIsNotNone(player._buffer)
        self.assertGreater(float(np.max(np.abs(player._buffer))), 0.01)

    def test_load_buffer_respects_level_scaling(self) -> None:
        player = BeatTrackPlayer(bpm=120.0, level=0.25)
        samples = np.ones(120, dtype=np.float32) * 0.8
        player.load_buffer(samples)
        scaled = float(player._buffer[0] * player._level)
        self.assertAlmostEqual(scaled, 0.2, places=5)

    def test_stop_clears_loaded_buffer(self) -> None:
        import asyncio

        player = BeatTrackPlayer(bpm=120.0)
        player.load_buffer(np.ones(120, dtype=np.float32))
        asyncio.run(player.stop())
        self.assertIsNone(player._buffer)
        self.assertFalse(player._external_buffer)

    def test_nudge_cursor_shifts_by_milliseconds(self) -> None:
        player = BeatTrackPlayer(bpm=120.0)
        player.load_buffer(np.arange(2400, dtype=np.float32))
        player._cursor = 0
        player.nudge_cursor(25.0)
        expected = int(round(25.0 * SAMPLE_RATE / 1000.0))
        self.assertEqual(player._cursor, expected % 2400)

    def test_nudge_cursor_wraps_around_buffer_length(self) -> None:
        player = BeatTrackPlayer(bpm=120.0)
        player.load_buffer(np.ones(1000, dtype=np.float32))
        player._cursor = 990
        player.nudge_cursor(25.0)
        shift = int(round(25.0 * SAMPLE_RATE / 1000.0))
        self.assertEqual(player._cursor, (990 + shift) % 1000)