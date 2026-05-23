import json
import unittest

from vokel.audio_routes import find_source, parse_pulse_sources_json


class AudioRoutesTests(unittest.TestCase):
    def test_parse_sources_marks_default_and_availability(self):
        raw = json.dumps(
            [
                {
                    "index": 1,
                    "state": "SUSPENDED",
                    "name": "mic1",
                    "description": "Internal Mic",
                    "active_port": "p1",
                    "ports": {"p1": {"available": "unknown"}},
                },
                {
                    "index": 2,
                    "state": "SUSPENDED",
                    "name": "mic2",
                    "description": "Headset Mic",
                    "active_port": "p2",
                    "ports": [{"name": "p2", "availability": "not available"}],
                },
            ]
        )

        sources = parse_pulse_sources_json(raw, default_name="mic1")

        self.assertTrue(sources[0].is_default)
        self.assertTrue(sources[0].usable_input)
        self.assertFalse(sources[1].usable_input)
        self.assertEqual(find_source(sources, "mic2"), sources[1])


if __name__ == "__main__":
    unittest.main()
