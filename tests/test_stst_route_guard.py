import argparse
import unittest
from unittest.mock import patch

from benchmarks.stst_latency import run_mic_lm_studio
from vokel.audio_routes import AudioSource


class StstRouteGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_require_profile_route_rejects_unavailable_source(self):
        args = argparse.Namespace(
            audio_profile="headset-wired",
            require_profile_route=True,
        )
        source = AudioSource(
            index=1,
            name="alsa_input.pci-0000_35_00.6.HiFi__Mic2__source",
            description="Headset Mic",
            state="SUSPENDED",
            active_port="[In] Mic2",
            active_port_available="not available",
            is_monitor=False,
            is_default=False,
        )

        with patch("vokel.audio_routes.load_pulse_sources", return_value=[source]):
            with self.assertRaisesRegex(RuntimeError, "not usable"):
                await run_mic_lm_studio(args)


if __name__ == "__main__":
    unittest.main()
