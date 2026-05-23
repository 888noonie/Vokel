from __future__ import annotations

from dataclasses import dataclass

from .audio import MicVadConfig


@dataclass(frozen=True)
class AudioProfile:
    name: str
    description: str
    mic: MicVadConfig
    preferred_source_name: str | None = None


PROFILES = {
    "laptop-mic-headphones": AudioProfile(
        name="laptop-mic-headphones",
        description=(
            "Recommended dev baseline: stable laptop digital mic input with "
            "headphone-isolated playback."
        ),
        preferred_source_name="alsa_input.pci-0000_35_00.6.HiFi__Mic1__source",
        mic=MicVadConfig(
            device=7,
            input_gain=2.0,
            threshold=0.35,
            min_silence_duration=0.35,
            min_speech_duration=0.1,
        ),
    ),
    "laptop-open": AudioProfile(
        name="laptop-open",
        description="Laptop mic or room mic with DC offset correction and stronger gain.",
        preferred_source_name="alsa_input.pci-0000_35_00.6.HiFi__Mic1__source",
        mic=MicVadConfig(
            device=8,
            input_gain=4.0,
            threshold=0.25,
            min_silence_duration=0.35,
            min_speech_duration=0.1,
        ),
    ),
    "headset-wired": AudioProfile(
        name="headset-wired",
        description="Wired headset/headphone mic baseline with stricter VAD.",
        preferred_source_name="alsa_input.pci-0000_35_00.6.HiFi__Mic2__source",
        mic=MicVadConfig(
            device=7,
            input_gain=2.0,
            threshold=0.35,
            min_silence_duration=0.35,
            min_speech_duration=0.1,
        ),
    ),
    "headset-bluetooth": AudioProfile(
        name="headset-bluetooth",
        description="Bluetooth headset profile; tune after selecting the active Pulse/PipeWire source.",
        preferred_source_name=None,
        mic=MicVadConfig(
            device=7,
            input_gain=2.0,
            threshold=0.4,
            min_silence_duration=0.45,
            min_speech_duration=0.15,
        ),
    ),
    "noisy-handset": AudioProfile(
        name="noisy-handset",
        description="Noisier open-air profile with stricter speech confirmation.",
        preferred_source_name="alsa_input.pci-0000_35_00.6.HiFi__Mic1__source",
        mic=MicVadConfig(
            device=8,
            input_gain=3.0,
            threshold=0.45,
            min_silence_duration=0.5,
            min_speech_duration=0.2,
        ),
    ),
}


def get_audio_profile(name: str) -> AudioProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        names = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown audio profile: {name}. Available profiles: {names}") from exc


def list_audio_profiles() -> list[AudioProfile]:
    return [PROFILES[name] for name in sorted(PROFILES)]
