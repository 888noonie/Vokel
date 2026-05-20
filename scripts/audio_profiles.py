from __future__ import annotations

from voyce.audio_profiles import list_audio_profiles


def main() -> None:
    for profile in list_audio_profiles():
        mic = profile.mic
        print(f"{profile.name}: {profile.description}")
        print(
            f"  device={mic.device} threshold={mic.threshold} "
            f"gain={mic.input_gain} silence={mic.min_silence_duration}s"
        )
        if profile.preferred_source_name:
            print(f"  preferred_source={profile.preferred_source_name}")


if __name__ == "__main__":
    main()
