from __future__ import annotations

import argparse

from vokel.audio_profiles import get_audio_profile, list_audio_profiles
from vokel.audio_routes import find_source, load_pulse_sources, set_default_source


def print_sources(profile_name: str | None = None) -> int:
    sources = load_pulse_sources()
    wanted_source = None
    if profile_name:
        wanted_source = get_audio_profile(profile_name).preferred_source_name

    exit_code = 0
    for source in sources:
        if source.is_monitor:
            continue
        markers = []
        if source.is_default:
            markers.append("default")
        if source.name == wanted_source:
            markers.append("profile")
            if not source.usable_input:
                exit_code = 2
        marker_text = f" [{' '.join(markers)}]" if markers else ""
        print(f"{source.index}: {source.description}{marker_text}")
        print(f"  name={source.name}")
        print(
            f"  state={source.state} port={source.active_port} "
            f"available={source.active_port_available} usable={source.usable_input}"
        )

    if wanted_source and not find_source(sources, wanted_source):
        print(f"profile_source_missing={wanted_source}")
        exit_code = 2
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and select Pulse/PipeWire input routes.")
    parser.add_argument("--profile", choices=[profile.name for profile in list_audio_profiles()])
    parser.add_argument("--set-profile-default", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.set_profile_default:
        if not args.profile:
            raise SystemExit("--set-profile-default requires --profile")
        source_name = get_audio_profile(args.profile).preferred_source_name
        if not source_name:
            raise SystemExit(f"Profile {args.profile} has no preferred source")
        set_default_source(source_name)

    raise SystemExit(print_sources(profile_name=args.profile))


if __name__ == "__main__":
    main()
