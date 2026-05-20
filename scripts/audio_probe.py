from __future__ import annotations

import argparse
import math
import time


def rms_dbfs(samples) -> float:
    import numpy as np

    rms = float(np.sqrt(np.mean(np.square(samples)))) if len(samples) else 0.0
    if rms <= 0:
        return -120.0
    return 20.0 * math.log10(min(rms, 1.0))


def probe_device(device: int | str | None, seconds: float, sample_rate: int) -> float:
    import sounddevice as sd

    frames = max(1, int(seconds * sample_rate))
    with sd.InputStream(
        device=device, channels=1, dtype="float32", samplerate=sample_rate
    ) as stream:
        data, overflowed = stream.read(frames)
        if overflowed:
            print(f"device={device} overflowed=true")
        return rms_dbfs(data.reshape(-1))


def list_input_devices() -> None:
    import sounddevice as sd

    for index, device in enumerate(sd.query_devices()):
        if int(device.get("max_input_channels", 0)) > 0:
            print(
                f"{index}: {device['name']} "
                f"inputs={device['max_input_channels']} "
                f"default_sr={device['default_samplerate']}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe microphone input levels.")
    parser.add_argument("--list", action="store_true", help="List input devices.")
    parser.add_argument("--device", default=None, help="Input device index or name.")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.list:
        list_input_devices()
        return

    device = int(args.device) if args.device and args.device.isdigit() else args.device
    print("Speak now...")
    time.sleep(0.2)
    level = probe_device(device=device, seconds=args.seconds, sample_rate=args.sample_rate)
    print(f"device={device if device is not None else 'default'} rms_dbfs={level:.1f}")


if __name__ == "__main__":
    main()
