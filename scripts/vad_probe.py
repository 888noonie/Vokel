from __future__ import annotations

import argparse
import math
import time


def rms_dbfs(samples) -> float:
    import numpy as np

    if len(samples) == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(samples))))
    if rms <= 0:
        return -120.0
    return 20.0 * math.log10(min(rms, 1.0))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe Sherpa-ONNX Silero VAD detection.")
    parser.add_argument("--vad-model", default="models/silero_vad.onnx")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--min-silence-duration", type=float, default=0.35)
    parser.add_argument("--min-speech-duration", type=float, default=0.1)
    parser.add_argument("--keep-dc-offset", action="store_true")
    parser.add_argument("--input-gain", type=float, default=1.0)
    return parser


def main() -> None:
    import numpy as np
    import sherpa_onnx
    import sounddevice as sd

    args = build_parser().parse_args()
    device = int(args.device) if args.device and args.device.isdigit() else args.device

    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = args.vad_model
    config.silero_vad.threshold = args.threshold
    config.silero_vad.min_silence_duration = args.min_silence_duration
    config.silero_vad.min_speech_duration = args.min_speech_duration
    config.sample_rate = args.sample_rate
    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)

    window_size = config.silero_vad.window_size
    deadline = time.monotonic() + args.seconds
    accepted = 0
    segments = 0
    print("Speak now...")
    with sd.InputStream(device=device, channels=1, dtype="float32", samplerate=args.sample_rate) as stream:
        while time.monotonic() < deadline:
            frame, _overflowed = stream.read(window_size)
            samples = frame.reshape(-1).astype(np.float32)
            raw_mean = float(np.mean(samples)) if len(samples) else 0.0
            if not args.keep_dc_offset:
                samples = samples - raw_mean
            if args.input_gain != 1.0:
                samples = np.clip(samples * args.input_gain, -1.0, 1.0).astype(np.float32)
            vad.accept_waveform(samples)
            accepted += len(samples)
            detected = vad.is_speech_detected()
            print(
                f"t={accepted / args.sample_rate:.2f}s "
                f"raw_mean={raw_mean:.3f} "
                f"rms_dbfs={rms_dbfs(samples):.1f} "
                f"speech_detected={detected} "
                f"queue_empty={vad.empty()}",
                flush=True,
            )
            while not vad.empty():
                segment = vad.front
                segments += 1
                print(
                    f"segment={segments} "
                    f"samples={len(segment.samples)} "
                    f"start={segment.start}",
                    flush=True,
                )
                vad.pop()

    vad.flush()
    while not vad.empty():
        segment = vad.front
        segments += 1
        print(f"segment={segments} samples={len(segment.samples)} start={segment.start}", flush=True)
        vad.pop()
    print(f"segments={segments}")


if __name__ == "__main__":
    main()
