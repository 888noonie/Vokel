from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Keep the checkout-local probe runnable before an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vokel.vision import (
    DEFAULT_VISION_PROMPT,
    analyze_frame,
    capture_frame,
    is_loopback_url,
    list_camera_devices,
)

DEFAULT_URL = "http://127.0.0.1:1234/v1/chat/completions"
DEFAULT_MODEL = "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q5_K_M"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture webcam frames and analyze them with a local LM Studio vision model."
    )
    parser.add_argument("--device", default="/dev/video4", help="V4L2 camera device.")
    parser.add_argument("--url", default=DEFAULT_URL, help="LM Studio Chat Completions endpoint.")
    parser.add_argument(
        "--model",
        default=os.environ.get("LM_STUDIO_MODEL", DEFAULT_MODEL),
        help="Loaded LM Studio vision model identifier.",
    )
    parser.add_argument(
        "--prompt", default=DEFAULT_VISION_PROMPT, help="Question to ask about each frame."
    )
    parser.add_argument("--count", type=int, default=0, help="Frames to analyze; 0 runs until Ctrl-C.")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between analyses.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--framerate", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--keep-frame", type=Path, help="Explicitly retain the latest captured JPEG.")
    parser.add_argument(
        "--allow-external",
        action="store_true",
        help="Allow a non-loopback LM Studio URL to receive captured frames.",
    )
    parser.add_argument("--list-cameras", action="store_true", help="List V4L2 camera nodes and exit.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.list_cameras:
        for camera in list_camera_devices():
            print(f"{camera.path}: {camera.name}")
        return
    if not args.allow_external and not is_loopback_url(args.url):
        raise SystemExit("Error: use --allow-external before sending camera frames off-device.")

    frame_number = 0
    try:
        while args.count == 0 or frame_number < args.count:
            frame_number += 1
            with tempfile.TemporaryDirectory(prefix="vokel-live-vision-") as tmp:
                started = time.monotonic()
                frame = capture_frame(
                    device=args.device,
                    output_dir=Path(tmp),
                    width=args.width,
                    height=args.height,
                    framerate=args.framerate,
                    warmup_frames=args.warmup_frames,
                )
                captured = time.monotonic()
                if args.keep_frame:
                    args.keep_frame.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(frame, args.keep_frame)
                description = analyze_frame(
                    url=args.url,
                    model=args.model,
                    frame=frame,
                    prompt=args.prompt,
                    max_tokens=args.max_tokens,
                )
                finished = time.monotonic()

            print(
                f"[frame {frame_number}] capture={captured - started:.2f}s "
                f"inference={finished - captured:.2f}s"
            )
            print(description, flush=True)
            if args.count == 0 or frame_number < args.count:
                time.sleep(max(0, args.interval))
    except KeyboardInterrupt:
        print("\nLive vision stopped.")
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
