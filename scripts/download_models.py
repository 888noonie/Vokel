from __future__ import annotations

import argparse
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


MODELS_DIR = Path("models")


@dataclass(frozen=True)
class ModelAsset:
    name: str
    url: str
    destination: Path
    archive_member_prefix: str | None = None


ASSETS = {
    "silero-vad": ModelAsset(
        name="silero-vad",
        url="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
        destination=MODELS_DIR / "silero_vad.onnx",
    ),
    "sense-voice-int8": ModelAsset(
        name="sense-voice-int8",
        url=(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
            "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2"
        ),
        destination=MODELS_DIR / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17",
        archive_member_prefix="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17",
    ),
    "streaming-zipformer-en": ModelAsset(
        name="streaming-zipformer-en",
        url=(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
            "sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2"
        ),
        destination=MODELS_DIR / "sherpa-onnx-streaming-zipformer-en-2023-06-26",
        archive_member_prefix="sherpa-onnx-streaming-zipformer-en-2023-06-26",
    ),
    "kokoro-v1.0": ModelAsset(
        name="kokoro-v1.0",
        url="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
        destination=MODELS_DIR / "kokoro-v1.0.onnx",
    ),
    "kokoro-voices": ModelAsset(
        name="kokoro-voices",
        url="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
        destination=MODELS_DIR / "voices-v1.0.bin",
    ),
}


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_destination = destination.with_suffix(destination.suffix + ".tmp")
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, tmp_destination)
    tmp_destination.replace(destination)


def safe_extract_tar_bz2(archive_path: Path, output_dir: Path) -> None:
    output_root = output_dir.resolve()
    with tarfile.open(archive_path, mode="r:bz2") as archive:
        for member in archive.getmembers():
            target = (output_dir / member.name).resolve()
            if not str(target).startswith(str(output_root)):
                raise RuntimeError(f"Unsafe archive member path: {member.name}")
        archive.extractall(output_dir)


def ensure_asset(asset: ModelAsset) -> None:
    if asset.destination.exists():
        print(f"Already present: {asset.destination}")
        return

    if asset.archive_member_prefix:
        archive_path = MODELS_DIR / f"{asset.archive_member_prefix}.tar.bz2"
        download_file(asset.url, archive_path)
        safe_extract_tar_bz2(archive_path, MODELS_DIR)
        archive_path.unlink()
        print(f"Extracted: {asset.destination}")
        return

    download_file(asset.url, asset.destination)
    print(f"Saved: {asset.destination}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download local model assets for Voyce.")
    parser.add_argument(
        "models",
        nargs="*",
        default=None,
        help="Model assets to download.",
    )
    parser.add_argument("--list", action="store_true", help="List available model assets.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.list:
        for name, asset in ASSETS.items():
            print(f"{name}: {asset.destination}")
        return

    MODELS_DIR.mkdir(exist_ok=True)
    model_names = args.models or [
        "silero-vad",
        "sense-voice-int8",
        "streaming-zipformer-en",
        "kokoro-v1.0",
        "kokoro-voices",
    ]
    for name in model_names:
        if name not in ASSETS:
            print(f"Error: invalid choice: '{name}' (choose from {', '.join(sorted(ASSETS))})")
            exit(2)
        ensure_asset(ASSETS[name])


if __name__ == "__main__":
    main()
