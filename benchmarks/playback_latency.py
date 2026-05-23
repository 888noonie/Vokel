from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass

from vokel.playback import (
    available_playback_backends,
    build_playback_sink,
)


LONG_TEXT = (
    "Vokel playback interruption benchmark. This sentence is intentionally long "
    "so there is enough time to interrupt the speech while it is still active."
)


@dataclass(frozen=True)
class PlaybackBenchmarkResult:
    backend: str
    interrupt_after_ms: float
    stop_latency_ms: float
    speak_completion_after_stop_ms: float

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=True)


def build_sink(backend: str):  # type: ignore[return-value]
    """Build a playback sink, including the benchmark-only 'fake' backend."""
    if backend == "fake":
        from vokel.playback import SubprocessPlaybackConfig, SubprocessPlaybackSink

        return SubprocessPlaybackSink(
            SubprocessPlaybackConfig(
                command=("python3", "-c", "import time; time.sleep(10)")
            )
        )
    return build_playback_sink(backend)


async def run_benchmark(backend: str, interrupt_after_ms: float) -> PlaybackBenchmarkResult:
    sink = build_sink(backend)
    speak_task = asyncio.create_task(sink.speak(LONG_TEXT))
    await asyncio.sleep(interrupt_after_ms / 1000)

    stop_start = time.perf_counter()
    await sink.stop()
    stop_done = time.perf_counter()

    await speak_task
    speak_done = time.perf_counter()

    return PlaybackBenchmarkResult(
        backend=backend,
        interrupt_after_ms=interrupt_after_ms,
        stop_latency_ms=(stop_done - stop_start) * 1000,
        speak_completion_after_stop_ms=(speak_done - stop_done) * 1000,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark playback stop latency.")
    parser.add_argument("--backend", choices=available_playback_backends(), default="fake")
    parser.add_argument("--interrupt-after-ms", type=float, default=500)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--list", action="store_true", help="List available playback backends.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.list:
        for backend in available_playback_backends():
            print(backend)
        return

    result = asyncio.run(run_benchmark(args.backend, args.interrupt_after_ms))
    if args.json:
        print(result.to_json())
        return

    print(f"backend={result.backend}")
    print(f"interrupt_after_ms={result.interrupt_after_ms:.1f}")
    print(f"stop_latency_ms={result.stop_latency_ms:.1f}")
    print(f"speak_completion_after_stop_ms={result.speak_completion_after_stop_ms:.1f}")


if __name__ == "__main__":
    main()
