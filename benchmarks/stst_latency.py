from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from voyce.config import LmStudioConfig
from voyce.engine import ConversationEngine
from voyce.lm_studio import ChatMessage, LmStudioClient
from voyce.playback import PlaybackSink
from voyce.telemetry import LatencyTrace
from voyce.turns import AsrEngine, AudioTurn, PassthroughAsr, TextTurnProducer


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    summary_ms: dict[str, float]
    events: list[dict[str, Any]]

    def to_json(self) -> str:
        return json.dumps(
            {
                "name": self.name,
                "summary_ms": self.summary_ms,
                "events": self.events,
            },
            indent=2,
            sort_keys=True,
        )


class DelayedAsr:
    def __init__(self, delay_ms: float):
        self.delay_s = delay_ms / 1000

    async def transcribe(self, turn: AudioTurn) -> str:
        await asyncio.sleep(self.delay_s)
        return turn.text


class ScriptedLlm:
    def __init__(self, tokens: list[str], first_token_ms: float, token_gap_ms: float):
        self.tokens = tokens
        self.first_token_s = first_token_ms / 1000
        self.token_gap_s = token_gap_ms / 1000

    async def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        await asyncio.sleep(self.first_token_s)
        for index, token in enumerate(self.tokens):
            if index:
                await asyncio.sleep(self.token_gap_s)
            yield token


class BenchmarkPlaybackSink:
    def __init__(self, first_audio_ms: float = 0):
        self.first_audio_s = first_audio_ms / 1000
        self.phrases: list[str] = []
        self.stop_count = 0

    async def speak(self, phrase: str) -> None:
        if not self.phrases and self.first_audio_s:
            await asyncio.sleep(self.first_audio_s)
        self.phrases.append(phrase)

    async def stop(self) -> None:
        self.stop_count += 1


async def run_engine_benchmark(
    name: str,
    llm: Any,
    asr: AsrEngine,
    playback: PlaybackSink,
    prompt: str,
) -> BenchmarkResult:
    trace = LatencyTrace()
    engine = ConversationEngine(llm=llm, playback=playback, trace=trace, echo_tokens=False)
    await engine.start()
    try:
        await engine.run_turns(TextTurnProducer([prompt]), asr=asr, max_turns=1)
    finally:
        await engine.close()

    return BenchmarkResult(
        name=name,
        summary_ms=trace.summary_ms(),
        events=[
            {
                "name": event.name,
                "timestamp_ns": event.timestamp_ns,
                "fields": event.fields,
            }
            for event in trace.events
        ],
    )


async def run_synthetic(args: argparse.Namespace) -> BenchmarkResult:
    llm = ScriptedLlm(
        tokens=args.synthetic_tokens.split("|"),
        first_token_ms=args.first_token_ms,
        token_gap_ms=args.token_gap_ms,
    )
    asr = DelayedAsr(args.asr_ms)
    playback = BenchmarkPlaybackSink(first_audio_ms=args.first_audio_ms)
    return await run_engine_benchmark(
        name="synthetic",
        llm=llm,
        asr=asr,
        playback=playback,
        prompt=args.prompt,
    )


async def run_lm_studio(args: argparse.Namespace) -> BenchmarkResult:
    lm_config = LmStudioConfig(url=args.url, model=args.model)
    playback = BenchmarkPlaybackSink()
    async with LmStudioClient(lm_config) as llm:
        return await run_engine_benchmark(
            name="lm-studio",
            llm=llm,
            asr=PassthroughAsr(),
            playback=playback,
            prompt=args.prompt,
        )


def print_table(result: BenchmarkResult) -> None:
    print(f"benchmark={result.name}")
    for name, value in result.summary_ms.items():
        print(f"{name}_ms={value:.1f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Voyce STST latency checkpoints.")
    parser.add_argument(
        "--mode",
        choices=("synthetic", "lm-studio"),
        default="synthetic",
        help="Benchmark backend to run.",
    )
    parser.add_argument(
        "--prompt",
        default="Give one short response for the Voyce latency benchmark.",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON events.")

    parser.add_argument("--url", default=LmStudioConfig.url)
    parser.add_argument("--model", default=LmStudioConfig.model)

    parser.add_argument("--asr-ms", type=float, default=120)
    parser.add_argument("--first-token-ms", type=float, default=350)
    parser.add_argument("--token-gap-ms", type=float, default=35)
    parser.add_argument("--first-audio-ms", type=float, default=40)
    parser.add_argument(
        "--synthetic-tokens",
        default="Voyce is alive,| measured,| and ready.",
        help="Pipe-separated scripted tokens for synthetic mode.",
    )
    return parser


async def async_main(args: argparse.Namespace) -> BenchmarkResult:
    if args.mode == "lm-studio":
        return await run_lm_studio(args)
    return await run_synthetic(args)


def main() -> None:
    args = build_parser().parse_args()
    result = asyncio.run(async_main(args))
    if args.json:
        print(result.to_json())
    else:
        print_table(result)


if __name__ == "__main__":
    main()
