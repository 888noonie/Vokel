from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from voyce.config import LmStudioConfig
from voyce.engine import ConversationEngine
from voyce.events import TextDeltaEvent
from voyce.web_search import create_default_registry
from voyce.inference import ChatMessage, LocalInferenceClient
from voyce.playback import PlaybackSink, build_playback_sink
from voyce.progress import ConsoleProgressObserver
from voyce.telemetry import LatencyTrace
from voyce.turns import AsrEngine, AudioTurn, PassthroughAsr, TextTurnProducer, TurnProducer


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
        return str(turn.text)


class ScriptedLlm:
    def __init__(self, tokens: list[str], first_token_ms: float, token_gap_ms: float):
        self.tokens = tokens
        self.first_token_s = first_token_ms / 1000
        self.token_gap_s = token_gap_ms / 1000

    async def stream_chat(
        self, messages: list[ChatMessage], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[TextDeltaEvent]:
        await asyncio.sleep(self.first_token_s)
        for index, token in enumerate(self.tokens):
            if index:
                await asyncio.sleep(self.token_gap_s)
            yield TextDeltaEvent(token)


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
    producer: TurnProducer,
    progress: bool = False,
    trace: LatencyTrace | None = None,
) -> BenchmarkResult:
    own_trace = trace is None
    trace = trace or LatencyTrace()
    if progress and own_trace:
        trace.add_observer(ConsoleProgressObserver())
    engine = ConversationEngine(llm=llm, playback=playback, trace=trace, echo_tokens=False, tool_registry=create_default_registry())
    await engine.start()
    try:
        await engine.run_turns(producer, asr=asr, max_turns=1)
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
        producer=TextTurnProducer([args.prompt]),
        progress=args.progress,
    )


async def run_lm_studio(args: argparse.Namespace) -> BenchmarkResult:
    lm_config = LmStudioConfig(url=args.url, model=args.model)
    playback = build_playback_sink(args.playback) if args.playback else BenchmarkPlaybackSink()
    async with LocalInferenceClient(lm_config) as llm:
        return await run_engine_benchmark(
            name="lm-studio",
            llm=llm,
            asr=PassthroughAsr(),
            playback=playback,
            producer=TextTurnProducer([args.prompt]),
            progress=args.progress,
        )


def _stst_profile_mic_config(args: argparse.Namespace) -> Any | None:
    """Resolve optional audio profile and enforce route guard when requested."""
    from voyce.audio_profiles import get_audio_profile
    from voyce.audio_routes import find_source, load_pulse_sources

    profile = get_audio_profile(args.audio_profile) if args.audio_profile else None
    profile_config = profile.mic if profile else None
    if args.require_profile_route and profile and profile.preferred_source_name:
        source = find_source(load_pulse_sources(), profile.preferred_source_name)
        if source is None:
            raise RuntimeError(f"Profile source is missing: {profile.preferred_source_name}")
        if not source.usable_input:
            raise RuntimeError(
                f"Profile source is not usable: {source.description} "
                f"available={source.active_port_available}"
            )
        if not source.is_default:
            raise RuntimeError(
                f"Profile source is not default: {source.description}. "
                "Run scripts/audio_routes.py for diagnostics."
            )
    return profile_config


def mic_vad_config_for_stst_mic(args: argparse.Namespace, profile_mic: Any | None) -> Any:
    """Shared MicVadConfig for VAD-bounded and streaming desktop mic benchmarks."""
    from voyce.audio import MicVadConfig

    return MicVadConfig(
        vad_model_path=args.vad_model,
        sample_rate=profile_mic.sample_rate if profile_mic else 16_000,
        read_seconds=profile_mic.read_seconds if profile_mic else 0.1,
        threshold=args.vad_threshold
        if args.vad_threshold is not None
        else profile_mic.threshold
        if profile_mic
        else 0.25,
        min_silence_duration=args.min_silence_duration
        if args.min_silence_duration is not None
        else profile_mic.min_silence_duration
        if profile_mic
        else 0.35,
        min_speech_duration=args.min_speech_duration
        if args.min_speech_duration is not None
        else profile_mic.min_speech_duration
        if profile_mic
        else 0.1,
        max_speech_seconds=args.max_speech_seconds
        if args.max_speech_seconds is not None
        else profile_mic.max_speech_seconds
        if profile_mic
        else 30,
        vad_buffer_seconds=profile_mic.vad_buffer_seconds if profile_mic else 30,
        remove_dc_offset=not args.keep_dc_offset,
        input_gain=args.input_gain
        if args.input_gain is not None
        else profile_mic.input_gain
        if profile_mic
        else 1.0,
        device=args.audio_device
        if args.audio_device is not None
        else profile_mic.device
        if profile_mic
        else None,
    )


async def run_mic_lm_studio(args: argparse.Namespace) -> BenchmarkResult:
    from voyce.audio import SherpaOfflineAsr, SherpaOfflineAsrConfig, SileroVadTurnProducer

    profile_config = _stst_profile_mic_config(args)
    producer = SileroVadTurnProducer(mic_vad_config_for_stst_mic(args, profile_config))
    asr = SherpaOfflineAsr(
        SherpaOfflineAsrConfig(
            tokens=args.asr_tokens,
            num_threads=args.asr_threads,
            sense_voice_model=args.sense_voice_model,
            moonshine_preprocessor=args.moonshine_preprocessor,
            moonshine_encoder=args.moonshine_encoder,
            moonshine_uncached_decoder=args.moonshine_uncached_decoder,
            moonshine_cached_decoder=args.moonshine_cached_decoder,
            whisper_encoder=args.whisper_encoder,
            whisper_decoder=args.whisper_decoder,
        )
    )
    playback = build_playback_sink(args.playback) if args.playback else BenchmarkPlaybackSink()
    lm_config = LmStudioConfig(url=args.url, model=args.model)
    async with LocalInferenceClient(lm_config) as llm:
        return await run_engine_benchmark(
            name="mic-lm-studio",
            llm=llm,
            asr=asr,
            playback=playback,
            producer=producer,
            progress=args.progress,
        )


async def run_mic_streaming_lm_studio(args: argparse.Namespace) -> BenchmarkResult:
    """One turn: streaming Zipformer endpointing + async mic; same engine loop as Phase 1."""
    from voyce.audio import AsynchronousMicStream, StreamingTurnProducer, create_streaming_asr

    profile_config = _stst_profile_mic_config(args)
    mic_cfg = mic_vad_config_for_stst_mic(args, profile_config)
    mic = AsynchronousMicStream(mic_cfg)
    online_asr = create_streaming_asr(args.streaming_asr_dir, provider=args.asr_provider)
    trace = LatencyTrace()
    if args.progress:
        trace.add_observer(ConsoleProgressObserver())
    producer = StreamingTurnProducer(mic, online_asr, trace)
    playback = build_playback_sink(args.playback) if args.playback else BenchmarkPlaybackSink()
    lm_config = LmStudioConfig(url=args.url, model=args.model)
    async with LocalInferenceClient(lm_config) as llm:
        return await run_engine_benchmark(
            name="mic-streaming-lm-studio",
            llm=llm,
            asr=online_asr,
            playback=playback,
            producer=producer,
            progress=False,
            trace=trace,
        )


def print_table(result: BenchmarkResult) -> None:
    print(f"benchmark={result.name}")
    for name, value in result.summary_ms.items():
        print(f"{name}_ms={value:.1f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Voyce STST latency checkpoints.")
    parser.add_argument(
        "--mode",
        choices=("synthetic", "lm-studio", "mic-lm-studio", "mic-streaming-lm-studio"),
        default="synthetic",
        help="Benchmark backend to run.",
    )
    parser.add_argument(
        "--prompt",
        default="Give one short response for the Voyce latency benchmark.",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON events.")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print human-readable pipeline cues.",
    )
    parser.add_argument(
        "--playback",
        choices=("console", "spd-say"),
        default=None,
        help="Use a real playback sink instead of the benchmark sink.",
    )

    parser.add_argument("--url", default=LmStudioConfig.url)
    parser.add_argument("--model", default=LmStudioConfig.model)
    parser.add_argument("--vad-model", default="models/silero_vad.onnx")
    parser.add_argument(
        "--audio-profile",
        choices=(
            "laptop-mic-headphones",
            "laptop-open",
            "headset-wired",
            "headset-bluetooth",
            "noisy-handset",
        ),
        default=None,
    )
    parser.add_argument(
        "--require-profile-route",
        action="store_true",
        help="Fail if the selected profile's preferred Pulse/PipeWire source is not usable/default.",
    )
    parser.add_argument("--asr-tokens", default="")
    parser.add_argument("--asr-threads", type=int, default=2)
    parser.add_argument("--sense-voice-model", default="")
    parser.add_argument("--moonshine-preprocessor", default="")
    parser.add_argument("--moonshine-encoder", default="")
    parser.add_argument("--moonshine-uncached-decoder", default="")
    parser.add_argument("--moonshine-cached-decoder", default="")
    parser.add_argument("--whisper-encoder", default="")
    parser.add_argument("--whisper-decoder", default="")
    parser.add_argument("--vad-threshold", type=float, default=None)
    parser.add_argument("--min-silence-duration", type=float, default=None)
    parser.add_argument("--min-speech-duration", type=float, default=None)
    parser.add_argument("--max-speech-seconds", type=float, default=None)
    parser.add_argument("--keep-dc-offset", action="store_true")
    parser.add_argument("--input-gain", type=float, default=None)
    parser.add_argument("--audio-device", default=None)
    parser.add_argument(
        "--streaming-asr-dir",
        default="models/sherpa-onnx-streaming-zipformer-en-2023-06-26",
        help="Sherpa-ONNX streaming Zipformer model directory (mic-streaming-lm-studio mode).",
    )
    parser.add_argument(
        "--asr-provider",
        default="cpu",
        help="Sherpa-ONNX execution provider for streaming Zipformer (e.g. cpu, cuda).",
    )

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
    if args.mode == "mic-streaming-lm-studio":
        return await run_mic_streaming_lm_studio(args)
    if args.mode == "mic-lm-studio":
        return await run_mic_lm_studio(args)
    if args.mode == "lm-studio":
        return await run_lm_studio(args)
    return await run_synthetic(args)


def main() -> None:
    args = build_parser().parse_args()
    if args.json:
        args.progress = False
    result = asyncio.run(async_main(args))
    if args.json:
        print(result.to_json())
    else:
        print_table(result)


if __name__ == "__main__":
    main()
