from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .audio import MicVadConfig, SherpaOfflineAsr, SherpaOfflineAsrConfig, SileroVadTurnProducer
from .config import LmStudioConfig
from .engine import ConversationEngine
from .web_search import create_default_registry
from .inference import LocalInferenceClient
from .memory import MemoryConfig, SQLiteMemoryStore
from .playback import ConsolePlaybackSink
from .turns import PassthroughAsr, TextTurnProducer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one Vokel turn or start the web app.")
    parser.add_argument("prompt", nargs="?", default="", help="Text prompt to send as the user turn.")
    parser.add_argument("--url", default=LmStudioConfig.url)
    parser.add_argument("--model", default=LmStudioConfig.model)
    parser.add_argument("--mic", action="store_true", help="Capture one microphone turn.")
    parser.add_argument("--vad-model", default=MicVadConfig.vad_model_path)
    parser.add_argument("--asr-tokens", default="")
    parser.add_argument("--sense-voice-model", default="")
    parser.add_argument("--moonshine-preprocessor", default="")
    parser.add_argument("--moonshine-encoder", default="")
    parser.add_argument("--moonshine-uncached-decoder", default="")
    parser.add_argument("--moonshine-cached-decoder", default="")
    parser.add_argument("--whisper-encoder", default="")
    parser.add_argument("--whisper-decoder", default="")
    parser.add_argument(
        "--streaming-asr-dir",
        default="",
        help=(
            "With --mic: use Sherpa streaming Zipformer from this directory "
            "(see scripts/download_models.py streaming-zipformer-en). "
            "If omitted, use Silero VAD + offline ASR instead."
        ),
    )
    parser.add_argument(
        "--metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print latency metrics after the turn.",
    )
    parser.add_argument("--web", action="store_true", help="Start the web API and UI dashboard server.")
    parser.add_argument("--host", default="0.0.0.0", help="Web server host binding.")
    parser.add_argument("--port", type=int, default=8000, help="Web server port.")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn hot reloading.")
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Let Vokel use saved local context for this turn.",
    )
    parser.add_argument("--memory-db", default=str(MemoryConfig.path), help="Saved context file path.")
    parser.add_argument(
        "--memory-results",
        type=int,
        default=MemoryConfig.max_results,
        help="Maximum saved context snippets to use per turn.",
    )
    parser.add_argument(
        "--remember",
        default="",
        help="Save a local note without sending a prompt to the model.",
    )
    parser.add_argument("--memory-list", action="store_true", help="List recent saved notes.")
    parser.add_argument("--memory-clear", action="store_true", help="Delete saved notes and turn history.")
    return parser


async def run(args: argparse.Namespace) -> None:
    if args.web:
        import uvicorn
        config = uvicorn.Config("vokel.web:app", host=args.host, port=args.port, reload=args.reload)
        server = uvicorn.Server(config)
        await server.serve()
        return

    from .telemetry import LatencyTrace
    from .turns import AsrEngine, TurnProducer

    memory_db = Path(args.memory_db)
    admin_memory = SQLiteMemoryStore(memory_db)
    if args.memory_clear:
        await admin_memory.clear()
        print(f"Cleared saved context at {memory_db}")
        return
    if args.remember.strip():
        await admin_memory.record_fact(args.remember)
        print(f"Saved locally: {args.remember.strip()}")
        return
    if args.memory_list:
        facts = await admin_memory.list_facts()
        if not facts:
            print("No saved notes.")
        else:
            for index, fact in enumerate(facts, start=1):
                print(f"{index}. {fact.user_text}")
        return

    trace = LatencyTrace()
    asr: AsrEngine
    producer: TurnProducer
    if args.mic and args.streaming_asr_dir.strip():
        from .audio import AsynchronousMicStream, StreamingTurnProducer, create_streaming_asr

        mic = AsynchronousMicStream(MicVadConfig(vad_model_path=args.vad_model))
        online = create_streaming_asr(args.streaming_asr_dir.strip())
        producer = StreamingTurnProducer(mic, online, trace)
        asr = online
    else:
        asr = PassthroughAsr()
        producer = TextTurnProducer([args.prompt])
        if args.mic:
            producer = SileroVadTurnProducer(MicVadConfig(vad_model_path=args.vad_model))
            asr = SherpaOfflineAsr(
                SherpaOfflineAsrConfig(
                    tokens=args.asr_tokens,
                    sense_voice_model=args.sense_voice_model,
                    moonshine_preprocessor=args.moonshine_preprocessor,
                    moonshine_encoder=args.moonshine_encoder,
                    moonshine_uncached_decoder=args.moonshine_uncached_decoder,
                    moonshine_cached_decoder=args.moonshine_cached_decoder,
                    whisper_encoder=args.whisper_encoder,
                    whisper_decoder=args.whisper_decoder,
                )
            )

    lm_config = LmStudioConfig(url=args.url, model=args.model)
    memory_config = MemoryConfig(
        enabled=args.memory,
        path=memory_db,
        max_results=args.memory_results,
    )
    memory_store = (
        SQLiteMemoryStore(memory_config.path, scan_limit=memory_config.scan_limit)
        if memory_config.enabled
        else None
    )
    async with LocalInferenceClient(lm_config) as llm:
        engine = ConversationEngine(
            llm=llm,
            playback=ConsolePlaybackSink(),
            trace=trace,
            echo_tokens=False,
            memory_store=memory_store,
            memory_config=memory_config,
            tool_registry=create_default_registry(),
        )
        await engine.start()
        try:
            await engine.run_turns(producer=producer, asr=asr, max_turns=1)
            if args.metrics:
                print("\n[latency]")
                for name, value in engine.trace.summary_ms().items():
                    print(f"{name}_ms={value:.1f}")
        finally:
            await engine.close()
    print()


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
