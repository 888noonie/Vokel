from __future__ import annotations

import argparse
import asyncio

from .audio import MicVadConfig, SherpaOfflineAsr, SherpaOfflineAsrConfig, SileroVadTurnProducer
from .config import LmStudioConfig
from .engine import ConversationEngine
from .lm_studio import LmStudioClient
from .playback import ConsolePlaybackSink
from .turns import PassthroughAsr, TextTurnProducer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one Voyce turn against LM Studio.")
    parser.add_argument("prompt", help="Text prompt to send as the user turn.")
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
        "--metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print latency metrics after the turn.",
    )
    return parser


async def run(args: argparse.Namespace) -> None:
    from .turns import AsrEngine, TurnProducer
    asr: AsrEngine = PassthroughAsr()
    producer: TurnProducer = TextTurnProducer([args.prompt])
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
    async with LmStudioClient(lm_config) as llm:
        engine = ConversationEngine(llm=llm, playback=ConsolePlaybackSink())
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
