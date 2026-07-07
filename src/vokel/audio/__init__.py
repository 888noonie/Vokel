"""Audio capture, VAD/ASR helpers, and optional musical-mode clock."""

from vokel.audio.beatclock import BeatClock, BeatInfo, ClockStopped
from vokel.audio.beattrack import BeatTrackPlayer
from vokel.audio.quantized_sink import QuantizedPlaybackSink, Quantum
from vokel.audio.capture import (
    AsynchronousMicStream,
    AudioDependencyError,
    MicVadConfig,
    SherpaOfflineAsr,
    SherpaOfflineAsrConfig,
    SherpaOnlineAsr,
    SherpaOnlineAsrConfig,
    SherpaOnlineStream,
    SileroVadTurnProducer,
    StreamingTurnProducer,
    _require_audio_dependencies,
    coerce_audio_device,
    create_streaming_asr,
)

__all__ = [
    "AsynchronousMicStream",
    "AudioDependencyError",
    "BeatClock",
    "BeatTrackPlayer",
    "BeatInfo",
    "ClockStopped",
    "QuantizedPlaybackSink",
    "Quantum",
    "MicVadConfig",
    "SherpaOfflineAsr",
    "SherpaOfflineAsrConfig",
    "SherpaOnlineAsr",
    "SherpaOnlineAsrConfig",
    "SherpaOnlineStream",
    "SileroVadTurnProducer",
    "StreamingTurnProducer",
    "_require_audio_dependencies",
    "coerce_audio_device",
    "create_streaming_asr",
]