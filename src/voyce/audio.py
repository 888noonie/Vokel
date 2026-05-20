from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .turns import AudioTurn


class AudioDependencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class MicVadConfig:
    vad_model_path: str = "models/silero_vad.onnx"
    sample_rate: int = 16_000
    read_seconds: float = 0.1
    min_silence_duration: float = 0.35
    max_speech_seconds: float = 30.0
    vad_buffer_seconds: float = 30.0
    device: int | str | None = None


def _require_audio_dependencies() -> tuple[Any, Any, Any]:
    missing: list[str] = []
    try:
        import numpy as np
    except ImportError:
        np = None
        missing.append("numpy")

    try:
        import sounddevice as sd
    except ImportError:
        sd = None
        missing.append("sounddevice")

    try:
        import sherpa_onnx
    except ImportError:
        sherpa_onnx = None
        missing.append("sherpa-onnx")

    if missing:
        raise AudioDependencyError(
            "Missing optional audio dependencies: "
            + ", ".join(missing)
            + '. Install them with `pip install -e ".[audio,dev]"` after '
            "installing PortAudio development headers."
        )

    return np, sd, sherpa_onnx


class SileroVadTurnProducer:
    """Microphone producer that yields completed speech turns using Sherpa-ONNX VAD."""

    def __init__(self, config: MicVadConfig | None = None):
        self.config = config or MicVadConfig()
        model_path = Path(self.config.vad_model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"{model_path} does not exist. Download silero_vad.onnx and set "
                "MicVadConfig.vad_model_path."
            )

        self._np, self._sd, sherpa_onnx = _require_audio_dependencies()

        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = str(model_path)
        vad_config.silero_vad.min_silence_duration = self.config.min_silence_duration
        vad_config.sample_rate = self.config.sample_rate
        self._window_size = vad_config.silero_vad.window_size
        self._vad = sherpa_onnx.VoiceActivityDetector(
            vad_config,
            buffer_size_in_seconds=self.config.vad_buffer_seconds,
        )
        self._pending = self._np.array([], dtype=self._np.float32)

    async def next_turn(self) -> AudioTurn:
        samples = await asyncio.to_thread(self._read_next_segment)
        return AudioTurn(
            text="",
            sample_rate=self.config.sample_rate,
            audio_samples=tuple(float(sample) for sample in samples),
        )

    def _read_next_segment(self):
        samples_per_read = max(1, int(self.config.read_seconds * self.config.sample_rate))
        max_samples = max(1, int(self.config.max_speech_seconds * self.config.sample_rate))
        captured = self._np.array([], dtype=self._np.float32)

        stream_kwargs: dict[str, Any] = {
            "channels": 1,
            "dtype": "float32",
            "samplerate": self.config.sample_rate,
        }
        if self.config.device is not None:
            stream_kwargs["device"] = self.config.device

        with self._sd.InputStream(**stream_kwargs) as stream:
            while True:
                frame, _overflowed = stream.read(samples_per_read)
                samples = frame.reshape(-1).astype(self._np.float32)
                self._pending = self._np.concatenate([self._pending, samples])

                while len(self._pending) >= self._window_size:
                    self._vad.accept_waveform(self._pending[: self._window_size])
                    self._pending = self._pending[self._window_size :]

                while not self._vad.empty():
                    segment = self._vad.front.samples.astype(self._np.float32)
                    self._vad.pop()
                    captured = self._np.concatenate([captured, segment])
                    if len(captured) >= max_samples:
                        return captured[:max_samples]

                if len(captured) > 0:
                    return captured


@dataclass(frozen=True)
class SherpaOfflineAsrConfig:
    tokens: str
    provider: str = "cpu"
    num_threads: int = 2
    sample_rate: int = 16_000
    feature_dim: int = 80
    decoding_method: str = "greedy_search"
    debug: bool = False
    sense_voice_model: str = ""
    moonshine_preprocessor: str = ""
    moonshine_encoder: str = ""
    moonshine_uncached_decoder: str = ""
    moonshine_cached_decoder: str = ""
    whisper_encoder: str = ""
    whisper_decoder: str = ""
    whisper_language: str = ""
    whisper_task: str = "transcribe"


class SherpaOfflineAsr:
    """Offline ASR adapter for completed VAD turns."""

    def __init__(self, config: SherpaOfflineAsrConfig):
        self.config = config
        _np, _sd, sherpa_onnx = _require_audio_dependencies()
        self._np = _np
        self._recognizer = self._create_recognizer(sherpa_onnx)

    async def transcribe(self, turn: AudioTurn) -> str:
        if not turn.audio_samples:
            return turn.text
        return await asyncio.to_thread(self._transcribe_sync, turn)

    def _transcribe_sync(self, turn: AudioTurn) -> str:
        sample_rate = turn.sample_rate or self.config.sample_rate
        samples = self._np.asarray(turn.audio_samples, dtype=self._np.float32)
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        self._recognizer.decode_stream(stream)
        return stream.result.text.strip()

    def _create_recognizer(self, sherpa_onnx: Any):
        self._assert_file(self.config.tokens)

        if self.config.sense_voice_model:
            self._assert_file(self.config.sense_voice_model)
            return sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=self.config.sense_voice_model,
                tokens=self.config.tokens,
                num_threads=self.config.num_threads,
                provider=self.config.provider,
                debug=self.config.debug,
                use_itn=True,
            )

        if self.config.moonshine_preprocessor:
            for filename in (
                self.config.moonshine_preprocessor,
                self.config.moonshine_encoder,
                self.config.moonshine_uncached_decoder,
                self.config.moonshine_cached_decoder,
            ):
                self._assert_file(filename)
            return sherpa_onnx.OfflineRecognizer.from_moonshine(
                preprocessor=self.config.moonshine_preprocessor,
                encoder=self.config.moonshine_encoder,
                uncached_decoder=self.config.moonshine_uncached_decoder,
                cached_decoder=self.config.moonshine_cached_decoder,
                tokens=self.config.tokens,
                num_threads=self.config.num_threads,
                provider=self.config.provider,
                decoding_method=self.config.decoding_method,
                debug=self.config.debug,
            )

        if self.config.whisper_encoder:
            self._assert_file(self.config.whisper_encoder)
            self._assert_file(self.config.whisper_decoder)
            return sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=self.config.whisper_encoder,
                decoder=self.config.whisper_decoder,
                tokens=self.config.tokens,
                num_threads=self.config.num_threads,
                provider=self.config.provider,
                decoding_method=self.config.decoding_method,
                debug=self.config.debug,
                language=self.config.whisper_language,
                task=self.config.whisper_task,
            )

        raise ValueError(
            "Configure one ASR model: sense_voice_model, moonshine_preprocessor, "
            "or whisper_encoder."
        )

    @staticmethod
    def _assert_file(filename: str) -> None:
        if not filename or not Path(filename).is_file():
            raise FileNotFoundError(f"Required ASR model file does not exist: {filename}")
