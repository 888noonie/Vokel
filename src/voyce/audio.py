from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .telemetry import LatencyTrace
from .turns import AudioTurn


class AudioDependencyError(RuntimeError):
    pass


def coerce_audio_device(device: int | str | None) -> int | str | None:
    if isinstance(device, str) and device.isdigit():
        return int(device)
    return device


@dataclass(frozen=True)
class MicVadConfig:
    vad_model_path: str = "models/silero_vad.onnx"
    sample_rate: int = 16_000
    read_seconds: float = 0.1
    threshold: float = 0.25
    min_silence_duration: float = 0.35
    min_speech_duration: float = 0.1
    max_speech_seconds: float = 30.0
    vad_buffer_seconds: float = 30.0
    remove_dc_offset: bool = True
    input_gain: float = 1.0
    device: int | str | None = None


def _require_audio_dependencies() -> tuple[Any, Any, Any]:
    missing: list[str] = []
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore[assignment]
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
        vad_config.silero_vad.threshold = self.config.threshold
        vad_config.silero_vad.min_silence_duration = self.config.min_silence_duration
        vad_config.silero_vad.min_speech_duration = self.config.min_speech_duration
        vad_config.silero_vad.max_speech_duration = self.config.max_speech_seconds
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

    def _read_next_segment(self) -> Any:
        samples_per_read = max(1, int(self.config.read_seconds * self.config.sample_rate))
        max_samples = max(1, int(self.config.max_speech_seconds * self.config.sample_rate))
        captured = self._np.array([], dtype=self._np.float32)

        stream_kwargs: dict[str, Any] = {
            "channels": 1,
            "dtype": "float32",
            "samplerate": self.config.sample_rate,
        }
        if self.config.device is not None:
            stream_kwargs["device"] = coerce_audio_device(self.config.device)

        with self._sd.InputStream(**stream_kwargs) as stream:
            while True:
                frame, _overflowed = stream.read(samples_per_read)
                samples = frame.reshape(-1).astype(self._np.float32)
                if self.config.remove_dc_offset:
                    samples = samples - self._np.mean(samples)
                if self.config.input_gain != 1.0:
                    samples = self._np.clip(
                        samples * self.config.input_gain,
                        -1.0,
                        1.0,
                    ).astype(self._np.float32)
                self._pending = self._np.concatenate([self._pending, samples])

                while len(self._pending) >= self._window_size:
                    self._vad.accept_waveform(self._pending[: self._window_size])
                    self._pending = self._pending[self._window_size :]

                while not self._vad.empty():
                    segment = self._np.asarray(self._vad.front.samples, dtype=self._np.float32)
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
        return str(stream.result.text).strip()

    def _create_recognizer(self, sherpa_onnx: Any) -> Any:
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


@dataclass(frozen=True)
class SherpaOnlineAsrConfig:
    tokens: str
    encoder: str
    decoder: str
    joiner: str
    provider: str = "cpu"
    num_threads: int = 2
    sample_rate: int = 16_000
    feature_dim: int = 80
    decoding_method: str = "greedy_search"
    debug: bool = False
    enable_endpoint_detection: bool = True
    rule1_min_trailing_silence: float = 2.4
    rule2_min_trailing_silence: float = 1.2
    rule3_min_utterance_length: float = 20.0


class SherpaOnlineStream:
    """Wrapper around sherpa_onnx.OnlineStream that holds reference to both recognizer and stream."""

    def __init__(self, recognizer: Any, stream: Any) -> None:
        self._recognizer = recognizer
        self._stream = stream

    def accept_waveform(self, sample_rate: float, samples: Any) -> None:
        self._stream.accept_waveform(sample_rate, samples)

    def decode(self) -> None:
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

    def get_result(self) -> str:
        return str(self._recognizer.get_result(self._stream)).strip()

    def is_endpoint(self) -> bool:
        return bool(self._recognizer.is_endpoint(self._stream))


class SherpaOnlineAsr:
    """Online (streaming) ASR engine using Sherpa-ONNX Zipformer."""

    def __init__(self, config: SherpaOnlineAsrConfig) -> None:
        self.config = config
        _np, _sd, sherpa_onnx = _require_audio_dependencies()
        self._np = _np
        self._recognizer = self._create_recognizer(sherpa_onnx)

    def _create_recognizer(self, sherpa_onnx: Any) -> Any:
        self._assert_file(self.config.tokens)
        self._assert_file(self.config.encoder)
        self._assert_file(self.config.decoder)
        self._assert_file(self.config.joiner)

        return sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=self.config.tokens,
            encoder=self.config.encoder,
            decoder=self.config.decoder,
            joiner=self.config.joiner,
            num_threads=self.config.num_threads,
            sample_rate=self.config.sample_rate,
            feature_dim=self.config.feature_dim,
            enable_endpoint_detection=self.config.enable_endpoint_detection,
            rule1_min_trailing_silence=self.config.rule1_min_trailing_silence,
            rule2_min_trailing_silence=self.config.rule2_min_trailing_silence,
            rule3_min_utterance_length=self.config.rule3_min_utterance_length,
            decoding_method=self.config.decoding_method,
            debug=self.config.debug,
            provider=self.config.provider,
        )

    def create_stream(self) -> SherpaOnlineStream:
        stream = self._recognizer.create_stream()
        return SherpaOnlineStream(self._recognizer, stream)

    async def transcribe(self, turn: AudioTurn) -> str:
        """Offline transcribe fallback for compatibility."""
        if turn.text:
            return turn.text
        if not turn.audio_samples:
            return turn.text
        return await asyncio.to_thread(self._transcribe_sync, turn)

    def _transcribe_sync(self, turn: AudioTurn) -> str:
        sample_rate = turn.sample_rate or self.config.sample_rate
        samples = self._np.asarray(turn.audio_samples, dtype=self._np.float32)

        stream = self.create_stream()
        stream.accept_waveform(sample_rate, samples)
        stream.decode()
        return stream.get_result()

    @staticmethod
    def _assert_file(filename: str) -> None:
        if not filename or not Path(filename).is_file():
            raise FileNotFoundError(f"Required online ASR model file does not exist: {filename}")


class AsynchronousMicStream:
    """Non-blocking audio capture stream using sounddevice with asyncio queue."""

    def __init__(self, config: MicVadConfig) -> None:
        self.config = config
        self._loop = asyncio.get_running_loop()
        self._np, self._sd, _ = _require_audio_dependencies()
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._stream: Any = None

    def _callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        # Copy the numpy array to avoid it being reused by sounddevice
        data_copy = indata.copy()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, data_copy)

    async def __aenter__(self) -> AsynchronousMicStream:
        stream_kwargs = {
            "channels": 1,
            "dtype": "float32",
            "samplerate": self.config.sample_rate,
            "callback": self._callback,
            "blocksize": int(self.config.read_seconds * self.config.sample_rate),
        }
        if self.config.device is not None:
            stream_kwargs["device"] = coerce_audio_device(self.config.device)

        self._stream = self._sd.InputStream(**stream_kwargs)
        self._stream.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def read(self) -> Any:
        frame = await self._queue.get()
        samples = frame.reshape(-1).astype(self._np.float32)
        if self.config.remove_dc_offset:
            samples = samples - self._np.mean(samples)
        if self.config.input_gain != 1.0:
            samples = self._np.clip(
                samples * self.config.input_gain,
                -1.0,
                1.0,
            ).astype(self._np.float32)
        return samples


class StreamingTurnProducer:
    """Turn producer that uses an asynchronous mic stream and streaming ASR for real-time transcription."""

    def __init__(
        self,
        mic: AsynchronousMicStream,
        asr: SherpaOnlineAsr,
        trace: LatencyTrace,
    ) -> None:
        self.mic = mic
        self.asr = asr
        self.trace = trace
        self._np, _, _ = _require_audio_dependencies()

    async def next_turn(self) -> AudioTurn:
        stream = self.asr.create_stream()
        all_samples: list[float] = []

        last_text = ""
        last_changed_time = asyncio.get_running_loop().time()
        stable_fired = False
        max_samples = int(self.mic.config.max_speech_seconds * self.mic.config.sample_rate)

        async with self.mic as active_mic:
            while len(all_samples) < max_samples:
                samples = await active_mic.read()
                all_samples.extend(samples.tolist())

                # Feed samples to ASR stream
                stream.accept_waveform(self.mic.config.sample_rate, samples)
                stream.decode()

                text = stream.get_result()
                current_time = asyncio.get_running_loop().time()

                if text != last_text:
                    last_text = text
                    last_changed_time = current_time
                    stable_fired = False
                    self.trace.mark("partial_transcript", text=text)
                elif text and not stable_fired:
                    if current_time - last_changed_time >= 0.6:
                        self.trace.mark("stable_transcript", text=text)
                        stable_fired = True

                if stream.is_endpoint():
                    if last_text and not stable_fired:
                        self.trace.mark("stable_transcript", text=last_text)
                    break

        return AudioTurn(
            text=last_text,
            sample_rate=self.mic.config.sample_rate,
            audio_samples=tuple(all_samples),
        )


def create_streaming_asr(model_dir: str | Path, provider: str = "cpu") -> SherpaOnlineAsr:
    """Convenience helper to instantiate SherpaOnlineAsr from a directory layout."""
    path = Path(model_dir)
    return SherpaOnlineAsr(
        SherpaOnlineAsrConfig(
            tokens=str(path / "tokens.txt"),
            encoder=str(path / "encoder-epoch-99-avg-1-chunk-16-left-128.onnx"),
            decoder=str(path / "decoder-epoch-99-avg-1-chunk-16-left-128.onnx"),
            joiner=str(path / "joiner-epoch-99-avg-1-chunk-16-left-128.onnx"),
            provider=provider,
        )
    )
