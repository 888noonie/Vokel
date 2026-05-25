import { useState, useRef, useEffect } from "react";

const TARGET_SAMPLE_RATE = 16_000;

/** Linear resample (mono float32). Browsers often ignore requested AudioContext sampleRate. */
function resampleLinear(input: Float32Array, fromRate: number, toRate: number): Float32Array {
  if (fromRate === toRate || input.length === 0) {
    return new Float32Array(input);
  }
  const outLen = Math.max(1, Math.round((input.length * toRate) / fromRate));
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const srcPos = (i * fromRate) / toRate;
    const j = Math.floor(srcPos);
    const frac = srcPos - j;
    const a = input[j] ?? 0;
    const b = input[j + 1] ?? a;
    out[i] = a + (b - a) * frac;
  }
  return out;
}

interface AudioStreamerOptions {
  outputMuted: boolean;
  outputVolume: number;
}

export const useAudioStreamer = ({ outputMuted, outputVolume }: AudioStreamerOptions) => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [micVolume, setMicVolume] = useState(0);

  const captureContextRef = useRef<AudioContext | null>(null);
  const playbackContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const gainNodeRef = useRef<GainNode | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);

  const activeSourcesRef = useRef<AudioBufferSourceNode[]>([]);
  const nextPlaybackTimeRef = useRef<number>(0);
  const outputGainRef = useRef<GainNode | null>(null);

  const socketRef = useRef<WebSocket | null>(null);
  const socketListenerRef = useRef<((event: MessageEvent) => void) | null>(null);

  useEffect(() => {
    if (outputGainRef.current) {
      outputGainRef.current.gain.value = outputMuted ? 0 : outputVolume;
    }
  }, [outputMuted, outputVolume]);

  const startStreaming = async (socket: WebSocket) => {
    try {
      socketRef.current = socket;

      const captureContext = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
      captureContextRef.current = captureContext;

      const mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      mediaStreamRef.current = mediaStream;

      const sourceNode = captureContext.createMediaStreamSource(mediaStream);
      sourceNodeRef.current = sourceNode;

      const processor = captureContext.createScriptProcessor(2048, 1, 1);
      processorRef.current = processor;

      // Keep graph alive without routing mic to speakers (avoids feedback into browser mic).
      const silentGain = captureContext.createGain();
      silentGain.gain.value = 0;
      gainNodeRef.current = silentGain;

      sourceNode.connect(processor);
      processor.connect(silentGain);
      silentGain.connect(captureContext.destination);

      const inRate = captureContext.sampleRate;

      const playbackContext = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)({
        sampleRate: 24_000,
      });
      playbackContextRef.current = playbackContext;
      const outputGain = playbackContext.createGain();
      outputGain.gain.value = outputMuted ? 0 : outputVolume;
      outputGain.connect(playbackContext.destination);
      outputGainRef.current = outputGain;

      const stopAllPlayback = () => {
        activeSourcesRef.current.forEach((src) => {
          try {
            src.stop();
          } catch {
            /* already stopped */
          }
        });
        activeSourcesRef.current = [];
        nextPlaybackTimeRef.current = 0;
      };

      const socketListener = async (event: MessageEvent) => {
        if (event.data instanceof Blob) {
          const arrayBuffer = await event.data.arrayBuffer();
          const float32Data = new Float32Array(arrayBuffer);
          if (float32Data.length === 0) return;

          const buffer = playbackContext.createBuffer(1, float32Data.length, playbackContext.sampleRate);
          buffer.getChannelData(0).set(float32Data);

          const source = playbackContext.createBufferSource();
          source.buffer = buffer;
          source.connect(outputGainRef.current ?? playbackContext.destination);

          const startTime = Math.max(playbackContext.currentTime, nextPlaybackTimeRef.current);
          source.start(startTime);
          nextPlaybackTimeRef.current = startTime + buffer.duration;
          activeSourcesRef.current.push(source);

          source.onended = () => {
            activeSourcesRef.current = activeSourcesRef.current.filter((s) => s !== source);
          };
        } else if (typeof event.data === "string") {
          const data = JSON.parse(event.data) as { type?: string };
          if (data.type === "playback_stop") {
            stopAllPlayback();
          }
        }
      };

      socket.addEventListener("message", socketListener);
      socketListenerRef.current = socketListener;

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        // Copy only this render quantum; inputData.buffer is often larger than one chunk.
        const chunk = new Float32Array(inputData.length);
        chunk.set(inputData);

        let sum = 0;
        for (let i = 0; i < chunk.length; i++) {
          sum += chunk[i] * chunk[i];
        }
        const rms = Math.sqrt(sum / chunk.length);
        setMicVolume(rms);

        const pcm16k = resampleLinear(chunk, inRate, TARGET_SAMPLE_RATE);

        if (socket.readyState === WebSocket.OPEN) {
          const copy = pcm16k.slice();
          socket.send(copy.buffer);
        }
      };

      setIsStreaming(true);
    } catch (err) {
      console.error("Failed to start client-side audio capture:", err);
      stopStreaming();
    }
  };

  const stopStreaming = () => {
    setIsStreaming(false);
    setMicVolume(0);

    const sock = socketRef.current;
    const listener = socketListenerRef.current;
    if (sock && listener) {
      sock.removeEventListener("message", listener);
    }
    socketListenerRef.current = null;
    socketRef.current = null;

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }

    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (gainNodeRef.current) {
      gainNodeRef.current.disconnect();
      gainNodeRef.current = null;
    }
    if (sourceNodeRef.current) {
      sourceNodeRef.current.disconnect();
      sourceNodeRef.current = null;
    }

    if (captureContextRef.current) {
      captureContextRef.current.close().catch(() => {});
      captureContextRef.current = null;
    }

    activeSourcesRef.current.forEach((source) => {
      try {
        source.stop();
      } catch {
        /* already stopped */
      }
    });
    activeSourcesRef.current = [];
    nextPlaybackTimeRef.current = 0;
    outputGainRef.current?.disconnect();
    outputGainRef.current = null;

    if (playbackContextRef.current) {
      playbackContextRef.current.close().catch(() => {});
      playbackContextRef.current = null;
    }
  };

  useEffect(() => {
    return () => {
      stopStreaming();
    };
  }, []);

  return {
    startStreaming,
    stopStreaming,
    isStreaming,
    micVolume,
  };
};
