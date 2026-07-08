import { useCallback, useState, useEffect, useRef } from "react";
import {
  Mic,
  MicOff,
  Radio,
  Settings,
  Flame,
  Cpu,
  Power,
  AlertCircle,
  Brain,
  Timer,
  Pause,
  Play,
  RotateCcw,
  Download,
  Save,
  Trash2,
  Wrench,
  Bot,
  Volume2,
  VolumeX,
  Loader2,
  Eye,
  Aperture,
} from "lucide-react";
import { LatencyScoreboard } from "./components/LatencyScoreboard";
import { WaveformVisualizer } from "./components/WaveformVisualizer";
import { TranscriptStream } from "./components/TranscriptStream";
import type { Message } from "./components/TranscriptStream";
import { WorkspaceTabs } from "./components/WorkspaceTabs";
import type { AgentEvent, ExecuteState } from "./components/AgentConsole";
import type { ChatSnapshot, WorkspaceTab } from "./components/WorkspaceTabs";
import { useAudioStreamer } from "./hooks/useAudioStreamer";
import { playToolEndClick, playToolStartClick } from "./audio/toolCue";
import { VoiceOrb } from "./components/VoiceOrb";
import { PersonaSelector } from "./components/PersonaSelector";
import { LiveVisionPanel } from "./components/LiveVisionPanel";
import type { VisionFrame } from "./components/LiveVisionPanel";
import { BeatIndicator } from "./components/BeatIndicator";
import type { BeatPulse } from "./components/BeatIndicator";
import { CollapsibleSection } from "./components/CollapsibleSection";

type Mode = "local" | "browser";
type AgentBackend = "builtin" | "hermes";
type LlmConnection = "jan" | "lmstudio" | "hermes";
type Status = "idle" | "listening" | "generating" | "speaking" | "paused" | "capturing_vision";

const JAN_MODEL = "HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive";
const LMSTUDIO_MODEL = "gemma-4-e4b-uncensored-hauhaucs-aggressive";

const LLM_PRESETS = {
  jan: {
    url: "http://127.0.0.1:6767/v1/chat/completions",
    model: JAN_MODEL,
    label: "Jan",
  },
  lmstudio: {
    url: "http://localhost:1234/v1/chat/completions",
    model: LMSTUDIO_MODEL,
    label: "LM Studio",
  },
} as const;

interface MemoryFact {
  id: number;
  text: string;
  created_at_ns: number;
}

const kokoroVoices = [
  "af_alloy",
  "af_aoede",
  "af_heart",
  "af_bella",
  "af_jessica",
  "af_kore",
  "af_nicole",
  "af_nova",
  "af_river",
  "af_sarah",
  "af_sky",
  "am_adam",
  "am_echo",
  "am_eric",
  "am_fenrir",
  "am_liam",
  "am_michael",
  "am_onyx",
  "am_puck",
  "am_santa",
  "bf_alice",
  "bf_emma",
  "bf_isabella",
  "bf_lily",
  "bm_daniel",
  "bm_fable",
  "bm_george",
  "bm_lewis",
];

const voicePrefsKey = "vokel.voicePrefs.v1";
const previousChatsKey = "vokel.previousChats.v1";
const hermesPrefsKey = "vokel.hermesPrefs.v1";
const connectionPrefsKey = "vokel.connectionPrefs.v1";

function loadJson<T>(key: string, fallback: T): T {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function App() {
  const [mode, setMode] = useState<Mode>("browser");
  const [isConnected, setIsConnected] = useState(false);
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [status, setStatus] = useState<Status>("idle");
  const [messages, setMessages] = useState<Message[]>([]);
  const [metrics, setMetrics] = useState<Record<string, number>>({});

  // Live Voice Presence + Persona Layer (Command #2)
  const [currentPersona, setCurrentPersona] = useState<string>("cosmic");
  const [personaPrompt, setPersonaPrompt] = useState<string>("You are a curious cosmic explorer exploring the edges of understanding. Be concise, wonder-filled, and speak naturally for voice.");
  const [error, setError] = useState<string | null>(null);
  const [memoryFacts, setMemoryFacts] = useState<MemoryFact[]>([]);
  const [selectedMemoryIds, setSelectedMemoryIds] = useState<Set<number>>(new Set());
  const [memoryDraft, setMemoryDraft] = useState("");
  const [isPaused, setIsPaused] = useState(false);
  const [workspaceTab, setWorkspaceTab] = useState<WorkspaceTab>("settings");
  const [previousChats, setPreviousChats] = useState<ChatSnapshot[]>(() =>
    loadJson<ChatSnapshot[]>(previousChatsKey, [])
  );
  const [agentEvents, setAgentEvents] = useState<AgentEvent[]>([]);
  const [activeToolName, setActiveToolName] = useState<string | null>(null);
  const [executeState, setExecuteState] = useState<ExecuteState>({
    armed: false,
    risk: "none",
    detail: "idle",
  });

  // Settings
  const savedVoicePrefs = loadJson(voicePrefsKey, {
    playbackBackend: "kokoro",
    voice: "af_heart",
    ttsSpeed: 1,
    outputMuted: false,
    outputVolume: 1,
    musicalMode: false,
    musicalBpm: 90,
    musicalLevel: 1,
    musicalStyle: "beat",
    sessionContext: "",
  });
  const savedConnection = loadJson<{ provider?: LlmConnection }>(connectionPrefsKey, {
    provider: "jan",
  });
  const initialConnection: LlmConnection =
    savedConnection.provider === "lmstudio" || savedConnection.provider === "hermes"
      ? savedConnection.provider
      : "jan";
  const initialPreset =
    initialConnection === "lmstudio" ? LLM_PRESETS.lmstudio : LLM_PRESETS.jan;

  const [lmStudioUrl, setLmStudioUrl] = useState<string>(initialPreset.url);
  const [lmStudioModel, setLmStudioModel] = useState<string>(initialPreset.model);
  const [useLmNativeMcp, setUseLmNativeMcp] = useState(false);
  const [lmMcpIntegrations, setLmMcpIntegrations] = useState("");  // comma sep e.g. mcp/playwright,mcp/fetch
  const [playbackBackend, setPlaybackBackend] = useState(savedVoicePrefs.playbackBackend);
  const [voice, setVoice] = useState(savedVoicePrefs.voice);
  const [ttsSpeed, setTtsSpeed] = useState(savedVoicePrefs.ttsSpeed);
  const [outputMuted, setOutputMuted] = useState(savedVoicePrefs.outputMuted);
  const [outputVolume, setOutputVolume] = useState(savedVoicePrefs.outputVolume);
  const [musicalMode, setMusicalMode] = useState(Boolean(savedVoicePrefs.musicalMode));
  const [musicalBpm, setMusicalBpm] = useState(
    typeof savedVoicePrefs.musicalBpm === "number" ? savedVoicePrefs.musicalBpm : 90
  );
  const [musicalLevel, setMusicalLevel] = useState(
    typeof savedVoicePrefs.musicalLevel === "number" ? savedVoicePrefs.musicalLevel : 1
  );
  const [musicalStyle, setMusicalStyle] = useState<"beat" | "metronome">(
    savedVoicePrefs.musicalStyle === "metronome" ? "metronome" : "beat"
  );
  const [sessionContext, setSessionContext] = useState(
    typeof savedVoicePrefs.sessionContext === "string" ? savedVoicePrefs.sessionContext : ""
  );
  const [musicalTrackInfo, setMusicalTrackInfo] = useState<{
    durationSeconds: number;
    samples: number;
  } | null>(null);
  const [musicalTrackUploading, setMusicalTrackUploading] = useState(false);
  const [beatPulse, setBeatPulse] = useState<BeatPulse | null>(null);
  const [previewingVoice, setPreviewingVoice] = useState<string | null>(null);
  const [memoryEnabled, setMemoryEnabled] = useState(false);
  const [autoFollowupEnabled, setAutoFollowupEnabled] = useState(true);
  const [autoFollowupSeconds, setAutoFollowupSeconds] = useState(8);
  const [llmConnection, setLlmConnection] = useState<LlmConnection>(initialConnection);
  const [agentBackend, setAgentBackend] = useState<AgentBackend>(
    initialConnection === "hermes" ? "hermes" : "builtin"
  );
  const savedHermesPrefs = loadJson(hermesPrefsKey, {
    url: "http://127.0.0.1:8642",
    apiKey: "",
    model: "hermes-agent",
  });
  const [hermesUrl, setHermesUrl] = useState(savedHermesPrefs.url);
  const [hermesApiKey, setHermesApiKey] = useState(savedHermesPrefs.apiKey);
  const [hermesSessionId, setHermesSessionId] = useState("");
  const [hermesModel, setHermesModel] = useState(savedHermesPrefs.model);
  const [activeHermesSessionId, setActiveHermesSessionId] = useState<string | null>(null);
  const [probeStatus, setProbeStatus] = useState<"idle" | "probing" | "ok" | "error">("idle");
  const [visionVoiceEnabled, setVisionVoiceEnabled] = useState(false);
  const [visionDevice, setVisionDevice] = useState("/dev/video0");
  const [voiceVisionFrame, setVoiceVisionFrame] = useState<VisionFrame | null>(null);
  const [liveVisionControl, setLiveVisionControl] = useState<{
    action: "start_live" | "stop_live";
    nonce: number;
  } | null>(null);
  const [visionPanelStatus, setVisionPanelStatus] = useState<{
    label: string;
    aiViewing: boolean;
  }>({ label: "Camera idle", aiViewing: false });
  const [beatCollapsedPulse, setBeatCollapsedPulse] = useState(false);
  const [ffmpegAvailable, setFfmpegAvailable] = useState(true);

  const socketRef = useRef<WebSocket | null>(null);
  // Set by LiveVisionPanel; grabs one JPEG from the live preview on the server's request.
  const capturePreviewFrameRef = useRef<(() => string | null) | null>(null);
  const previewingVoiceRef = useRef<string | null>(null);
  const previewAudioContextRef = useRef<AudioContext | null>(null);
  const previewPlaybackTimeRef = useRef(0);
  const toolAudioContextRef = useRef<AudioContext | null>(null);
  const toolCueActiveRef = useRef(false);
  const messagesRef = useRef<Message[]>([]);
  const musicalLevelSendTimerRef = useRef<number | null>(null);
  const musicalTrackSlotRef = useRef<string | null>(null);
  const tapTempoTimesRef = useRef<number[]>([]);

  // Hook for audio streaming and local queue scheduling in browser mode
  const { startStreaming, stopStreaming, isStreaming, micVolume } = useAudioStreamer({
    outputMuted,
    outputVolume,
  });

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  useEffect(() => {
    window.localStorage.setItem(
      voicePrefsKey,
      JSON.stringify({
        playbackBackend,
        voice,
        ttsSpeed,
        outputMuted,
        outputVolume,
        musicalMode,
        musicalBpm,
        musicalLevel,
        musicalStyle,
        sessionContext,
      })
    );
  }, [
    playbackBackend,
    voice,
    ttsSpeed,
    outputMuted,
    outputVolume,
    musicalMode,
    musicalBpm,
    musicalLevel,
    musicalStyle,
    sessionContext,
  ]);

  const sendMusicalLevel = useCallback(
    (level: number) => {
      if (!socketRef.current || !isConnected || !isSessionActive) return;
      socketRef.current.send(
        JSON.stringify({
          type: "set_musical_level",
          level,
        })
      );
    },
    [isConnected, isSessionActive]
  );

  const handleMusicalLevelChange = useCallback(
    (level: number) => {
      setMusicalLevel(level);
      if (musicalLevelSendTimerRef.current !== null) {
        window.clearTimeout(musicalLevelSendTimerRef.current);
      }
      musicalLevelSendTimerRef.current = window.setTimeout(() => {
        sendMusicalLevel(level);
        musicalLevelSendTimerRef.current = null;
      }, 100);
    },
    [sendMusicalLevel]
  );

  const sendMusicalNudge = useCallback(
    (ms: number) => {
      if (!socketRef.current || !isConnected || !isSessionActive) return;
      socketRef.current.send(
        JSON.stringify({
          type: "set_musical_nudge",
          ms,
        })
      );
    },
    [isConnected, isSessionActive]
  );

  const handleTapTempo = useCallback(() => {
    const now = performance.now();
    const times = tapTempoTimesRef.current;
    if (times.length > 0 && now - times[times.length - 1] > 2000) {
      times.length = 0;
    }
    times.push(now);
    if (times.length > 5) {
      times.shift();
    }
    if (times.length < 2) {
      return;
    }
    const intervals: number[] = [];
    for (let index = 1; index < times.length; index += 1) {
      intervals.push(times[index] - times[index - 1]);
    }
    const recent = intervals.slice(-4);
    const averageMs = recent.reduce((sum, value) => sum + value, 0) / recent.length;
    const nextBpm = Math.round(60000 / averageMs);
    setMusicalBpm(Math.min(160, Math.max(60, Math.round(nextBpm / 5) * 5)));
  }, []);

  useEffect(() => {
    if (!beatPulse) {
      setBeatCollapsedPulse(false);
      return;
    }
    setBeatCollapsedPulse(true);
    const timer = window.setTimeout(() => setBeatCollapsedPulse(false), 180);
    return () => window.clearTimeout(timer);
  }, [beatPulse?.receivedAt]);

  const clearMusicalTrack = useCallback(async () => {
    const slot = musicalTrackSlotRef.current;
    setMusicalTrackInfo(null);
    if (!slot) {
      return;
    }
    try {
      const response = await fetch(`/api/musical/track?slot=${encodeURIComponent(slot)}`, {
        method: "DELETE",
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        console.debug(
          "Backing track clear failed:",
          typeof payload.detail === "string" ? payload.detail : response.status
        );
      }
    } catch (clearError) {
      console.debug("Backing track clear failed:", clearError);
    }
  }, []);

  const uploadMusicalTrack = useCallback(async (file: File) => {
    const slot = musicalTrackSlotRef.current;
    if (!slot) {
      setError("Connect to Vokel before uploading a backing track.");
      return;
    }
    setMusicalTrackUploading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const response = await fetch(`/api/musical/track?slot=${encodeURIComponent(slot)}`, {
        method: "POST",
        body: form,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(
          typeof payload.detail === "string"
            ? payload.detail
            : "Backing track upload failed"
        );
      }
      setMusicalTrackInfo({
        durationSeconds: Number(payload.duration_seconds ?? 0),
        samples: Number(payload.samples ?? 0),
      });
      setError(null);
    } catch (uploadError) {
      const message =
        uploadError instanceof Error ? uploadError.message : "Backing track upload failed";
      setError(message);
    } finally {
      setMusicalTrackUploading(false);
    }
  }, []);

  useEffect(() => {
    window.localStorage.setItem(previousChatsKey, JSON.stringify(previousChats));
  }, [previousChats]);

  useEffect(() => {
    window.localStorage.setItem(
      hermesPrefsKey,
      JSON.stringify({ url: hermesUrl, apiKey: hermesApiKey, model: hermesModel })
    );
  }, [hermesUrl, hermesApiKey, hermesModel]);

  const getToolAudioContext = () => {
    if (!toolAudioContextRef.current) {
      toolAudioContextRef.current = new AudioContext();
    }
    return toolAudioContextRef.current;
  };

  const startToolCue = async () => {
    toolCueActiveRef.current = true;
    await playToolStartClick(getToolAudioContext());
  };

  const stopToolCue = async (playEnd = true) => {
    if (!toolCueActiveRef.current) return;
    toolCueActiveRef.current = false;
    const audioContext = toolAudioContextRef.current;
    if (playEnd && audioContext) {
      await playToolEndClick(audioContext);
    }
  };

  const playPreviewAudio = async (blob: Blob) => {
    if (outputMuted) return;
    const arrayBuffer = await blob.arrayBuffer();
    const float32Data = new Float32Array(arrayBuffer);
    if (float32Data.length === 0) return;

    if (!previewAudioContextRef.current) {
      previewAudioContextRef.current = new AudioContext({ sampleRate: 24_000 });
    }
    const audioContext = previewAudioContextRef.current;
    if (audioContext.state === "suspended") {
      await audioContext.resume();
    }

    const buffer = audioContext.createBuffer(1, float32Data.length, audioContext.sampleRate);
    buffer.getChannelData(0).set(float32Data);

    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    const gain = audioContext.createGain();
    gain.gain.value = outputVolume;
    source.connect(gain);
    gain.connect(audioContext.destination);

    const startTime = Math.max(audioContext.currentTime, previewPlaybackTimeRef.current);
    source.start(startTime);
    previewPlaybackTimeRef.current = startTime + buffer.duration;
  };

  const stopVoicePreview = () => {
    previewingVoiceRef.current = null;
    setPreviewingVoice(null);
    previewPlaybackTimeRef.current = 0;
  };

  // Close websocket on unmount
  useEffect(() => {
    return () => {
      toolCueActiveRef.current = false;
      previewingVoiceRef.current = null;
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, []);

  const handleConnect = () => {
    if (isConnected) {
      if (socketRef.current) socketRef.current.close();
      return;
    }

    setError(null);
    const wsUrl = `ws://${window.location.host}/api/ws`;
    const socket = new WebSocket(wsUrl);
    socketRef.current = socket;

    socket.onopen = () => {
      setIsConnected(true);
      setError(null);
    };

    socket.onclose = () => {
      setIsConnected(false);
      setIsSessionActive(false);
      setStatus("idle");
      setIsPaused(false);
      setActiveToolName(null);
      setExecuteState({ armed: false, risk: "none", detail: "idle" });
      void stopToolCue(false);
      stopVoicePreview();
      stopStreaming();
    };

    socket.onerror = () => {
      setError("WebSocket connection failed. Ensure the Vokel server is running with --web.");
    };

    socket.onmessage = async (event) => {
      if (event.data instanceof Blob) {
        if (previewingVoiceRef.current) {
          await playPreviewAudio(event.data);
        }
        return;
      }

      if (typeof event.data === "string") {
        // Handle JSON messages from server
        const data = JSON.parse(event.data);

        switch (data.type) {
          case "agent_event": {
            if (data.event === "gateway_health_ok") setProbeStatus("ok");
            if (data.event === "gateway_health_failed") setProbeStatus("error");
            
            const now = new Date();
            const event: AgentEvent = {
              id: `${now.getTime()}-${Math.random()}`,
              event: String(data.event ?? "event"),
              level: data.level === "error" || data.level === "warning" ? data.level : "info",
              backend: String(data.backend ?? "vokel"),
              detail: String(data.detail ?? ""),
              timestamp: now.toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
              }),
              session_id: typeof data.session_id === "string" ? data.session_id : null,
              risk: typeof data.risk === "string" ? data.risk : undefined,
            };
            setAgentEvents((prev) => [...prev.slice(-79), event]);
            break;
          }

          case "execute_state":
            if (typeof data.musical_track_slot === "string") {
              musicalTrackSlotRef.current = data.musical_track_slot;
            }
            if (typeof data.ffmpeg_available === "boolean") {
              setFfmpegAvailable(data.ffmpeg_available);
            }
            setExecuteState({
              armed: Boolean(data.armed),
              risk: String(data.risk ?? "none"),
              detail: String(data.detail ?? ""),
            });
            break;

          case "session_started":
            setIsSessionActive(true);
            setIsPaused(false);
            setStatus("listening");
            setActiveToolName(null);
            if (typeof data.hermes_session_id === "string") {
              setActiveHermesSessionId(data.hermes_session_id);
              if (!hermesSessionId.trim()) {
                setHermesSessionId(data.hermes_session_id);
              }
            } else {
              setActiveHermesSessionId(null);
            }
            if (data.mode === "browser") {
              startStreaming(socket);
            }
            break;

          case "session_stopped":
            setIsSessionActive(false);
            setStatus("idle");
            setIsPaused(false);
            setActiveToolName(null);
            setActiveHermesSessionId(null);
            setBeatPulse(null);
            setExecuteState({ armed: false, risk: "none", detail: "idle" });
            void stopToolCue(false);
            stopVoicePreview();
            stopStreaming();
            break;

          case "beat":
            setBeatPulse({
              bar: Number(data.bar ?? 0),
              beat: Number(data.beat ?? 0),
              bpm: Number(data.bpm ?? musicalBpm),
              receivedAt: Date.now(),
            });
            break;

          case "status":
            setStatus(data.status);
            setIsPaused(data.status === "paused");
            break;

          case "session_reset":
            setMessages([]);
            setMetrics({});
            break;

          case "memory_facts": {
            const facts = (data.facts ?? []) as MemoryFact[];
            setMemoryFacts(facts);
            setSelectedMemoryIds((current) => {
              const valid = new Set(facts.map((fact) => fact.id));
              return new Set([...current].filter((id) => valid.has(id)));
            });
            break;
          }

          case "voice_preview_started":
            previewingVoiceRef.current = data.voice as string;
            setPreviewingVoice(data.voice as string);
            previewPlaybackTimeRef.current = 0;
            break;

          case "voice_preview_finished":
            window.setTimeout(stopVoicePreview, 250);
            if (data.error) {
              setError(`Voice preview failed: ${data.error}`);
            }
            break;

          case "partial_transcript":
            setMessages((prev) => {
              const last = prev[prev.length - 1];
              if (last && last.role === "user" && last.isPartial) {
                return [...prev.slice(0, -1), { ...last, text: data.text }];
              } else {
                return [
                  ...prev,
                  { id: Math.random().toString(), role: "user", text: data.text, isPartial: true },
                ];
              }
            });
            break;

          case "stable_transcript":
            setMessages((prev) => {
              const last = prev[prev.length - 1];
              if (last && last.role === "user" && last.isPartial) {
                return [...prev.slice(0, -1), { ...last, text: data.text, isPartial: false }];
              } else {
                return [
                  ...prev,
                  { id: Math.random().toString(), role: "user", text: data.text, isPartial: false },
                ];
              }
            });
            break;

          case "final_transcript":
            // Replace any partial text with the final verified transcript
            setMessages((prev) => {
              const filtered = prev.filter((m) => !m.isPartial);
              const last = filtered[filtered.length - 1];
              if (last && last.role === "user" && last.text === data.text) {
                return filtered;
              }
              return [...filtered, { id: Math.random().toString(), role: "user", text: data.text }];
            });
            break;

          case "user_transcript":
            setMessages((prev) => {
              const filtered = prev.filter((m) => !m.isPartial);
              const last = filtered[filtered.length - 1];
              if (last && last.role === "user" && last.text === data.text) {
                return filtered;
              }
              return [
                ...filtered,
                { id: `u-${Date.now()}`, role: "user", text: data.text as string, isPartial: false },
              ];
            });
            break;

          case "assistant_reply":
            setMessages((prev) => {
              const text = data.text as string;
              const last = prev[prev.length - 1];
              if (last && last.role === "assistant") {
                return [...prev.slice(0, -1), { ...last, text }];
              }
              return [...prev, { id: `a-${Date.now()}`, role: "assistant", text }];
            });
            break;

          case "assistant_phrase":
            setMessages((prev) => {
              const last = prev[prev.length - 1];
              if (last && last.role === "assistant") {
                // If appending to existing assistant turn, join with space
                return [
                  ...prev.slice(0, -1),
                  { ...last, text: `${last.text} ${data.phrase}`.trim() },
                ];
              } else {
                return [
                  ...prev,
                  { id: Math.random().toString(), role: "assistant", text: data.phrase },
                ];
              }
            });
            break;

          case "playback_stop":
            // Trigger browser-side player queue clearance (handled automatically inside useAudioStreamer)
            setActiveToolName(null);
            void stopToolCue(false);
            break;

          case "telemetry":
            if (
              data.event === "tool_call_forced" ||
              data.event === "tool_call_started" ||
              data.event === "tool_reassurance_queued"
            ) {
              const fields = data.fields && typeof data.fields === "object" ? data.fields : {};
              const toolName = typeof fields.tool_name === "string" ? fields.tool_name : null;
              setActiveToolName(toolName);
              void startToolCue();
            }
            if (
              data.event === "tool_call_finished" ||
              data.event === "tool_call_failed" ||
              data.event === "generation_finished" ||
              data.event === "generation_cancelled"
            ) {
              setActiveToolName(null);
              void stopToolCue();
            }
            break;

          case "summary":
            setMetrics(data.metrics);
            break;

          case "vision_context_captured":
            setVoiceVisionFrame({
              device: String(data.device ?? ""),
              description: "Fresh frame attached to your spoken question. Gemma's answer appears in the transcript.",
              image_data_url: String(data.image_data_url ?? ""),
              capture_seconds: Number(data.capture_seconds ?? 0),
              inference_seconds: 0,
            });
            break;

          case "vision_context_state":
            // Backend armed Camera Questions from a spoken command ("watch me"/"shoot").
            setVisionVoiceEnabled(Boolean(data.enabled));
            if (typeof data.device === "string" && data.device) {
              setVisionDevice(data.device);
            }
            break;

          case "vision_control":
            // "action"/"cut" drive the Local Vision Window's live AI watch loop.
            if (data.action === "start_live" || data.action === "stop_live") {
              setLiveVisionControl({ action: data.action, nonce: Date.now() });
            }
            break;

          case "request_camera_frame": {
            // Server wants a still while the browser holds the camera: grab one from
            // the live preview (null if no preview is active) and send it back.
            const grab = capturePreviewFrameRef.current;
            const imageDataUrl = grab ? grab() : null;
            socketRef.current?.send(
              JSON.stringify({ type: "camera_frame", image_data_url: imageDataUrl })
            );
            break;
          }

          case "auto_followup":
            break;

          case "error":
            void stopToolCue(false);
            setError(data.message);
            break;
        }
      }
    };
  };

  const handleStartSession = () => {
    if (!socketRef.current || !isConnected) return;

    setMessages([]);
    setMetrics({});
    setAgentEvents([]);
    setActiveToolName(null);
    setExecuteState({ armed: false, risk: "none", detail: "idle" });
    setError(null);

    socketRef.current.send(
      JSON.stringify({
        type: "start_session",
        mode,
        agent_backend: agentBackend,
        url: lmStudioUrl,
        model: lmStudioModel,
        hermes_url: hermesUrl,
        hermes_api_key: hermesApiKey,
        hermes_session_id: hermesSessionId.trim(),
        hermes_model: hermesModel,
        playback: playbackBackend,
        voice,
        tts_speed: ttsSpeed,
        memory: agentBackend === "builtin" && memoryEnabled,
        auto_followup: agentBackend === "builtin" && autoFollowupEnabled,
        auto_followup_seconds: autoFollowupSeconds,
        vision_voice_enabled: visionVoiceEnabled,
        vision_device: visionDevice,
        use_lm_native_chat: useLmNativeMcp,
        lm_mcp_integrations: lmMcpIntegrations,
        // Persona + Live Voice Presence (Command #2)
        persona: currentPersona,
        persona_prompt: personaPrompt,
        musical_mode: musicalMode,
        musical_bpm: musicalBpm,
        musical_level: musicalLevel,
        musical_style: musicalStyle,
        session_context: sessionContext,
      })
    );
  };

  const handleStopSession = () => {
    if (!socketRef.current || !isConnected) return;
    persistCurrentChat();
    setActiveToolName(null);

    socketRef.current.send(
      JSON.stringify({
        type: "stop_session",
      })
    );
  };

  const handleInterrupt = () => {
    if (!socketRef.current || !isConnected) return;
    setActiveToolName(null);

    socketRef.current.send(
      JSON.stringify({
        type: "interrupt",
      })
    );
  };

  const persistCurrentChat = () => {
    const transcript = messagesRef.current.filter((message) => !message.isPartial);
    if (transcript.length === 0) return;
    const firstUser = transcript.find((message) => message.role === "user");
    const title = (firstUser?.text ?? transcript[0].text).slice(0, 72) || "Untitled session";
    const snapshot: ChatSnapshot = {
      id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      title,
      createdAt: new Date().toISOString(),
      backend: agentBackend,
      messages: transcript,
    };
    setPreviousChats((current) => [snapshot, ...current].slice(0, 20));
  };

  const restorePreviousChat = (chat: ChatSnapshot) => {
    setMessages(chat.messages);
    setWorkspaceTab("previous");
  };

  const clearPreviousChats = () => {
    setPreviousChats([]);
  };

  const handleArmExecute = () => {
    sendControl("arm_execute", { risk: "medium" });
  };

  const handleCancelExecute = () => {
    sendControl("cancel_execute");
  };

  const handleConfirmExecute = () => {
    sendControl("confirm_execute");
  };

  const sendControl = (type: string, payload: Record<string, unknown> = {}) => {
    if (!socketRef.current || !isConnected) return;
    socketRef.current.send(JSON.stringify({ type, ...payload }));
  };

  // Persona + Live Presence (Command #2)
  const handlePersonaChange = (id: string, prompt: string) => {
    setCurrentPersona(id);
    setPersonaPrompt(prompt);

    // Send to backend so the next session / turn can use the persona system prompt
    if (socketRef.current && isConnected) {
      socketRef.current.send(JSON.stringify({
        type: "set_persona",
        persona: id,
        prompt,
      }));
    }
  };

  // Derive reactive orb state from existing pipeline status
  const getOrbState = (): 'idle' | 'listening' | 'thinking' | 'speaking' => {
    if (status === "listening") return "listening";
    if (status === "generating") return "thinking";
    if (status === "speaking") return "speaking";
    return "idle";
  };

  const orbState = getOrbState();
  const orbAmplitude = status === "listening" ? micVolume : 0;

  const handlePauseResume = () => {
    if (!isSessionActive) return;
    sendControl(isPaused ? "resume_session" : "pause_session");
    if (mode === "browser") {
      if (isPaused) {
        if (socketRef.current) startStreaming(socketRef.current);
      } else {
        stopStreaming();
      }
    }
  };

  const handleReset = () => {
    if (!isSessionActive) return;
    sendControl("reset_session");
    setMessages([]);
    setMetrics({});
  };

  const handleSaveMemory = () => {
    const text = memoryDraft.trim();
    if (!text) return;
    sendControl("memory_save", { text });
    setMemoryDraft("");
  };

  const handleUpdateMemory = (id: number, text: string) => {
    sendControl("memory_update", { id, text });
  };

  const handlePreviewVoice = (voiceName: string) => {
    if (!socketRef.current || !isConnected || isSessionActive) return;
    setError(null);
    setVoice(voiceName);
    previewingVoiceRef.current = voiceName;
    setPreviewingVoice(voiceName);
    previewPlaybackTimeRef.current = 0;
    socketRef.current.send(JSON.stringify({
      type: "preview_voice",
      voice: voiceName,
      tts_speed: ttsSpeed,
    }));
  };

  const handleDeleteMemory = (id: number) => {
    sendControl("memory_delete", { id });
  };

  const toggleMemorySelection = (id: number) => {
    setSelectedMemoryIds((current) => {
      const next = new Set(current);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const handleProbeHermes = () => {
    if (!socketRef.current || !isConnected) return;
    setProbeStatus("probing");
    socketRef.current.send(
      JSON.stringify({
        type: "probe_hermes",
        hermes_url: hermesUrl,
        hermes_api_key: hermesApiKey,
      })
    );
  };

  const handleVisionVoiceEnabledChange = useCallback((enabled: boolean) => {
    setVisionVoiceEnabled(enabled);
    if (socketRef.current && isConnected && isSessionActive) {
      socketRef.current.send(JSON.stringify({
        type: "set_vision_context",
        enabled,
        device: visionDevice,
      }));
    }
  }, [isConnected, isSessionActive, visionDevice]);

  const handleVisionDeviceChange = useCallback((device: string) => {
    setVisionDevice(device);
    if (socketRef.current && isConnected && isSessionActive && visionVoiceEnabled) {
      socketRef.current.send(JSON.stringify({
        type: "set_vision_context",
        enabled: true,
        device,
      }));
    }
  }, [isConnected, isSessionActive, visionVoiceEnabled]);

  const handleExportMarkdown = () => {
    const selectedFacts = memoryFacts.filter((fact) => selectedMemoryIds.has(fact.id));
    const lines = [
      "# Vokel Session Export",
      "",
      `Exported: ${new Date().toISOString()}`,
      "",
      "## Transcript",
      "",
      ...(
        messages.length
          ? messages
              .filter((message) => !message.isPartial)
              .flatMap((message) => [
                `### ${message.role === "user" ? "User" : "Assistant"}`,
                "",
                message.text.trim(),
                "",
              ])
          : ["No transcript captured.", ""]
      ),
      "## Selected Memory Notes",
      "",
      ...(
        selectedFacts.length
          ? selectedFacts.map((fact) => `- ${fact.text}`)
          : ["No memory notes selected."]
      ),
      "",
    ];

    const blob = new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `vokel-session-${new Date().toISOString().replace(/[:.]/g, "-")}.md`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  };

  const selectLlmConnection = (provider: LlmConnection) => {
    setLlmConnection(provider);
    localStorage.setItem(connectionPrefsKey, JSON.stringify({ provider }));
    if (provider === "hermes") {
      setAgentBackend("hermes");
      return;
    }
    setAgentBackend("builtin");
    const preset = LLM_PRESETS[provider];
    setLmStudioUrl(preset.url);
    setLmStudioModel(preset.model);
    if (provider === "jan") {
      setUseLmNativeMcp(false);
    }
  };

  const activeConnectionLabel =
    llmConnection === "hermes" ? "Hermes" : llmConnection === "jan" ? "Jan" : "LM Studio";
  const routeLabel = llmConnection === "hermes" ? "External agent" : "Local model";
  const toolsOwnerLabel =
    llmConnection === "hermes"
      ? "Hermes-owned"
      : llmConnection === "jan"
        ? "Jan-owned"
        : "LM Studio-owned";
  const privacyLabel = llmConnection === "hermes" ? "External agent active" : "Local";
  const voiceLabel =
    playbackBackend === "kokoro"
      ? `Local Kokoro (${voice})`
      : playbackBackend === "spd-say"
        ? "Local Speech Dispatcher"
        : "Local Console";
  const interruptLabel = isConnected && isSessionActive ? "Available" : "Not available";
  const consentLabel = executeState.armed
    ? "Armed"
    : executeState.risk !== "none"
      ? "Required"
      : "Not armed";
  const trustSummary = `${activeConnectionLabel} • ${routeLabel} • ${interruptLabel}`;
  const sessionStateLabel = isPaused
    ? "paused"
    : isSessionActive
      ? status
      : "idle";
  const activeActionLabel = activeToolName
    ? activeToolName === "search_web"
      ? "Searching the web"
      : `Using ${activeToolName}`
    : status === "generating"
      ? "Working on your request"
      : status === "capturing_vision"
        ? "Capturing local camera context"
      : status === "speaking"
        ? "Speaking response"
        : null;
  const platformCapabilitiesCopy =
    llmConnection === "hermes"
      ? "Hermes owns reasoning, tools, and memory. Vokel is the voice front-end."
      : llmConnection === "jan"
        ? "Jan serves the model on this machine. Vokel handles speech; Jan handles text generation."
        : "LM Studio serves the model and can own MCP tools. Enable native MCP below if you use ~/.lmstudio/mcp.json.";
  const sessionStatusBadge = (
    <span
      className={`rounded-full border px-2 py-1 text-[10px] font-bold uppercase tracking-wide ${
        isPaused
          ? "border-amber-500/30 bg-amber-500/10 text-amber-300"
          : status === "listening"
            ? "border-emerald-500/25 bg-emerald-500/10 text-emerald-300"
            : status === "capturing_vision"
              ? "border-rose-500/30 bg-rose-500/10 text-rose-200"
              : status === "generating" || status === "speaking"
                ? "border-blue-500/25 bg-blue-500/10 text-blue-300"
                : "border-zinc-800 bg-zinc-950 text-zinc-500"
      }`}
    >
      {isPaused ? "Paused" : status}
    </span>
  );
  const beatCollapsedBadge =
    musicalMode && isSessionActive ? (
      <span
        className={`beat-indicator-dot beat-indicator-dot--downbeat ${
          beatCollapsedPulse ? "beat-indicator-dot--active" : ""
        }`}
        aria-hidden
      />
    ) : null;
  const visionCollapsedBadge = visionPanelStatus.aiViewing ? (
    <span className="flex items-center gap-1.5 rounded-full border border-rose-400/50 bg-rose-950/85 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide text-rose-50">
      <Aperture className="h-3 w-3 animate-pulse" />
      AI viewing
    </span>
  ) : null;
  const llmEndpointLabel =
    llmConnection === "jan"
      ? "Jan API Endpoint"
      : llmConnection === "lmstudio"
        ? "LM Studio API Endpoint"
        : "LLM API Endpoint";

  return (
    <div className="app-shell min-h-screen text-zinc-100 flex flex-col font-sans selection:bg-purple-500/30 selection:text-purple-200">
      {/* Header */}
      <header className="border-b border-white/10 bg-zinc-950/70 backdrop-blur-xl sticky top-0 z-40">
        <div className="safe-container min-h-16 py-3 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-2xl bg-gradient-to-tr from-purple-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-purple-500/30 ring-1 ring-white/10">
              <Radio className="w-5 h-5 text-white stroke-[1.8] animate-pulse" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight bg-gradient-to-r from-zinc-100 to-zinc-400 bg-clip-text text-transparent leading-none">
                Vokel
              </h1>
              <p className="text-[10px] text-zinc-500 font-mono tracking-wider mt-0.5">
                VOICE-INVOKED LOCAL INTELLIGENCE
              </p>
            </div>
          </div>

          <div className="flex items-center justify-between gap-3 sm:justify-end">
            <div className="flex items-center space-x-2">
              <span
                className={`w-2.5 h-2.5 rounded-full ${
                  isConnected ? "bg-emerald-500 animate-pulse" : "bg-zinc-700"
                }`}
              />
              <span className="text-xs font-medium text-zinc-400 font-mono">
                {isConnected ? "SERVER CONNECTED" : "SERVER DISCONNECTED"}
              </span>
            </div>

            <button
              onClick={handleConnect}
              className={`touch-button flex items-center space-x-1.5 px-4 rounded-xl text-xs font-semibold font-mono transition-all duration-200 border ${
                isConnected
                  ? "bg-zinc-900 hover:bg-zinc-800 text-zinc-400 border-zinc-850"
                  : "bg-purple-650 hover:bg-purple-600 text-white border-purple-500/30 shadow-lg shadow-purple-500/10"
              }`}
            >
              <Power className="w-3.5 h-3.5" />
              <span>{isConnected ? "DISCONNECT" : "CONNECT"}</span>
            </button>
          </div>
        </div>
      </header>

      {/* Mobile Privacy & Connection Banner */}
      <div className="bg-zinc-900 border-b border-zinc-800 xl:hidden">
        <div className="safe-container py-2 flex items-center justify-between text-[10px] font-mono tracking-wider text-zinc-400">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1">
              <Cpu className="w-3 h-3 text-purple-400" />
              {llmConnection === "hermes" ? "HERMES" : llmConnection === "jan" ? "JAN" : "LM STUDIO"}
            </span>
            <span className="flex items-center gap-1">
              <Wrench className="w-3 h-3 text-indigo-400" />
              PLATFORM
            </span>
          </div>
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1">
              {outputMuted ? <VolumeX className="w-3 h-3 text-amber-400" /> : <Volume2 className="w-3 h-3 text-emerald-400" />}
              {outputMuted ? "MUTED" : "ON"}
            </span>
            {executeState.armed && (
              <span className="flex items-center gap-1 text-rose-400 font-bold">
                <Flame className="w-3 h-3 animate-pulse" />
                ARMED
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="safe-container flex-1 py-6 sm:py-8 grid grid-cols-1 xl:grid-cols-[minmax(280px,360px)_1fr] gap-5 lg:gap-7">
        {/* Left column: Controls & settings */}
        <div className="space-y-5 lg:space-y-6">
          <CollapsibleSection
            id="session-control"
            title={
              <span className="flex items-center space-x-2">
                <Cpu className="w-4 h-4 text-purple-400" />
                <span>Session Control</span>
              </span>
            }
            badge={sessionStatusBadge}
          >
            <div className="space-y-4">
              <div className="flex rounded-xl bg-zinc-950 p-1 border border-zinc-900">
                <button
                  disabled={isSessionActive}
                  onClick={() => setMode("browser")}
                  className={`touch-button flex-1 rounded-lg text-xs font-semibold font-mono transition-all duration-200 flex items-center justify-center space-x-1.5 ${
                    mode === "browser"
                      ? "bg-zinc-900 text-purple-400 border border-zinc-800 shadow"
                      : "text-zinc-500 hover:text-zinc-300 disabled:opacity-50"
                  }`}
                >
                  <Radio className="w-3.5 h-3.5" />
                  <span>BROWSER AUDIO</span>
                </button>
                <button
                  disabled={isSessionActive}
                  onClick={() => setMode("local")}
                  className={`touch-button flex-1 rounded-lg text-xs font-semibold font-mono transition-all duration-200 flex items-center justify-center space-x-1.5 ${
                    mode === "local"
                      ? "bg-zinc-900 text-purple-400 border border-zinc-800 shadow"
                      : "text-zinc-500 hover:text-zinc-300 disabled:opacity-50"
                  }`}
                >
                  <Cpu className="w-3.5 h-3.5" />
                  <span>LOCAL HARDWARE</span>
                </button>
              </div>

              {error && (
                <div className="bg-rose-950/40 border border-rose-900/50 rounded-xl p-3 text-xs text-rose-300 flex items-start gap-2">
                  <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                  <p className="leading-normal">{error}</p>
                </div>
              )}

              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => setOutputMuted((muted) => !muted)}
                  className={`touch-button rounded-xl border px-4 text-sm font-bold tracking-wide flex items-center justify-center space-x-2 transition-all ${
                    outputMuted
                      ? "border-amber-500/35 bg-amber-500/10 text-amber-100"
                      : "border-zinc-850 bg-zinc-950 text-zinc-300 hover:bg-zinc-900"
                  }`}
                >
                  {outputMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
                  <span>{outputMuted ? "MUTED" : "MUTE"}</span>
                </button>
                <button
                  type="button"
                  disabled={!isSessionActive}
                  onClick={handlePauseResume}
                  className="touch-button rounded-xl border border-zinc-850 bg-zinc-950 px-4 text-sm font-bold tracking-wide flex items-center justify-center space-x-2 text-zinc-300 transition-all hover:bg-zinc-900 disabled:opacity-40"
                >
                  {isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
                  <span>{isPaused ? "RESUME" : "PAUSE"}</span>
                </button>
              </div>

              <button
                type="button"
                role="switch"
                aria-checked={autoFollowupEnabled}
                disabled={isSessionActive || agentBackend === "hermes"}
                onClick={() => setAutoFollowupEnabled((enabled) => !enabled)}
                className="touch-button vokel-panel-subtle w-full rounded-2xl px-4 py-3 text-left text-xs text-zinc-400 transition-all hover:border-purple-500/30 disabled:opacity-55"
              >
                <span className="flex items-center justify-between gap-4">
                  <span>
                    <span className="flex items-center gap-1.5 font-bold text-zinc-300 font-mono uppercase">
                      <Timer className="w-3.5 h-3.5 text-purple-400" />
                      Auto Follow-up
                    </span>
                    <span className="block mt-1 leading-normal">
                      After silence, Vokel re-engages so the conversation does not stall.
                    </span>
                  </span>
                  <span className="flex shrink-0 flex-col items-end gap-1">
                    <span className="text-[10px] font-mono uppercase text-zinc-500">
                      {autoFollowupEnabled ? "On" : "Off"}
                    </span>
                    <span className="recall-switch" data-enabled={autoFollowupEnabled}>
                      <span className="recall-knob" />
                    </span>
                  </span>
                </span>
              </button>

              <div>
                <label className="block text-xs font-bold text-zinc-500 font-mono mb-1.5 uppercase">
                  Follow-up Timer
                </label>
                <select
                  disabled={isSessionActive || !autoFollowupEnabled || agentBackend === "hermes"}
                  value={autoFollowupSeconds}
                  onChange={(e) => setAutoFollowupSeconds(Number(e.target.value))}
                  className="vokel-field"
                >
                  <option value={5}>5 seconds</option>
                  <option value={8}>8 seconds</option>
                  <option value={12}>12 seconds</option>
                  <option value={15}>15 seconds</option>
                  <option value={20}>20 seconds</option>
                </select>
                <p className="mt-2 text-[11px] leading-relaxed text-zinc-500">
                  Default is 8s. Applies on the next session start.
                </p>
              </div>

              <div className="grid grid-cols-2 gap-4">
                {!isSessionActive ? (
                  <button
                    disabled={!isConnected}
                    onClick={handleStartSession}
                    className="touch-button col-span-2 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-550 hover:to-indigo-550 disabled:from-zinc-900 disabled:to-zinc-900 disabled:text-zinc-600 disabled:border-zinc-850 disabled:shadow-none px-4 rounded-xl text-sm font-bold tracking-wide flex items-center justify-center space-x-2 text-white shadow-lg shadow-purple-500/10 transition-all border border-purple-500/20"
                  >
                    <Mic className="w-4 h-4" />
                    <span>START SESSION</span>
                  </button>
                ) : (
                  <>
                    <button
                      onClick={handleStopSession}
                      className="touch-button bg-zinc-950 border border-zinc-850 hover:bg-zinc-900 px-4 rounded-xl text-sm font-bold tracking-wide flex items-center justify-center space-x-2 text-zinc-300 transition-all"
                    >
                      <MicOff className="w-4 h-4" />
                      <span>STOP</span>
                    </button>
                    <button
                      onClick={handleInterrupt}
                      className="touch-button bg-rose-950 hover:bg-rose-900 border border-rose-900/40 px-4 rounded-xl text-sm font-bold tracking-wide flex items-center justify-center space-x-2 text-rose-100 shadow-lg shadow-rose-950/20 transition-all"
                    >
                      <Flame className="w-4 h-4 animate-bounce" />
                      <span>BARGE IN</span>
                    </button>
                    <button
                      onClick={handleReset}
                      className="touch-button col-span-2 bg-zinc-950 border border-zinc-850 hover:bg-zinc-900 px-4 rounded-xl text-sm font-bold tracking-wide flex items-center justify-center space-x-2 text-zinc-300 transition-all"
                    >
                      <RotateCcw className="w-4 h-4" />
                      <span>RESET</span>
                    </button>
                  </>
                )}
              </div>
            </div>
          </CollapsibleSection>

          <CollapsibleSection
            id="connection"
            title={
              <span className="flex items-center space-x-2">
                <Bot className="w-4 h-4 text-purple-400" />
                <span>Connection</span>
              </span>
            }
          >
            <p className="text-[11px] leading-relaxed text-zinc-500 mb-3">
              Vokel handles voice. Pick where the model runs.
            </p>

            <div className="grid grid-cols-3 gap-1.5 rounded-xl bg-zinc-950 p-1 border border-zinc-900 mb-3">
              <button
                disabled={isSessionActive}
                onClick={() => selectLlmConnection("jan")}
                className={`touch-button rounded-lg py-2 px-1 text-[10px] font-bold font-mono tracking-wide transition-all ${
                  llmConnection === "jan"
                    ? "bg-zinc-900 text-purple-400 border border-zinc-800 shadow"
                    : "text-zinc-500 hover:text-zinc-300 disabled:opacity-50"
                }`}
              >
                JAN
              </button>
              <button
                disabled={isSessionActive}
                onClick={() => selectLlmConnection("lmstudio")}
                className={`touch-button rounded-lg py-2 px-1 text-[10px] font-bold font-mono tracking-wide transition-all ${
                  llmConnection === "lmstudio"
                    ? "bg-zinc-900 text-purple-400 border border-zinc-800 shadow"
                    : "text-zinc-500 hover:text-zinc-300 disabled:opacity-50"
                }`}
              >
                LM STUDIO
              </button>
              <button
                disabled={isSessionActive}
                onClick={() => selectLlmConnection("hermes")}
                className={`touch-button rounded-lg py-2 px-1 text-[10px] font-bold font-mono tracking-wide transition-all ${
                  llmConnection === "hermes"
                    ? "bg-zinc-900 text-purple-400 border border-zinc-800 shadow"
                    : "text-zinc-500 hover:text-zinc-300 disabled:opacity-50"
                }`}
              >
                HERMES
              </button>
            </div>

            {llmConnection === "jan" && (
              <p className="text-[10px] leading-relaxed text-zinc-600 font-mono mb-3">
                Preset: <span className="text-zinc-500">127.0.0.1:6767</span> · Gemma via{" "}
                <span className="text-zinc-500">~/.local/bin/jan-voice</span>
              </p>
            )}

            {llmConnection === "hermes" && (
              <div className="space-y-3">
                <div>
                  <label className="block text-xs font-bold text-zinc-500 font-mono mb-1.5 uppercase">
                    Hermes Gateway URL
                  </label>
                  <input
                    type="text"
                    disabled={isSessionActive}
                    value={hermesUrl}
                    onChange={(e) => setHermesUrl(e.target.value)}
                    className="vokel-field"
                  />
                </div>
                <div>
                  <label className="block text-xs font-bold text-zinc-500 font-mono mb-1.5 uppercase">
                    API Key
                  </label>
                  <input
                    type="password"
                    disabled={isSessionActive}
                    value={hermesApiKey}
                    onChange={(e) => setHermesApiKey(e.target.value)}
                    className="vokel-field"
                    placeholder="Matches API_SERVER_KEY in ~/.hermes/.env"
                  />
                </div>
                <div>
                  <label className="block text-xs font-bold text-zinc-500 font-mono mb-1.5 uppercase">
                    Conversation ID (optional)
                  </label>
                  <input
                    type="text"
                    disabled={isSessionActive}
                    value={hermesSessionId}
                    onChange={(e) => setHermesSessionId(e.target.value)}
                    className="vokel-field"
                    placeholder="Leave blank to start a new Hermes conversation"
                  />
                </div>
                <div>
                  <label className="block text-xs font-bold text-zinc-500 font-mono mb-1.5 uppercase">
                    Hermes Model Name
                  </label>
                  <input
                    type="text"
                    disabled={isSessionActive}
                    value={hermesModel}
                    onChange={(e) => setHermesModel(e.target.value)}
                    className="vokel-field"
                  />
                </div>
                
                <div className="pt-2">
                  <div className="flex items-center gap-3">
                    <button
                      type="button"
                      disabled={isSessionActive || !isConnected || probeStatus === "probing"}
                      onClick={handleProbeHermes}
                      className="touch-button rounded-xl border border-zinc-850 bg-zinc-950 px-4 py-2 text-xs font-bold tracking-wide text-zinc-300 transition-all hover:bg-zinc-900 disabled:opacity-40"
                    >
                      {probeStatus === "probing" ? "TESTING..." : "TEST CONNECTION"}
                    </button>
                    {probeStatus === "ok" && <span className="text-xs font-bold text-emerald-400 font-mono">OK</span>}
                    {probeStatus === "error" && <span className="text-xs font-bold text-rose-400 font-mono">FAILED</span>}
                  </div>
                </div>

                {activeHermesSessionId && (
                  <p className="text-[11px] font-mono text-purple-300/90">
                    Active conversation: {activeHermesSessionId}
                  </p>
                )}
              </div>
            )}
          </CollapsibleSection>

          <CollapsibleSection
            id="platform-capabilities"
            title={
              <span className="flex items-center space-x-2">
                <Wrench className="w-4 h-4 text-purple-400" />
                <span>Platform Capabilities</span>
              </span>
            }
          >
            <p className="text-[11px] leading-relaxed text-zinc-500">{platformCapabilitiesCopy}</p>
            <p
              className={`mt-2 text-[11px] leading-relaxed ${
                ffmpegAvailable ? "text-zinc-500" : "text-amber-400/90"
              }`}
            >
              Backing-track upload:{" "}
              {ffmpegAvailable ? "ready (ffmpeg found)" : "unavailable — install ffmpeg"}
            </p>
          </CollapsibleSection>

          <CollapsibleSection
            id="model-pipeline"
            title={
              <span className="flex items-center space-x-2">
                <Settings className="w-4 h-4 text-purple-400" />
                <span>Model & Pipeline Specs</span>
              </span>
            }
          >
            <div className="space-y-4">
              {llmConnection !== "hermes" && (
                <>
              <div>
                <label className="block text-xs font-bold text-zinc-500 font-mono mb-1.5 uppercase">
                  {llmEndpointLabel}
                </label>
                <input
                  type="text"
                  disabled={isSessionActive}
                  value={lmStudioUrl}
                  onChange={(e) => setLmStudioUrl(e.target.value)}
                  className="vokel-field"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-zinc-500 font-mono mb-1.5 uppercase">
                  Target Model Name
                </label>
                <input
                  type="text"
                  disabled={isSessionActive}
                  value={lmStudioModel}
                  onChange={(e) => setLmStudioModel(e.target.value)}
                  className="vokel-field"
                />
              </div>

              {llmConnection === "lmstudio" && (
              <div className="pt-1">
                <label className="flex items-center gap-2 text-xs font-mono text-zinc-400 mb-1.5">
                  <input
                    type="checkbox"
                    disabled={isSessionActive}
                    checked={useLmNativeMcp}
                    onChange={(e) => setUseLmNativeMcp(e.target.checked)}
                  />
                  Use native /api/v1/chat (MCP integrations from ~/.lmstudio/mcp.json)
                </label>
                {useLmNativeMcp && (
                  <input
                    type="text"
                    disabled={isSessionActive}
                    placeholder="mcp/playwright, mcp/fetch (comma separated)"
                    value={lmMcpIntegrations}
                    onChange={(e) => setLmMcpIntegrations(e.target.value)}
                    className="vokel-field text-xs"
                  />
                )}
              </div>
              )}
                </>
              )}

              <div>
                <label className="block text-xs font-bold text-zinc-500 font-mono mb-1.5 uppercase">
                  Local TTS Engine backend
                </label>
                <select
                  disabled={isSessionActive}
                  value={playbackBackend}
                  onChange={(e) => setPlaybackBackend(e.target.value)}
                  className="vokel-field"
                >
                  <option value="kokoro">Kokoro ONNX (Streaming synthesis)</option>
                  <option value="spd-say">Speech Dispatcher (spd-say CLI)</option>
                  <option value="console">Console Only (Silent telemetry)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-bold text-zinc-500 font-mono mb-1.5 uppercase">
                  Kokoro Voice
                </label>
                <div className="flex gap-2">
                  <select
                    disabled={isSessionActive || playbackBackend !== "kokoro"}
                    value={voice}
                    onChange={(e) => setVoice(e.target.value)}
                    className="vokel-field"
                  >
                    {kokoroVoices.map((voiceName) => (
                      <option key={voiceName} value={voiceName}>
                        {voiceName}
                      </option>
                    ))}
                  </select>
                  <button
                    type="button"
                    disabled={!isConnected || isSessionActive || playbackBackend !== "kokoro"}
                    onClick={() => handlePreviewVoice(voice)}
                    className="touch-button shrink-0 rounded-xl border border-purple-500/30 bg-purple-600/10 px-3 text-purple-100 transition hover:bg-purple-600/20 disabled:opacity-40"
                    aria-label={`Preview selected voice ${voice}`}
                    title={`Preview selected voice ${voice}`}
                  >
                    <Play className={`h-4 w-4 ${previewingVoice === voice ? "animate-pulse" : ""}`} />
                  </button>
                </div>
                <div className="mt-3 flex items-center justify-between text-[10px] font-mono uppercase tracking-wide text-zinc-500">
                  <span>Audition Voices</span>
                  <span>{previewingVoice ? `Playing ${previewingVoice}` : "Click play to sample"}</span>
                </div>
                <div className="mt-2 max-h-44 overflow-y-auto rounded-2xl border border-zinc-850 bg-zinc-950/50 p-2">
                  <div className="grid grid-cols-1 gap-1.5">
                    {kokoroVoices.map((voiceName) => {
                      const isSelected = voiceName === voice;
                      const isPreviewing = previewingVoice === voiceName;
                      return (
                        <div
                          key={voiceName}
                          className={`flex items-center justify-between gap-2 rounded-xl border px-2.5 py-2 text-xs transition ${
                            isSelected
                              ? "border-purple-500/35 bg-purple-500/10 text-purple-100"
                              : "border-zinc-850 bg-zinc-950/50 text-zinc-400"
                          }`}
                        >
                          <button
                            type="button"
                            disabled={isSessionActive || playbackBackend !== "kokoro"}
                            onClick={() => setVoice(voiceName)}
                            className="min-h-0 flex-1 text-left font-mono disabled:opacity-50"
                          >
                            {voiceName}
                          </button>
                          <button
                            type="button"
                            disabled={!isConnected || isSessionActive || playbackBackend !== "kokoro"}
                            onClick={() => handlePreviewVoice(voiceName)}
                            className="min-h-0 inline-flex h-8 w-8 items-center justify-center rounded-lg border border-zinc-800 bg-zinc-900 text-zinc-300 transition hover:border-purple-500/40 hover:text-purple-200 disabled:opacity-40"
                            aria-label={`Preview ${voiceName}`}
                            title={`Preview ${voiceName}`}
                          >
                            <Play className={`h-3.5 w-3.5 ${isPreviewing ? "animate-pulse text-purple-300" : ""}`} />
                          </button>
                        </div>
                      );
                    })}
                  </div>
                </div>
                <p className="mt-2 text-[11px] leading-relaxed text-zinc-500">
                  Preview uses a short local Kokoro sample and does not touch transcript or memory.
                </p>
              </div>

              <div>
                <label className="block text-xs font-bold text-zinc-500 font-mono mb-1.5 uppercase">
                  Voice Effects
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {[
                    { label: "Clear", speed: 0.9 },
                    { label: "Natural", speed: 1 },
                    { label: "Brisk", speed: 1.12 },
                  ].map((profile) => (
                    <button
                      key={profile.label}
                      type="button"
                      disabled={isSessionActive || playbackBackend !== "kokoro"}
                      onClick={() => setTtsSpeed(profile.speed)}
                      className={`touch-button rounded-xl border px-3 text-xs font-bold uppercase tracking-wide transition ${
                        Math.abs(ttsSpeed - profile.speed) < 0.01
                          ? "border-purple-500/35 bg-purple-500/10 text-purple-100"
                          : "border-zinc-850 bg-zinc-950 text-zinc-400 hover:bg-zinc-900"
                      } disabled:opacity-40`}
                    >
                      {profile.label}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-xs font-bold text-zinc-500 font-mono mb-1.5 uppercase">
                  Speech Speed: {ttsSpeed.toFixed(2)}x
                </label>
                <input
                  type="range"
                  min="0.75"
                  max="1.25"
                  step="0.05"
                  disabled={isSessionActive || playbackBackend !== "kokoro"}
                  value={ttsSpeed}
                  onChange={(e) => setTtsSpeed(Number(e.target.value))}
                  className="w-full accent-purple-500"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-zinc-500 font-mono mb-1.5 uppercase">
                  {musicalMode
                    ? "What are we rapping about?"
                    : "What are we going to talk about?"}
                </label>
                <input
                  type="text"
                  maxLength={200}
                  value={sessionContext}
                  onChange={(e) => setSessionContext(e.target.value)}
                  placeholder="optional — grounds the session topic"
                  className="w-full rounded-xl border border-zinc-850 bg-zinc-950 px-3 py-2 text-xs text-zinc-200 placeholder:text-zinc-600 focus:border-purple-500/40 focus:outline-none"
                />
                {isSessionActive && (
                  <p className="mt-1.5 text-[10px] font-mono text-zinc-500">
                    applies on next session start
                  </p>
                )}
              </div>

              <button
                type="button"
                role="switch"
                aria-checked={musicalMode}
                disabled={
                  isSessionActive ||
                  mode !== "local" ||
                  (playbackBackend !== "kokoro" && playbackBackend !== "spd-say")
                }
                onClick={() => setMusicalMode((enabled) => !enabled)}
                className="touch-button vokel-panel-subtle w-full rounded-2xl px-4 py-3 text-left text-xs text-zinc-400 transition-all hover:border-purple-500/30 disabled:opacity-55"
              >
                <span className="flex items-center justify-between gap-4">
                  <span>
                    <span className="flex items-center gap-1.5 font-bold text-zinc-300 font-mono uppercase">
                      <Radio className="w-3.5 h-3.5 text-purple-400" />
                      Musical Mode
                    </span>
                    <span className="block mt-1 leading-normal">
                      Quantize TTS to the beat grid with a low backing loop. Local Kokoro or
                      spd-say only.
                    </span>
                  </span>
                  <span className="flex shrink-0 flex-col items-end gap-1">
                    <span className="text-[10px] font-mono uppercase text-zinc-500">
                      {musicalMode ? "On" : "Off"}
                    </span>
                    <span className="recall-switch" data-enabled={musicalMode}>
                      <span className="recall-knob" />
                    </span>
                  </span>
                </span>
              </button>

              {musicalMode && (
                <>
                  <div>
                    <label className="block text-xs font-bold text-zinc-500 font-mono mb-1.5 uppercase">
                      Backing Style
                    </label>
                    <div className="grid grid-cols-2 gap-2">
                      {(["beat", "metronome"] as const).map((style) => (
                        <button
                          key={style}
                          type="button"
                          disabled={
                            isSessionActive ||
                            mode !== "local" ||
                            (playbackBackend !== "kokoro" && playbackBackend !== "spd-say")
                          }
                          onClick={() => setMusicalStyle(style)}
                          className={`touch-button rounded-xl border px-3 py-2 text-xs font-mono uppercase transition ${
                            musicalStyle === style
                              ? "border-purple-500/50 bg-purple-500/10 text-purple-200"
                              : "border-zinc-850 bg-zinc-950 text-zinc-400 hover:bg-zinc-900"
                          }`}
                        >
                          {style === "beat" ? "Beat Loop" : "Metronome"}
                        </button>
                      ))}
                    </div>
                    {musicalTrackInfo && (
                      <p className="mt-2 flex flex-wrap items-center gap-2 text-[10px] font-mono text-zinc-500">
                        <span>
                          Uploaded track overrides {musicalStyle === "beat" ? "Beat Loop" : "Metronome"}.
                        </span>
                        <button
                          type="button"
                          onClick={clearMusicalTrack}
                          className="text-purple-300 underline-offset-2 hover:text-purple-200 hover:underline"
                        >
                          Clear track
                        </button>
                      </p>
                    )}
                  </div>

                  <div>
                    <div className="mb-1.5 flex items-center justify-between gap-3">
                      <label className="block text-xs font-bold text-zinc-500 font-mono uppercase">
                        Tempo: {musicalBpm} BPM
                      </label>
                      <button
                        type="button"
                        disabled={
                          isSessionActive ||
                          mode !== "local" ||
                          (playbackBackend !== "kokoro" && playbackBackend !== "spd-say")
                        }
                        onClick={handleTapTempo}
                        className="touch-button rounded-xl border border-zinc-850 bg-zinc-950 px-3 py-1.5 text-[10px] font-mono uppercase tracking-wide text-zinc-300 transition hover:bg-zinc-900 disabled:opacity-55"
                      >
                        Tap
                      </button>
                    </div>
                    <input
                      type="range"
                      min="60"
                      max="160"
                      step="5"
                      disabled={
                        isSessionActive ||
                        mode !== "local" ||
                        (playbackBackend !== "kokoro" && playbackBackend !== "spd-say")
                      }
                      value={musicalBpm}
                      onChange={(e) => setMusicalBpm(Number(e.target.value))}
                      className="w-full accent-purple-500"
                    />
                  </div>

                  <div className={!ffmpegAvailable ? "opacity-50" : undefined}>
                    <label className="block text-xs font-bold text-zinc-500 font-mono mb-1.5 uppercase">
                      Load Track
                    </label>
                    <input
                      type="file"
                      accept="audio/*,.mp3,.wav,.flac,.aac,.ogg,.m4a"
                      disabled={musicalTrackUploading || !isConnected || !ffmpegAvailable}
                      title={
                        ffmpegAvailable
                          ? undefined
                          : "Install ffmpeg to enable backing-track upload"
                      }
                      onChange={(event) => {
                        const file = event.target.files?.[0];
                        if (file) {
                          void uploadMusicalTrack(file);
                        }
                        event.currentTarget.value = "";
                      }}
                      className="block w-full text-xs text-zinc-400 file:mr-3 file:rounded-lg file:border-0 file:bg-purple-500/15 file:px-3 file:py-2 file:font-mono file:text-[10px] file:uppercase file:text-purple-200 disabled:cursor-not-allowed"
                    />
                    {musicalTrackInfo && (
                      <p className="mt-1.5 text-[10px] font-mono text-zinc-500">
                        Loaded {musicalTrackInfo.durationSeconds.toFixed(1)}s ·{" "}
                        {musicalTrackInfo.samples.toLocaleString()} samples
                      </p>
                    )}
                  </div>
                </>
              )}

              <div>
                <label className="block text-xs font-bold text-zinc-500 font-mono mb-1.5 uppercase">
                  Browser Output Volume: {outputMuted ? "Muted" : `${Math.round(outputVolume * 100)}%`}
                </label>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setOutputMuted((muted) => !muted)}
                    className="touch-button shrink-0 rounded-xl border border-zinc-850 bg-zinc-950 px-3 text-zinc-300 transition hover:bg-zinc-900"
                    aria-label={outputMuted ? "Unmute output" : "Mute output"}
                    title={outputMuted ? "Unmute output" : "Mute output"}
                  >
                    {outputMuted ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
                  </button>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={outputVolume}
                    onChange={(e) => {
                      setOutputVolume(Number(e.target.value));
                      setOutputMuted(false);
                    }}
                    className="w-full accent-purple-500"
                  />
                </div>
              </div>

              <button
                type="button"
                role="switch"
                aria-checked={memoryEnabled}
                disabled={isSessionActive || agentBackend === "hermes"}
                onClick={() => setMemoryEnabled((enabled) => !enabled)}
                className="touch-button vokel-panel-subtle w-full rounded-2xl px-4 py-3 text-left text-xs text-zinc-400 transition-all hover:border-purple-500/30 disabled:opacity-55"
              >
                <span className="flex items-center justify-between gap-4">
                  <span>
                    <span className="flex items-center gap-1.5 font-bold text-zinc-300 font-mono uppercase">
                      <Brain className="w-3.5 h-3.5 text-purple-400" />
                      Conversation Recall
                    </span>
                    <span className="block mt-1 leading-normal">
                      Off by default. When enabled, Vokel can use saved local notes from this machine.
                    </span>
                  </span>
                  <span className="flex shrink-0 flex-col items-end gap-1">
                    <span className="text-[10px] font-mono uppercase text-zinc-500">
                      {memoryEnabled ? "On" : "Off"}
                    </span>
                    <span className="recall-switch" data-enabled={memoryEnabled}>
                      <span className="recall-knob" />
                    </span>
                  </span>
                </span>
              </button>
            </div>
          </CollapsibleSection>

          <CollapsibleSection
            id="memory-notes"
            title={
              <span className="flex items-center space-x-2">
                <Brain className="w-4 h-4 text-purple-400" />
                <span>Memory Notes</span>
              </span>
            }
          >
            <div className="space-y-3">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={memoryDraft}
                  onChange={(e) => setMemoryDraft(e.target.value)}
                  className="vokel-field"
                  placeholder="Save a local note"
                />
                <button
                  onClick={handleSaveMemory}
                  disabled={!isConnected || !memoryDraft.trim()}
                  className="touch-button shrink-0 bg-zinc-950 border border-zinc-850 hover:bg-zinc-900 disabled:opacity-50 px-3 rounded-xl text-zinc-300"
                  aria-label="Save memory note"
                  title="Save memory note"
                >
                  <Save className="w-4 h-4" />
                </button>
              </div>

              <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
                {memoryFacts.length === 0 ? (
                  <p className="text-xs text-zinc-500 leading-relaxed">
                    No saved notes yet.
                  </p>
                ) : (
                  memoryFacts.map((fact) => (
                    <div key={fact.id} className="memory-row">
                      <input
                        type="checkbox"
                        checked={selectedMemoryIds.has(fact.id)}
                        onChange={() => toggleMemorySelection(fact.id)}
                        aria-label="Include note in Markdown export"
                      />
                      <input
                        type="text"
                        value={fact.text}
                        onChange={(e) => {
                          const text = e.target.value;
                          setMemoryFacts((facts) =>
                            facts.map((item) =>
                              item.id === fact.id ? { ...item, text } : item
                            )
                          );
                        }}
                        onBlur={(e) => handleUpdateMemory(fact.id, e.target.value)}
                        className="memory-edit"
                      />
                      <button
                        onClick={() => handleDeleteMemory(fact.id)}
                        className="memory-icon-button"
                        aria-label="Delete memory note"
                        title="Delete memory note"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  ))
                )}
              </div>

              <button
                onClick={handleExportMarkdown}
                className="touch-button w-full bg-zinc-950 border border-zinc-850 hover:bg-zinc-900 px-4 rounded-xl text-sm font-bold tracking-wide flex items-center justify-center space-x-2 text-zinc-300 transition-all"
              >
                <Download className="w-4 h-4" />
                <span>EXPORT .MD</span>
              </button>
            </div>
          </CollapsibleSection>
        </div>

        {/* Right column: Spectrums and Transcript (2 columns wide) */}
        <div className="space-y-5 lg:space-y-6 min-w-0">
          {/* Trust State Surface (Slice B) */}
          <details className="vokel-panel rounded-3xl p-5 sm:p-6" open>
            <summary className="cursor-pointer list-none">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <div className="text-[10px] font-mono uppercase tracking-wider text-zinc-500">
                    Trust State
                  </div>
                  <div className="mt-1 text-xs font-mono text-zinc-300">{trustSummary}</div>
                </div>
                <span className="text-[10px] font-mono uppercase text-zinc-500">Details</span>
              </div>
            </summary>

            <div className="mt-4 grid gap-2 sm:grid-cols-2">
              {[
                { label: "Active Connection", value: activeConnectionLabel },
                { label: "Route", value: routeLabel },
                { label: "Voice", value: voiceLabel },
                { label: "Tools", value: toolsOwnerLabel },
                { label: "Privacy", value: privacyLabel },
                { label: "Interrupt", value: interruptLabel },
                { label: "Consent", value: consentLabel },
              ].map((item) => (
                <div key={item.label} className="vokel-panel-subtle rounded-2xl px-3 py-2">
                  <div className="text-[10px] font-mono uppercase text-zinc-500">{item.label}</div>
                  <div className="mt-1 text-xs font-mono text-zinc-300">{item.value}</div>
                </div>
              ))}
            </div>
          </details>

          {/* Live Voice Presence — Command #2 */}
          <div className="vokel-panel rounded-3xl p-8 flex flex-col items-center">
            <PersonaSelector current={currentPersona} onChange={handlePersonaChange} />

            <div className="mt-6 mb-2">
              <VoiceOrb state={orbState} amplitude={orbAmplitude} />
            </div>

            <div className="text-center mt-2">
              <div className="text-[10px] font-mono tracking-[4px] text-white/50">PRESENCE</div>
              <div className="text-xl font-semibold tracking-tight mt-0.5 capitalize">{currentPersona}</div>
              <div className="text-[11px] text-white/50 mt-0.5">Live voice, reactive to every turn</div>
            </div>
          </div>

          {musicalMode && isSessionActive && (
            <CollapsibleSection
              id="beat-indicator"
              title={
                <span className="flex items-center space-x-2">
                  <Radio className="w-4 h-4 text-purple-400" />
                  <span>Musical Grid</span>
                </span>
              }
              badge={beatCollapsedBadge}
            >
              <BeatIndicator
                musicalMode={musicalMode}
                isSessionActive={isSessionActive}
                beatPulse={beatPulse}
                musicalLevel={musicalLevel}
                onMusicalLevelChange={handleMusicalLevelChange}
                onMusicalNudge={sendMusicalNudge}
              />
            </CollapsibleSection>
          )}

          <CollapsibleSection
            id="waveform-visualizer"
            title={
              <span className="flex items-center space-x-2">
                <Volume2 className="w-4 h-4 text-purple-400" />
                <span>Audio Pipeline Spectrum</span>
              </span>
            }
            badge={
              <span className="text-[10px] font-mono uppercase tracking-wide text-zinc-500">
                {status}
              </span>
            }
          >
            <WaveformVisualizer status={status} volume={isStreaming ? micVolume : 0} />
          </CollapsibleSection>

          <CollapsibleSection
            id="live-vision"
            title={
              <span className="flex items-center space-x-2">
                <Eye className="w-4 h-4 text-purple-400" />
                <span>Local Vision Window</span>
              </span>
            }
            badge={visionCollapsedBadge}
          >
            <LiveVisionPanel
              visionUrl={llmConnection === "hermes" ? hermesUrl : lmStudioUrl}
              visionModel={llmConnection === "hermes" ? hermesModel : lmStudioModel}
              visionApiKey={llmConnection === "hermes" ? hermesApiKey : ""}
              visionBackend={llmConnection === "hermes" ? "hermes" : "local"}
              voiceContextEnabled={visionVoiceEnabled}
              voiceContextFrame={voiceVisionFrame}
              sessionStatus={status}
              liveControlSignal={liveVisionControl}
              registerFrameGrabber={(grab) => {
                capturePreviewFrameRef.current = grab;
              }}
              onVoiceContextEnabledChange={handleVisionVoiceEnabledChange}
              onSelectedDeviceChange={handleVisionDeviceChange}
              onStatusChange={setVisionPanelStatus}
            />
          </CollapsibleSection>

          {/* Near-transcript session controls (UI-1) */}
          <div className="vokel-panel rounded-2xl p-3 sm:p-4 sticky top-20 z-20">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="min-w-0">
                <div className="text-[10px] font-mono uppercase tracking-wider text-zinc-500">
                  Session cockpit
                </div>
                <div className="mt-1 text-xs font-mono text-zinc-300 truncate">
                  {trustSummary} • Output {outputMuted ? "muted" : "on"}
                </div>
              </div>
              <span
                className={`rounded-full border px-2 py-1 text-[10px] font-bold uppercase tracking-wide ${
                  sessionStateLabel === "paused"
                    ? "border-amber-500/30 bg-amber-500/10 text-amber-300"
                    : sessionStateLabel === "listening"
                      ? "border-emerald-500/25 bg-emerald-500/10 text-emerald-300"
                      : sessionStateLabel === "capturing_vision"
                        ? "border-rose-500/30 bg-rose-500/10 text-rose-200"
                      : sessionStateLabel === "generating" || sessionStateLabel === "speaking"
                        ? "border-blue-500/25 bg-blue-500/10 text-blue-300"
                        : "border-zinc-800 bg-zinc-950 text-zinc-500"
                }`}
              >
                {sessionStateLabel}
              </span>
            </div>

            {activeActionLabel && (
              <div className="mt-3 overflow-hidden rounded-xl border border-blue-500/20 bg-blue-500/10">
                <div className="flex items-center justify-between gap-3 px-3 py-2">
                  <div className="flex min-w-0 items-center gap-2 text-xs font-mono text-blue-100">
                    <Loader2 className="h-4 w-4 shrink-0 animate-spin text-blue-300" />
                    <span className="truncate">{activeActionLabel}</span>
                  </div>
                  <span className="h-2 w-2 shrink-0 rounded-full bg-blue-300 shadow-[0_0_18px_rgba(147,197,253,0.85)]" />
                </div>
                <div className="h-0.5 w-full overflow-hidden bg-blue-950/60">
                  <div className="h-full w-1/2 animate-pulse bg-gradient-to-r from-blue-400 via-cyan-300 to-blue-400" />
                </div>
              </div>
            )}

            <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-5">
              {!isSessionActive ? (
                <button
                  disabled={!isConnected}
                  onClick={handleStartSession}
                  className="touch-button col-span-2 sm:col-span-2 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-550 hover:to-indigo-550 disabled:from-zinc-900 disabled:to-zinc-900 disabled:text-zinc-600 disabled:border-zinc-850 disabled:shadow-none px-3 rounded-xl text-xs font-bold tracking-wide flex items-center justify-center space-x-2 text-white shadow-lg shadow-purple-500/10 transition-all border border-purple-500/20"
                >
                  <Mic className="w-4 h-4" />
                  <span>START</span>
                </button>
              ) : (
                <>
                  <button
                    onClick={handleStopSession}
                    className="touch-button bg-zinc-950 border border-zinc-850 hover:bg-zinc-900 px-3 rounded-xl text-xs font-bold tracking-wide flex items-center justify-center space-x-1.5 text-zinc-300 transition-all"
                  >
                    <MicOff className="w-4 h-4" />
                    <span>STOP</span>
                  </button>
                  <button
                    onClick={handlePauseResume}
                    className="touch-button bg-zinc-950 border border-zinc-850 hover:bg-zinc-900 px-3 rounded-xl text-xs font-bold tracking-wide flex items-center justify-center space-x-1.5 text-zinc-300 transition-all"
                  >
                    {isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
                    <span>{isPaused ? "RESUME" : "PAUSE"}</span>
                  </button>
                  <button
                    onClick={handleInterrupt}
                    className="touch-button bg-rose-950 hover:bg-rose-900 border border-rose-900/40 px-3 rounded-xl text-xs font-bold tracking-wide flex items-center justify-center space-x-1.5 text-rose-100 shadow-lg shadow-rose-950/20 transition-all"
                  >
                    <Flame className="w-4 h-4" />
                    <span>BARGE IN</span>
                  </button>
                </>
              )}
              <button
                type="button"
                onClick={() => setOutputMuted((muted) => !muted)}
                className={`touch-button rounded-xl border px-3 text-xs font-bold tracking-wide flex items-center justify-center space-x-1.5 transition-all ${
                  outputMuted
                    ? "border-amber-500/35 bg-amber-500/10 text-amber-100"
                    : "border-zinc-850 bg-zinc-950 text-zinc-300 hover:bg-zinc-900"
                }`}
              >
                {outputMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
                <span>{outputMuted ? "UNMUTE" : "MUTE"}</span>
              </button>
              <button
                disabled={!isSessionActive}
                onClick={handleReset}
                className="touch-button rounded-xl border border-zinc-850 bg-zinc-950 hover:bg-zinc-900 px-3 text-xs font-bold tracking-wide flex items-center justify-center space-x-1.5 text-zinc-300 transition-all disabled:opacity-40"
              >
                <RotateCcw className="w-4 h-4" />
                <span>RESET</span>
              </button>
            </div>
          </div>

          {/* Chat transcript stream */}
          <TranscriptStream
            messages={messages}
            status={status}
            activeConnection={llmConnection === "hermes" ? "hermes" : llmConnection === "jan" ? "jan" : "lm_studio"}
            activeRoute={agentBackend === "hermes" ? "external" : "local"}
            activePrivacy={agentBackend === "hermes" ? "external_active" : "local"}
            activeAction={activeActionLabel}
          />

          <WorkspaceTabs
            activeTab={workspaceTab}
            onTabChange={setWorkspaceTab}
            voice={voice}
            ttsSpeed={ttsSpeed}
            outputMuted={outputMuted}
            outputVolume={outputVolume}
            previousChats={previousChats}
            onRestoreChat={restorePreviousChat}
            onClearPreviousChats={clearPreviousChats}
            backend={activeConnectionLabel}
            isSessionActive={isSessionActive}
            activeSessionId={activeHermesSessionId}
            events={agentEvents}
            executeState={executeState}
            onArmExecute={handleArmExecute}
            onCancelExecute={handleCancelExecute}
            onConfirmExecute={handleConfirmExecute}
          />
        </div>

        {/* Bottom row: latency budget scorecard */}
        <div className="xl:col-span-2">
          <LatencyScoreboard metrics={metrics} />
        </div>
      </main>
    </div>
  );
}

export default App;
