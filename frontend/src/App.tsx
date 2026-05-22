import { useState, useEffect, useRef } from "react";
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
} from "lucide-react";
import { LatencyScoreboard } from "./components/LatencyScoreboard";
import { WaveformVisualizer } from "./components/WaveformVisualizer";
import { TranscriptStream } from "./components/TranscriptStream";
import type { Message } from "./components/TranscriptStream";
import { useAudioStreamer } from "./hooks/useAudioStreamer";

type Mode = "local" | "browser";
type Status = "idle" | "listening" | "generating" | "speaking";

function App() {
  const [mode, setMode] = useState<Mode>("browser");
  const [isConnected, setIsConnected] = useState(false);
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [status, setStatus] = useState<Status>("idle");
  const [messages, setMessages] = useState<Message[]>([]);
  const [metrics, setMetrics] = useState<Record<string, number>>({});
  const [error, setError] = useState<string | null>(null);

  // Settings
  const [lmStudioUrl, setLmStudioUrl] = useState("http://localhost:1234/v1/chat/completions");
  const [lmStudioModel, setLmStudioModel] = useState("gemma-4-e4b-it-ultra-uncensored-heretic");
  const [playbackBackend, setPlaybackBackend] = useState("kokoro");
  const [memoryEnabled, setMemoryEnabled] = useState(false);

  const socketRef = useRef<WebSocket | null>(null);

  // Hook for audio streaming and local queue scheduling in browser mode
  const { startStreaming, stopStreaming, isStreaming, micVolume } = useAudioStreamer();

  // Close websocket on unmount
  useEffect(() => {
    return () => {
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
      stopStreaming();
    };

    socket.onerror = () => {
      setError("WebSocket connection failed. Ensure the Voyce server is running with --web.");
    };

    socket.onmessage = async (event) => {
      if (typeof event.data === "string") {
        // Handle JSON messages from server
        const data = JSON.parse(event.data);

        switch (data.type) {
          case "session_started":
            setIsSessionActive(true);
            setStatus("listening");
            if (data.mode === "browser") {
              startStreaming(socket);
            }
            break;

          case "session_stopped":
            setIsSessionActive(false);
            setStatus("idle");
            stopStreaming();
            break;

          case "status":
            setStatus(data.status);
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
            break;

          case "telemetry":
            // Log or stream markers directly
            break;

          case "summary":
            setMetrics(data.metrics);
            break;

          case "error":
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
    setError(null);

    socketRef.current.send(
      JSON.stringify({
        type: "start_session",
        mode,
        url: lmStudioUrl,
        model: lmStudioModel,
        playback: playbackBackend,
        memory: memoryEnabled,
      })
    );
  };

  const handleStopSession = () => {
    if (!socketRef.current || !isConnected) return;

    socketRef.current.send(
      JSON.stringify({
        type: "stop_session",
      })
    );
  };

  const handleInterrupt = () => {
    if (!socketRef.current || !isConnected) return;

    socketRef.current.send(
      JSON.stringify({
        type: "interrupt",
      })
    );
  };

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
                Voyce
              </h1>
              <p className="text-[10px] text-zinc-500 font-mono tracking-wider mt-0.5">
                REAL-TIME VOICE LOOP
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

      {/* Main Content */}
      <main className="safe-container flex-1 py-6 sm:py-8 grid grid-cols-1 xl:grid-cols-[minmax(280px,360px)_1fr] gap-5 lg:gap-7">
        {/* Left column: Controls & settings */}
        <div className="space-y-5 lg:space-y-6">
          {/* Active Session Status Card */}
          <div className="voyce-panel rounded-3xl p-5 sm:p-6">
            <h2 className="text-sm font-bold text-zinc-400 tracking-wider font-mono uppercase mb-4 flex items-center space-x-2">
              <Cpu className="w-4 h-4 text-purple-400" />
              <span>Session Control</span>
            </h2>

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
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Model Configuration / Settings */}
          <div className="voyce-panel rounded-3xl p-5 sm:p-6">
            <h2 className="text-sm font-bold text-zinc-400 tracking-wider font-mono uppercase mb-4 flex items-center space-x-2">
              <Settings className="w-4 h-4 text-purple-400" />
              <span>Model & Pipeline Specs</span>
            </h2>

            <div className="space-y-4">
              <div>
                <label className="block text-xs font-bold text-zinc-500 font-mono mb-1.5 uppercase">
                  LM Studio Client Endpoint
                </label>
                <input
                  type="text"
                  disabled={isSessionActive}
                  value={lmStudioUrl}
                  onChange={(e) => setLmStudioUrl(e.target.value)}
                  className="voyce-field"
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
                  className="voyce-field"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-zinc-500 font-mono mb-1.5 uppercase">
                  Local TTS Engine backend
                </label>
                <select
                  disabled={isSessionActive}
                  value={playbackBackend}
                  onChange={(e) => setPlaybackBackend(e.target.value)}
                  className="voyce-field"
                >
                  <option value="kokoro">Kokoro ONNX (Streaming synthesis)</option>
                  <option value="spd-say">Speech Dispatcher (spd-say CLI)</option>
                  <option value="console">Console Only (Silent telemetry)</option>
                </select>
              </div>

              <button
                type="button"
                role="switch"
                aria-checked={memoryEnabled}
                disabled={isSessionActive}
                onClick={() => setMemoryEnabled((enabled) => !enabled)}
                className="touch-button voyce-panel-subtle w-full rounded-2xl px-4 py-3 text-left text-xs text-zinc-400 transition-all hover:border-purple-500/30 disabled:opacity-55"
              >
                <span className="flex items-center justify-between gap-4">
                  <span>
                    <span className="flex items-center gap-1.5 font-bold text-zinc-300 font-mono uppercase">
                      <Brain className="w-3.5 h-3.5 text-purple-400" />
                      Conversation Recall
                    </span>
                    <span className="block mt-1 leading-normal">
                      Off by default. When enabled, Voyce can use saved local notes from this machine.
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
          </div>
        </div>

        {/* Right column: Spectrums and Transcript (2 columns wide) */}
        <div className="space-y-5 lg:space-y-6 min-w-0">
          {/* Waveform visualizer */}
          <WaveformVisualizer status={status} volume={isStreaming ? micVolume : 0} />

          {/* Chat transcript stream */}
          <TranscriptStream messages={messages} status={status} />
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
