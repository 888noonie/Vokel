import { useCallback, useEffect, useRef, useState } from "react";
import { Camera, Eye, Loader2, RefreshCw, ShieldCheck, Square } from "lucide-react";

interface CameraDevice {
  path: string;
  name: string;
}

export interface VisionFrame {
  device: string;
  description: string;
  image_data_url: string;
  capture_seconds: number;
  inference_seconds: number;
}

interface CameraDiscovery {
  cameras: CameraDevice[];
  default_device: string;
}

interface LiveVisionPanelProps {
  lmStudioUrl: string;
  lmStudioModel: string;
  voiceContextEnabled: boolean;
  voiceContextFrame: VisionFrame | null;
  onVoiceContextEnabledChange: (enabled: boolean) => void;
  onSelectedDeviceChange: (device: string) => void;
}

const defaultPrompt = "Describe only what is visible in this image in one short sentence.";

export function LiveVisionPanel({
  lmStudioUrl,
  lmStudioModel,
  voiceContextEnabled,
  voiceContextFrame,
  onVoiceContextEnabledChange,
  onSelectedDeviceChange,
}: LiveVisionPanelProps) {
  const [cameras, setCameras] = useState<CameraDevice[]>([]);
  const [selectedDevice, setSelectedDevice] = useState("");
  const [prompt, setPrompt] = useState(defaultPrompt);
  const [latestFrame, setLatestFrame] = useState<VisionFrame | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isLive, setIsLive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestInFlightRef = useRef(false);
  const liveRequestedRef = useRef(false);

  const loadCameras = useCallback(async () => {
    setError(null);
    try {
      const response = await fetch("/api/vision/cameras");
      if (!response.ok) throw new Error(`Camera discovery failed: HTTP ${response.status}`);
      const data = (await response.json()) as CameraDiscovery;
      setCameras(data.cameras);
      setSelectedDevice((current) => {
        if (current && data.cameras.some((camera) => camera.path === current)) return current;
        return data.default_device;
      });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Camera discovery failed.");
    }
  }, []);

  useEffect(() => {
    const timeout = window.setTimeout(() => void loadCameras(), 0);
    return () => window.clearTimeout(timeout);
  }, [loadCameras]);

  useEffect(() => {
    if (!voiceContextFrame) return;
    const timeout = window.setTimeout(() => setLatestFrame(voiceContextFrame), 0);
    return () => window.clearTimeout(timeout);
  }, [voiceContextFrame]);

  useEffect(() => {
    if (selectedDevice) onSelectedDeviceChange(selectedDevice);
  }, [onSelectedDeviceChange, selectedDevice]);

  const analyzeOnce = useCallback(async () => {
    if (!selectedDevice || requestInFlightRef.current) return false;
    requestInFlightRef.current = true;
    setIsAnalyzing(true);
    setError(null);
    try {
      const response = await fetch("/api/vision/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          device: selectedDevice,
          url: lmStudioUrl,
          model: lmStudioModel,
          prompt,
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(String(data.detail ?? `Vision analysis failed: HTTP ${response.status}`));
      setLatestFrame(data as VisionFrame);
      return true;
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Vision analysis failed.");
      return false;
    } finally {
      requestInFlightRef.current = false;
      setIsAnalyzing(false);
    }
  }, [lmStudioModel, lmStudioUrl, prompt, selectedDevice]);

  useEffect(() => {
    if (!isLive) return;
    let cancelled = false;

    const run = async () => {
      while (!cancelled && liveRequestedRef.current) {
        const succeeded = await analyzeOnce();
        if (!succeeded) {
          liveRequestedRef.current = false;
          setIsLive(false);
          break;
        }
        await new Promise((resolve) => window.setTimeout(resolve, 1_000));
      }
    };

    void run();
    return () => {
      cancelled = true;
    };
  }, [analyzeOnce, isLive]);

  const startLive = () => {
    liveRequestedRef.current = true;
    setIsLive(true);
  };

  const stopLive = () => {
    liveRequestedRef.current = false;
    setIsLive(false);
  };

  return (
    <section className="vokel-panel overflow-hidden rounded-3xl">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 px-5 py-4 sm:px-6">
        <div>
          <div className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-zinc-300 font-mono">
            <Eye className="h-4 w-4 text-purple-400" />
            <span>Local Vision Window</span>
          </div>
          <p className="mt-1 text-[11px] leading-relaxed text-zinc-500">
            Explicit camera capture to local LM Studio. Frames are discarded after analysis.
          </p>
        </div>
        <span
          className={`rounded-full border px-2.5 py-1 text-[10px] font-bold uppercase tracking-wide ${
            isLive
              ? "border-rose-500/40 bg-rose-500/10 text-rose-200"
              : isAnalyzing
                ? "border-blue-500/30 bg-blue-500/10 text-blue-200"
                : "border-emerald-500/25 bg-emerald-500/10 text-emerald-300"
          }`}
        >
          {isLive ? "Live armed" : isAnalyzing ? "Capturing" : "Local idle"}
        </span>
      </div>

      <div className="grid gap-0 lg:grid-cols-[minmax(0,1.35fr)_minmax(250px,0.65fr)]">
        <div className="relative min-h-72 overflow-hidden bg-black/60">
          {latestFrame ? (
            <img
              src={latestFrame.image_data_url}
              alt="Latest explicitly captured local camera frame"
              className="h-full min-h-72 w-full object-cover"
            />
          ) : (
            <div className="flex min-h-72 flex-col items-center justify-center gap-3 px-6 text-center text-zinc-600">
              <Camera className="h-10 w-10" />
              <p className="max-w-sm text-xs leading-relaxed">
                Camera output is dormant. Use Look Now for one frame or arm the visible live loop.
              </p>
            </div>
          )}

          {(isAnalyzing || isLive) && (
            <div className="absolute left-3 top-3 flex items-center gap-2 rounded-full border border-rose-400/40 bg-rose-950/80 px-3 py-1.5 text-[10px] font-bold uppercase tracking-wide text-rose-100 backdrop-blur">
              <span className="h-2 w-2 rounded-full bg-rose-400 animate-pulse" />
              {isAnalyzing ? "Capture active" : "Live armed"}
            </div>
          )}
        </div>

        <div className="flex flex-col gap-4 p-5 sm:p-6">
          <div>
            <div className="text-[10px] font-mono uppercase tracking-wider text-zinc-500">Gemma sees</div>
            <p className="mt-2 min-h-20 text-sm leading-relaxed text-zinc-200">
              {latestFrame?.description ?? "No visual description yet."}
            </p>
          </div>

          {latestFrame && (
            <div className="grid grid-cols-2 gap-2">
              <div className="vokel-panel-subtle rounded-xl px-3 py-2">
                <div className="text-[10px] font-mono uppercase text-zinc-500">Capture</div>
                <div className="mt-1 text-xs font-mono text-zinc-300">{latestFrame.capture_seconds.toFixed(2)}s</div>
              </div>
              <div className="vokel-panel-subtle rounded-xl px-3 py-2">
                <div className="text-[10px] font-mono uppercase text-zinc-500">Inference</div>
                <div className="mt-1 text-xs font-mono text-zinc-300">{latestFrame.inference_seconds.toFixed(2)}s</div>
              </div>
            </div>
          )}

          <div>
            <label className="mb-1.5 block text-[10px] font-bold uppercase tracking-wide text-zinc-500 font-mono">
              Camera
            </label>
            <div className="flex gap-2">
              <select
                value={selectedDevice}
                disabled={isLive || isAnalyzing}
                onChange={(event) => {
                  setSelectedDevice(event.target.value);
                }}
                className="vokel-field"
              >
                {cameras.length === 0 && <option value="">No cameras found</option>}
                {cameras.map((camera) => (
                  <option key={camera.path} value={camera.path}>
                    {camera.path} - {camera.name}
                  </option>
                ))}
              </select>
              <button
                type="button"
                disabled={isLive || isAnalyzing}
                onClick={() => void loadCameras()}
                className="touch-button shrink-0 rounded-xl border border-zinc-850 bg-zinc-950 px-3 text-zinc-300 transition hover:bg-zinc-900 disabled:opacity-40"
                aria-label="Refresh camera list"
                title="Refresh camera list"
              >
                <RefreshCw className="h-4 w-4" />
              </button>
            </div>
          </div>

          <button
            type="button"
            role="switch"
            aria-checked={voiceContextEnabled}
            onClick={() => onVoiceContextEnabledChange(!voiceContextEnabled)}
            className="touch-button vokel-panel-subtle w-full rounded-2xl px-4 py-3 text-left text-xs text-zinc-400 transition-all hover:border-purple-500/30"
          >
            <span className="flex items-center justify-between gap-4">
              <span>
                <span className="flex items-center gap-1.5 font-bold text-zinc-300 font-mono uppercase">
                  <Eye className="h-3.5 w-3.5 text-purple-400" />
                  Camera Questions in Voice Loop
                </span>
                <span className="mt-1 block leading-normal text-[11px]">
                  When armed, phrases like "what am I holding?" capture one fresh local frame.
                </span>
              </span>
              <span className="flex shrink-0 flex-col items-end gap-1">
                <span className="text-[10px] font-mono uppercase text-zinc-500">
                  {voiceContextEnabled ? "Armed" : "Off"}
                </span>
                <span className="recall-switch" data-enabled={voiceContextEnabled}>
                  <span className="recall-knob" />
                </span>
              </span>
            </span>
          </button>

          <div>
            <label className="mb-1.5 block text-[10px] font-bold uppercase tracking-wide text-zinc-500 font-mono">
              Visual Prompt
            </label>
            <textarea
              value={prompt}
              disabled={isLive || isAnalyzing}
              onChange={(event) => setPrompt(event.target.value)}
              rows={2}
              className="vokel-field min-h-20 resize-y"
            />
          </div>

          {error && <p className="text-xs leading-relaxed text-rose-300">{error}</p>}

          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              disabled={!selectedDevice || isAnalyzing || isLive}
              onClick={() => void analyzeOnce()}
              className="touch-button rounded-xl border border-purple-500/30 bg-purple-600/10 px-3 text-xs font-bold uppercase tracking-wide text-purple-100 transition hover:bg-purple-600/20 disabled:opacity-40"
            >
              <span className="flex items-center justify-center gap-2">
                {isAnalyzing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Eye className="h-4 w-4" />}
                Look Now
              </span>
            </button>
            {isLive ? (
              <button
                type="button"
                onClick={stopLive}
                className="touch-button rounded-xl border border-rose-500/40 bg-rose-600/15 px-3 text-xs font-bold uppercase tracking-wide text-rose-100 transition hover:bg-rose-600/25"
              >
                <span className="flex items-center justify-center gap-2">
                  <Square className="h-3.5 w-3.5 fill-current" />
                  Stop Live
                </span>
              </button>
            ) : (
              <button
                type="button"
                disabled={!selectedDevice || isAnalyzing}
                onClick={startLive}
                className="touch-button rounded-xl border border-zinc-800 bg-zinc-950 px-3 text-xs font-bold uppercase tracking-wide text-zinc-300 transition hover:bg-zinc-900 disabled:opacity-40"
              >
                <span className="flex items-center justify-center gap-2">
                  <Camera className="h-4 w-4" />
                  Start Live
                </span>
              </button>
            )}
          </div>

          <div className="flex items-center gap-2 text-[10px] leading-relaxed text-emerald-300/80 font-mono">
            <ShieldCheck className="h-3.5 w-3.5 shrink-0" />
            Loopback-only endpoint. Voice camera questions capture one frame only when armed.
          </div>
        </div>
      </div>
    </section>
  );
}
