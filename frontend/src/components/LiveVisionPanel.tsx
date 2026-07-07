import { useCallback, useEffect, useRef, useState } from "react";
import { Aperture, Camera, Eye, Loader2, Radio, RefreshCw, ShieldCheck, Square, Video } from "lucide-react";

interface CameraDevice {
  path: string;
  name: string;
}

/** A camera as the *browser* sees it (deviceId/label), used only for the live preview. */
interface BrowserCamera {
  deviceId: string;
  label: string;
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

type SessionStatus = "idle" | "listening" | "generating" | "speaking" | "paused" | "capturing_vision";

interface LiveVisionPanelProps {
  visionUrl: string;
  visionModel: string;
  visionApiKey?: string;
  visionBackend?: "local" | "hermes";
  voiceContextEnabled: boolean;
  voiceContextFrame: VisionFrame | null;
  sessionStatus: SessionStatus;
  /** Voice-driven "action"/"cut" commands that start/stop the AI watch loop. */
  liveControlSignal?: { action: "start_live" | "stop_live"; nonce: number } | null;
  /** Registers a grabber so the server can pull one still from the live preview. */
  registerFrameGrabber?: (grab: (() => string | null) | null) => void;
  onVoiceContextEnabledChange: (enabled: boolean) => void;
  onSelectedDeviceChange: (device: string) => void;
}

const defaultPrompt = "Describe only what is visible in this image in one short sentence.";

export function LiveVisionPanel({
  visionUrl,
  visionModel,
  visionApiKey = "",
  visionBackend = "local",
  voiceContextEnabled,
  voiceContextFrame,
  sessionStatus,
  liveControlSignal = null,
  registerFrameGrabber,
  onVoiceContextEnabledChange,
  onSelectedDeviceChange,
}: LiveVisionPanelProps) {
  const [cameras, setCameras] = useState<CameraDevice[]>([]);
  const [selectedDevice, setSelectedDevice] = useState("");
  const [prompt, setPrompt] = useState(defaultPrompt);
  const [latestFrame, setLatestFrame] = useState<VisionFrame | null>(null);
  const [liveFeedEnabled, setLiveFeedEnabled] = useState(false);
  const [browserCameras, setBrowserCameras] = useState<BrowserCamera[]>([]);
  const [activePreviewIds, setActivePreviewIds] = useState<string[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isAiWatchLoop, setIsAiWatchLoop] = useState(false);
  const [shutterFlash, setShutterFlash] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestInFlightRef = useRef(false);
  const aiWatchRequestedRef = useRef(false);
  const prevVoiceFrameRef = useRef<VisionFrame | null>(null);
  // One MediaStream + <video> per previewed browser camera, so several feeds (e.g.
  // the built-in webcam and the PS3 Eye) can be shown at once.
  const streamMapRef = useRef<Map<string, MediaStream>>(new Map());
  const videoElsRef = useRef<Map<string, HTMLVideoElement>>(new Map());

  const liveFeedActive = activePreviewIds.length > 0;
  const voiceCapturing = sessionStatus === "capturing_vision";
  const aiViewing = isAnalyzing || voiceCapturing;

  const triggerShutter = useCallback(() => {
    setShutterFlash(true);
    window.setTimeout(() => setShutterFlash(false), 320);
  }, []);

  const attachVideoEl = useCallback((deviceId: string) => (el: HTMLVideoElement | null) => {
    if (el) {
      videoElsRef.current.set(deviceId, el);
      const stream = streamMapRef.current.get(deviceId);
      if (stream && el.srcObject !== stream) {
        el.srcObject = stream;
        void el.play().catch(() => undefined);
      }
    } else {
      videoElsRef.current.delete(deviceId);
    }
  }, []);

  const stopPreview = useCallback((deviceId: string) => {
    streamMapRef.current.get(deviceId)?.getTracks().forEach((track) => track.stop());
    streamMapRef.current.delete(deviceId);
    const el = videoElsRef.current.get(deviceId);
    if (el) el.srcObject = null;
    setActivePreviewIds((ids) => ids.filter((id) => id !== deviceId));
  }, []);

  const stopAllPreviews = useCallback(() => {
    streamMapRef.current.forEach((stream) => stream.getTracks().forEach((track) => track.stop()));
    streamMapRef.current.clear();
    videoElsRef.current.forEach((el) => {
      el.srcObject = null;
    });
    setActivePreviewIds([]);
  }, []);

  const startPreview = useCallback(async (deviceId: string) => {
    if (streamMapRef.current.has(deviceId)) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("This browser does not expose camera access.");
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      video: deviceId
        ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
        : { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    streamMapRef.current.set(deviceId, stream);
    setActivePreviewIds((ids) => (ids.includes(deviceId) ? ids : [...ids, deviceId]));
  }, []);

  const enumerateBrowserCameras = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return [] as BrowserCamera[];
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams = devices
      .filter((device) => device.kind === "videoinput")
      .map((device, index) => ({
        deviceId: device.deviceId,
        label: device.label || `Camera ${index + 1}`,
      }));
    setBrowserCameras(cams);
    return cams;
  }, []);

  const enableLiveFeed = useCallback(async () => {
    setError(null);
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("This browser does not expose camera access.");
      }
      // Prime permission once so enumerateDevices() returns real deviceIds + labels.
      const primer = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      primer.getTracks().forEach((track) => track.stop());
      const cams = await enumerateBrowserCameras();
      setLiveFeedEnabled(true);
      await startPreview(cams[0]?.deviceId ?? "");
    } catch (reason) {
      stopAllPreviews();
      setLiveFeedEnabled(false);
      const message =
        reason instanceof Error ? reason.message : "Could not open the browser camera.";
      setError(`${message} Allow camera access for this site (http://127.0.0.1:8000).`);
    }
  }, [enumerateBrowserCameras, startPreview, stopAllPreviews]);

  const togglePreviewCamera = useCallback(
    (deviceId: string) => {
      if (streamMapRef.current.has(deviceId)) {
        stopPreview(deviceId);
        return;
      }
      void startPreview(deviceId).catch((reason) => {
        setError(reason instanceof Error ? reason.message : "Could not open that camera.");
      });
    },
    [startPreview, stopPreview],
  );

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
    return () => {
      stopAllPreviews();
    };
  }, [stopAllPreviews]);

  // Bind freshly opened streams to their <video> elements after they render.
  useEffect(() => {
    activePreviewIds.forEach((deviceId) => {
      const el = videoElsRef.current.get(deviceId);
      const stream = streamMapRef.current.get(deviceId);
      if (el && stream && el.srcObject !== stream) {
        el.srcObject = stream;
        void el.play().catch(() => undefined);
      }
    });
  }, [activePreviewIds]);

  // Expose a grabber so the voice loop can capture a still from the live preview
  // (the browser holds the camera, so a server-side grab would hit "device busy").
  useEffect(() => {
    if (!registerFrameGrabber) return;
    registerFrameGrabber(() => {
      for (const deviceId of activePreviewIds) {
        const el = videoElsRef.current.get(deviceId);
        if (el && el.videoWidth > 0 && el.videoHeight > 0) {
          const canvas = document.createElement("canvas");
          canvas.width = el.videoWidth;
          canvas.height = el.videoHeight;
          const ctx = canvas.getContext("2d");
          if (!ctx) return null;
          ctx.drawImage(el, 0, 0, canvas.width, canvas.height);
          return canvas.toDataURL("image/jpeg", 0.85);
        }
      }
      return null;
    });
    return () => registerFrameGrabber(null);
  }, [registerFrameGrabber, activePreviewIds]);

  useEffect(() => {
    if (!voiceContextFrame) return;
    if (voiceContextFrame !== prevVoiceFrameRef.current) {
      prevVoiceFrameRef.current = voiceContextFrame;
      triggerShutter();
    }
    const timeout = window.setTimeout(() => setLatestFrame(voiceContextFrame), 0);
    return () => window.clearTimeout(timeout);
  }, [triggerShutter, voiceContextFrame]);

  useEffect(() => {
    if (!voiceCapturing) return;
    triggerShutter();
  }, [triggerShutter, voiceCapturing]);

  useEffect(() => {
    if (selectedDevice) onSelectedDeviceChange(selectedDevice);
  }, [onSelectedDeviceChange, selectedDevice]);

  const analyzeOnce = useCallback(async () => {
    if (!selectedDevice || requestInFlightRef.current) return false;
    requestInFlightRef.current = true;
    setIsAnalyzing(true);
    triggerShutter();
    setError(null);
    try {
      const response = await fetch("/api/vision/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          device: selectedDevice,
          url: visionUrl,
          model: visionModel,
          prompt,
          api_key: visionApiKey,
          backend: visionBackend,
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
  }, [prompt, selectedDevice, triggerShutter, visionApiKey, visionBackend, visionModel, visionUrl]);

  useEffect(() => {
    if (!isAiWatchLoop) return;
    let cancelled = false;

    const run = async () => {
      while (!cancelled && aiWatchRequestedRef.current) {
        const succeeded = await analyzeOnce();
        if (!succeeded) {
          aiWatchRequestedRef.current = false;
          setIsAiWatchLoop(false);
          break;
        }
        await new Promise((resolve) => window.setTimeout(resolve, 1_200));
      }
    };

    void run();
    return () => {
      cancelled = true;
    };
  }, [analyzeOnce, isAiWatchLoop]);

  const startAiWatchLoop = () => {
    aiWatchRequestedRef.current = true;
    setIsAiWatchLoop(true);
  };

  const stopAiWatchLoop = () => {
    aiWatchRequestedRef.current = false;
    setIsAiWatchLoop(false);
  };

  // Voice commands "action" (start) and "cut" (stop) drive the same AI watch loop.
  const lastLiveControlNonceRef = useRef<number | null>(null);
  useEffect(() => {
    if (!liveControlSignal) return;
    if (lastLiveControlNonceRef.current === liveControlSignal.nonce) return;
    lastLiveControlNonceRef.current = liveControlSignal.nonce;
    if (liveControlSignal.action === "start_live") {
      aiWatchRequestedRef.current = true;
      setIsAiWatchLoop(true);
    } else {
      aiWatchRequestedRef.current = false;
      setIsAiWatchLoop(false);
    }
  }, [liveControlSignal]);

  const toggleLiveFeed = () => {
    if (liveFeedEnabled) {
      stopAllPreviews();
      setLiveFeedEnabled(false);
      return;
    }
    void enableLiveFeed();
  };

  const showLiveVideo = liveFeedEnabled && liveFeedActive;
  const showCapturedStill = !showLiveVideo && latestFrame?.image_data_url;

  const statusLabel = aiViewing
    ? "AI viewing"
    : isAiWatchLoop
      ? "AI watch loop"
      : liveFeedActive
        ? "Live feed"
        : "Camera idle";

  const statusTone = aiViewing
    ? "border-rose-500/45 bg-rose-500/15 text-rose-100"
    : isAiWatchLoop
      ? "border-purple-500/35 bg-purple-500/10 text-purple-200"
      : liveFeedActive
        ? "border-emerald-500/35 bg-emerald-500/10 text-emerald-200"
        : "border-zinc-700 bg-zinc-950 text-zinc-500";

  return (
    <section className="vokel-panel overflow-hidden rounded-3xl">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 px-5 py-4 sm:px-6">
        <div>
          <div className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-zinc-300 font-mono">
            <Eye className="h-4 w-4 text-purple-400" />
            <span>Local Vision Window</span>
          </div>
          <p className="mt-1 text-[11px] leading-relaxed text-zinc-500">
            Live Feed previews your browser cameras — show one or both at once. AI captures use the device selected below.
          </p>
        </div>
        <span
          className={`rounded-full border px-2.5 py-1 text-[10px] font-bold uppercase tracking-wide ${statusTone}`}
        >
          {statusLabel}
        </span>
      </div>

      <div className="grid gap-0 lg:grid-cols-[minmax(0,1.35fr)_minmax(250px,0.65fr)]">
        <div className="relative min-h-72 overflow-hidden bg-black/60">
          {showLiveVideo && (
            <div
              className={`absolute inset-0 grid h-full w-full gap-px ${
                activePreviewIds.length > 1 ? "grid-cols-2" : "grid-cols-1"
              }`}
            >
              {activePreviewIds.map((deviceId) => {
                const cam = browserCameras.find((camera) => camera.deviceId === deviceId);
                return (
                  <div key={deviceId || "default"} className="relative h-full min-h-72 overflow-hidden bg-black">
                    <video
                      ref={attachVideoEl(deviceId)}
                      autoPlay
                      playsInline
                      muted
                      className={`h-full w-full object-cover transition-opacity duration-150 ${
                        aiViewing ? "opacity-75" : "opacity-100"
                      }`}
                    />
                    {activePreviewIds.length > 1 && (
                      <span className="absolute bottom-2 left-2 rounded bg-black/70 px-2 py-0.5 text-[9px] font-mono uppercase tracking-wide text-zinc-200">
                        {cam?.label ?? "Camera"}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {showCapturedStill ? (
            <img
              src={latestFrame!.image_data_url}
              alt="Latest captured camera frame"
              className={`h-full min-h-72 w-full object-cover transition-opacity duration-150 ${
                aiViewing ? "opacity-75" : "opacity-100"
              }`}
            />
          ) : (
            !showLiveVideo && (
              <div className="flex min-h-72 flex-col items-center justify-center gap-3 px-6 text-center text-zinc-600">
                <Camera className="h-10 w-10" />
                <p className="max-w-sm text-xs leading-relaxed">
                  Turn on <span className="text-zinc-400">Live Feed</span> for a real-time preview, or use{" "}
                  <span className="text-zinc-400">Look Now</span> for a single AI capture.
                </p>
              </div>
            )
          )}

          {liveFeedActive && (
            <div className="absolute left-3 top-3 flex items-center gap-2 rounded-full border border-emerald-400/35 bg-emerald-950/80 px-3 py-1.5 text-[10px] font-bold uppercase tracking-wide text-emerald-100 backdrop-blur">
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              <Radio className="h-3 w-3" />
              Live
            </div>
          )}

          {aiViewing && (
            <div className="absolute right-3 top-3 flex items-center gap-2 rounded-full border border-rose-400/50 bg-rose-950/85 px-3 py-1.5 text-[10px] font-bold uppercase tracking-wide text-rose-50 backdrop-blur vision-ai-viewing-badge">
              <Aperture className="h-3.5 w-3.5 animate-pulse" />
              {voiceCapturing ? "Voice capture" : isAnalyzing ? "AI shot" : "AI viewing"}
            </div>
          )}

          {isAiWatchLoop && !aiViewing && (
            <div className="absolute bottom-3 left-3 rounded-full border border-purple-400/35 bg-purple-950/80 px-3 py-1.5 text-[10px] font-bold uppercase tracking-wide text-purple-100 backdrop-blur">
              AI watch loop armed
            </div>
          )}

          {shutterFlash && <div className="vision-shutter-flash pointer-events-none absolute inset-0" aria-hidden />}
        </div>

        <div className="flex flex-col gap-4 p-5 sm:p-6">
          <div>
            <div className="text-[10px] font-mono uppercase tracking-wider text-zinc-500">Model sees</div>
            <p className="mt-2 min-h-20 text-sm leading-relaxed text-zinc-200">
              {aiViewing && !latestFrame?.description
                ? "Capturing a fresh frame for the model..."
                : latestFrame?.description ??
                  visionBackend === "hermes"
                    ? "Arm Camera Questions for voice turns, or use Look Now to send one frame to Hermes."
                    : "Text-only models cannot describe images. Live Feed preview still works."}
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
              Camera (for AI captures)
            </label>
            <div className="flex gap-2">
              <select
                value={selectedDevice}
                disabled={isAiWatchLoop || isAnalyzing}
                onChange={(event) => {
                  setSelectedDevice(event.target.value);
                }}
                className="vokel-field"
              >
                {cameras.length === 0 && <option value="">No capture cameras found</option>}
                {cameras.map((camera) => (
                  <option key={camera.path} value={camera.path}>
                    {camera.path} — {camera.name}
                  </option>
                ))}
              </select>
              <button
                type="button"
                disabled={isAiWatchLoop || isAnalyzing}
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
            aria-checked={liveFeedEnabled}
            onClick={toggleLiveFeed}
            className="touch-button vokel-panel-subtle w-full rounded-2xl px-4 py-3 text-left text-xs text-zinc-400 transition-all hover:border-emerald-500/30"
          >
            <span className="flex items-center justify-between gap-4">
              <span>
                <span className="flex items-center gap-1.5 font-bold text-zinc-300 font-mono uppercase">
                  <Video className="h-3.5 w-3.5 text-emerald-400" />
                  Live Feed
                </span>
                <span className="mt-1 block leading-normal text-[11px]">
                  Real-time browser preview — allow camera when prompted, then pick which cameras to show.
                </span>
              </span>
              <span className="flex shrink-0 flex-col items-end gap-1">
                <span className="text-[10px] font-mono uppercase text-zinc-500">
                  {liveFeedEnabled ? "On" : "Off"}
                </span>
                <span className="recall-switch" data-enabled={liveFeedEnabled}>
                  <span className="recall-knob" />
                </span>
              </span>
            </span>
          </button>

          {liveFeedEnabled && browserCameras.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-bold uppercase tracking-wide text-zinc-500 font-mono">
                  Preview cameras{browserCameras.length > 1 ? " — tap to show both" : ""}
                </span>
                <button
                  type="button"
                  onClick={() => void enumerateBrowserCameras()}
                  className="text-[10px] font-mono text-zinc-500 underline-offset-2 hover:text-zinc-300 hover:underline"
                >
                  Refresh
                </button>
              </div>
              <div className="flex flex-wrap gap-2">
                {browserCameras.map((cam) => {
                  const active = activePreviewIds.includes(cam.deviceId);
                  return (
                    <button
                      key={cam.deviceId || "default"}
                      type="button"
                      onClick={() => togglePreviewCamera(cam.deviceId)}
                      aria-pressed={active}
                      className={`touch-button rounded-lg border px-2.5 py-1.5 text-[11px] font-mono transition ${
                        active
                          ? "border-emerald-500/40 bg-emerald-600/15 text-emerald-100"
                          : "border-zinc-800 bg-zinc-950 text-zinc-400 hover:bg-zinc-900"
                      }`}
                    >
                      <span className="flex items-center gap-1.5">
                        <Video className="h-3 w-3" />
                        {cam.label}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

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
                  Say &quot;shoot&quot; for a photo, &quot;action&quot; to start a live loop, &quot;cut&quot; to stop. &quot;Watch me&quot; arms it hands-free.
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
              disabled={isAiWatchLoop || isAnalyzing}
              onChange={(event) => setPrompt(event.target.value)}
              rows={2}
              className="vokel-field min-h-20 resize-y"
            />
          </div>

          {error && <p className="text-xs leading-relaxed text-rose-300">{error}</p>}

          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              disabled={!selectedDevice || isAnalyzing || isAiWatchLoop}
              onClick={() => void analyzeOnce()}
              className="touch-button rounded-xl border border-purple-500/30 bg-purple-600/10 px-3 text-xs font-bold uppercase tracking-wide text-purple-100 transition hover:bg-purple-600/20 disabled:opacity-40"
            >
              <span className="flex items-center justify-center gap-2">
                {isAnalyzing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Aperture className="h-4 w-4" />}
                Look Now
              </span>
            </button>
            {isAiWatchLoop ? (
              <button
                type="button"
                onClick={stopAiWatchLoop}
                className="touch-button rounded-xl border border-rose-500/40 bg-rose-600/15 px-3 text-xs font-bold uppercase tracking-wide text-rose-100 transition hover:bg-rose-600/25"
              >
                <span className="flex items-center justify-center gap-2">
                  <Square className="h-3.5 w-3.5 fill-current" />
                  Stop Watch
                </span>
              </button>
            ) : (
              <button
                type="button"
                disabled={!selectedDevice || isAnalyzing}
                onClick={startAiWatchLoop}
                className="touch-button rounded-xl border border-zinc-800 bg-zinc-950 px-3 text-xs font-bold uppercase tracking-wide text-zinc-300 transition hover:bg-zinc-900 disabled:opacity-40"
              >
                <span className="flex items-center justify-center gap-2">
                  <Eye className="h-4 w-4" />
                  AI Watch Loop
                </span>
              </button>
            )}
          </div>

          <div className="flex items-center gap-2 text-[10px] leading-relaxed text-emerald-300/80 font-mono">
            <ShieldCheck className="h-3.5 w-3.5 shrink-0" />
            {visionBackend === "hermes"
              ? "Local capture only. Look Now routes one frame to your Hermes gateway (Grok vision)."
              : "Local capture only. Load a vision model in LM Studio, or use Jan with JAN_VISION=1."}
          </div>
        </div>
      </div>
    </section>
  );
}
