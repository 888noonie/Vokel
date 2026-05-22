import React, { useEffect, useRef } from "react";

interface WaveformVisualizerProps {
  status: "idle" | "listening" | "generating" | "speaking";
  volume?: number; // 0.0 to 1.0 representing mic input volume
}

export const WaveformVisualizer: React.FC<WaveformVisualizerProps> = ({ status, volume = 0 }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationFrameId: number;
    let phase = 0;

    const resizeCanvas = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    };

    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    const draw = () => {
      const width = canvas.width / window.devicePixelRatio;
      const height = canvas.height / window.devicePixelRatio;

      ctx.clearRect(0, 0, width, height);

      phase += 0.05;

      // Color scheme based on state
      let primaryColor = "rgba(161, 98, 247, 0.8)"; // Purple for idle
      let secondaryColor = "rgba(161, 98, 247, 0.2)";
      let waveCount = 3;
      let amplitudeMultiplier = 0.15;
      let frequencyMultiplier = 1.0;

      if (status === "listening") {
        primaryColor = "rgba(16, 185, 129, 0.8)"; // Emerald green
        secondaryColor = "rgba(16, 185, 129, 0.15)";
        waveCount = 4;
        amplitudeMultiplier = 0.2 + volume * 0.8; // Modulated by volume
        frequencyMultiplier = 1.5;
      } else if (status === "generating") {
        primaryColor = "rgba(245, 158, 11, 0.8)"; // Amber
        secondaryColor = "rgba(245, 158, 11, 0.15)";
        waveCount = 2;
        amplitudeMultiplier = 0.05;
        frequencyMultiplier = 0.5; // Slow breathing wave
      } else if (status === "speaking") {
        primaryColor = "rgba(59, 130, 246, 0.8)"; // Blue
        secondaryColor = "rgba(59, 130, 246, 0.15)";
        waveCount = 4;
        amplitudeMultiplier = 0.35 + Math.sin(phase * 2) * 0.1; // Simulated speech wave
        frequencyMultiplier = 1.2;
      }

      ctx.lineWidth = 2;

      for (let i = 0; i < waveCount; i++) {
        ctx.beginPath();
        const wavePhase = phase + i * (Math.PI / waveCount);
        const amp = height * amplitudeMultiplier * (1 - i * 0.2);

        for (let x = 0; x < width; x++) {
          const progress = x / width;
          // Fade waves at the edges
          const edgeFade = Math.sin(progress * Math.PI);
          const y =
            height / 2 +
            Math.sin(progress * Math.PI * 2 * frequencyMultiplier + wavePhase) *
              amp *
              edgeFade;

          if (x === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }

        ctx.strokeStyle = i === 0 ? primaryColor : secondaryColor;
        ctx.stroke();
      }

      animationFrameId = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener("resize", resizeCanvas);
    };
  }, [status, volume]);

  return (
    <div className="voyce-panel-subtle rounded-3xl p-4 h-32 sm:h-40 flex flex-col justify-between overflow-hidden">
      <div className="flex justify-between gap-4 text-[11px] sm:text-xs text-zinc-500 font-medium select-none">
        <span>AUDIO PIPELINE SPECTRUM</span>
        <span className="uppercase font-mono">{status}</span>
      </div>
      <canvas ref={canvasRef} className="w-full h-20 sm:h-24 block" />
    </div>
  );
};
