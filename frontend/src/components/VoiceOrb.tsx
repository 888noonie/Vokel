import React, { useEffect, useRef } from 'react';

interface VoiceOrbProps {
  state: 'idle' | 'listening' | 'thinking' | 'speaking';
  amplitude?: number; // 0-1 for mic input intensity
}

export const VoiceOrb: React.FC<VoiceOrbProps> = ({ state, amplitude = 0 }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d')!;
    let animationFrame: number;
    let phase = 0;

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      const baseRadius = 48;
      
      let radius = baseRadius;
      let opacity = 0.9;

      if (state === 'listening') {
        radius = baseRadius + Math.sin(phase) * 8 * (0.5 + amplitude);
        opacity = 0.95;
      } else if (state === 'thinking') {
        radius = baseRadius + Math.sin(phase * 1.5) * 4;
        opacity = 0.85;
      } else if (state === 'speaking') {
        radius = baseRadius + Math.sin(phase * 2) * 6;
        opacity = 1;
      }

      // Glow
      ctx.shadowColor = state === 'listening' ? '#22c55e' : 
                        state === 'thinking' ? '#a855f7' : '#3b82f6';
      ctx.shadowBlur = 25;

      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
      ctx.fillStyle = state === 'listening' ? '#22c55e' : 
                      state === 'thinking' ? '#a855f7' : '#3b82f6';
      ctx.globalAlpha = opacity;
      ctx.fill();

      // Inner ring
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius * 0.7, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255,255,255,0.4)';
      ctx.lineWidth = 2;
      ctx.stroke();

      phase += 0.08;
      animationFrame = requestAnimationFrame(draw);
    };

    draw();

    return () => cancelAnimationFrame(animationFrame);
  }, [state, amplitude]);

  return (
    <div className="relative flex items-center justify-center">
      <canvas 
        ref={canvasRef} 
        width={140} 
        height={140} 
        className="drop-shadow-2xl"
      />
      <div className="absolute text-xs font-mono text-white/60 tracking-[3px]">
        {state.toUpperCase()}
      </div>
    </div>
  );
};
