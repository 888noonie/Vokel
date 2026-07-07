import { useEffect, useState } from "react";

export interface BeatPulse {
  bar: number;
  beat: number;
  bpm: number;
  receivedAt: number;
}

interface BeatIndicatorProps {
  musicalMode: boolean;
  isSessionActive: boolean;
  beatPulse: BeatPulse | null;
  musicalLevel: number;
  onMusicalLevelChange: (level: number) => void;
  onMusicalNudge: (ms: number) => void;
  beatsPerBar?: number;
}

export function BeatIndicator({
  musicalMode,
  isSessionActive,
  beatPulse,
  musicalLevel,
  onMusicalLevelChange,
  onMusicalNudge,
  beatsPerBar = 4,
}: BeatIndicatorProps) {
  const [inPocket, setInPocket] = useState(false);

  useEffect(() => {
    if (!beatPulse) {
      setInPocket(false);
      return;
    }
    const intervalMs = (60_000 / beatPulse.bpm);
    const update = () => {
      setInPocket(Date.now() - beatPulse.receivedAt <= intervalMs * 1.5);
    };
    update();
    const timer = window.setInterval(update, 120);
    return () => window.clearInterval(timer);
  }, [beatPulse]);

  if (!musicalMode || !isSessionActive) {
    return null;
  }

  const bpm = beatPulse?.bpm ?? 90;
  const bar = beatPulse?.bar ?? 0;
  const activeBeat = beatPulse?.beat ?? 0;

  return (
    <div className="vokel-panel rounded-3xl px-5 py-4">
      <div className="flex items-center justify-between gap-3">
        <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-500">
          Musical Grid
        </div>
        <span
          className={`beat-pocket-badge text-[10px] font-mono uppercase tracking-wide ${
            inPocket ? "beat-pocket-badge--live" : ""
          }`}
        >
          In Pocket
        </span>
      </div>

      <div className="mt-3 flex items-center justify-center gap-3">
        {Array.from({ length: beatsPerBar }, (_, index) => {
          const isActive = beatPulse !== null && index === activeBeat;
          const isDownbeat = index === 0;
          return (
            <span
              key={index}
              className={[
                "beat-indicator-dot",
                isActive ? "beat-indicator-dot--active" : "",
                isDownbeat ? "beat-indicator-dot--downbeat" : "",
              ]
                .filter(Boolean)
                .join(" ")}
              aria-hidden
            />
          );
        })}
      </div>

      <div className="mt-3 text-center font-mono text-xs tracking-wide text-zinc-300">
        <span className="text-purple-300">♩</span> {bpm} BPM · BAR {bar + 1}
      </div>

      <div className="mt-4">
        <label className="mb-1.5 flex items-center justify-between text-[10px] font-mono uppercase tracking-wide text-zinc-500">
          <span>Beat Level</span>
          <span>{Math.round(musicalLevel * 100)}%</span>
        </label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={musicalLevel}
          onChange={(event) => onMusicalLevelChange(Number(event.target.value))}
          className="w-full accent-purple-500"
          aria-label="Backing beat level"
        />
      </div>

      <div className="mt-4">
        <div className="mb-1.5 text-[10px] font-mono uppercase tracking-wide text-zinc-500">
          Track Align
        </div>
        <div className="grid grid-cols-2 gap-2">
          <button
            type="button"
            onClick={() => onMusicalNudge(-25)}
            className="touch-button rounded-xl border border-zinc-850 bg-zinc-950 px-3 py-2 text-xs font-mono uppercase text-zinc-300 transition hover:bg-zinc-900"
          >
            -25ms
          </button>
          <button
            type="button"
            onClick={() => onMusicalNudge(25)}
            className="touch-button rounded-xl border border-zinc-850 bg-zinc-950 px-3 py-2 text-xs font-mono uppercase text-zinc-300 transition hover:bg-zinc-900"
          >
            +25ms
          </button>
        </div>
      </div>
    </div>
  );
}