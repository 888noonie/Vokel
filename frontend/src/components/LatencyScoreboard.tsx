import React from "react";

interface LatencyScoreboardProps {
  metrics: Record<string, number>;
}

export const LatencyScoreboard: React.FC<LatencyScoreboardProps> = ({ metrics }) => {
  // Budget targets from docs/latency-budget.md
  const budgets: Record<string, { target: number; desc: string; label: string }> = {
    turn_to_first_token: { target: 500, label: "Turn to First Token", desc: "Keeps model feeling present" },
    turn_to_playback_start: { target: 1000, label: "Turn to Playback Start", desc: "The key response moment" },
    asr_duration: { target: 350, label: "ASR Duration", desc: "Voice transcription time" },
    turn_to_first_phrase: { target: 900, label: "Turn to First Phrase", desc: "First speakable text unit" },
    capture_duration: { target: 15000, label: "Capture Duration", desc: "Time user spent speaking" },
    generation_duration: { target: 3000, label: "LLM Generation", desc: "Total text generation duration" },
  };

  const getStatusColor = (key: string, value: number) => {
    const budget = budgets[key];
    if (!budget) return "text-zinc-400 bg-zinc-900/40 border-zinc-800";
    if (value <= budget.target) return "text-emerald-400 bg-emerald-950/40 border-emerald-900/50";
    if (value <= budget.target * 1.5) return "text-amber-400 bg-amber-950/40 border-amber-900/50";
    return "text-rose-400 bg-rose-950/40 border-rose-900/50";
  };

  return (
    <div className="bg-zinc-900/80 border border-zinc-800 rounded-2xl p-6 backdrop-blur">
      <h2 className="text-xl font-semibold text-zinc-100 mb-4 flex items-center justify-between">
        <span>Latency Scoreboard</span>
        <span className="text-xs font-normal text-zinc-500">Targets from docs/latency-budget.md</span>
      </h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {Object.entries(budgets).map(([key, budget]) => {
          const value = metrics[`${key}_ms`] ?? metrics[key];
          const hasValue = value !== undefined && value !== null;

          return (
            <div
              key={key}
              className={`border rounded-xl p-4 transition-all duration-300 ${
                hasValue ? getStatusColor(key, value) : "border-zinc-800 bg-zinc-900/30 text-zinc-500"
              }`}
            >
              <div className="flex justify-between items-start mb-1">
                <span className="text-sm font-medium text-zinc-300">{budget.label}</span>
                <span className="text-xs opacity-60">Target: &lt;{budget.target}ms</span>
              </div>
              <div className="text-2xl font-bold font-mono tracking-tight my-1">
                {hasValue ? `${value.toFixed(1)} ms` : "--"}
              </div>
              <div className="text-xs opacity-80 leading-relaxed">{budget.desc}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
