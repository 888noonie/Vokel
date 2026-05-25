import { useEffect, useRef, useState } from "react";
import { Activity, ShieldCheck, Terminal } from "lucide-react";

export interface AgentEvent {
  id: string;
  event: string;
  level: "info" | "warning" | "error";
  backend: string;
  detail: string;
  timestamp: string;
  session_id?: string | null;
  risk?: string;
}

export interface ExecuteState {
  armed: boolean;
  risk: string;
  detail: string;
}

interface AgentConsoleProps {
  framed?: boolean;
  backend: string;
  isSessionActive: boolean;
  activeSessionId: string | null;
  events: AgentEvent[];
  executeState: ExecuteState;
  onArmExecute: () => void;
  onCancelExecute: () => void;
  onConfirmExecute: () => void;
}

const HOLD_MS = 3000;

export function AgentConsole({
  framed = true,
  backend,
  isSessionActive,
  activeSessionId,
  events,
  executeState,
  onArmExecute,
  onCancelExecute,
  onConfirmExecute,
}: AgentConsoleProps) {
  const [holdStartedAt, setHoldStartedAt] = useState<number | null>(null);
  const [holdProgress, setHoldProgress] = useState(0);
  const holdTimerRef = useRef<number | null>(null);

  useEffect(() => {
    if (holdStartedAt === null) {
      setHoldProgress(0);
      return;
    }

    const tick = () => {
      const elapsed = Date.now() - holdStartedAt;
      const progress = Math.min(1, elapsed / HOLD_MS);
      setHoldProgress(progress);
      if (progress >= 1) {
        setHoldStartedAt(null);
        onConfirmExecute();
        return;
      }
      holdTimerRef.current = window.setTimeout(tick, 50);
    };

    tick();
    return () => {
      if (holdTimerRef.current !== null) {
        window.clearTimeout(holdTimerRef.current);
      }
    };
  }, [holdStartedAt, onConfirmExecute]);

  const stopHold = () => setHoldStartedAt(null);

  const content = (
    <>
      <div className="mb-4 flex items-center justify-between gap-3">
        <h2 className="flex items-center space-x-2 text-sm font-bold uppercase tracking-wider text-zinc-400 font-mono">
          <Terminal className="h-4 w-4 text-purple-400" />
          <span>Agent Console</span>
        </h2>
        <span
          className={`rounded-full border px-2 py-1 text-[10px] font-bold uppercase tracking-wide ${
            isSessionActive
              ? "border-emerald-500/25 bg-emerald-500/10 text-emerald-300"
              : "border-zinc-800 bg-zinc-950 text-zinc-500"
          }`}
        >
          {backend}
        </span>
      </div>

      <div className="mb-4 grid gap-2 sm:grid-cols-3">
        <div className="vokel-panel-subtle rounded-2xl px-3 py-2">
          <div className="text-[10px] font-mono uppercase text-zinc-500">Session</div>
          <div className="mt-1 truncate text-xs font-mono text-zinc-300">
            {activeSessionId ?? "none"}
          </div>
        </div>
        <div className="vokel-panel-subtle rounded-2xl px-3 py-2">
          <div className="text-[10px] font-mono uppercase text-zinc-500">Consent</div>
          <div className="mt-1 flex items-center gap-1.5 text-xs font-mono text-zinc-300">
            <ShieldCheck className="h-3.5 w-3.5 text-purple-400" />
            <span>{executeState.armed ? "armed" : "idle"}</span>
          </div>
        </div>
        <div className="vokel-panel-subtle rounded-2xl px-3 py-2">
          <div className="text-[10px] font-mono uppercase text-zinc-500">Risk</div>
          <div className="mt-1 text-xs font-mono text-zinc-300">{executeState.risk}</div>
        </div>
      </div>

      <div className="mb-4 grid grid-cols-2 gap-2 sm:grid-cols-3">
        <button
          type="button"
          disabled={!isSessionActive || executeState.armed}
          onClick={onArmExecute}
          className="touch-button rounded-xl border border-purple-500/25 bg-purple-600/10 px-3 text-xs font-bold uppercase tracking-wide text-purple-100 transition hover:bg-purple-600/20 disabled:opacity-40"
        >
          Arm Execute
        </button>
        <button
          type="button"
          disabled={!executeState.armed}
          onPointerDown={() => setHoldStartedAt(Date.now())}
          onPointerUp={stopHold}
          onPointerLeave={stopHold}
          onPointerCancel={stopHold}
          className="touch-button relative overflow-hidden rounded-xl border border-amber-400/25 bg-amber-500/10 px-3 text-xs font-bold uppercase tracking-wide text-amber-100 transition hover:bg-amber-500/20 disabled:opacity-40"
        >
          <span
            className="absolute inset-y-0 left-0 bg-amber-300/20"
            style={{ width: `${holdProgress * 100}%` }}
          />
          <span className="relative">Hold 3s</span>
        </button>
        <button
          type="button"
          disabled={!executeState.armed}
          onClick={onCancelExecute}
          className="touch-button col-span-2 rounded-xl border border-zinc-800 bg-zinc-950 px-3 text-xs font-bold uppercase tracking-wide text-zinc-300 transition hover:bg-zinc-900 disabled:opacity-40 sm:col-span-1"
        >
          Cancel
        </button>
      </div>

      <div className="agent-console-log rounded-2xl border border-zinc-850 bg-zinc-950/70 p-3">
        {events.length === 0 ? (
          <div className="flex h-24 items-center justify-center gap-2 text-xs text-zinc-600">
            <Activity className="h-4 w-4" />
            <span>No agent events yet.</span>
          </div>
        ) : (
          <div className="space-y-2">
            {events.slice(-40).map((event) => (
              <div key={event.id} className="grid grid-cols-[4.25rem_minmax(0,1fr)] gap-2 text-xs">
                <span className="font-mono text-[10px] text-zinc-600">{event.timestamp}</span>
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span
                      className={`h-1.5 w-1.5 rounded-full ${
                        event.level === "error"
                          ? "bg-rose-400"
                          : event.level === "warning"
                            ? "bg-amber-300"
                            : "bg-emerald-400"
                      }`}
                    />
                    <span className="font-mono uppercase text-zinc-400">{event.backend}</span>
                    <span className="font-mono text-zinc-600">{event.event}</span>
                  </div>
                  {event.detail && (
                    <div className="mt-0.5 break-words leading-normal text-zinc-300">
                      {event.detail}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );

  if (!framed) {
    return content;
  }

  return <div className="vokel-panel rounded-3xl p-5 sm:p-6">{content}</div>;
}
