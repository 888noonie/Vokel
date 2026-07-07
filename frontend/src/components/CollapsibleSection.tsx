import { useState, type ReactNode } from "react";
import { ChevronDown } from "lucide-react";

const panelStateKey = "vokel.panelState.v1";

type PanelStateMap = Record<string, boolean>;

let cachedPanelState: PanelStateMap | null = null;

function loadJson<T>(key: string, fallback: T): T {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function loadPanelState(): PanelStateMap {
  if (cachedPanelState === null) {
    cachedPanelState = loadJson<PanelStateMap>(panelStateKey, {});
  }
  return cachedPanelState;
}

function savePanelState(state: PanelStateMap): void {
  cachedPanelState = state;
  window.localStorage.setItem(panelStateKey, JSON.stringify(state));
}

export interface CollapsibleSectionProps {
  id: string;
  title: ReactNode;
  badge?: ReactNode;
  defaultOpen?: boolean;
  className?: string;
  children: ReactNode;
}

export function CollapsibleSection({
  id,
  title,
  badge,
  defaultOpen = true,
  className = "",
  children,
}: CollapsibleSectionProps) {
  const [open, setOpen] = useState(() => {
    const saved = loadPanelState()[id];
    return saved !== undefined ? saved : defaultOpen;
  });

  const toggle = () => {
    setOpen((current) => {
      const next = !current;
      savePanelState({ ...loadPanelState(), [id]: next });
      return next;
    });
  };

  return (
    <section className={`vokel-panel rounded-3xl overflow-hidden ${className}`.trim()}>
      <button
        type="button"
        onClick={toggle}
        aria-expanded={open}
        className="touch-button flex w-full min-h-[44px] items-center justify-between gap-3 px-5 py-3 text-left transition-colors hover:bg-white/[0.02] sm:px-6 sm:py-4"
      >
        <div className="min-w-0 text-sm font-bold uppercase tracking-wider text-zinc-400 font-mono">
          {title}
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {badge}
          <ChevronDown
            className={`h-4 w-4 text-zinc-500 transition-transform duration-[180ms] ease ${
              open ? "rotate-0" : "-rotate-90"
            }`}
            aria-hidden
          />
        </div>
      </button>
      <div className={open ? "px-5 pb-5 sm:px-6 sm:pb-6" : "hidden"}>{children}</div>
    </section>
  );
}