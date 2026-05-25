import { Settings, History, Terminal } from "lucide-react";
import { AgentConsole } from "./AgentConsole";
import type { AgentEvent, ExecuteState } from "./AgentConsole";
import type { Message } from "./TranscriptStream";

type WorkspaceTab = "settings" | "previous" | "console";

interface ChatSnapshot {
  id: string;
  title: string;
  createdAt: string;
  backend: string;
  messages: Message[];
}

interface WorkspaceTabsProps {
  activeTab: WorkspaceTab;
  onTabChange: (tab: WorkspaceTab) => void;
  voice: string;
  ttsSpeed: number;
  outputMuted: boolean;
  outputVolume: number;
  previousChats: ChatSnapshot[];
  onRestoreChat: (chat: ChatSnapshot) => void;
  onClearPreviousChats: () => void;
  backend: string;
  isSessionActive: boolean;
  activeSessionId: string | null;
  events: AgentEvent[];
  executeState: ExecuteState;
  onArmExecute: () => void;
  onCancelExecute: () => void;
  onConfirmExecute: () => void;
}

const tabs = [
  { key: "settings" as const, label: "Settings", icon: Settings },
  { key: "previous" as const, label: "Previous Chats", icon: History },
  { key: "console" as const, label: "Console", icon: Terminal },
];

export function WorkspaceTabs({
  activeTab,
  onTabChange,
  voice,
  ttsSpeed,
  outputMuted,
  outputVolume,
  previousChats,
  onRestoreChat,
  onClearPreviousChats,
  backend,
  isSessionActive,
  activeSessionId,
  events,
  executeState,
  onArmExecute,
  onCancelExecute,
  onConfirmExecute,
}: WorkspaceTabsProps) {
  return (
    <section className="vokel-panel rounded-3xl p-5 sm:p-6">
      <div className="mb-4 flex rounded-xl border border-zinc-900 bg-zinc-950 p-1">
        {tabs.map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            type="button"
            onClick={() => onTabChange(key)}
            className={`touch-button flex-1 rounded-lg text-xs font-semibold font-mono transition-all duration-200 flex items-center justify-center gap-1.5 ${
              activeTab === key
                ? "bg-zinc-900 text-purple-400 border border-zinc-800 shadow"
                : "text-zinc-500 hover:text-zinc-300"
            }`}
          >
            <Icon className="h-3.5 w-3.5" />
            <span>{label}</span>
          </button>
        ))}
      </div>

      {activeTab === "settings" && (
        <div className="grid gap-3 sm:grid-cols-4">
          <div className="vokel-panel-subtle rounded-2xl px-3 py-2">
            <div className="text-[10px] font-mono uppercase text-zinc-500">Voice</div>
            <div className="mt-1 truncate text-xs font-mono text-zinc-300">{voice}</div>
          </div>
          <div className="vokel-panel-subtle rounded-2xl px-3 py-2">
            <div className="text-[10px] font-mono uppercase text-zinc-500">Speed</div>
            <div className="mt-1 text-xs font-mono text-zinc-300">{ttsSpeed.toFixed(2)}x</div>
          </div>
          <div className="vokel-panel-subtle rounded-2xl px-3 py-2">
            <div className="text-[10px] font-mono uppercase text-zinc-500">Output</div>
            <div className="mt-1 text-xs font-mono text-zinc-300">
              {outputMuted ? "muted" : `${Math.round(outputVolume * 100)}%`}
            </div>
          </div>
          <div className="vokel-panel-subtle rounded-2xl px-3 py-2">
            <div className="text-[10px] font-mono uppercase text-zinc-500">Backend</div>
            <div className="mt-1 text-xs font-mono text-zinc-300">{backend}</div>
          </div>
        </div>
      )}

      {activeTab === "previous" && (
        <div className="space-y-3">
          <div className="flex items-center justify-between gap-3">
            <p className="text-xs text-zinc-500">{previousChats.length} saved transcript snapshots</p>
            <button
              type="button"
              disabled={previousChats.length === 0}
              onClick={onClearPreviousChats}
              className="min-h-0 rounded-lg border border-zinc-800 bg-zinc-950 px-2.5 py-1.5 text-[10px] font-bold uppercase tracking-wide text-zinc-400 transition hover:bg-zinc-900 disabled:opacity-40"
            >
              Clear
            </button>
          </div>
          {previousChats.length === 0 ? (
            <div className="rounded-2xl border border-zinc-850 bg-zinc-950/60 p-5 text-center text-xs text-zinc-500">
              Completed sessions will appear here.
            </div>
          ) : (
            <div className="grid gap-2">
              {previousChats.map((chat) => (
                <button
                  key={chat.id}
                  type="button"
                  onClick={() => onRestoreChat(chat)}
                  className="touch-button rounded-2xl border border-zinc-850 bg-zinc-950/60 px-4 py-3 text-left transition hover:border-purple-500/30 hover:bg-zinc-900/70"
                >
                  <div className="flex items-center justify-between gap-3">
                    <span className="truncate text-sm font-semibold text-zinc-200">{chat.title}</span>
                    <span className="shrink-0 text-[10px] font-mono uppercase text-zinc-500">
                      {chat.backend}
                    </span>
                  </div>
                  <div className="mt-1 text-[11px] text-zinc-500">
                    {new Date(chat.createdAt).toLocaleString()} · {chat.messages.length} turns
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {activeTab === "console" && (
        <AgentConsole
          framed={false}
          backend={backend}
          isSessionActive={isSessionActive}
          activeSessionId={activeSessionId}
          events={events}
          executeState={executeState}
          onArmExecute={onArmExecute}
          onCancelExecute={onCancelExecute}
          onConfirmExecute={onConfirmExecute}
        />
      )}
    </section>
  );
}

export type { ChatSnapshot, WorkspaceTab };
