import React, { useEffect, useRef } from "react";
import { User, Bot, Volume2 } from "lucide-react";

export interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  isPartial?: boolean;
}

interface TranscriptStreamProps {
  messages: Message[];
  status: "idle" | "listening" | "generating" | "speaking";
}

export const TranscriptStream: React.FC<TranscriptStreamProps> = ({ messages, status }) => {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    // Auto-scroll to the bottom of the transcript as new tokens/messages arrive
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [messages, status]);

  return (
    <div className="bg-zinc-900/80 border border-zinc-800 rounded-2xl p-6 flex flex-col h-[400px] backdrop-blur">
      <h2 className="text-xl font-semibold text-zinc-100 mb-4 flex items-center justify-between border-b border-zinc-800 pb-3">
        <span>Live Transcript</span>
        <span className="text-xs bg-zinc-800 text-zinc-400 px-2 py-1 rounded-md font-mono uppercase">
          {messages.length} Turn{messages.length !== 1 ? "s" : ""}
        </span>
      </h2>

      <div
        ref={containerRef}
        className="flex-1 overflow-y-auto space-y-4 pr-2 scroll-smooth"
      >
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-zinc-500 text-sm select-none">
            <Volume2 className="w-12 h-12 mb-3 stroke-[1.5] animate-pulse text-zinc-600" />
            <p>Ready to converse. Start speaking or click start!</p>
          </div>
        ) : (
          messages.map((msg) => {
            const isUser = msg.role === "user";

            return (
              <div
                key={msg.id}
                className={`flex gap-3 max-w-[85%] ${isUser ? "ml-auto flex-row-reverse" : "mr-auto"}`}
              >
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 border ${
                    isUser
                      ? "bg-emerald-950/50 text-emerald-400 border-emerald-900/50"
                      : "bg-blue-950/50 text-blue-400 border-blue-900/50"
                  }`}
                >
                  {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                </div>

                <div
                  className={`rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
                    isUser
                      ? msg.isPartial
                        ? "bg-zinc-850/60 text-zinc-400 border border-dashed border-zinc-700 font-sans"
                        : "bg-emerald-600/10 text-emerald-100 border border-emerald-550/20"
                      : "bg-blue-600/10 text-blue-100 border border-blue-550/20"
                  }`}
                >
                  <p className={msg.isPartial ? "italic animate-pulse" : ""}>
                    {msg.text}
                  </p>
                </div>
              </div>
            );
          })
        )}

        {status === "generating" && (
          <div className="flex gap-3 mr-auto max-w-[85%] animate-pulse">
            <div className="w-8 h-8 rounded-full bg-amber-950/50 text-amber-400 border border-amber-900/50 flex items-center justify-center shrink-0">
              <Bot className="w-4 h-4" />
            </div>
            <div className="bg-amber-600/5 border border-amber-500/10 rounded-2xl px-4 py-2.5 flex items-center space-x-1.5 h-10">
              <div className="w-2 h-2 bg-amber-400 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
              <div className="w-2 h-2 bg-amber-400 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
              <div className="w-2 h-2 bg-amber-400 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
