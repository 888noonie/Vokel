import React, { useEffect, useRef } from "react";
import { User, Radio, Volume2 } from "lucide-react";

export interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  isPartial?: boolean;
}

interface TranscriptStreamProps {
  messages: Message[];
  status: "idle" | "listening" | "generating" | "speaking" | "paused";
}

const urlPattern = /(https?:\/\/[^\s)]+)/g;
const imagePattern = /!\[([^\]]*)\]\((https?:\/\/[^)]+)\)/g;

function renderTextWithLinks(text: string) {
  // First split on markdown images, then handle URLs in text fragments
  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  // Reset regex state
  imagePattern.lastIndex = 0;

  while ((match = imagePattern.exec(text)) !== null) {
    // Text before this image
    if (match.index > lastIndex) {
      parts.push(...renderUrlsInText(text.slice(lastIndex, match.index), lastIndex));
    }

    const rawAlt = match[1];
    const src = match[2];
    const isGif = rawAlt.startsWith("gif:");
    const alt = isGif ? rawAlt.slice(4) : rawAlt;

    parts.push(
      isGif ? (
        <figure key={`gif-${match.index}`} className="my-3 rounded-2xl overflow-hidden border border-purple-500/20 shadow-lg shadow-purple-500/10 bg-zinc-950/40 max-w-[280px]">
          <img
            src={src}
            alt={alt}
            className="w-full rounded-t-2xl"
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = "none";
            }}
          />
          <div className="px-3 py-1.5 text-[10px] text-zinc-500 flex items-center justify-between">
            <span className="truncate">{alt}</span>
            <span className="shrink-0 font-mono text-purple-400/60">GIF</span>
          </div>
        </figure>
      ) : (
        <figure key={`img-${match.index}`} className="my-3 rounded-xl overflow-hidden border border-white/10 shadow-lg shadow-purple-500/5">
          <img
            src={src}
            alt={alt}
            loading="lazy"
            className="w-full max-h-72 object-cover"
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = "none";
            }}
          />
          {alt && (
            <figcaption className="px-3 py-2 text-[11px] text-zinc-400 bg-zinc-950/60 leading-relaxed">
              {alt}
            </figcaption>
          )}
        </figure>
      )
    );

    lastIndex = match.index + match[0].length;
  }

  // Remaining text after last image
  if (lastIndex < text.length) {
    parts.push(...renderUrlsInText(text.slice(lastIndex), lastIndex));
  }

  return parts.length > 0 ? parts : [text];
}

function renderUrlsInText(text: string, keyOffset: number): React.ReactNode[] {
  return text.split(urlPattern).map((part, index) => {
    if (!part.match(urlPattern)) {
      return part;
    }

    const href = part.replace(/[.,;!?]+$/, "");
    const trailing = part.slice(href.length);

    return (
      <React.Fragment key={`url-${keyOffset}-${index}`}>
        <a
          href={href}
          target="_blank"
          rel="noreferrer"
          className="text-sky-300 underline decoration-sky-400/40 underline-offset-2 hover:text-sky-200"
        >
          {href}
        </a>
        {trailing}
      </React.Fragment>
    );
  });
}

export const TranscriptStream: React.FC<TranscriptStreamProps> = ({ messages, status }) => {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    // Auto-scroll to the bottom of the transcript as new tokens/messages arrive
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [messages, status]);

  const assistantMark = (
    <span className="relative flex h-4 w-4 items-center justify-center rounded-full bg-blue-400/15">
      <Radio className="h-3 w-3" />
    </span>
  );

  return (
    <div className="vokel-panel rounded-3xl p-5 sm:p-6 flex flex-col min-h-[360px] h-[52vh] max-h-[560px]">
      <h2 className="text-lg sm:text-xl font-semibold text-zinc-100 mb-4 flex items-center justify-between border-b border-white/10 pb-3">
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
          <div className="h-full flex flex-col items-center justify-center text-zinc-500 text-sm select-none text-center px-4">
            <Volume2 className="w-12 h-12 mb-3 stroke-[1.5] animate-pulse text-zinc-600" />
            <p>Ready to converse. Start speaking or press Start.</p>
          </div>
        ) : (
          messages.map((msg) => {
            const isUser = msg.role === "user";

            return (
              <div
                key={msg.id}
                className={`flex gap-3 max-w-[92%] sm:max-w-[85%] ${isUser ? "ml-auto flex-row-reverse" : "mr-auto"}`}
              >
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 border ${
                    isUser
                      ? "bg-emerald-950/50 text-emerald-400 border-emerald-900/50"
                      : "bg-blue-950/50 text-blue-400 border-blue-900/50"
                  }`}
                >
                  {isUser ? <User className="w-4 h-4" /> : assistantMark}
                </div>

                <div
                  className={`rounded-2xl px-4 py-2.5 text-sm leading-relaxed shadow-sm ${
                    isUser
                      ? msg.isPartial
                        ? "bg-zinc-850/60 text-zinc-400 border border-dashed border-zinc-700 font-sans"
                        : "bg-emerald-600/10 text-emerald-100 border border-emerald-550/20"
                      : "bg-blue-600/10 text-blue-100 border border-blue-550/20"
                  }`}
                >
                  <p className={msg.isPartial ? "italic animate-pulse whitespace-pre-wrap" : "whitespace-pre-wrap"}>
                    {renderTextWithLinks(msg.text)}
                  </p>
                </div>
              </div>
            );
          })
        )}

        {status === "generating" && (
          <div className="flex gap-3 mr-auto max-w-[85%] animate-pulse">
            <div className="w-8 h-8 rounded-full bg-amber-950/50 text-amber-400 border border-amber-900/50 flex items-center justify-center shrink-0">
              <Radio className="w-4 h-4" />
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
