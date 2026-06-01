import React, { useEffect, useRef } from "react";
import { User, Radio, Volume2, Loader2 } from "lucide-react";
import { TranscriptMediaCard } from "./TranscriptMediaCard";
import type { MediaCard } from "./mediaCardTypes";

export interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  isPartial?: boolean;
}

interface TranscriptStreamProps {
  messages: Message[];
  status: "idle" | "listening" | "generating" | "speaking" | "paused" | "capturing_vision";
  activeConnection: "lm_studio" | "hermes";
  activeRoute: "local" | "external";
  activePrivacy?: "local" | "external_active" | "unknown";
  activeAction?: string | null;
}

const urlPattern = /(https?:\/\/[^\s)]+)/g;
const imagePattern = /!\[([^\]]*)\]\(([^)\s]+)\)/g;
const markdownLinkPattern = /\[([^\]]+)\]\(([^)\s]+)\)/g;
const toolPattern = /\[tool_call:([^\]]+)\]/g;
const imageFilePattern = /\.(png|jpe?g|webp|gif|svg)(?:[?#][^\s)]*)?$/i;

function isLikelyImageLinkTarget(label: string, target: string): boolean {
  const lowerTarget = target.toLowerCase();
  const lowerLabel = label.toLowerCase();
  if (imageFilePattern.test(lowerTarget)) return true;
  if (lowerTarget.includes("images.unsplash.com")) return true;
  if (lowerTarget.includes("media.giphy.com")) return true;
  if (lowerLabel.includes("image") || lowerLabel.includes("photo") || lowerLabel.includes("picture") || lowerLabel.includes("gif")) {
    return /^https?:\/\//.test(lowerTarget);
  }
  return false;
}

function isLikelyPlainImageUrl(target: string): boolean {
  const lowerTarget = target.toLowerCase();
  return (
    imageFilePattern.test(lowerTarget) ||
    lowerTarget.includes("images.unsplash.com") ||
    lowerTarget.includes("media.giphy.com")
  );
}

function inferMediaCards(
  text: string,
  activeConnection: "lm_studio" | "hermes",
  activeRoute: "local" | "external",
  activePrivacy: "local" | "external_active" | "unknown"
): MediaCard[] {
  const cards: MediaCard[] = [];
  let index = 0;
  const imageSources = new Set<string>();

  const imageMatches = [...text.matchAll(imagePattern)];
  for (const m of imageMatches) {
    const rawAlt = m[1] ?? "";
    const src = m[2] ?? "";
    const isGif = rawAlt.startsWith("gif:");
    const alt = isGif ? rawAlt.slice(4) : rawAlt;
    cards.push({
      id: `img-${index++}`,
      kind: isGif ? "gif" : "image",
      title: alt || undefined,
      imageUrl: src,
      sourceUrl: src,
      connection: activeConnection,
      route: activeRoute,
      privacy: activePrivacy,
    });
    imageSources.add(src);
  }

  const strippedImages = text.replace(imagePattern, " ");
  const markdownLinks = [...strippedImages.matchAll(markdownLinkPattern)];
  for (const m of markdownLinks) {
    const label = (m[1] ?? "").trim();
    const target = (m[2] ?? "").trim();
    if (!target || imageSources.has(target) || !isLikelyImageLinkTarget(label, target)) {
      continue;
    }
    const isGif = label.toLowerCase().includes("gif") || /\.gif(?:[?#].*)?$/i.test(target);
    cards.push({
      id: `img-link-${index++}`,
      kind: isGif ? "gif" : "image",
      title: label || undefined,
      imageUrl: target,
      sourceUrl: target,
      connection: activeConnection,
      route: activeRoute,
      privacy: activePrivacy,
    });
    imageSources.add(target);
  }

  const strippedMarkdownLinks = strippedImages.replace(markdownLinkPattern, " ");
  const plainUrlMatches = [...strippedMarkdownLinks.matchAll(urlPattern)];
  for (const m of plainUrlMatches) {
    const raw = m[1] ?? "";
    const href = raw.replace(/[.,;!?]+$/, "");
    if (!href || imageSources.has(href) || !isLikelyPlainImageUrl(href)) {
      continue;
    }
    const isGif = /\.gif(?:[?#].*)?$/i.test(href) || href.toLowerCase().includes("media.giphy.com");
    cards.push({
      id: `img-url-${index++}`,
      kind: isGif ? "gif" : "image",
      imageUrl: href,
      sourceUrl: href,
      connection: activeConnection,
      route: activeRoute,
      privacy: activePrivacy,
    });
    imageSources.add(href);
  }

  const hasMediaCard = imageSources.size > 0;
  const toolMatches = [...strippedImages.matchAll(toolPattern)];
  for (const m of toolMatches) {
    const toolName = (m[1] ?? "").trim();
    cards.push({
      id: `tool-${index++}`,
      kind: "tool",
      title: toolName ? `Tool call: ${toolName}` : "Tool call",
      toolName: toolName || undefined,
      connection: activeConnection,
      route: activeRoute,
      privacy: activePrivacy,
    });
  }

  // If we already have an image/GIF card, avoid extra web-source cards from attribution links.
  if (hasMediaCard) {
    return cards;
  }

  const strippedForUrls = strippedImages.replace(toolPattern, " ");
  const seen = new Set<string>();
  const urlMatches = [...strippedForUrls.matchAll(urlPattern)];
  for (const m of urlMatches) {
    const raw = m[1] ?? "";
    const href = raw.replace(/[.,;!?]+$/, "");
    if (!href || seen.has(href) || imageSources.has(href)) continue;
    seen.add(href);
    cards.push({
      id: `web-${index++}`,
      kind: "web",
      title: "Web source",
      sourceUrl: href,
      connection: activeConnection,
      route: activeRoute,
      privacy: activePrivacy,
    });
  }

  return cards;
}

function renderTextWithLinks(text: string) {
  // Remove raw media markdown/image links from text body; cards render media separately.
  const cleaned = text
    .replace(imagePattern, "")
    .replace(markdownLinkPattern, (full, label, target) => (
      isLikelyImageLinkTarget(String(label ?? ""), String(target ?? "")) ? "" : full
    ))
    .replace(urlPattern, (full) => (isLikelyPlainImageUrl(String(full ?? "")) ? "" : full));
  // Then handle plain URLs in remaining text fragments.
  const parts: React.ReactNode[] = [];
  parts.push(...renderUrlsInText(cleaned, 0));
  return parts.length > 0 ? parts : [cleaned];
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

export const TranscriptStream: React.FC<TranscriptStreamProps> = ({
  messages,
  status,
  activeConnection,
  activeRoute,
  activePrivacy = "unknown",
  activeAction = null,
}) => {
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
      <div className="mb-4 border-b border-white/10 pb-3">
        <h2 className="flex items-center justify-between text-lg font-semibold text-zinc-100 sm:text-xl">
          <span>Live Transcript</span>
          <span className="rounded-md bg-zinc-800 px-2 py-1 font-mono text-xs uppercase text-zinc-400">
            {messages.length} Turn{messages.length !== 1 ? "s" : ""}
          </span>
        </h2>
        {activeAction && (
          <div className="mt-3 flex items-center gap-2 rounded-lg border border-blue-500/20 bg-blue-500/10 px-3 py-2 text-xs font-mono text-blue-100">
            <Loader2 className="h-4 w-4 shrink-0 animate-spin text-blue-300" />
            <span className="truncate">{activeAction}</span>
            <span className="ml-auto h-1.5 w-1.5 shrink-0 rounded-full bg-blue-300" />
          </div>
        )}
      </div>

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
                  {!isUser &&
                    inferMediaCards(msg.text, activeConnection, activeRoute, activePrivacy).map((card) => (
                      <TranscriptMediaCard key={`${msg.id}-${card.id}`} card={card} />
                    ))}
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
