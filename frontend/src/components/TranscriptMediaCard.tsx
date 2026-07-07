import React from "react";
import { Clapperboard, Globe, Image as ImageIcon, Wrench } from "lucide-react";
import type { MediaCard } from "./mediaCardTypes";

interface TranscriptMediaCardProps {
  card: MediaCard;
}

const kindMeta = {
  web: { label: "Web", Icon: Globe, badge: "text-sky-300 border-sky-500/30" },
  image: { label: "Image", Icon: ImageIcon, badge: "text-purple-300 border-purple-500/30" },
  gif: { label: "GIF", Icon: Clapperboard, badge: "text-purple-300 border-purple-500/30" },
  tool: { label: "Tool", Icon: Wrench, badge: "text-emerald-300 border-emerald-500/30" },
} as const;

function connectionLabel(connection?: MediaCard["connection"]): string | null {
  if (!connection) return null;
  if (connection === "hermes") return "Hermes";
  if (connection === "jan") return "Jan";
  return "LM Studio";
}

function routeLabel(route?: MediaCard["route"]): string | null {
  if (!route) return null;
  return route === "external" ? "External" : "Local";
}

function privacyLabel(privacy?: MediaCard["privacy"]): string | null {
  if (!privacy || privacy === "unknown") return null;
  return privacy === "external_active" ? "External agent active" : "Local";
}

export const TranscriptMediaCard: React.FC<TranscriptMediaCardProps> = ({ card }) => {
  const meta = kindMeta[card.kind];
  const { Icon } = meta;

  return (
    <figure className="my-3 rounded-2xl overflow-hidden border border-white/10 shadow-lg shadow-purple-500/8 bg-zinc-950/40 max-w-[320px]">
      {card.imageUrl && (
        <img
          src={card.imageUrl}
          alt={card.title ?? card.kind}
          className={card.kind === "gif" ? "w-full rounded-t-2xl" : "w-full max-h-72 object-cover"}
          loading="lazy"
          onError={(e) => {
            (e.target as HTMLImageElement).style.display = "none";
          }}
        />
      )}
      <figcaption className="px-3 py-2 bg-zinc-950/60 space-y-2">
        <div className="flex flex-wrap items-center gap-1.5 text-[10px] font-mono">
          <span className={`inline-flex items-center gap-1 rounded border px-1.5 py-0.5 ${meta.badge}`}>
            <Icon className="w-3 h-3" />
            {meta.label}
          </span>
          {connectionLabel(card.connection) && (
            <span className="inline-flex items-center rounded border border-zinc-700 px-1.5 py-0.5 text-zinc-400">
              {connectionLabel(card.connection)}
            </span>
          )}
          {routeLabel(card.route) && (
            <span className="inline-flex items-center rounded border border-zinc-700 px-1.5 py-0.5 text-zinc-400">
              {routeLabel(card.route)}
            </span>
          )}
          {privacyLabel(card.privacy) && (
            <span className="inline-flex items-center rounded border border-zinc-700 px-1.5 py-0.5 text-zinc-400">
              {privacyLabel(card.privacy)}
            </span>
          )}
        </div>
        {(card.title || card.content || card.toolName) && (
          <div className="text-[11px] leading-relaxed text-zinc-300">
            {card.title || card.content || card.toolName}
          </div>
        )}
        {card.sourceUrl && (
          <a
            href={card.sourceUrl}
            target="_blank"
            rel="noreferrer"
            className="text-[10px] text-sky-300 underline decoration-sky-400/40 underline-offset-2 break-all"
          >
            {card.sourceUrl}
          </a>
        )}
      </figcaption>
    </figure>
  );
};

