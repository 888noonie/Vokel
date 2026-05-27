import React, { useState } from "react";
import { Image, Clapperboard, Globe, FileText, ArrowRight, ChevronDown, ChevronUp } from "lucide-react";

export interface MediaCardProps {
  type: "image" | "gif" | "web" | "structured";
  title?: string;
  content: string;
  imageUrl?: string;
  source?: string;
  attribution?: string;
  onUse?: () => void;
  onSourceClick?: () => void;
  // Optional structured details for transparency (user-requested expandable data)
  details?: Record<string, any>;
  // New action callbacks (user request: Save / Hide / Remove / Delete)
  onSave?: () => void;
  onHide?: () => void;
  onRemove?: () => void;
  onDelete?: () => void;
}

const typeMeta = {
  image: { label: "Image", icon: Image, accent: "text-purple-400 border-purple-500/30" },
  gif: { label: "GIF", icon: Clapperboard, accent: "text-purple-400 border-purple-500/30" },
  web: { label: "Web", icon: Globe, accent: "text-sky-400 border-sky-500/30" },
  structured: { label: "Result", icon: FileText, accent: "text-emerald-400 border-emerald-500/30" },
};

export const MediaCard: React.FC<MediaCardProps> = ({
  type,
  title,
  content,
  imageUrl,
  source,
  attribution,
  onUse,
  onSourceClick,
  details,
  onSave,
  onHide,
  onRemove,
  onDelete,
}) => {
  const meta = typeMeta[type];
  const Icon = meta.icon;
  const [showDetails, setShowDetails] = useState(false);
  const [isHidden, setIsHidden] = useState(false);

  if (isHidden) return null;

  return (
    <div className="group bg-zinc-950 border border-white/10 rounded-2xl overflow-hidden max-w-md shadow-xl shadow-black/40">
      {imageUrl && (
        <div className="relative">
          <img
            src={imageUrl}
            alt={title || content}
            className="w-full h-auto object-cover max-h-[320px] bg-zinc-900"
            loading="lazy"
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = "none";
            }}
          />
          {type === "gif" && (
            <div className="absolute top-2 right-2 px-2 py-0.5 bg-black/80 text-[10px] font-mono tracking-wider text-purple-300 rounded">
              GIF
            </div>
          )}
          <div className="absolute top-2 left-2 px-2 py-0.5 bg-black/70 text-[10px] font-mono text-white/60 rounded flex items-center gap-1">
            <Icon className="w-3 h-3" />
            {meta.label}
          </div>
        </div>
      )}

      <div className="p-4 space-y-2">
        {title && (
          <div className="font-medium text-white text-sm leading-snug tracking-tight">
            {title}
          </div>
        )}

        {type === "web" ? (
          <div className="text-sm text-white/80 leading-relaxed space-y-1.5">
            {content
              .split(/\n\s*\n/)
              .slice(0, 3)
              .map((block, i) => (
                <div key={i} className="border-l-2 border-sky-500/30 pl-2">
                  {block}
                </div>
              ))}
          </div>
        ) : (
          <div className="text-sm text-white/80 leading-relaxed">
            {content}
          </div>
        )}

        {attribution && (
          <div className="mt-1 text-[10px] text-white/50 flex items-center gap-1.5 font-mono">
            {attribution}
          </div>
        )}

        {/* Expandable details for full transparency (user vision: drop-down with raw data) */}
        {(details || imageUrl) && (
          <div className="mt-2 border-t border-white/10 pt-2">
            <button
              onClick={() => setShowDetails(!showDetails)}
              className="flex w-full items-center justify-between text-[10px] font-mono text-white/50 hover:text-white/80 transition-colors"
            >
              <span>Details</span>
              {showDetails ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            </button>
            {showDetails && (
              <div className="mt-2 rounded-lg bg-black/40 p-3 text-[10px] text-white/70 font-mono leading-relaxed overflow-auto max-h-40">
                {details ? (
                  <div className="space-y-1">
                    {Object.entries(details).map(([key, value]) => {
                      const str = String(value);
                      const isUrl = /^https?:\/\//i.test(str);
                      return (
                        <div key={key}>
                          <span className="text-white/50">{key}: </span>
                          {isUrl ? (
                            <a
                              href={str}
                              target="_blank"
                              rel="noreferrer"
                              className="text-sky-300 underline decoration-sky-400/40 hover:text-sky-200 break-all"
                            >
                              {str}
                            </a>
                          ) : (
                            <span>{str}</span>
                          )}
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div>
                    {imageUrl && (
                      <div>
                        URL:{" "}
                        <a
                          href={imageUrl}
                          target="_blank"
                          rel="noreferrer"
                          className="text-sky-300 underline decoration-sky-400/40 hover:text-sky-200 break-all"
                        >
                          {imageUrl}
                        </a>
                      </div>
                    )}
                    {source && <div>Source: {source}</div>}
                    {attribution && <div>Attribution: {attribution}</div>}
                  </div>
                )}

                {/* Action bar requested by user: Save / Hide / Remove from chat / Delete */}
                <div className="mt-3 flex flex-wrap gap-2 border-t border-white/10 pt-2">
                  {onSave && (
                    <button
                      onClick={onSave}
                      className="px-2 py-0.5 text-[9px] rounded border border-white/20 hover:bg-white/10"
                    >
                      Save
                    </button>
                  )}
                  <button
                    onClick={() => {
                      if (onHide) onHide();
                      setIsHidden(true);
                    }}
                    className="px-2 py-0.5 text-[9px] rounded border border-white/20 hover:bg-white/10"
                  >
                    Hide
                  </button>
                  {onRemove && (
                    <button
                      onClick={onRemove}
                      className="px-2 py-0.5 text-[9px] rounded border border-amber-500/30 hover:bg-amber-500/10 text-amber-300"
                    >
                      Remove from chat
                    </button>
                  )}
                  {onDelete && (
                    <button
                      onClick={onDelete}
                      className="px-2 py-0.5 text-[9px] rounded border border-rose-500/30 hover:bg-rose-500/10 text-rose-300"
                    >
                      Delete
                    </button>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {(source || onSourceClick) && (
          <a
            href={source}
            target="_blank"
            rel="noreferrer"
            className="mt-1 inline-flex items-center gap-1 text-[10px] text-sky-300 hover:text-sky-200 underline decoration-sky-400/40 underline-offset-2 transition-colors"
            title={source}
          >
            <Globe className="w-3 h-3" />
            <span className="truncate max-w-[220px]">{source || "View source"}</span>
          </a>
        )}

        {onUse && (
          <button
            onClick={onUse}
            className="mt-3 w-full flex items-center justify-center gap-2 text-xs px-4 py-1.5 rounded-full border border-white/25 hover:bg-white/5 hover:border-white/40 active:bg-white/10 transition-all text-white/90 font-medium"
          >
            Use this in next turn
            <ArrowRight className="w-3.5 h-3.5 opacity-70 group-hover:translate-x-0.5 transition" />
          </button>
        )}
      </div>
    </div>
  );
};

export default MediaCard;
