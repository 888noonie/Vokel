export type MediaCardKind = "web" | "image" | "gif" | "tool";

export interface MediaCard {
  id: string;
  kind: MediaCardKind;
  title?: string;
  content?: string;
  imageUrl?: string;
  sourceUrl?: string;
  toolName?: string;
  // Real now when provided by current session state.
  connection?: "jan" | "lm_studio" | "hermes";
  route?: "local" | "external";
  // Reserved for later richer routing states.
  privacy?: "local" | "external_active" | "unknown";
}

