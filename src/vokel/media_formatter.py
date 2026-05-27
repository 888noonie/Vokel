"""Unified rich media formatter for tool results.

Produces structured payloads for MediaCard rendering in the frontend.
Used by both Built-in (ConversationEngine) and Hermes paths for consistency.
Vokel owns all presentation; external agents only supply the raw result text.
"""

from __future__ import annotations

import re
from typing import Any


def _extract_image_url(text: str) -> str | None:
    m = re.search(r"!\[[^\]]*\]\((https?://[^)]+)\)", text)
    return m.group(1) if m else None


def _extract_gif_url(text: str) -> str | None:
    # GIFs use the same markdown but with gif: prefix in alt
    m = re.search(r"!\[[^\]]*\]\((https?://[^)]+\.(?:gif|webp|png|jpg)[^)]*)\)", text, re.I)
    if m:
        return m.group(1)
    # fallback any url in giphy context
    m = re.search(r"(https?://[^)\s]+\.giphy\.com[^)\s]*)", text, re.I)
    return m.group(1) if m else None


def _extract_attribution(text: str) -> str | None:
    # Common patterns from our tools
    for pat in (
        r"(Photo by .+? on Unsplash)",
        r"(Powered by GIPHY[^)]*)",
        r"(Source: .+)",
        r"(via .+)",
    ):
        m = re.search(pat, text, re.I)
        if m:
            return m.group(1).strip()
    # last line heuristic
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if lines:
        last = lines[-1]
        if len(last) < 120 and any(k in last.lower() for k in ("unsplash", "giphy", "source", "credit", "by ")):
            return last
    return None


def _extract_source(text: str) -> str | None:
    # Prefer explicit links after numbers or "Source:"
    m = re.search(r"(https?://[^\s)]+)", text)
    if m:
        return m.group(1)
    m = re.search(r"Source:\s*(.+?)(?:\n|$)", text, re.I)
    if m:
        return m.group(1).strip()
    return None


def _extract_title(text: str, tool_name: str) -> str | None:
    # Try to pull a clean title from alt text or first heading-ish line
    m = re.search(r"!\[([^\]]*)\]\(", text)
    if m:
        alt = m.group(1)
        if alt.startswith("gif:"):
            alt = alt[4:]
        if alt and len(alt) > 2:
            return alt.strip()[:80]
    # first non-empty line that's not a url or credit
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("http") and not any(x in line.lower() for x in ("unsplash", "giphy", "powered", "photo by", "source:")):
            return line[:80]
    return None


def format_tool_result_for_display(result: str, tool_name: str) -> dict[str, Any]:
    """Returns structured data for beautiful MediaCard rendering.

    Works for Built-in tool results and (when Hermes emits similar blocks)
    for external agent results. Frontend decides final presentation.
    """
    if not result or not isinstance(result, str):
        return {"type": "structured", "content": str(result or "")}

    lower = result.lower()
    image_url = _extract_image_url(result) or _extract_gif_url(result)

    if tool_name in ("search_image", "search_gif") or (image_url and "gif" not in tool_name.lower()):
        is_gif = tool_name == "search_gif" or "gif:" in result or ".gif" in (image_url or "").lower()
        return {
            "type": "gif" if is_gif else "image",
            "title": _extract_title(result, tool_name),
            "content": result.split("\n\n")[0][:280] if "\n\n" in result else (result[:280] if len(result) > 280 else result),
            "imageUrl": image_url,
            "attribution": _extract_attribution(result),
            "source": _extract_source(result),
        }

    if tool_name == "search_web" or "search" in tool_name.lower() or "http" in lower:
        snippet = result[:320] + "..." if len(result) > 320 else result
        return {
            "type": "web",
            "title": _extract_title(result, tool_name),
            "content": snippet,
            "source": _extract_source(result),
            "attribution": _extract_attribution(result),
        }

    # Default structured (any other tool output)
    return {
        "type": "structured",
        "title": _extract_title(result, tool_name),
        "content": result[:480] + "..." if len(result) > 480 else result,
        "attribution": _extract_attribution(result),
    }
