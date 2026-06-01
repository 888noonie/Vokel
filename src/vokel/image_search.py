from __future__ import annotations

import asyncio
import httpx

from .tools import ToolDefinition

UNSPLASH_ACCESS_KEY = "vF-5RKDX3tFdMmT2f2Vx7bCz0bFMlr_H6OmQsOD6HIw"

_UNSPLASH_SEARCH_URL = "https://api.unsplash.com/search/photos"
_UNSPLASH_TIMEOUT = 8.0
_UNSPLASH_ATTEMPTS = 3


def _rate_limit_message(retry_after: str | None) -> str:
    if retry_after and retry_after.isdigit():
        return (
            "Image search is temporarily rate-limited. "
            f"Please try again in about {retry_after} seconds."
        )
    return "Image search is temporarily rate-limited. Please try again shortly."


async def search_unsplash(query: str) -> str:
    """Fetch a single relevant image from Unsplash and return a display-ready block."""
    headers = {
        "Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}",
        "Accept-Version": "v1",
    }
    params = {
        "query": query,
        "per_page": "1",
        "orientation": "landscape",
    }
    async with httpx.AsyncClient(timeout=_UNSPLASH_TIMEOUT) as client:
        data: dict[str, object] | None = None
        for attempt in range(1, _UNSPLASH_ATTEMPTS + 1):
            try:
                resp = await client.get(_UNSPLASH_SEARCH_URL, headers=headers, params=params)
                resp.raise_for_status()
                data = resp.json()
                break
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status == 429:
                    retry_after = exc.response.headers.get("Retry-After")
                    return _rate_limit_message(retry_after)
                if status >= 500 and attempt < _UNSPLASH_ATTEMPTS:
                    await asyncio.sleep(0.25 * attempt)
                    continue
                return f"Image search failed (HTTP {status}). Please try again."
            except httpx.TimeoutException:
                if attempt < _UNSPLASH_ATTEMPTS:
                    await asyncio.sleep(0.25 * attempt)
                    continue
                return "Image search timed out. Please try again."
            except httpx.RequestError:
                if attempt < _UNSPLASH_ATTEMPTS:
                    await asyncio.sleep(0.25 * attempt)
                    continue
                return "Image search is temporarily unavailable due to a network issue."
            except Exception:
                return "Image search failed unexpectedly. Please try again."

        if data is None:
            return "Image search failed. Please try again."

    results = data.get("results") or []
    if not results:
        return f"No images found for '{query}'."

    photo = results[0]
    image_url = photo.get("urls", {}).get("regular", "")
    alt = photo.get("alt_description") or query
    photographer = photo.get("user", {}).get("name", "Unknown")
    profile_url = photo.get("user", {}).get("links", {}).get("html", "")
    unsplash_link = photo.get("links", {}).get("html", "")

    # Unsplash API guidelines require attribution
    credit = f"Photo by {photographer}"
    if profile_url:
        credit += f" ({profile_url})"
    credit += " on Unsplash"
    if unsplash_link:
        credit += f" ({unsplash_link})"

    return (
        f"![{alt}]({image_url})\n"
        f"{alt.capitalize()}. {credit}"
    )


def get_image_search_tool() -> ToolDefinition:
    return ToolDefinition(
        name="search_image",
        description=(
            "Search for a photograph or image matching a description. "
            "Call this tool when the user says 'show me an image of', "
            "'picture of', 'photo of', or asks to see something visually."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Short description of the desired image, e.g. 'golden retriever puppy'.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        func=search_unsplash,
    )
