from __future__ import annotations

import httpx

from .tools import ToolDefinition

UNSPLASH_ACCESS_KEY = "vF-5RKDX3tFdMmT2f2Vx7bCz0bFMlr_H6OmQsOD6HIw"

_UNSPLASH_SEARCH_URL = "https://api.unsplash.com/search/photos"


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
    async with httpx.AsyncClient(timeout=8.0) as client:
        try:
            resp = await client.get(_UNSPLASH_SEARCH_URL, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return f"Image search failed: {e}"

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
