from __future__ import annotations

import httpx

from .tools import ToolDefinition

GIPHY_API_KEY = "Wmia5cpP9Red8EC0XNoCT5fOXgdPUg4F"

_GIPHY_SEARCH_URL = "https://api.giphy.com/v1/gifs/search"


async def search_giphy(query: str) -> str:
    """Fetch a relevant GIF from Giphy and return a display-ready block."""
    params = {
        "api_key": GIPHY_API_KEY,
        "q": query,
        "limit": "1",
        "rating": "g",
        "lang": "en",
    }
    async with httpx.AsyncClient(timeout=8.0) as client:
        try:
            resp = await client.get(_GIPHY_SEARCH_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return f"GIF search failed: {e}"

    results = data.get("data") or []
    if not results:
        return f"No GIFs found for '{query}'."

    gif = results[0]
    gif_url = gif.get("images", {}).get("fixed_height", {}).get("url", "")
    if not gif_url:
        gif_url = gif.get("images", {}).get("original", {}).get("url", "")
    title = gif.get("title") or query
    giphy_url = gif.get("url", "")

    # Giphy requires "Powered by GIPHY" attribution
    return (
        f"![gif:{title}]({gif_url})\n"
        f"Powered by GIPHY"
        + (f" — {giphy_url}" if giphy_url else "")
    )


def get_giphy_search_tool() -> ToolDefinition:
    return ToolDefinition(
        name="search_gif",
        description=(
            "Search for an animated GIF, reaction, meme, or sticker. "
            "Call this tool when the user asks for a GIF, reaction, meme, "
            "emoji, sticker, or wants something funny or expressive."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Short description of the desired GIF, e.g. 'mind blown', 'happy dance', 'thumbs up'.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        func=search_giphy,
    )
