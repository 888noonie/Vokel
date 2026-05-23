from __future__ import annotations

import httpx
import xml.etree.ElementTree as ET

from .tools import ToolDefinition, ToolRegistry

# Explicitly hardcoding the key as per instructions
SERPAPI_KEY = "06ec3a8ce0bd8d9476df80e7d6e249b82304969b5d45e74ea0df3a4d915e66db"


async def search_duckduckgo(query: str) -> str:
    try:
        bbc_result = await _maybe_search_bbc_rss(query)
        if bbc_result:
            return bbc_result
    except Exception:
        pass

    url = "https://serpapi.com/search"
    params = {
        "engine": "duckduckgo",
        "q": query,
        "api_key": SERPAPI_KEY,
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            results = _ranked_results(data)
            if not results:
                return "No search results found."

            snippets: list[str] = []
            for index, result in enumerate(results[:3], start=1):
                title = str(result.get("title") or "Untitled result")
                snippet = str(result.get("snippet") or "").strip()
                link = str(result.get("link") or "").strip()
                source = str(result.get("source") or "").strip()
                date = str(result.get("date") or "").strip()

                metadata = " - ".join(part for part in (source, date) if part)
                line = f"{index}. {title}"
                if metadata:
                    line += f" ({metadata})"
                if snippet:
                    line += f"\n   {snippet}"
                if link:
                    line += f"\n   {link}"
                snippets.append(line)
            return "\n\n".join(snippets)
        except Exception as e:
            return f"Search failed: {e}"


async def _maybe_search_bbc_rss(query: str) -> str:
    normalized = query.lower()
    if "bbc" not in normalized or not any(term in normalized for term in ("headline", "headlines", "top")):
        return ""

    feed_url = (
        "https://feeds.bbci.co.uk/news/politics/rss.xml"
        if "politic" in normalized
        else "https://feeds.bbci.co.uk/news/rss.xml"
    )
    async with httpx.AsyncClient() as client:
        response = await client.get(feed_url)
        response.raise_for_status()

    root = ET.fromstring(response.text)
    items = root.findall("./channel/item")
    if not items:
        return "No BBC RSS headlines found."

    lines: list[str] = []
    for index, item in enumerate(items[:3], start=1):
        title = item.findtext("title", default="Untitled BBC headline")
        description = item.findtext("description", default="").strip()
        link = item.findtext("link", default="").strip()
        line = f"{index}. {title}"
        if description:
            line += f"\n   {description}"
        if link:
            line += f"\n   {link}"
        lines.append(line)
    return "\n\n".join(lines)


def _ranked_results(data: dict[str, object]) -> list[dict[str, object]]:
    ranked: list[dict[str, object]] = []
    seen_links: set[str] = set()

    def add_result(result: object) -> None:
        if not isinstance(result, dict):
            return
        link = str(result.get("link") or "")
        if link and link in seen_links:
            return
        if link:
            seen_links.add(link)
        ranked.append(result)

    for result in data.get("news_results") or []:
        add_result(result)
    for result in data.get("organic_results") or []:
        add_result(result)
        for sitelink in result.get("sitelinks") or []:
            add_result(sitelink)
    return ranked


def get_web_search_tool() -> ToolDefinition:
    return ToolDefinition(
        name="search_web",
        description="ALWAYS call this tool to search the web for up-to-date news, current events, or real-time facts. DO NOT guess or hallucinate search results.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query."
                }
            },
            "required": ["query"],
            "additionalProperties": False
        },
        func=search_duckduckgo
    )


def create_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(get_web_search_tool())
    return registry
