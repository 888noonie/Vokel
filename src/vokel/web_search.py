from __future__ import annotations

import re
from html.parser import HTMLParser

import httpx
import xml.etree.ElementTree as ET

from .tools import ToolDefinition, ToolRegistry

SERPAPI_KEY = "06ec3a8ce0bd8d9476df80e7d6e249b82304969b5d45e74ea0df3a4d915e66db"

_THIN_SNIPPET_MAX_LEN = 120
_PAGE_SCRAPE_MAX_CHARS = 1200
_PAGE_SCRAPE_TIMEOUT = 6.0


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
    async with httpx.AsyncClient(timeout=10.0) as client:
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

                if _snippet_is_thin(snippet) and link:
                    deeper = await _scrape_page_content(client, link)
                    if deeper:
                        snippet = deeper

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


def _snippet_is_thin(snippet: str) -> bool:
    if not snippet or len(snippet) < 50:
        return True
    thin_phrases = (
        "is your online source",
        "breaking news",
        "get the latest",
        "stay updated",
        "latest news",
        "top stories",
        "detailed forecast including",
    )
    normalized = snippet.lower()
    return any(phrase in normalized for phrase in thin_phrases)


async def _scrape_page_content(client: httpx.AsyncClient, url: str) -> str:
    try:
        resp = await client.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; Vokel/1.0)"},
            follow_redirects=True,
            timeout=_PAGE_SCRAPE_TIMEOUT,
        )
        resp.raise_for_status()
    except Exception:
        return ""

    extractor = _ParagraphExtractor()
    try:
        extractor.feed(resp.text)
    except Exception:
        return ""

    joined = " ".join(extractor.paragraphs)
    if len(joined) > _PAGE_SCRAPE_MAX_CHARS:
        joined = joined[:_PAGE_SCRAPE_MAX_CHARS].rsplit(" ", 1)[0] + "..."
    return joined


class _ParagraphExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.paragraphs: list[str] = []
        self._in_p = False
        self._buf: list[str] = []
        self._skip_tags = {"script", "style", "nav", "footer", "header"}
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._skip_tags:
            self._skip_depth += 1
        if tag == "p" and self._skip_depth == 0:
            self._in_p = True
            self._buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag in self._skip_tags and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag == "p" and self._in_p:
            self._in_p = False
            text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
            if len(text) > 60:
                self.paragraphs.append(text)

    def handle_data(self, data: str) -> None:
        if self._in_p:
            self._buf.append(data)


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
    from .giphy_search import get_giphy_search_tool
    from .image_search import get_image_search_tool

    registry = ToolRegistry()
    registry.register(get_web_search_tool())
    registry.register(get_image_search_tool())
    registry.register(get_giphy_search_tool())
    return registry
