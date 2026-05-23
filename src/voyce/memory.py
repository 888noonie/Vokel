from __future__ import annotations

import asyncio
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class MemoryConfig:
    enabled: bool = False
    path: Path = Path("data/voyce-memory.sqlite3")
    max_results: int = 3
    max_context_chars: int = 900
    scan_limit: int = 200


@dataclass(frozen=True)
class MemoryEntry:
    user_text: str
    assistant_text: str
    created_at_ns: int
    score: int = 0
    kind: str = "turn"
    id: int | None = None

    def to_prompt_snippet(self) -> str:
        if self.kind == "fact":
            return f"- {self.user_text}"
        return f"User: {self.user_text}\nAssistant: {self.assistant_text}"


class MemoryStore(Protocol):
    async def retrieve(self, query: str, limit: int) -> list[MemoryEntry]:
        pass

    async def record_turn(self, user_text: str, assistant_text: str) -> None:
        pass

    async def close(self) -> None:
        pass


class NullMemoryStore:
    async def retrieve(self, query: str, limit: int) -> list[MemoryEntry]:
        return []

    async def record_turn(self, user_text: str, assistant_text: str) -> None:
        return None

    async def close(self) -> None:
        return None


class SQLiteMemoryStore:
    """Small local memory store backed by portable SQLite.

    SQLite calls are short and run in worker threads so microphone callbacks and
    playback scheduling never wait on disk I/O.
    """

    def __init__(self, path: Path | str, scan_limit: int = 200) -> None:
        self.path = Path(path)
        self.scan_limit = scan_limit

    async def retrieve(self, query: str, limit: int) -> list[MemoryEntry]:
        if limit <= 0 or not query.strip():
            return []
        return await asyncio.to_thread(self._retrieve_sync, query, limit)

    async def record_turn(self, user_text: str, assistant_text: str) -> None:
        if not user_text.strip() or not assistant_text.strip():
            return
        await asyncio.to_thread(self._record_turn_sync, user_text.strip(), assistant_text.strip())

    async def record_fact(self, fact_text: str) -> None:
        if not fact_text.strip():
            return
        await asyncio.to_thread(self._record_fact_sync, _clean_fact(fact_text))

    async def list_facts(self, limit: int = 20) -> list[MemoryEntry]:
        if limit <= 0:
            return []
        return await asyncio.to_thread(self._list_facts_sync, limit)

    async def update_fact(self, fact_id: int, fact_text: str) -> None:
        if fact_id <= 0 or not fact_text.strip():
            return
        await asyncio.to_thread(self._update_fact_sync, fact_id, _clean_fact(fact_text))

    async def delete_fact(self, fact_id: int) -> None:
        if fact_id <= 0:
            return
        await asyncio.to_thread(self._delete_fact_sync, fact_id)

    async def clear(self) -> None:
        await asyncio.to_thread(self._clear_sync)

    async def close(self) -> None:
        return None

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at_ns INTEGER NOT NULL,
                user_text TEXT NOT NULL,
                assistant_text TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversation_memory_created "
            "ON conversation_memory(created_at_ns)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at_ns INTEGER NOT NULL,
                fact_text TEXT NOT NULL UNIQUE
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_facts_created "
            "ON memory_facts(created_at_ns)"
        )
        return conn

    def _record_turn_sync(self, user_text: str, assistant_text: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_memory (created_at_ns, user_text, assistant_text)
                VALUES (?, ?, ?)
                """,
                (time.time_ns(), user_text, assistant_text),
            )

    def _record_fact_sync(self, fact_text: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_facts (created_at_ns, fact_text)
                VALUES (?, ?)
                """,
                (time.time_ns(), fact_text),
            )

    def _list_facts_sync(self, limit: int) -> list[MemoryEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at_ns, fact_text
                FROM memory_facts
                ORDER BY created_at_ns DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            MemoryEntry(
                user_text=str(fact_text),
                assistant_text="",
                created_at_ns=int(created_at_ns),
                kind="fact",
                id=int(row_id),
            )
            for row_id, created_at_ns, fact_text in rows
        ]

    def _update_fact_sync(self, fact_id: int, fact_text: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE memory_facts
                SET created_at_ns = ?, fact_text = ?
                WHERE id = ?
                """,
                (time.time_ns(), fact_text, fact_id),
            )

    def _delete_fact_sync(self, fact_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM memory_facts WHERE id = ?", (fact_id,))

    def _clear_sync(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM conversation_memory")
            conn.execute("DELETE FROM memory_facts")

    def _retrieve_sync(self, query: str, limit: int) -> list[MemoryEntry]:
        terms = _query_terms(query)
        if not terms:
            return []

        with self._connect() as conn:
            turn_rows = conn.execute(
                """
                SELECT created_at_ns, user_text, assistant_text
                FROM conversation_memory
                ORDER BY created_at_ns DESC
                LIMIT ?
                """,
                (min(self.scan_limit, max(limit, 1) * 100),),
            ).fetchall()
            fact_rows = conn.execute(
                """
                SELECT id, created_at_ns, fact_text
                FROM memory_facts
                ORDER BY created_at_ns DESC
                LIMIT ?
                """,
                (min(self.scan_limit, max(limit, 1) * 20),),
            ).fetchall()

        scored: list[MemoryEntry] = []
        for row_id, created_at_ns, fact_text in fact_rows:
            score = _score_text(str(fact_text), terms)
            if score:
                scored.append(
                    MemoryEntry(
                        user_text=str(fact_text),
                        assistant_text="",
                        created_at_ns=int(created_at_ns),
                        score=score + 1,
                        kind="fact",
                        id=int(row_id),
                    )
                )

        for created_at_ns, user_text, assistant_text in turn_rows:
            haystack = f"{user_text} {assistant_text}".lower()
            score = _score_text(haystack, terms)
            if score:
                scored.append(
                    MemoryEntry(
                        user_text=str(user_text),
                        assistant_text=str(assistant_text),
                        created_at_ns=int(created_at_ns),
                        score=score,
                    )
                )

        scored.sort(key=lambda entry: (entry.score, entry.created_at_ns), reverse=True)
        return scored[:limit]


def build_memory_context(entries: list[MemoryEntry], max_chars: int) -> str:
    if not entries or max_chars <= 0:
        return ""

    parts: list[str] = []
    used = 0
    for entry in entries:
        snippet = entry.to_prompt_snippet()
        remaining = max_chars - used
        if remaining <= 0:
            break
        if len(snippet) > remaining:
            snippet = snippet[: max(0, remaining - 3)].rstrip() + "..."
        parts.append(snippet)
        used += len(snippet)

    if not parts:
        return ""
    return "Relevant saved context:\n" + "\n\n".join(parts)


def _query_terms(query: str) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for term in re.findall(r"[a-zA-Z0-9']+", query.lower()):
        if len(term) < 3 or term in _STOPWORDS or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def _score_text(text: str, terms: list[str]) -> int:
    haystack = text.lower()
    return sum(1 for term in terms if term in haystack)


def _clean_fact(fact_text: str) -> str:
    return re.sub(r"\s+", " ", fact_text).strip()


_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "because",
    "but",
    "can",
    "did",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "that",
    "the",
    "this",
    "was",
    "what",
    "when",
    "where",
    "who",
    "why",
    "with",
    "you",
}
