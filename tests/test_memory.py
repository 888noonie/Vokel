from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from voyce.memory import MemoryEntry, SQLiteMemoryStore, build_memory_context


class SQLiteMemoryStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_records_and_retrieves_relevant_turns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteMemoryStore(Path(tmp) / "memory.sqlite3")
            await store.record_turn(
                "I loved the local model conversation about orchids.",
                "We talked about keeping orchid roots airy.",
            )
            await store.record_turn(
                "The latency benchmark was under a second.",
                "That keeps the voice loop feeling live.",
            )

            entries = await store.retrieve("What did we say about orchids?", limit=2)

        self.assertEqual(len(entries), 1)
        self.assertIn("orchids", entries[0].user_text)

    async def test_explicit_facts_are_listed_and_prioritized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteMemoryStore(Path(tmp) / "memory.sqlite3")
            await store.record_turn(
                "We mentioned Android memory once.",
                "Keep the app portable.",
            )
            await store.record_fact("Default audio profile is laptop-mic-headphones.")

            entries = await store.retrieve("Which audio profile should I use?", limit=2)
            facts = await store.list_facts()

        self.assertEqual(entries[0].kind, "fact")
        self.assertIn("laptop-mic-headphones", entries[0].user_text)
        self.assertEqual(facts[0].user_text, "Default audio profile is laptop-mic-headphones.")

    async def test_clear_removes_facts_and_turns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteMemoryStore(Path(tmp) / "memory.sqlite3")
            await store.record_turn("Remember orchids.", "Orchids need airy roots.")
            await store.record_fact("Richard prefers low-latency wired output.")

            await store.clear()
            entries = await store.retrieve("orchids latency", limit=3)
            facts = await store.list_facts()

        self.assertEqual(entries, [])
        self.assertEqual(facts, [])

    async def test_facts_can_be_edited_and_deleted_by_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteMemoryStore(Path(tmp) / "memory.sqlite3")
            await store.record_fact("Use the first draft voice.")
            fact = (await store.list_facts())[0]

            await store.update_fact(fact.id or 0, "Use af_heart for baseline tests.")
            edited = await store.list_facts()
            await store.delete_fact(edited[0].id or 0)
            facts = await store.list_facts()

        self.assertEqual(edited[0].user_text, "Use af_heart for baseline tests.")
        self.assertEqual(facts, [])

    def test_build_memory_context_respects_char_limit(self) -> None:
        context = build_memory_context([], max_chars=100)

        self.assertEqual(context, "")

    def test_build_memory_context_formats_facts(self) -> None:
        # Covered through SQLite retrieval above; this asserts the prompt shape stays compact.
        entry = SQLiteMemoryStoreTests._fact_entry("Voyce should avoid Bluetooth for now.")

        context = build_memory_context([entry], max_chars=200)

        self.assertIn("Relevant saved context", context)
        self.assertIn("- Voyce should avoid Bluetooth for now.", context)

    @staticmethod
    def _fact_entry(text: str) -> MemoryEntry:
        return MemoryEntry(user_text=text, assistant_text="", created_at_ns=1, kind="fact")


if __name__ == "__main__":
    unittest.main()
