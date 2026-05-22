# Conversation Recall

Conversation Recall is the Phase 4 first slice. It helps the local model use
saved context without turning recall into a latency dependency.

## Defaults

- Off unless explicitly enabled with `--memory` or the dashboard toggle.
- Stored only on this machine under `data/`, which is ignored by Git.
- Retrieved before generation with bounded result and character limits.
- Written after `generation_finished`, so first token and first playback timing
  stay focused on the live loop.
- Saved notes are stored separately from raw turns and are ranked first when
  they match the current turn.

## CLI

```bash
python3 -m voyce.cli "What did we discuss about orchids?" --memory
```

Optional controls:

```bash
python3 -m voyce.cli "Remind me what matters for latency." \
  --memory \
  --memory-db data/voyce-memory.sqlite3 \
  --memory-results 3
```

Save notes without calling the model:

```bash
python3 -m voyce.cli --remember "Avoid Bluetooth output for latency benchmarks."
python3 -m voyce.cli --remember "Default desktop audio profile is laptop-mic-headphones."
```

Inspect or clear saved context:

```bash
python3 -m voyce.cli --memory-list
python3 -m voyce.cli --memory-clear
```

These commands operate on `--memory-db` too, so separate experiments can keep
separate saved context files.

## Portability

`ConversationEngine` depends on `MemoryStore`, not the desktop storage detail
directly. The desktop reference uses SQLite; Android can keep the same interface
and replace the implementation with platform-managed SQLite.

## Telemetry

Recall adds:

- `memory_retrieval_ms`
- `memory_write_ms`
- `memory_to_first_token_ms`

Those metrics should stay small before memory graduates beyond this first slice.

## Phase 4 Boundary

This slice deliberately avoids embeddings, summarization calls, and cloud sync.
Saved notes plus recent completed turns are enough to prove the product behavior
while keeping the Android port straightforward.
