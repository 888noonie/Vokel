# Slice 11: Session context field ("What are we talking about?")

**Branch:** create `feature/context-field` off `feature/musical-mode` (NOT off main — this
slice extends the musical addendum, which only exists on the musical-mode branch).
**Implementer:** Composer. **Auditor:** Fable. Report back after implementation, before
any commit; note every deviation from this plan explicitly.

## Motivation

Two problems, one field:

1. **Homophone mishearings.** Our STT stack (sherpa-onnx SenseVoice/Moonshine/Whisper,
   `src/vokel/audio.py:195-240`) exposes no vocabulary-biasing hook — the recognizer
   will keep hearing "wrap" when the user says "rap". The fix belongs one layer up:
   with the session topic in the system prompt, the LLM repairs homophones itself.
   **Accepted limitation:** the displayed transcript will still show the mishearing;
   only the response is corrected. Do not attempt transcript correction — out of scope.
2. **Cold-open confabulation.** A grounded topic gives the model something concrete to
   perform about instead of inventing tool calls (see the fake weather/image-tool
   narration family).

## 11a — Backend (web.py only; engine.py MUST NOT change)

### Track filename capture
- Add `filename: str | None = None` to `_MusicalTrackSlot` (web.py:96-99).
- In `upload_musical_track` (web.py:~372), store a sanitized name:
  `Path(file.filename).name` truncated to 80 chars, or `None` if empty.
- Clear it (`slot.filename = None`) in `delete_musical_track` alongside the buffer.

### Session context on session start
- In the session-start handler, read `session_context` from the payload next to the
  existing `musical_mode` parse (web.py:1092):
  ```python
  session_context = " ".join(str(data.get("session_context", "")).split())[:200]
  ```
  Whitespace-collapsed, capped at 200 chars. Empty string ⇒ feature entirely inert.
- **Non-musical sessions:** when `session_context` is non-empty, build the same
  `VoiceLoopConfig(system_prompt=base.system_prompt + ...)` pattern used by musical
  mode (web.py:1130-1141), appending:
  ```
  " Session topic: {session_context}. The user's words arrive via imperfect
  speech-to-text; interpret likely mishearings in favor of this topic (e.g.
  'wrap' when the topic is rap)."
  ```
  Note: today `voice_loop_config` stays `None` outside musical mode — preserve that
  when `session_context` is empty so the default path is byte-identical.
- **Musical sessions:** extend the existing addendum (web.py:1134-1141) with, when
  present:
  - topic: `" The performance topic is: {session_context}."`
  - track: `" The backing track file is named '{slot.filename}' — treat the name as
    a hint about the vibe."` (only if a track slot filename exists; the BPM is
    already in the addendum — do not duplicate it)
  - plus the same mishearing sentence as the non-musical path.
- Ordering: keep the existing performance addendum first, then topic, then track,
  then the mishearing sentence — so the current no-context prompt remains a strict
  prefix (tests below rely on this).

### Explicit non-goal: no live mid-session update
The system prompt is baked into `VoiceLoopConfig` at session start and owned by the
engine thereafter. A `set_session_context` WS message would require touching
engine.py — **do not add one.** The field takes effect on the next session start.

## 11b — Frontend (App.tsx)

- New text input in the session settings area, near the musical-mode controls.
  Label: `musicalMode ? "What are we rapping about?" : "What are we going to talk
  about?"`. Placeholder: "optional — grounds the session topic". `maxLength={200}`.
- Persist in `savedVoicePrefs` exactly like `musicalLevel` (App.tsx:153/179/258).
- Send as `session_context` in the session-start payload (App.tsx:~835).
- When a session is live, show a small hint under the field: "applies on next
  session start" (input stays editable; it simply won't take effect until restart).

## Tests (extend existing web tests; follow their fixtures)

1. Upload stores sanitized filename; DELETE clears it; path components stripped
   (`"../../evil.mp3"` → `"evil.mp3"`); >80 chars truncated.
2. `session_context` sanitization: whitespace collapsed, 200-char cap, empty/missing
   ⇒ `voice_loop_config is None` on the non-musical path (zero-behavior-change guard).
3. Non-musical + context ⇒ system prompt contains topic + mishearing sentence.
4. Musical + context + named track ⇒ addendum contains BPM (existing), topic,
   filename, mishearing sentence, in that order; no-context musical prompt is
   unchanged from Slice 10 (prefix assertion).
5. Musical + context, no track uploaded ⇒ no track sentence.

## Gates

- `.venv/bin/python -m pytest` (venv, not system python) — all green, count reported.
- `npm run build` in `frontend/` — green.
- Grep-proof: `engine.py` untouched (`git diff --stat feature/musical-mode` must not
  list it).
- Smoke (Richard): start a musical session with topic "rapping about breakfast",
  say something containing "rap" — verify the response treats it as rap, not wrap.

## Standing constraints (unchanged)

No commits without Richard's OK. Empty context field ⇒ zero behavior change.
Report deviations, don't silently improve the plan.
