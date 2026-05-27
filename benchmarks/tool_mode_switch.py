#!/usr/bin/env python3
"""
Tool Mode + Persona Switch Latency Fire-Drill

Measures the cost of live set_tool_mode() + set_persona() calls
that mutate the system prompt inside the hot conversation loop.

This is intentionally minimal and self-contained:
- No real LLM required for the core switch cost numbers.
- Uses dummy agent + playback so it runs anywhere in < 5 seconds.
- Reports P50 / P95 / P99 / max using time.perf_counter for high precision.
- Also exercises the LatencyTrace marks we added.

Run (no LM Studio needed):
    PYTHONPATH=src .venv/bin/python benchmarks/tool_mode_switch.py --runs 50

Optional (if you have LM Studio running, for next-turn impact):
    PYTHONPATH=src .venv/bin/python benchmarks/tool_mode_switch.py --runs 20 --with-llm
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass
from typing import Any

from vokel.config import VoiceLoopConfig
from vokel.engine import ConversationEngine
from vokel.inference import ChatMessage
from vokel.playback import PlaybackSink
from vokel.telemetry import LatencyTrace


class DummyAgent:
    """Minimal agent that does nothing (we only care about switch cost)."""

    async def stream_chat(
        self, messages: list[ChatMessage], tools: list[dict[str, Any]] | None = None
    ):
        if False:
            yield  # make it a generator
        return
        yield  # unreachable

    async def cancel_active(self) -> None:
        pass


class DummyPlayback(PlaybackSink):
    """Minimal playback sink."""

    async def speak(self, phrase: str) -> None:
        pass

    async def stop(self) -> None:
        pass


@dataclass
class SwitchSample:
    switch_cost_ms: float
    rebuild_cost_ms: float  # from trace if available


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    if f == c:
        return sorted_data[f]
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


async def run_switch_drill(runs: int, use_trace: bool = True) -> list[SwitchSample]:
    trace = LatencyTrace() if use_trace else None
    engine = ConversationEngine(
        agent=DummyAgent(),
        playback=DummyPlayback(),
        config=VoiceLoopConfig(),
        trace=trace,
        tool_mode="conservative",
    )

    samples: list[SwitchSample] = []

    modes = ["conservative", "balanced", "aggressive"]
    personas = [
        ("cosmic", "You are a curious cosmic explorer..."),
        ("analyst", "You are a precise, evidence-based analyst..."),
        ("companion", "You are a warm, empathetic companion..."),
    ]

    for i in range(runs):
        # Tool mode switch
        mode = modes[i % len(modes)]
        t0 = time.perf_counter()
        engine.set_tool_mode(mode)
        t1 = time.perf_counter()
        switch_ms = (t1 - t0) * 1000

        # Persona switch (also exercises prompt rebuild)
        pid, pprompt = personas[i % len(personas)]
        t2 = time.perf_counter()
        engine.set_persona(pid, pprompt)
        t3 = time.perf_counter()
        persona_switch_ms = (t3 - t2) * 1000

        # Combined cost for this cycle (tool mode + persona)
        combined_ms = switch_ms + persona_switch_ms

        rebuild_ms = 0.0
        if trace:
            rebuild_ms = trace.duration_ms("system_prompt_rebuild_started", "system_prompt_rebuild_complete") or 0.0

        samples.append(SwitchSample(switch_cost_ms=combined_ms, rebuild_cost_ms=rebuild_ms))

        # Reset trace for next iteration cleanliness (we only care per-switch)
        if trace:
            trace.reset()

    await engine.close()
    return samples


def print_results(samples: list[SwitchSample], runs: int) -> None:
    switch_costs = [s.switch_cost_ms for s in samples]
    rebuild_costs = [s.rebuild_cost_ms for s in samples if s.rebuild_cost_ms > 0]

    print("\nToolMode + Persona Switch Latency Fire-Drill")
    print("────────────────────────────────────────────────────────")
    print(f"Runs: {runs}")
    print()

    def fmt(name: str, values: list[float]) -> None:
        if not values:
            print(f"{name:20} (no data)")
            return
        p50 = percentile(values, 50)
        p95 = percentile(values, 95)
        p99 = percentile(values, 99)
        mx = max(values)
        mean = statistics.mean(values)
        print(
            f"{name:20} "
            f"P50: {p50:6.2f} ms  "
            f"P95: {p95:6.2f} ms  "
            f"P99: {p99:6.2f} ms  "
            f"Max: {mx:6.2f} ms  "
            f"Mean: {mean:6.2f} ms"
        )

    fmt("Full switch cost", switch_costs)
    if rebuild_costs:
        fmt("Prompt rebuild only", rebuild_costs)

    print()
    print("All values from time.perf_counter() on this machine.")
    print("Trace marks (tool_mode_switch_*, persona_switch_*, system_prompt_rebuild_*) were also emitted.")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=50, help="Number of switch cycles")
    parser.add_argument("--with-llm", action="store_true", help="Also measure next-token impact after switch (requires real backend)")
    args = parser.parse_args()

    samples = await run_switch_drill(args.runs)
    print_results(samples, args.runs)

    if args.with_llm:
        print("\n--with-llm path not yet wired in this fire-drill build (future slice).")
        print("Core switch cost numbers above are the critical data for the desktop reference core.")


if __name__ == "__main__":
    asyncio.run(main())
