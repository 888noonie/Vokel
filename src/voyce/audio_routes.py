from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AudioSource:
    index: int
    name: str
    description: str
    state: str
    active_port: str | None
    active_port_available: str | None
    is_monitor: bool
    is_default: bool = False

    @property
    def usable_input(self) -> bool:
        return not self.is_monitor and self.active_port_available != "not available"


def _run_text(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def load_pulse_sources() -> list[AudioSource]:
    raw = _run_text(["pactl", "--format=json", "list", "sources"])
    default_name = get_default_source()
    return parse_pulse_sources_json(raw, default_name=default_name)


def get_default_source() -> str | None:
    try:
        value = _run_text(["pactl", "get-default-source"])
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return value or None


def parse_pulse_sources_json(raw: str, default_name: str | None = None) -> list[AudioSource]:
    data = json.loads(raw)
    return [_parse_source(source, default_name=default_name) for source in data]


def find_source(sources: list[AudioSource], source_name: str) -> AudioSource | None:
    return next((source for source in sources if source.name == source_name), None)


def set_default_source(source_name: str) -> None:
    subprocess.check_call(["pactl", "set-default-source", source_name])


def _parse_source(source: dict[str, Any], default_name: str | None) -> AudioSource:
    active_port = source.get("active_port")
    active_port_available = None
    if active_port:
        port = _find_port(source.get("ports"), active_port)
        active_port_available = port.get("available") or port.get("availability")

    name = source.get("name", "")
    return AudioSource(
        index=int(source.get("index", -1)),
        name=name,
        description=source.get("description", ""),
        state=source.get("state", ""),
        active_port=active_port,
        active_port_available=active_port_available,
        is_monitor=name.endswith(".monitor"),
        is_default=name == default_name,
    )


def _find_port(ports: Any, active_port: str) -> dict[str, Any]:
    if isinstance(ports, dict):
        return ports.get(active_port) or {}
    if isinstance(ports, list):
        return next((port for port in ports if port.get("name") == active_port), {})
    return {}
