from __future__ import annotations

import re
import subprocess
from urllib.parse import urlparse

JAN_SERVE_PORTS = {6767}


_API_KEY_RE = re.compile(r"--api-key\s+(\S+)")


def jan_port_from_url(url: str) -> int | None:
    """Return the TCP port when *url* targets a local Jan-style OpenAI endpoint."""

    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return None
    if parsed.port is None or parsed.port not in JAN_SERVE_PORTS:
        return None
    return parsed.port


def discover_jan_api_key(port: int) -> str:
    """Read the Bearer token from a running Jan llama-server router on *port*."""

    try:
        output = subprocess.check_output(
            ["pgrep", "-af", f"llama-server.*--port {port}"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""

    for line in output.splitlines():
        match = _API_KEY_RE.search(line)
        if match:
            return match.group(1)
    return ""


def resolve_llm_api_key(url: str, explicit_key: str = "", env_key: str = "") -> str:
    """Pick the API key for a local OpenAI-compatible endpoint."""

    if explicit_key.strip():
        return explicit_key.strip()
    if env_key.strip():
        return env_key.strip()
    port = jan_port_from_url(url)
    if port is None:
        return ""
    return discover_jan_api_key(port)
