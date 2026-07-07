#!/usr/bin/env bash
#
# Start a local LLM / agent backend for Vokel.
#
#   scripts/start-backend.sh jan        # Jan serve  -> OpenAI-compatible API on :6767
#   scripts/start-backend.sh lmstudio   # LM Studio local server          on :1234
#   scripts/start-backend.sh hermes     # Hermes gateway API server       on :8642
#   scripts/start-backend.sh vokel      # Vokel FastAPI + built UI          on :8000
#
# Vokel talks to exactly one LLM backend at a time (jan OR lmstudio), plus an
# optional Hermes gateway. Each backend runs as its own long-lived process, so
# these are wired as separate VS Code "Run Task" entries (.vscode/tasks.json).
#
# Config is read from the repo .env when present (VOKEL_LLM_MODEL, ports, keys).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

# Load .env (export every assignment) without choking on comments/blank lines.
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

require() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "error: '$1' not found on PATH. Install it before starting this backend." >&2
    exit 127
  }
}

usage() {
  sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

start_jan() {
  require jan
  local model="${VOKEL_LLM_MODEL:-${LM_STUDIO_MODEL:-}}"
  local port="${JAN_PORT:-6767}"
  echo ">> Starting Jan serve on http://127.0.0.1:${port}/v1"
  echo "   model: ${model:-<interactive pick>}"

  local args=(serve --port "${port}")
  [[ -n "${model}" ]] && args+=("${model}")
  # If a key is configured, enforce it; otherwise Jan serves with no auth and
  # Vokel auto-discovers from the running process (see src/vokel/jan_key.py).
  [[ -n "${VOKEL_LLM_API_KEY:-}" ]] && args+=(--api-key "${VOKEL_LLM_API_KEY}")

  exec jan "${args[@]}"
}

start_lmstudio() {
  require lms
  local port="${LMSTUDIO_PORT:-1234}"
  echo ">> Starting LM Studio local server on http://127.0.0.1:${port}"
  lms server start --port "${port}"
  lms server status || true
  echo ">> LM Studio server is running. Load a model from the LM Studio app or 'lms load'."
  echo "   This task stays open and streams server logs (Ctrl-C to stop tailing; server keeps running)."
  exec lms log stream
}

start_hermes() {
  require hermes
  echo ">> Starting Hermes gateway (API server reads ~/.hermes/.env)"
  echo "   Expecting: [API Server] API server listening on http://127.0.0.1:8642"
  exec hermes gateway run
}

start_vokel() {
  local port="${VOKEL_PORT:-8000}"
  local host="${VOKEL_HOST:-0.0.0.0}"
  if [[ ! -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    echo "error: .venv/bin/python not found. Run 'make install' first." >&2
    exit 1
  fi
  if [[ ! -d "${REPO_ROOT}/frontend/dist" ]]; then
    echo ">> frontend/dist missing — building UI (one-time)..."
    make -C "${REPO_ROOT}" build
  fi
  echo ">> Starting Vokel web server on http://127.0.0.1:${port}"
  echo "   Local musical mode: kokoro/spd-say + musical_mode in start_session"
  cd "${REPO_ROOT}"
  exec env PYTHONPATH=src .venv/bin/python -m vokel.cli --web --host "${host}" --port "${port}"
}

main() {
  local backend="${1:-}"
  case "${backend}" in
    jan)              start_jan ;;
    lmstudio|lms)     start_lmstudio ;;
    hermes)           start_hermes ;;
    vokel|web)        start_vokel ;;
    -h|--help|help|"") usage 0 ;;
    *)
      echo "error: unknown backend '${backend}'" >&2
      usage 1
      ;;
  esac
}

main "$@"
