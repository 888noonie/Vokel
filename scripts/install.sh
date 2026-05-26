#!/usr/bin/env bash
set -euo pipefail

echo "╔════════════════════════════════════════════════════════════╗"
echo "║           VOKEL — Live Natural Voice with AI               ║"
echo "║              Co-built with Grok • Install                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo

# Prerequisites
command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3.11+ required"; exit 1; }
python3 -c "import sys; assert sys.version_info >= (3,11)" || { echo "❌ Python 3.11+ required"; exit 1; }

echo "✅ Python $(python3 --version) detected"

# Virtual environment
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

echo "📦 Upgrading pip & installing uv (recommended)..."
pip install --upgrade pip uv -q

# Core + dev install
echo "📦 Installing Vokel (core + dev)..."
uv pip install -e ".[dev]" --quiet

# Audio stack (optional but recommended)
read -p "Install full audio stack (Sherpa-ONNX + Kokoro TTS)? [Y/n] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🎤 Installing audio dependencies (this may take a few minutes)..."
    uv pip install -e ".[audio]" --quiet
    sudo apt-get update -qq && sudo apt-get install -y -qq portaudio19-dev 2>/dev/null || true
fi

# Model downloader (calls your existing script + adds sensible defaults)
echo "📥 Ensuring core models are present..."
python -m scripts.download_models --minimal || true

# .env setup
if [ ! -f ".env" ]; then
    echo "📝 Creating .env from template..."
    cp .env.example .env
    echo "✏️  Please edit .env with your API keys (LM_STUDIO_URL, SerpApi, etc.)"
fi

echo
echo "✅ Vokel installed successfully!"
echo
echo "Next steps:"
echo "  # Beautiful web dashboard (recommended first run)"
echo "  vokel --web"
echo
echo "  # Or instant CLI voice turn"
echo "  vokel \"Tell me something interesting about voice interfaces\""
echo
echo "Welcome to the future of natural voice AI."
echo "— Grok & the Vokel team"
