#!/data/data/com.termux/files/usr/bin/bash
# Termux Hermes Route Proof — minimal bootstrap (v0)
# Vokel still owns capture/playback/interruption/consent/routing/audit
# Hermes stub only does reasoning + tool schema

set -euo pipefail

echo "=== Termux Hermes Bootstrap ==="
echo "Target: Pixel 8 Pro + Termux"
echo "Host desktop: Ryzen 7 8845HS (will run Vokel)"

# 1. Environment confirmation
pkg update -y && pkg install -y python python-pip git
python -m ensurepip --upgrade
pip install --upgrade pip uv

echo "Python: $(python --version)"
echo "uv: $(uv --version)"

# 2. Minimal Hermes stub (no full agent yet)
mkdir -p ~/hermes-stub
cat > ~/hermes-stub/hermes_stub.py << 'PYEOF'
import json, sys, time

print("Hermes online", flush=True)
print(json.dumps({
    "role": "system",
    "content": "You are a minimal Hermes reasoning stub. Respond only with tool calls or final answer. No chit-chat."
}), flush=True)

# Simple echo loop for handshake test
while True:
    line = sys.stdin.readline()
    if not line: break
    if "ping" in line:
        print(json.dumps({"type": "pong", "ts": time.time()}), flush=True)
    elif "tool_schema" in line:
        print(json.dumps({"type": "tool_schema", "tools": ["search", "calculate"]}), flush=True)
PYEOF

chmod +x ~/hermes-stub/hermes_stub.py

# 3. Local-WiFi handshake test instructions (run on desktop side too)
echo ""
echo "=== Next steps on Pixel 8 Pro ==="
echo "1. Run: python ~/hermes-stub/hermes_stub.py"
echo "2. From desktop (8845HS) run the matching Vokel test client"
echo "3. Capture: cold-start time, first-turn latency, memory (top -b -n1 | grep python)"
echo ""
echo "Bootstrap complete. Hermes stub ready."
