#!/data/data/com.termux/files/usr/bin/bash
# Termux Hermes Route Proof — minimal bootstrap (v1)
# Real WebSocket transport boundary test
# Vokel still owns capture/playback/interruption/consent/routing/audit
# Hermes stub only does reasoning + tool schema

set -euo pipefail

echo "=== Termux Hermes Bootstrap (WebSocket v1) ==="
echo "Target: Pixel 8 Pro + Termux"
echo "Host desktop: Ryzen 7 8845HS (will run Vokel)"

# 1. Environment confirmation
pkg update -y && pkg install -y python python-pip git
python -m ensurepip --upgrade
pip install --upgrade pip uv

echo "Python: $(python --version)"
echo "uv: $(uv --version)"

# 2. Install websockets for the real transport
pip install websockets

# 3. Minimal Hermes WebSocket stub (handshake only, no reasoning yet)
mkdir -p ~/hermes-stub
cat > ~/hermes-stub/hermes_stub.py << 'PYEOF'
#!/data/data/com.termux/files/usr/bin/env python
"""Minimal Hermes WebSocket stub — v1 (handshake only)"""
import asyncio
import json
import time
import websockets

async def handler(websocket):
    print("Hermes connected", flush=True)
    await websocket.send(json.dumps({"type": "ready", "ts": time.time()}))

    async for message in websocket:
        data = json.loads(message)
        if data.get("type") == "ping":
            await websocket.send(json.dumps({"type": "pong", "ts": time.time()}))
        elif data.get("type") == "get_schema":
            await websocket.send(json.dumps({
                "type": "tool_schema",
                "tools": ["web_search", "calculate"]
            }))

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765):
        print("Hermes WebSocket listening on :8765", flush=True)
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
PYEOF

chmod +x ~/hermes-stub/hermes_stub.py

# 4. Local-WiFi WebSocket handshake test instructions
echo ""
echo "=== Next steps on Pixel 8 Pro ==="
echo "1. Run the WebSocket stub:"
echo "   python ~/hermes-stub/hermes_stub.py"
echo ""
echo "2. From desktop (8845HS) run:"
echo "   PYTHONPATH=src python benchmarks/hermes_handshake.py --host <PIXEL_IP>"
echo ""
echo "3. Capture on Pixel:"
echo "   - cold-start time (first 'Hermes WebSocket listening')"
echo "   - memory: top -b -n1 | grep python"
echo ""
echo "Bootstrap complete. Hermes WebSocket stub ready on port 8765."
