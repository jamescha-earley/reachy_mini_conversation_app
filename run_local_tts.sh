#!/bin/bash
# Run TTS server locally on your laptop
# The model will download on first run (~3GB)
# Requires: cd tts_server && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

cd "$(dirname "$0")"

# Use the TTS server's isolated venv (avoids dependency conflicts with main app)
if [ ! -d "tts_server/.venv" ]; then
    echo "Error: tts_server/.venv not found."
    echo "Set it up first:  cd tts_server && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi
source tts_server/.venv/bin/activate

PORT=${1:-8001}
VOICE_SAMPLE="src/reachy_mini_conversation_app/profiles/snowflake_community/voice_sample.wav"

echo "Starting local TTS server on port $PORT..."
echo "The model will download on first run (may take a few minutes)."
echo ""

PYTHONPATH="$(pwd)" python -m tts_server.app --port $PORT &
TTS_PID=$!

echo "Waiting for model to load (~45s)..."
for i in {1..60}; do
    if curl -s "http://localhost:$PORT/health" | grep -q "healthy"; then
        echo "TTS service is ready!"
        break
    fi
    sleep 1
done

# Register voice
if [ -f "$VOICE_SAMPLE" ]; then
    echo "Registering voice sample..."
    curl -s -X POST "http://localhost:$PORT/register_voice_file?voice_id=cartoon" \
        -F "file=@$VOICE_SAMPLE" | grep -q "success" && echo "Voice registered as 'cartoon'"
else
    echo "Warning: Voice sample not found at $VOICE_SAMPLE"
fi

echo ""
echo "TTS service running (PID: $TTS_PID)"
echo "To use from robot, set: export TTS_SERVICE_URL=http://YOUR_LAPTOP_IP:$PORT"
echo "To stop: Ctrl+C"
echo ""

wait $TTS_PID
