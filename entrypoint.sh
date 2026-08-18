#!/bin/bash
set -e

# ── Ollama's OLLAMA_MODELS vs. Texel Studio's OLLAMA_MODELS ──
#
# Ollama itself reads an env var called OLLAMA_MODELS to mean "directory to
# store downloaded models in". Texel Studio's server.py/agent.py reuse the
# exact same env var name to mean "comma-separated list of model names to
# expose in the UI". If both processes see the same value, `ollama serve`
# tries to use a model *name* (e.g. "qwen3:8b") as a storage *path*, which
# silently breaks its model cache. So: read the app's meaning first, start
# the daemon with that var unset (its own default storage dir), then export
# the app's meaning only for the Python process.
APP_OLLAMA_MODELS="${OLLAMA_MODELS:-qwen2.5:3b}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

echo "[entrypoint] starting ollama serve..."
env -u OLLAMA_MODELS ollama serve &
OLLAMA_PID=$!

echo "[entrypoint] waiting for ollama to become ready..."
for i in $(seq 1 60); do
    if curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

IFS=',' read -ra MODELS <<< "$APP_OLLAMA_MODELS"
for m in "${MODELS[@]}"; do
    m_trimmed="$(echo "$m" | xargs)"
    if [ -n "$m_trimmed" ]; then
        echo "[entrypoint] pulling ollama model: $m_trimmed (first run only — cached after)"
        ollama pull "$m_trimmed" || echo "[entrypoint] WARNING: failed to pull $m_trimmed"
    fi
done

export OLLAMA_MODELS="$APP_OLLAMA_MODELS"

echo "[entrypoint] starting Texel Studio server..."
python server.py &
APP_PID=$!

trap 'kill $OLLAMA_PID $APP_PID 2>/dev/null' TERM INT
wait $APP_PID
