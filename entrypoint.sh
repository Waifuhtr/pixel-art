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
APP_OLLAMA_MODELS="${OLLAMA_MODELS:-${PIXELART_OLLAMA_MODELS:-qwen2.5:3b}}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

# ── CPU budget ──
#
# os.cpu_count()/llama.cpp both see the HOST's cores (64 on a Space), not the
# container's cgroup quota (8 vCPU). llama.cpp then spawns ~32 spin-waiting
# threads on 8 usable CPUs and every token crawls. Read the real quota and pin
# Ollama to that many CPUs so its own auto-detection lands on a sane number.
detect_cpus() {
    if [ -r /sys/fs/cgroup/cpu.max ]; then
        read -r quota period < /sys/fs/cgroup/cpu.max
        if [ "$quota" != "max" ] && [ -n "$period" ] && [ "$period" -gt 0 ] 2>/dev/null; then
            echo $(( quota / period > 0 ? quota / period : 1 ))
            return
        fi
    fi
    if [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ] && [ -r /sys/fs/cgroup/cpu/cpu.cfs_period_us ]; then
        quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
        period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
        if [ "$quota" -gt 0 ] 2>/dev/null && [ "$period" -gt 0 ] 2>/dev/null; then
            echo $(( quota / period > 0 ? quota / period : 1 ))
            return
        fi
    fi
    nproc 2>/dev/null || echo 4
}

CPUS="$(detect_cpus)"
echo "[entrypoint] usable CPUs (cgroup-aware): $CPUS"

OLLAMA_LAUNCH=(env -u OLLAMA_MODELS ollama serve)
if command -v taskset >/dev/null 2>&1 && [ "$CPUS" -ge 1 ] 2>/dev/null; then
    # Pinning to CPUS cores keeps llama.cpp from oversubscribing. Harmless if
    # the scheduler would have given us those cores anyway.
    OLLAMA_LAUNCH=(taskset -c "0-$((CPUS - 1))" env -u OLLAMA_MODELS ollama serve)
fi

echo "[entrypoint] starting ollama serve..."
"${OLLAMA_LAUNCH[@]}" &
OLLAMA_PID=$!

echo "[entrypoint] waiting for ollama to become ready..."
for i in $(seq 1 60); do
    if curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

# The Dockerfile pre-pulls these at build time, so normally this loop just
# confirms they're present. Only pull when actually missing — an unconditional
# `ollama pull` hits the network on every cold start and fails the whole boot
# if the registry is unreachable.
INSTALLED="$(ollama list 2>/dev/null || true)"
IFS=',' read -ra MODELS <<< "$APP_OLLAMA_MODELS"
for m in "${MODELS[@]}"; do
    m_trimmed="$(echo "$m" | xargs)"
    [ -z "$m_trimmed" ] && continue
    if grep -qF "$m_trimmed" <<< "$INSTALLED"; then
        echo "[entrypoint] ollama model already in image cache: $m_trimmed"
    else
        echo "[entrypoint] pulling ollama model: $m_trimmed"
        ollama pull "$m_trimmed" || echo "[entrypoint] WARNING: failed to pull $m_trimmed"
    fi
done

export OLLAMA_MODELS="$APP_OLLAMA_MODELS"

echo "[entrypoint] starting Texel Studio server..."
python server.py &
APP_PID=$!

trap 'kill $OLLAMA_PID $APP_PID 2>/dev/null' TERM INT
wait $APP_PID
