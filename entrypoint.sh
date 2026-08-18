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
# os.cpu_count(), Ollama and llama.cpp all see the HOST's cores (64 on a Space),
# not the container's cgroup quota (8 vCPU), so every thread-pool default is
# wildly oversized. Read the real quota once and feed it to each consumer.
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

# Keep the model resident. Ollama's 5-minute default unloads it while the user
# is reading the result, and reloading costs ~75s (mmap is disabled on CPU, so
# it re-reads 1.8GB and redoes the CPU_REPACK).
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1}"

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

# ── Thread-count override ──
#
# Ollama sizes llama.cpp's thread pool from the HOST's core count (64 here), so
# it launched llama-server with n_threads=32 on 8 usable vCPUs — 4x
# oversubscription of spin-waiting threads, which is why prompt ingestion
# crawled at ~14 tok/s. Neither an env var nor the OpenAI-compatible endpoint
# can override that per request, but a derived model with PARAMETER num_thread
# can. Building it is cheap: it reuses the base model's blobs.
#
# num_ctx also matters — the default 4096 barely fits the ~1.6k-token system
# prompt plus a few tool rounds before the context window starts shifting and
# the agent forgets its instructions mid-sprite.
BASE_MODEL="$(echo "${MODELS[0]}" | xargs)"
TUNED_MODEL="pixelart-agent"
TUNED_CTX="${AGENT_NUM_CTX:-8192}"

if [ -n "$BASE_MODEL" ]; then
    echo "[entrypoint] building $TUNED_MODEL from $BASE_MODEL (num_thread=$CPUS, num_ctx=$TUNED_CTX)"
    printf 'FROM %s\nPARAMETER num_thread %s\nPARAMETER num_ctx %s\n' \
        "$BASE_MODEL" "$CPUS" "$TUNED_CTX" > /tmp/Modelfile.pixelart
    if ollama create "$TUNED_MODEL" -f /tmp/Modelfile.pixelart; then
        APP_OLLAMA_MODELS="$TUNED_MODEL,$APP_OLLAMA_MODELS"
    else
        echo "[entrypoint] WARNING: could not create $TUNED_MODEL, using base model as-is"
    fi
fi

export OLLAMA_MODELS="$APP_OLLAMA_MODELS"

# Load the agent model now so the first user request doesn't pay the ~75s
# llama-server startup on top of its own inference time.
FIRST_MODEL="$(echo "$APP_OLLAMA_MODELS" | cut -d, -f1 | xargs)"
if [ -n "$FIRST_MODEL" ]; then
    echo "[entrypoint] pre-loading $FIRST_MODEL into memory..."
    (curl -sf -m 600 "${OLLAMA_URL}/api/generate" \
        -d "{\"model\":\"$FIRST_MODEL\",\"prompt\":\"hi\",\"stream\":false,\"keep_alive\":-1}" \
        >/dev/null 2>&1 && echo "[entrypoint] agent model warm" \
        || echo "[entrypoint] WARNING: warmup request failed (model loads on first use)") &
fi

echo "[entrypoint] starting Texel Studio server..."
python server.py &
APP_PID=$!

trap 'kill $OLLAMA_PID $APP_PID 2>/dev/null' TERM INT
wait $APP_PID
