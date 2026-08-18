FROM python:3.11-slim-bookworm

# build-essential + cmake: stable-diffusion-cpp-python compiles stable-diffusion.cpp
# from source at pip-install time. curl: installs Ollama and is used by the
# entrypoint's readiness check. zstd: Ollama's own installer now ships its
# archive zstd-compressed and fails ("requires zstd for extraction") without it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl ca-certificates zstd procps \
    && rm -rf /var/lib/apt/lists/*

# Official installer — CPU-only host is fine, Ollama falls back to CPU automatically.
RUN curl -fsSL https://ollama.com/install.sh | sh

# Hugging Face Spaces runs containers as an arbitrary non-root UID by default;
# give that UID a writable home (Ollama's model store + HF hub cache both live
# under $HOME) instead of assuming UID 1000 like most Docker Space examples do.
ENV HOME=/home/user
RUN mkdir -p $HOME && chmod 777 $HOME
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# stable-diffusion-cpp-python compiles stable-diffusion.cpp (C++) from source —
# its own layer so: (1) -v actually shows compiler progress instead of pip's
# default silence during a multi-minute build (that silence is what looked
# like a hang), (2) a capped parallel level avoids the builder OOMing under
# several concurrent GCC jobs (that OOM is what actually causes a real hang),
# (3) GGML_NATIVE=OFF avoids -march=native, since the container that BUILDS
# this image can have different CPU features than the one that later RUNS
# it — a native build can crash with "Illegal instruction" at runtime.
ENV CMAKE_BUILD_PARALLEL_LEVEL=4
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF -DSD_CUDA=OFF -DSD_HIPBLAS=OFF -DSD_METAL=OFF -DSD_VULKAN=OFF -DSD_SYCL=OFF"
RUN pip install --no-cache-dir -v "stable-diffusion-cpp-python>=0.3.0"

# ── Model weights, baked into the image ───────────────────────────────────
#
# A Space's writable layer is NOT persisted across restarts unless paid
# persistent storage is attached, so anything downloaded at runtime is
# re-downloaded on every cold start (~4GB, several minutes before the first
# image appears). Fetching both models at BUILD time puts them in immutable
# image layers instead: every container start already has them on disk.
#
# These live above `COPY . .` on purpose — editing app code then rebuilds in
# seconds instead of re-downloading 4GB of weights.
#
# Set --build-arg PREFETCH_MODELS=0 to skip and download lazily at runtime
# (much smaller image, much slower first run).
ARG PREFETCH_MODELS=1

ENV SD_GGUF_REPO=gpustack/stable-diffusion-v2-1-turbo-GGUF
ENV SD_GGUF_FILE=stable-diffusion-v2-1-turbo-Q8_0.gguf
ENV SD_MODEL_DIR=/app/models
RUN if [ "$PREFETCH_MODELS" = "1" ]; then \
        python -c "import os; from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id=os.environ['SD_GGUF_REPO'], filename=os.environ['SD_GGUF_FILE'], local_dir=os.environ['SD_MODEL_DIR'])"; \
    fi

# NOTE: deliberately NOT named OLLAMA_MODELS. Ollama's own daemon reads that
# name to mean "directory to store models in", while this app uses it to mean
# "list of model names to expose". Setting it as a container-wide ENV would
# break `ollama serve`; entrypoint.sh bridges the two meanings.
ENV PIXELART_OLLAMA_MODELS=qwen2.5:3b
RUN if [ "$PREFETCH_MODELS" = "1" ]; then \
        ollama serve & \
        OLLAMA_PID=$! && \
        for i in $(seq 1 60); do curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break; sleep 1; done && \
        ollama pull "$PIXELART_OLLAMA_MODELS" && \
        kill "$OLLAMA_PID"; \
    fi

COPY . .
# static/ ships with only a placeholder page in git — the custom HTML/CSS/JS
# frontend is unzipped into it before the image is built.
# The 777s matter because Spaces picks the runtime UID, and both the SQLite DB
# and Ollama's model store must stay writable for whichever UID that is.
RUN mkdir -p static references output models && chmod -R 777 /app $HOME

# HF Spaces' Docker SDK expects the app to listen on this port (declared as
# `app_port` in README.md's Space metadata — keep the two in sync).
ENV PORT=7860
ENV OLLAMA_URL=http://localhost:11434
EXPOSE 7860

RUN chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
