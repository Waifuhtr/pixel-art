FROM python:3.11-slim-bookworm

# build-essential + cmake: stable-diffusion-cpp-python compiles stable-diffusion.cpp
# from source at pip-install time. curl: installs Ollama and is used by the
# entrypoint's readiness check. zstd: Ollama's own installer now ships its
# archive zstd-compressed and fails ("requires zstd for extraction") without it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl ca-certificates zstd \
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

COPY . .
# static/ is intentionally close to empty in this repo — the custom HTML/CSS/JS
# frontend (delivered separately) gets unzipped in here before/at build time.
RUN mkdir -p static references output models && chmod -R 777 /app

# HF Spaces' Docker SDK expects the app to listen on this port (declared as
# `app_port` in README.md's Space metadata — keep the two in sync).
ENV PORT=7860
ENV OLLAMA_URL=http://localhost:11434
EXPOSE 7860

RUN chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
