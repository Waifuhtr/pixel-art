# Pixel Party XL — Hugging Face Space (Docker SDK, Nvidia T4 small)
#
# Deliberately built on python:3.10-slim rather than an nvidia/cuda base image:
# the PyPI torch wheel already ships every CUDA 12.1 runtime library it needs
# as nvidia-*-cu12 dependencies, so the CUDA base image would only duplicate
# ~3GB of the same .so files. The host driver comes from the Space hardware.

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# libgomp1 is the OpenMP runtime torch's CPU-side kernels link against, and it
# is not in python:3.10-slim. Nothing else is needed — this stack has no
# opencv, so no libGL/libglib.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Torch gets its own layer: it is by far the slowest thing to install and it
# never changes when application code does, so every later rebuild reuses it.
RUN pip install --no-cache-dir torch==2.4.1

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Spaces run containers as UID 1000. Anything written at runtime — the HF
# cache, matplotlib-style dotfiles — has to live under a directory that user
# owns, or the process dies on a permission error at startup.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PORT=7860

WORKDIR $HOME/app

# Weights are baked in. Copied on its own first so that editing app code or
# the frontend does not invalidate the 7GB download layer underneath it.
COPY --chown=user:user download_models.py ./
RUN python download_models.py

COPY --chown=user:user . ./

EXPOSE 7860

# -u so uvicorn's startup logs reach the Space's log panel unbuffered; without
# it a slow model load looks like a hung container.
CMD ["python", "-u", "app.py"]
