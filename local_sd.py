"""
Local Stable Diffusion (GGUF, CPU) — replaces Gemini for the concept-art
reference-image step.

Uses stable-diffusion.cpp via the `stable-diffusion-cpp-python` bindings to
run a quantized GGUF checkpoint (default: SD-Turbo Q8_0) fully on CPU, so the
"Generate concept art" step needs no API key and no GPU.

Model file is fetched from the Hugging Face Hub on first use and cached under
SD_MODEL_DIR (default ./models) so later generations skip the download. The
Dockerfile pre-fetches it at build time, so in the shipped image it is already
on disk and this module never downloads anything at runtime.

This does NOT replace the pixel-painting agent's LLM (agent.py / OLLAMA_MODELS
etc.) — a text-to-image diffusion model cannot make tool calls. It only
produces the one-shot reference image the agent paints from.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path


def _available_cpus() -> int:
    """CPU count the container may actually use.

    os.cpu_count() reports the HOST's cores, not the container's cgroup quota —
    on a Hugging Face Space it returns 64 while the Space is capped at 8 vCPU.
    Handing 64 threads to a 8-vCPU container makes ggml oversubscribe badly and
    every stage crawls (observed: 34s just to run CLIP text encoding). Read the
    cgroup quota first and only fall back to the raw core count.
    """
    # cgroup v2
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except Exception:
        pass
    # cgroup v1
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0 and period > 0:
            return max(1, quota // period)
    except Exception:
        pass
    # No quota set — fall back to the affinity mask, then the raw count.
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return max(1, os.cpu_count() or 4)


SD_GGUF_REPO = os.getenv("SD_GGUF_REPO", "gpustack/stable-diffusion-v2-1-turbo-GGUF")
SD_GGUF_FILE = os.getenv("SD_GGUF_FILE", "stable-diffusion-v2-1-turbo-Q8_0.gguf")
SD_MODEL_DIR = Path(os.getenv("SD_MODEL_DIR", str(Path(__file__).parent / "models")))
SD_THREADS = int(os.getenv("SD_THREADS", "0")) or _available_cpus()
SD_STEPS = int(os.getenv("SD_STEPS", "4"))            # SD-Turbo: sane range is 1-4
SD_CFG_SCALE = float(os.getenv("SD_CFG_SCALE", "1.0")) # Turbo checkpoints want ~0 guidance
SD_SIZE = int(os.getenv("SD_SIZE", "512"))             # SD 2.x degrades badly below 512

_model = None
# One lock guards BOTH loading and generating. stable-diffusion.cpp holds mutable
# state on the context, so two concurrent generate_image calls on one instance
# are unsafe — and they will happen, because FastAPI runs `def` endpoints in a
# threadpool and the user can click "generate" twice.
_lock = threading.RLock()


def enabled() -> bool:
    """True when local SD should be used instead of Gemini for reference images.

    Defaults to True — this fork exists to run without any API key, so an
    operator who forgets to set USE_LOCAL_SD in the Space's env vars should
    still get the local/free path, not a "No Gemini credentials" crash.
    Set USE_LOCAL_SD=false explicitly to opt back into Gemini.
    """
    return os.getenv("USE_LOCAL_SD", "true").lower() not in ("0", "false", "no")


def _ensure_weights() -> str:
    from huggingface_hub import hf_hub_download

    SD_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return hf_hub_download(
        repo_id=SD_GGUF_REPO,
        filename=SD_GGUF_FILE,
        local_dir=str(SD_MODEL_DIR),
    )


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from stable_diffusion_cpp import StableDiffusion

                weights_path = _ensure_weights()
                print(f"[local_sd] loading {weights_path} with n_threads={SD_THREADS}", flush=True)
                _model = StableDiffusion(
                    model_path=weights_path,
                    n_threads=SD_THREADS,
                    wtype="default",
                )
    return _model


def warmup() -> None:
    """Load the model ahead of the first request.

    Loading an SD 2.x checkpoint runs a one-off v-parameterization probe that
    costs over a minute on CPU. Doing it during container startup keeps that
    cost off the first user-visible request. Safe to call from a daemon thread;
    failures are logged and swallowed so a broken warmup never blocks the API.
    """
    try:
        _get_model()
        print("[local_sd] model ready", flush=True)
    except Exception as e:  # noqa: BLE001 - warmup must never take the server down
        print(f"[local_sd] warmup failed (will retry on first request): {e}", flush=True)


def generate_reference_png(prompt: str, size: int | None = None) -> bytes:
    """Generate a PNG (as raw bytes) for the given prompt using the local GGUF model."""
    import io

    px = size or SD_SIZE
    with _lock:
        sd = _get_model()
        images = sd.generate_image(
            prompt=prompt,
            width=px,
            height=px,
            sample_steps=SD_STEPS,
            cfg_scale=SD_CFG_SCALE,
        )
    buf = io.BytesIO()
    images[0].save(buf, format="PNG")
    return buf.getvalue()
