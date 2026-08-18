"""
Local Stable Diffusion (GGUF, CPU) — replaces Gemini for the concept-art
reference-image step.

Uses stable-diffusion.cpp via the `stable-diffusion-cpp-python` bindings to
run a quantized GGUF checkpoint (default: SD-Turbo Q8_0) fully on CPU, so the
"Generate concept art" step needs no API key and no GPU.

Model file is fetched from the Hugging Face Hub on first use and cached under
SD_MODEL_DIR (default ./models) so later generations skip the download.

This does NOT replace the pixel-painting agent's LLM (agent.py / OLLAMA_MODELS
etc.) — a text-to-image diffusion model cannot make tool calls. It only
produces the one-shot reference image the agent paints from.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

SD_GGUF_REPO = os.getenv("SD_GGUF_REPO", "gpustack/stable-diffusion-v2-1-turbo-GGUF")
SD_GGUF_FILE = os.getenv("SD_GGUF_FILE", "stable-diffusion-v2-1-turbo-Q8_0.gguf")
SD_MODEL_DIR = Path(os.getenv("SD_MODEL_DIR", str(Path(__file__).parent / "models")))
SD_THREADS = int(os.getenv("SD_THREADS", str(os.cpu_count() or 4)))
SD_STEPS = int(os.getenv("SD_STEPS", "4"))            # SD-Turbo: sane range is 1-4
SD_CFG_SCALE = float(os.getenv("SD_CFG_SCALE", "1.0")) # Turbo checkpoints want ~0 guidance

_model = None
_lock = threading.Lock()


def enabled() -> bool:
    """True when local SD should be used instead of Gemini for reference images."""
    return os.getenv("USE_LOCAL_SD", "").lower() in ("1", "true", "yes")


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
                _model = StableDiffusion(
                    model_path=weights_path,
                    n_threads=SD_THREADS,
                    wtype="default",
                )
    return _model


def generate_reference_png(prompt: str, size: int = 512) -> bytes:
    """Generate a PNG (as raw bytes) for the given prompt using the local GGUF model."""
    import io

    sd = _get_model()
    images = sd.generate_image(
        prompt=prompt,
        width=size,
        height=size,
        sample_steps=SD_STEPS,
        cfg_scale=SD_CFG_SCALE,
    )
    buf = io.BytesIO()
    images[0].save(buf, format="PNG")
    return buf.getvalue()
