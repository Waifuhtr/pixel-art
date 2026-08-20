"""Pre-fetch every weight the Space needs, at image build time.

pixel-party-xl is a UNet-only repo — it ships a replacement denoiser for SDXL
and nothing else. A working pipeline therefore needs three repos:

    pixelparty/pixel-party-xl          the fine-tuned UNet          (5.14 GB)
    stabilityai/stable-diffusion-xl-base-1.0
                                       text encoders + tokenizers
                                       + scheduler config           (1.64 GB)
    madebyollin/sdxl-vae-fp16-fix      fp16-safe VAE                (0.33 GB)

Baking them into the image costs ~7GB of layer, and buys a cold start of
roughly 40s instead of the ~5min it takes to pull 7GB from the Hub. On billed
GPU hardware that download would be charged every single time the Space wakes
from sleep, so the trade is strongly worth it.

Run as part of `docker build`; app.py never downloads anything at runtime.
"""

from __future__ import annotations

import sys
import time

from huggingface_hub import snapshot_download

BASE_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
UNET_REPO = "pixelparty/pixel-party-xl"
VAE_REPO = "madebyollin/sdxl-vae-fp16-fix"

# Only the pieces the pipeline actually assembles.
#
# The base repo's own unet/ (5.1 GB) and vae/ (167 MB) are deliberately NOT
# listed: pixel-party-xl replaces the UNet and sdxl-vae-fp16-fix replaces the
# VAE, so fetching them would add 5.3 GB to the image for files nothing opens.
# Same for the flax / onnx / openvino copies and the single-file
# sd_xl_base_1.0.safetensors checkpoint — all dead weight here.
TARGETS: list[tuple[str, list[str]]] = [
    (
        BASE_REPO,
        [
            "model_index.json",
            "scheduler/scheduler_config.json",
            "tokenizer/*.json",
            "tokenizer/merges.txt",
            "tokenizer_2/*.json",
            "tokenizer_2/merges.txt",
            "text_encoder/config.json",
            "text_encoder/model.fp16.safetensors",
            "text_encoder_2/config.json",
            "text_encoder_2/model.fp16.safetensors",
        ],
    ),
    (UNET_REPO, ["config.json", "diffusion_pytorch_model.safetensors"]),
    (VAE_REPO, ["config.json", "diffusion_pytorch_model.safetensors"]),
]

MAX_ATTEMPTS = 4


def fetch(repo_id: str, allow_patterns: list[str]) -> None:
    """Download one repo's subset, retrying on the Hub's transient failures."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(f"[download] {repo_id} (attempt {attempt}/{MAX_ATTEMPTS})", flush=True)
            # No local_dir: this lands in the standard HF_HOME cache layout,
            # which is exactly what from_pretrained() looks in at runtime.
            path = snapshot_download(
                repo_id=repo_id,
                allow_patterns=allow_patterns,
                max_workers=4,
            )
            print(f"[download] ok -> {path}", flush=True)
            return
        except Exception as exc:  # noqa: BLE001 - build step: report and retry
            print(f"[download] failed: {type(exc).__name__}: {exc}", flush=True)
            if attempt == MAX_ATTEMPTS:
                raise
            backoff = 2**attempt
            print(f"[download] retrying in {backoff}s", flush=True)
            time.sleep(backoff)


def main() -> int:
    started = time.time()
    for repo_id, patterns in TARGETS:
        fetch(repo_id, patterns)
    print(f"[download] all weights cached in {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
