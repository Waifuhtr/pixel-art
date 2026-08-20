---
title: Pixel Party XL
emoji: 👾
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
license: creativeml-openrail-m
short_description: Pixel-art generation with Pixel Party XL on SDXL
suggested_hardware: t4-small
models:
  - pixelparty/pixel-party-xl
  - stabilityai/stable-diffusion-xl-base-1.0
  - madebyollin/sdxl-vae-fp16-fix
---

# Pixel Party XL

Text-to-image pixel art, served by a custom HTML/CSS/JS frontend over a
FastAPI backend. Built for **Nvidia T4 small** (4 vCPU · 15 GB RAM · 16 GB VRAM).

## How the model is assembled

`pixelparty/pixel-party-xl` is a **UNet-only repo** — it publishes a replacement
denoiser for SDXL and nothing else. A working pipeline is therefore built from
three sources:

| Component | Source | Size |
|---|---|---|
| UNet | `pixelparty/pixel-party-xl` | 5.14 GB (already fp16) |
| VAE | `madebyollin/sdxl-vae-fp16-fix` | 0.33 GB |
| Text encoders, tokenizers, scheduler | `stabilityai/stable-diffusion-xl-base-1.0` | 1.64 GB |

The base repo's own UNet and VAE are deliberately never downloaded — both are
replaced, so pulling them would add 5.3 GB to the image for files nothing opens.

Two rules from the model card are enforced in code rather than left to the user:

- **The fp16-fix VAE is mandatory.** The stock SDXL VAE emits NaNs in fp16, and
  a T4 (Turing) has no usable bf16 path, so fp16 is the only sensible dtype.
- **No refiner.** The model card says not to use one, so none is loaded.

## The downscale step

The model paints pixel art at 1024px, where every intended pixel is an 8×8
block. The model card's instruction — *"downsize the image 8x using nearest
neighbor"* — is what recovers the real 128×128 sprite, so the backend does it
automatically and returns both the sprite and the 1024px render. Skipping this
leaves you with a blurry upscale of the actual art.

## Features

- Automatic `. in pixel art style` suffix (the model card's instance prompt)
- 7 SDXL-native aspect buckets, all multiples of 64
- Downscale factors 1×–16×, with the resulting sprite size shown inline
- Optional palette quantization (8–64 colours, no dithering)
- img2img with an init image — drop, paste, or feed a previous result back in
- 4 samplers, seed locking, 1–4 images per run
- Live progress with real step counts, queue position, ETA, and cancel

## Runtime behaviour

Generation runs on a **single worker thread**. The Space has one GPU, and
letting two requests into the pipeline at once would race on the same weights
and risk an OOM that kills both. Requests return a job id immediately and the
browser polls, so a long batch never sits inside one HTTP request waiting to be
timed out by a proxy.

Weights are baked into the image at build time. A cold start then costs ~40s of
GPU time instead of the ~5 min it takes to pull 7 GB from the Hub — which on
billed hardware would otherwise be charged every time the Space wakes from
sleep.

Guardrails, since GPU time is metered: max 12 queued jobs, max 2 in flight per
caller, results capped at 384 MB / 40 jobs in RAM, and cancel actually aborts
the denoising loop rather than just hiding the result.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/config` | Canvases, samplers, factors, defaults, limits |
| `GET /api/health` | Load state, GPU name, VRAM, queue depth |
| `POST /api/generate` | Enqueue a job, returns `job_id` |
| `GET /api/job/{id}` | Status, progress, ETA, results |
| `POST /api/job/{id}/cancel` | Abort a queued or running job |
| `GET /api/image/{id}/{n}/{raw\|pixel}` | PNG, with `?scale=` and `?download=` |

## A note on hosting

The model card asks: *"please do not host this model"*, while licensing it under
CreativeML-OpenRAIL-M, which does permit redistribution. That is a request from
the authors ([PixelLab](https://www.pixellab.ai)) rather than a licence term —
worth knowing before making a Space public. Running it privately, or supporting
their work if you publish, respects the intent.

## Local run

```bash
docker build -t pixel-party-xl .
docker run --gpus all -p 7860:7860 pixel-party-xl
```

Without a GPU the app still starts and serves the UI, but generation on CPU is
impractically slow for SDXL (minutes per image rather than seconds).
