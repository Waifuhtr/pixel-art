"""Model loading and image generation for Pixel Party XL.

pixel-party-xl publishes a UNet and nothing else, so the pipeline is assembled
by hand from three repos rather than loaded with a single from_pretrained():

    unet           <- pixelparty/pixel-party-xl          (fine-tuned denoiser)
    vae            <- madebyollin/sdxl-vae-fp16-fix      (required, see below)
    everything else<- stabilityai/stable-diffusion-xl-base-1.0

Assembling explicitly, instead of calling from_pretrained(BASE_REPO, unet=...),
is what lets download_models.py skip the base repo's own 5.1GB UNet: nothing
here ever asks for a component that was not fetched.

Two constraints come straight from the model card:
  * the stock SDXL VAE produces NaNs in fp16, so the fp16-fix VAE is mandatory
    on a T4 (which has no usable bf16 path)
  * the refiner must NOT be used
"""

from __future__ import annotations

import threading
import time

import torch
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from PIL import Image
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

BASE_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
UNET_REPO = "pixelparty/pixel-party-xl"
VAE_REPO = "madebyollin/sdxl-vae-fp16-fix"

# Model card: 'Append ". in pixel art style" to your prompt.'
STYLE_SUFFIX = ". in pixel art style"
DEFAULT_NEGATIVE = "mixels. amateur. multiple"

# Schedulers are rebuilt from the base scheduler's config every time rather
# than mutated in place, so switching back and forth cannot accumulate state.
SCHEDULERS: dict[str, tuple[str, object, dict]] = {
    "euler": ("Euler", EulerDiscreteScheduler, {}),
    "euler_a": ("Euler Ancestral", EulerAncestralDiscreteScheduler, {}),
    "dpmpp_2m": ("DPM++ 2M", DPMSolverMultistepScheduler, {"algorithm_type": "dpmsolver++"}),
    "dpmpp_2m_karras": (
        "DPM++ 2M Karras",
        DPMSolverMultistepScheduler,
        {"algorithm_type": "dpmsolver++", "use_karras_sigmas": True},
    ),
}
DEFAULT_SCHEDULER = "euler"


class GenerationCancelled(Exception):
    """Raised out of the step callback when a job is cancelled mid-run."""


def pixelate(image: Image.Image, factor: int, palette_colors: int = 0) -> Image.Image:
    """Downscale with nearest-neighbour into true pixel art.

    This is the step the model card calls for ("Downsize the image 8x using
    nearest neighbor"): the model paints pixel art at 1024px, where every
    intended pixel is an 8x8 block, and the downscale recovers the real
    128x128 sprite. Skipping it leaves a blurry upscale of the actual art.
    """
    width, height = image.size
    small = image.resize(
        (max(1, width // factor), max(1, height // factor)),
        Image.Resampling.NEAREST,
    )
    if palette_colors:
        small = (
            small.convert("RGB")
            .quantize(
                colors=palette_colors,
                method=Image.Quantize.MEDIANCUT,
                dither=Image.Dither.NONE,
            )
            .convert("RGB")
        )
    return small


def _fit_init_image(image: Image.Image, width: int, height: int) -> Image.Image:
    """Resize an init image to the target canvas, preserving pixel edges.

    A small source is almost always a sprite — often one of this Space's own
    128px results being fed back in — and LANCZOS would smear exactly the hard
    edges the model is being asked to keep. Anything large enough to be a photo
    or a render gets the smooth filter instead.
    """
    image = image.convert("RGB")
    is_sprite = max(image.size) <= 512
    resample = Image.Resampling.NEAREST if is_sprite else Image.Resampling.LANCZOS
    return image.resize((width, height), resample)


def upscale_nearest(image: Image.Image, scale: int) -> Image.Image:
    """Blow a sprite back up with hard edges, for viewing and downloading."""
    return image.resize(
        (image.width * scale, image.height * scale), Image.Resampling.NEAREST
    )


class PixelPartyEngine:
    """Owns the pipeline. One generation at a time — a T4 has one GPU."""

    def __init__(self) -> None:
        self.pipe: StableDiffusionXLPipeline | None = None
        self.img2img: StableDiffusionXLImg2ImgPipeline | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.status = "starting"
        self.error: str | None = None
        self.load_seconds: float | None = None
        self._scheduler_config: dict | None = None
        self._active_scheduler = DEFAULT_SCHEDULER
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- loading

    def load(self) -> None:
        """Assemble the pipeline and move it to the GPU.

        Called on a background thread so uvicorn can bind the port immediately;
        a Space that does not answer on its port quickly is treated as failed.
        """
        started = time.time()
        try:
            self.status = "loading"
            print(f"[engine] device={self.device} dtype={self.dtype}", flush=True)

            tokenizer = CLIPTokenizer.from_pretrained(BASE_REPO, subfolder="tokenizer")
            tokenizer_2 = CLIPTokenizer.from_pretrained(BASE_REPO, subfolder="tokenizer_2")
            text_encoder = CLIPTextModel.from_pretrained(
                BASE_REPO, subfolder="text_encoder", torch_dtype=self.dtype, variant="fp16"
            )
            text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                BASE_REPO, subfolder="text_encoder_2", torch_dtype=self.dtype, variant="fp16"
            )
            scheduler = EulerDiscreteScheduler.from_pretrained(
                BASE_REPO, subfolder="scheduler"
            )

            # No variant= here: pixel-party-xl ships one safetensors file that
            # is already fp16 (5.14GB for 2.57B params).
            unet = UNet2DConditionModel.from_pretrained(UNET_REPO, torch_dtype=self.dtype)
            vae = AutoencoderKL.from_pretrained(VAE_REPO, torch_dtype=self.dtype)

            pipe = StableDiffusionXLPipeline(
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                unet=unet,
                scheduler=scheduler,
                force_zeros_for_empty_prompt=True,
                # invisible-watermark is not installed; being explicit keeps
                # diffusers from probing for it.
                add_watermarker=False,
            )
            pipe.to(self.device)
            pipe.set_progress_bar_config(disable=True)
            # Cheap on batch-of-1, and it keeps VAE decode off the peak when a
            # large canvas lands next to a nearly full 16GB of VRAM.
            pipe.enable_vae_slicing()

            # img2img shares every weight with the txt2img pipeline — this
            # costs no extra VRAM, it is just a second set of call semantics.
            # The model card notes init images help this model a lot.
            self.img2img = StableDiffusionXLImg2ImgPipeline(
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                text_encoder_2=pipe.text_encoder_2,
                tokenizer=pipe.tokenizer,
                tokenizer_2=pipe.tokenizer_2,
                unet=pipe.unet,
                scheduler=pipe.scheduler,
                requires_aesthetics_score=False,
                force_zeros_for_empty_prompt=True,
                add_watermarker=False,
            )
            self.img2img.set_progress_bar_config(disable=True)

            self._scheduler_config = dict(pipe.scheduler.config)
            self.pipe = pipe
            self.load_seconds = time.time() - started
            self.status = "ready"
            print(f"[engine] ready in {self.load_seconds:.1f}s", flush=True)
            self._warmup()
        except Exception as exc:  # noqa: BLE001 - surfaced through /api/health
            self.error = f"{type(exc).__name__}: {exc}"
            self.status = "error"
            print(f"[engine] load failed: {self.error}", flush=True)
            raise

    def _warmup(self) -> None:
        """One throwaway step so the first real request pays no cuDNN autotune."""
        if self.device != "cuda":
            return
        try:
            started = time.time()
            self.generate(prompt="pixel art sword", steps=1, width=1024, height=1024)
            print(f"[engine] warmup done in {time.time() - started:.1f}s", flush=True)
        except Exception as exc:  # noqa: BLE001 - warmup is best effort only
            print(f"[engine] warmup skipped: {type(exc).__name__}: {exc}", flush=True)

    # ------------------------------------------------------------- generation

    def _scheduler_for(self, name: str):
        _label, cls, kwargs = SCHEDULERS.get(name, SCHEDULERS[DEFAULT_SCHEDULER])
        return cls.from_config(self._scheduler_config, **kwargs)

    def _apply_scheduler(self, name: str) -> None:
        if name == self._active_scheduler:
            return
        scheduler = self._scheduler_for(name)
        self.pipe.scheduler = scheduler
        self.img2img.scheduler = scheduler
        self._active_scheduler = name

    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str = DEFAULT_NEGATIVE,
        steps: int = 25,
        guidance: float = 7.5,
        width: int = 1024,
        height: int = 1024,
        seed: int | None = None,
        scheduler: str = DEFAULT_SCHEDULER,
        init_image: Image.Image | None = None,
        strength: float = 0.6,
        on_step=None,
        should_cancel=None,
    ) -> Image.Image:
        """Render one image. Callers loop for batches — see app.py for why."""
        if self.pipe is None:
            raise RuntimeError("model is still loading")

        with self._lock:
            self._apply_scheduler(scheduler)

            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(int(seed))

            def callback(_pipe, step_index, _timestep, callback_kwargs):
                if should_cancel is not None and should_cancel():
                    raise GenerationCancelled()
                if on_step is not None:
                    # num_timesteps is the count the pipeline actually runs,
                    # which for img2img is int(steps * strength), not steps.
                    total = getattr(_pipe, "num_timesteps", None) or steps
                    on_step(step_index + 1, total)
                return {}

            common = dict(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
                num_images_per_prompt=1,
                callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=["latents"],
            )

            try:
                if init_image is not None:
                    result = self.img2img(
                        image=_fit_init_image(init_image, width, height),
                        strength=strength,
                        **common,
                    )
                else:
                    result = self.pipe(width=width, height=height, **common)
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "GPU ran out of memory. Try a smaller canvas or fewer images."
                ) from exc

            return result.images[0]

    # ------------------------------------------------------------------ state

    def snapshot(self) -> dict:
        info: dict = {
            "status": self.status,
            "ready": self.status == "ready",
            "device": self.device,
            "error": self.error,
            "load_seconds": round(self.load_seconds, 1) if self.load_seconds else None,
        }
        if self.device == "cuda":
            try:
                info["gpu"] = torch.cuda.get_device_name(0)
                info["vram_total_gb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
                )
                info["vram_used_gb"] = round(torch.cuda.memory_allocated(0) / 1024**3, 1)
            except Exception:  # noqa: BLE001 - diagnostics only
                pass
        return info
