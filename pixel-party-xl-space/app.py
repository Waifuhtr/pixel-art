"""FastAPI front door for the Pixel Party XL Space.

Generation is queued onto a single worker thread. That is not a simplification
— the Space has one T4, and letting two requests into the pipeline at once
would race on the same weights and risk an OOM that kills both. One worker
also makes queue position and progress honest numbers rather than guesses.

Requests return a job id immediately and the browser polls, so a 90-second
batch never sits inside a single HTTP request waiting to be timed out by a
proxy somewhere between the browser and the container.
"""

from __future__ import annotations

import base64
import io
import os
import random
import threading
import time
import uuid
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field, model_validator

from pipeline import (
    DEFAULT_NEGATIVE,
    DEFAULT_SCHEDULER,
    SCHEDULERS,
    STYLE_SUFFIX,
    GenerationCancelled,
    PixelPartyEngine,
    pixelate,
    upscale_nearest,
)

STATIC_DIR = Path(__file__).parent / "static"

# SDXL's native aspect buckets. Every one is a multiple of 64, so all the
# offered pixel factors divide them exactly. Off-bucket sizes cost quality and
# VRAM for nothing, so the API refuses them outright.
SIZES: list[tuple[int, int]] = [
    (1024, 1024),
    (1152, 896),
    (896, 1152),
    (1216, 832),
    (832, 1216),
    (1344, 768),
    (768, 1344),
]
PIXEL_FACTORS = [1, 2, 4, 8, 16]

MAX_IMAGES = 4
MAX_QUEUE = 12
MAX_ACTIVE_PER_CLIENT = 2
MAX_INIT_BYTES = 8 * 1024 * 1024
MAX_SEED = 2**31 - 1

# Results live in RAM. Both caps matter: the byte budget stops a run of large
# canvases from eating the box's 15GB, the job cap stops a long tail of tiny
# jobs from growing the dict without bound.
RESULT_BYTE_BUDGET = 384 * 1024 * 1024
MAX_JOBS_KEPT = 40

engine = PixelPartyEngine()


# --------------------------------------------------------------------- models


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=800)
    negative_prompt: str = Field(default=DEFAULT_NEGATIVE, max_length=800)
    append_style: bool = True
    steps: int = Field(default=25, ge=8, le=50)
    guidance: float = Field(default=7.5, ge=1.0, le=15.0)
    width: int = 1024
    height: int = 1024
    num_images: int = Field(default=1, ge=1, le=MAX_IMAGES)
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
    scheduler: str = DEFAULT_SCHEDULER
    pixel_factor: int = 8
    palette_colors: int = Field(default=0, ge=0, le=64)
    init_image: str | None = None
    strength: float = Field(default=0.6, ge=0.1, le=0.95)

    @model_validator(mode="after")
    def _validate(self) -> "GenerateRequest":
        if (self.width, self.height) not in SIZES:
            raise ValueError(f"unsupported canvas {self.width}x{self.height}")
        if self.scheduler not in SCHEDULERS:
            raise ValueError(f"unknown scheduler {self.scheduler!r}")
        if self.pixel_factor not in PIXEL_FACTORS:
            raise ValueError(f"pixel_factor must be one of {PIXEL_FACTORS}")
        if self.palette_colors and self.palette_colors < 2:
            raise ValueError("palette_colors must be 0 (off) or at least 2")
        return self


@dataclass
class Job:
    id: str
    params: GenerateRequest
    client: str
    status: str = "queued"  # queued | running | done | error | cancelled
    progress: float = 0.0
    step: int = 0
    total_steps: int = 0
    images: list[dict] = field(default_factory=list)
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    cancel_requested: bool = False

    def public(self, queue_position: int | None) -> dict:
        eta = None
        if self.status == "running" and self.started and self.step > 0:
            per_step = (time.time() - self.started) / self.step
            eta = round(per_step * max(0, self.total_steps - self.step), 1)
        return {
            "id": self.id,
            "status": self.status,
            "progress": round(self.progress, 4),
            "step": self.step,
            "total_steps": self.total_steps,
            "queue_position": queue_position,
            # Copied, not referenced: the worker appends to this list while a
            # poll is being serialised.
            "images": list(self.images),
            "error": self.error,
            "eta_seconds": eta,
            "elapsed": round((self.finished or time.time()) - (self.started or self.created), 1),
        }


# ---------------------------------------------------------------- job storage


class JobStore:
    """Jobs and their rendered PNGs, evicted together so no blob is orphaned."""

    def __init__(self) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._blobs: dict[str, bytes] = {}
        self._bytes = 0
        self._lock = threading.Lock()

    def add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job
            self._evict()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def active_for(self, client: str) -> int:
        with self._lock:
            return sum(
                1
                for job in self._jobs.values()
                if job.client == client and job.status in ("queued", "running")
            )

    def put_blob(self, key: str, data: bytes) -> None:
        with self._lock:
            existing = self._blobs.get(key)
            if existing is not None:
                self._bytes -= len(existing)
            self._blobs[key] = data
            self._bytes += len(data)
            self._evict()

    def get_blob(self, key: str) -> bytes | None:
        with self._lock:
            return self._blobs.get(key)

    def _evict(self) -> None:
        """Drop oldest finished jobs until both budgets are satisfied.

        Unfinished jobs are skipped rather than stopping the sweep: their blobs
        are still being written and their poller is still watching, but a long
        run at the head of the dict must not block reclaiming everything behind
        it.
        """
        while self._bytes > RESULT_BYTE_BUDGET or len(self._jobs) > MAX_JOBS_KEPT:
            victim = next(
                (
                    job_id
                    for job_id, job in self._jobs.items()
                    if job.status not in ("queued", "running")
                ),
                None,
            )
            if victim is None:
                break
            self._jobs.pop(victim)
            for key in [k for k in self._blobs if k.startswith(f"{victim}:")]:
                self._bytes -= len(self._blobs.pop(key))


store = JobStore()
_pending: deque[str] = deque()
_queue_cv = threading.Condition()


def queue_position(job_id: str) -> int | None:
    with _queue_cv:
        try:
            return list(_pending).index(job_id) + 1
        except ValueError:
            return None


# ------------------------------------------------------------------- helpers


def client_key(request: Request) -> str:
    # Spaces sit behind a proxy, so the socket address is the proxy's. The
    # left-most X-Forwarded-For hop is the real caller.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def build_prompt(prompt: str, append_style: bool) -> str:
    """Apply the model card's required style suffix, without doubling it."""
    text = prompt.strip()
    if not append_style or "pixel art" in text.lower():
        return text
    return text.rstrip(" .,") + STYLE_SUFFIX


def decode_init_image(raw: str) -> Image.Image:
    payload = raw.strip()
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    try:
        data = base64.b64decode(payload, validate=True)
    except Exception as exc:  # noqa: BLE001 - any decode failure is a bad request
        raise HTTPException(status_code=400, detail="init_image is not valid base64") from exc
    if len(data) > MAX_INIT_BYTES:
        raise HTTPException(status_code=400, detail="init_image is larger than 8MB")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:  # noqa: BLE001 - unreadable/hostile image data
        raise HTTPException(status_code=400, detail="init_image could not be decoded") from exc
    return image


def encode_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


# -------------------------------------------------------------------- worker


def run_job(job: Job) -> None:
    params = job.params
    prompt = build_prompt(params.prompt, params.append_style)
    init_image = decode_init_image(params.init_image) if params.init_image else None

    job.status = "running"
    job.started = time.time()
    job.total_steps = params.num_images * params.steps
    completed = 0

    for index in range(params.num_images):
        if job.cancel_requested:
            break

        seed = (
            (params.seed + index) % (MAX_SEED + 1)
            if params.seed is not None
            else random.randint(0, MAX_SEED)
        )

        def on_step(step: int, per_image_total: int, _completed: int = completed) -> None:
            # per_image_total comes from the pipeline itself, because img2img
            # runs int(steps * strength) steps rather than the requested steps.
            job.total_steps = per_image_total * params.num_images
            job.step = _completed * per_image_total + step
            job.progress = min(1.0, job.step / max(1, job.total_steps))

        image = engine.generate(
            prompt=prompt,
            negative_prompt=params.negative_prompt,
            steps=params.steps,
            guidance=params.guidance,
            width=params.width,
            height=params.height,
            seed=seed,
            scheduler=params.scheduler,
            init_image=init_image,
            strength=params.strength,
            on_step=on_step,
            should_cancel=lambda: job.cancel_requested,
        )

        sprite = pixelate(image, params.pixel_factor, params.palette_colors)
        store.put_blob(f"{job.id}:{index}:raw", encode_png(image))
        store.put_blob(f"{job.id}:{index}:pixel", encode_png(sprite))

        job.images.append(
            {
                "index": index,
                "seed": seed,
                "width": image.width,
                "height": image.height,
                "pixel_width": sprite.width,
                "pixel_height": sprite.height,
                "raw_url": f"/api/image/{job.id}/{index}/raw",
                "pixel_url": f"/api/image/{job.id}/{index}/pixel",
            }
        )
        completed += 1

    if job.cancel_requested and completed < params.num_images:
        job.status = "cancelled"
    else:
        job.status = "done"
        job.progress = 1.0


def worker_loop() -> None:
    while True:
        with _queue_cv:
            while not _pending:
                _queue_cv.wait()
            job_id = _pending.popleft()

        job = store.get(job_id)
        if job is None or job.cancel_requested:
            if job is not None:
                job.status = "cancelled"
            continue

        try:
            run_job(job)
        except GenerationCancelled:
            job.status = "cancelled"
        except HTTPException as exc:
            job.status = "error"
            job.error = str(exc.detail)
        except Exception as exc:  # noqa: BLE001 - never let the worker die
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            print(f"[worker] job {job.id} failed: {job.error}", flush=True)
        finally:
            job.finished = time.time()


# ----------------------------------------------------------------------- app


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=engine.load, name="engine-load", daemon=True).start()
    threading.Thread(target=worker_loop, name="worker", daemon=True).start()
    yield


app = FastAPI(title="Pixel Party XL", lifespan=lifespan, docs_url=None, redoc_url=None)


@app.get("/api/config")
def get_config() -> dict:
    return {
        "sizes": [{"width": w, "height": h} for w, h in SIZES],
        "pixel_factors": PIXEL_FACTORS,
        "schedulers": [{"id": key, "label": value[0]} for key, value in SCHEDULERS.items()],
        "defaults": {
            "negative_prompt": DEFAULT_NEGATIVE,
            "steps": 25,
            "guidance": 7.5,
            "width": 1024,
            "height": 1024,
            "scheduler": DEFAULT_SCHEDULER,
            "pixel_factor": 8,
            "strength": 0.6,
        },
        "limits": {
            "max_images": MAX_IMAGES,
            "max_steps": 50,
            "max_queue": MAX_QUEUE,
        },
        "style_suffix": STYLE_SUFFIX,
    }


@app.get("/api/health")
def get_health() -> dict:
    snapshot = engine.snapshot()
    with _queue_cv:
        snapshot["queued"] = len(_pending)
    return snapshot


@app.post("/api/generate")
def post_generate(payload: GenerateRequest, request: Request) -> dict:
    snapshot = engine.snapshot()
    if snapshot["status"] == "error":
        raise HTTPException(status_code=503, detail=f"Model failed to load: {engine.error}")
    if not snapshot["ready"]:
        raise HTTPException(status_code=503, detail="Model is still loading, try again shortly.")

    with _queue_cv:
        if len(_pending) >= MAX_QUEUE:
            raise HTTPException(status_code=429, detail="Queue is full, try again in a moment.")

    caller = client_key(request)
    if store.active_for(caller) >= MAX_ACTIVE_PER_CLIENT:
        raise HTTPException(
            status_code=429,
            detail=f"You already have {MAX_ACTIVE_PER_CLIENT} jobs in flight.",
        )

    # Decode here rather than in the worker so a malformed upload fails fast
    # with a 400 the browser can show, instead of a job that dies later.
    if payload.init_image:
        decode_init_image(payload.init_image)

    job = Job(id=uuid.uuid4().hex[:12], params=payload, client=caller)
    store.add(job)
    with _queue_cv:
        _pending.append(job.id)
        _queue_cv.notify()

    return {"job_id": job.id, "queue_position": queue_position(job.id)}


@app.get("/api/job/{job_id}")
def get_job(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown or expired job.")
    return job.public(queue_position(job_id))


@app.post("/api/job/{job_id}/cancel")
def post_cancel(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown or expired job.")
    if job.status in ("queued", "running"):
        job.cancel_requested = True
    return {"ok": True, "status": job.status}


@app.get("/api/image/{job_id}/{index}/{kind}")
def get_image(job_id: str, index: int, kind: str, scale: int = 1, download: int = 0) -> Response:
    if kind not in ("raw", "pixel"):
        raise HTTPException(status_code=404, detail="Unknown image kind.")
    data = store.get_blob(f"{job_id}:{index}:{kind}")
    if data is None:
        raise HTTPException(status_code=404, detail="Image expired or never existed.")

    scale = max(1, min(16, scale))
    if scale > 1:
        image = upscale_nearest(Image.open(io.BytesIO(data)), scale)
        data = encode_png(image)

    headers = {
        # Blob keys are unique per job, so a hit can never be stale.
        "Cache-Control": "public, max-age=31536000, immutable",
    }
    if download:
        suffix = f"-{kind}{f'-{scale}x' if scale > 1 else ''}"
        headers["Content-Disposition"] = (
            f'attachment; filename="pixel-party-{job_id}-{index}{suffix}.png"'
        )
    return Response(content=data, media_type="image/png", headers=headers)


# Mounted last: StaticFiles at "/" would otherwise shadow every /api route.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        log_level="info",
    )
