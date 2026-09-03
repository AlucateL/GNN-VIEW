import os
from pathlib import Path

import torch
from fastapi import FastAPI
import logging
import re
import traceback

from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware

from services.seed_db import init_db
from routes.graph import router as graph_router
from routes.model import router as model_router
from routes.inference import router as inference_router, initialize_inference_runtime
from routes.integrated_gradients import router as integrated_gradients_router
from services.inference import load_timeline, PRESET_MODELS
from services.era5_window_manager import manager as era5_manager


logger = logging.getLogger(__name__)

origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")

cors_middleware = Middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# CORS must stay outermost so its headers land on every response, including ones
# gzip never touches (e.g. small/error responses below minimum_size).
# compresslevel=1: Starlette defaults to 9, which costs ~255 ms of event-loop time
# on a ~950 KB payload for ~10% fewer bytes than level 1 (~11 ms). Compression runs
# in the ASGI send path, so that time blocks every other in-flight request.
gzip_middleware = Middleware(GZipMiddleware, minimum_size=1024, compresslevel=1)

app = FastAPI(
    title="Verification and Interpretation of GNN-Based Weather Models Backend API",
    version="1.0.0",
    middleware=[cors_middleware, gzip_middleware],
)

# Unhandled exceptions otherwise return a bare "Internal Server Error" with no body, and
# because a 500 escapes before CORSMiddleware can attach its headers, the browser reports it
# as an origin failure -- so the actual fault is invisible from both the client and the
# console, and only reachable through kubectl on a pod whose logs the health probes flood.
#
# Return the exception type and message instead. The traceback stays server-side unless
# DEBUG_ERRORS is set, so the response is useful without publishing internals by default.
# Absolute paths leak the identity of whoever built an artefact, and TorchScript is the
# reason this is not hypothetical: it records the source paths of the machine that traced a
# model and replays them in every traceback, so an ocean-model error carries the exporter's
# home directory into the response body -- on a site with no authentication, during
# double-blind review. Redact rather than drop the message: "CUDA out of memory. Tried to
# allocate 2.16 GiB" is what diagnoses anything; the paths never helped.
# Greedy to the LAST slash, so only the directory goes: ".../interaction_net.py(320): message"
# keeps the file, the line and the frame, which is what actually locates a fault. Covers the
# roots that carry an identity -- /Users and /home (developer machines), /scratch (the HPC
# project path baked into the checkpoint) and /builds (the CI checkout).
_PATH_RE = re.compile(r"""/(?:Users|home|scratch|builds)/[^\s"']*/""")


def _redact(text: str) -> str:
    """Strip directory prefixes from absolute paths, keeping basename and line number."""
    return _PATH_RE.sub(".../", text)


# HTTPException is handled by FastAPI's own handler, not the Exception one below, so the
# ~12 call sites that do `detail=str(e)` would return an exception message unredacted --
# including _get_or_load_model, where `e` is a TorchScript error carrying export paths.
# Wrapping the default handler covers every one of them without touching the call sites.
@app.exception_handler(StarletteHTTPException)
async def redacting_http_exception_handler(request: Request, exc: StarletteHTTPException):
    if isinstance(exc.detail, str):
        exc = StarletteHTTPException(
            status_code=exc.status_code,
            detail=_redact(exc.detail),
            headers=getattr(exc, "headers", None),
        )
    return await http_exception_handler(request, exc)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    logger.error("Unhandled %s on %s %s\n%s", type(exc).__name__, request.method, request.url.path, tb)
    body = {
        "detail": "Internal Server Error",
        "error": _redact(f"{type(exc).__name__}: {exc}"),
        "path": request.url.path,
    }
    if os.getenv("DEBUG_ERRORS", "").strip().lower() in ("1", "true", "yes", "on"):
        body["traceback"] = [_redact(l) for l in tb.splitlines()[-40:]]
    # CORSMiddleware never sees a response raised this deep, so set the header here or the
    # browser hides the payload we just went to the trouble of producing.
    origin = request.headers.get("origin")
    headers = {"Access-Control-Allow-Origin": origin} if origin else {}
    return JSONResponse(status_code=500, content=body, headers=headers)


# Initialize database on startup
@app.on_event("startup")
def startup():
    # §2k diagnostic: torch sizes its CPU thread pool from hardware_concurrency(),
    # which does not read the container's cgroup quota. Zero behaviour change —
    # just logs whether torch_threads is oversized relative to the cgroup limit.
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    print(f"[startup] torch_threads={torch.get_num_threads()} cpu_count={os.cpu_count()} "
          f"cpu.max={cpu_max.read_text().strip() if cpu_max.exists() else 'n/a'}", flush=True)

    init_db()
    initialize_inference_runtime()
    try:
        default_model_path = next(iter(PRESET_MODELS.values()))
        timeline_path = default_model_path.parent / f"{default_model_path.stem}_timeline_errors.npy"
        era5_manager.initialize(load_timeline(timeline_path))
    except Exception as e:
        print(f"[startup] ERA5WindowManager: load_timeline() failed — manager not initialized: {e}")

# app.include_router(dummy_router)
app.include_router(graph_router)
app.include_router(model_router)
app.include_router(inference_router)
app.include_router(integrated_gradients_router)


@app.get("/health", tags=["Monitoring"])
async def health():
    """
    Liveness plus the runtime facts that are otherwise invisible from outside the pod.

    `device`/`ocean_device` are the ones worth watching: torch falls back to CPU silently
    when no GPU is attached, so a cluster that was allocated one but never requests it in
    the pod spec looks identical to a CPU-only deployment — except that a forward pass takes
    ~30 s instead of ~1 s. Reading this over HTTP avoids needing cluster credentials.
    """
    import torch  # noqa: PLC0415

    from routes.inference import OCEAN_DEVICE, VRAM_LIMIT_MIB  # noqa: PLC0415
    from services.inference import DEVICE  # noqa: PLC0415

    # Container memory, read from the cgroup rather than the host: inside a pod, psutil and
    # /proc/meminfo report the NODE's memory, not the limit the container is held to, so they
    # cannot tell you whether a forward pass will fit. "max" means no limit is set.
    def _cgroup(name: str) -> str:
        for base in ("/sys/fs/cgroup", "/sys/fs/cgroup/memory"):
            try:
                return Path(f"{base}/{name}").read_text().strip()
            except OSError:
                continue
        return "unavailable"

    limit = _cgroup("memory.max")
    if limit == "unavailable":
        limit = _cgroup("memory.limit_in_bytes")
    current = _cgroup("memory.current")
    if current == "unavailable":
        current = _cgroup("memory.usage_in_bytes")

    def _gb(v: str) -> str:
        try:
            return f"{int(v) / 1_073_741_824:.2f} GB"
        except ValueError:
            return v          # "max" (no limit) or "unavailable"

    return {
        "status": "ok",
        "memory_limit": _gb(limit),
        "memory_in_use": _gb(current),
        "device": DEVICE,
        "ocean_device": OCEAN_DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        # Reported so the VRAM ceiling can be confirmed from outside the cluster: the shared-GPU
        # reservation in helm/values.yaml is a promise, and this is the evidence we keep it.
        "vram_limit_mib": VRAM_LIMIT_MIB if torch.cuda.is_available() else None,
        "cuda_devices": [
            {
                "name": torch.cuda.get_device_properties(i).name,
                "total_mib": round(torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)),
                "allocated_mib": round(torch.cuda.memory_allocated(i) / (1024 ** 2)),
                "reserved_mib": round(torch.cuda.memory_reserved(i) / (1024 ** 2)),
                # High-water marks since process start. allocated_mib/reserved_mib are
                # sampled between requests and so never show what a pass actually costs --
                # an ocean attribution peaks for under a second and frees on the way out.
                # These are the only figures that reveal the real requirement, and they are
                # what a successful pass leaves behind.
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated(i) / (1024 ** 2)),
                "peak_reserved_mib": round(torch.cuda.max_memory_reserved(i) / (1024 ** 2)),
            }
            for i in range(torch.cuda.device_count())
        ],
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }
