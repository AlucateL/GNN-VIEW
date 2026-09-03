from __future__ import annotations

import asyncio
import torch
import os
import pickle
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from services.bbox_torch import bbox_node_mask
from services.crud import GraphCRUD
from services.database import get_db
from services.era5_window_manager import manager
from services.graph import load_graph
from services.inference import DEVICE
from services.integrated_gradients import IGCancelled, compute_integrated_gradients, compute_input_x_gradients
from services.json_response import ORJSONResponse
from routes import inference as inference_route  # reuse _model initialized there

router = APIRouter(prefix="/integrated_gradients", tags=["IntegratedGradients"], default_response_class=ORJSONResponse)


# Riemann-sum steps for IG. Ignored when method = "ixg".
# Hardcoded server-side: client cannot vary this, which keeps the cache safe by
# construction (no risk of returning a steps=4 cached result to a steps=32 caller).
# At k=4 the completeness ratio (Σ attributions / Δprediction) is ~94% vs the
# steps=64 reference — accepted as the latency/fidelity trade-off on CPU.
# Measured, not guessed: completeness error is 0.009% at 4 steps and does not improve with
# more — the residual is the float32 rounding floor, not Riemann error, and cosine similarity
# against a 64-step reference is 1.00000. See IG_STEPS.md. Raising it is not memory-limited
# (steps run sequentially at batch_size=1); there is simply nothing to gain.
_IG_STEPS = 4

# Same (target, pred_time) tuple → reuse the full attribution tensor instead of re-running
# IG. The autograd pass produces all features in one shot; `source` query filtering just
# slices the cached result. Stored as float32 numpy arrays — ~19 MB per entry instead of
# ~400 MB if we kept the JSON-shaped Python lists.
_IG_CACHE_MAX = 8
_IG_CACHE: "OrderedDict[tuple, dict[str, Any]]" = OrderedDict()
_IG_CACHE_LOCK = threading.Lock()

# One attribution pass at a time. A single pass peaks near 14 GiB; the pool was later observed
# at 20,258 MiB against a 20,480 cap, after which even isolated requests failed because the
# allocator stayed fragmented. The drawer fires Panel A and Panel B as separate requests in
# Variables compare mode, so overlap is routine rather than exceptional. Queueing turns a 500
# into a slower response, which is the right trade for a deliberate user action. Wraps only the
# compute -- cache hits never acquire it.
_ATTR_COMPUTE_LOCK = asyncio.Semaphore(1)

# Per-key async locks so concurrent requests for the same (method, pred_time, target...)
# dedup onto a single compute instead of racing two forward(+backward) passes. Requests
# for different keys are unaffected. Only ever touched from the event loop thread, so no
# extra guard around the dict itself is needed.
_IG_INFLIGHT: dict[tuple, asyncio.Lock] = {}

# Disk cache: speeds up repeated IG queries within a single container lifetime. Cache files
# live in /tmp/ig_cache and are cleared automatically on container restart — they are not
# persisted across deploys. (Any legacy .pkl files in backend/data/ig_cache/ from the old
# DATA_ROOT-backed location can be manually removed; they are no longer read or written.)
# Cache key includes _IG_STEPS so changing the step count doesn't return stale entries.
_IG_DISK_CACHE_DIR = Path("/tmp/ig_cache")
_IG_DISK_CACHE_MAX = 64  # ~64 × 19 MB ≈ 1.2 GB on disk; oldest pruned by mtime
_IG_DISK_CACHE_LOCK = threading.Lock()

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _disk_cache_path(model_id: str, method: str, pred_time: str, target_var: str, level: int | None, target_node: int | None) -> Path:
    safe_model = _SAFE_NAME_RE.sub("_", str(model_id))
    safe_pt   = _SAFE_NAME_RE.sub("", str(pred_time))
    safe_var  = _SAFE_NAME_RE.sub("_", str(target_var))
    lv_part   = "x" if level is None else str(level)
    node_part = "all" if target_node is None else str(target_node)
    method_tag = f"ig_k{_IG_STEPS}" if method == "ig" else "ixg"
    return _IG_DISK_CACHE_DIR / f"m{safe_model}_{method_tag}_{safe_pt}_{safe_var}_lv{lv_part}_n{node_part}.pkl"


def _disk_cache_load(model_id: str, method: str, pred_time: str, target_var: str, level: int | None, target_node: int | None) -> dict[str, Any] | None:
    p = _disk_cache_path(model_id, method, pred_time, target_var, level, target_node)
    if not p.exists():
        return None
    try:
        with p.open("rb") as fh:
            return pickle.load(fh)
    except Exception as e:
        # A corrupt or partial file shouldn't take the endpoint down — log and recompute.
        print(f"[ig-cache] failed to load {p.name}: {e}")
        return None


def _disk_cache_save(model_id: str, method: str, pred_time: str, target_var: str, level: int | None, target_node: int | None, value: dict[str, Any]) -> None:
    """Fire-and-forget disk save. Pickling ~12 MB and writing through Docker's bind-mount FS
    (virtiofs/gRPC FUSE on macOS) can take several seconds — running it on a daemon thread
    keeps the request-response path off that latency. Worst case if the process exits
    mid-write: a `.tmp` orphan that's safely ignored, and the entry recomputes next time."""
    t = threading.Thread(target=_disk_cache_save_sync, args=(model_id, method, pred_time, target_var, level, target_node, value), daemon=True)
    t.start()


def _disk_cache_save_sync(model_id: str, method: str, pred_time: str, target_var: str, level: int | None, target_node: int | None, value: dict[str, Any]) -> None:
    with _IG_DISK_CACHE_LOCK:
        try:
            _IG_DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            p = _disk_cache_path(model_id, method, pred_time, target_var, level, target_node)
            tmp = p.with_suffix(p.suffix + ".tmp")
            with tmp.open("wb") as fh:
                pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(p)  # atomic on POSIX — readers never see a half-written file
            _disk_cache_prune()
        except Exception as e:
            print(f"[ig-cache] failed to save {method} {pred_time} {target_var}: {e}")


def _disk_cache_prune() -> None:
    # LRU by mtime. Called under _IG_DISK_CACHE_LOCK; safe to scan + delete.
    files = sorted(_IG_DISK_CACHE_DIR.glob("*.pkl"), key=lambda p: p.stat().st_mtime)
    for stale in files[:-_IG_DISK_CACHE_MAX]:
        try:
            stale.unlink()
        except OSError:
            pass


def _check_ready() -> None:
    if inference_route._era5_cache is None:
        raise HTTPException(status_code=503, detail="Inference runtime not ready")


def _parse_source_spec(spec: str) -> tuple[str, int | None]:
    """Parse 'temperature@500' or '2m_temperature' into (variable, level | None)."""
    if "@" in spec:
        variable, level_str = spec.split("@", 1)
        try:
            level: int | None = int(level_str)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Bad source spec '{spec}': level after '@' must be an integer.",
            )
    else:
        variable, level = spec, None
    if not variable:
        raise HTTPException(status_code=400, detail=f"Bad source spec '{spec}': empty variable name.")
    return variable, level


def _cache_get(key: tuple) -> dict[str, Any] | None:
    with _IG_CACHE_LOCK:
        full = _IG_CACHE.get(key)
        if full is not None:
            _IG_CACHE.move_to_end(key)
        return full


def _cache_put(key: tuple, value: dict[str, Any]) -> None:
    with _IG_CACHE_LOCK:
        _IG_CACHE[key] = value
        while len(_IG_CACHE) > _IG_CACHE_MAX:
            _IG_CACHE.popitem(last=False)


def _compact(full: dict[str, Any]) -> dict[str, Any]:
    """Convert list-shaped attribution dicts to float32 numpy arrays for cache storage.

    Python lists of float carry ~84 bytes per element (28-byte float + list pointer +
    bookkeeping). float32 numpy arrays carry 4. ~50× shrink, no math change.
    """
    return {
        **full,
        "attributions_prev":      {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in full["attributions_prev"].items()},
        "attributions_prev_prev": {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in full["attributions_prev_prev"].items()},
    }


def _slice_response(
    full: dict[str, Any],
    wanted_keys: set[str] | None,
    visible_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    """Narrow `full` to the requested feature keys, optionally also slicing each per-feature
    array down to nodes inside the viewport bbox. Arrays stay as float32 numpy — the app's
    ORJSONResponse subclass serializes them directly via OPT_SERIALIZE_NUMPY, avoiding
    ~2.6M PyFloat allocations per response.

    Spatial slicing: when `visible_indices` is given, each per-feature array is reduced from
    length N_grid (~16k) to length N_visible (e.g. ~1k for a typical zoom). This shrinks the
    response payload roughly proportionally — the dominant network/serialization cost on
    cold computes. The frontend gets `visible_indices` back so it can map per-row positions
    in the returned arrays back to the original grid-node IDs.

    Numpy advanced indexing (`arr[indices]`) always returns a new C-contiguous array, which
    is what orjson's NUMPY serialization requires.
    """
    keys = wanted_keys if wanted_keys is not None else set(full["attributions_prev"].keys())

    def slice_arr(v: np.ndarray) -> np.ndarray:
        return v[visible_indices] if visible_indices is not None else v

    return {
        "sources":                [m for m in full["sources"]                       if m["key"] in keys],
        "attributions_prev":      {k: slice_arr(v) for k, v in full["attributions_prev"].items()      if k in keys},
        "attributions_prev_prev": {k: slice_arr(v) for k, v in full["attributions_prev_prev"].items() if k in keys},
        "n_nodes":   full["n_nodes"],
        "visible_indices": visible_indices,  # null = whole graph; else original grid-node IDs in row order
        "target":    full["target"],
        "steps":     full["steps"],
        "pred_time": full["pred_time"],
        "source_ranking": full["source_ranking"],  # always full set, regardless of slice
    }


def _resolve_visible_indices(
    graph_folder,
    min_lat: float | None,
    max_lat: float | None,
    min_lon: float | None,
    max_lon: float | None,
) -> np.ndarray | None:
    """Returns the int32 indices of grid nodes inside the bbox, or None if any bound is missing
    or the graph folder isn't loadable. Reuses load_graph (cached) and the same bbox_node_mask
    helper the graph endpoint uses, so the IG slicing matches the frontend's existing bbox
    filtering exactly."""
    if any(b is None for b in (min_lat, max_lat, min_lon, max_lon)):
        return None
    if graph_folder is None:
        return None
    try:
        raw = load_graph(graph_folder)
    except FileNotFoundError:
        return None
    lat_lon = raw["grid_lat_lon"]  # torch tensor [N_grid, 2]
    mask = bbox_node_mask(lat_lon, min_lat, max_lat, min_lon, max_lon)
    return mask.nonzero(as_tuple=True)[0].numpy().astype(np.int32, copy=False)


def compute_source_ranking(full: dict[str, Any]) -> list[dict[str, Any]]:
    """For every input *feature* (atmospheric levels kept separate, e.g.
    `temperature@500` and `temperature@850` rank independently), sum |attribution|
    across all grid nodes, tracked separately for each input timestep. Return
    descending by total magnitude with each feature's share as a percentage of
    the grand total.

    Per-level ranking avoids the level-count bias that aggregating by variable
    introduces — each input channel that the model receives gets its own row.
    `variable` is the bare variable name (used by the frontend's source-var
    dropdown filter); `feature` is the level-qualified key shown to the user.

    Caller is expected to invoke this on the full (un-sliced) attribution dict so that
    rankings stay consistent regardless of which `source` filter the client requested.
    """
    sums_prev: dict[str, float] = {}
    sums_prev_prev: dict[str, float] = {}
    var_for_key: dict[str, str] = {}
    level_for_key: dict[str, int | None] = {}

    for meta in full["sources"]:
        var = meta["variable"]
        key = meta["key"]
        prev = full["attributions_prev"].get(key)
        prev_prev = full["attributions_prev_prev"].get(key)
        # nansum, not sum: cells the channel has no data for are NaN (see _mask_absent), and a
        # single NaN would make the whole feature's score -- and therefore every percentage in
        # the ranking table -- NaN.
        s_prev = float(np.nansum(np.abs(np.asarray(prev)))) if prev is not None else 0.0
        s_prev_prev = float(np.nansum(np.abs(np.asarray(prev_prev)))) if prev_prev is not None else 0.0
        sums_prev[key] = sums_prev.get(key, 0.0) + s_prev
        sums_prev_prev[key] = sums_prev_prev.get(key, 0.0) + s_prev_prev
        var_for_key[key] = var
        if key not in level_for_key:
            level_for_key[key] = int(key.split("@", 1)[1]) if "@" in key else None

    all_keys = set(var_for_key)
    total = sum(sums_prev.get(k, 0.0) + sums_prev_prev.get(k, 0.0) for k in all_keys) or 1.0

    rows = []
    for k in all_keys:
        s_p = sums_prev.get(k, 0.0)
        s_pp = sums_prev_prev.get(k, 0.0)
        score = s_p + s_pp
        rows.append({
            "variable":       var_for_key[k],
            "feature":        k,
            "level":          level_for_key[k],
            "score":          score,
            "score_prev_prev": s_pp,
            "score_prev":     s_p,
            "percentage":     score / total * 100.0,
        })
    rows.sort(key=lambda r: -r["score"])
    return rows


@router.get("/{model_id}/attributions")
@router.get("/attributions")
async def get_attributions(
    request: Request,
    pred_time: str,
    target_variable: str,
    target_level: int | None = None,
    target_node: int | None = None,
    source: list[str] = Query(default=[]),
    method: str = Query("ixg"),
    min_lat: float | None = None,
    max_lat: float | None = None,
    min_lon: float | None = None,
    max_lon: float | None = None,
    graph_id: str | None = None,
    model_id: str | None = None,
    db: Session = Depends(get_db),
):
    """
    Per-input-node Integrated Gradients attributions for a single prediction step.

    Caching: the full attribution tensor (all features) is computed once per
    (method, pred_time, target_variable, target_level, target_node) and cached in-process.
    Subsequent requests that vary only `source` or the bbox slice the cached tensor —
    typically <100 ms.

    - pred_time:       ISO timestamp of the prediction step (must be in the loaded ERA5 cache).
    - target_variable: Output variable to explain.
    - target_level:    Pressure level (hPa) — required iff target_variable is atmospheric.
    - target_node:     Optional grid-node index. When set, attributes the prediction at this
                       single output node. When omitted, sums the target feature across all
                       output nodes (global integral, rarely the question you want).
    - source:          Repeat query param. Each entry is 'variable' (surface) or
                       'variable@level' (atmospheric). Omit to return *all* features.
    - method:          Attribution method. Options: "ixg" (fast, ~0.9s) or "ig" (principled, ~3.5s).
                       Default: "ixg". Both methods coexist in the cache — switching does not
                       invalidate the other method's results.
    - min_lat / max_lat / min_lon / max_lon:
                       Optional viewport bbox. When all four are given AND graph_id resolves
                       to a loadable graph folder, per-feature attribution arrays are sliced
                       down to nodes inside the bbox and the response includes
                       `visible_indices` (original grid-node IDs in row order). Cuts the
                       response payload roughly proportionally — the biggest cost on cold
                       computes is JSON over the wire.
    - graph_id:        Graph DB id used to resolve the grid lat/lon for bbox filtering.
                       Ignored when no bbox is given. If unset, the response is unsliced.
    """
    # Resolves the model AND the cache for its domain together. Taking the cache from the
    # manager independently would always hand back ERA5, so an ocean model would be
    # attributed against weather data -- shapes differ (28 vs 83 channels), but the failure
    # would surface deep inside TorchScript rather than here.
    resolved_model_id, model, domain, era5_cache = inference_route.get_explain_context(db, model_id)
    device = inference_route._device_for(domain)

    # Validate method parameter
    if method not in ("ixg", "ig"):
        raise HTTPException(status_code=400, detail=f"Invalid method '{method}'. Must be 'ixg' or 'ig'.")
    
    t0 = time.perf_counter()

    sources = [_parse_source_spec(s) for s in source] if source else None
    wanted_keys = (
        {f"{v}@{lv}" if lv is not None else v for v, lv in sources}
        if sources is not None
        else None
    )

    cache_key = (resolved_model_id, method, pred_time, target_variable, target_level, target_node)
    full = _cache_get(cache_key)
    cache_status = "mem-hit" if full is not None else "mem-miss"

    t_lookup = time.perf_counter() - t0
    t_compute = t_compact = t_rank = 0.0

    if full is None:
        # A second concurrent request for the identical key waits here instead of
        # launching a duplicate forward(+backward) pass. Different keys use different
        # locks, so unrelated requests are never serialized against each other.
        inflight_lock = _IG_INFLIGHT.setdefault(cache_key, asyncio.Lock())
        try:
            async with inflight_lock:
                # Re-check: the request that held the lock may have already populated
                # the cache while we were waiting on it.
                full = _cache_get(cache_key)
                if full is not None:
                    cache_status = "mem-hit"

                if full is None:
                    # In-memory miss → check disk before recomputing (saves ~7 s after a
                    # container restart).
                    full = _disk_cache_load(resolved_model_id, method, pred_time, target_variable, target_level, target_node)
                    if full is not None:
                        _cache_put(cache_key, full)
                        cache_status = "disk-hit"

                if full is None:
                    cache_status = "compute"
                    ts = time.perf_counter()
                    common = dict(
                        model=model,
                        era5_cache=era5_cache,
                        pred_time=pred_time,
                        target_variable=target_variable,
                        target_level=target_level,
                        target_node=target_node,
                        sources=None,  # always full tensor — keeps cache reusable across `source` filters
                        device=device,
                    )

                    # IG is the only expensive path worth cancelling — IxG is a single forward+backward
                    # pass with no loop to break out of. For IG, we drive the torch compute on a thread
                    # and poll request.is_disconnected() from the event loop. If the client goes away,
                    # we set a threading.Event that the IG loop checks between mini-batches, raising
                    # IGCancelled which we catch and turn into a 499.
                    cancel_event = threading.Event()

                    async def _watch_disconnect() -> None:
                        try:
                            while not cancel_event.is_set():
                                if await request.is_disconnected():
                                    cancel_event.set()
                                    return
                                await asyncio.sleep(0.2)
                        except asyncio.CancelledError:
                            pass

                    monitor_task: asyncio.Task | None = None
                    # Set inside the except and acted on after it, deliberately: while the
                    # handler is running, the live exception still references the frames that
                    # hold the activation tensors, so empty_cache() there frees nothing. Once
                    # the block exits, Python drops the exception and the memory becomes
                    # reclaimable -- which is why the pool stayed poisoned until a pod restart.
                    oom: BaseException | None = None
                    async with _ATTR_COMPUTE_LOCK:
                      try:
                          if method == "ixg":
                              # IxG is non-cancellable, but this is an async route — running it sync would
                              # block the event loop (and every other in-flight request) for its ~0.9s
                              # duration. Offload to a thread so the loop stays free.
                              raw = await asyncio.to_thread(compute_input_x_gradients, **common)
                          else:
                              # batch_size=1 → _IG_STEPS sequential mini-batches → cancellation latency ≈
                              # one mini-batch (~2–3s on CPU). With _IG_STEPS=4 the per-batch overhead is
                              # negligible vs the single-batch path, so we accept it as the cost of being
                              # interruptible.
                              monitor_task = asyncio.create_task(_watch_disconnect())
                              raw = await asyncio.to_thread(
                                  compute_integrated_gradients,
                                  **common,
                                  steps=_IG_STEPS,
                                  batch_size=1,
                                  should_continue=lambda: not cancel_event.is_set(),
                              )
                      except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                          # Both shapes: a direct allocation failure raises OutOfMemoryError,
                          # but one inside the model surfaces as a plain RuntimeError -- the
                          # TorchScript interpreter catches the inner error and re-raises it
                          # wrapped, so type(exc) is RuntimeError and the OOM is only visible
                          # in the message. Catching just the former would miss every real case.
                          if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower():
                              oom = e
                          else:
                              raise
                      except IGCancelled as e:
                          print(f"[{method}] cancelled ({e}) target_node={target_node}", flush=True)
                          raise HTTPException(status_code=499, detail="Client cancelled")
                      except ValueError as e:
                          raise HTTPException(status_code=400, detail=str(e))
                      finally:
                          cancel_event.set()
                          if monitor_task is not None:
                              monitor_task.cancel()
                          # Hand back fragments while the GPU is shared. Attribution frees a
                          # full activation set per step, so reserved memory can sit well above
                          # what is live; releasing here keeps repeated requests from walking
                          # the pool up to the cap. No-ops below the threshold and on CPU.
                          from services.ocean import release_vram_if_high  # noqa: PLC0415

                          release_vram_if_high()

                    if oom is not None:
                        # The exception is out of scope here, so its frames no longer pin the
                        # activations and this actually returns memory to the driver.
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        raise HTTPException(
                            status_code=507,
                            detail=(
                                "Not enough GPU memory for this attribution. The pass needs "
                                "~14 GiB; try again in a moment."
                            ),
                        ) from None

                    t_compute = time.perf_counter() - ts
                    ts = time.perf_counter()
                    full = _compact(raw)
                    t_compact = time.perf_counter() - ts
                    ts = time.perf_counter()
                    full["source_ranking"] = compute_source_ranking(full)
                    t_rank = time.perf_counter() - ts
                    _cache_put(cache_key, full)
                    _disk_cache_save(resolved_model_id, method, pred_time, target_variable, target_level, target_node, full)  # async; doesn't block the response
        finally:
            _IG_INFLIGHT.pop(cache_key, None)

    ts = time.perf_counter()
    graph_folder = GraphCRUD.get_path_by_id(db, graph_id) if graph_id is not None else None
    visible_indices = _resolve_visible_indices(graph_folder, min_lat, max_lat, min_lon, max_lon)
    n_visible = "all" if visible_indices is None else len(visible_indices)
    sliced = _slice_response(full, wanted_keys, visible_indices)
    # Add method to response so frontend knows which attribution method was used
    sliced["method"] = method
    # Diagnostic: surface cache path so the frontend can log it without scraping docker logs
    sliced["cache_status"] = cache_status
    t_slice = time.perf_counter() - ts

    ts = time.perf_counter()
    response = ORJSONResponse(content=sliced)
    t_serialize = time.perf_counter() - ts

    print(
        f"[{method}] {cache_status} total={(time.perf_counter() - t0):.2f}s "
        f"lookup={t_lookup * 1000:.0f}ms compute={t_compute:.2f}s "
        f"compact={t_compact * 1000:.0f}ms rank={t_rank * 1000:.0f}ms "
        f"slice={t_slice * 1000:.0f}ms serialize={t_serialize * 1000:.0f}ms "
        f"n_visible={n_visible} target_node={target_node}",
        flush=True,
    )
    return response