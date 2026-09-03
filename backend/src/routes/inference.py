from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
import threading

from services.crud import ModelCRUD
from services.database import SessionLocal, get_db
from services.domains import DOMAIN_ERA5, DOMAIN_OCEAN, normalize_domain
from services.era5_window_manager import WindowAlreadyLoadingError, derive_month_bands, manager
from services.graph import grid_spacing_deg
from services.json_response import ORJSONResponse
from services.inference import (
    DEVICE,
    FIELDS_ZARR_PATH,
    MODEL_PATH,
    PRESET_MODELS,
    _clear_inference_cache,
    compute_error_range,
    compute_physical_scales,
    compute_prediction_errors,
    get_available_pred_timestamps,
    get_era5_step_inputs,
    get_prediction_field,
    load_grid,
    load_model,
    load_timeline,
    resolve_timeline_path,
    preload_era5_tensors,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/inference", tags=["Inference"], default_response_class=ORJSONResponse)

DEFAULT_MODEL_ID = "1"

# Single source of truth in services/ocean.py: the window manager builds the ocean cache on
# that device, and the tensors it holds must match the ones built per request.
from services.ocean import OCEAN_DEVICE, VRAM_LIMIT_MIB  # noqa: E402

# Hard ceiling on VRAM, matching the `vram: "16384"` request in helm/values.yaml.
#
# The cluster shares GPUs by VRAM reservation rather than handing out whole cards, so the
# request is a promise to other tenants, not something the driver enforces -- nothing stops a
# process from allocating past it. set_per_process_memory_fraction makes it real: allocations
# beyond the cap raise torch.cuda.OutOfMemoryError, which is a normal catchable exception, so
# an oversized attribution returns a 500 instead of eating into somebody else's share.


# Ocean attribution is a backward pass over a 3.4M-edge graph, which retains every layer's
# activations: an estimated 9-30 GB, against ~0.4 GB for ERA5. On CUDA an over-allocation
# raises a catchable OutOfMemoryError and only that request fails. On CPU the kernel SIGKILLs
# the process instead, so a shared deployment loses every in-flight request and the API is down
# for the ~2 minutes the pod needs to restart.
#
# Enabled by default: on a workstation that trade-off belongs to whoever is running it, and
# defaulting to off would make the feature impossible to develop locally. Set
# OCEAN_ATTRIBUTION=0 on a deployment that cannot absorb the restart.
OCEAN_ATTRIBUTION_ENABLED: bool = (
    os.environ.get("OCEAN_ATTRIBUTION", "1").strip().lower() not in ("0", "false", "no", "off")
)


def _cap_vram() -> None:
    """Apply VRAM_LIMIT_MIB to every visible CUDA device. No-op without CUDA."""
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        total_mib = torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)
        # A card smaller than the cap needs no restriction; fraction > 1 is rejected anyway.
        fraction = min(1.0, VRAM_LIMIT_MIB / total_mib)
        torch.cuda.set_per_process_memory_fraction(fraction, device=i)
        logger.info(
            "[vram] device %d: %.0f MiB total, capped at %.0f MiB (fraction %.3f)",
            i, total_mib, min(VRAM_LIMIT_MIB, total_mib), fraction,
        )

_model: torch.nn.Module | None = None
_model_cache: dict[str, torch.nn.Module] = {}   # preset keys + DB model IDs
_model_cache_lock = threading.Lock()
_active_model_key: str = "graphcast"             # which preset is currently active
_era5_cache: dict | None = None
_grid: dict | None = None
_timelines: dict[str, dict] = {}                 # preset key → precomputed timeline dict


class _LoadWindowRequest(BaseModel):
    start: str
    end: str


def _warmup_torchscript() -> None:
    """One forward pass at startup so the first user-triggered IG request doesn't pay
    TorchScript JIT-compile cost (~1–2 s on first call). Failure here must never block
    the server from coming up — explainability is not required for the rest of the API."""
    if _era5_cache is None:
        return
    try:
        model = _model_cache.get(DEFAULT_MODEL_ID)
        if model is None:
            return
        timestamps = get_available_pred_timestamps(_era5_cache, as_str=True)
        if not timestamps:
            return
        prev, prev_prev, forcing = get_era5_step_inputs(_era5_cache, timestamps[0], DEVICE)
        with torch.no_grad():
            model(prev, prev_prev, forcing)
    except Exception as e:
        print(f"[warmup] TorchScript warmup skipped: {e}")


def _resolve_model_record(db: Session, model_id: str | None):
    requested = model_id or DEFAULT_MODEL_ID
    model_record = ModelCRUD.get(db, requested)
    if model_record is None and model_id is None:
        all_models = ModelCRUD.list_all(db)
        if all_models:
            model_record = all_models[0]
    if model_record is None:
        raise HTTPException(status_code=404, detail=f"Model '{requested}' not found")
    return model_record


def _window_for_model(db: Session, model_id: str | None):
    """
    The active window for a model's domain, building the domain if needed.

    Used by the endpoints that only advertise which timestamps are selectable -- they need
    the right domain's timestamps but not its model, so they skip the model load that
    get_explain_context does.
    """
    model_record = _resolve_model_record(db, model_id)
    domain = _model_domain(model_record)
    try:
        manager.ensure_domain(domain)
        return manager.get_current_window(domain)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"No '{domain}' window available: {e}")


@lru_cache(maxsize=1)
def _era5_grid_spacing() -> float:
    """
    Spacing of the ERA5 inference grid, from the same helper the graph endpoint uses.

    Cached: `_grid` is loaded once at startup and never changes, and the computation sorts
    every latitude. Falls back to the 1.5 deg ERA5 step if the grid is unavailable, so a
    missing grid file degrades cell sizing rather than failing the request.
    """
    if _grid is None:
        return 1.5
    gll = torch.stack([
        torch.as_tensor(_grid["lat"], dtype=torch.float32),
        torch.as_tensor(_grid["lon"], dtype=torch.float32),
    ], dim=1)
    return grid_spacing_deg(gll)


def _device_for(domain: str) -> str:
    """Device a domain's tensors and model live on. They must agree, or TorchScript fails
    deep inside with 'Passed CPU tensor to MPS op'."""
    return OCEAN_DEVICE if domain == DOMAIN_OCEAN else DEVICE


def _attribution_available(domain: str) -> bool:
    """Whether attribution is offered for a domain. ERA5 is always fine -- two orders of
    magnitude smaller, and it ran on CPU for the whole project. Ocean follows
    OCEAN_ATTRIBUTION; see the note there for why it is a deployment decision."""
    return True if domain != DOMAIN_OCEAN else OCEAN_ATTRIBUTION_ENABLED


def _model_domain(model_record) -> str:
    """The domain a model record routes to, validated."""
    try:
        return normalize_domain(getattr(model_record, "domain", None))
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


def _get_or_load_model(model_record) -> torch.nn.Module:
    model_id = str(model_record.model_id)
    with _model_cache_lock:
        cached = _model_cache.get(model_id)
    if cached is not None:
        return cached

    domain = _model_domain(model_record)
    try:
        if domain == DOMAIN_OCEAN:
            # The SeaCast export takes a fourth atmosphere argument; load_ocean_model wraps
            # it so it presents the 3-argument signature the rest of the backend calls.
            # Going through load_model here yields a bare ScriptModule that fails at call
            # time with "forward() is missing value for argument 'a'".
            # CPU, not DEVICE: one pass over 675,219 nodes overruns Apple MPS's cap.
            from services.ocean import load_ocean_model  # noqa: PLC0415

            loaded = load_ocean_model(model_record.model_path, device=OCEAN_DEVICE)
        else:
            loaded = load_model(path=model_record.model_path, device=DEVICE)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model '{model_id}': {e}")

    with _model_cache_lock:
        _model_cache[model_id] = loaded
    return loaded


def get_runtime_for_request(db: Session, model_id: str | None) -> tuple[str, torch.nn.Module, dict]:
    _check_ready()
    # When no explicit model_id is in the URL and a preset is active (set via
    # POST /inference/switch_model), use the preset directly — bypassing the DB
    # lookup entirely.  Explicit model_id in the URL always takes precedence and
    # goes through the DB path as before.
    # Presets are ERA5-only (see PRESET_MODELS), so this branch never serves ocean.
    if model_id is None and _active_model_key in _model_cache:
        return _active_model_key, _model_cache[_active_model_key], _era5_cache
    model_record = _resolve_model_record(db, model_id)
    model = _get_or_load_model(model_record)
    return str(model_record.model_id), model, _era5_cache


def get_explain_context(db: Session, model_id: str | None) -> tuple[str, torch.nn.Module, str, dict]:
    """
    Resolve everything one Explain request needs: (model_key, model, domain, cache).

    Replaces the `manager.is_initialized()` / `manager.get_current_cache()` pair that every
    Explain endpoint used to open with. The order matters and is the reason this exists: the
    domain is a property of the *model*, so the model must be resolved before the manager can
    be asked for the matching data. Checking the manager first -- as the endpoints did --
    can only ever check ERA5.

    The domain is resolved per request rather than held as manager state. The manager is
    process-wide and shared across concurrent requests, so a mutable "active domain" would
    let one caller's model switch redirect another caller's in-flight request to the wrong
    dataset, and both would still return a plausible-looking field.
    """
    _check_ready()

    if model_id is None and _active_model_key in _model_cache:
        return _active_model_key, _model_cache[_active_model_key], DOMAIN_ERA5, _era5_cache

    model_record = _resolve_model_record(db, model_id)
    domain = _model_domain(model_record)
    model = _get_or_load_model(model_record)

    try:
        manager.ensure_domain(domain)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Data for domain '{domain}' could not be loaded: {e}",
        )
    if not manager.is_initialized(domain):
        raise HTTPException(status_code=503, detail=f"No '{domain}' window is loaded")

    cache = manager.get_current_cache(domain)
    # Assert the pairing rather than trusting it. An ocean model fed the ERA5 cache does not
    # fail loudly -- the shapes differ (675,219x28 vs 29,040x83), but the first thing it hits
    # is a timestamp lookup, so it surfaces as "Timestamp not found in loaded ERA5 subset"
    # from a route the user asked about the ocean. That cost an afternoon once; a domain is
    # cheap to check and the failure names the real problem.
    if bool(cache.get("_ocean")) != (domain == DOMAIN_OCEAN):
        raise HTTPException(
            status_code=500,
            detail=(
                f"Domain/cache mismatch: model '{model_record.model_id}' resolves to domain "
                f"'{domain}' but the loaded cache is "
                f"{'ocean' if cache.get('_ocean') else 'ERA5'}. Refusing to run inference."
            ),
        )
    return str(model_record.model_id), model, domain, cache


def initialize_inference_runtime() -> None:
    # TODO: _model, _era5_cache, and _grid are only consumed by Explain mode.
    # Move all three to lazy-on-first-Explain-request once thread-safety is handled.
    global _model, _model_cache, _active_model_key, _era5_cache, _grid, _timelines
    # Before anything touches the GPU: the cap only binds allocations made after it is set.
    _cap_vram()
    # Preload all preset models so switching is instant (no reload cost).
    for key, path in PRESET_MODELS.items():
        if path.exists():
            try:
                _model_cache[key] = load_model(path=path, device=DEVICE)
                print(f"[startup] Loaded preset model '{key}' from {path}")
            except Exception as e:
                print(f"[startup] Failed to load preset model '{key}': {e}")
        else:
            print(f"[startup] Preset model '{key}' not found at {path} — skipping")
    _model = _model_cache.get(_active_model_key)
    if _model is None:
        try:
            _model = load_model(device=DEVICE)
        except Exception as e:
            print(f"[startup] Default model load skipped: {e}", flush=True)
    # Lazy: this module-global cache is a fallback (warmup, the IG seed path) — every Explain
    # endpoint reads the manager's window, which is itself built lazily. Loading it eagerly
    # materialised the whole seed month up front and pinned ~1.1 GB for the life of the
    # process, which is most of what pushed an ocean forward pass past the container limit.
    # Missing PVC data must not take the process down — /health has to stay up so the
    # pod is Running and the claim can be filled by hand.
    try:
        _era5_cache = preload_era5_tensors(FIELDS_ZARR_PATH, device=DEVICE, lazy=True)
    except Exception as e:
        print(f"[startup] ERA5 preload skipped: {e}", flush=True)

    # Eager-load default model when available so first explain request is faster.
    try:
        db = SessionLocal()
        try:
            model_record = _resolve_model_record(db, None)
            _get_or_load_model(model_record)
        finally:
            db.close()
    except Exception as e:
        print(f"[startup] Default model warmload skipped: {e}")

    _warmup_torchscript()
    try:
        _grid = load_grid()
    except FileNotFoundError as e:
        print(f"[startup] Grid file not found — /inference/grid will return 503: {e}")


def _check_ready() -> None:
    if _era5_cache is None:
        raise HTTPException(status_code=503, detail="Inference runtime not ready")


@router.get("/{model_id}/model_info")
@router.get("/model_info")
def get_model_info():
    """Return info about the currently active model."""
    path = PRESET_MODELS.get(_active_model_key, MODEL_PATH)
    return {
        "name": path.stem,
        "path": str(path),
        "key": _active_model_key,
    }


class _SwitchModelRequest(BaseModel):
    key: str


@router.post("/switch_model")
def switch_model(body: _SwitchModelRequest, db: Session = Depends(get_db)):
    """Switch the active inference model to one of the preloaded presets."""
    global _model, _active_model_key
    if body.key not in PRESET_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model key '{body.key}'. Valid keys: {list(PRESET_MODELS)}")
    if body.key not in _model_cache:
        raise HTTPException(status_code=503, detail=f"Model '{body.key}' was not loaded at startup (file missing?)")
    _model = _model_cache[body.key]
    _active_model_key = body.key
    _clear_inference_cache()
    path = PRESET_MODELS[body.key]
    model_record = ModelCRUD.get_by_filename(db, path.name)
    return {
        "name": path.stem,
        "path": str(path),
        "key": _active_model_key,
        "model_id": str(model_record.model_id) if model_record else None,
    }


@router.get("/{model_id}/variables")
@router.get("/variables")
def list_variables(model_id: str | None = None, db: Session = Depends(get_db)):
    """
    Return the variable names this model predicts.

    ERA5 splits them into atmospheric variables x pressure levels plus surface variables.
    Ocean state has no level axis -- each feature carries its own depth ("thetao_47m") -- so
    they are reported as surface variables with an empty level list, which is what the
    frontend already does for 2m_temperature and friends. Serving the ERA5 list for an ocean
    model would offer variables it does not predict.
    """
    model_record = _resolve_model_record(db, model_id)
    domain = _model_domain(model_record)
    _check_ready()

    if domain == DOMAIN_OCEAN:
        try:
            manager.ensure_domain(domain)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Ocean variables unavailable: {e}")
        # `is_ocean` + `state_vars` are what the right panel keys off to switch to the
        # ocean variable list and its depth-aware labels ("thetao_47m" -> "Temperature
        # (47m)"). `surf_vars` carries the same names so anything reading the ERA5 shape
        # still works -- ocean channels behave like surface variables, having no level.
        state_vars = list(manager.get_current_cache(domain)["state_features"])
        return {
            "is_ocean": True,
            "state_vars": state_vars,
            "atm_vars": [],
            "surf_vars": state_vars,
            "levels": [],
            "attribution_available": _attribution_available(domain),
        }

    return {
        "is_ocean": False,
        "atm_vars": _era5_cache["atm_vars"],
        "surf_vars": _era5_cache["surf_vars"],
        "levels": _era5_cache["levels"],
        "attribution_available": _attribution_available(domain),
    }


@router.get("/{model_id}/cached_timestamps")
@router.get("/cached_timestamps")
def get_cached_timestamps(model_id: str | None = None, db: Session = Depends(get_db)):
    """
    Return all prediction-valid timestamps from every zarr in the local ERA5 cache dir.

    ERA5-only, and empty for every other domain: the cache dir holds GCS-fetched ERA5
    windows. The ocean window is whatever is on disk and is already reported in full by
    /loaded_window, so there is nothing additional to advertise -- and returning ERA5's 2020
    timestamps here would merge them into an October 2024 ocean timeline, marking dates as
    available that the ocean model cannot predict.
    """
    model_record = _resolve_model_record(db, model_id)
    if _model_domain(model_record) != DOMAIN_ERA5:
        return {"timestamps": []}
    if not manager.is_initialized():
        raise HTTPException(status_code=503, detail="ERA5WindowManager not initialized")
    return {"timestamps": manager.get_all_cached_timestamps()}


@router.get("/{model_id}/loaded_window")
@router.get("/loaded_window")
def get_loaded_window(model_id: str | None = None, db: Session = Depends(get_db)):
    """Return the active window's range and prediction-valid timestamps, for this model's domain."""
    window = _window_for_model(db, model_id)
    return {
        "range_start": window.range_start,
        "range_end": window.range_end,
        "timestamps": list(window.timestamps),
    }


@router.post("/{model_id}/load_window")
@router.post("/load_window")
def load_window(body: _LoadWindowRequest, model_id: str | None = None, db: Session = Depends(get_db)):
    """Download (or reuse cached) ERA5 data for the given month band and swap the active window."""
    _resolve_model_record(db, model_id)
    if not manager.is_initialized():
        raise HTTPException(status_code=503, detail="ERA5WindowManager not initialized")
    try:
        result = manager.load_window(body.start, body.end)
    except WindowAlreadyLoadingError:
        raise HTTPException(status_code=409, detail="A window load is already in progress")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@router.get("/{model_id}/timestamps")
@router.get("/timestamps")
def list_prediction_timestamps(model_id: str | None = None, db: Session = Depends(get_db)):
    """Return the valid prediction timestamps from the active window, for this model's domain."""
    timestamps = list(_window_for_model(db, model_id).timestamps)
    return {"timestamps": timestamps, "count": len(timestamps)}


@router.get("/{model_id}/grid")
@router.get("/grid")
def get_grid(model_id: str | None = None, db: Session = Depends(get_db)):
    """
    Return the inference grid positions for this model's domain.

    Response: { lat: [...], lon: [...], grid_spacing_deg: float } — one entry per grid node.
    Longitude is normalised to [-180, 180]. Index i pairs with errors[i] and attribution[i]
    from all other Explain mode endpoints.

    MUST follow the model's domain. ERA5 has 29,040 nodes at 1.5 deg, the ocean grid 675,219
    at 0.25 deg, and the frontend places value i at position i -- so serving the ERA5 grid for
    an ocean model paints 29,040 of 675,219 values onto the wrong coordinates and silently
    draws a thin strip near the dateline instead of a global field.

    `grid_spacing_deg` is reported for the same reason the graph endpoint reports it: cell
    size is a property of the grid, and a constant tuned for one is wrong for the other by 6x.
    """
    model_record = _resolve_model_record(db, model_id)
    domain = _model_domain(model_record)

    if domain == DOMAIN_OCEAN:
        try:
            manager.ensure_domain(domain)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Ocean grid unavailable: {e}")
        gll = manager.get_current_cache(domain)["grid_lat_lon"]
        return ORJSONResponse(content={
            "lat": gll[:, 0].contiguous().numpy(),
            "lon": gll[:, 1].contiguous().numpy(),
            "grid_spacing_deg": grid_spacing_deg(gll),
        })

    if _grid is None:
        raise HTTPException(status_code=503, detail="Grid file not found — check data/grid/grid_lat_lon.pt")
    # Returning a Response directly bypasses FastAPI's default jsonable_encoder pass,
    # which does not know how to handle numpy arrays — required for OPT_SERIALIZE_NUMPY
    # passthrough to actually apply instead of erroring or silently falling back.
    return ORJSONResponse(content={**_grid, "grid_spacing_deg": _era5_grid_spacing()})


@router.get("/{model_id}/prediction_errors")
@router.get("/prediction_errors")
def predict_errors(
    pred_time: str,
    variable: str = Query("total", description="Variable name or 'total'"),
    z_level: str = Query("total", description="Pressure level (e.g. '500') or 'total'"),
    with_field: bool = Query(False, description="Also include the pred slice (only meaningful for single-component cases 1 and 5)"),
    model_id: str | None = None,
    db: Session = Depends(get_db),
):
    """
    Run one-step inference and return normalised error (RMSE or absolute) per node.

    Query parameters:
      pred_time:  ISO timestamp (e.g. "2020-06-01T12:00")
      variable:   Variable name, "total" (default: "total")
      z_level:    Pressure level as string (e.g. "500") or "total" (default: "total")
                  Ignored for surface variables.
      with_field: If true, also returns the underlying `pred` slice (normalized)
                  so the frontend can show the prediction map. Only populated for
                  single-component cases (1 and 5); aggregated cases return
                  `pred=null` because a per-node "average prediction across levels"
                  has no physical meaning.

                  The matching `truth` slice is deliberately NOT bundled here. It is
                  a third of the payload and nothing in the frontend reads it -- the
                  error map is served by `errors` and the prediction map by `pred`.
                  `/prediction_field` still returns it for callers that want to
                  recompute `|pred - truth|` themselves.

    Response:
      with_field=false (default, backwards compatible):
        list[float] of length N_grid = 29,040 — error per node.
      with_field=true:
        {
          "errors":         list[float],  # length N_grid, same as default response
          "pred":           list[float] | null,  # length N_grid (cases 1, 5) else null
          "component_idx":  int | null,
        }

    Cases:
      1. variable=<specific atm_var>, z_level=<specific hPa>
         → |truth - pred| per node (absolute error, no averaging)
      2. variable=<specific atm_var>, z_level="total"
         → √(mean((truth - pred)² over all levels)) per node
      3. variable="total", z_level=<specific hPa>
         → √(mean((truth - pred)² over all variables at that level)) per node
      4. variable="total", z_level="total"
         → √(mean((truth - pred)² over all variables+levels)) per node
      5. variable=<surface_var>, z_level=<ignored>
         → |truth - pred| per node (surface has no levels)
    """
    resolved_model_id, model, domain, era5_cache = get_explain_context(db, model_id)
    if not manager.is_in_window(pred_time, domain):
        raise HTTPException(
            status_code=404,
            detail=f"pred_time '{pred_time}' is outside the active '{domain}' window",
        )
    try:
        errors = compute_prediction_errors(
            model=model,
            era5_cache=era5_cache,
            pred_time=pred_time,
            variable=variable,
            z_level=z_level,
            model_cache_key=resolved_model_id,
            device=_device_for(domain),
        )
        if not with_field:
            # See get_grid above: explicit Response bypasses jsonable_encoder so the
            # numpy array reaches orjson's OPT_SERIALIZE_NUMPY passthrough untouched.
            return ORJSONResponse(content=errors)

        # Single-component cases only — aggregated cases (2, 3, 4) have no meaningful
        # single pred/truth slice to attach.
        if era5_cache.get("_ocean"):
            # The ocean cache has no surf_vars/atm_vars keys at all: its features are flat,
            # with depth carried in the name, so there is no level axis to disambiguate and
            # any named feature is already a single component. Reading the ERA5 keys here
            # raised KeyError, which `except ValueError` below does not catch — so it escaped
            # as a 500 on every with_field=true request, before CORS headers were attached.
            is_single_component = variable in era5_cache["state_features"]
        else:
            is_single_component = (
                variable in era5_cache["surf_vars"]
                or (variable in era5_cache["atm_vars"] and z_level not in (None, "total"))
            )
        if is_single_component:
            field = get_prediction_field(
                model=model,
                era5_cache=era5_cache,
                pred_time=pred_time,
                variable=variable,
                z_level=z_level,
                model_cache_key=resolved_model_id,
                # Must match the model's own device, as the errors call above does. DEVICE is
                # the ERA5 device; using it for an ocean model puts the inputs somewhere the
                # loaded module is not.
                device=_device_for(domain),
            )
            # field["truth"] is computed but not forwarded -- see the with_field note
            # in the docstring. Dropping it here rather than teaching
            # get_prediction_field to skip it keeps /prediction_field's contract intact;
            # the discarded work is one column copy, well under a millisecond.
            return ORJSONResponse(content={
                "errors":        errors,
                "pred":          field["pred"],
                "component_idx": field["component_idx"],
            })
        return ORJSONResponse(content={
            "errors":        errors,
            "pred":          None,
            "component_idx": None,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{model_id}/prediction_errors/range")
@router.get("/prediction_errors/range")
def predict_errors_range(pred_time: str, model_id: str | None = None, db: Session = Depends(get_db)):
    """
    Compute the four colormap scale maxes for a given prediction timestep.

    One inference call (served from cache if already computed), four tensor reductions.
    The frontend fetches this once per pred_time and caches it locally; the colormap
    scale is held fixed when the user switches variables or levels within the same
    aggregation mode.

    Response:
      {
        "single_component": float,   # max |pred−truth| over all 83 components × nodes
        "variable_total":   float,   # max per-variable RMSE-over-levels, over atm_vars × nodes
        "total_by_level":   float,   # max per-level RMSE-over-atm-vars, over levels × nodes
        "global_total":     float,   # max per-node RMSE over all 83 components
      }
    """
    resolved_model_id, model, domain, era5_cache = get_explain_context(db, model_id)
    try:
        return compute_error_range(
            model=model,
            era5_cache=era5_cache,
            pred_time=pred_time,
            model_cache_key=resolved_model_id,
            device=_device_for(domain),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{model_id}/normalization")
@router.get("/normalization")
def get_normalization(model_id: str | None = None, db: Session = Depends(get_db)):
    """Return per-component normalization stats + per-feature physical colorbar ranges.

    Meant to be fetched once on app load and cached client-side. The frontend uses
    `mean`/`std` to denormalize any `pred`/`truth` array returned by
    `/prediction_field`. `physical_scales` gives a stable [min, max] per feature
    across the loaded ERA5 window, so the colorbar doesn't rescale as the user
    scrubs the timeline.

    Response:
      {
        "mean": float[d_state],                   # 83 entries — one per stacked component
        "std":  float[d_state],
        "physical_scales": { "<key>": [min, max] },  # key is "var@level" or "var" (surface)
      }
    """
    _, _, domain, era5_cache = get_explain_context(db, model_id)
    return {
        "domain":          domain,
        "mean":            era5_cache["data_mean"].detach().cpu().numpy().reshape(-1).tolist(),
        "std":             era5_cache["data_std"].detach().cpu().numpy().reshape(-1).tolist(),
        "physical_scales": compute_physical_scales(era5_cache),
    }


@router.get("/{model_id}/prediction_field")
@router.get("/prediction_field")
def get_prediction_field_route(
    pred_time: str,
    variable: str,
    z_level: str | None = Query(None, description="Pressure level (e.g. '500'); omit for surface vars"),
    model_id: str | None = None,
    db: Session = Depends(get_db),
):
    """Return raw `pred` + `truth` slices for one (variable, level) in normalized space.

    Frontend denormalizes locally with the (mean, std) from `/normalization`,
    computes `|pred - truth|` for error display, or shows pred / truth directly.

    Inference is reused via `run_inference_cached`, so calling this for many
    (variable, level) at the same pred_time costs zero extra forward passes.

    Response:
      {
        "pred":          float[N_grid],   # 29040, normalized
        "truth":         float[N_grid],   # 29040, normalized
        "component_idx": int,             # column index into d_state (debug aid)
      }
    """
    resolved_model_id, model, domain, era5_cache = get_explain_context(db, model_id)
    if not manager.is_in_window(pred_time, domain):
        raise HTTPException(
            status_code=404,
            detail=f"pred_time '{pred_time}' is outside the active '{domain}' window",
        )
    try:
        field = get_prediction_field(
            model=model,
            era5_cache=era5_cache,
            pred_time=pred_time,
            variable=variable,
            z_level=z_level,
            model_cache_key=resolved_model_id,
            device=_device_for(domain),
        )
        return ORJSONResponse(content=field)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{model_id}/prediction_errors/timeline")
def get_errors_timeline(model_id: str, db: Session = Depends(get_db)):
    """
    Return the precomputed per-timestep error series for the timeline bar.

    Looks up the model by DB model_id, then loads
    <model_dir>/<model_stem>_timeline_errors.npy. Results are cached after first load.

    Response: { range_start, range_end, timestamps: [...ISO], mean_error: [...float],
                peak_error, error_concentration, covered_timestamps, stale, month_bands }

    peak_error and error_concentration are null for artifacts written before they
    existed; the frontend disables their chips and falls back to mean_error rather
    than blanking the chart. `rmse` is kept as an alias of mean_error for one release
    in case anything else still reads it.

    The timeline is a precomputed artifact and the loaded data window is not; nothing keeps
    them in step. Re-download a different window and the file still describes the old dates,
    the chart renders happily, and every scrub 404s against a window that does not contain
    them. So the two are compared here and the overlap is reported rather than assumed --
    `covered_timestamps` is the subset that is actually scrubbable, and `stale` is True when
    the file and the window are disjoint, which means the artifact needs regenerating.
    """
    model_record = _resolve_model_record(db, model_id)
    domain = _model_domain(model_record)
    cache_key = str(model_record.model_id)

    if cache_key not in _timelines:
        model_pt_path = Path(model_record.model_path)
        timeline_path = resolve_timeline_path(model_pt_path)
        try:
            _timelines[cache_key] = load_timeline(timeline_path)
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))

    timeline = _timelines[cache_key]
    stamps = [str(np.datetime64(t, "s")) for t in timeline["timestamps"]]

    # Intersect against the window this model's domain actually has loaded. Building the
    # domain would be wasteful for a chart, so an unbuilt one reports no coverage rather
    # than paying to find out -- the frontend refetches once the window is up.
    covered: list[str] = []
    if manager.is_initialized(domain):
        window = set(manager.get_current_window(domain).timestamps)
        covered = [s for s in stamps if s in window]
    stale = bool(stamps) and not covered and manager.is_initialized(domain)
    if stale:
        logger.warning(
            "[timeline] '%s' covers %s..%s but the loaded '%s' window shares none of it — "
            "the precomputed file is for a different dataset and needs regenerating.",
            model_record.name, stamps[0], stamps[-1], domain,
        )

    return {
        "range_start": str(np.datetime64(timeline["range_start"], "s")),
        "range_end": str(np.datetime64(timeline["range_end"], "s")),
        "timestamps": stamps,
        "mean_error": timeline["mean_error"],
        "peak_error": timeline["peak_error"],
        "error_concentration": timeline["error_concentration"],
        # Deprecated alias of mean_error, kept for one release.
        "rmse": timeline["mean_error"],
        # Which of the above can actually be scrubbed to, and whether the artifact is stale.
        "covered_timestamps": covered,
        "stale": stale,
        # Derived from the timeline actually being served, not from manager state. The bands
        # describe this series, and taking them from the manager would attach ERA5's 2020
        # bands to an October 2024 ocean series -- every band would miss and the frontend
        # would silently drop it (ExplainTimeline.tsx guards on indexOf < 0). Equivalent to
        # the old behaviour for ERA5, whose manager bands come from this same timeline.
        # Derived from `stamps`, NOT from timeline["timestamps"]. The frontend locates a
        # band with timestamps.indexOf(band.start) and silently drops it on a miss
        # (ExplainTimeline.tsx guards on indexOf < 0), so the bands must be built from the
        # exact strings served above. The raw artifact timestamps carry nanosecond precision
        # ("2020-01-01T06:00:00.000000000") while `stamps` are normalised to seconds, so
        # deriving from the raw list dropped every ERA5 band. The window manager always
        # normalised first (era5_window_manager.py:108); this is what "equivalent to the old
        # behaviour for ERA5" was meant to mean. Ocean was unaffected only because
        # _run_ocean already writes second-precision timestamps.
        "month_bands": [
            {"label": b.label, "start": b.start, "end": b.end, "count": b.count}
            for b in derive_month_bands(stamps)
        ],
    }


@router.get("/{model_id}/prediction_errors/timeline/components")
def get_errors_timeline_components(model_id: str, db: Session = Depends(get_db)):
    """
    Per-timestep top-3 peak components, for marking them in the right panel.

    Served separately from the timeline itself because it is only needed once a user
    looks at the Peak error series, and it is the one part of the artifact that scales
    with d_state rather than being three floats.

    Response: { timestamps, peak_top3: [[i, j, k], ...], column_names: [...] }

    peak_top3 holds indices into column_names; -1 pads a domain with fewer than three
    components. Names follow the domain's own layout -- "vertical_velocity@150" for an
    ERA5 atmospheric column, a bare name for an ERA5 surface variable, and a fully
    qualified "thetao_47m" for the ocean, whose channels carry their depth and have no
    separate level axis. Splitting on "@" is therefore the only decoding needed, and it
    yields no level for exactly the columns that have none.

    404s when the artifact predates peak_top3, rather than returning empty arrays that
    the caller would have to tell apart from "no peaks".
    """
    model_record = _resolve_model_record(db, model_id)
    cache_key = str(model_record.model_id)

    if cache_key not in _timelines:
        model_pt_path = Path(model_record.model_path)
        timeline_path = resolve_timeline_path(model_pt_path)
        try:
            _timelines[cache_key] = load_timeline(timeline_path)
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))

    timeline = _timelines[cache_key]
    if timeline["peak_top3"] is None or timeline["column_names"] is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"'{model_record.name}' has no peak component data. Its timeline artifact "
                "predates peak_top3 — re-run scripts/precompute_timeline.py for this model."
            ),
        )

    return {
        "timestamps": [str(np.datetime64(t, "s")) for t in timeline["timestamps"]],
        "peak_top3": timeline["peak_top3"],
        "column_names": timeline["column_names"],
    }
