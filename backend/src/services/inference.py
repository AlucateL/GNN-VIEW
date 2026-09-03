"""
Model loaders for neural-lam inference.

Two entry points:
  load_torchscript(pt_path)   — loads a TorchScript .pt export, no neural_lam needed.
  load_checkpoint(ckpt_path)  — loads a PL .ckpt, reconstructs the full nn.Module from neural-lam/neural_lam/model

Both functions cache on the resolved path so repeated calls are free.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project-relative paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT    = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[3]))
_MODEL_ROOT      = Path(os.environ.get("MODEL_ROOT", _PROJECT_ROOT / "models"))
_NEURAL_LAM_PATH = Path(os.environ.get("NEURAL_LAM_PATH", _MODEL_ROOT / "neural-lam"))
_DATA_ROOT: Path = Path(os.environ["DATA_ROOT"]) if "DATA_ROOT" in os.environ else _PROJECT_ROOT / "data"
_GRAPHS_BASE     = Path(os.environ.get("GRAPHS_ROOT", _DATA_ROOT / "graphs"))
_GRAPH_FALLBACKS = ("global_multiscale", "global_hierarchical")

# Per-checkpoint inference overrides (flags not reliably stored in ckpt hparams).
# Canonical source: model_configs.json co-located with the paper checkpoints.
_MODEL_CONFIGS_PATH = _DATA_ROOT / "paper_checkpoints" / "model_configs.json"

def _load_model_configs() -> dict:
    if _MODEL_CONFIGS_PATH.exists():
        with open(_MODEL_CONFIGS_PATH) as f:
            cfg = json.load(f)
        return {k: v for k, v in cfg.items() if not k.startswith("_")}
    return {}

_MODEL_CONFIGS: dict = _load_model_configs()

# ── Configuration — swap these when upgrading datasets or models ──────────────
MODEL_PATH: Path = _DATA_ROOT / "paper_checkpoints" / "graphcast_scripted.pt"
# Switch to .ckpt to use the full PL checkpoint instead:
# MODEL_PATH: Path = _DATA_ROOT / "paper_checkpoints" / "graphcast.ckpt"

# Preset models available for selection in the UI
PRESET_MODELS: dict[str, Path] = {
    "graphcast":    _DATA_ROOT / "paper_checkpoints" / "graphcast_scripted.pt",
    "hierarchical": _DATA_ROOT / "paper_checkpoints" / "graph_fm_scripted.pt",
}

# Seed window the ERA5WindowManager boots from. The 9-step Aug-2020 subset is
# shipped instead of the full 124-step aug2020.zarr (903 MB) to keep the repo
# under GitHub's push limit; any other month is streamed from GCS on demand via
# the timeline band UI, so only the pre-scrub default window is narrower.
FIELDS_ZARR_PATH: Path = (
    _DATA_ROOT / "era5_datasets" / "small_2020-08-27T00:00_for_10_steps.zarr"
)
STATIC_FILES_PATH: Path = _DATA_ROOT / "era5_datasets" / "static"
# Global TOA normalization constant recovered from forcing.zarr.
# None → fall back to local-window normalization (incorrect for short subsets).
TOA_NORM_PATH: Path | None = _DATA_ROOT / "era5_datasets" / "forcing" / "toa_solar_radiation_max.pt"

GRID_LAT_LON_PATH: Path = _DATA_ROOT / "grid" / "grid_lat_lon.pt"
_PRECOMPUTED_DIR:  Path = _DATA_ROOT / "precomputed"

DEVICE: str = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

_MODEL_CLASSES: dict | None = None   # populated lazily on first checkpoint load


# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------

_jit_cache:  dict[str, torch.jit.ScriptModule] = {}
_ckpt_cache: dict[str, torch.nn.Module]         = {}


# ---------------------------------------------------------------------------
# TorchScript loader
# ---------------------------------------------------------------------------

def load_torchscript(
    pt_path: str | Path,
    device: str | None = None,
) -> torch.jit.ScriptModule:
    """
    Load a TorchScript .pt model exported from a neural-lam checkpoint.

    The returned module is callable as:
        pred = model(prev_state, prev_prev_state, forcing)  # (B, N_grid, d_state)

    No neural_lam imports are required. Input gradients flow through the traced
    graph normally, so saliency maps and integrated gradients work as expected.

    Args:
        pt_path: Path to the .pt file produced by export_torchscript.py.
        device:  'cpu', 'cuda', etc. Defaults to cuda if available.

    Returns:
        torch.jit.ScriptModule in eval mode on the requested device.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pt_path = Path(pt_path).resolve()
    cache_key = f"{pt_path}|{device}"

    if cache_key in _jit_cache:
        return _jit_cache[cache_key]

    if not pt_path.exists():
        raise FileNotFoundError(f"TorchScript model not found: {pt_path}")

    model = torch.jit.load(str(pt_path), map_location=device)
    model.eval()
    _jit_cache[cache_key] = model
    return model


# ---------------------------------------------------------------------------
# Checkpoint loader helpers
# ---------------------------------------------------------------------------

class _ModelArgs:
    """Plain namespace for model constructor args — module-level so it's picklable."""
    pass


def _ensure_neural_lam() -> dict:
    """Lazily import neural_lam model classes, adding the submodule to sys.path once."""
    global _MODEL_CLASSES
    if _MODEL_CLASSES is not None:
        return _MODEL_CLASSES

    candidates = [
        _NEURAL_LAM_PATH,
    ]
    nl_path = next((p for p in candidates if p.exists()), None)
    if nl_path is None:
        raise ModuleNotFoundError(
            "neural_lam sources not found. Expected a 'neural-lam' directory under a "
            "model root (e.g. /models/neural-lam or <repo>/models/neural-lam). "
            "Mount the models folder into the container or set NEURAL_LAM_PATH/MODEL_ROOT."
        )

    nl_path_str = str(nl_path)
    if nl_path_str not in sys.path:
        sys.path.insert(0, nl_path_str)

    from neural_lam.models.graph_efm import GraphEFM   # noqa: PLC0415
    from neural_lam.models.graph_fm  import GraphFM    # noqa: PLC0415
    from neural_lam.models.graphcast import GraphCast  # noqa: PLC0415

    _MODEL_CLASSES = {
        "graphcast": GraphCast,
        "graph_fm":  GraphFM,
        "graph_efm": GraphEFM,
    }
    return _MODEL_CLASSES


def _read_hparams(ckpt_path: str | Path) -> object:
    """
    Return the hparams namespace stored in a PL checkpoint.
    PL nests them under hp["args"] when save_hyperparameters() receives an
    argparse Namespace; falls back to a flat dict otherwise.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    return hp.get("args", hp)


def _get(hp: object, key: str, default):
    """Fetch a value from either a namespace or a dict."""
    if hasattr(hp, key):
        return getattr(hp, key)
    if isinstance(hp, dict):
        return hp.get(key, default)
    return default


def _build_args(model_name: str, ckpt_path: str | Path) -> _ModelArgs:
    hp   = _read_hparams(ckpt_path)
    cfg  = _MODEL_CONFIGS.get(model_name, {})  # overrides from model_configs.json

    def _cfg_or_hp(key, default):
        if key in cfg:
            return cfg[key]
        return _get(hp, key, default)

    args = _ModelArgs()
    args.model                    = model_name
    args.dataset                  = ""   # overridden in load_checkpoint from STATIC_FILES_PATH
    args.graph                    = _cfg_or_hp("graph", "global_multiscale")
    args.hidden_dim               = _cfg_or_hp("hidden_dim", 64)
    args.hidden_layers            = _cfg_or_hp("hidden_layers", 1)
    args.processor_layers         = _cfg_or_hp("processor_layers", 4)
    args.prior_processor_layers   = _cfg_or_hp("prior_processor_layers", 2)
    args.encoder_processor_layers = _cfg_or_hp("encoder_processor_layers", 2)
    args.mesh_aggr                = _cfg_or_hp("mesh_aggr", "sum")
    args.ar_steps                 = _cfg_or_hp("ar_steps", 1)
    args.lr                       = _cfg_or_hp("lr", 1e-3)
    args.loss                     = _cfg_or_hp("loss", "mse")
    args.step_length              = _cfg_or_hp("step_length", 6)
    args.num_past_forcing_steps   = _cfg_or_hp("num_past_forcing_steps", 1)
    args.num_future_forcing_steps = _cfg_or_hp("num_future_forcing_steps", 1)
    args.eval_leads               = _cfg_or_hp("eval_leads", 1)
    args.vertical_propnets        = _cfg_or_hp("vertical_propnets", False)
    args.output_std               = _cfg_or_hp("output_std", False)
    args.ensemble_size            = _cfg_or_hp("ensemble_size", 1)
    args.eval                     = None
    args.n_example_pred           = 0
    args.batch_size               = 1
    args.restore_opt              = False
    args.load                     = None
    args.save_forecasts           = False
    args.n_workers                = 0
    args.seed                     = 42
    return args


def _resolve_graph(args: _ModelArgs) -> None:
    """
    Replace args.graph with a graph directory that exists under _GRAPHS_BASE.
    Checkpoints store the training-time name (e.g. 'global_ms4') which may not
    match any local directory.
    """
    if (_GRAPHS_BASE / args.graph).exists():
        return
    for candidate in _GRAPH_FALLBACKS:
        if (_GRAPHS_BASE / candidate).exists():
            print(f"[inference] Graph '{args.graph}' not found — using '{candidate}'")
            args.graph = candidate
            return
    raise FileNotFoundError(
        f"No graph directory found under {_GRAPHS_BASE}. "
        f"Tried '{args.graph}' and fallbacks {_GRAPH_FALLBACKS}."
    )


def _detect_model_name(ckpt_path: str | Path, model_classes: dict) -> str:
    """Infer the model type from the checkpoint's stored hparams or filename."""
    hp   = _read_hparams(ckpt_path)
    name = _get(hp, "model", None)
    if name and name in model_classes:
        return name
    stem = Path(ckpt_path).stem.lower()
    for key in model_classes:
        if key in stem:
            return key
    raise ValueError(
        f"Could not detect model type from checkpoint. "
        f"Stored name: '{name}'. Known types: {list(model_classes)}. "
        f"Pass model_name explicitly to load_checkpoint()."
    )


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_checkpoint(
    ckpt_path: str | Path,
    model_name: str | None = None,
    device: str | None = None,
) -> torch.nn.Module:
    """
    Load a neural-lam PL checkpoint and return a ready-to-use nn.Module.

    The returned model:
      - is in eval() mode with gradients enabled (suitable for saliency / IG)
      - has the correct architecture reconstructed from the checkpoint's own
        stored hyperparameters (hidden_dim, graph, etc.)
      - uses normalization stats from STATIC_FILES_PATH (set at the top of this file)

    Usage:
        model = load_checkpoint("data/paper_checkpoints/graphcast.ckpt")
        pred, _ = model.predict_step(prev_state, prev_prev_state, forcing)

        # gradient-based attribution
        prev_state.requires_grad_(True)
        pred, _ = model.predict_step(prev_state, prev_prev_state, forcing)
        grad = torch.autograd.grad(pred.sum(), prev_state)[0]

    Args:
        ckpt_path:   Path to the .ckpt file.
        model_name:  Model architecture ('graphcast', 'graph_fm', 'graph_efm').
                     Auto-detected from the checkpoint if omitted.
        device:      'cpu', 'cuda', etc. Defaults to cuda if available.

    Returns:
        nn.Module in eval() mode, gradients enabled, on the requested device.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = Path(ckpt_path).resolve()
    cache_key = f"{ckpt_path}|{model_name}|{device}"

    if cache_key in _ckpt_cache:
        return _ckpt_cache[cache_key]

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model_classes = _ensure_neural_lam()

    if model_name is None:
        model_name = _detect_model_name(ckpt_path, model_classes)
        print(f"[inference] Detected model type: '{model_name}'")

    if model_name not in model_classes:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from {list(model_classes)}."
        )

    # neural_lam resolves both graph and dataset paths relative to CWD via
    # os.path.join("graphs", name) and os.path.join("data", name, "static").
    # Injecting an absolute path as the middle component causes Python to drop
    # the preceding segment, making resolution CWD-independent.
    args = _build_args(model_name, ckpt_path)
    _resolve_graph(args)
    args.graph   = str(_GRAPHS_BASE / args.graph)   # absolute, CWD-independent
    # neural_lam resolves os.path.join("data", dataset, "static"); injecting an
    # absolute path as the middle component makes Python drop the "data/" prefix.
    args.dataset = str(STATIC_FILES_PATH.parent)    # → STATIC_FILES_PATH/static

    model_cls = model_classes[model_name]
    model = model_cls.load_from_checkpoint(
        str(ckpt_path), args=args, strict=True, map_location=device
    )
    model.eval()
    # Do NOT call requires_grad_(False) — gradients through the model are needed
    # for saliency maps and integrated gradients.

    _ckpt_cache[cache_key] = model
    return model


# ---------------------------------------------------------------------------
# Unified model loader
# ---------------------------------------------------------------------------

def load_model(path: str | Path | None = None, device: str | None = None) -> torch.nn.Module:
    """
    Load a model from a .pt (TorchScript) or .ckpt (PyTorch Lightning) file.
    Defaults to MODULE_PATH when path is omitted.
    """
    path = Path(path) if path is not None else MODEL_PATH
    device = device or DEVICE
    if path.suffix == ".pt":
        return load_torchscript(path, device=device)
    if path.suffix == ".ckpt":
        return load_checkpoint(path, device=device)
    raise ValueError(f"Unknown model extension '{path.suffix}'. Expected .pt or .ckpt.")


# ---------------------------------------------------------------------------
# ERA5 data loading
# ---------------------------------------------------------------------------
# Load ERA5 data from a local zarr written by fetch_era5_and_inference.ipynb
# and prepare the normalized (prev_state, prev_prev_state, forcing) tensors
# expected by model.predict_step().
# Normalization stats come from STATIC_FILES_PATH — same source as load_checkpoint().

_nlam_constants = None   # populated lazily by _ensure_constants()


def _ensure_constants():
    global _nlam_constants
    if _nlam_constants is not None:
        return _nlam_constants
    _ensure_neural_lam()              # adds models/neural-lam to sys.path
    from neural_lam import constants  # noqa: PLC0415
    _nlam_constants = constants
    return constants


def _stack_state(ds) -> np.ndarray:
    """
    Stack model state variables from an xr.Dataset into (T, N_grid, d_state).
    Order: ATMOSPHERIC_PARAMS (×levels) then SURFACE_PARAMS — matches neural-lam.
    """
    c = _ensure_constants()
    T     = len(ds.time)
    Nlon  = len(ds.longitude)
    Nlat  = len(ds.latitude)
    Ngrid = Nlon * Nlat

    atm_vars  = [v for v in c.ATMOSPHERIC_PARAMS if v in ds.data_vars]
    surf_vars = [v for v in c.SURFACE_PARAMS      if v in ds.data_vars]

    atm = (
        ds[atm_vars]
        .to_dataarray("state_var")
        .transpose("time", "longitude", "latitude", "state_var", "level")
        .values.astype(np.float32)
        .reshape(T, Ngrid, len(atm_vars) * len(c.PRESSURE_LEVELS))
    )
    surf = (
        ds[surf_vars]
        .to_dataarray("state_var")
        .transpose("time", "longitude", "latitude", "state_var")
        .values.astype(np.float32)
        .reshape(T, Ngrid, len(surf_vars))
    )
    return np.concatenate([atm, surf], axis=-1)   # (T, N_grid, d_state)


def _load_toa_normalization(norm_path: Path) -> tuple[float, float] | None:
    """
    Load (toa_min=0, toa_max) from toa_solar_radiation_max.pt.
    Returns None if the file does not exist.
    """
    if not norm_path.exists():
        print(f"[inference] TOA normalization file not found: {norm_path}\n")
        return None
    stats = torch.load(str(norm_path), map_location="cpu", weights_only=True)
    return 0.0, float(stats["toa_max"])


def _compute_forcing(ds, toa_norm: tuple[float, float] | None = None) -> np.ndarray:
    """
    Compute forcing features with the same datetime logic as neural-lam's
    create_global_forcing.py (GraphCast data_utils + solar_radiation).

    Feature order (kept checkpoint-compatible):
      [cos_day_progress, cos_year_progress, sin_day_progress,
       sin_year_progress, toa_incident_radiation]

    toa_norm: (toa_min, toa_max) from the full training dataset, produced by
        scripts/extract_toa_subset.py. When provided, TOA is computed on the fly
        and normalized with global statistics. When None, local-window min/max is
        used which is incorrect for short subsets.

    Returns:
      (T, N_grid, 5) float32.
    """
    time_vals = ds.time.values.astype("datetime64[s]")
    lon = ds.longitude.values.astype(np.float32)
    lat = ds.latitude.values.astype(np.float32)
    T, Nlon, Nlat = len(time_vals), len(lon), len(lat)
    Ngrid = Nlon * Nlat

    sec = time_vals.astype(np.int64)

    try:
        import graphcast.data_utils as _gc_du  # noqa: PLC0415

        year_progress = _gc_du.get_year_progress(sec)  # (T,)
        day_progress = _gc_du.get_day_progress(sec, lon)  # (T, Nlon)
    except Exception:
        # Fallback if graphcast utilities are unavailable.
        year_start = (
            time_vals.astype("datetime64[Y]").astype("datetime64[s]").astype(np.int64)
        )
        year_next = (
            (time_vals.astype("datetime64[Y]") + np.timedelta64(1, "Y"))
            .astype("datetime64[s]").astype(np.int64)
        )
        year_progress = (sec - year_start) / np.maximum(year_next - year_start, 1)

        utc_day_frac = (sec % 86400) / 86400.0
        day_progress = (utc_day_frac[:, None] + lon[None, :] / 360.0) % 1.0

    sin_year = ((np.sin(2 * np.pi * year_progress) + 1.0) / 2.0).astype(np.float32)
    cos_year = ((np.cos(2 * np.pi * year_progress) + 1.0) / 2.0).astype(np.float32)
    sin_day = ((np.sin(2 * np.pi * day_progress) + 1.0) / 2.0).astype(np.float32)
    cos_day = ((np.cos(2 * np.pi * day_progress) + 1.0) / 2.0).astype(np.float32)

    try:
        import graphcast.solar_radiation as _gc_sr  # noqa: PLC0415
        toa = _gc_sr.get_toa_incident_solar_radiation(time_vals, lat, lon).astype(np.float32)
        toa = np.transpose(toa, (0, 2, 1))  # → (T, Nlon, Nlat)
        if toa_norm is not None:
            toa_min, toa_max = toa_norm
        else:
            # Local-window fallback: normalization will be off for short subsets
            toa_min, toa_max = float(toa.min()), float(toa.max())
        toa = (
            (toa - toa_min) / (toa_max - toa_min)
            if toa_max > toa_min
            else np.zeros((T, Nlon, Nlat), dtype=np.float32)
        )
    except Exception:
        toa = np.zeros((T, Nlon, Nlat), dtype=np.float32)

    # Broadcast scalars to (T, Nlon, Nlat) before stacking
    cos_day_3d  = np.repeat(cos_day[:, :, None],       Nlat, axis=2)
    sin_day_3d  = np.repeat(sin_day[:, :, None],       Nlat, axis=2)
    cos_year_3d = np.repeat(cos_year[:, None, None], Nlon, axis=1).repeat(Nlat, axis=2)
    sin_year_3d = np.repeat(sin_year[:, None, None], Nlon, axis=1).repeat(Nlat, axis=2)

    return np.stack(
        [cos_day_3d, cos_year_3d, sin_day_3d, sin_year_3d, toa], axis=-1
    ).reshape(T, Ngrid, 5).astype(np.float32)


def _load_static(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (data_mean, data_std) from STATIC_FILES_PATH."""
    if not STATIC_FILES_PATH.exists():
        raise FileNotFoundError(
            f"Static dir not found: {STATIC_FILES_PATH}\n"
            "Run fetch_era5_and_inference.ipynb to create it, or update "
            "STATIC_FILES_PATH at the top of this file."
        )
    mean = torch.load(STATIC_FILES_PATH / "parameter_mean.pt", map_location=device, weights_only=True)
    std  = torch.load(STATIC_FILES_PATH / "parameter_std.pt",  map_location=device, weights_only=True)
    return mean, std


# Small LRUs on the two uncached ERA5 zarr reads (§S8). Both are keyed on `idx`
# alone and store CPU tensors — the expensive part is the zarr load + stack +
# normalization, not the (cheap) device transfer, which is applied after the
# cache lookup so one cache entry serves every device. `_clear_era5_slice_caches`
# is called from era5_window_manager.load_window as a guard: if the active ERA5
# window ever changes at runtime, a stale idx→tensor mapping must not leak across
# windows. Today GCS_FETCH_ENABLED is false and the window never changes, but the
# guard keeps that assumption from being silently load-bearing.
_ERA5_SLICE_CACHE_MAX = 4
_era5_lazy_window_cache: OrderedDict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = OrderedDict()
_era5_state_cache: OrderedDict[int, torch.Tensor] = OrderedDict()


def _clear_era5_slice_caches() -> None:
    """Drop cached ERA5 slices. Call whenever the active window changes."""
    _era5_lazy_window_cache.clear()
    _era5_state_cache.clear()


def _load_lazy_window(
    era5_cache: dict[str, Any],
    idx: int,
    device: str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load timesteps [idx-2, idx-1, idx] from the lazy zarr cache on demand."""
    cached = _era5_lazy_window_cache.get(idx)
    if cached is not None:
        _era5_lazy_window_cache.move_to_end(idx)
        prev, prev_prev, forcing = cached
    else:
        window_ds = era5_cache["ds"].isel(time=slice(idx - 2, idx + 1)).load()

        state_np   = _stack_state(window_ds)                              # (3, N_grid, d_state)
        forcing_np = _compute_forcing(window_ds, era5_cache["toa_norm"])  # (3, N_grid, 5)

        # Forcing window at prediction time idx: stack [t-2, t-1, t] on last axis
        forcing_win_np = np.concatenate(
            [forcing_np[0], forcing_np[1], forcing_np[2]], axis=-1
        )[None]  # (1, N_grid, 15)

        data_mean = era5_cache["data_mean"].cpu()
        data_std  = era5_cache["data_std"].cpu()
        state_t   = torch.tensor(state_np, dtype=torch.float32)
        state_norm = (state_t - data_mean) / data_std  # (3, N_grid, d_state)

        prev_prev = state_norm[0].unsqueeze(0)  # state at idx-2
        prev      = state_norm[1].unsqueeze(0)  # state at idx-1
        forcing   = torch.tensor(forcing_win_np, dtype=torch.float32)

        _era5_lazy_window_cache[idx] = (prev, prev_prev, forcing)
        if len(_era5_lazy_window_cache) > _ERA5_SLICE_CACHE_MAX:
            _era5_lazy_window_cache.popitem(last=False)

    if device is not None:
        prev_prev = prev_prev.to(device)
        prev      = prev.to(device)
        forcing   = forcing.to(device)

    return prev, prev_prev, forcing


def get_era5_normalized_state(
    era5_cache: dict[str, Any],
    idx: int,
    device: str | None = None,
) -> torch.Tensor:
    """
    Return the normalized state tensor at time index idx.

    Works with both lazy and eager caches. Used by the precompute script and
    compute_prediction_errors to obtain ground-truth in normalized space.
    """
    # Ocean is checked first: its cache also sets _lazy, but the ERA5 lazy branch below
    # reads cache["ds"] and stacks ERA5 variable names, neither of which an ocean cache has.
    if era5_cache.get("_ocean"):
        from services.ocean import ocean_state_at  # noqa: PLC0415

        state_norm = ocean_state_at(era5_cache, idx, device)
    elif era5_cache.get("_lazy"):
        cached = _era5_state_cache.get(idx)
        if cached is not None:
            _era5_state_cache.move_to_end(idx)
            state_norm = cached
        else:
            slice_ds   = era5_cache["ds"].isel(time=slice(idx, idx + 1)).load()
            state_np   = _stack_state(slice_ds)  # (1, N_grid, d_state)
            data_mean  = era5_cache["data_mean"].cpu()
            data_std   = era5_cache["data_std"].cpu()
            state_norm = (torch.tensor(state_np[0], dtype=torch.float32) - data_mean) / data_std
            _era5_state_cache[idx] = state_norm
            if len(_era5_state_cache) > _ERA5_SLICE_CACHE_MAX:
                _era5_state_cache.popitem(last=False)
    else:
        state_norm = era5_cache["state_normalized"][idx]

    if device is not None:
        state_norm = state_norm.to(device)
    return state_norm


def preload_era5_tensors(
    fields_zarr: str | Path,
    time_start: str | None = None,
    n_steps: int | None = None,
    device: str = "cpu",
    toa_norm_path: str | Path | None = "auto",
    lazy: bool = True,
    storage_options: dict | None = None,
) -> dict[str, Any]:
    """
    Open an ERA5 zarr and return a cache dict for downstream inference helpers.

    fields_zarr may be a local path or a remote URL (gs://, s3://, filecache::gs://, …).
    For GCS access pass storage_options, e.g.:
        storage_options={
            "filecache": {"cache_storage": "/tmp/era5_cache"},
            "gs": {"token": "anon"},
        }
    The filecache layer saves downloaded chunks to disk so repeat accesses
    to the same timestep window are served locally at <50 ms.

    lazy=True (default): only the time coordinate and static stats are loaded at
    startup (~2 MB total). Per-timestep data is read on demand inside
    get_era5_step_inputs — RAM stays flat regardless of dataset size.

    lazy=False: full eager load (original behaviour). Useful for short dev datasets
    or notebooks where repeated random access into the tensor is convenient.

    toa_norm_path:
      "auto" (default) — use TOA_NORM_PATH from module config if it exists.
      Path             — explicit path to a toa_normalization.pt file.
      None             — skip; compute TOA with local-window min/max (incorrect
                         for short subsets).
    """
    import xarray as xr

    zarr_url = str(fields_zarr)
    is_remote = "://" in zarr_url

    if is_remote:
        if zarr_url.startswith("gs://"):
            # gcsfs mapper — works with zarr 3 (filecache:: wrapper is incompatible
            # with zarr 3's async _cat_file protocol).
            import gcsfs  # noqa: PLC0415
            gs_opts = (storage_options or {}).get("gs", {})
            token = gs_opts.get("token", "anon")
            fs = gcsfs.GCSFileSystem(token=token)
            store = fs.get_mapper(zarr_url)
            ds = xr.open_zarr(store, consolidated=True)
        else:
            ds = xr.open_zarr(zarr_url, storage_options=storage_options or {}, consolidated=True)
    else:
        fields_zarr = Path(zarr_url)
        if not fields_zarr.is_absolute():
            fields_zarr = _DATA_ROOT / fields_zarr
        if not fields_zarr.exists():
            raise FileNotFoundError(f"fields.zarr not found: {fields_zarr}")
        ds = xr.open_zarr(str(fields_zarr), consolidated=True)
    if time_start is not None:
        ds = ds.sel(time=slice(time_start, None))
    if n_steps is not None:
        ds = ds.isel(time=slice(0, n_steps))

    # Time coordinate is tiny — read it regardless of lazy flag.
    time_vals = ds.time.values.astype("datetime64[ns]")
    T = len(time_vals)
    if T < 3:
        raise ValueError(f"Need at least 3 timesteps, got {T}")

    c = _ensure_constants()
    atm_vars = [v for v in c.ATMOSPHERIC_PARAMS if v in ds.data_vars]
    surf_vars = [v for v in c.SURFACE_PARAMS if v in ds.data_vars]

    # Resolve TOA normalization constants
    if toa_norm_path == "auto":
        toa_norm_path = TOA_NORM_PATH
    toa_norm: tuple[float, float] | None = None
    if toa_norm_path is not None:
        toa_norm = _load_toa_normalization(Path(toa_norm_path))
        if toa_norm is not None:
            print(f"[inference] Using global TOA normalization from {Path(toa_norm_path).name}")
        else:
            print("[inference] toa_normalization.pt not found — using local-window normalization (may be off).")

    data_mean, data_std = _load_static(device)

    if lazy:
        # Store the open (unloaded) dataset. Each get_era5_step_inputs call will
        # slice exactly 3 timesteps and call .load() on just those (~10 MB freed
        # after each call). OS page cache warms naturally for hot timesteps.
        return {
            "_lazy": True,
            "ds": ds,
            "time": time_vals,
            "toa_norm": toa_norm,
            "atm_vars": atm_vars,
            "surf_vars": surf_vars,
            "levels": [int(lv) for lv in c.PRESSURE_LEVELS],
            "data_mean": data_mean,
            "data_std": data_std,
        }

    # --- Eager path (lazy=False) ---
    ds = ds.load()
    state_np = _stack_state(ds)                      # (T, N_grid, d_state)
    forcing_np = _compute_forcing(ds, toa_norm)      # (T, N_grid, 5)

    forcing_win_np = np.concatenate(
        [forcing_np[:-2], forcing_np[1:-1], forcing_np[2:]], axis=-1
    )  # (T-2, N_grid, 15)

    state_t = torch.tensor(state_np, dtype=torch.float32, device=device)
    state_norm = (state_t - data_mean) / data_std
    forcing_win = torch.tensor(forcing_win_np, dtype=torch.float32, device=device)

    return {
        "state_normalized": state_norm,
        "forcing_window": forcing_win,
        "time": time_vals,
        "atm_vars": atm_vars,
        "surf_vars": surf_vars,
        "levels": [int(lv) for lv in c.PRESSURE_LEVELS],
        "data_std": data_std,
    }


def _to_datetime64_ns(pred_time: str | np.datetime64) -> np.datetime64:
    """Parse timestamp input into numpy datetime64[ns]."""
    if isinstance(pred_time, np.datetime64):
        return pred_time.astype("datetime64[ns]")
    if isinstance(pred_time, str) and pred_time.endswith("Z"):
        pred_time = pred_time[:-1]
    return np.datetime64(pred_time, "ns")


def get_available_pred_timestamps(
    era5_cache: dict[str, Any],
    as_str: bool = True,
) -> list[str] | np.ndarray:
    """
    Return valid prediction timestamps from a preloaded ERA5 cache.

    Notes:
      - Cache stores full time axis of length T.
      - Valid prediction indices are [2, T-1], so valid prediction timestamps
        are time[2:].

    Args:
        era5_cache: Output of preload_era5_tensors(...).
        as_str: If True, return ISO strings; else return np.datetime64[ns] array.
    """
    time_vals = era5_cache.get("time")
    if time_vals is None:
        raise ValueError(
            "ERA5 cache has no 'time'. Recreate cache with preload_era5_tensors()."
        )

    pred_times = time_vals[2:];  # pred_idx valid range starts at 2
    if as_str:
        return [str(t) for t in pred_times]
    return pred_times


def list_era5_timestamps(
    fields_zarr: str | Path,
    time_start: str | None = None,
    n_steps: int | None = None,
    prediction_only: bool = False,
    as_str: bool = True,
) -> list[str] | np.ndarray:
    """
    Read available timestamps directly from an ERA5 zarr without preloading tensors.

    Args:
        fields_zarr: Path to fields.zarr (absolute or project-relative).
        time_start: Optional lower bound for time selection.
        n_steps: Optional cap on number of steps after slicing.
        prediction_only: If True, return only times valid as prediction targets
            for get_era5_step_inputs/run_inference, i.e. time[2:].
        as_str: If True, return ISO strings; else return np.datetime64[ns] array.
    """
    import xarray as xr

    fields_zarr = Path(fields_zarr)
    if not fields_zarr.is_absolute():
        fields_zarr = _DATA_ROOT / fields_zarr
    if not fields_zarr.exists():
        raise FileNotFoundError(f"fields.zarr not found: {fields_zarr}")

    ds = xr.open_zarr(str(fields_zarr), consolidated=True)
    if time_start is not None:
        ds = ds.sel(time=slice(time_start, None))
    if n_steps is not None:
        ds = ds.isel(time=slice(0, n_steps))

    time_vals = ds.time.values.astype("datetime64[ns]")
    if prediction_only:
        time_vals = time_vals[2:]

    if as_str:
        return [str(t) for t in time_vals]
    return time_vals


def resolve_pred_idx(
    era5_cache: dict[str, Any],
    pred_idx: int | None = None,
    pred_time: str | np.datetime64 | None = None,
    nearest: bool = False,
) -> int:
    """
    Resolve prediction index either from pred_idx or pred_time.

    Valid prediction indices are [2, T-1].
    """
    if era5_cache.get("_lazy"):
        T = len(era5_cache["time"])
    else:
        T = era5_cache["state_normalized"].shape[0]

    if pred_idx is not None:
        idx = int(pred_idx)
    else:
        if pred_time is None:
            raise ValueError("Provide either pred_idx or pred_time.")

        time_vals = era5_cache.get("time")
        if time_vals is None:
            raise ValueError(
                "ERA5 cache has no 'time'. Recreate cache with preload_era5_tensors()."
            )

        target = _to_datetime64_ns(pred_time)
        if nearest:
            idx = int(np.argmin(np.abs(time_vals - target)))
        else:
            matches = np.where(time_vals == target)[0]
            if len(matches) == 0:
                raise ValueError(
                    f"Timestamp {target} not found in loaded ERA5 subset. "
                    f"Available range: [{time_vals[0]}, {time_vals[-1]}]"
                )
            idx = int(matches[0])

    if idx < 2 or idx > T - 1:
        raise ValueError(f"pred_idx={idx} out of range; valid range is [2, {T-1}]")

    return idx


def get_era5_step_inputs(
    era5_cache: dict[str, Any],
    pred_time: str | np.datetime64,
    device: str | None = None,
    nearest: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build one model input triplet to predict state at a selected timepoint.

    Meaning:
    pred_time = target time to predict
    prev_prev = state[idx - 2]
    prev      = state[idx - 1]
    forcing   = forcing_window[idx - 2]  # uses times [idx-2, idx-1, idx]
    """
    idx = resolve_pred_idx(era5_cache, pred_time=pred_time, nearest=nearest)

    if era5_cache.get("_lazy"):
        return _load_lazy_window(era5_cache, idx, device)

    state_norm  = era5_cache["state_normalized"]   # (T, N, d)
    forcing_win = era5_cache["forcing_window"]     # (T-2, N, 15)

    i         = idx - 2
    prev_prev = state_norm[i].unsqueeze(0)
    prev      = state_norm[i + 1].unsqueeze(0)
    forcing   = forcing_win[i].unsqueeze(0)

    if device is not None:
        prev_prev = prev_prev.to(device)
        prev      = prev.to(device)
        forcing   = forcing.to(device)

    return prev, prev_prev, forcing


def denormalize(pred: torch.Tensor) -> torch.Tensor:
    """
    Convert a normalized model output back to physical units using STATIC_FILES_PATH.

    Args:
        pred: (B, N_grid, d_state) tensor in normalized space.

    Returns:
        Tensor in physical units, same shape and device as pred.
    """
    mean, std = _load_static(str(pred.device))
    return pred * std + mean

# ---------------------------------------------------------------------------
# Grid loading
# ---------------------------------------------------------------------------

def load_grid(path: Path | None = None) -> dict[str, np.ndarray]:
    """
    Load the ERA5 inference grid from grid_lat_lon.pt.

    Returns { "lat": [...], "lon": [...] } with N_grid = 29,040 entries, as
    numpy arrays (ORJSONResponse serializes these directly, no list conversion).
    Longitude is normalised from [0, 360] to [-180, 180].
    """
    p = path or GRID_LAT_LON_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Grid file not found: {p}\n"
            "Copy data/grid/grid_lat_lon.pt into the project data directory."
        )
    g = torch.load(str(p), map_location="cpu", weights_only=True)  # (N_grid, 2)
    # Column selects are non-contiguous views; orjson's numpy passthrough requires
    # C-contiguous arrays.
    lat = g[:, 0].contiguous().numpy()
    lon = ((g[:, 1] + 180.0) % 360.0 - 180.0).contiguous().numpy()
    return {"lat": lat, "lon": lon}


# ---------------------------------------------------------------------------
# Timeline loading
# ---------------------------------------------------------------------------

# Directory of timeline artifacts baked into the image, set by backend/Dockerfile. Unset
# outside the container, where the artifacts sit next to their checkpoints already.
BUNDLED_TIMELINE_DIR: Path | None = (
    Path(os.environ["BUNDLED_TIMELINE_DIR"]) if os.environ.get("BUNDLED_TIMELINE_DIR") else None
)


def resolve_timeline_path(model_path: Path | str) -> Path:
    """
    Where to read '<model_stem>_timeline_errors.npy' from, bundled copy first.

    In production the checkpoint lives on the data-neurip PVC, which is written out of band
    and therefore drifts from the code: a deploy that adds a new series keeps serving the
    PVC's older artifact, and load_timeline degrades it to mean_error alone with the other
    chips disabled. The bundled copy is built from the same commit as the reader, so it wins.

    Falls through to the checkpoint's own directory when there is no bundled artifact for
    this stem -- which is every uploaded model, whose timeline is written beside it.
    """
    model_path = Path(model_path)
    name = f"{model_path.stem}_timeline_errors.npy"
    if BUNDLED_TIMELINE_DIR is not None:
        bundled = BUNDLED_TIMELINE_DIR / name
        if bundled.exists():
            return bundled
    return model_path.parent / name


def load_timeline(path: Path) -> dict:
    """
    Load the precomputed per-timestep error series from a .npy file.

    Returns { range_start, range_end, timestamps, mean_error, peak_error, peak_top3,
              error_concentration, column_names }.

    mean_error           RMSE over every scoreable (node, component) cell.
    peak_error           largest single component error anywhere on the grid, before
                         any averaging.
    peak_top3            (T, 3) component indices into column_names -- the three
                         components with the largest per-component maxima. -1 pads a
                         domain with fewer than three components.
    error_concentration  max over nodes of how many components put their own global
                         argmax on that node. A unit-free count, whose ceiling is
                         d_state and which is therefore not comparable across domains.
    column_names         d_state names, so peak_top3 is decodable without knowing the
                         domain layout ("vertical_velocity@150" / "thetao_47m").

    Backward compatibility: files written before this format stored only `rmse` (and,
    on one intermediate branch, a (T, d_state) `comp_max` matrix). Those are mapped
    forward -- `rmse` becomes mean_error, and peak_error / peak_top3 are derived from
    comp_max when it is there. error_concentration cannot be recovered from an old
    file and is returned as None, which disables its chip in the UI. Anything absent
    is None rather than an error, so an un-regenerated artifact degrades to the series
    it does carry instead of failing the request. Re-run scripts/precompute_timeline.py
    to populate the rest.

    Raises FileNotFoundError if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Timeline file not found: {path}. "
            "Expected a '<model_stem>_timeline_errors.npy' file co-located with the model."
        )

    data = np.load(path, allow_pickle=True).item()
    timestamps = [str(ts) for ts in data["timestamps"]]
    n_t = len(timestamps)

    def _series(key: str) -> list[float] | None:
        """A float series, or None when absent or the wrong length for these timestamps."""
        vals = data.get(key)
        if vals is None:
            return None
        vals = [float(v) for v in vals]
        return vals if len(vals) == n_t else None

    mean_error = _series("mean_error")
    peak_error = _series("peak_error")

    concentration = data.get("error_concentration")
    if concentration is not None:
        concentration = [int(v) for v in concentration]
        if len(concentration) != n_t:
            concentration = None

    top3 = data.get("peak_top3")
    if top3 is not None:
        top3 = np.asarray(top3, dtype=np.int64)
        top3 = top3.tolist() if top3.ndim == 2 and top3.shape[0] == n_t else None

    column_names = data.get("column_names")
    column_names = [str(c) for c in column_names] if column_names is not None else None

    # --- legacy formats -----------------------------------------------------
    if mean_error is None:
        mean_error = _series("rmse")
    if mean_error is None:
        raise ValueError(
            f"Timeline file {path} has neither 'mean_error' nor 'rmse'. "
            "Re-run scripts/precompute_timeline.py."
        )
    if peak_error is None or top3 is None:
        comp_max = data.get("comp_max")
        if comp_max is not None:
            comp_max = np.asarray(comp_max, dtype=np.float32)
            if comp_max.ndim == 2 and comp_max.shape[0] == n_t:
                if peak_error is None:
                    peak_error = comp_max.max(axis=1).tolist()
                if top3 is None:
                    k = min(3, comp_max.shape[1])
                    idx = np.argsort(-comp_max, axis=1)[:, :k]
                    if k < 3:                       # pad a degenerate d_state < 3
                        idx = np.pad(idx, ((0, 0), (0, 3 - k)), constant_values=-1)
                    top3 = idx.astype(np.int64).tolist()

    return {
        "range_start":         timestamps[0],
        "range_end":           timestamps[-1],
        "timestamps":          timestamps,
        "mean_error":          mean_error,
        "peak_error":          peak_error,
        "peak_top3":           top3,
        "error_concentration": concentration,
        "column_names":        column_names,
    }


# ---------------------------------------------------------------------------
# Error computation
# ---------------------------------------------------------------------------

def state_column_names(era5_cache: dict) -> list[str]:
    """
    Names of the d_state columns, in the cache's own channel order.

    ERA5 stacks atmospheric variables x pressure levels first, then surface variables, so a
    column is addressed as "var@level" or "var". The ocean datastore instead carries fully
    qualified names ("thetao_47m") in an order of its own. Both layouts are dense and neither
    errors when indexed with the other's arithmetic, so the two must never be mixed.
    """
    if era5_cache.get("_ocean"):
        return list(era5_cache["state_features"])
    atm_vars = era5_cache["atm_vars"]
    surf_vars = era5_cache["surf_vars"]
    levels = era5_cache["levels"]
    return [f"{v}@{lv}" for v in atm_vars for lv in levels] + list(surf_vars)


def resolve_state_columns(
    era5_cache: dict, variable: str | None, z_level: str | int | None
) -> list[int]:
    """
    Columns of d_state selected by (variable, z_level), for either domain.

    Returns the indices only; how they are aggregated is the caller's business. "total" in
    either position means "every value along that axis". Ocean features carry their depth in
    the name and have no separate level axis, so z_level is accepted only as "total"/None
    there -- silently ignoring a real level would hand back the surface field labelled as
    depth.
    """
    if variable is None:
        variable = "total"
    if z_level is None:
        z_level = "total"

    if era5_cache.get("_ocean"):
        names = era5_cache["state_features"]
        if z_level != "total":
            raise ValueError(
                f"Ocean features have no separate level axis -- depth is part of the name "
                f"(e.g. 'thetao_47m'). Got z_level={z_level!r}."
            )
        if variable == "total":
            return list(range(len(names)))
        if variable in names:
            return [names.index(variable)]
        raise ValueError(f"Unknown ocean feature '{variable}'. Available: {names}")

    atm_vars = era5_cache["atm_vars"]
    surf_vars = era5_cache["surf_vars"]
    levels = era5_cache["levels"]
    n_levels = len(levels)
    n_atm = len(atm_vars)

    if variable in surf_vars:
        return [n_atm * n_levels + surf_vars.index(variable)]
    if variable in atm_vars:
        i = atm_vars.index(variable)
        if z_level == "total":
            return list(range(i * n_levels, (i + 1) * n_levels))
        return [i * n_levels + levels.index(int(z_level))]
    if variable == "total":
        if z_level == "total":
            return list(range(n_atm * n_levels + len(surf_vars)))
        # All atmospheric variables at one level. Surface variables have no level, so they
        # are excluded rather than folded in at an arbitrary one.
        li = levels.index(int(z_level))
        return [v * n_levels + li for v in range(n_atm)]

    raise ValueError(
        f"Invalid variable='{variable}' or z_level='{z_level}'. "
        f"Atmospheric vars: {atm_vars}. Surface vars: {surf_vars}. Levels: {levels}."
    )


def compute_prediction_errors(
    model: torch.nn.Module,
    era5_cache: dict,
    pred_time: str,
    variable: str | None = None,
    z_level: str | int | None = None,
    model_cache_key: str = "default",
    device: str | None = None,
) -> np.ndarray:
    """
    Run inference and return normalised error per node.

    Uses run_inference_cached for the underlying inference, so switching variables
    for the same pred_time does NOT re-run the model — all cases slice from the
    same cached prediction tensor.

    variable: Variable name or "total" (default: "total").
    z_level:  Pressure level (as string, e.g. "500") or "total" (default: "total").
              Ignored for surface variables.

    Cases:
      1. variable=<specific>, z_level=<specific>
         → |truth - pred| per node (absolute error, no averaging)
      2. variable=<specific>, z_level="total"
         → √(mean((truth - pred)² over all levels)) per node
      3. variable="total", z_level=<specific>
         → √(mean((truth - pred)² over all variables at that level)) per node
      4. variable="total", z_level="total"
         → √(mean((truth - pred)² over all variables+levels)) per node
      5. variable=<surface>, z_level=<ignored>
         → |truth - pred| per node (surface has no levels)

    Returns:
        np.ndarray of length N_grid — error per node.
    """
    if variable is None:
        variable = "total"
    if z_level is None:
        z_level = "total"

    # Use cached inference keyed on (model, pred_time, device).
    pred_norm = run_inference_cached(
        model=model,
        era5_cache=era5_cache,
        pred_time=pred_time,
        model_cache_key=model_cache_key,
        device=device,
    )[0]  # (N_grid, d_state)

    idx        = resolve_pred_idx(era5_cache, pred_time=pred_time, nearest=False)
    truth_norm = get_era5_normalized_state(era5_cache, idx, device=device) # (N_grid, d_state)

    # Absolute error (dimensionless, in normalized space)
    error = torch.abs(pred_norm - truth_norm)  # (N_grid, d_state)

    # All five cases above are one operation: select the columns the (variable, z_level) pair
    # names, then take the per-node RMSE across them. For a single column that reduces to
    # |pred - truth| exactly, which is what cases 1 and 5 asked for, so the aggregation is
    # written once and the domain difference lives entirely in resolve_state_columns.
    cols = resolve_state_columns(era5_cache, variable, z_level)
    sq = error[:, cols] ** 2                             # (N_grid, len(cols))

    if era5_cache.get("_ocean"):
        # Divide by each node's OWN count of valid channels, not by len(cols).
        #
        # Ocean channels die off with depth: a node in 30 m of water has real data in ~7 of
        # 27 channels, the rest being below the seafloor and zeroed to the training mean.
        # Averaging over all 27 counts those dead cells as near-perfect predictions, so a
        # shallow node's error is divided by 27 while a deep node's is divided by 27 real
        # values. The map then shades coastlines and shelf seas cool -- reading as "well
        # predicted" when it actually means "little here to predict".
        #
        # This is the same valid_mask the timeline RMSE and the colorbar ranges already use;
        # applying it here makes the three consistent. ERA5 is untouched: it has no dead
        # cells, so masking there would be a no-op anyway.
        valid = era5_cache["valid_mask"][:, cols].to(sq.device)     # (N_grid, len(cols))
        count = valid.sum(dim=-1)
        result = (sq * valid).sum(dim=-1) / count.clamp(min=1)
        # A node with no valid channel in the selection (e.g. thetao_541m where the water is
        # shallower than 541 m) has nothing to score. Both sides were zeroed to the training
        # mean there, so the computed value is ~0 -- indistinguishable from a perfect
        # prediction, and the map shaded whole shelf seas as if the model nailed them.
        # NaN marks it as absent instead; orjson serialises NaN to JSON null, so this needs no
        # change to the wire format and the frontend can tell "no data" from "no error".
        # Only reachable when a single deep channel is selected: every node is valid at the
        # surface, so "total" and zos always have at least one live channel.
        result = result.masked_fill(count == 0, float("nan"))
    else:
        result = sq.mean(dim=-1)

    return result.sqrt().detach().cpu().contiguous().numpy()


# ---------------------------------------------------------------------------
# Prediction field (raw pred + truth slices for the frontend)
# ---------------------------------------------------------------------------

def _resolve_component_idx(era5_cache: dict, variable: str, z_level: str | int | None) -> int:
    """Map (variable, level) → column index into d_state.

    Matches the encoding used by `_stack_state`: atmospheric vars come first, indexed
    `var_idx * n_levels + level_idx`, then surface vars in their list order.

    z_level is ignored for surface variables. For atmospheric vars, must be an int
    or numeric string; "total" is not valid here (use `compute_prediction_errors`
    for aggregated views).
    """
    if era5_cache.get("_ocean"):
        # Ocean features carry their depth in the name, so there is nothing to combine.
        cols = resolve_state_columns(era5_cache, variable, z_level)
        if len(cols) != 1:
            raise ValueError(
                f"'{variable}' selects {len(cols)} ocean features; a prediction field needs "
                "exactly one. Name the feature including its depth, e.g. 'thetao_47m'."
            )
        return cols[0]

    atm_vars  = era5_cache["atm_vars"]
    surf_vars = era5_cache["surf_vars"]
    levels    = era5_cache["levels"]
    n_levels  = len(levels)

    if variable in surf_vars:
        return len(atm_vars) * n_levels + surf_vars.index(variable)
    if variable in atm_vars:
        if z_level in (None, "total"):
            raise ValueError(
                f"Atmospheric variable '{variable}' requires an explicit pressure level "
                f"(one of {levels}); got z_level={z_level!r}."
            )
        try:
            level = int(z_level)
        except (TypeError, ValueError):
            raise ValueError(f"z_level must be an integer hPa value, got {z_level!r}.")
        if level not in levels:
            raise ValueError(f"Level {level} not in available levels {levels}.")
        return atm_vars.index(variable) * n_levels + levels.index(level)
    raise ValueError(
        f"Unknown variable '{variable}'. Atm: {atm_vars}. Surf: {surf_vars}."
    )


def get_prediction_field(
    model: torch.nn.Module,
    era5_cache: dict,
    pred_time: str,
    variable: str,
    z_level: str | int | None = None,
    model_cache_key: str = "default",
    device: str | None = None,
) -> dict:
    """Return raw pred + truth slices for a single (variable, level) in **normalized** space.

    The frontend denormalizes to physical units using the (mean, std) it fetched
    once from `/inference/normalization`. Both arrays are length N_grid.

    Reuses `run_inference_cached`, so calling this for different (variable, level)
    at the same pred_time does not re-run the model.

    Returns:
        {
            "pred":  list[float],  # length N_grid, normalized
            "truth": list[float],  # length N_grid, normalized
            "component_idx": int,  # column index into d_state, useful for debugging
        }
    """
    pred_norm = run_inference_cached(
        model=model,
        era5_cache=era5_cache,
        pred_time=pred_time,
        model_cache_key=model_cache_key,
        device=device,
    )[0]  # (N_grid, d_state)

    idx = resolve_pred_idx(era5_cache, pred_time=pred_time, nearest=False)
    truth_norm = get_era5_normalized_state(era5_cache, idx, device=device)  # (N_grid, d_state)

    c_idx = _resolve_component_idx(era5_cache, variable, z_level)
    pred_col  = pred_norm[:, c_idx]
    truth_col = truth_norm[:, c_idx]

    if era5_cache.get("_ocean"):
        # Below-seafloor cells were zeroed *after* standardising, so they sit at exactly 0 in
        # normalized space and denormalize to the training mean -- displayed as a perfectly
        # plausible temperature or salinity for water that isn't there. Mark them absent so
        # the frontend paints no-data instead; orjson writes NaN as JSON null.
        valid = era5_cache["valid_mask"][:, c_idx].to(pred_col.device)
        pred_col  = pred_col.masked_fill(~valid, float("nan"))
        truth_col = truth_col.masked_fill(~valid, float("nan"))

    return {
        "pred":          pred_col.detach().cpu().contiguous().numpy(),
        "truth":         truth_col.detach().cpu().contiguous().numpy(),
        "component_idx": c_idx,
    }


def compute_physical_scales(era5_cache: dict) -> dict[str, list[float]]:
    """Per-feature physical [min, max] across the loaded ERA5 window.

    Keys are `"{variable}@{level}"` for atmospheric variables and `"{variable}"` for
    surface ones. Values are `[min, max]` in physical units, suitable as a fixed
    colorbar range when the frontend displays predictions (or truth) for a given
    feature across the timeline.

    Cached on `era5_cache["physical_scales"]` after first computation. The cache is
    discarded automatically when the ERA5 window is reloaded (the whole `era5_cache`
    dict is rebuilt).

    First call in lazy mode triggers a window-wide zarr scan (one xarray reduction
    per (variable, level) — chunked, but not free). Subsequent calls hit the cache.
    """
    cached = era5_cache.get("physical_scales")
    if cached is not None:
        return cached

    if era5_cache.get("_ocean"):
        # Ocean state is a flat list of features (zos, thetao_47m, ...) with no
        # variable/level split, so the "var@level" key scheme does not apply -- the feature
        # name already carries its depth. The density channel is dropped: it is an input
        # construct with no physical units, and the model does not emit it as state.
        from services.ocean import ocean_state_at  # noqa: PLC0415

        names = era5_cache["state_features"]
        data_mean = era5_cache["data_mean"].detach().cpu().numpy().reshape(-1)
        data_std = era5_cache["data_std"].detach().cpu().numpy().reshape(-1)
        # Land and below-seafloor cells were zeroed after standardising, so they sit at the
        # training mean rather than at a real value. Including them would drag every range
        # toward the mean; the valid mask is the same one the error metrics use.
        # Runs on the cache's own device. Forcing CPU here does not work: ocean_state_at
        # normalises the step against _land, data_mean, data_std and _ref_idx, which are cache
        # constants living wherever the cache was built -- so a CPU step meets CUDA constants
        # and raises "Expected all tensors to be on the same device". Device is a property of
        # the cache, not a choice this function gets to make.
        valid = era5_cache["valid_mask"]
        dev = valid.device
        n_feat = len(names)
        # Streamed one timestep at a time rather than over a (T, N, 27) array: the lazy cache
        # has no such array, and materialising one here would reintroduce the ~900 MB the
        # lazy path exists to avoid. Running min/max is exact, not an approximation. One step
        # at a time is also why this is affordable on the GPU -- 76 MB resident, not the whole
        # window.
        mins = torch.full((n_feat,), float("inf"), device=dev)
        maxs = torch.full((n_feat,), float("-inf"), device=dev)
        for t in range(len(era5_cache["time"])):
            step = ocean_state_at(era5_cache, t)[..., :n_feat]   # (N_grid, 27) on `dev`
            for i in range(n_feat):
                col = step[:, i][valid[:, i]]
                if col.numel() == 0:
                    continue
                mins[i] = torch.minimum(mins[i], col.min())
                maxs[i] = torch.maximum(maxs[i], col.max())

        scales: dict[str, list[float]] = {}
        for i, name in enumerate(names):
            if not torch.isfinite(mins[i]):
                continue
            # float() must wrap the whole expression, not just the tensor element: a Python
            # float times a numpy float32 is a numpy float32, which jsonable_encoder cannot
            # serialize -- and the resulting 500 escapes before CORS headers are attached,
            # so the browser reports it as an origin error rather than a server fault.
            scales[name] = [
                float(mins[i] * data_std[i] + data_mean[i]),
                float(maxs[i] * data_std[i] + data_mean[i]),
            ]
        era5_cache["physical_scales"] = scales
        return scales

    atm_vars  = era5_cache["atm_vars"]
    surf_vars = era5_cache["surf_vars"]
    levels    = era5_cache["levels"]
    n_levels  = len(levels)

    state_norm = era5_cache.get("state_normalized")
    if state_norm is not None:
        # Eager path: actual window min/max in normalized space, convert back to physical.
        data_mean = era5_cache["data_mean"].detach().cpu().numpy()
        data_std  = era5_cache["data_std"].detach().cpu().numpy()
        sn = state_norm.detach().cpu().numpy()  # (T, N_grid, d_state)
        mins = sn.min(axis=(0, 1)) * data_std + data_mean
        maxs = sn.max(axis=(0, 1)) * data_std + data_mean
        scales: dict[str, list[float]] = {}
        for vi, var in enumerate(atm_vars):
            for li, level in enumerate(levels):
                idx = vi * n_levels + li
                scales[f"{var}@{level}"] = [float(mins[idx]), float(maxs[idx])]
        for si, var in enumerate(surf_vars):
            idx = len(atm_vars) * n_levels + si
            scales[var] = [float(mins[idx]), float(maxs[idx])]
        era5_cache["physical_scales"] = scales
        return scales

    # Lazy path: walk the open xarray Dataset and compute true min/max per (var, level).
    # Atmospheric vars have dims (time, longitude, latitude, level); surface vars omit level.
    ds = era5_cache["ds"]
    scales = {}
    for var in atm_vars:
        if var not in ds.data_vars:
            continue
        da = ds[var]
        for level in levels:
            slc = da.sel(level=level)
            scales[f"{var}@{level}"] = [float(slc.min().values), float(slc.max().values)]
    for var in surf_vars:
        if var not in ds.data_vars:
            continue
        da = ds[var]
        scales[var] = [float(da.min().values), float(da.max().values)]

    era5_cache["physical_scales"] = scales
    return scales


def compute_error_range(
    model: torch.nn.Module,
    era5_cache: dict,
    pred_time: str,
    model_cache_key: str = "default",
    device: str | None = None,
) -> dict[str, float]:
    """
    Compute four colormap scale maxes for the error heatmap at a given timestep.

    One inference call (cached), four tensor reductions.  The four values are used
    by the frontend to fix the colormap scale per aggregation mode so that switching
    variables/levels within a mode does not rescale the colorbar.

    Returns:
        {
            "single_component": float,   # max |pred−truth| over all 83 components × nodes
            "variable_total":   float,   # max per-variable RMSE-over-levels, over (atm_vars × nodes)
            "total_by_level":   float,   # max per-level RMSE-over-atm-vars, over (levels × nodes)
            "global_total":     float,   # max per-node RMSE over all 83 components
        }
    """
    pred_norm = run_inference_cached(
        model=model,
        era5_cache=era5_cache,
        pred_time=pred_time,
        model_cache_key=model_cache_key,
        device=device,
    )[0]  # (N_grid, d_state)

    idx        = resolve_pred_idx(era5_cache, pred_time=pred_time, nearest=False)
    truth_norm = get_era5_normalized_state(era5_cache, idx, device=device)  # (N_grid, d_state)

    error = torch.abs(pred_norm - truth_norm)  # (N_grid, d_state)

    # single_component: max absolute error across all nodes and all state dimensions
    single_component = float(error.max())

    if era5_cache.get("_ocean"):
        # Ocean state has no variable x level grid: each feature already names its depth, so
        # there is nothing to aggregate *within* a variable or *across* a level. Both grouped
        # bounds therefore collapse onto the ungrouped ones rather than being invented. The
        # response keeps all four keys so the frontend's colorbar logic is unchanged.
        return {
            "single_component": single_component,
            "variable_total":   single_component,
            "total_by_level":   float((error ** 2).mean(dim=-1).sqrt().max()),
            "global_total":     float((error ** 2).mean(dim=-1).sqrt().max()),
        }

    atm_vars = era5_cache["atm_vars"]
    n_levels = len(era5_cache["levels"])
    n_atm    = len(atm_vars)

    # variable_total: for each atm var, RMSE over its 13 levels per node → max over (vars × nodes)
    var_rmses = []
    for v in range(n_atm):
        var_slice = error[:, v * n_levels : (v + 1) * n_levels]  # (N_grid, n_levels)
        var_rmses.append((var_slice ** 2).mean(dim=-1).sqrt())    # (N_grid,)
    variable_total = float(torch.stack(var_rmses).max())

    # total_by_level: for each level, RMSE over all atm vars per node → max over (levels × nodes)
    level_rmses = []
    for l in range(n_levels):
        level_indices = [v * n_levels + l for v in range(n_atm)]
        level_slice = error[:, level_indices]                       # (N_grid, n_atm)
        level_rmses.append((level_slice ** 2).mean(dim=-1).sqrt()) # (N_grid,)
    total_by_level = float(torch.stack(level_rmses).max())

    # global_total: per-node RMSE over all 83 components → scalar max
    global_total = float((error ** 2).mean(dim=-1).sqrt().max())

    return {
        "single_component": single_component,
        "variable_total":   variable_total,
        "total_by_level":   total_by_level,
        "global_total":     global_total,
    }


# ---------------------------------------------------------------------------
# Inference caching — keyed on model + pred_time + device
# ---------------------------------------------------------------------------

# Module-level cache for raw inference output. Keyed on
# (model_cache_key, pred_time_str, device),
# value is the normalized prediction tensor (1, N_grid, d_state).
# Bounded by BYTES, not entry count. An ERA5 prediction is (29040, 83) = 9.6 MB, but an
# ocean one is (675219, 28) = 75.6 MB — 8x larger. A 50-entry cap that means 480 MB for
# weather means 3.8 GB for ocean, which on its own exceeds a typical container limit, so
# scrubbing a handful of ocean timestamps would OOM the process. The budget below holds
# either domain to the same footprint; it just stores fewer ocean entries.
# All variable slices and error aggregations (compute_prediction_errors cases 1-4)
# derive from this one cached tensor, so switching variables for the same timestep
# costs zero extra inference.
_INFERENCE_CACHE_MAX_BYTES = int(os.environ.get("INFERENCE_CACHE_MB", "480")) * 1024 * 1024
_inference_cache: OrderedDict[tuple[str, str, str], torch.Tensor] = OrderedDict()

# Per-key locks so two concurrent requests for the *same* pred_time dedup onto a
# single forward pass, while requests for *different* keys still run fully
# concurrent (a single global lock would serialize unrelated work while still
# looking correct). Guarded by _inference_locks_guard, which is only ever held
# for dict bookkeeping — never across the actual compute.
_inference_locks: dict[tuple[str, str, str], threading.Lock] = {}
_inference_locks_guard = threading.Lock()


def _clear_inference_cache() -> None:
    """Manually clear the inference cache (useful for testing or forced refresh)."""
    global _inference_cache
    _inference_cache.clear()


def run_inference_cached(
    model: torch.nn.Module,
    era5_cache: dict[str, Any],
    pred_time: str | np.datetime64,
    model_cache_key: str = "default",
    device: str | None = None,
) -> torch.Tensor:
    """
    Run inference with caching on pred_time.

    Returns the normalized prediction tensor (1, N_grid, d_state) — shape is consistent
    with run_inference(..., batch_size=1, return_denormalized=False)["prediction_normalized"].

    Cache key: (model_cache_key, str(pred_time), device).
    Cache size: up to 50 entries (~480 MB total).
    When cache is full, oldest entry (FIFO) is evicted.

    Args:
        model:      Loaded GNN.
        era5_cache: Result of preload_era5_tensors.
        pred_time:  Prediction timestamp (ISO string or numpy datetime64).
        device:     Device to run inference on. Defaults to DEVICE.

    Returns:
        Normalized prediction tensor of shape (1, N_grid, d_state) in float32.
    """
    global _inference_cache

    if device is None:
        # The cache records the device its tensors were built on. Prefer that over the global
        # DEVICE, which is ERA5's: on a Mac it is "mps", and an ocean cache is never built
        # there (the graph overruns Apple's allocation cap), so defaulting to it would fail
        # with a device mismatch rather than an obvious error.
        device = era5_cache.get("_device") or DEVICE

    # Normalize pred_time to string for cache key
    pred_time_str = str(pred_time)
    # Include model identity so switching models never returns a stale cached prediction.
    cache_key = (pred_time_str, device, id(model))

    # Check cache. Entries live in RAM (see the store below), so every hit moves to the
    # compute device -- ~70 MB over PCIe, order 10 ms, against a multi-second forward pass.
    if cache_key in _inference_cache:
        return _inference_cache[cache_key].to(device)

    # Acquire (or create) this key's lock so a second concurrent caller for the
    # same pred_time blocks here instead of launching a duplicate forward pass.
    with _inference_locks_guard:
        if cache_key in _inference_cache:
            return _inference_cache[cache_key].to(device)
        lock = _inference_locks.setdefault(cache_key, threading.Lock())

    with lock:
        # Re-check: the lock holder that actually computed it may have already
        # populated the cache while we were waiting.
        if cache_key in _inference_cache:
            return _inference_cache[cache_key].to(device)

        # Cache miss — run inference
        result_dict = run_inference(
            model=model,
            era5_cache=era5_cache,
            pred_time=pred_time_str,
            device=device,
            nearest=False,
            with_grad=False,
            return_denormalized=False,
            batch_size=1,
        )
        pred_norm = result_dict["prediction_normalized"]  # (1, N_grid, d_state)

        with _inference_locks_guard:
            # Cached on CPU: an attribution backward pass allocates ~15 GiB against a 16 GiB
            # VRAM cap, so anything resident on the accelerator between requests comes
            # straight off that margin. Returned tensor stays on its compute device; only the
            # stored copy moves.
            _inference_cache[cache_key] = pred_norm.detach().to("cpu")

            # Evict oldest entries until the cache fits its byte budget. Always keeps at
            # least the entry just inserted, so a single prediction larger than the budget
            # still works rather than being evicted before it can be returned.
            while len(_inference_cache) > 1 and sum(
                t.numel() * t.element_size() for t in _inference_cache.values()
            ) > _INFERENCE_CACHE_MAX_BYTES:
                _inference_cache.popitem(last=False)

            _inference_locks.pop(cache_key, None)

    return pred_norm


# ---------------------------------------------------------------------------
# Run Inference
# ---------------------------------------------------------------------------

def run_inference(
    model: torch.nn.Module,
    era5_cache: dict[str, Any],
    pred_time: str | np.datetime64,
    device: str | None = None,
    nearest: bool = False,
    with_grad: bool = False,
    return_denormalized: bool = False,
    batch_size: int = 1,
) -> dict[str, Any]:
    """
    Unified one-step inference for TorchScript and checkpoint-loaded models.

    Dispatch strategy:
      - TorchScript (.pt): call model(prev, prev_prev, forcing)
      - Checkpoint wrappers (e.g. LightningModule without forward): call
        model.predict_step(prev, prev_prev, forcing)

    Args:
        with_grad:   If True, run under torch.enable_grad() and mark inputs as
                     requiring grad (useful for saliency / IG).
        batch_size:  Repeat the single-timestep input batch_size times along
                     the batch dimension before the forward pass.  The returned
                     prediction_normalized has shape (batch_size, N_grid, d_state).
                     Useful for throughput profiling: the model sees a realistic
                     batch without requiring multiple distinct ERA5 timesteps.
    """
    idx = resolve_pred_idx(
        era5_cache,
        pred_time=pred_time,
        nearest=nearest,
    )

    if era5_cache.get("_ocean"):
        # Ocean caches carry their own step-inputs builder. It returns the same three-tuple,
        # with the day-of-year and atmospheric forcing windows concatenated into one block;
        # the model adapter splits them again. Nothing below this point differs.
        from services.ocean import get_ocean_step_inputs  # noqa: PLC0415

        prev, prev_prev, forcing = get_ocean_step_inputs(
            era5_cache, pred_time=era5_cache["time"][idx], device=device
        )
    else:
        prev, prev_prev, forcing = get_era5_step_inputs(
            era5_cache=era5_cache,
            pred_time=era5_cache["time"][idx],
            device=device,
        )

    if batch_size > 1:
        prev      = prev.expand(batch_size, -1, -1).contiguous()
        prev_prev = prev_prev.expand(batch_size, -1, -1).contiguous()
        forcing   = forcing.expand(batch_size, -1, -1).contiguous()

    if with_grad:
        prev      = prev.requires_grad_(True)
        prev_prev = prev_prev.requires_grad_(True)
        forcing   = forcing.requires_grad_(True)

    model.eval()
    grad_ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with grad_ctx:
        if isinstance(model, torch.jit.ScriptModule):
            out = model(prev, prev_prev, forcing)
        elif hasattr(model, "predict_step"):
            out = model.predict_step(prev, prev_prev, forcing)
        else:
            out = model(prev, prev_prev, forcing)

    pred_norm = out[0] if isinstance(out, tuple) else out

    result = {
        "prediction_normalized": pred_norm,
        "pred_idx": idx,
        "batch_size": batch_size,
    }
    if "time" in era5_cache:
        result["pred_time"] = era5_cache["time"][idx]
    if with_grad:
        result["inputs"] = {
            "prev": prev,
            "prev_prev": prev_prev,
            "forcing": forcing,
        }
    if return_denormalized:
        if era5_cache.get("_ocean"):
            # Ocean constants are 27 features, ERA5's are 83 -- using the wrong ones is a
            # shape error rather than a silent rescale, but only because the counts differ.
            from services.ocean import denormalize as _ocean_denormalize  # noqa: PLC0415

            result["prediction"] = _ocean_denormalize(pred_norm)
        else:
            result["prediction"] = denormalize(pred_norm)

    return result


def run_inference_batch(
    model: torch.nn.Module,
    inputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: str = DEVICE,
) -> torch.Tensor:
    """
    Run one batched model call over multiple distinct timestep inputs.

    Each element of `inputs` is a (prev, prev_prev, forcing) tuple as returned
    by get_era5_step_inputs for a single timestep — shape (1, N_grid, d_state).
    All tuples are stacked along the batch dimension and passed through the model
    in a single forward call.

    Args:
        model:  Loaded GNN (TorchScript or checkpoint).
        inputs: List of (prev, prev_prev, forcing) tuples, one per timestep.
        device: Device to move tensors to before stacking.

    Returns:
        Tensor of shape (B, N_grid, d_state) — one normalized prediction per input.
    """
    if not inputs:
        raise ValueError("inputs must not be empty")

    prevs     = torch.cat([p.to(device)   for p, _, _ in inputs], dim=0)  # (B, N_grid, d_state)
    prev_prevs = torch.cat([pp.to(device) for _, pp, _ in inputs], dim=0)
    forcings  = torch.cat([f.to(device)   for _, _, f in inputs], dim=0)  # (B, N_grid, 15)

    model.eval()
    with torch.no_grad():
        if isinstance(model, torch.jit.ScriptModule):
            out = model(prevs, prev_prevs, forcings)
        elif hasattr(model, "predict_step"):
            out = model.predict_step(prevs, prev_prevs, forcings)
        else:
            out = model(prevs, prev_prevs, forcings)

    pred = out[0] if isinstance(out, tuple) else out  # (B, N_grid, d_state)
    return pred
