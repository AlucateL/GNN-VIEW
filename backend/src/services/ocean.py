"""
Ocean-state preprocessing, mirroring the ERA5 path in `inference.py`.

`prepare_OceanBench()` is the counterpart of `preload_era5_tensors()` and returns a cache dict with
the same key names, so the surrounding machinery (timestamp resolution, caching, the Explain
endpoints) carries over. `get_ocean_step_inputs()` is the counterpart of
`get_era5_step_inputs()`.

WHERE THE OCEAN DIFFERS FROM ERA5
    1. Forcing is not derived.  ERA5's `_compute_forcing` builds 5 astronomical features from
       timestamps alone. The ocean model predicts the ocean, so the atmosphere is an external
       driver it cannot compute: 8 real fields from `ifs_forcing.zarr`, plus day-of-year
       sin/cos from the ocean datastore itself.

    2. The model takes five inputs, not three:
           predict_step(prev_state, prev_prev_state, forcing, boundary_forcing,
                        atmosphere_forcing)
       Atmosphere stays separate from forcing. `boundary_forcing` is for the regional Baltic
       configuration and is None globally.

    3. An ice density channel is appended to the state, so the model sees 28 channels rather
       than 27 (`weather_dataset.py:858-878`). It is 1 where `siconc` > 0 in physical space
       and 0 over land, with the threshold expressed in standardised space as -mean/std.
       Being derived from `siconc`, it cannot be built without that variable.

    4. Grid points must be selected by the mask. The datastore carries all 979,200 lat/lon
       cells; the model and the graphs use only the 675,219 that are sea.

Clamping (a sigmoid/softplus reparameterisation of the state update) and the zeroing of
`siconc`/`sithick` where density < 0.5 both live inside the model, so a TorchScript export
carries them and they are deliberately not reimplemented here.
"""

from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Same resolution as inference.py: DATA_ROOT is set to /usr/data in the container, where
# backend/data is mounted. Falls back to the project-relative path for local runs.
_PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[3]))
_DATA_ROOT: Path = (
    Path(os.environ["DATA_ROOT"]) if "DATA_ROOT" in os.environ else _PROJECT_ROOT / "data"
)
_OCEAN_ROOT = _DATA_ROOT / "ocean_datasets"

STATE_ZARR = _OCEAN_ROOT / "analysis_ocean.zarr"
ATMO_ZARR = _OCEAN_ROOT / "ifs_forcing.zarr"
STATIC_FILES_PATH = _OCEAN_ROOT / "static"

# Index of the reference variable for the ice density channel, resolved from
# state_features.json rather than hardcoded -- the paper lists the variables in a different
# order than the data, so any hardcoded index would be a coin flip.
_DENSITY_REFERENCE_VAR = "siconc"

# Widths the checkpoint expects, read off the instantiated model during export:
# interior_input_dim 93 = 2*28 state + 7 static + 6 forcing + 24 atmosphere.
# Both are their base feature count times a 3-step window (num_past_forcing_steps=2,
# num_future_forcing_steps=0, include_current).
FORCING_DIM = 6
ATMOSPHERE_DIM = 24


def _load_ocean_static(device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mean, std) for the 27 state features, shaped (1, 1, 27) for broadcasting."""
    mean = torch.load(STATIC_FILES_PATH / "parameter_mean.pt", map_location=device, weights_only=True)
    std = torch.load(STATIC_FILES_PATH / "parameter_std.pt", map_location=device, weights_only=True)
    return mean.view(1, 1, -1).float(), std.view(1, 1, -1).float()


def state_feature_names() -> list[str]:
    return json.loads((STATIC_FILES_PATH / "state_features.json").read_text())


def _load_forcing_stats(prefix: str, n_features: int, device: str = "cpu"):
    """
    Return (mean, std) for a forcing block, shaped (1, 1, n_features) for broadcasting.

    `prefix` is "forcing" (day-of-year, from the ocean statistics datastore) or "atmosphere"
    (the 8 IFS fields, from the ERA5 one). Both are standardised the same way the training
    pipeline does it -- `weather_dataset.py:715,724` subtract the mean and divide by the std
    of the *statistics* datastore, which is a different store from the one the data comes
    from. See the provenance table in ocean_datasets/README.md.
    """
    mean = torch.load(STATIC_FILES_PATH / f"{prefix}_mean.pt", map_location=device, weights_only=True)
    std = torch.load(STATIC_FILES_PATH / f"{prefix}_std.pt", map_location=device, weights_only=True)
    if mean.numel() != n_features or std.numel() != n_features:
        raise ValueError(
            f"{prefix}_mean/std.pt hold {mean.numel()}/{std.numel()} values, expected "
            f"{n_features}. These must line up with the forcing_feature axis of the store."
        )
    return mean.view(1, 1, -1).float(), std.view(1, 1, -1).float()


def _surface_mask(ds) -> np.ndarray:
    """
    Boolean selector for the sea points the model runs on.

    The mask is [mask_feature, grid_index] with one feature per state variable and depth;
    deeper levels have fewer valid cells as the seafloor cuts them off. Feature 0 is the
    surface, and it is the one that corresponds to the graph's grid nodes -- 675,219 of
    979,200 for the analysis grid.
    """
    return np.asarray(ds["mask"].isel(mask_feature=0).values).astype(bool)


def _ice_density(state_norm: torch.Tensor, ref_idx: int, mean: torch.Tensor,
                 std: torch.Tensor) -> torch.Tensor:
    """
    The 28th input channel: 1 where sea ice is present, 0 elsewhere.

    The test is `siconc > 0` in physical units. Since the state is already standardised, the
    threshold becomes -mean/std (`weather_dataset.py:248`). Land is excluded upstream by the
    mask selection, so no separate land zeroing is needed here.
    """
    thr = float(-mean[0, 0, ref_idx] / std[0, 0, ref_idx])
    return (state_norm[..., ref_idx] > thr).float().unsqueeze(-1)


def _present_timesteps(ds, store: Path, probe: int) -> np.ndarray:
    """
    Time indices that actually have data, without decompressing any of it.

    Zarr returns fill-value NaN for absent chunks rather than raising, so a partial download
    looks like a complete store and has to be detected. The obvious test -- read one grid
    point across time and check `isfinite` -- is ruinously expensive here: `state` is chunked
    (27, 1, 979200), so a single cell pulls its whole 105.7 MB chunk, and probing the shipped
    12-step window costs ~1.3 GB to answer a question about metadata.

    Chunk keys for a 3-D zarr v2 array are "{feature}.{time}.{grid}", so listing the
    directory gives the same answer for the price of a readdir. Falls back to the probe when
    the layout is anything else (v3, sharded, remote), so correctness never depends on the
    fast path being applicable.
    """
    chunk_dir = store / "state"
    try:
        t_from_names = sorted({
            int(p.name.split(".")[1])
            for p in chunk_dir.iterdir()
            if not p.name.startswith(".") and p.name.count(".") == 2
        })
        if t_from_names:
            return np.array(t_from_names, dtype=int)
    except (OSError, ValueError, IndexError):
        pass   # fall through to the read-based probe

    return np.flatnonzero(
        np.isfinite(ds["state"].isel(state_feature=0, grid_index=probe).values)
    )


def _present_inits(atmo, store: Path, probe: int) -> list[int]:
    """
    IFS init indices that have chunks on disk, from the chunk names.

    Same trick and same reason as _present_timesteps: `forcing` is chunked
    (8, 1, 1, 979200), so probing one grid cell pulls an 11.8 MB chunk per init. Keys are
    "{feature}.{lead}.{init}.{grid}", so the init is the third field. Falls back to the read
    probe for any other layout.
    """
    chunk_dir = store / "forcing"
    try:
        found = sorted({
            int(p.name.split(".")[2])
            for p in chunk_dir.iterdir()
            if not p.name.startswith(".") and p.name.count(".") == 3
        })
        if found:
            return found
    except (OSError, ValueError, IndexError):
        pass

    return [int(i) for i in np.flatnonzero(np.isfinite(
        atmo["forcing"].isel(forcing_feature=0, lead_time=1, grid_index=probe).values
    ))]


def _normalize_state(
    state: torch.Tensor, land: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
    ref_idx: int, device: str,
) -> torch.Tensor:
    """
    Standardise raw state and append the ice-density channel: (..., 27) -> (..., 28).

    Shared by the eager and lazy paths so both produce byte-identical tensors -- the whole
    point of the lazy path is that it changes when work happens, never what it computes.
    `land` is (N_grid, 27) and broadcasts over any leading time dimension.
    """
    # Broadcast on the feature axis alone. The stored stats are (1, 1, 27), which lines up
    # with an eager (T, N, 27) block but silently prepends a time axis to a single (N, 27)
    # step -- yielding (1, N, 28) and, after unsqueeze, a 4-D input the model rejects.
    # Flattening to (27,) is rank-agnostic and correct for both.
    m = mean.reshape(-1)
    s = std.reshape(-1)
    state_norm = (state - m) / s
    state_norm = torch.where(land, torch.zeros((), device=device), state_norm)
    density = _ice_density(state_norm, ref_idx, mean, std)
    return torch.cat([state_norm, density], dim=-1)


def _window_last_axis(t: torch.Tensor) -> torch.Tensor:
    """
    Flatten a 3-step window feature-major: channel index = feature*3 + step.

    `t` is (3, N_grid, d) oldest-first; the result is (N_grid, d*3).

    The interleaving is not a free choice. weather_dataset.py:577 flattens with
        da.stack({"forcing_feature_windowed": ("forcing_feature", "window")})
    and xarray's stack varies the LAST dim fastest, so a feature's three timesteps sit
    adjacent: [f0_t-2, f0_t-1, f0_t, f1_t-2, ...]. Concatenating whole timesteps instead
    (the ERA5 path's cat([t[:-2], t[1:-1], t[2:]])) yields step*d + feature -- the same
    numbers in a different order, correctly shaped, silently wrong. Measured: 0.241 vs
    0.175 normalised RMSE on the Milton window.
    """
    steps = torch.stack([t[0], t[1], t[2]], dim=-1)      # (N_grid, d, 3)
    return steps.reshape(steps.shape[0], -1)             # (N_grid, d*3)


def prepare_OceanBench(
    state_zarr: str | Path | None = None,
    atmo_zarr: str | Path | None = None,
    device: str = "cpu",
    init_index: int | None = None,
    lazy: bool = True,
) -> dict[str, Any]:
    """
    Load the ocean state and both forcing sources into a cache dict.

    Counterpart of `preload_era5_tensors`, including its `lazy` flag.

    LAZY BY DEFAULT, because eager does not fit. Holding the whole window costs ~1.5 GB --
    907 MB of state (T, 675219, 28), 648 MB of windowed atmosphere, 162 MB of forcing -- on
    top of the ERA5 cache and the weather models the same process already holds. A 3.8 GB
    container is OOM-killed building it, and even 5.8 GB was not enough. Only three timesteps
    are ever needed at once, so lazy keeps the open zarr plus the small per-node constants
    (mask, positions, statistics) and materialises steps on demand through a small LRU,
    trading a zarr read per timestep for roughly a fifth of the memory.

    Pass lazy=False for offline work where the whole window is scanned anyway and the memory
    is available -- it avoids re-reading the same chunks.

    `init_index` selects which IFS forecast to take the atmospheric forcing from. When None,
    the single init present in the store is used -- the shipped slice has only one.

    Returns a dict with the same key names the ERA5 cache uses, plus the ocean-specific
    `atmosphere` and `mask` entries.
    """
    import xarray as xr

    ds = xr.open_zarr(str(state_zarr or STATE_ZARR))
    atmo = xr.open_zarr(str(atmo_zarr or ATMO_ZARR))

    names = state_feature_names()
    store_names = [str(x) for x in ds.state_feature.values]
    if store_names != names:
        raise ValueError(
            "state_feature order in the store does not match state_features.json.\n"
            f"  store: {store_names[:5]} …\n  json : {names[:5]} …\n"
            "Channel order drives every downstream index; refusing to guess."
        )

    mask = _surface_mask(ds)
    n_grid = int(mask.sum())

    # Node positions, selected by the same mask as the state so index i of the model output
    # pairs with position i. Verified byte-identical (max diff 0.0 on both axes) to
    # graphs/ocean_hierarchical/grid_lat_lon.pt, which is built independently -- so the map
    # and the graph view place the same node in the same place.
    _lat = np.asarray(ds["latitude"].values)[mask]
    _lon = np.asarray(ds["longitude"].values)[mask]
    _lon = ((_lon + 180.0) % 360.0) - 180.0
    grid_lat_lon = torch.tensor(np.stack([_lat, _lon], axis=1), dtype=torch.float32)

    # --- state -------------------------------------------------------------------------
    # Stored as [state_feature, time, grid_index]; the model wants (T, N_grid, d_state).
    times_all = ds.time.values.astype("datetime64[ns]")
    # Probe a sea point, not grid_index 0 -- that cell is land and reads NaN at every
    # timestep, which would make a fully-present window look empty. Absent chunks read as
    # fill-value NaN rather than raising, so this is also how a partial download is caught.
    probe = int(np.flatnonzero(mask)[0])
    present = _present_timesteps(ds, Path(state_zarr or STATE_ZARR), probe)
    if present.size < 3:
        raise ValueError(
            f"Only {present.size} timesteps have data. Need at least 3 "
            "(2 input steps plus a target)."
        )
    t_idx = present

    mean, std = _load_ocean_static(device)

    # Per-feature land -> 0, matching weather_dataset.py:819-824 ("convert land from nan to
    # zero"). The surface mask keeps 675,219 points, but deeper levels are valid at only
    # 586,638 of them where the seafloor cuts them off, so those channels still carry NaN
    # after selection. neural-lam zeroes them *after* standardising, so 0 here means "the
    # training mean", not zero in physical units.
    land = torch.tensor(
        ~np.transpose(np.asarray(ds["mask"].values).astype(bool))[mask, :],
        dtype=torch.bool, device=device,
    )  # (N_grid, 27), True where that feature has no data

    ref_idx = names.index(_DENSITY_REFERENCE_VAR)

    state_norm = None
    if not lazy:
        state_np = ds["state"].isel(time=t_idx).values          # (27, T, 979200)
        state_np = np.transpose(state_np, (1, 2, 0))[:, mask, :]  # (T, N_grid, 27)
        state = torch.tensor(state_np, dtype=torch.float32, device=device)
        state_norm = _normalize_state(state, land, mean, std, ref_idx, device)

    # --- forcing: day-of-year sin/cos, from the ocean datastore ------------------------
    f_mean, f_std = _load_forcing_stats("forcing", 2, device)
    forcing = None
    if not lazy:
        forcing_np = ds["forcing"].isel(time=t_idx).values      # (2, T, 979200)
        forcing_np = np.transpose(forcing_np, (1, 2, 0))[:, mask, :]
        forcing = torch.tensor(forcing_np, dtype=torch.float32, device=device)
        forcing = (forcing - f_mean) / f_std

    # --- atmosphere: 8 IFS fields, aligned onto the state timestamps -------------------
    # One init covers only 12 days (lead -1 .. +10), so any window longer than that needs
    # several stitched together -- a month needs five. Each state timestep therefore carries
    # its own (init, lead) pair rather than sharing one init.
    #
    # Where inits overlap (they are weekly, so they overlap by five days) the MOST RECENT
    # init wins: that is the freshest forecast for the date and also the shortest lead, which
    # is how these are used operationally. Iterating inits in ascending order and letting
    # later ones overwrite gives exactly that.
    if init_index is not None:
        inits_present = [init_index]
    else:
        inits_present = _present_inits(atmo, Path(atmo_zarr or ATMO_ZARR), probe)
        if not inits_present:
            raise ValueError(
                f"No atmospheric forcing chunks found in {atmo_zarr or ATMO_ZARR}. "
                "Zarr returns NaN for missing chunks instead of raising, so an incomplete "
                "download looks like an empty store."
            )

    init_times = atmo.init_time.values.astype("datetime64[ns]")
    lead_deltas = atmo.lead_time.values.astype("timedelta64[ns]")
    one_day = np.timedelta64(1, "D").astype("timedelta64[ns]")

    # Two passes, because leads <= 0 are the forecast's own input steps -- the analysis at
    # initialisation, not a forecast. Driving the model with those gives it atmosphere it
    # would not have ahead of time, which flatters it, so genuine forecast leads (>= 1) win.
    # Leads 1..7 from weekly inits tile a timeline exactly: contiguous, no gaps.
    #
    # The fallback is not optional. The shipped window is 12 state steps against a single
    # init, and its first two dates -- the conditioning steps -- are reachable ONLY at leads
    # -1 and 0. A strict >= 1 rule leaves them uncovered, prepare_OceanBench raises, and the
    # whole ocean domain fails to build: the timeline still renders from its .npy while no
    # timestamp is selectable. Preferring forecast leads and falling back keeps thin windows
    # working, and the dates that fall back are inputs rather than scored predictions.
    best: dict[Any, tuple[int, int]] = {}
    for min_lead in (one_day, None):
        for i in sorted(inits_present):
            for li, delta in enumerate(lead_deltas):
                if min_lead is not None and delta < min_lead:
                    continue
                key = np.datetime64(init_times[i] + delta, "ns")
                # First pass wins: never downgrade a forecast lead to an analysis one.
                if min_lead is not None or key not in best:
                    best[key] = (i, li)

    missing = [str(t)[:10] for t in times_all[t_idx] if np.datetime64(t, "ns") not in best]
    if missing:
        covered = sorted(best)
        raise ValueError(
            f"No atmospheric forcing for {missing[:5]}{' ...' if len(missing) > 5 else ''}. "
            f"Inits {sorted(inits_present)} cover {str(covered[0])[:10]} to "
            f"{str(covered[-1])[:10]}; state and forcing windows must line up. Each init "
            "spans 12 days, so a longer window needs more of them downloaded."
        )
    init_lead = [best[np.datetime64(t, "ns")] for t in times_all[t_idx]]

    # Which prediction steps are driven by forecast forcing alone.
    #
    # A prediction at step i reads the forcing window [i-2, i-1, i], so a fallback at any of
    # those three feeds analysis-lead atmosphere into a scored prediction. On the shipped
    # 12-step window that is the first two predictions; with enough inits it is none. Those
    # steps are still perfectly renderable, but they are not measuring the same thing as the
    # rest -- analysis forcing is information the model would not have ahead of time -- so
    # they are excluded from error computation rather than silently averaged in.
    _lead_is_forecast = [lead_deltas[li] >= one_day for _, li in init_lead]
    forecast_only_times = [
        times_all[t_idx][i]
        for i in range(2, len(init_lead))
        if all(_lead_is_forecast[i - 2 : i + 1])
    ]
    # The IFS fields arrive in physical units -- K, m/s, W/m^2, Pa -- so this store reads as
    # "physically plausible" while being four orders of magnitude outside anything the model
    # was trained on. Standardising is not optional; without it the model scores twice as
    # badly as persistence. Stats come from the ERA5 statistics datastore, not from this
    # store, which carries no stats of its own.
    a_mean, a_std = _load_forcing_stats("atmosphere", 8, device)
    atmosphere = None
    if not lazy:
        # One read per timestep: consecutive steps can come from different inits, so this
        # cannot be a single vectorised isel over a shared init.
        atmo_np = np.stack([
            atmo["forcing"].isel(init_time=i, lead_time=li).values[:, mask]   # (8, N_grid)
            for i, li in init_lead
        ])                                                                     # (T, 8, N_grid)
        atmo_np = np.transpose(atmo_np, (0, 2, 1))                             # (T, N_grid, 8)
        atmosphere = torch.tensor(atmo_np, dtype=torch.float32, device=device)
        atmosphere = (atmosphere - a_mean) / a_std

    # --- window forcing and atmosphere over 3 steps -----------------------------------
    # The checkpoint records num_past_forcing_steps=2, num_future_forcing_steps=0, and
    # weather_dataset.py:452-455 slices lead_time[current-2 : current+1] with
    # include_current=True -- i.e. [t-2, t-1, t], oldest first. Widths must come out as
    # 2*3 = 6 and 8*3 = 24; the model's interior_input_dim of 93 (= 2*28 state + 7 static
    # + 6 forcing + 24 atmosphere) does not admit anything else.
    forcing_window = atmosphere_window = None
    if not lazy:
        forcing_window = torch.stack(
            [_window_last_axis(forcing[i:i + 3]) for i in range(forcing.shape[0] - 2)]
        )
        atmosphere_window = torch.stack(
            [_window_last_axis(atmosphere[i:i + 3]) for i in range(atmosphere.shape[0] - 2)]
        )

    return {
        # Marks this as an ocean cache. inference.run_inference keys off it to pick the
        # right step-inputs builder; everything after that is identical to the ERA5 path.
        "_ocean": True,
        "_lazy": lazy,
        # None in lazy mode -- read through ocean_state_at() / ocean_window_at(), never
        # directly, so a caller that assumes the eager shape fails loudly instead of
        # silently reading a partial window.
        "state_normalized": state_norm,        # (T, N_grid, 28)
        "forcing_window": forcing_window,      # (T-2, N_grid, 6)
        "atmosphere_window": atmosphere_window,  # (T-2, N_grid, 24)
        # Everything below is what the lazy loaders need to materialise a step. Small:
        # the open zarr handles are metadata, and the constants are per-node, not per-time.
        "_ds": ds,
        "_atmo": atmo,
        "_t_idx": t_idx,
        # (init, lead) per timestep — a window longer than 12 days spans several inits.
        "_init_lead": init_lead,
        # Prediction timestamps whose whole forcing window is genuine forecast (lead >= 1).
        # The timeline scores only these; see the note above init_lead.
        "forecast_only_times": np.array(forecast_only_times, dtype="datetime64[ns]"),
        "_land": land,
        "_ref_idx": ref_idx,
        "_forcing_stats": (f_mean, f_std),
        "_atmo_stats": (a_mean, a_std),
        "_device": device,
        "time": times_all[t_idx],
        "mask": mask,
        "grid_lat_lon": grid_lat_lon,   # (N_grid, 2) float32, lon in [-180, 180]
        # (N_grid, 27) True where that channel carries real data. The surface mask keeps all
        # 675,219 points, but the depth channels are valid at only 586,638 of them. Error
        # metrics must average over these cells only -- the rest were zeroed above, so
        # counting them scores a filled-in training mean as a perfect prediction.
        "valid_mask": ~land,
        "n_grid": n_grid,
        "state_features": names,
        "data_mean": mean,
        "data_std": std,
        # The density channel is always appended last, so this is len(names) in both modes.
        "density_index": len(names),
    }


# Materialised timesteps, keyed by (id(cache), position in t_idx). Small and bounded: one
# step is (675219, 28) float32 = 76 MB, and a forward pass needs three (idx, idx-1, idx-2).
# Sized to hold one step's worth of context on either side so scrubbing to an adjacent
# timestamp reuses two of the three, at ~380 MB rather than the 1.5 GB the eager cache cost.
# Device for all ocean work: the cache's per-node constants, the model, and the tensors built
# per request. It lives here rather than in routes/inference.py because era5_window_manager
# builds the cache and must agree with whatever the request path later uses -- when the two
# disagreed (cache pinned to "cpu" while requests ran on cuda) every ocean inference raised a
# device mismatch immediately, while ERA5 was unaffected.
#
# The ocean graph is 23x the weather one and a single backward pass over its 675,219 nodes
# overruns Apple MPS's allocation cap, so MPS is excluded; that cap is specific to MPS, not a
# general limit. Override with OCEAN_DEVICE if a GPU is present but short on memory.
OCEAN_DEVICE: str = os.environ.get(
    "OCEAN_DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu",
)

# Ceiling this process may use on a shared GPU, matching the `vram: "16384"` reservation in
# helm/values.yaml. The reservation is a promise to other tenants; set_per_process_memory_
# fraction (applied at startup) is what makes it binding.
VRAM_LIMIT_MIB: int = int(os.environ.get("VRAM_LIMIT_MIB", "16384"))


def release_vram_if_high(threshold: float = 0.6) -> None:
    """Hand PyTorch's cached-but-unused blocks back to the driver when the pool is filling.

    "Reserved" memory is what PyTorch holds from the driver, including blocks it has freed
    internally and kept for reuse; "allocated" is live tensors. Attribution allocates and
    frees a full activation set per step, so reserved can sit far above allocated as stranded
    fragments -- an IG pass failed a 2.16 GiB request while holding 3.45 GiB of them.

    Only releases above `threshold` of the cap: empty_cache() forces later allocations back
    through cudaMalloc, so doing it unconditionally would slow every request to protect
    against a problem that only appears near the ceiling.
    """
    if not torch.cuda.is_available():
        return
    cap_bytes = VRAM_LIMIT_MIB * 1024 * 1024
    reserved = torch.cuda.memory_reserved()
    if reserved > threshold * cap_bytes:
        allocated = torch.cuda.memory_allocated()
        torch.cuda.empty_cache()
        logging.getLogger(__name__).info(
            "[vram] released: reserved %.2f GiB (allocated %.2f) -> %.2f GiB, cap %.2f GiB",
            reserved / 1024**3, allocated / 1024**3,
            torch.cuda.memory_reserved() / 1024**3, cap_bytes / 1024**3,
        )


_OCEAN_STATE_LRU = int(os.environ.get("OCEAN_STATE_LRU", "5"))
# Entries are held in RAM, never on the accelerator, and moved to the compute device on read.
# Each is 72 MB; five of them plus the inference and IG caches came to ~2 GiB of VRAM, which
# is the exact margin an attribution backward pass needs -- it allocates ~15 GiB against a
# 16 GiB cap, so the caches were the difference between fitting and OOM. A host-to-device copy
# costs ~10 ms against a 2.7 s computation, so this keeps the speedup and returns the VRAM.
_ocean_state_cache: "OrderedDict[tuple[int, int], torch.Tensor]" = OrderedDict()


def clear_ocean_state_cache() -> None:
    """Drop all materialised steps. Call when a cache is rebuilt, so stale steps cannot
    outlive the window they came from."""
    _ocean_state_cache.clear()


def ocean_state_at(cache: dict[str, Any], idx: int, device: str | None = None) -> torch.Tensor:
    """
    Normalised state at time position `idx`: (N_grid, 28).

    The single entry point for ocean state, eager or lazy -- callers must not index
    `state_normalized` themselves, since it is None in lazy mode.
    """
    dev = device or cache.get("_device", "cpu")
    if not cache.get("_lazy"):
        return cache["state_normalized"][idx].to(dev)

    key = (id(cache), int(idx))
    hit = _ocean_state_cache.get(key)
    if hit is not None:
        _ocean_state_cache.move_to_end(key)
        return hit.to(dev)    # cached on CPU; this is the only copy

    t = int(cache["_t_idx"][idx])
    raw = cache["_ds"]["state"].isel(time=t).values          # (27, 979200)
    sel = np.transpose(raw, (1, 0))[cache["mask"], :]        # (N_grid, 27)
    state = torch.tensor(sel, dtype=torch.float32, device=dev)
    out = _normalize_state(
        state, cache["_land"], cache["data_mean"], cache["data_std"], cache["_ref_idx"], dev
    )

    # Store on CPU regardless of where it was computed, so the cache never occupies VRAM.
    _ocean_state_cache[key] = out.detach().to("cpu")
    while len(_ocean_state_cache) > _OCEAN_STATE_LRU:
        _ocean_state_cache.popitem(last=False)
    return out


def ocean_window_at(
    cache: dict[str, Any], wi: int, device: str | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forcing (N_grid, 6) and atmosphere (N_grid, 24) for window `wi`, covering the state
    steps [wi, wi+1, wi+2] oldest-first -- i.e. the window ending at prediction step wi+2.

    Not cached: the two together are ~80 MB per step and are consumed immediately, so
    holding them would cost more than re-reading them.
    """
    dev = device or cache.get("_device", "cpu")
    if not cache.get("_lazy"):
        return cache["forcing_window"][wi].to(dev), cache["atmosphere_window"][wi].to(dev)

    mask = cache["mask"]
    times = [int(cache["_t_idx"][wi + k]) for k in range(3)]

    f_raw = cache["_ds"]["forcing"].isel(time=times).values          # (2, 3, 979200)
    f_sel = np.transpose(f_raw, (1, 2, 0))[:, mask, :]               # (3, N_grid, 2)
    f_mean, f_std = cache["_forcing_stats"]
    forcing = (torch.tensor(f_sel, dtype=torch.float32, device=dev) - f_mean) / f_std

    # Each of the three steps carries its own (init, lead): a window straddling a weekly
    # init boundary draws from two different forecasts.
    a_sel = np.stack([
        cache["_atmo"]["forcing"].isel(
            init_time=cache["_init_lead"][wi + k][0],
            lead_time=cache["_init_lead"][wi + k][1],
        ).values[:, mask]                                             # (8, N_grid)
        for k in range(3)
    ]).transpose(0, 2, 1)                                             # (3, N_grid, 8)
    a_mean, a_std = cache["_atmo_stats"]
    atmosphere = (torch.tensor(a_sel, dtype=torch.float32, device=dev) - a_mean) / a_std

    return _window_last_axis(forcing), _window_last_axis(atmosphere)


def get_ocean_step_inputs(
    cache: dict[str, Any],
    pred_time: str | np.datetime64,
    device: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build one model input tuple for the state at `pred_time`.

    Returns the same three-element tuple as `get_era5_step_inputs`, with the day-of-year and
    atmospheric windows concatenated into one (N, 30) forcing block. `load_ocean_model`
    splits them again before calling the model, so every downstream caller -- run_inference,
    the IxG/IG path -- stays identical to the weather case.

        prev_prev = state[idx - 2]
        prev      = state[idx - 1]
        forcing / atmosphere are taken at the target step.
    """
    times = cache["time"]
    target = np.datetime64(pred_time, "ns")
    hits = np.flatnonzero(times == target)
    if hits.size == 0:
        raise ValueError(
            f"{str(target)[:10]} not in the loaded window "
            f"({str(times[0])[:10]} to {str(times[-1])[:10]})."
        )
    idx = int(hits[0])
    if idx < 2:
        raise ValueError(
            f"{str(target)[:10]} is at index {idx}; the model needs two preceding steps."
        )

    prev_prev = ocean_state_at(cache, idx - 2, device).unsqueeze(0)
    prev = ocean_state_at(cache, idx - 1, device).unsqueeze(0)
    # Window i covers [i, i+1, i+2], so the entry ending at the target step is idx-2 --
    # the same indexing as get_era5_step_inputs.
    f_win, a_win = ocean_window_at(cache, idx - 2, device)
    forcing = torch.cat([f_win, a_win], dim=-1).unsqueeze(0)

    if device is not None:
        prev_prev, prev, forcing = prev_prev.to(device), prev.to(device), forcing.to(device)

    return prev, prev_prev, forcing


class _OceanModelAdapter(torch.nn.Module):
    """
    Presents the ocean model with the three-argument signature the rest of the backend uses.

    The scripted SeaCast module takes (prev, prev_prev, forcing, atmosphere) -- four inputs,
    where ERA5 models take three. Rather than teach every call site about that difference,
    the two forcing tensors travel concatenated as one (N, 30) block and get split here.

    That keeps `run_inference`, `_forward` in integrated_gradients, and the whole IxG/IG path
    working unmodified, which is what the task asked for. It also means attributions are
    computed over the combined forcing block, so the existing per-feature indexing still
    lines up: features 0-5 are the day-of-year window, 6-29 the atmospheric one.
    """

    def __init__(self, scripted: torch.nn.Module, forcing_dim: int = FORCING_DIM):
        super().__init__()
        self.model = scripted
        self.forcing_dim = forcing_dim

    def forward(self, prev: torch.Tensor, prev_prev: torch.Tensor,
                forcing: torch.Tensor) -> torch.Tensor:
        f = forcing[..., : self.forcing_dim]
        a = forcing[..., self.forcing_dim :]
        out = self.model(prev, prev_prev, f, a)
        return out[0] if isinstance(out, tuple) else out


def load_ocean_model(path: str | Path | None = None, device: str = "cpu") -> torch.nn.Module:
    """
    Load the scripted ocean model behind the three-argument adapter.

    Counterpart of `inference.load_torchscript`. Needs nothing but torch -- no neural_lam,
    no BSF, no zarr -- because TorchScript carries the graph and the weights.

    Defaults to CPU deliberately, unlike the weather loaders which follow DEVICE. A single
    backward pass over 675,219 nodes and a 3-level mesh needs ~9 GB, which overruns Apple
    MPS's allocation cap outright; the weather graph is 23x smaller and fits comfortably.
    Pass device explicitly if you have the headroom. Whatever you choose must match the
    device the cache and the attribution call use, or TorchScript fails deep inside with
    "Passed CPU tensor to MPS op".
    """
    p = Path(path) if path is not None else (
        _DATA_ROOT / "paper_checkpoints" / "seacast_scripted.pt"
    )
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Build it with scripts/export_seacast.py (needs .venv-export)."
        )
    scripted = torch.jit.load(str(p), map_location=device)
    scripted.eval()
    return _OceanModelAdapter(scripted).to(device).eval()


def denormalize(pred: torch.Tensor) -> torch.Tensor:
    """
    Convert normalised model output back to physical units.

    Counterpart of `inference.denormalize`. The model emits 27 state channels; if a density
    channel is still attached it is dropped first, since it is an input construct with no
    physical scaling.
    """
    mean, std = _load_ocean_static(str(pred.device))
    if pred.shape[-1] == mean.shape[-1] + 1:
        pred = pred[..., :-1]
    return pred * std + mean
