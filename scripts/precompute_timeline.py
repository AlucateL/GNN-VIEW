#!/usr/bin/env python3
"""
Precompute the per-timestep error timeline for all prediction timesteps in an
ERA5 or ocean dataset and save it beside the model checkpoint, where
GET /{model_id}/prediction_errors/timeline reads it.

FOUR SERIES ARE STORED (all in the model's own normalised space, so they are
comparable across variables without unit conversion):

  mean_error           sqrt(mean(err^2)) over every scoreable (node, component)
                       cell. This is what used to be stored as `rmse`.
  peak_error           max(err) over every scoreable (node, component) pair --
                       the largest single component error anywhere on the grid,
                       before any averaging.
  peak_top3            (T, 3) int16 -- the three components with the largest
                       per-component maxima, as indices into `column_names`.
                       Padded with -1 if d_state < 3.
  error_concentration  max over nodes of how many components put their own
                       global argmax on that node. A raw count.

peak_top3 indexes the COMPONENT axis; error_concentration reduces over the NODE
axis. They are different axes of the same error matrix -- easy to conflate.

error_concentration is deliberately a count, which makes it unit-free: it needs
no normalisation to be comparable across timesteps, and it is invariant to any
per-component rescaling, since argmax_n (s_c * err[n,c]) == argmax_n err[n,c]
for any s_c > 0. Sigma-space and physical units give the identical count.

`column_names` is stored alongside so peak_top3 is decodable without knowing the
domain's layout. It is `services.inference.state_column_names` output: ERA5 gives
"var@level" for atmospheric columns then bare names for surface ones; the ocean
gives fully qualified names ("thetao_47m") that carry their depth and have no
separate level axis.

Usage (from project root, in the cute_GNNs conda env):

    # Dev run -- uses the small 10-step zarr already in the repo
    python scripts/precompute_timeline.py

    # Full test-set streamed from WeatherBench2 GCS -- no local download needed.
    # Several models in one pass: each fetched chunk is scored by every model
    # before the next fetch, so N models cost one ~14 GB stream, not N.
    python scripts/precompute_timeline.py --gcs \\
        --model backend/data/paper_checkpoints/graphcast_scripted.pt \\
        --model backend/data/paper_checkpoints/graph_fm_scripted.pt

    # GPU run with larger inference batch
    python scripts/precompute_timeline.py --gcs --batch-size 8 --device cuda

    # Ocean (SeaCast) over the downloaded analysis window.
    #
    # DATA_ROOT is required for local runs: services/ocean.py defaults it to <repo>/data,
    # but the ocean zarrs and the SeaCast export live under backend/data, which is what is
    # mounted at /usr/data in the container.
    DATA_ROOT=backend/data python scripts/precompute_timeline.py --ocean

    # ERA5 runs also need neural_lam on the path. The container mounts it at
    # /models/neural-lam; locally it lives under backend/.
    NEURAL_LAM_PATH=backend/neural-lam python scripts/precompute_timeline.py ...

OUTPUT LOCATION
    Defaults to '<model_stem>_timeline_errors.npy' beside each checkpoint, which is
    the only place the backend looks. Do not write to data/precomputed/ -- that path
    is not read by anything, and hand-copying out of it is what let the shipped
    artifacts drift from the script's output.

OCEAN DIFFERS FROM ERA5 IN THREE WAYS (see services/ocean.py):

  source     There is no archive to slice or stream. The state and the IFS atmospheric
             forcing both come from the local zarrs, and the window is whatever was
             downloaded -- --time-start/--time-end only narrow it further.
  channels   The cached state carries 28 channels (27 state + a derived ice-density input),
             but the model emits 27. Density is an input construct with no target to score
             against, so it is dropped before differencing.
  masking    Land and below-seafloor cells are zeroed after standardising, so they are not
             predictions and must not be averaged over. Every reduction runs over valid
             cells only, which for the depth channels is 586,638 of 675,219 nodes. This
             makes the ocean numbers NOT directly comparable to the ERA5 ones, which are
             unmasked. error_concentration is additionally not comparable across domains
             because its ceiling is d_state: 27 for the ocean, 83 for ERA5.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup — make backend services importable without installing the package
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "backend" / "src"))

from services.inference import (  # noqa: E402
    DEVICE,
    FIELDS_ZARR_PATH,
    MODEL_PATH,
    TOA_NORM_PATH,
    _compute_forcing,   # private but same project
    _ensure_constants,
    _load_static,
    _load_toa_normalization,
    _stack_state,
    get_available_pred_timestamps,
    get_era5_normalized_state,
    get_era5_step_inputs,
    load_model,
    preload_era5_tensors,
    run_inference_batch,
)

# WeatherBench2 public ERA5 zarr — anonymous GCS access, no credentials needed.
_GCS_URL = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
)

# Test-set date range from ERA5Dataset in models/neural-lam:
#   split_slices["test"] = slice("2019-12-31T18", "2021-01-10T18")
# The first two steps are conditioning; valid predictions begin at 2020-01-01T06.
_DEFAULT_TIME_START = "2019-12-31T18"
_DEFAULT_TIME_END = "2021-01-10T18"
# No default output directory: every artifact is written as
# '<model_stem>_timeline_errors.npy' beside its own checkpoint, which is the only
# path routes/inference.py::get_errors_timeline reads. The former default wrote to
# data/precomputed/, which nothing loads -- the shipped artifacts had to be copied
# out of it by hand, and drifted as a result.
# Variables neural-lam actually uses (must match ATMOSPHERIC_PARAMS + SURFACE_PARAMS
# from neural_lam/constants.py in the same order).
_NEURAL_LAM_VARS = [
    "geopotential", "specific_humidity", "temperature",
    "u_component_of_wind", "v_component_of_wind", "vertical_velocity",
    "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
    "mean_sea_level_pressure", "total_precipitation_6hr",
]

# How many timesteps to load from GCS in one batch. 40 steps ≈ 10 days ≈ 380 MB
# uncompressed. Fits comfortably in RAM and minimises round-trips.
_GCS_CHUNK = 40

# How many top components to record per timestep. Three fits the right panel's
# marking and costs 6 bytes/timestep.
_TOP_K = 3


# ---------------------------------------------------------------------------
# Per-timestep statistics
# ---------------------------------------------------------------------------

def _step_stats(
    pred: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor | None = None
) -> tuple[float, float, np.ndarray, int]:
    """
    Reduce one (N_grid, d_state) prediction/truth pair to
    (mean_error, peak_error, peak_top3, error_concentration).

    valid: optional (N_grid, d_state) bool mask of scoreable cells. None means every
    cell counts (ERA5). The ocean passes cache["valid_mask"], which excludes land and
    below-seafloor cells -- those are zeroed rather than predicted, so averaging or
    maximising over them measures the masking, not the model.

    Masked cells are filled with -1 before the max/argmax reductions. Errors are
    absolute values and therefore >= 0, so -1 can never win either one.
    """
    err = (pred - truth).abs()                     # (N_grid, d_state)

    if valid is None:
        mean_error = float(err.pow(2).mean().sqrt())
        masked = err
    else:
        mean_error = float(err[valid].pow(2).mean().sqrt())
        masked = err.masked_fill(~valid, -1.0)

    comp_max = masked.amax(dim=0)                  # (d_state,) per-component max over nodes
    peak_error = float(comp_max.max())

    k = min(_TOP_K, int(comp_max.numel()))
    top = torch.topk(comp_max, k=k).indices
    if k < _TOP_K:                                 # pad a degenerate d_state < 3
        top = torch.cat([top, top.new_full((_TOP_K - k,), -1)])

    # argmax over NODES for each component, then how many components the most-won
    # node took. unique() rather than bincount(minlength=N_grid): the ocean grid has
    # 675,219 nodes and only 27 components, so a full-length histogram would allocate
    # a 5 MB int64 tensor per timestep to hold at most 27 non-zero entries.
    winners = masked.argmax(dim=0)                 # (d_state,) node index
    conc = int(torch.unique(winners, return_counts=True)[1].max())

    return mean_error, peak_error, top.detach().cpu().numpy().astype(np.int16), conc


def _new_series() -> dict[str, list]:
    return {"mean_error": [], "peak_error": [], "peak_top3": [], "error_concentration": []}


def _append_stats(series: dict[str, list], stats: tuple[float, float, np.ndarray, int]) -> None:
    series["mean_error"].append(stats[0])
    series["peak_error"].append(stats[1])
    series["peak_top3"].append(stats[2])
    series["error_concentration"].append(stats[3])


def _era5_column_names() -> list[str]:
    """
    Flat d_state column names in the order _stack_state produces them.

    Mirrors services.inference.state_column_names for the ERA5 layout, which the GCS
    path cannot call because it never builds an era5_cache. Atmospheric variables are
    var-major / level-minor, matching the (state_var, level) reshape in _stack_state.
    """
    c = _ensure_constants()
    return (
        [f"{v}@{lv}" for v in c.ATMOSPHERIC_PARAMS for lv in c.PRESSURE_LEVELS]
        + list(c.SURFACE_PARAMS)
    )


def _save_timeline(out_path: Path, timestamps: list[str], s: dict[str, list],
                   column_names: list[str]) -> None:
    """Write one model's series. Layout must stay readable by services.inference.load_timeline."""
    peak_top3 = np.stack(s["peak_top3"]).astype(np.int16)            # (T, 3)
    conc      = np.array(s["error_concentration"], dtype=np.int16)   # (T,)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, {
        "timestamps":          timestamps,
        "mean_error":          np.array(s["mean_error"], dtype=np.float32),
        "peak_error":          np.array(s["peak_error"], dtype=np.float32),
        "peak_top3":           peak_top3,
        "error_concentration": conc,
        "column_names":        list(column_names),
    })
    print(f"[precompute] saved {out_path}")
    print(f"[precompute]   mean_error          {min(s['mean_error']):.4f} - {max(s['mean_error']):.4f}")
    print(f"[precompute]   peak_error          {min(s['peak_error']):.4f} - {max(s['peak_error']):.4f}")
    print(f"[precompute]   error_concentration {conc.min()} - {conc.max()} of {len(column_names)} components")
    top_names = [column_names[i] for i in peak_top3[int(np.argmax(s['peak_error']))] if i >= 0]
    print(f"[precompute]   top-3 at peak       {top_names}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    src = p.add_mutually_exclusive_group()
    src.add_argument("--zarr", default=None,
                     help="Local path to ERA5 fields.zarr (default: dev dataset in repo)")
    src.add_argument("--gcs",  action="store_true",
                     help="Stream from WeatherBench2 GCS — no local download needed")
    src.add_argument("--ocean", action="store_true",
                     help="Ocean (SeaCast) over the local analysis window in data/ocean_datasets")

    p.add_argument("--state-zarr", default=None,
                   help="--ocean only: override the ocean state zarr (default: analysis_ocean.zarr)")
    p.add_argument("--atmo-zarr",  default=None,
                   help="--ocean only: override the IFS forcing zarr (default: ifs_forcing.zarr)")

    p.add_argument("--model",      action="append", default=None, dest="models",
                   help="Path to model checkpoint (.pt or .ckpt). Repeat to score several "
                        "models in one pass over the data. Defaults to the ERA5 model, or "
                        "the SeaCast export under --ocean.")
    p.add_argument("--time-start", default=_DEFAULT_TIME_START,
                   help="Slice from this ISO timestamp onwards")
    p.add_argument("--time-end",   default=_DEFAULT_TIME_END,
                   help="Slice up to this ISO timestamp (inclusive). Default: end of test set.")
    p.add_argument("--out",        default=None,
                   help="Output .npy path. Only valid with a single --model. The default -- "
                        "'<model_stem>_timeline_errors.npy' beside each checkpoint -- is the "
                        "only location the backend reads, so prefer it.")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Timesteps per model forward pass (increase on GPU)")
    p.add_argument("--device",     default=None,
                   help="Torch device: cpu / cuda / mps. Defaults to the ERA5 device, or cpu "
                        "under --ocean.")
    args = p.parse_args()
    if args.device is None:
        # The ocean graph is 23x larger than the weather one; a single forward pass over
        # 675,219 nodes overruns Apple MPS's allocation cap outright (services/ocean.py:328).
        args.device = "cpu" if args.ocean else DEVICE
    if not args.models:
        # Resolved in main(): the ocean default lives in services.ocean, not here.
        args.models = [None] if args.ocean else [str(MODEL_PATH)]
    if args.out and len(args.models) > 1:
        p.error("--out takes a single path; with several --model the output is always "
                "'<model_stem>_timeline_errors.npy' beside each checkpoint.")
    return args


# ---------------------------------------------------------------------------
# GCS path: chunk-based loading to avoid redundant network fetches
# ---------------------------------------------------------------------------

def _run_local(args, models: dict[str, torch.nn.Module]) -> tuple[list[str], dict, list[str]]:
    """Per-step lazy loading from a local zarr."""
    zarr_path = args.zarr or str(FIELDS_ZARR_PATH)
    print(f"[precompute] source:     {zarr_path} (local)")

    era5_cache = preload_era5_tensors(
        zarr_path,
        time_start=args.time_start,
        lazy=True,
        device="cpu",
    )
    if args.time_end:
        mask = era5_cache["time"] <= np.datetime64(args.time_end, "ns")
        era5_cache["time"] = era5_cache["time"][mask]
        era5_cache["ds"]   = era5_cache["ds"].sel(time=slice(None, args.time_end))

    timestamps = get_available_pred_timestamps(era5_cache, as_str=True)
    n = len(timestamps)
    print(f"[precompute] {n} valid prediction timesteps")
    if n == 0:
        raise SystemExit("[precompute] No timestamps found.")

    from tqdm import tqdm  # noqa: PLC0415

    series = {name: _new_series() for name in models}
    first  = next(iter(models))
    bs = args.batch_size

    with tqdm(total=n, unit="step", dynamic_ncols=True) as bar:
        for start in range(0, n, bs):
            chunk = timestamps[start : start + bs]
            inputs = [get_era5_step_inputs(era5_cache, t, device=args.device) for t in chunk]
            truths = []
            for t in chunk:
                ts_np = np.datetime64(t.replace(" ", "T").rstrip("Z"), "ns")
                idx   = int(np.where(era5_cache["time"] == ts_np)[0][0])
                truths.append(get_era5_normalized_state(era5_cache, idx, device=args.device))
            for name, model in models.items():
                with torch.no_grad():
                    preds = run_inference_batch(model, inputs, device=args.device)
                for b in range(len(chunk)):
                    _append_stats(series[name], _step_stats(preds[b], truths[b]))
            bar.update(len(chunk))
            bar.set_postfix(mean_error=f"{series[first]['mean_error'][-1]:.4f}")

    return timestamps, series, _era5_column_names()


def _run_ocean(args, models: dict[str, torch.nn.Module]) -> tuple[list[str], dict, list[str]]:
    """
    Ocean path: local analysis window, masked reductions over the 27 predicted channels.

    There is no streaming variant. `prepare_OceanBench` loads the whole window eagerly (it is
    a handful of days, not an archive) and reports which timesteps actually carry data --
    zarr returns NaN for absent chunks, so a partially downloaded store looks complete until
    something reads it.
    """
    from services.ocean import (  # noqa: PLC0415
        get_ocean_step_inputs, ocean_state_at, prepare_OceanBench,
    )

    print("[precompute] source:     local ocean zarrs (analysis state + IFS forcing)")
    cache = prepare_OceanBench(
        state_zarr=args.state_zarr,
        atmo_zarr=args.atmo_zarr,
        device=args.device,
    )

    times = cache["time"]
    # The first two steps condition the model; predictions start at index 2. Of those, score
    # only the ones whose whole 3-step forcing window is genuine forecast (lead >= 1) -- a
    # step forced partly by analysis is not measuring the same thing, since analysis is
    # information the model would not have ahead of time. On a window with enough inits this
    # drops nothing; on the shipped single-init window it drops the first two.
    pred_times = cache["forecast_only_times"]
    skipped = len(times) - 2 - len(pred_times)
    if skipped:
        print(f"[precompute] skipping {skipped} step(s) whose forcing window contains "
              f"analysis leads (<= 0); they are not comparable to the rest")

    # --time-start/--time-end only narrow what was downloaded. Their ERA5 defaults refer to
    # the 2020 test year and would silently empty an ocean window, so they apply only when
    # the caller set them explicitly.
    if args.time_start and args.time_start != _DEFAULT_TIME_START:
        pred_times = pred_times[pred_times >= np.datetime64(args.time_start, "ns")]
    if args.time_end and args.time_end != _DEFAULT_TIME_END:
        pred_times = pred_times[pred_times <= np.datetime64(args.time_end, "ns")]

    n = len(pred_times)
    print(f"[precompute] window:     {str(times[0])[:10]} .. {str(times[-1])[:10]}  "
          f"({len(times)} steps with data)")
    print(f"[precompute] {n} valid prediction timesteps")
    if n == 0:
        raise SystemExit(
            "[precompute] No prediction timesteps in range. The downloaded window is "
            f"{str(times[0])[:10]} to {str(times[-1])[:10]}; --time-start/--time-end must "
            "fall inside it."
        )

    # Read through the accessor: the cache is lazy by default, so state_normalized is None
    # and steps are materialised on demand.
    n_state = len(cache["state_features"])          # 27 -- excludes the ice-density input
    valid = cache["valid_mask"].to(args.device)     # (N_grid, 27) True where scoreable
    n_valid = int(valid.sum())
    print(f"[precompute] scoring over {n_valid:,} of {valid.numel():,} "
          f"(node, channel) cells — land and below-seafloor excluded")

    from tqdm import tqdm  # noqa: PLC0415

    series = {name: _new_series() for name in models}
    first  = next(iter(models))
    bs = args.batch_size

    with tqdm(total=n, unit="step", dynamic_ncols=True) as bar:
        for start in range(0, n, bs):
            chunk = pred_times[start : start + bs]
            inputs = [get_ocean_step_inputs(cache, t, device=args.device) for t in chunk]
            truths = []
            for t in chunk:
                idx = int(np.flatnonzero(times == t)[0])
                truths.append(ocean_state_at(cache, idx, args.device)[..., :n_state])
            for name, model in models.items():
                with torch.no_grad():
                    preds = run_inference_batch(model, inputs, device=args.device)
                for b in range(len(chunk)):
                    _append_stats(
                        series[name],
                        _step_stats(preds[b][..., :n_state], truths[b], valid=valid),
                    )
            bar.update(len(chunk))
            bar.set_postfix(mean_error=f"{series[first]['mean_error'][-1]:.4f}")

    timestamps = [str(np.datetime64(t, "s")) for t in pred_times]
    return timestamps, series, list(cache["state_features"])


def _run_gcs(args, models: dict[str, torch.nn.Module]) -> tuple[list[str], dict, list[str]]:
    """
    Chunk-based GCS streaming. Opens the WeatherBench2 zarr lazily, then loads
    _GCS_CHUNK timesteps at a time to avoid re-fetching the same zarr chunks.

    Every model is scored on each chunk before the next fetch, so adding models costs
    inference time but not bandwidth.

    For the 2020 test year (~1460 steps):
      ceil(1460 / _GCS_CHUNK) = 37 GCS batches
      each batch ≈ 40 steps × 9.6 MB = 384 MB uncompressed
      total fetched ≈ 14 GB (first run); instant on re-run if OS cache is warm.
    """
    import gcsfs  # noqa: PLC0415
    import xarray as xr  # noqa: PLC0415

    print(f"[precompute] source:     GCS  {_GCS_URL}")

    fs  = gcsfs.GCSFileSystem(token="anon")
    ds_full = (
        xr.open_zarr(fs.get_mapper(_GCS_URL), consolidated=True)
        [_NEURAL_LAM_VARS]
        .sel(time=slice(args.time_start, args.time_end or None))
    )

    time_vals = ds_full.time.values.astype("datetime64[ns]")
    T = len(time_vals)
    if T < 3:
        raise SystemExit(f"[precompute] Only {T} timesteps in range — need ≥ 3.")

    pred_timestamps = [str(t) for t in time_vals[2:]]
    n = len(pred_timestamps)
    print(f"[precompute] {n} valid prediction timesteps")
    est_batches = int(np.ceil(n / _GCS_CHUNK))
    print(f"[precompute] {est_batches} GCS batches of {_GCS_CHUNK} steps each")

    # TOA normalization and static stats
    toa_norm = _load_toa_normalization(TOA_NORM_PATH) if TOA_NORM_PATH else None
    data_mean, data_std = _load_static("cpu")

    from tqdm import tqdm  # noqa: PLC0415

    series = {name: _new_series() for name in models}
    first  = next(iter(models))
    bs = args.batch_size

    with tqdm(total=n, unit="step", dynamic_ncols=True) as bar:
        for pred_start in range(0, n, _GCS_CHUNK):
            pred_end = min(pred_start + _GCS_CHUNK, n)
            z_start  = pred_start
            z_end    = pred_end + 2

            bar.set_description(f"fetch {pred_start}–{pred_end-1}")
            chunk_ds   = ds_full.isel(time=slice(z_start, z_end)).load()
            state_np   = _stack_state(chunk_ds)
            forcing_np = _compute_forcing(chunk_ds, toa_norm)
            state_t    = torch.tensor(state_np, dtype=torch.float32)
            state_norm = (state_t - data_mean) / data_std

            chunk_preds  = pred_end - pred_start
            local_inputs: list[tuple] = []
            local_truths: list[torch.Tensor] = []
            for li in range(2, 2 + chunk_preds):
                fw = torch.tensor(
                    np.concatenate([forcing_np[li-2], forcing_np[li-1], forcing_np[li]], axis=-1)[None],
                    dtype=torch.float32,
                )
                local_inputs.append((
                    state_norm[li-1].unsqueeze(0).to(args.device),
                    state_norm[li-2].unsqueeze(0).to(args.device),
                    fw.to(args.device),
                ))
                local_truths.append(state_norm[li].to(args.device))

            bar.set_description("infer")
            for sub_start in range(0, len(local_inputs), bs):
                sub_inputs = local_inputs[sub_start : sub_start + bs]
                sub_truths = local_truths[sub_start : sub_start + bs]
                for name, model in models.items():
                    with torch.no_grad():
                        preds = run_inference_batch(model, sub_inputs, device=args.device)
                    for b in range(len(sub_inputs)):
                        _append_stats(series[name], _step_stats(preds[b], sub_truths[b]))
                bar.update(len(sub_inputs))
                bar.set_postfix(mean_error=f"{series[first]['mean_error'][-1]:.4f}")

    return pred_timestamps, series, _era5_column_names()


def main() -> None:
    args = parse_args()

    # (name, checkpoint path, output path) per model. The name keys the series dicts and
    # is the checkpoint stem, which is also what the output filename is built from -- the
    # backend looks for '<model_stem>_timeline_errors.npy' beside the checkpoint and
    # nowhere else.
    models: dict[str, torch.nn.Module] = {}
    out_paths: dict[str, Path] = {}

    if args.ocean:
        from services import ocean as _ocean  # noqa: PLC0415

        # load_ocean_model resolves the default checkpoint itself and wraps the four-argument
        # SeaCast module in the three-argument adapter run_inference_batch expects. Reuse its
        # own DATA_ROOT for the output path rather than guessing a second time -- locally that
        # is backend/data (set DATA_ROOT), in the container /usr/data.
        for m in args.models:
            model_path = Path(m) if m else None
            model = _ocean.load_ocean_model(model_path, device=args.device)
            resolved = model_path or (
                _ocean._DATA_ROOT / "paper_checkpoints" / "seacast_scripted.pt"
            )
            models[resolved.stem] = model
            out_paths[resolved.stem] = resolved.parent / f"{resolved.stem}_timeline_errors.npy"
    else:
        for m in args.models:
            resolved = Path(m)
            models[resolved.stem] = load_model(resolved, device=args.device)
            out_paths[resolved.stem] = resolved.parent / f"{resolved.stem}_timeline_errors.npy"

    if args.out:
        out_paths[next(iter(models))] = Path(args.out)

    for model in models.values():
        model.eval()

    print(f"[precompute] models:     {', '.join(models)}")
    if not args.ocean:
        print(f"[precompute] time-start: {args.time_start}")
        print(f"[precompute] time-end:   {args.time_end or 'all available'}")
    print(f"[precompute] device:     {args.device}  batch-size: {args.batch_size}")

    t0 = time.time()
    if args.ocean:
        timestamps, series, column_names = _run_ocean(args, models)
    elif args.gcs:
        timestamps, series, column_names = _run_gcs(args, models)
    else:
        timestamps, series, column_names = _run_local(args, models)

    for name, s in series.items():
        _save_timeline(out_paths[name], timestamps, s, column_names)

    print(f"[precompute] Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
