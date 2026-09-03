#!/usr/bin/env python3
"""
Pre-download an ERA5 time window from WeatherBench2 GCS to a local zarr.

Eliminates cold-start GCS latency during development. Not used in deployment.

Usage (from project root, in the cute_GNNs conda env):

    python scripts/download_era5_window.py \\
        --start 2020-01-01T00:00:00 \\
        --end   2020-01-31T18:00:00

    # Custom output path:
    python scripts/download_era5_window.py \\
        --start 2020-01-01T00:00:00 \\
        --end   2020-01-31T18:00:00 \\
        --output /tmp/era5_cache/jan2020.zarr
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import xarray as xr

_GCS_URL = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
)

# must match `_NEURAL_LAM_VARS` in `scripts/precompute_timeline.py`
_NEURAL_LAM_VARS = [
    "geopotential", "specific_humidity", "temperature",
    "u_component_of_wind", "v_component_of_wind", "vertical_velocity",
    "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
    "mean_sea_level_pressure", "total_precipitation_6hr",
]

_UPPER_AIR_VARS = 6   # geopotential, specific_humidity, temperature, u, v, w
_PRESSURE_LEVELS = 13
_SURFACE_VARS = 5
_GRID_NODES = 29040
_BYTES_PER_FLOAT = 4

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _safe(ts: str) -> str:
    return ts.replace(":", "-")


def _default_output(start: str, end: str) -> Path:
    return _REPO_ROOT / "data" / "era5_cache" / f"{_safe(start)}_{_safe(end)}.zarr"


def _size_estimate(n_steps: int) -> str:
    total = n_steps * (_UPPER_AIR_VARS * _PRESSURE_LEVELS + _SURFACE_VARS) * _GRID_NODES * _BYTES_PER_FLOAT
    if total >= 1_073_741_824:
        return f"{total / 1_073_741_824:.1f} GB"
    return f"{total / 1_048_576:.0f} MB"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start",  required=True, help="Start ISO timestamp (inclusive), e.g. 2020-01-01T00:00:00")
    p.add_argument("--end",    required=True, help="End ISO timestamp (inclusive), e.g. 2020-01-31T18:00:00")
    p.add_argument("--output", default=None,  help="Output zarr path (default: ./data/era5_cache/{start}_{end}.zarr)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output = Path(args.output) if args.output else _default_output(args.start, args.end)

    if output.exists():
        sys.exit(f"[download_era5] Output path already exists: {output}\nDelete it first or choose a different --output path.")

    print(f"[download_era5] source:  {_GCS_URL}")
    print(f"[download_era5] range:   {args.start}  →  {args.end}")
    print(f"[download_era5] output:  {output}")
    print(f"[download_era5] Opening GCS zarr lazily (anonymous) …")

    ds = xr.open_zarr(
        _GCS_URL,
        storage_options={"token": "anon"},
        consolidated=True,
    )

    ds_sliced = ds[_NEURAL_LAM_VARS].sel(time=slice(args.start, args.end))

    n_steps = ds_sliced.sizes["time"]
    if n_steps == 0:
        sys.exit(f"[download_era5] No timesteps found in range {args.start} – {args.end}.")

    est = _size_estimate(n_steps)
    print(f"[download_era5] {n_steps} timesteps  |  uncompressed estimate: {est} (on-disk ~3–5× smaller after zarr compression)")
    print(f"[download_era5] Downloading …")

    os.makedirs(output.parent, exist_ok=True)

    # align_chunks=True strips the inherited source chunk encoding so it can't
    # conflict with the dask chunks of an arbitrary time slice.
    ds_sliced.to_zarr(str(output), mode="w", consolidated=True, zarr_format=2, align_chunks=True)

    print(f"[download_era5] Done — written to {output}")


if __name__ == "__main__":
    main()
