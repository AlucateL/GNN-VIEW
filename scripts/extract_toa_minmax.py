"""
One-time utility: extract the global TOA normalization constants (min, max)
from the full forcing.zarr and save them alongside the static files.

Run this once while the full forcing.zarr is on disk:

    python scripts/extract_toa_minmax.py \\
        --forcing  /path/to/full/forcing.zarr \\
        --out      data/era5_datasets/static/toa_normalization.pt

Or if TOA is accumulated over 1 hour (common in ERA5):

    python scripts/extract_toa_minmax.py \\
        --forcing  /path/to/full/forcing.zarr \\
        --out      data/era5_datasets/static/toa_normalization.pt \\
        --timestep-hours 1

TOA radiation is deterministic (pure astronomy — no ERA5 data required), so it
can always be recomputed on the fly for any time window. The only thing that
cannot be derived locally is the normalization constants from the full training
period. Storing these two scalars (<1 KB) is all that is needed.

After running this once, inference.py will compute TOA on the fly for any
requested window and normalize it correctly using the full-dataset statistics.

Note: The --timestep-hours parameter specifies the ACCUMULATION PERIOD of the
TOA data (how long it was accumulated), not the sampling interval. For WeatherBench2
ERA5 data with 6-hour sampling, TOA is typically accumulated over 1 hour.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import xarray as xr


def extract_toa_normalization(
    forcing_zarr: str | Path,
    out_pt: str | Path,
    timestep_hours: int | None = None,
) -> None:
    forcing_zarr = Path(forcing_zarr)
    out_pt       = Path(out_pt)

    if not forcing_zarr.exists():
        raise FileNotFoundError(f"forcing.zarr not found: {forcing_zarr}")

    print(f"Opening forcing.zarr … {forcing_zarr}")
    ds = xr.open_zarr(str(forcing_zarr), consolidated=True)
    
    # Check metadata for accumulation info
    print("\n=== Dataset Info ===")
    print(f"Dimensions: {dict(ds.dims)}")
    if hasattr(ds, 'attrs'):
        print(f"Attributes: {ds.attrs}")
    
    # Auto-detect timestep if not provided
    if timestep_hours is None:
        times = ds.time.values
        if len(times) >= 2:
            dt = (times[1] - times[0]) / np.timedelta64(1, 'h')
            timestep_hours = int(dt)
            print(f"\nAuto-detected sampling timestep: {timestep_hours} hour(s)")
            print("⚠  NOTE: This is the sampling interval, NOT necessarily the accumulation period.")
            print("   For ERA5, TOA radiation is typically accumulated over 1 hour.")
            print("   You may need to specify --timestep-hours 1 if TOA is 1-hour accumulated.")
        else:
            timestep_hours = 6  # default fallback
            print(f"\nCould not auto-detect timestep (only 1 time value), assuming {timestep_hours} hours")
    else:
        print(f"\nUsing provided accumulation period: {timestep_hours} hour(s)")
    
    timestep_seconds = timestep_hours * 3600

    # Case 1: Dataset with toa_incident_radiation as a named variable
    if "toa_incident_radiation" in ds.data_vars:
        toa = ds["toa_incident_radiation"]
    # Case 2: DataArray saved to zarr — xarray stores it as __xarray_dataarray_variable__
    # with forcing_var as a coordinate dimension
    elif "forcing_var" in ds.dims:
        var_name = list(ds.data_vars)[0]  # typically __xarray_dataarray_variable__
        toa = ds[var_name].sel(forcing_var="toa_incident_radiation")
    else:
        raise KeyError(
            "Cannot locate toa_incident_radiation in forcing zarr.\n"
            f"  data_vars : {list(ds.data_vars)}\n"
            f"  coords    : {list(ds.coords)}\n"
            "Expected either a 'toa_incident_radiation' variable or a "
            "'forcing_var' dimension with that label."
        )

    print(f"Finding global min/max over {len(toa.time)} timesteps …")
    
    # Find the indices of the global minimum and maximum normalized values
    toa_computed = toa.compute()
    toa_min_val = float(toa_computed.min().item())
    toa_max_val = float(toa_computed.max().item())
    
    # Find the grid point where these extrema occur using numpy to avoid xarray warning
    toa_data = toa_computed.values
    toa_min_idx = np.argmin(toa_data)
    toa_max_idx = np.argmax(toa_data)
    
    # Convert flat index to time, lon, lat indices
    dims = toa_computed.dims  # typically ('time', 'longitude', 'latitude')
    shape = toa_data.shape
    
    # Unravel flat index to multi-dimensional indices
    min_indices = np.unravel_index(toa_min_idx, shape)
    max_indices = np.unravel_index(toa_max_idx, shape)
    
    print(f"\n  Normalized min = {toa_min_val:.6f} at index {min_indices}")
    print(f"  Normalized max = {toa_max_val:.6f} at index {max_indices}")
    
    # Extract the coordinates for min and max points
    min_time = toa_computed.time.values[min_indices[0]].astype("datetime64[s]")
    min_lon = float(toa_computed.longitude.values[min_indices[1]])
    min_lat = float(toa_computed.latitude.values[min_indices[2]])
    
    max_time = toa_computed.time.values[max_indices[0]].astype("datetime64[s]")
    max_lon = float(toa_computed.longitude.values[max_indices[1]])
    max_lat = float(toa_computed.latitude.values[max_indices[2]])
    
    print(f"\n  Min point: time={str(min_time)[:10]}, lat={min_lat:.1f}, lon={min_lon:.1f}")
    print(f"  Max point: time={str(max_time)[:10]}, lat={max_lat:.1f}, lon={max_lon:.1f}")
    
    # Now recover the true min/max by computing TOA with GraphCast for these points
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "models" / "neural-lam"))
        import graphcast.solar_radiation as gc_sr
        
        print(f"\nRecovering true TOA min/max using GraphCast…")
        
        # Compute raw TOA at the extremal points
        raw_toa_min = gc_sr.get_toa_incident_solar_radiation(
            np.array([min_time]), np.array([min_lat]), np.array([min_lon])
        )[0, 0, 0]
        
        raw_toa_max = gc_sr.get_toa_incident_solar_radiation(
            np.array([max_time]), np.array([max_lat]), np.array([max_lon])
        )[0, 0, 0]
        
        print(f"  Raw TOA at min point: {raw_toa_min:.2f} J/m² (accumulated over {timestep_hours}h)")
        print(f"  Raw TOA at max point: {raw_toa_max:.2f} J/m² (accumulated over {timestep_hours}h)")
        
        # Save in J/m² (accumulated) to match what inference.py computes
        # inference.py calls GraphCast which returns J/m², so we keep same units for consistency
        print(f"\n  Keeping values in J/m² (accumulated) for consistency with inference.py")
        
        # If normalized values are very close to 0 or 1, use them directly
        # Otherwise, warn that the extrema may not be pure 0/1
        if abs(toa_min_val) < 0.01:
            true_toa_min = raw_toa_min
            print(f"  ✓ Using raw min ({raw_toa_min:.2f} J/m²) — norm is very close to 0")
        else:
            print(f"  ⚠ Warning: min norm is {toa_min_val:.6f}, not exactly 0")
            print(f"     Deriving toa_min from raw value")
            true_toa_min = raw_toa_min / toa_min_val if toa_min_val > 0 else raw_toa_min
        
        if abs(toa_max_val - 1.0) < 0.01:
            true_toa_max = raw_toa_max
            print(f"  ✓ Using raw max ({raw_toa_max:.2f} J/m²) — norm is very close to 1")
        else:
            print(f"  ⚠ Warning: max norm is {toa_max_val:.6f}, not exactly 1")
            print(f"     Deriving toa_max from raw value")
            true_toa_max = raw_toa_max / toa_max_val if toa_max_val > 0 else raw_toa_max
        
    except Exception as e:
        print(f"\n✗ Could not import GraphCast: {e}")
        print(f"  Falling back to normalized values (this is incorrect).")
        true_toa_min = toa_min_val
        true_toa_max = toa_max_val

    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"toa_min": true_toa_min, "toa_max": true_toa_max}, str(out_pt))

    print(f"\nSaved → {out_pt}")
    print(f"  toa_min = {true_toa_min:.2f} J/m² (was normalized: {toa_min_val:.6f})")
    print(f"  toa_max = {true_toa_max:.2f} J/m² (was normalized: {toa_max_val:.6f})")
    print("  Units: J/m² accumulated over {timestep_hours}h (matches GraphCast output)")
    print("Done. You can now delete forcing.zarr from disk.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forcing", required=True, help="Path to full forcing.zarr")
    ap.add_argument(
        "--out",
        default="data/era5_datasets/static/toa_normalization.pt",
        help="Output path for the normalization constants (default: data/era5_datasets/static/toa_normalization.pt)",
    )
    ap.add_argument(
        "--timestep-hours",
        type=int,
        default=None,
        help="Timestep duration in hours (auto-detected if not provided)",
    )
    args = ap.parse_args()
    extract_toa_normalization(args.forcing, args.out, timestep_hours=args.timestep_hours)


if __name__ == "__main__":
    main()
