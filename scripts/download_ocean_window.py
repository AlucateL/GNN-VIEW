#!/usr/bin/env python3
"""
Pre-download an ocean analysis time window to a local zarr, and check that its land-sea
mask matches the graph it will be paired with.

Ocean counterpart of `download_era5_window.py`. Same shape: open a remote store lazily,
select variables and a time slice, write a local zarr.

WHY THE MASK CHECK MATTERS
    Model output is one value per sea point; the frontend maps those onto graph nodes by
    position. The two must therefore describe the *same* set of sea points. The analysis and
    reanalysis products differ by ~1,500 nodes, and pairing across them produces a map that
    is silently shifted rather than one that errors -- so this script refuses to be useful
    quietly. Pass --graph to assert the match.

    Confirmed node counts:
        global_cluster_analysis        675,219   (Njord,   K-means cluster mesh)
        global_hierarchical_analysis   675,219   (SeaCast, icosahedral mesh)

    Both are built from the analysis dataset, hence identical grids despite different meshes.

Usage:

    python scripts/download_ocean_window.py \\
        --source <oceanbench-store-url> \\
        --start 2024-01-01 --end 2024-01-10 \\
        --graph backend/data/graphs/global_hierarchical_analysis

    # Inspect what a store contains without downloading anything:
    python scripts/download_ocean_window.py --source <url> --describe
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

_REPO_ROOT = Path(__file__).resolve().parents[1]

# State variables and depths, from analysis_ocean_mdp_config.yaml. Order matters: it defines
# the channel order the model was trained on, and a mismatch silently reorders inputs rather
# than failing. 3 surface + 4 x 6 depths = 27 state features.
_SURFACE_VARS = ["zos", "siconc", "sithick"]
_DEPTH_VARS = ["thetao", "so", "uo", "vo"]
_DEPTHS = [0, 47, 92, 222, 318, 541]

_MASK_CANDIDATES = ["mask", "land_sea_mask", "lsm", "sea_mask"]
_BYTES_PER_FLOAT = 4


def _open(source: str) -> xr.Dataset:
    """Open a zarr store: local path, gs://, s3:// or https://."""
    opts = {"token": "anon"} if str(source).startswith("gs://") else None
    try:
        return xr.open_zarr(source, storage_options=opts, consolidated=True)
    except Exception:
        # Older stores may lack consolidated metadata.
        return xr.open_zarr(source, storage_options=opts, consolidated=False)


def _find_mask(ds: xr.Dataset) -> str | None:
    for name in _MASK_CANDIDATES:
        if name in ds or name in ds.coords:
            return name
    return None


def _is_datastore(ds: xr.Dataset) -> bool:
    """
    True for an mllam-data-prep datastore, where variables are already stacked into
    `state` over a `state_feature` dim, rather than a raw store with named variables.
    """
    return "state" in ds and "state_feature" in ds.coords


def _surface_mask(ds: xr.Dataset, name: str) -> np.ndarray:
    """
    The surface land-sea mask as a flat boolean array.

    A datastore mask is [mask_feature, grid_index] with one feature per state variable and
    depth -- deeper levels have fewer valid cells because the seafloor cuts them off. Only
    the surface feature corresponds to the graph's grid nodes, so summing all features would
    give a meaningless total.
    """
    m = np.asarray(ds[name].values)
    if m.ndim == 1:
        return m.astype(bool)
    if "mask_feature" in ds[name].dims:
        return np.asarray(ds[name].isel(mask_feature=0).values).ravel().astype(bool)
    return m.reshape(-1).astype(bool) if m.ndim == 2 and m.shape[0] > 64 else m[0].ravel().astype(bool)


def _graph_grid_nodes(graph_dir: Path) -> int:
    """
    Number of grid nodes the graph expects.

    Read from grid_lat_lon.pt when the graph has been converted, otherwise derived from the
    g2m edge index, where grid indices are offset by the mesh node count. g2m is used rather
    than m2g because m2g does not reach input-only boundary nodes.
    """
    import torch

    ll = graph_dir / "grid_lat_lon.pt"
    if ll.exists():
        t = torch.load(ll, map_location="cpu", weights_only=True)
        return int((t[0] if isinstance(t, list) else t).shape[0])

    mesh = torch.load(graph_dir / "mesh_features.pt", map_location="cpu", weights_only=True)
    num_mesh = (mesh[0] if isinstance(mesh, list) else mesh).shape[0]
    g2m = torch.load(graph_dir / "g2m_edge_index.pt", map_location="cpu", weights_only=True)
    if isinstance(g2m, list):
        g2m = g2m[0]
    return int(g2m.max()) + 1 - int(num_mesh)


def _describe(ds: xr.Dataset, source: str) -> None:
    print(f"[ocean] source: {source}\n")
    print(f"  dims   : {dict(ds.sizes)}")
    print(f"  coords : {list(ds.coords)}")
    print(f"  vars   : {list(ds.data_vars)}\n")

    if _is_datastore(ds):
        feats = [str(f) for f in ds["state_feature"].values]
        print(f"  form   : built datastore — {len(feats)} state features already stacked")
        print(f"           {', '.join(feats[:8])}{' …' if len(feats) > 8 else ''}")
    else:
        missing = [v for v in _SURFACE_VARS + _DEPTH_VARS if v not in ds]
        print("  form   : raw store with named variables")
        if missing:
            print(f"  ! missing expected state variables: {missing}")
        else:
            print("  all 7 state variables present")

    mask = _find_mask(ds)
    print(f"  mask   : {mask if mask else 'NOT FOUND — cannot build grid_lat_lon without it'}")
    if mask:
        surface = _surface_mask(ds, mask)
        print(f"           shape {tuple(ds[mask].shape)}, {int(surface.sum()):,} sea points (surface)")

    if "depth" in ds.coords:
        have = [d for d in _DEPTHS if float(d) in set(float(x) for x in ds["depth"].values)]
        print(f"  depths : need {_DEPTHS}, exactly present: {have}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--source", required=True, help="Zarr store: local path, gs://, s3:// or https://")
    p.add_argument("--start", help="Start date (inclusive), e.g. 2024-01-01")
    p.add_argument("--end", help="End date (inclusive), e.g. 2024-01-10")
    p.add_argument("--output", default=None, help="Output zarr (default: ./data/ocean_cache/{start}_{end}.zarr)")
    p.add_argument("--graph", type=Path, default=None,
                   help="Graph directory to verify the mask against — strongly recommended")
    p.add_argument("--describe", action="store_true",
                   help="Print the store's contents and exit without downloading")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[ocean] opening {args.source} …")
    ds = _open(args.source)

    if args.describe:
        _describe(ds, args.source)
        return

    if not args.start or not args.end:
        sys.exit("[ocean] --start and --end are required unless --describe is passed.")

    # Verify the grid before spending bandwidth on it.
    if args.graph:
        expected = _graph_grid_nodes(args.graph)
        mask_name = _find_mask(ds)
        if mask_name is None:
            sys.exit(
                f"[ocean] no land-sea mask in the store, so it cannot be checked against "
                f"{args.graph.name} ({expected:,} grid nodes). Looked for: {_MASK_CANDIDATES}"
            )
        sea = int(_surface_mask(ds, mask_name).sum())
        print(f"[ocean] mask sea points : {sea:,}")
        print(f"[ocean] graph grid nodes: {expected:,}  ({args.graph.name})")
        if sea != expected:
            sys.exit(
                f"[ocean] MISMATCH of {abs(sea - expected):,} nodes. This is the analysis vs "
                "reanalysis difference -- the two products have slightly different land-sea "
                "masks. Use the dataset the graph was built from rather than reindexing: a "
                "positional pairing across mismatched grids shifts every value after the "
                "first missing cell, and looks plausible on the map."
            )
        print("[ocean] mask matches the graph — safe to pair.\n")

    if _is_datastore(ds):
        # Already stacked and ordered by the config that built it — take it as-is.
        sub = ds[[v for v in ("state", "forcing", "static", "mask") if v in ds]]
    else:
        present = [v for v in _SURFACE_VARS + _DEPTH_VARS if v in ds]
        missing = [v for v in _SURFACE_VARS + _DEPTH_VARS if v not in ds]
        if missing:
            print(f"[ocean] WARNING: missing state variables {missing} — run --describe to inspect")
        sub = ds[present]
        if "depth" in sub.dims:
            sub = sub.sel(depth=_DEPTHS, method="nearest")

    time_dim = "time" if "time" in sub.dims else next(iter(sub.dims))
    sub = sub.sel({time_dim: slice(args.start, args.end)})

    n_steps = sub.sizes.get(time_dim, 0)
    if n_steps == 0:
        sys.exit(f"[ocean] no timesteps in range {args.start} – {args.end}")

    output = Path(args.output) if args.output else (
        _REPO_ROOT / "data" / "ocean_cache" / f"{args.start}_{args.end}.zarr"
    )
    if output.exists():
        sys.exit(f"[ocean] output already exists: {output}")

    n_feat = len(_SURFACE_VARS) + len(_DEPTH_VARS) * len(_DEPTHS)
    n_pts = int(np.prod([v for k, v in sub.sizes.items() if k in ("longitude", "latitude")]) or 0)
    if n_pts:
        mb = n_steps * n_feat * n_pts * _BYTES_PER_FLOAT / 1_048_576
        print(f"[ocean] {n_steps} timesteps | uncompressed estimate: {mb:,.0f} MB")

    print(f"[ocean] writing {output} …")
    os.makedirs(output.parent, exist_ok=True)
    sub.to_zarr(str(output), mode="w", consolidated=True, zarr_format=2, align_chunks=True)
    print(f"[ocean] done — {output}")


if __name__ == "__main__":
    main()
