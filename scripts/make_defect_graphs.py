"""
Build ocean graphs that contain known structural defects, for use as GNN-VIEW fixtures.

Every graph here is produced by the *unmodified* vendored constructor,
`vendor/baltic-sea-forecasting/neural_lam/create_global_graph.py`. Nothing about the
topology is synthesised or edited afterwards -- the defects come out of the real script
run with different values of the two flags that govern them:

  --max_chord_len        Drops any mesh edge longer than this chord on the unit sphere
                         (level 0; scaled by refinement_factor**(0.5*level) above it).
                         Land filtering is otherwise only a *midpoint* test, so raising
                         this admits long edges that step over a landmass but happen to
                         have a wet midpoint. Lowering it below the mesh's natural
                         spacing deletes legitimate edges and fragments the mesh.
  --connect_disconnected Adds a nearest-neighbour edge for grid nodes left with no g2m
                         out-edge / no m2g in-edge. It does NOT repair m2m, so isolated
                         *mesh* nodes survive it.

Two things are done differently from `neural_lam.create_global_graph.cli`, both of them
plumbing rather than algorithm:

1. The grid is read straight from `backend/data/ocean_datasets/analysis_ocean.zarr`
   instead of through a neural-lam datastore. `load_grid_from_datastore` returns the
   surface mask split into (sea_xy, land_xy) and that is exactly what this reproduces --
   675,219 sea points, 303,981 land points, on the same 0.25 degree lattice the shipped
   `global_*_analysis` graphs were built on. It avoids needing mllam_data_prep and the
   original zarr paths, which are not available outside the HPC system.

2. K-means cluster centres are cached on disk. The node set at every mesh level is
   independent of `max_chord_len` -- `filter_global_edges_land` decides node retention
   with `d_sea <= d_land` and applies the length cap to *edges* only -- so one clustering
   is reused across the whole sweep. Cache key includes the input points, so a changed
   resolution recomputes.

Usage:
    python scripts/make_defect_graphs.py --list
    python scripts/make_defect_graphs.py --variants ico_baseline ico_noconnect
    python scripts/make_defect_graphs.py            # everything

Output, per variant:
    <outdir>/bsf/<name>/       raw constructor output (BSF format)
    <outdir>/upload/<name>/    converted neural-lam bundle -- this is what to upload
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / "vendor" / "baltic-sea-forecasting" / "neural_lam"
OCEAN_ZARR = REPO / "backend" / "data" / "ocean_datasets" / "analysis_ocean.zarr"

sys.path.insert(0, str(REPO / "backend" / "src"))


# ---------------------------------------------------------------------------
# Importing the vendored constructor without its training stack
# ---------------------------------------------------------------------------

def load_vendored():
    """
    Import `create_global_graph` from the vendored tree.

    `neural_lam/__init__.py` imports the models, which need mlflow / lightning, and
    `create_global_graph` imports the datastore, which needs mllam_data_prep. None of that
    is used to build a graph. Registering the package under an alias with an empty init and
    stubbing the two datastore-side names lets the real module import untouched.
    """
    if "nlam" not in sys.modules:
        pkg = types.ModuleType("nlam")
        pkg.__path__ = [str(VENDOR)]
        sys.modules["nlam"] = pkg

        cfg = types.ModuleType("nlam.config")
        cfg.load_config_and_datastores = None
        sys.modules["nlam.config"] = cfg

        ds = types.ModuleType("nlam.datastore")
        ds.__path__ = []
        sys.modules["nlam.datastore"] = ds
        base = types.ModuleType("nlam.datastore.base")

        class BaseRegularGridDatastore:  # only referenced as a type hint
            pass

        base.BaseRegularGridDatastore = BaseRegularGridDatastore
        sys.modules["nlam.datastore.base"] = base

    return importlib.import_module("nlam.create_global_graph")


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

GRID_STEP_DEG = 0.25


def load_sea_land(stride: int = 1):
    """
    (sea_xy, land_xy) as [longitude, latitude] in degrees.

    Mirrors `create_global_graph.load_grid_from_datastore`: the datastore mask is
    [mask_feature, grid_index] with one feature per state variable, and feature 0 is the
    surface one, which is what the graph's grid nodes correspond to.

    stride > 1 keeps every stride-th lattice line in both axes, giving a genuine
    (0.25 * stride) degree land/sea mask rather than a random subsample. Selection is by
    coordinate value, not array index, so it does not depend on the raveling order.
    """
    import zarr

    z = zarr.open(str(OCEAN_ZARR), mode="r")
    lon = np.asarray(z["longitude"]).astype(np.float32)
    lat = np.asarray(z["latitude"]).astype(np.float32)
    surface = np.asarray(z["mask"][0]).astype(bool)
    xy = np.stack([lon, lat], axis=1)

    if stride > 1:
        li = np.rint((lon - lon.min()) / GRID_STEP_DEG).astype(np.int64)
        ai = np.rint((lat - lat.min()) / GRID_STEP_DEG).astype(np.int64)
        on_lattice = (li % stride == 0) & (ai % stride == 0)
        xy, surface = xy[on_lattice], surface[on_lattice]

    return xy[surface], xy[~surface]


# ---------------------------------------------------------------------------
# K-means cache
# ---------------------------------------------------------------------------

def install_chunked_estep():
    """
    Make the k-means E-step memory-bounded so production-resolution meshes are buildable.

    `_spherical_labels_inertia` forms `X @ centers.T` in one allocation. At the production
    cluster resolution that is (675,219 x 33,761) float32 = 91 GB and the process dies.
    Chunking over rows gives the identical result -- each row's argmax is independent of how
    the rows are grouped -- at ~1.4 GB per chunk. Only the float summation order of the
    reported inertia changes, which is not used to pick labels.

    Patched on the module so `_spherical_kmeans_single_lloyd` picks it up through its global.
    """
    import importlib

    skm = importlib.import_module("nlam.graphs.spherical_kmeans")

    def chunked(X, sample_weight, centers, rows=20_000):
        n = X.shape[0]
        labels = np.empty(n, dtype=np.int64)
        inertia = 0.0
        ct = np.ascontiguousarray(centers.T)
        for lo in range(0, n, rows):
            hi = min(lo + rows, n)
            sim = np.asarray(X[lo:hi] @ ct)
            lb = np.argmax(sim, axis=1)
            labels[lo:hi] = lb
            best = sim[np.arange(hi - lo), lb]
            inertia += float(np.sum(sample_weight[lo:hi] * 2.0 * (1.0 - best)))
        return labels, inertia

    skm._spherical_labels_inertia = chunked
    print("    [kmeans] chunked E-step installed (memory-bounded)", flush=True)


def install_kmeans_cache(cache_dir: Path):
    """
    Replace SphericalKMeans inside the cluster-mesh builder with a disk-memoised subclass.

    Only `cluster_centers_` is read by the caller, so that is all that is stored. The key
    covers the input points, the weights, k and the seed, so nothing is reused across
    resolutions by accident.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    gcm = importlib.import_module("nlam.graphs.global_cluster_mesh")
    Base = gcm.SphericalKMeans

    class CachedSphericalKMeans(Base):
        def fit(self, X, y=None, sample_weight=None):
            h = hashlib.sha256()
            h.update(np.ascontiguousarray(X, dtype=np.float64).tobytes())
            if sample_weight is not None:
                h.update(np.ascontiguousarray(sample_weight, dtype=np.float64).tobytes())
            h.update(f"{self.n_clusters}|{self.random_state}|{self.n_init}".encode())
            path = cache_dir / f"kmeans_{self.n_clusters}_{h.hexdigest()[:16]}.npy"

            if path.exists():
                self.cluster_centers_ = np.load(path)
                print(f"    [kmeans cache hit] k={self.n_clusters} -> {path.name}", flush=True)
                return self

            print(f"    [kmeans] fitting k={self.n_clusters} on {X.shape[0]:,} points ...", flush=True)
            t0 = time.time()
            super().fit(X, y=y, sample_weight=sample_weight)
            print(f"    [kmeans] done in {time.time() - t0:.0f}s ({self.n_iter_} iterations)", flush=True)
            np.save(path, self.cluster_centers_)
            return self

    gcm.SphericalKMeans = CachedSphericalKMeans


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------
#
# `ico_*` reproduce the shipped `global_hierarchical_analysis` construction exactly
# (--type hierarchical --splits 6 --levels 3). Free to build -- no clustering.
#
#   NOTE: for the icosahedral types, create_global_graph does NOT forward --max_chord_len
#   to the m2m land filter; that call uses the function default of 0.1 at every level.
#   The flag only reaches the m2g filter. So the m2m sweep has to be done on `cluster`.
#
# `cluster_*` are the Njord/SeaCast construction (spherical k-means + spherical Delaunay
# + land filter), at 1/16 of the production mesh resolution so a full sweep is buildable
# on a laptop -- production used --grid_to_first_mesh_refinement 20 (33,777 level-0 mesh
# nodes), these use 320 (~2,100). Refinement factor and level count are unchanged.
#
# Chord to great-circle: km = 6371 * 2 * asin(chord / 2).
#   0.03 ~ 191 km | 0.05 ~ 319 km | 0.10 ~ 637 km | 0.20 ~ 1275 km | 0.40 ~ 2557 km

CLUSTER_BASE = dict(
    graph_type="cluster",
    levels=3,
    mesh_refinement_factor=4,
    grid_to_first_mesh_refinement=320,
    g2m_radius=0.67,
    m2g_k=3,
)
ICO_BASE = dict(
    graph_type="hierarchical",
    splits=6,
    levels=3,
    g2m_radius=0.67,
    m2g_k=3,
)

VARIANTS = {
    # -- icosahedral, production parameters -------------------------------------------
    "ico_baseline": dict(
        **ICO_BASE, max_chord_len=0.1, connect_disconnected=True,
        note="reproduces shipped global_hierarchical_analysis",
    ),
    "ico_noconnect": dict(
        **ICO_BASE, max_chord_len=0.1, connect_disconnected=False,
        note="isolated grid nodes: no g2m out-edge and/or no m2g in-edge",
    ),

    # -- cluster, chord sweep with connect_disconnected on -----------------------------
    # Level-0 mesh spacing at this resolution is ~0.071 chord (~450 km), so the cap starts
    # cutting into the bulk of the distribution below ~0.08 and stops binding above ~0.10.
    "cluster_c006": dict(
        **CLUSTER_BASE, max_chord_len=0.06, connect_disconnected=True,
        note="cap below the bulk of the length distribution -- mesh collapses",
    ),
    "cluster_c007": dict(
        **CLUSTER_BASE, max_chord_len=0.07, connect_disconnected=True,
        note="cap at the median edge length -- heavy fragmentation, isolated mesh nodes",
    ),
    "cluster_c008": dict(
        **CLUSTER_BASE, max_chord_len=0.08, connect_disconnected=True,
        note="cap in the upper tail -- mild fragmentation",
    ),
    "cluster_c010": dict(
        **CLUSTER_BASE, max_chord_len=0.10, connect_disconnected=True,
        note="constructor default -- reference point",
    ),
    "cluster_c020": dict(
        **CLUSTER_BASE, max_chord_len=0.20, connect_disconnected=True,
        note="cap relaxed -- long edges start surviving the midpoint-only land test",
    ),
    "cluster_c040": dict(
        **CLUSTER_BASE, max_chord_len=0.40, connect_disconnected=True,
        note="cap effectively off -- long edges crossing land",
    ),

    # -- both defect classes at once ----------------------------------------------------
    "cluster_c040_noconnect": dict(
        **CLUSTER_BASE, max_chord_len=0.40, connect_disconnected=False,
        note="land-crossing mesh edges AND isolated grid nodes",
    ),
    "cluster_c007_noconnect": dict(
        **CLUSTER_BASE, max_chord_len=0.07, connect_disconnected=False,
        note="fragmented mesh AND isolated grid nodes",
    ),
}


# Flat (single-scale) counterparts of the set above: three subgraphs instead of nine, no
# mesh_up / mesh_down, no coarse mesh levels.
#
#   cluster    --levels 1        one k-means level; --max_chord_len still governs m2m
#   multiscale --levels 1        the icosahedral level-0 mesh on its own, which is exactly
#                                the bottom level of the hierarchical build, so the two are
#                                directly comparable
#
# Note these are NOT much smaller on disk. The weight is the 675,219-node grid: g2m and m2g
# are ~98 MB of the ~105 MB bundle either way. What flattening buys is a simpler graph to
# explore. Use --grid-stride for an actually lighter file.

FLAT_CLUSTER = dict(graph_type="cluster", levels=1, mesh_refinement_factor=4,
                    grid_to_first_mesh_refinement=320, g2m_radius=0.67, m2g_k=3)
FLAT_ICO = dict(graph_type="multiscale", splits=6, levels=1, g2m_radius=0.67, m2g_k=3)

VARIANTS.update({
    "flat_ico_baseline": dict(
        **FLAT_ICO, max_chord_len=0.1, connect_disconnected=True,
        note="flat: icosahedral level-0 mesh alone, repaired",
    ),
    "flat_ico_noconnect": dict(
        **FLAT_ICO, max_chord_len=0.1, connect_disconnected=False,
        note="flat: isolated grid nodes in g2m / m2g",
    ),
    "flat_cluster_c006": dict(
        **FLAT_CLUSTER, max_chord_len=0.06, connect_disconnected=True,
        note="flat: cap below the length distribution -- mesh collapses",
    ),
    "flat_cluster_c007": dict(
        **FLAT_CLUSTER, max_chord_len=0.07, connect_disconnected=True,
        note="flat: cap at the median edge length -- isolated mesh nodes",
    ),
    "flat_cluster_c010": dict(
        **FLAT_CLUSTER, max_chord_len=0.10, connect_disconnected=True,
        note="flat: constructor default -- reference point",
    ),
    "flat_cluster_c020": dict(
        **FLAT_CLUSTER, max_chord_len=0.20, connect_disconnected=True,
        note="flat: cap relaxed -- long edges start surviving the midpoint test",
    ),
    "flat_cluster_c040": dict(
        **FLAT_CLUSTER, max_chord_len=0.40, connect_disconnected=True,
        note="flat: cap effectively off -- long edges crossing land",
    ),
    "flat_cluster_c040_noconnect": dict(
        **FLAT_CLUSTER, max_chord_len=0.40, connect_disconnected=False,
        note="flat: land-crossing mesh edges AND isolated grid nodes",
    ),
    "flat_cluster_c007_noconnect": dict(
        **FLAT_CLUSTER, max_chord_len=0.07, connect_disconnected=False,
        note="flat: isolated mesh nodes AND isolated grid nodes",
    ),
})


# Production-resolution cluster graphs -- the construction behind the shipped
# `global_cluster_analysis` (33,777 / 8,409 / 2,088 mesh nodes), from
# graph_scripts/create_graph.sh: --type cluster --levels 3 --mesh_refinement_factor 4
# --grid_to_first_mesh_refinement 20 --connect_disconnected.
#
# Reproducing it needs k-means at k = 33,761 on 675,219 points, which is only tractable with
# install_chunked_estep(). Expect ~30-60 min for the level-0 clustering; it is cached, so the
# rest of the sweep is minutes.
#
# At this resolution the default cap does NOT bind: level-0 chords run 0.0099-0.0882 against
# a cap of 0.10, so nothing is deleted by length and only the midpoint test shapes the mesh.
# Measured on the shipped graph, the useful range is therefore much lower than it looks:
#   cap 0.012 -> keeps  0.3% of level-0 edges      cap 0.020 -> keeps 77.7%
#   cap 0.016 -> keeps 26.4%  (the low-cut choice) cap 0.030 -> keeps 99.7% (no effect)
# Level-0 edge lengths: median 112 km, p99 158 km, max 562 km.
# Cap in km: 0.016 ~ 102 | 0.10 ~ 637 (default, inert) | 0.40 ~ 2557 (disables the cap)

PROD_CLUSTER = dict(graph_type="cluster", levels=3, mesh_refinement_factor=4,
                    grid_to_first_mesh_refinement=20, g2m_radius=0.67, m2g_k=3)

VARIANTS.update({
    "prod_cluster_default": dict(
        **PROD_CLUSTER, max_chord_len=0.10, connect_disconnected=True,
        note="reproduces the shipped global_cluster_analysis -- reference",
    ),
    "prod_cluster_noconnect": dict(
        **PROD_CLUSTER, max_chord_len=0.10, connect_disconnected=False,
        note="same graph, --connect_disconnected off -- isolated grid nodes",
    ),
    "prod_cluster_lowcut": dict(
        **PROD_CLUSTER, max_chord_len=0.016, connect_disconnected=True,
        note="same graph, lower edge-length cutoff -- fragmentation",
    ),
    "prod_cluster_highcut": dict(
        **PROD_CLUSTER, max_chord_len=0.40, connect_disconnected=True,
        note="same graph, higher edge-length cutoff -- long edges crossing land",
    ),
    # No length cap at all. filter_global_edges_land takes max_chord_len=None and skips
    # the test outright, but the cluster path multiplies the cap per level
    # (global_cluster_mesh.py:200), so None raises TypeError there. 2.0 is the exact
    # equivalent: chord length on a unit sphere is bounded by 2 (antipodal points), and
    # the per-level scaling only makes the cap larger, so `chord <= cap` is always true.
    # Leaves the midpoint-over-sea test as the only thing filtering edges.
    "prod_cluster_nocap": dict(
        **PROD_CLUSTER, max_chord_len=2.0, connect_disconnected=True,
        note="no length cap -- midpoint land test is the only filter",
    ),
})


# ---------------------------------------------------------------------------
# Conversion to an uploadable bundle
# ---------------------------------------------------------------------------

def to_upload_bundle(bsf_dir: Path, out_dir: Path, sea_xy: np.ndarray) -> dict:
    """
    Convert with the backend's own converter, falling back to the known grid if it refuses.

    `recover_grid_positions` rebuilds grid coordinates from g2m/m2g edge vdiff. Two things
    defeat it, and both are properties of the graphs this script exists to produce:

      * A grid node with neither a g2m nor an m2g edge has no vdiff to recover from, and
        the converter raises. That disconnection is the defect being demonstrated.
      * Without --connect_disconnected the g2m edge features are (len, dlat, dlon) in
        degrees, 3 columns; with it, connect_disconnected_g2m rebuilds them through
        add_edge_features_pyg as (len, dx, dy, dz) on the unit sphere, 4 columns. The
        converter was written against graphs built with the flag and assumes the latter,
        so it fails on the width. (The degree-valued form is unusable anyway: the raw
        lon difference is not antimeridian-corrected and reaches 360.)

    In either case the grid is written from the true sea point coordinates instead --
    known here, since they are the constructor's own input.
    """
    import torch
    from services.bsf_converter import convert_bsf_to_neural_lam

    try:
        summary = convert_bsf_to_neural_lam(bsf_dir, out_dir)
        summary["grid_source"] = "recovered from edge vdiff (backend converter)"
        return summary
    except (ValueError, RuntimeError) as e:
        print(f"    backend converter could not run ({type(e).__name__}: {e});"
              f" writing grid from known sea points", flush=True)

    from services.bsf_converter import (
        detect_flavour, mesh_positions, to_lat_lon, localize_hierarchy, _load, _grid_row,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_features = _load(bsf_dir, "mesh_features.pt")
    if not isinstance(mesh_features, list):
        mesh_features = [mesh_features]
    flavour = detect_flavour(mesh_features)
    mesh_pos = mesh_positions(mesh_features, flavour)
    num_mesh = mesh_pos[0].shape[0]
    mesh_lat_lon = [to_lat_lon(p, flavour, None) for p in mesh_pos]

    # grid_lat_lon is [lat, lon]; sea_xy is [lon, lat]
    grid_lat_lon = torch.from_numpy(np.stack([sea_xy[:, 1], sea_xy[:, 0]], axis=1)).to(torch.float32)

    m2m = _load(bsf_dir, "m2m_edge_index.pt")
    if not isinstance(m2m, list):
        m2m = [m2m]

    torch.save(grid_lat_lon, out_dir / "grid_lat_lon.pt")
    torch.save(mesh_lat_lon, out_dir / "mesh_lat_lon.pt")
    torch.save([e.to(torch.int64) for e in m2m], out_dir / "m2m_edge_index.pt")

    for name in ("g2m_edge_index.pt", "m2g_edge_index.pt"):
        edges = _load(bsf_dir, name)
        if isinstance(edges, list):
            edges = edges[0]
        edges = edges.to(torch.int64).clone()
        edges[_grid_row(edges, num_mesh, name)] -= num_mesh
        torch.save(edges, out_dir / name)

    if len(mesh_lat_lon) > 1:
        # Localized here for the same reason convert_bsf_to_neural_lam does it: the cluster
        # builder writes each mesh_up/mesh_down step in the combined [level i | level i+1]
        # index space, and a bundle carrying that convention is out of bounds against its own
        # per-level mesh_lat_lon. This path is the one every cluster variant takes -- the
        # converter above declines them precisely because they have disconnected grid nodes,
        # which is the defect being demonstrated -- so skipping it here shipped every cluster
        # bundle unlocalized.
        counts = [int(m.shape[0]) for m in mesh_lat_lon]
        for name in ("mesh_up_edge_index.pt", "mesh_down_edge_index.pt"):
            edges = _load(bsf_dir, name)
            if not isinstance(edges, list):
                edges = [edges]
            edges, _ = localize_hierarchy(edges, counts, name)
            torch.save(edges, out_dir / name)

    for name in ("g2m_features.pt", "m2g_features.pt", "m2m_features.pt",
                 "mesh_up_features.pt", "mesh_down_features.pt"):
        feats = _load(bsf_dir, name)
        if feats is not None:
            torch.save(feats, out_dir / name)

    return {
        "output_dir": str(out_dir),
        "flavour": flavour,
        "format": "hierarchical" if len(mesh_lat_lon) > 1 else "single-scale",
        "levels": len(mesh_lat_lon),
        "mesh_nodes_per_level": [int(m.shape[0]) for m in mesh_lat_lon],
        "grid_nodes": int(grid_lat_lon.shape[0]),
        "grid_source": "known sea points (backend converter not applicable)",
    }


# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", type=Path, default=REPO / "data" / "defect_graphs")
    p.add_argument("--variants", nargs="*", default=None, help="subset of names to build")
    p.add_argument("--list", action="store_true", help="print the variant table and exit")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--grid-stride", type=int, default=1,
                   help="keep every N-th lattice line in both axes: 1 = the native 0.25 deg "
                        "grid (675,219 sea points), 2 = 0.5 deg, 4 = 1 deg. Lowers the grid "
                        "node count, which is what actually shrinks a bundle. The mesh "
                        "refinement is rescaled by N**2 so the mesh stays the same size.")
    p.add_argument("--suffix", default="", help="appended to each output directory name")
    p.add_argument("--chunked-estep", action="store_true",
                   help="force the memory-bounded k-means E-step (implied by prod_* variants)")
    args = p.parse_args(argv)

    if args.list:
        w = max(len(k) for k in VARIANTS)
        for name, cfg in VARIANTS.items():
            print(f"{name:<{w}}  type={cfg['graph_type']:<12} chord={cfg['max_chord_len']:<5} "
                  f"connect={str(cfg['connect_disconnected']):<5}  {cfg['note']}")
        return

    names = args.variants or list(VARIANTS)
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        p.error(f"unknown variants: {unknown}")

    cgg = load_vendored()
    install_kmeans_cache(args.outdir / "_kmeans_cache")
    if any(n.startswith("prod_") for n in names) or args.chunked_estep:
        install_chunked_estep()

    print("Loading ocean grid ...", flush=True)
    sea_xy, land_xy = load_sea_land(args.grid_stride)
    print(f"  sea {sea_xy.shape[0]:,}  land {land_xy.shape[0]:,}"
          f"  (stride {args.grid_stride} = {GRID_STEP_DEG * args.grid_stride} deg)", flush=True)

    summaries = {}
    for name in names:
        cfg = dict(VARIANTS[name])
        note = cfg.pop("note")
        # A coarser grid would otherwise drag the mesh coarser with it, since the mesh size
        # is expressed as a ratio of grid nodes. Rescaling keeps the mesh fixed so the only
        # thing the stride changes is the grid.
        if args.grid_stride > 1 and "grid_to_first_mesh_refinement" in cfg:
            cfg["grid_to_first_mesh_refinement"] /= args.grid_stride ** 2
        out_name = name + args.suffix
        bsf_dir = args.outdir / "bsf" / out_name
        up_dir = args.outdir / "upload" / out_name

        if args.skip_existing and (up_dir / "grid_lat_lon.pt").exists():
            print(f"\n=== {out_name}: exists, skipping ===", flush=True)
            continue

        print(f"\n{'=' * 78}\n{out_name}  --  {note}\n{'=' * 78}", flush=True)
        print(f"  {cfg}", flush=True)
        t0 = time.time()
        try:
            cgg.create_global_graph(
                graph_dir_path=str(bsf_dir),
                grid_xy=sea_xy,
                sea_xy=sea_xy,
                land_xy=land_xy,
                create_plot=False,
                **cfg,
            )
        except ValueError as e:
            # A cap below the whole length distribution leaves the bottom mesh with no
            # edges at all and the constructor refuses. Degenerate rather than a failure --
            # record it and move on so one bad point does not abort the sweep.
            print(f"  SKIPPED: constructor refused these parameters -- {e}", flush=True)
            summaries[out_name] = {"variant": out_name, "note": note, "params": cfg,
                               "skipped": str(e)}
            continue
        print(f"  built in {time.time() - t0:.0f}s -> {bsf_dir}", flush=True)

        summary = to_upload_bundle(bsf_dir, up_dir, sea_xy)
        summary["variant"] = out_name
        summary["note"] = note
        summary["params"] = {k: (v if not isinstance(v, np.generic) else float(v)) for k, v in cfg.items()}
        summary["grid_stride"] = args.grid_stride
        summaries[out_name] = summary
        print(f"  converted -> {up_dir}", flush=True)
        print(json.dumps(summary, indent=2), flush=True)

    manifest = args.outdir / "manifest.json"
    existing = json.loads(manifest.read_text()) if manifest.exists() else {}
    existing.update(summaries)
    manifest.write_text(json.dumps(existing, indent=2))
    print(f"\nManifest: {manifest}")


if __name__ == "__main__":
    main()
