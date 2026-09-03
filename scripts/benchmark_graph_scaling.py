#!/usr/bin/env python3
"""
How does backend latency scale with graph size?

Measures per-request latency of the graph endpoints against GraphCast-style graphs built by
the same code neural-lam uses, over a controlled ladder of sizes, and reports the scaling as
O(f(g, m, e)) where

    g = number of grid nodes
    m = number of mesh nodes
    e = total edges (g2m + m2m + m2g)

WHY GENERATED GRAPHS, AND WHY REAL ONES ARE NOT ENOUGH
------------------------------------------------------
The three shipped graphs give effectively two distinct sizes -- global_multiscale and
global_hierarchical have the *same* 29,040 grid nodes, so only ocean_hierarchical is a
separate point. Two points cannot establish an exponent. This builds a ladder of real graphs
instead, using graphcast's own icosahedral_mesh and grid_mesh_connectivity -- the modules
neural-lam's create_global_mesh.py imports -- so the mesh topology, the grid-to-mesh radius
query and the mesh-to-grid containing-triangle query are the genuine article, not a stand-in.
The three shipped graphs are then held out and used to check the result.

g AND e ARE NOT INDEPENDENT
---------------------------
This is the important structural fact, and it is a property of the construction rather than
of this benchmark. GraphCast connectivity fixes the edge counts once g and m are chosen:

    m2g = 3 per grid node       (each grid node joins the 3 vertices of its containing triangle)
    g2m ~ 1.55 per grid node    (mesh nodes within 0.6 of a max edge length)
    m2m = 6 per mesh node       (each icosahedral vertex has 6 neighbours; 8 when multiscale
                                 levels are merged, as global_multiscale does)

    => e ~ 4.55*g + 6*m,  and since g >> m in every practical graph, e ~ 4.55*g

Verified against the shipped graphs: ERA5 m2g/g = 3.00 exactly, ocean 2.98; g2m/g = 1.59 and
1.61. So there is no such thing as "hold g fixed and raise e" in a real graph -- asking for
the exponent wrt e *at fixed g* describes a graph that GraphCast cannot build. What the
ladders below do instead is vary g (mesh fixed) and vary m (grid fixed), and report how
latency moves along the manifold real graphs actually live on.

Mesh sizes are quantised by icosahedral refinement -- 42, 162, 642, 2562, 10242, 40962 -- so
the m ladder is those, not arbitrary values. The shipped ERA5 hierarchical mesh levels are
exactly 2562/642/162/42, which is the same sequence.

WHAT IS FITTED
--------------
1. The structural law e = a*g + b*m, over the generated ladder, so the coupling above is
   measured rather than asserted.
2. Per-endpoint latency against the number of elements that endpoint actually serializes
   (grid nodes for /nodes/grid, that subgraph's edges for /edges/...), pooled over every
   graph including the shipped ones. If the endpoints are linear in their own payload this
   collapses to one line per endpoint across three orders of magnitude.
3. Per-ladder latency against g and against m, which is the practical scaling: what happens
   to the tool when someone swaps in a finer grid.

COLD VS WARM
------------
services/graph.py caches raw tensors and derived metrics per folder, forever. First touch of
a graph pays torch.load plus registry construction; first touch of a subgraph's metrics pays
a haversine over every edge plus a sort. Neither recurs, and they are reported separately.

RUNNING
-------
Needs network on first run to fetch the graphcast mesh modules (cached under --gc-cache).

    docker cp scripts/benchmark_graph_scaling.py xai-iml-26-backend-1:/tmp/bench.py
    docker exec -w /usr/src/app xai-iml-26-backend-1 python /tmp/bench.py all \
        --out /usr/data/bench_scaling

`all` runs generate -> bench -> analyze -> cleanup; cleanup removes the generated graphs from
the DB and disk so they do not linger in the graph picker. Restart the backend afterwards to
release them from _raw_cache.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "http://localhost:8080"   # overridden by --api
GRAPHS_DIR = Path("/usr/data/graphs")
BENCH_PREFIX = "bench_scaling_"

# ---------------------------------------------------------------------------
# Sweep definition
#
# Two ladders, because GraphCast connectivity leaves exactly two free knobs: the grid
# resolution and the number of icosahedral refinements. Everything else -- and therefore e --
# follows from those. See the module docstring.
# ---------------------------------------------------------------------------

# Degrees per grid cell. Spans 990 grid nodes to 1,036,800, bracketing both shipped grids
# (ERA5 is 1.5 deg / 29,040 nodes; the ocean grid is 0.25 deg / 675,219 sea points).
GRID_LADDER_DEG = [8.0, 4.0, 2.0, 1.5, 1.0, 0.5, 0.25]
# Icosahedral splits -> mesh nodes 162, 642, 2562, 10242, 40962.
MESH_LADDER_SPLITS = [2, 3, 4, 5, 6]

# Held fixed while the other knob moves. splits=4 gives 2,562 mesh nodes, which is exactly
# ERA5's finest hierarchical level; 2 deg gives 16,200 grid nodes.
BASE_SPLITS = 4
BASE_RES_DEG = 2.0

# Full-world bbox: every edge and node passes the filter, so latency is a clean function of
# the graph size rather than of how much of it happens to be on screen. This is also the
# worst case, and it is reachable in the UI (minZoom is 1).
WORLD = "min_lat=-90&max_lat=90&min_lon=-180&max_lon=180"

# The shipped graphs, held out of every fit and used only to check it. graph_id -> label.
REAL_GRAPHS = {1: "global_multiscale", 2: "global_hierarchical", 3: "ocean_hierarchical"}

GC_ZIP = "https://github.com/google-deepmind/graphcast/archive/refs/heads/main.zip"

# ---------------------------------------------------------------------------
# Random-generator sweeps (--generator random)
#
# The only design that decouples all three dimensions. GraphCast connectivity forces
# e = 4.56*g + 8*m, and any realistic connectivity forces e >= g because every grid node needs
# at least one m2g edge -- so "g varies 30x while e stays fixed" is only reachable by assigning
# edges at random and leaving some grid nodes unconnected. That is unphysical as a GNN graph
# but exact for the question being asked here, which is how latency responds to element counts.
#
# The assumption it rests on was measured, not asserted: pooling latency against payload size
# across random graphs, GraphCast-generated graphs and the shipped graphs gives k ~ 1.0 with
# R^2 >= 0.99 per endpoint. Cost tracks how many elements are serialized, not how they are
# wired. The real graphs are overlaid on every plot so that stays visible rather than implied.
# ---------------------------------------------------------------------------

BASE_G, BASE_M, BASE_E = 100_000, 8_000, 300_000
G_SWEEP = [25_000, 50_000, 100_000, 200_000, 400_000, 800_000]
M_SWEEP = [2_000, 4_000, 8_000, 16_000, 32_000, 64_000]
E_SWEEP = [50_000, 100_000, 200_000, 400_000, 800_000, 1_600_000]
EDGE_SPLIT = {"g2m": 0.30, "m2m": 0.15, "m2g": 0.55}

# Overlay graphs, resolved from the catalog by folder basename so ids do not have to be
# hardcoded. Njord is the converted copy of global_cluster_analysis.
REAL_GRAPH_FOLDERS = {
    "global_multiscale":            "GraphCast multiscale",
    "global_hierarchical":          "GraphCast hierarchical",
    "ocean_hierarchical":           "SeaCast ocean",
    "bench_scaling_njord_cluster":  "Njord cluster",
}


def random_sweep_points() -> list[dict]:
    """(g, m, e) points for the random generator: one dimension moves, the other two pinned."""
    pts: dict[tuple[int, int, int], dict] = {}
    def add(g, m, e, which):
        pts.setdefault((g, m, e), {"g": g, "m": m, "e": e, "sweeps": set()})["sweeps"].add(which)
    for g in G_SWEEP:
        add(g, BASE_M, BASE_E, "g")
    for m in M_SWEEP:
        add(BASE_G, m, BASE_E, "m")
    for e in E_SWEEP:
        add(BASE_G, BASE_M, e, "e")
    out = []
    for pt in pts.values():
        pt["sweeps"] = sorted(pt["sweeps"])
        pt["name"] = f"{BENCH_PREFIX}g{pt['g']}_m{pt['m']}_e{pt['e']}"
        out.append(pt)
    return sorted(out, key=lambda pt: (pt["g"], pt["m"], pt["e"]))


def graphcast_sweep_points() -> list[dict]:
    """The (resolution, splits) pairs to build, deduplicated and tagged with their ladder."""
    pts: dict[tuple[float, int], dict] = {}
    for res in GRID_LADDER_DEG:
        pts.setdefault((res, BASE_SPLITS),
                       {"res_deg": res, "splits": BASE_SPLITS, "ladders": set()})["ladders"].add("grid")
    for sp in MESH_LADDER_SPLITS:
        pts.setdefault((BASE_RES_DEG, sp),
                       {"res_deg": BASE_RES_DEG, "splits": sp, "ladders": set()})["ladders"].add("mesh")
    out = []
    for pt in pts.values():
        pt["ladders"] = sorted(pt["ladders"])
        pt["name"] = f"{BENCH_PREFIX}res{pt['res_deg']}_s{pt['splits']}".replace(".", "p")
        out.append(pt)
    return sorted(out, key=lambda pt: (-pt["res_deg"], pt["splits"]))


# ---------------------------------------------------------------------------
# Phase: generate  (real GraphCast connectivity)
# ---------------------------------------------------------------------------

def _ensure_graphcast(cache: Path):
    """
    Puts graphcast's mesh modules on sys.path, downloading them once into `cache`.

    Only two modules are needed -- icosahedral_mesh and grid_mesh_connectivity -- and both are
    plain numpy/scipy/trimesh. The package as a whole pulls in jax and a git-based build, which
    is why this fetches the archive and extracts rather than pip-installing. The upstream repo
    now redirects to `weathernext`, and the internal imports use that name, so the original
    package layout is preserved on disk.
    """
    pkg = cache / "weathernext"
    if not (pkg / "utils" / "icosahedral_mesh.py").exists():
        import io
        import urllib.request
        import zipfile
        cache.mkdir(parents=True, exist_ok=True)
        print(f"  fetching graphcast mesh modules -> {cache}")
        blob = urllib.request.urlopen(GC_ZIP, timeout=300).read()
        zf = zipfile.ZipFile(io.BytesIO(blob))
        prefix = next(n for n in zf.namelist() if n.endswith("/weathernext/")).rsplit(
            "weathernext/", 1)[0]
        for name in zf.namelist():
            if name.startswith(prefix + "weathernext/") and name.endswith(".py"):
                tgt = cache / name[len(prefix):]
                tgt.parent.mkdir(parents=True, exist_ok=True)
                tgt.write_bytes(zf.read(name))
    if str(cache) not in sys.path:
        sys.path.insert(0, str(cache))

    # grid_mesh_connectivity needs trimesh for the containing-triangle query. Installed into
    # the same cache with --target so the running backend's site-packages is left alone.
    try:
        import trimesh  # noqa: F401, PLC0415
    except ModuleNotFoundError:
        import subprocess  # noqa: PLC0415
        print("  installing trimesh into the cache")
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
                        "--target", str(cache), "trimesh"], check=True)

    from weathernext.utils import icosahedral_mesh as im  # noqa: PLC0415
    from weathernext.utils.legacy import grid_mesh_connectivity as gm  # noqa: PLC0415
    return im, gm


def build_graph(res_deg: float, splits: int, folder: Path, im, gm) -> dict:
    """
    Writes one GraphCast-style graph in single-scale neural-lam format.

    Mirrors create_global_mesh.py: a merged multiscale mesh for m2m (which is what the shipped
    global_multiscale uses, and why its m2m is ~8 edges per node rather than the 6 a single
    level gives), a radius query for g2m, and the containing-triangle query for m2g.

    Feature tables are written as [E, 1] zeros. load_graph's single-scale branch opens them
    unconditionally so they have to exist, but none of the benchmarked endpoints reads their
    values -- /edges passes feats=None and graph_metrics derives lengths from positions.
    """
    import numpy as np
    import torch

    lat = np.arange(-90 + res_deg / 2, 90, res_deg, dtype=np.float32)
    lon = np.arange(-180 + res_deg / 2, 180, res_deg, dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lon, lat, indexing="xy")
    grid_lat_lon = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=1)

    meshes = im.get_hierarchy_of_triangular_meshes_for_sphere(splits)
    finest = meshes[-1]
    merged = im.merge_meshes(meshes) if hasattr(im, "merge_meshes") else finest

    mesh_xyz = finest.vertices
    mesh_lat = np.degrees(np.arcsin(np.clip(mesh_xyz[:, 2], -1, 1)))
    mesh_lon = np.degrees(np.arctan2(mesh_xyz[:, 1], mesh_xyz[:, 0]))
    mesh_lat_lon = np.stack([mesh_lat, mesh_lon], axis=1).astype(np.float32)

    # 0.6 * the longest edge of the finest mesh -- the radius create_global_mesh.py uses.
    edge_len = np.linalg.norm(
        finest.vertices[finest.faces[:, 0]] - finest.vertices[finest.faces[:, 1]], axis=-1).max()
    g2m_grid, g2m_mesh = gm.radius_query_indices(
        grid_latitude=lat, grid_longitude=lon, mesh=finest, radius=0.6 * float(edge_len))
    m2g_grid, m2g_mesh = gm.in_mesh_triangle_indices(
        grid_latitude=lat, grid_longitude=lon, mesh=finest)
    m2m_src, m2m_dst = im.faces_to_edges(merged.faces)

    edges = {
        "g2m": np.stack([g2m_grid, g2m_mesh]),
        "m2m": np.stack([m2m_src, m2m_dst]),
        "m2g": np.stack([m2g_mesh, m2g_grid]),
    }

    folder.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(grid_lat_lon), folder / "grid_lat_lon.pt")
    torch.save(torch.from_numpy(mesh_lat_lon), folder / "mesh_lat_lon.pt")
    for key, ei in edges.items():
        torch.save(torch.from_numpy(np.ascontiguousarray(ei)).long(),
                   folder / f"{key}_edge_index.pt")
        torch.save(torch.zeros((ei.shape[1], 1), dtype=torch.float32),
                   folder / f"{key}_features.pt")

    return {
        "g": int(grid_lat_lon.shape[0]),
        "m": int(mesh_lat_lon.shape[0]),
        "e": int(sum(ei.shape[1] for ei in edges.values())),
        "subgraph_edges_built": {k: int(v.shape[1]) for k, v in edges.items()},
    }


# Degree steps that are exact in binary, so a coordinate survives the float32 round trip with
# few significant digits and orjson writes it short. Both shipped grids already sit on one of
# these (ERA5 on 1.5, ocean on 0.25) -- matching that is what makes the bytes per node
# comparable to a real graph's. A step like 0.1 would be fine in decimal and terrible here:
# float32(0.1) prints as 0.10000000149011612 and inflates every node by ~20 characters.
_EXACT_STEPS = [4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.03125]


def _lattice(n: int, torch):
    """
    n points on a global lat/lon lattice, as a [n, 2] float32 tensor.

    A real lattice rather than random positions, because grid_spacing_deg() runs at load time
    and branches on whether the latitudes are evenly spaced -- the regular branch is the one
    both shipped grids take. The step is the coarsest exact-binary step whose full lattice
    holds at least n points; the surplus is truncated. So g=25,000 lands on 1.5 deg (ERA5's
    grid) and g=800,000 on 0.25 deg (the ocean grid).
    """
    step = next((st for st in _EXACT_STEPS
                 if int(180 / st) * int(360 / st) >= n), _EXACT_STEPS[-1])
    rows, cols = int(180 / step), int(360 / step)
    lat = torch.arange(rows, dtype=torch.float64) * step - 90.0 + step / 2
    lon = torch.arange(cols, dtype=torch.float64) * step - 180.0 + step / 2
    grid = torch.stack(torch.meshgrid(lat, lon, indexing="ij"), dim=-1).reshape(-1, 2)
    return grid[:n].to(torch.float32).contiguous()


def build_random_graph(g: int, m: int, e: int, folder: Path, seed_name: str) -> dict:
    """
    Writes one graph with real lat/lon positions but randomly assigned edges.

    Positions come from the same exact-binary-step lattice the GraphCast builder uses, so the
    JSON bytes per node match a real grid's. Only the connectivity is random: endpoints are
    drawn uniformly, which is what makes the surviving-edge count a pure function of `e`
    rather than of the spatial layout. Under the full-world bbox nothing is filtered anyway.

    The edge total is split across the three subgraphs in fixed proportions -- roughly the
    average of the shipped graphs, where m2g is always largest and m2m smallest -- so that `e`
    stays a single knob.
    """
    import torch

    counts = {k: int(round(e * f)) for k, f in EDGE_SPLIT.items()}
    counts["m2g"] += e - sum(counts.values())   # absorb rounding drift so the total is exact

    folder.mkdir(parents=True, exist_ok=True)
    torch.save(_lattice(g, torch), folder / "grid_lat_lon.pt")
    torch.save(_lattice(m, torch), folder / "mesh_lat_lon.pt")
    for key, (n_src, n_tgt) in {"g2m": (g, m), "m2m": (m, m), "m2g": (m, g)}.items():
        gen = torch.Generator().manual_seed(abs(hash((seed_name, key))) % (2**31))
        ei = torch.stack([
            torch.randint(0, max(1, n_src), (counts[key],), generator=gen, dtype=torch.int64),
            torch.randint(0, max(1, n_tgt), (counts[key],), generator=gen, dtype=torch.int64),
        ])
        torch.save(ei, folder / f"{key}_edge_index.pt")
        # load_graph's single-scale branch opens the feature tables unconditionally, so they
        # must exist; nothing benchmarked reads their values.
        torch.save(torch.zeros((counts[key], 1), dtype=torch.float32),
                   folder / f"{key}_features.pt")
    return {"g": g, "m": m, "e": e, "subgraph_edges_built": counts}


def generate(points: list[dict], gc_cache: Path, generator: str = "graphcast") -> None:
    """Builds each graph with real GraphCast connectivity, then registers it."""
    sys.path.insert(0, "/usr/src/app")
    im = gm = None
    if generator == "graphcast":
        im, gm = _ensure_graphcast(gc_cache)
    from models.models import Graph
    from models.schemas import GraphCreate
    from services.crud import GraphCRUD
    from services.database import SessionLocal

    db = SessionLocal()
    try:
        for i, pt in enumerate(points):
            folder = GRAPHS_DIR / pt["name"]
            existing = db.query(Graph).filter(Graph.folder_path == str(folder)).first()
            t0 = time.perf_counter()
            if generator == "random":
                info = build_random_graph(pt["g"], pt["m"], pt["e"], folder, pt["name"])
            else:
                info = build_graph(pt["res_deg"], pt["splits"], folder, im, gm)
            pt.update(info)
            pt["build_s"] = time.perf_counter() - t0
            if existing:
                pt["graph_id"] = existing.graph_id
            else:
                label = (f"g={pt['g']} m={pt['m']} e={pt['e']}" if generator == "random"
                         else f"{pt['res_deg']}deg splits={pt['splits']}")
                row = GraphCRUD.create(db, GraphCreate(
                    name=f"[bench] {label}",
                    folder_path=str(folder),
                    description="GraphCast-style scaling benchmark graph - safe to delete",
                ))
                pt["graph_id"] = row.graph_id
            sub = info["subgraph_edges_built"]
            print(f"  [{i+1}/{len(points)}] g={info['g']:>9,} m={info['m']:>7,} "
                  f"e={info['e']:>10,}  (g2m={sub['g2m']:,} m2m={sub['m2m']:,} "
                  f"m2g={sub['m2g']:,})  {pt['build_s']:5.1f}s -> id={pt['graph_id']}")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Phase: bench
# ---------------------------------------------------------------------------

def _time_get(url: str, accept_gzip: bool = True, decode: bool = True) -> tuple[float, int, int]:
    """
    Returns (seconds, bytes_on_wire, bytes_decoded) for one GET.

    gzip is requested because the browser requests it, and Starlette compresses in the ASGI
    send path -- so its cost is part of the latency the tool actually exhibits, not an
    artifact. Both sizes are recorded so the transfer and serialization terms can be told
    apart afterwards.
    """
    req = urllib.request.Request(url)
    if accept_gzip:
        req.add_header("Accept-Encoding", "gzip")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        raw = r.read()
        encoded = r.headers.get("Content-Encoding", "")
    dt = time.perf_counter() - t0
    # Decompression sits outside the timer, but it still holds the GIL -- so in the parallel
    # pass, one thread decompressing blocks every other thread's socket read and masks
    # whatever concurrency the server actually has. `decode=False` skips it there. The browser
    # decompresses too, but in native code without a GIL, so leaving it out is the more
    # faithful measurement of what the server can deliver.
    if not decode:
        return dt, len(raw), 0
    decoded = len(gzip.decompress(raw)) if encoded == "gzip" else len(raw)
    return dt, len(raw), decoded


def _endpoints(gid: int, node_types: list[str], subgraph_keys: list[str]) -> dict[str, tuple[str, str | None, str | None]]:
    """
    The URLs to time, as label -> (url, subgraph_key).

    The third element is the node type, for the node endpoints. Both are carried through
    because on a hierarchical graph "edges/m2m" resolves
    to m2m_0 only -- one level out of several. Attributing that request's latency to the
    graph's *total* edge count would be wrong by the ratio between them (5% vs 15% on the
    ocean graph), so the model is fitted against the edge count of the subgraph actually
    requested, not the total.
    """
    eps: dict[str, tuple[str, str | None, str | None]] = {}
    if "grid" in node_types:
        eps["nodes/grid"] = (f"{API}/graphs/{gid}/nodes/grid?{WORLD}", None, "grid")
    mesh0 = next((t for t in node_types if t.startswith("mesh")), None)
    if mesh0:
        eps["nodes/mesh"] = (f"{API}/graphs/{gid}/nodes/{mesh0}?{WORLD}", None, mesh0)
    for key in ("g2m", "m2m", "m2g"):
        real = key if key in subgraph_keys else next(
            (k for k in subgraph_keys if k.startswith(key)), None)
        if real:
            eps[f"edges/{key}"] = (f"{API}/graphs/{gid}/edges/{real}?{WORLD}", real, None)
    if "g2m" in subgraph_keys:
        eps["graph_metrics/g2m"] = (f"{API}/graphs/{gid}/graph_metrics/g2m", "g2m", None)
    return eps


def _total_latency(comps: list[dict], reps: int, warmup: int) -> dict:
    """
    Cost of one complete Explore render: every node type plus every subgraph.

    Records each component individually as well as the two totals. The per-component times
    come from the same passes as the totals, so they cost nothing extra -- and they are what
    makes the real graphs usable as scaling data on their own: four graphs are four points,
    but their components are 48, spanning 42 elements to 2 million.

    Two totals, because they answer different questions:

    `sequential_s` is the sum of the individual request times -- additive by construction, so
    its slope against g/m/e is exactly "seconds of server work per extra element".

    `parallel_s` is the wall clock with every request in flight at once, which is what the
    frontend does (Promise.all over all node types and subgraphs). The backend is a single
    uvicorn worker and the work is GIL-bound Python, so these barely differ; that gap is the
    measurement of how little concurrency currently buys.
    """
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    urls = [c["url"] for c in comps]
    for _ in range(warmup):
        for u in urls:
            _time_get(u)

    per: dict[str, list[float]] = {u: [] for u in urls}
    sizes: dict[str, tuple[int, int]] = {}
    seq, par = [], []
    for _ in range(reps):
        total = 0.0
        for u in urls:
            dt, wire, decoded = _time_get(u)
            per[u].append(dt)
            sizes[u] = (wire, decoded)
            total += dt
        seq.append(total)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(urls)) as pool:
            list(pool.map(lambda u: _time_get(u, decode=False)[0], urls))
        par.append(time.perf_counter() - t0)

    out_comps = []
    for c in comps:
        wire, decoded = sizes[c["url"]]
        out_comps.append({**{k: v for k, v in c.items() if k != "url"},
                          "latency_s": min(per[c["url"]]),
                          "bytes_wire": wire, "bytes_json": decoded})

    def _sum(pred):
        return sum(c["latency_s"] for c in out_comps if pred(c))

    return {
        "sequential_s": min(seq), "sequential_median_s": statistics.median(seq),
        "parallel_s": min(par),   "parallel_median_s": statistics.median(par),
        "n_requests": len(urls),
        "components": out_comps,
        # Per-dimension aggregates, so the y axis can be swapped to "just the part that the
        # x axis actually drives" without re-running anything.
        "grid_s":  _sum(lambda c: c["kind"] == "nodes" and c["key"] == "grid"),
        "mesh_s":  _sum(lambda c: c["kind"] == "nodes" and c["key"] != "grid"),
        "edges_s": _sum(lambda c: c["kind"] == "edges"),
        "grid_bytes":  sum(c["bytes_json"] for c in out_comps if c["kind"] == "nodes" and c["key"] == "grid"),
        "mesh_bytes":  sum(c["bytes_json"] for c in out_comps if c["kind"] == "nodes" and c["key"] != "grid"),
        "edges_bytes": sum(c["bytes_json"] for c in out_comps if c["kind"] == "edges"),
    }


def bench_one(gid: int, reps: int, warmup: int, totals_only: bool = False) -> dict:
    """Cold then warm timings for one graph. Requests are sequential -- the backend is a
    single uvicorn worker, so overlapping them would measure contention, not scaling."""
    # First touch: pays torch.load plus registry construction inside load_graph.
    t_meta_cold, _, _ = _time_get(f"{API}/graphs/{gid}/meta")
    meta = json.loads(urllib.request.urlopen(f"{API}/graphs/{gid}/meta", timeout=600).read())
    node_types = meta.get("node_types", [])
    subgraph_keys = [s["key"] for s in meta.get("subgraphs", [])]
    eps = _endpoints(gid, node_types, subgraph_keys)

    out: dict = {
        "meta_cold_s": t_meta_cold,
        "n_grid": meta.get("n_grid"),
        "n_mesh": meta.get("n_mesh"),
        "n_edges_total": sum(s.get("n_edges", 0) for s in meta.get("subgraphs", [])),
        "subgraph_edges": {s["key"]: s.get("n_edges", 0) for s in meta.get("subgraphs", [])},
        "endpoints": {},
    }

    # Metrics cold, before anything warms _statistics_cache: haversine over every edge plus
    # the ascending sort. Measured separately because it is O(e log e) where warm is O(e).
    if "graph_metrics/g2m" in eps:
        t_cold, _, _ = _time_get(eps["graph_metrics/g2m"][0])
        out["graph_metrics_cold_s"] = t_cold

    node_counts = meta.get("node_counts", {})
    out["node_counts"] = node_counts

    # Total over every component the UI fetches for one Explore render, plus each component
    # on its own. `elements` is what that request actually serializes -- node count for a node
    # type, edge count for a subgraph -- which is the x axis the component points belong on.
    comps = [{"kind": "nodes", "key": nt, "elements": node_counts.get(nt, 0),
              "url": f"{API}/graphs/{gid}/nodes/{nt}?{WORLD}"} for nt in node_types]
    comps += [{"kind": "edges", "key": k, "elements": out["subgraph_edges"].get(k, 0),
               "url": f"{API}/graphs/{gid}/edges/{k}?{WORLD}"} for k in subgraph_keys]
    out["total"] = _total_latency(comps, reps, warmup)
    t = out["total"]
    print(f"      {'TOTAL (' + str(len(comps)) + ' components)':24} "
          f"seq {t['sequential_s']*1000:9.0f} ms   parallel {t['parallel_s']*1000:9.0f} ms   "
          f"[grid {t['grid_s']*1000:.0f} / mesh {t['mesh_s']*1000:.0f} / "
          f"edges {t['edges_s']*1000:.0f} ms]")

    if totals_only:
        # The per-endpoint sampling below re-measures a subset of what the totals pass already
        # covered. Skipping it roughly halves the run when only the component figure is wanted.
        return out

    for label, (url, sub_key, node_key) in eps.items():
        for _ in range(warmup):
            _time_get(url)
        samples, wire, decoded = [], 0, 0
        for _ in range(reps):
            dt, w, d = _time_get(url)
            samples.append(dt)
            wire, decoded = w, d
        # The number of elements this request actually serializes -- recorded here rather
        # than reconstructed later, because on a hierarchical graph the node endpoint returns
        # one mesh level while `n_mesh` is the sum over all of them.
        sub_edges = out["subgraph_edges"].get(sub_key, 0) if sub_key else 0
        out["endpoints"][label] = {
            "subgraph_key": sub_key,
            "node_key": node_key,
            "subgraph_edges": sub_edges,
            "element_count": node_counts.get(node_key, 0) if node_key else sub_edges,
            "median_s": statistics.median(samples),
            "min_s": min(samples),
            "stdev_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
            "samples": samples,
            "bytes_wire": wire,
            "bytes_decoded": decoded,
        }
        print(f"      {label:22} {statistics.median(samples)*1000:9.1f} ms  "
              f"{wire/1e6:7.2f} MB wire  {decoded/1e6:8.2f} MB json")
    return out


def bench(points: list[dict], reps: int, warmup: int, include_real: bool,
          skip_synthetic: bool = False, totals_only: bool = False) -> dict:
    results = {"synthetic": [], "real": [], "config": {
        "reps": reps, "warmup": warmup, "bbox": "world",
        "base": {"res_deg": BASE_RES_DEG, "splits": BASE_SPLITS},
        "grid_ladder_deg": GRID_LADDER_DEG, "mesh_ladder_splits": MESH_LADDER_SPLITS,
    }}
    for i, p in enumerate(points):
        if "graph_id" not in p or skip_synthetic:
            continue
        tags = p.get("sweeps") or p.get("ladders") or []
        print(f"  [{i+1}/{len(points)}] g={p['g']:,} m={p['m']:,} e={p['e']:,} "
              f"(sweeps={','.join(tags)})")
        rec = bench_one(p["graph_id"], reps, warmup, totals_only)
        rec.update({"g": p["g"], "m": p["m"], "e": p["e"], "name": p["name"]})
        for k in ("res_deg", "splits", "sweeps", "ladders"):
            if k in p:
                rec[k] = p[k]
        results["synthetic"].append(rec)

    if include_real:
        # Every registered graph that is not one of this run's own sweep points counts as a
        # real graph. Discovered rather than listed, so graphs registered later -- the defect
        # graphs at assorted resolutions, say -- are picked up without touching this file.
        catalog = json.loads(urllib.request.urlopen(f"{API}/graphs", timeout=120).read())
        sweep_names = {pt["name"] for pt in points}
        found = {}
        for row in catalog:
            base = row["folder_path"].rstrip("/").split("/")[-1]
            if base in sweep_names:
                continue
            found[row["graph_id"]] = REAL_GRAPH_FOLDERS.get(base, base)
        print(f"  [real] {len(found)} graphs: {', '.join(sorted(found.values()))}")
        for gid, label in found.items():
            try:
                print(f"  [real] {label}")
                rec = bench_one(gid, reps, warmup, totals_only)
                rec.update({"g": rec["n_grid"], "m": rec["n_mesh"],
                            "e": rec["n_edges_total"], "name": label, "sweeps": []})
                results["real"].append(rec)
            except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                    json.JSONDecodeError, KeyError) as exc:
                print(f"      skipped ({exc})")
    return results


# ---------------------------------------------------------------------------
# Phase: zoom  (crop latency)
# ---------------------------------------------------------------------------

# Web Mercator, 256 px tiles, and the viewport the app actually renders into. Leaflet is
# configured minZoom 1 / maxZoom 8 and opens at zoom 4 (GNNExplorer.tsx), so that is the range
# worth measuring.
ZOOM_LEVELS = [1, 2, 3, 4, 5, 6, 7, 8]
VIEWPORT_PX = (1200, 800)
# Centre of the crop. The Atlantic, because the ocean graphs carry sea points only -- crop
# latency depends on where you are looking, and centring on land would measure an empty
# viewport rather than a representative one. This is the single largest caveat on these
# numbers: they are one location, not an average over the map.
ZOOM_CENTRE = (20.0, -40.0)
MERC_MAX_LAT = 85.05112878


def bbox_for_zoom(zoom: int, centre=ZOOM_CENTRE, viewport=VIEWPORT_PX) -> dict:
    """The lat/lon bounds Leaflet would report at this zoom, for the configured viewport."""
    lat0, lon0 = centre
    w_px, h_px = viewport
    world = 256 * (2 ** zoom)

    def lat_to_y(lat):
        lat = max(-MERC_MAX_LAT, min(MERC_MAX_LAT, lat))
        return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

    def y_to_lat(y):
        return math.degrees(2 * math.atan(math.exp(y)) - math.pi / 2)

    y_half = (h_px / 2) * (2 * math.pi / world)
    y0 = lat_to_y(lat0)
    lat_min = y_to_lat(max(-math.pi, y0 - y_half))
    lat_max = y_to_lat(min(math.pi, y0 + y_half))
    lon_half = (w_px / 2) * (360.0 / world)
    if lon_half >= 180:
        lon_min, lon_max = -180.0, 180.0
    else:
        lon_min, lon_max = lon0 - lon_half, lon0 + lon_half
    # The route treats a bbox reaching the Mercator limit as the pole, as getBbox() does.
    if lat_min <= -(MERC_MAX_LAT - 1):
        lat_min = -90.0
    if lat_max >= MERC_MAX_LAT - 1:
        lat_max = 90.0
    return {"min_lat": lat_min, "max_lat": lat_max, "min_lon": lon_min, "max_lon": lon_max}


def _count_items(body: bytes, kind: str) -> int:
    """
    Elements in a response, by counting a marker that appears exactly once per record.

    A byte scan rather than json.loads: the full-view ocean m2g response is 245 MB and would
    cost seconds and gigabytes to parse, for a number that is just a count.
    """
    return body.count(b'"source":' if kind == "edges" else b'"id":')


def bench_zoom(reps: int, warmup: int, levels=None) -> dict:
    """
    Whole-render latency at each zoom level, for every registered non-sweep graph.

    Measures what the full-view benchmark cannot: `/edges` masks and filters over the *entire*
    graph on every request and only serializes what survives, so cropping shrinks one term and
    leaves the others untouched. Full view cannot separate them -- there, everything survives.
    """
    levels = levels or ZOOM_LEVELS
    catalog = json.loads(urllib.request.urlopen(f"{API}/graphs", timeout=120).read())
    out = {"config": {"reps": reps, "warmup": warmup, "centre": ZOOM_CENTRE,
                      "viewport_px": VIEWPORT_PX, "levels": levels}, "rows": []}

    for row in catalog:
        gid = row["graph_id"]
        base = row["folder_path"].rstrip("/").split("/")[-1]
        if base.startswith(BENCH_PREFIX):
            continue
        label = REAL_GRAPH_FOLDERS.get(base, base)
        try:
            meta = json.loads(urllib.request.urlopen(f"{API}/graphs/{gid}/meta", timeout=600).read())
        except Exception as exc:                                    # noqa: BLE001
            print(f"  [{label}] skipped ({exc})")
            continue
        node_types = meta.get("node_types", [])
        sub_keys = [s["key"] for s in meta.get("subgraphs", [])]
        n_grid, n_mesh = meta.get("n_grid", 0), meta.get("n_mesh", 0)
        e_tot = sum(s.get("n_edges", 0) for s in meta.get("subgraphs", []))
        print(f"  [{label}] g={n_grid:,} m={n_mesh:,} e={e_tot:,}")

        for z in levels:
            bb = bbox_for_zoom(z)
            q = "&".join(f"{k}={v}" for k, v in bb.items())
            comps = ([{"kind": "nodes", "key": nt,
                       "url": f"{API}/graphs/{gid}/nodes/{nt}?{q}"} for nt in node_types]
                     + [{"kind": "edges", "key": k,
                         "url": f"{API}/graphs/{gid}/edges/{k}?{q}"} for k in sub_keys])
            t = _total_latency(comps, reps, warmup)
            # One extra fetch per component, outside the timing, purely to count what came back.
            returned = 0
            for c in comps:
                req = urllib.request.Request(c["url"])
                with urllib.request.urlopen(req, timeout=1800) as r:
                    returned += _count_items(r.read(), c["kind"])
            out["rows"].append({
                "graph": label, "graph_id": gid, "zoom": z,
                "g": n_grid, "m": n_mesh, "e": e_tot,
                "bbox": bb,
                "lat_span": bb["max_lat"] - bb["min_lat"],
                "lon_span": bb["max_lon"] - bb["min_lon"],
                "elements_returned": returned,
                "sequential_s": t["sequential_s"], "parallel_s": t["parallel_s"],
                "bytes_wire": sum(c["bytes_wire"] for c in t["components"]),
                "components": t["components"],
            })
            print(f"      z={z}  lat/lon span {bb['max_lat']-bb['min_lat']:6.1f}/"
                  f"{bb['max_lon']-bb['min_lon']:6.1f}  returned {returned:>9,} el  "
                  f"seq {t['sequential_s']*1000:8.0f} ms  par {t['parallel_s']*1000:8.0f} ms")
    return out


# ---------------------------------------------------------------------------
# Phase: analyze
# ---------------------------------------------------------------------------

def _power_fit(xs, ys):
    """Least-squares slope of log(y) on log(x), with R^2. The slope is the exponent k in
    t ~ x^k."""
    import numpy as np
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    keep = (xs > 0) & (ys > 0)
    xs, ys = xs[keep], ys[keep]
    if len(xs) < 3:
        return None
    lx, ly = np.log(xs), np.log(ys)
    k, c = np.polyfit(lx, ly, 1)
    resid = ly - (k * lx + c)
    ss_tot = ((ly - ly.mean()) ** 2).sum()
    r2 = 1.0 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else 1.0
    return {"exponent": float(k), "intercept": float(c), "r2": float(r2), "n": int(len(xs))}


def _additive_fit(recs, label):
    """
    Non-negative least squares for t = a*g + b*m + c*e + d.

    Non-negative because a negative per-element cost is not physically meaningful here; an
    unconstrained fit will happily produce one to absorb noise when a term is irrelevant, and
    then the coefficient reads as though the endpoint gets faster with more nodes.
    """
    import numpy as np
    from scipy.optimize import nnls
    rows, ts = [], []
    for r in recs:
        ep = r["endpoints"].get(label)
        if not ep:
            continue
        # e_sub, not the graph total -- see _endpoints. For the node endpoints it is 0, so the
        # edge coefficient is simply not identified from them and nnls leaves it at zero.
        rows.append([r["g"], r["m"], ep.get("subgraph_edges", 0), 1.0])
        ts.append(ep["min_s"])
    if len(rows) < 4:
        return None
    A, y = np.asarray(rows, float), np.asarray(ts, float)
    coef, _ = nnls(A, y)
    pred = A @ coef
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return {
        "ns_per_grid_node": float(coef[0] * 1e9),
        "ns_per_mesh_node": float(coef[1] * 1e9),
        "ns_per_subgraph_edge": float(coef[2] * 1e9),
        "const_ms": float(coef[3] * 1e3),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0,
        "n": int(len(rows)),
    }


def _predict(fit, g, m, e_sub):
    """Predicted latency in seconds. `e_sub` is the requested subgraph's edge count."""
    return (fit["ns_per_grid_node"] * g + fit["ns_per_mesh_node"] * m
            + fit["ns_per_subgraph_edge"] * e_sub) / 1e9 + fit["const_ms"] / 1e3


def _linear_fit(xs, ys):
    """Least-squares y = a*x + b with R^2. `a` is the derivative being reported."""
    import numpy as np
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    if len(x) < 3:
        return None
    a, b = np.polyfit(x, y, 1)
    pred = a * x + b
    ss_tot = ((y - y.mean()) ** 2).sum()
    return {"slope_s_per_element": float(a), "intercept_s": float(b),
            "r2": float(1 - ((y - pred) ** 2).sum() / ss_tot) if ss_tot > 0 else 1.0,
            "n": int(len(x))}


def analyze_totals(results: dict, outdir: Path) -> dict:
    """
    Total Explore-render latency against one dimension at a time, with the derivative.

    Linear axes on purpose: the question is the slope in seconds per element, and a log-log
    plot hides an intercept while making every straight line look alike. R^2 on the linear fit
    is what says whether "linear" is actually the right description.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import nnls

    syn = [r for r in results["synthetic"] if r.get("total")]
    real = [r for r in results.get("real", []) if r.get("total")]
    axes_def = [("g", "g", "grid nodes", (BASE_M, BASE_E), ("m", "e")),
                ("m", "m", "mesh nodes", (BASE_G, BASE_E), ("g", "e")),
                ("e", "e", "total edges", (BASE_G, BASE_M), ("g", "m"))]
    series = [("sequential_s", "sequential sum (server work)", "#2563eb", "o"),
              ("parallel_s",   "parallel wall clock (what the UI waits)", "#ea580c", "s")]

    # Additive model on the TOTAL, fitted across every synthetic point, used to place the
    # overlay graphs -- they have different m and e from the sweep baseline, so they cannot
    # be expected to sit on a line that holds those fixed.
    A = np.array([[r["g"], r["m"], r["e"], 1.0] for r in syn], float)
    models = {}
    for key, _, _, _ in series:
        coef, _ = nnls(A, np.array([r["total"][key] for r in syn], float))
        models[key] = coef

    fits, fig = {}, plt.figure(figsize=(17, 5.2))
    for i, (sw, xkey, xlabel, held, heldnames) in enumerate(axes_def):
        ax = fig.add_subplot(1, 3, i + 1)
        pts = sorted((r[xkey], r) for r in syn if sw in r["sweeps"])
        xs = [x for x, _ in pts]
        for key, lbl, colour, mk in series:
            ys = [r["total"][key] for _, r in pts]
            f = _linear_fit(xs, ys)
            fits.setdefault(sw, {})[key] = f
            slope_us = f["slope_s_per_element"] * 1e6
            ax.plot(xs, ys, marker=mk, ms=5, color=colour, lw=1.8,
                    label=f"{lbl}\n  {slope_us:.2f} \u00b5s/element  (R\u00b2={f['r2']:.3f})")
            ax.plot(xs, [f["slope_s_per_element"] * x + f["intercept_s"] for x in xs],
                    color=colour, lw=.8, ls="--", alpha=.55)
        for r in real:
            x = r[xkey]
            ax.scatter([x], [r["total"]["sequential_s"]], marker="*", s=170,
                       color="#111", zorder=5)
            pred = float(models["sequential_s"] @ np.array([r["g"], r["m"], r["e"], 1.0]))
            ax.scatter([x], [pred], marker="*", s=170, facecolors="none",
                       edgecolors="#111", linewidths=1.1, zorder=5)
            ax.annotate(r["name"], (x, r["total"]["sequential_s"]), fontsize=6.5,
                        xytext=(4, 5), textcoords="offset points")
        ax.set_xlabel(xlabel); ax.set_ylabel("total latency, all components (s)")
        ax.set_title(f"vs {xlabel}   ({heldnames[0]}={held[0]:,}, {heldnames[1]}={held[1]:,} fixed)",
                     fontsize=10)
        ax.grid(True, alpha=.25, lw=.5)
        ax.legend(fontsize=6.5, loc="upper left")
    fig.suptitle("Total Explore-render latency vs one dimension at a time  \u2014  "
                 "filled star = real graph measured, hollow star = model prediction",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "total_latency_derivative.png", dpi=140)
    plt.close(fig)

    overlay = [{"name": r["name"], "g": r["g"], "m": r["m"], "e": r["e"],
                "sequential_s": r["total"]["sequential_s"],
                "parallel_s": r["total"]["parallel_s"],
                "predicted_sequential_s": float(
                    models["sequential_s"] @ np.array([r["g"], r["m"], r["e"], 1.0]))}
               for r in real]
    return {"fits": fits, "overlay": overlay,
            "model_sequential": {"s_per_grid_node": float(models["sequential_s"][0]),
                                 "s_per_mesh_node": float(models["sequential_s"][1]),
                                 "s_per_edge": float(models["sequential_s"][2]),
                                 "const_s": float(models["sequential_s"][3])}}


def export_tables(results: dict, outdir: Path) -> None:
    """
    Writes every measurement to CSV so plots can be redone without re-running anything.

    Two files, because there are two natural units of analysis:

    components.csv -- one row per request per graph: what it returned, how many elements that
    was, how long it took, how many bytes. This is the table to re-plot from when the question
    is about a single kind of component.

    graphs.csv -- one row per graph: g, m, e, the per-dimension latency aggregates, and both
    totals. This is the table for whole-render questions. `grid_s` / `mesh_s` / `edges_s` are
    the latency of just the requests that the corresponding dimension drives, so a plot with g
    on the x axis can use either `total_seq_s` or `grid_s` on the y without new measurements.
    """
    import csv

    rows_c, rows_g = [], []
    for kind, recs in (("synthetic", results.get("synthetic", [])),
                       ("real", results.get("real", []))):
        for r in recs:
            t = r.get("total")
            if not t:
                continue
            for c in t.get("components", []):   # absent in pre-per-component runs
                rows_c.append({"set": kind, "graph": r["name"],
                               "sweeps": "|".join(r.get("sweeps") or r.get("ladders") or []),
                               "g": r["g"], "m": r["m"],
                               "e": r["e"], "component_kind": c["kind"],
                               "component_key": c["key"], "elements": c["elements"],
                               "latency_s": round(c["latency_s"], 6),
                               "bytes_wire": c["bytes_wire"], "bytes_json": c["bytes_json"]})
            # Which sweep(s) this point belongs to: "g" means g moved while m and e were
            # pinned. Without it the synthetic rows cannot be filtered back into their sweeps
            # from the CSV alone, and a fit over all of them at once is meaningless.
            rows_g.append({"set": kind, "graph": r["name"],
                           "sweeps": "|".join(r.get("sweeps") or r.get("ladders") or []),
                           "g": r["g"], "m": r["m"], "e": r["e"],
                           "n_components": t["n_requests"],
                           "total_seq_s": round(t["sequential_s"], 6),
                           "total_parallel_s": round(t["parallel_s"], 6),
                           # Empty, not 0, when the run predates per-component recording:
                           # a written 0 reads as "the grid requests took no time", which is
                           # false and silently poisons any fit that includes these rows.
                           **{k: (round(t[k], 6) if k in t else "")
                              for k in ("grid_s", "mesh_s", "edges_s")},
                           **{k: t.get(k, "") for k in
                              ("grid_bytes", "mesh_bytes", "edges_bytes")}})
    for name, rows in (("components.csv", rows_c), ("graphs.csv", rows_g)):
        if not rows:
            continue
        with (outdir / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {outdir / name}  ({len(rows)} rows)")


def analyze_real_graphs(results: dict, outdir: Path) -> None:
    """
    The shipped graphs on their own: total render latency against g, against m, against e.

    Shared y axis across the three panels on purpose -- it is the same quantity each time, so
    the panels are directly comparable and the eye can see which dimension actually moves it.

    Each panel also carries the component-only series (hollow markers): the latency of just
    the requests that panel's dimension drives. Both are in graphs.csv, so either can become
    the y axis without re-measuring.

    No line is fitted. Four graphs occupy roughly two distinct sizes -- the two ERA5 graphs are
    nearly identical, as are the two ocean ones -- and a slope through two clusters would be
    a fit to two points wearing a disguise. The decoupled synthetic sweeps are where the
    derivative comes from; this figure is for reading real magnitudes off real graphs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real = [r for r in results.get("real", [])
            if r.get("total") and r["total"].get("components")]
    if not real:
        print("  no per-component real-graph data in this results file - skipping that figure")
        return
    panels = [("g", "grid nodes", "grid_s", "grid-node requests only"),
              ("m", "mesh nodes", "mesh_s", "mesh-node requests only"),
              ("e", "total edges", "edges_s", "edge requests only")]
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
    for ax, (xkey, xlabel, ckey, clabel) in zip(axes, panels):
        for i, r in enumerate(sorted(real, key=lambda r: r["e"])):
            x, t = r[xkey], r["total"]
            ax.scatter([x], [t["sequential_s"]], s=110, color=colours[i % len(colours)],
                       zorder=5, label=r["name"] if xkey == "g" else None)
            ax.scatter([x], [t[ckey]], s=110, facecolors="none",
                       edgecolors=colours[i % len(colours)], linewidths=1.6, zorder=5)
            ax.annotate(f"{t['sequential_s']:.1f}s", (x, t["sequential_s"]), fontsize=7,
                        xytext=(5, 4), textcoords="offset points")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(f"vs {xlabel}", fontsize=10)
        ax.grid(True, which="both", alpha=.25, lw=.5)
    axes[0].set_ylabel("latency (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles += [Line2D([], [], marker="o", ls="", color="#555", label="total render (filled)"),
                Line2D([], [], marker="o", ls="", mfc="none", color="#555",
                       label="that component only (hollow)")]
    axes[0].legend(handles=handles, fontsize=7, loc="upper left")
    fig.suptitle("Shipped graphs: total Explore-render latency vs each dimension "
                 "(shared y axis; log-log)", fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "real_graph_scaling.png", dpi=140)
    plt.close(fig)


def analyze(results: dict, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    syn = results["synthetic"]
    if not syn:
        export_tables(results, outdir)
        analyze_real_graphs(results, outdir)
        print("  real-only results: wrote tables and the shipped-graph figure")
        return
    labels = sorted({l for r in syn for l in r.get("endpoints", {})})
    gen = results.get("config", {}).get("generator", "graphcast")
    # The random generator sweeps all three dimensions independently; GraphCast leaves only two
    # free knobs, and reports `e` as an outcome of them rather than something sweepable.
    var_axis = ({"g": ("g", "grid nodes"), "m": ("m", "mesh nodes"), "e": ("e", "total edges")}
                if gen == "random"
                else {"grid": ("g", "grid nodes"), "mesh": ("m", "mesh nodes")})
    def SW(r):  # noqa: N802 - sweep tags, named per generator
        return r.get("sweeps") or r.get("ladders") or []

    # -- per-sweep power-law exponents -------------------------------------------------
    power: dict[str, dict] = {}
    for var, (key, _) in var_axis.items():
        for label in labels:
            pts = [(r[key], r["endpoints"][label]["min_s"])
                   for r in syn if var in SW(r) and label in r["endpoints"]]
            pts.sort()
            fit = _power_fit([p[0] for p in pts], [p[1] for p in pts])
            if fit:
                power.setdefault(label, {})[var] = fit

    # Cold-path exponents against e, where the O(e log e) sort lives.
    cold = {}
    for field in ("meta_cold_s", "graph_metrics_cold_s"):
        pts = [(r["e"], r[field]) for r in syn if r.get(field)]
        pts.sort()
        f = _power_fit([p[0] for p in pts], [p[1] for p in pts])
        if f:
            cold[field] = f

    additive = {label: _additive_fit(syn, label) for label in labels}
    additive = {k: v for k, v in additive.items() if v}

    # -- the structural law e = a*g + b*m, measured over the generated ladder --------
    # Only meaningful for graphcast: with the random generator e is chosen independently, so
    # there is no law to recover and fitting one would report noise as structure.
    import numpy as np
    from scipy.optimize import nnls
    A = np.array([[r["g"], r["m"]] for r in syn], float)  # noqa: N806
    coef, _ = nnls(A, np.array([r["e"] for r in syn], float))
    pred_e = A @ coef
    obs_e = np.array([r["e"] for r in syn], float)
    structural = None if gen == "random" else {
        "edges_per_grid_node": float(coef[0]),
        "edges_per_mesh_node": float(coef[1]),
        "r2": float(1 - ((obs_e - pred_e) ** 2).sum() / ((obs_e - obs_e.mean()) ** 2).sum()),
        "real": [{"name": r["name"], "g": r["g"], "m": r["m"], "e": r["e"],
                  "e_over_g": r["e"] / r["g"]} for r in results.get("real", [])],
    }

    # -- latency vs the payload each endpoint actually serializes ---------------------
    # Pooled over generated AND shipped graphs. If an endpoint is linear in its own payload
    # this collapses to a single line, which is a stronger statement than any per-ladder
    # exponent: it says the graph the payload came from does not matter.
    pooled = {}
    for label in labels:
        pts = []
        for r in syn + results.get("real", []):
            if label not in r["endpoints"]:
                continue
            n = r["endpoints"][label].get("element_count", 0)
            if n:
                pts.append((n, r["endpoints"][label]["min_s"]))
        fit = _power_fit([x for x, _ in pts], [y for _, y in pts])
        if fit:
            fit["ns_per_element"] = 1e9 * float(np.median([y / x for x, y in pts]))
            pooled[label] = fit

    # -- Figure 1: one panel per variable ----------------------------------------------
    fig, axes = plt.subplots(1, len(var_axis), figsize=(5.8 * len(var_axis), 5))
    for ax, (var, (key, xlabel)) in zip(axes, var_axis.items()):
        for label in labels:
            pts = sorted((r[key], r["endpoints"][label]["min_s"])
                         for r in syn if var in SW(r) and label in r["endpoints"])
            if len(pts) < 2:
                continue
            k = power.get(label, {}).get(var, {}).get("exponent")
            ax.plot([p[0] for p in pts], [p[1] * 1e3 for p in pts], "o-", ms=4,
                    label=f"{label}  (k={k:.2f})" if k is not None else label)
        ax.set_xscale("log"); ax.set_yscale("log")
        # Explicit ticks at the swept values. Matplotlib's default log locator labels minor
        # ticks too, which overlap illegibly on a 6-point decade-and-a-half sweep.
        ticks = sorted({r[key] for r in syn if var in SW(r)})
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t/1000:g}k" if t < 1e6 else f"{t/1e6:g}M" for t in ticks],
                           fontsize=8)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(xlabel); ax.set_ylabel("latency, min of reps (ms)")
        if gen == "random":
            held = ", ".join(f"{v}={ {'g': BASE_G, 'm': BASE_M, 'e': BASE_E}[v] :,}"
                             for v in ("g", "m", "e") if v != var)
            ax.set_title(f"vs {xlabel}  ({held} fixed)", fontsize=9)
        else:
            held = (f"mesh fixed at splits={BASE_SPLITS}" if var == "grid"
                    else f"grid fixed at {BASE_RES_DEG}\u00b0")
            ax.set_title(f"vs {xlabel}  ({held};\ne follows from the construction)", fontsize=9)
        ax.grid(True, which="both", alpha=.25, lw=.5)
        ax.legend(fontsize=7)
    fig.suptitle("Endpoint latency along each GraphCast ladder (log-log; slope k = exponent)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "scaling_by_dimension.png", dpi=140)
    plt.close(fig)

    # -- Figure 2: where the time goes, per element ------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    import numpy as np
    ordered = sorted(additive)
    x = np.arange(len(ordered)); w = 0.27
    for off, key, name in ((-w, "ns_per_grid_node", "per grid node"),
                           (0.0, "ns_per_mesh_node", "per mesh node"),
                           (w, "ns_per_subgraph_edge", "per edge (own subgraph)")):
        ax.bar(x + off, [max(additive[l][key], 1e-3) for l in ordered], w, label=name)
    ax.set_xticks(x); ax.set_xticklabels(ordered, rotation=20, ha="right", fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel("nanoseconds per element")
    ax.set_title("Additive model  t = a·g + b·m + c·e + d   (fitted coefficients)")
    ax.grid(True, axis="y", which="both", alpha=.25, lw=.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "cost_per_element.png", dpi=140)
    plt.close(fig)

    # -- Figure 3: predict the real graphs from the synthetic fit -----------------------
    validation = []
    if results.get("real"):
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        lo = hi = None
        palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        markers = ["o", "s", "^", "D", "v", "P"]
        for gi, r in enumerate(results["real"]):
            colour = palette[gi % len(palette)]
            first = True
            for label, ep in r["endpoints"].items():
                fit = additive.get(label)
                if not fit:
                    continue
                pred = _predict(fit, r["g"], r["m"], ep.get("subgraph_edges", 0))
                meas = ep["min_s"]
                validation.append({"graph": r["name"], "endpoint": label,
                                   "predicted_s": pred, "measured_s": meas,
                                   "ratio": meas / pred if pred > 0 else float("nan")})
                ax.scatter(pred * 1e3, meas * 1e3, s=48, alpha=.85, color=colour,
                           marker=markers[sorted(additive).index(label) % len(markers)],
                           edgecolors="white", linewidths=.6,
                           label=r["name"] if first else None)
                first = False
                vals = [v for v in (pred * 1e3, meas * 1e3) if v > 0]
                lo = min([lo] + vals) if lo else min(vals)
                hi = max([hi] + vals) if hi else max(vals)
        # Marker shape encodes the endpoint, colour the graph -- two legends, or the reader
        # cannot tell which point is which.
        from matplotlib.lines import Line2D
        shape_legend = [Line2D([], [], marker=markers[i % len(markers)], ls="", color="#555",
                               ms=7, label=l) for i, l in enumerate(sorted(additive))]
        ax.add_artist(ax.legend(handles=shape_legend, fontsize=7, loc="lower right",
                                title="endpoint", title_fontsize=7))
        if lo and hi:
            ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect prediction")
            for f, s in ((2, ":"), (0.5, ":")):
                ax.plot([lo, hi], [lo * f, hi * f], "k", ls=s, lw=.6, alpha=.5)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("predicted latency from synthetic fit (ms)")
        ax.set_ylabel("measured latency on real graph (ms)")
        ax.set_title("Held-out validation: real graphs vs the model\n(dotted lines = ±2x)",
                     fontsize=10)
        ax.grid(True, which="both", alpha=.25, lw=.5)
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        fig.savefig(outdir / "validation_real_graphs.png", dpi=140)
        plt.close(fig)

    export_tables(results, outdir)
    analyze_real_graphs(results, outdir)
    if not syn:
        # A real-only run (--only-real): the tables and the shipped-graph figure are the whole
        # output. Everything below needs the decoupled sweeps to mean anything.
        print("  no synthetic points in this results file - tables and real-graph figure only")
        return
    totals = analyze_totals(results, outdir) if any(r.get("total") for r in syn) else None
    if not labels:
        print("  --totals-only run: skipping the per-endpoint figures")
        results["analysis"] = {"totals": totals, "power_law": {}, "additive": {},
                               "cold": {}, "validation": [], "structural": None, "pooled": {}}
        _write_report(results, outdir)
        return
    results["analysis"] = {"totals": totals,
                           "power_law": power, "additive": additive, "cold": cold,
                           "validation": validation, "structural": structural,
                           "pooled": pooled}
    _write_report(results, outdir)


def _write_report(results: dict, outdir: Path) -> None:
    a = results["analysis"]
    L: list[str] = []
    L.append("# Latency scaling with graph size\n")
    L.append(f"Generated by `scripts/benchmark_graph_scaling.py`. "
             f"{results['config']['reps']} timed reps per point, full-world bbox, gzip "
             f"enabled, requests sequential.\n")
    L.append("Fits use the **minimum** of the reps, not the median. This runs on a shared "
             "machine where transient load adds occasional ~1.8x spikes on top of a stable "
             "floor (measured: two graphs sharing an identical g2m file read 553 and 554 ms "
             "at the floor, with excursions to 987 ms). The minimum estimates the uncontended "
             "cost; drift across the run was checked at 0.82-1.00x and is not correlated with "
             "graph size, so the small-to-large ordering does not bias the exponents.\n")
    L.append(f"Graphs built with graphcast's own icosahedral mesh and grid/mesh connectivity "
             f"(the modules neural-lam's `create_global_mesh.py` uses). Grid ladder at "
             f"splits={BASE_SPLITS}; mesh ladder at {BASE_RES_DEG}\u00b0.\n")

    if a.get("totals"):
        tt = a["totals"]
        L.append("\n## Total Explore-render latency \u2014 the derivative\n")
        L.append("Sum over every node type and every subgraph, i.e. one full Explore render. "
                 "One dimension moves per panel, the other two pinned. Linear fit; the slope "
                 "is the answer to \"how much longer per extra element\".\n")
        L.append("| dimension | \u00b5s per element (sequential) | s per 1,000 added | R\u00b2 | "
                 "\u00b5s per element (parallel) | R\u00b2 |")
        L.append("|---|---|---|---|---|---|")
        for sw, name in (("g", "grid node"), ("m", "mesh node"), ("e", "edge")):
            f = tt["fits"].get(sw, {})
            fs, fp = f.get("sequential_s"), f.get("parallel_s")
            if not fs:
                continue
            L.append(f"| per **{name}** | **{fs['slope_s_per_element']*1e6:.2f}** | "
                     f"**{fs['slope_s_per_element']*1000:.3f} s** | {fs['r2']:.4f} | "
                     f"{fp['slope_s_per_element']*1e6:.2f} | {fp['r2']:.4f} |")
        m = tt["model_sequential"]
        L.append(f"\nCombined: **T \u2248 {m['s_per_grid_node']*1e6:.2f} \u00b5s\u00b7g + "
                 f"{m['s_per_mesh_node']*1e6:.2f} \u00b5s\u00b7m + "
                 f"{m['s_per_edge']*1e6:.2f} \u00b5s\u00b7e** (sequential total).\n")
        if tt["overlay"]:
            L.append("\n### Overlay: real graphs\n")
            L.append("| graph | g | m | e | sequential | parallel | predicted seq | ratio |")
            L.append("|---|---|---|---|---|---|---|---|")
            for o in sorted(tt["overlay"], key=lambda o: o["e"]):
                ratio = o["sequential_s"] / o["predicted_sequential_s"] if o["predicted_sequential_s"] else float("nan")
                L.append(f"| {o['name']} | {o['g']:,} | {o['m']:,} | {o['e']:,} | "
                         f"{o['sequential_s']:.2f} s | {o['parallel_s']:.2f} s | "
                         f"{o['predicted_sequential_s']:.2f} s | {ratio:.2f}\u00d7 |")

    st = a.get("structural")
    if st:
        L.append("\n## The structural law: g, m and e are not independent\n")
        L.append("Fitted over the generated ladder, `e = a\u00b7g + b\u00b7m`:\n")
        L.append(f"- **{st['edges_per_grid_node']:.2f} edges per grid node**")
        L.append(f"- **{st['edges_per_mesh_node']:.2f} edges per mesh node**")
        L.append(f"- R\u00b2 = {st['r2']:.4f}\n")
        L.append("Since g \u226b m in every practical graph, **e \u2248 "
                 f"{st['edges_per_grid_node']:.1f}\u00b7g**. The shipped graphs agree:\n")
        L.append("| graph | g | m | e | e/g |")
        L.append("|---|---|---|---|---|")
        for r in st["real"]:
            L.append(f"| {r['name']} | {r['g']:,} | {r['m']:,} | {r['e']:,} | {r['e_over_g']:.2f} |")
        L.append("\nSo \"hold g fixed and raise e\" is not a graph GraphCast can build; the "
                 "exponents below are along the manifold real graphs occupy.\n")
    else:
        L.append("\n## Note on the generator\n")
        L.append("Edges were assigned at random, so g, m and e are independent by "
                 "construction and there is no structural law to recover. That is the point: "
                 "it is the only way to hold two of them fixed while the third moves. The "
                 "real graphs overlaid on the plots are where the coupling reasserts itself "
                 "\u2014 in a GraphCast-built graph e is pinned at roughly 4.6\u00b7g.\n")

    L.append("\n## Latency vs the payload each endpoint serializes\n")
    L.append("Pooled over the generated ladder **and** the shipped graphs. `k` \u2248 1 with a "
             "tight fit means latency depends only on how many elements come back, not on "
             "which graph they came from.\n")
    L.append("| endpoint | elements counted | k | R\u00b2 | n | \u00b5s per element |")
    L.append("|---|---|---|---|---|---|")
    _what = {"nodes/grid": "grid nodes", "nodes/mesh": "mesh nodes (level 0)"}
    for label in sorted(a["pooled"]):
        f = a["pooled"][label]
        L.append(f"| {label} | {_what.get(label, 'edges in that subgraph')} | "
                 f"**{f['exponent']:.2f}** | {f['r2']:.3f} | {f['n']} | "
                 f"{f['ns_per_element']/1000:.2f} |")

    L.append("\n## Measured exponents — warm path\n")
    L.append("`t ~ x^k` along each ladder, R\u00b2 in parentheses. Along the grid ladder e "
             "rises with g by construction, so an edge endpoint's `k wrt g` reflects that "
             "coupling rather than an independent dependence on g \u2014 which is exactly the "
             "practical question: what happens when someone swaps in a finer grid.\n")
    _sweeps = sorted({v for lab in a["power_law"].values() for v in lab})
    L.append("| endpoint | " + " | ".join(f"k wrt {v}" for v in _sweeps) + " |")
    L.append("|---" * (1 + len(_sweeps)) + "|")
    for label in sorted(a["power_law"]):
        row = [label]
        for var in _sweeps:
            f = a["power_law"][label].get(var)
            row.append(f"**{f['exponent']:.2f}** ({f['r2']:.2f})" if f else "\u2013")
        L.append("| " + " | ".join(row) + " |")

    if a["cold"]:
        L.append("\n## Cold path (first touch of a graph / subgraph), wrt e\n")
        L.append("| stage | k wrt e | R² |")
        L.append("|---|---|---|")
        names = {"meta_cold_s": "`/meta` — torch.load + registry build",
                 "graph_metrics_cold_s": "`/graph_metrics` — haversine + sort"}
        for k, f in a["cold"].items():
            L.append(f"| {names.get(k, k)} | **{f['exponent']:.2f}** | {f['r2']:.2f} |")

    L.append("\n## Additive model\n")
    L.append("`t = a·g + b·m + c·e + d`, non-negative least squares over every synthetic "
             "point. This is the model the code actually implements; the per-element costs "
             "say which term you pay for.\n")
    L.append("Fitted against the *requested subgraph's* edge count, not the graph total, so "
             "the coefficients transfer to graphs with a different subgraph mix.\n")
    L.append("| endpoint | ns/grid node | ns/mesh node | ns/edge (that subgraph) | const (ms) | R² |")
    L.append("|---|---|---|---|---|---|")
    for label in sorted(a["additive"]):
        f = a["additive"][label]
        L.append(f"| {label} | {f['ns_per_grid_node']:.1f} | {f['ns_per_mesh_node']:.1f} | "
                 f"{f['ns_per_subgraph_edge']:.1f} | {f['const_ms']:.1f} | {f['r2']:.3f} |")

    if a["validation"]:
        L.append("\n## Held-out validation on the real graphs\n")
        L.append("The fit never saw these. `ratio` = measured / predicted.\n")
        L.append("| graph | endpoint | predicted | measured | ratio |")
        L.append("|---|---|---|---|---|")
        for v in sorted(a["validation"], key=lambda v: (v["graph"], v["endpoint"])):
            L.append(f"| {v['graph']} | {v['endpoint']} | {v['predicted_s']*1e3:.0f} ms | "
                     f"{v['measured_s']*1e3:.0f} ms | {v['ratio']:.2f}× |")

    L.append("\n## Raw measurements\n")
    L.append("| graph | g | m | e | " + " | ".join(
        sorted({l for r in results["synthetic"] for l in r["endpoints"]})) + " |")
    labels = sorted({l for r in results["synthetic"] for l in r["endpoints"]})
    L.append("|---" * (4 + len(labels)) + "|")
    for r in results["synthetic"] + results.get("real", []):
        cells = [r["name"], f"{r['g']:,}", f"{r['m']:,}", f"{r['e']:,}"]
        for label in labels:
            ep = r["endpoints"].get(label)
            cells.append(f"{ep['median_s']*1e3:.0f} ms" if ep else "–")
        L.append("| " + " | ".join(cells) + " |")

    (outdir / "scaling_report.md").write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# Phase: cleanup
# ---------------------------------------------------------------------------

def cleanup() -> None:
    """Removes every synthetic graph from the DB and disk. Safe to run repeatedly."""
    sys.path.insert(0, "/usr/src/app")
    from services.database import SessionLocal
    from models.models import Graph

    db = SessionLocal()
    n = 0
    try:
        for row in db.query(Graph).all():
            if Path(row.folder_path).name.startswith(BENCH_PREFIX):
                db.delete(row)
                n += 1
        db.commit()
    finally:
        db.close()
    d = 0
    for folder in GRAPHS_DIR.glob(f"{BENCH_PREFIX}*"):
        shutil.rmtree(folder, ignore_errors=True)
        d += 1
    print(f"  removed {n} DB rows, {d} folders")
    print("  NOTE: restart the backend to release the synthetic graphs from _raw_cache")


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["generate", "bench", "zoom", "analyze",
                                      "cleanup", "all"])
    ap.add_argument("--out", type=Path, default=Path("/usr/data/bench_scaling"))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--no-real", action="store_true", help="skip the real-graph validation points")
    ap.add_argument("--api", default=None, help="backend base URL to benchmark")
    ap.add_argument("--totals-only", action="store_true",
                    help="skip the per-endpoint sampling; measure only the whole-render "
                         "totals and their per-component breakdown")
    ap.add_argument("--only-real", action="store_true",
                    help="benchmark only the real graphs (for collecting per-component data "
                         "without rebuilding the synthetic sweep)")
    ap.add_argument("--generator", choices=["random", "graphcast"], default="graphcast",
                    help="random decouples g/m/e fully; graphcast builds realizable graphs "
                         "but forces e = 4.56g + 8m")
    ap.add_argument("--gc-cache", type=Path, default=Path("/tmp/graphcast_mesh"),
                    help="where to cache the downloaded graphcast mesh modules")
    ap.add_argument("--quick", action="store_true",
                    help="3 points per ladder and 2 reps, for checking the harness works")
    args = ap.parse_args()

    global API, GRID_LADDER_DEG, MESH_LADDER_SPLITS, G_SWEEP, M_SWEEP, E_SWEEP
    if args.api:
        API = args.api
    if args.quick:
        GRID_LADDER_DEG, MESH_LADDER_SPLITS = GRID_LADDER_DEG[:3], MESH_LADDER_SPLITS[:3]
        G_SWEEP, M_SWEEP, E_SWEEP = G_SWEEP[:3], M_SWEEP[:3], E_SWEEP[:3]
        args.reps = min(args.reps, 2)

    args.out.mkdir(parents=True, exist_ok=True)
    points = (random_sweep_points() if args.generator == "random"
              else graphcast_sweep_points())
    results_path = args.out / "results.json"

    if args.phase in ("generate", "all"):
        print(f"[generate] {len(points)} {args.generator} graphs -> {GRAPHS_DIR}")
        generate(points, args.gc_cache, args.generator)
        (args.out / "points.json").write_text(json.dumps(points, indent=2))

    if args.phase in ("bench", "all"):
        if args.phase == "bench":
            saved = json.loads((args.out / "points.json").read_text())
            points = saved
        print(f"[bench] reps={args.reps} warmup={args.warmup}")
        results = bench(points, args.reps, args.warmup, include_real=not args.no_real,
                        skip_synthetic=args.only_real, totals_only=args.totals_only)
        results["config"]["generator"] = args.generator
        results_path.write_text(json.dumps(results, indent=2))
        print(f"[bench] wrote {results_path}")

    if args.phase == "zoom":
        print(f"[zoom] levels={ZOOM_LEVELS} centre={ZOOM_CENTRE} viewport={VIEWPORT_PX}")
        z = bench_zoom(args.reps, args.warmup)
        (args.out / "results_zoom.json").write_text(json.dumps(z, indent=2))
        import csv as _csv
        with (args.out / "zoom.csv").open("w", newline="") as fh:
            cols = ["graph", "zoom", "g", "m", "e", "lat_span", "lon_span",
                    "elements_returned", "sequential_s", "parallel_s", "bytes_wire"]
            w = _csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(z["rows"])
        print(f"[zoom] wrote {args.out}/results_zoom.json and zoom.csv ({len(z['rows'])} rows)")

    if args.phase in ("analyze", "all"):
        results = json.loads(results_path.read_text())
        print("[analyze] fitting and plotting")
        analyze(results, args.out)
        results_path.write_text(json.dumps(results, indent=2))
        print(f"[analyze] wrote {args.out}/scaling_report.md and the figures")

    if args.phase in ("cleanup", "all"):
        print("[cleanup]")
        cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
