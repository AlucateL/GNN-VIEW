"""
Converter for neural-lam ocean graphs (Njord / baltic-sea-forecasting) to the graph format
this backend serves.

Sibling of `anemoi_converter.py`, and called from the same place in `routes/graph.py`.

Those graph directories ship edge indices and feature tables but NOT the two position files
the frontend needs to place nodes on a map. Both are reconstructed from the .pt files alone:

  mesh_lat_lon.pt   mesh_features.pt carries node positions in its leading columns.
  grid_lat_lon.pt   grid position = mesh position -/+ the edge `vdiff`, since every g2m/m2g
                    edge stores vdiff = (receiver - sender).

  ! Use g2m, not just m2g. On baltic_cluster, m2g reaches only 147,087 of 147,701 grid
    nodes -- 614 are input-only (the boundary forcing region). Recovering from m2g alone
    drops them silently and shifts every subsequent index.

TWO mesh_up/mesh_down INDEX CONVENTIONS, auto-detected per hierarchy step:

  level-local  each row already indexes its own level (create_graph.py's hierarchical path,
               e.g. global_hierarchical_analysis).
  combined     each step is written in the joint space of the pair it connects,
               [level i | level i+1], so the coarse row is offset by the finer level's node
               count. Emitted by the K-means cluster path -- global_cluster_analysis
               (offsets 33777, 8409) and baltic_cluster (16358, 1767).

  Only the first was represented among the graphs converted before, so the position-recovery
  checks below never exercised the second. See localize_hierarchy.

TWO FLAVOURS, auto-detected from the width of mesh_features:

  3 columns -> (x, y, voronoi_area) in projected metres, limited-area. Edge features are
               [len, vdiff_x, vdiff_y]. NEEDS a projection to reach lat/lon.
  5 columns -> (sin_lon, cos_lon, sin_lat, cos_lat, area), global. Edge features are
               [len, vdiff_x, vdiff_y, vdiff_z] on the unit sphere. Self-contained.

Verified on baltic_cluster (147,701 grid nodes, 4e-4 m agreement between independent
per-edge estimates) and on the two global analysis graphs (675,219 nodes each, land/sea
distribution matching Earth's ~71% ocean, and byte-identical grids after snapping).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

_PASSTHROUGH = (
    "g2m_features.pt",
    "m2g_features.pt",
    "m2m_features.pt",
    "mesh_up_features.pt",
    "mesh_down_features.pt",
)

PROJECTED, SPHERICAL = "projected2d", "spherical3d"

_HIERARCHY_FILES = ("mesh_up_edge_index.pt", "mesh_down_edge_index.pt")


def is_bsf_format(saved_files: set[str]) -> bool:
    """
    True for an ocean graph directory that still needs converting.

    The signature is mesh_features.pt present while the two position files are absent --
    that is exactly what create_graph.py / create_global_graph.py emit. A directory that
    already has grid_lat_lon.pt is a ready-made neural-lam graph and must not be touched.
    """
    return (
        "mesh_features.pt" in saved_files
        and "grid_lat_lon.pt" not in saved_files
        and "mesh_lat_lon.pt" not in saved_files
    )


def _load(graph_dir: Path, name: str):
    path = graph_dir / name
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=True)


def detect_flavour(mesh_features: list[torch.Tensor]) -> str:
    """Classify a graph from the width of its mesh feature table."""
    width = mesh_features[0].shape[1]
    if width == 3:
        return PROJECTED
    if width == 5:
        return SPHERICAL
    raise ValueError(
        f"mesh_features has {width} columns; expected 3 (x, y, area) or "
        "5 (sin_lon, cos_lon, sin_lat, cos_lat, area)."
    )


def mesh_positions(mesh_features: list[torch.Tensor], flavour: str) -> list[torch.Tensor]:
    """
    Per-level mesh node positions, in the graph's native space.

    PROJECTED -> [N, 2] (x, y) metres, straight out of columns 0-1.
    SPHERICAL -> [N, 3] unit-sphere Cartesian, rebuilt from the sin/cos columns so it shares
                 a space with the 3D vdiff on the edges.
    """
    out = []
    for feats in mesh_features:
        f = feats.to(torch.float64)
        if flavour == PROJECTED:
            out.append(f[:, :2].clone())
        else:
            lon = torch.atan2(f[:, 0], f[:, 1])
            lat = torch.atan2(f[:, 2], f[:, 3])
            out.append(
                torch.stack(
                    [torch.cos(lat) * torch.cos(lon), torch.cos(lat) * torch.sin(lon), torch.sin(lat)],
                    dim=1,
                )
            )
    return out


def _grid_row(edge_index: torch.Tensor, num_mesh: int, name: str) -> int:
    """Which row of a combined-space edge index holds the (offset) grid nodes."""
    rows = [bool(int(edge_index[r].min()) >= num_mesh) for r in (0, 1)]
    if rows[0] == rows[1]:
        raise ValueError(
            f"{name}: cannot identify the grid row -- row minima are "
            f"({int(edge_index[0].min())}, {int(edge_index[1].min())}) against "
            f"num_mesh={num_mesh}. Edge index may already be localized."
        )
    return 0 if rows[0] else 1


def _localize_hierarchy_row(
    row: torch.Tensor, level_count: int, offset: int, name: str
) -> tuple[torch.Tensor, bool]:
    """
    Bring one row of a mesh_up/mesh_down edge index into its own level's index space.

    Upstream builders disagree here. create_graph.py's hierarchical path writes indices that
    are already level-local, but the K-means cluster path (Njord's global_cluster_analysis,
    baltic_cluster) writes them in the COMBINED two-level space of the pair it connects:
    [level i | level i+1], so the coarse row is offset by the finer level's node count.

    The offset is a known quantity, so it is neither guessed nor read off the data. Taking
    the row minimum instead -- the obvious shortcut -- silently shifts every index whenever
    the lowest-numbered node of a level happens to have no edge in that direction.

    Returns (row, was_rebased). Raises when the row fits neither convention, rather than
    letting a wrong index reach the frontend as an out-of-bounds lookup.

    An empty row carries no evidence of either convention and nothing to rebase, so it is
    returned untouched. It is not a hypothetical: a mesh level whose edges are all cut by
    --max_chord_len leaves the step with zero edges, and min()/max() on an empty tensor
    raise, which would fail the whole conversion over a subgraph that is simply empty.
    """
    if row.numel() == 0:
        return row, False
    lo, hi = int(row.min()), int(row.max())
    if 0 <= lo and hi < level_count:
        return row, False
    if offset and lo >= offset and hi < offset + level_count:
        return row - offset, True
    raise ValueError(
        f"{name}: indices span [{lo}, {hi}], which is neither local to the level "
        f"({level_count} nodes) nor that level offset by {offset} in the combined "
        "two-level space. Unrecognised mesh_up/mesh_down index convention."
    )


def localize_hierarchy(
    edges_per_step: list[torch.Tensor], counts: list[int], name: str
) -> tuple[list[torch.Tensor], list[str]]:
    """
    Localize every step of a mesh_up or mesh_down edge index list.

    `counts` is the mesh node count per level. Step i connects levels i and i+1; the row
    holding the coarse (i+1) side is the one that may carry the offset, which is counts[i].
    The fine row is checked too, so a graph in some third convention fails loudly here
    instead of at render time.

    Returns (localized, rebased_step_names) -- the second element names the steps that were
    actually rebased, for the conversion summary.
    """
    coarse_row = 1 if name.startswith("mesh_up") else 0  # up: i -> i+1, down: i+1 -> i
    out, rebased = [], []

    for i, edges in enumerate(edges_per_step):
        edges = edges.to(torch.int64).clone()
        step = f"{name.removesuffix('_edge_index.pt')}_{i}"
        edges[coarse_row], moved = _localize_hierarchy_row(
            edges[coarse_row], counts[i + 1], counts[i], f"{step} (level {i + 1} row)"
        )
        edges[1 - coarse_row], _ = _localize_hierarchy_row(
            edges[1 - coarse_row], counts[i], 0, f"{step} (level {i} row)"
        )
        if moved:
            rebased.append(step)
        out.append(edges)

    return out, rebased


def recover_grid_positions(
    graph_dir: Path, mesh_pos_level0: torch.Tensor, num_mesh: int
) -> tuple[torch.Tensor, dict]:
    """
    Rebuild grid node positions from the g2m/m2g edge geometry.

    Every g2m/m2g edge stores vdiff = receiver - sender in the same space as the mesh
    positions, so each edge yields one estimate of its grid endpoint. Estimates are averaged
    per node; the spread across independent estimates is returned as a correctness check.

    Returns (positions [N_grid, D], diagnostics).
    """
    dst_all, est_all = [], []

    for name in ("g2m_edge_index.pt", "m2g_edge_index.pt"):
        edges = _load(graph_dir, name)
        feats = _load(graph_dir, name.replace("_edge_index", "_features"))
        if edges is None or feats is None:
            continue
        if isinstance(edges, list):
            edges, feats = edges[0], feats[0]
        edges = edges.to(torch.int64)
        vdiff = feats.to(torch.float64)[:, 1:]  # column 0 is edge length

        grid_row = _grid_row(edges, num_mesh, name)
        mesh_row = 1 - grid_row
        grid_idx = edges[grid_row] - num_mesh
        mesh_pos = mesh_pos_level0[edges[mesh_row]]

        # vdiff points receiver <- sender: add it when the grid node receives (m2g),
        # subtract it when the grid node sends (g2m).
        est = mesh_pos + vdiff if grid_row == 1 else mesh_pos - vdiff

        dst_all.append(grid_idx)
        est_all.append(est)

    if not dst_all:
        raise FileNotFoundError(
            f"{graph_dir}: need g2m/m2g edge_index + features to reconstruct grid positions."
        )

    dst = torch.cat(dst_all)
    est = torch.cat(est_all)
    num_grid = int(dst.max()) + 1
    dim = est.shape[1]

    total = torch.zeros(num_grid, dim, dtype=torch.float64)
    count = torch.zeros(num_grid, dtype=torch.float64)
    total.index_add_(0, dst, est)
    count.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float64))

    uncovered = int((count == 0).sum())
    if uncovered:
        raise ValueError(
            f"{graph_dir}: {uncovered} of {num_grid} grid nodes have no g2m or m2g edge, so "
            "their positions cannot be recovered. The graph is not fully connected."
        )

    positions = total / count.unsqueeze(1)
    spread = (est - positions[dst]).norm(dim=1)
    return positions, {
        "grid_nodes": num_grid,
        "max_estimate_spread": float(spread.max()),
        "mean_estimate_spread": float(spread.mean()),
    }


# WGS84
_A = 6378137.0
_E = math.sqrt(2 * (1 / 298.257223563) - (1 / 298.257223563) ** 2)


def lcc_inverse(
    x: np.ndarray, y: np.ndarray, lat_0: float, lon_0: float, lat_1: float, lat_2: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inverse Lambert Conformal Conic on the WGS84 ellipsoid (Snyder, eqs. 14-15 / 15-9).

    Implemented directly rather than via pyproj so this script has no geospatial
    dependencies -- neither pyproj nor cartopy is guaranteed to be installed.

    lat_1 == lat_2 gives the tangent (single standard parallel) case.
    Returns (lon_degrees, lat_degrees).
    """
    m = lambda p: math.cos(p) / math.sqrt(1 - _E**2 * math.sin(p) ** 2)  # noqa: E731
    t = lambda p: math.tan(math.pi / 4 - p / 2) / (  # noqa: E731
        (1 - _E * math.sin(p)) / (1 + _E * math.sin(p))
    ) ** (_E / 2)

    p0, l0, p1, p2 = map(math.radians, (lat_0, lon_0, lat_1, lat_2))
    if abs(p1 - p2) < 1e-12:
        n = math.sin(p1)
    else:
        n = (math.log(m(p1)) - math.log(m(p2))) / (math.log(t(p1)) - math.log(t(p2)))
    big_f = m(p1) / (n * t(p1) ** n)
    rho_0 = _A * big_f * t(p0) ** n

    sign = np.sign(n)
    rho = sign * np.sqrt(x**2 + (rho_0 - y) ** 2)
    theta = np.arctan2(sign * x, sign * (rho_0 - y))
    lon = theta / n + l0

    t_prime = (rho / (_A * big_f)) ** (1 / n)
    lat = math.pi / 2 - 2 * np.arctan(t_prime)
    for _ in range(12):  # Snyder 3-5, converges in a handful of passes
        lat = math.pi / 2 - 2 * np.arctan(
            t_prime * ((1 - _E * np.sin(lat)) / (1 + _E * np.sin(lat))) ** (_E / 2)
        )
    return np.degrees(lon), np.degrees(lat)


def to_lat_lon(
    positions: torch.Tensor, flavour: str, projection: dict | None
) -> torch.Tensor:
    """Native-space positions -> [N, 2] float32 (lat, lon) degrees, lon in [-180, 180]."""
    if flavour == SPHERICAL:
        p = positions / positions.norm(dim=1, keepdim=True).clamp(min=1e-12)
        lat = torch.rad2deg(torch.asin(p[:, 2].clamp(-1, 1)))
        lon = torch.rad2deg(torch.atan2(p[:, 1], p[:, 0]))
    else:
        if projection is None:
            raise ValueError(
                "This is a limited-area graph: its coordinates are in projected metres, so "
                "the Lambert Conformal Conic parameters are needed to turn them into "
                "lat/lon. Supply them as 'lat_0,lon_0[,lat_1,lat_2]' (the `lcc` field on "
                "upload, or --lcc on the command line). They come from the `extra.projection` "
                "block of the datastore config the graph was built with. Global ocean graphs "
                "need nothing here."
            )
        lon_np, lat_np = lcc_inverse(
            positions[:, 0].numpy(), positions[:, 1].numpy(), **projection
        )
        lon = torch.as_tensor(lon_np, dtype=torch.float64)
        lat = torch.as_tensor(lat_np, dtype=torch.float64)

    lon = (lon + 180.0) % 360.0 - 180.0
    return torch.stack([lat, lon], dim=1).to(torch.float32)


def infer_lattice_step(lat_lon: torch.Tensor) -> float | None:
    """
    Infer the source grid's lattice step from the reconstructed latitudes.

    A regular lon/lat datastore places every point on whole multiples of one step, so the
    smallest gap between distinct latitudes is that step. Latitudes are rounded first to
    absorb reconstruction noise (which measured <= 5e-5 deg, far below any real step).

    Returns None when no step can be read off, in which case snapping is skipped.
    """
    lat = torch.unique(torch.round(lat_lon[:, 0].double() * 1000) / 1000)
    if lat.numel() < 3:
        return None
    gaps = torch.diff(lat)
    gaps = gaps[gaps > 1e-6]
    if gaps.numel() == 0:
        return None
    return round(float(gaps.min()), 6)


def snap_to_lattice(lat_lon: torch.Tensor, step: float) -> tuple[torch.Tensor, dict]:
    """
    Snap reconstructed coordinates onto a regular lat/lon lattice of the given step.

    Grid positions are rebuilt by adding a float32 Cartesian `vdiff` to a float32 mesh
    position, then converting back with atan2. Near the poles that round-trip amplifies a
    ~1e-7 Cartesian error into a visible longitude swing (up to ~0.01 deg above 87N, though
    only ~5 m on the ground). When the source datastore is a regular lon/lat grid -- as the
    1/4 degree global ocean one is -- every true coordinate is an exact multiple of `step`,
    so snapping removes the error outright instead of merely bounding it.

    Only valid for regular lat/lon grids. Do NOT use it on limited-area graphs: their grids
    are regular in projected space, which is irregular in lat/lon.

    Raises if the snap is ambiguous, i.e. any node sits near a cell boundary or two nodes
    would land in the same cell.
    """
    snapped = (lat_lon.double() / step).round() * step
    offset = (lat_lon.double() - snapped).abs()
    worst = float(offset.max()) / step

    # Rounding can lift a longitude just below 180 onto exactly +180, which is the same
    # meridian as -180 but outside the normalized range. Re-wrap so the output stays in
    # [-180, 180) and two graphs over the same grid compare equal.
    snapped[:, 1] = (snapped[:, 1] + 180.0) % 360.0 - 180.0

    if worst > 0.4:
        raise ValueError(
            f"snap: a node sits {worst:.0%} of a cell from the nearest lattice point, too "
            f"close to the boundary to snap safely. Is the grid really regular at {step} deg?"
        )

    keys = (
        ((snapped[:, 0] + 90.0) / step).round().long() * int(360 / step + 1)
        + ((snapped[:, 1] + 180.0) / step).round().long()
    )
    unique = len(torch.unique(keys))
    if unique != lat_lon.shape[0]:
        raise ValueError(
            f"snap: {lat_lon.shape[0] - unique} nodes collapsed onto shared lattice cells. "
            "The grid is not a regular lat/lon lattice at this step."
        )

    return snapped.to(torch.float32), {
        "snap_step_deg": step,
        "snap_worst_offset_frac_of_cell": round(worst, 4),
    }


def convert_bsf_to_neural_lam(
    bsf_graph_dir: Path,
    output_dir: Path,
    projection: dict | None = None,
) -> dict:
    """Write a complete neural-lam graph bundle from a BSF graph directory."""
    bsf_graph_dir, output_dir = Path(bsf_graph_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_features = _load(bsf_graph_dir, "mesh_features.pt")
    if mesh_features is None:
        raise FileNotFoundError(f"{bsf_graph_dir}/mesh_features.pt not found.")
    if not isinstance(mesh_features, list):
        mesh_features = [mesh_features]

    flavour = detect_flavour(mesh_features)
    mesh_pos = mesh_positions(mesh_features, flavour)
    num_mesh = mesh_pos[0].shape[0]

    grid_pos, diagnostics = recover_grid_positions(bsf_graph_dir, mesh_pos[0], num_mesh)
    num_grid = grid_pos.shape[0]

    mesh_lat_lon = [to_lat_lon(p, flavour, projection) for p in mesh_pos]
    grid_lat_lon = to_lat_lon(grid_pos, flavour, projection)

    # Grid positions come back through a float32 Cartesian round-trip, which near the poles
    # turns a ~1e-7 error into a visible longitude swing. Global graphs sit on a regular
    # lon/lat lattice, so the exact values can be recovered by snapping to it. Done
    # automatically and only when it is provably safe -- an ambiguous fit is skipped, not
    # forced. Limited-area grids are regular in projected space, not lat/lon, so never snap.
    snap_info = {}
    if flavour == SPHERICAL:
        step = infer_lattice_step(grid_lat_lon)
        if step:
            try:
                grid_lat_lon, snap_info = snap_to_lattice(grid_lat_lon, step)
            except ValueError as e:
                snap_info = {"snap_skipped": str(e)}

    m2m = _load(bsf_graph_dir, "m2m_edge_index.pt")
    if m2m is None:
        raise FileNotFoundError(f"{bsf_graph_dir}/m2m_edge_index.pt not found.")
    if not isinstance(m2m, list):
        m2m = [m2m]

    # g2m/m2g need the grid side rebased to 0. m2m is always level-local; mesh_up/down are
    # only sometimes -- see localize_hierarchy, which handles both conventions.
    localized = {}
    for name in ("g2m_edge_index.pt", "m2g_edge_index.pt"):
        edges = _load(bsf_graph_dir, name)
        if isinstance(edges, list):
            edges = edges[0]
        edges = edges.to(torch.int64).clone()
        edges[_grid_row(edges, num_mesh, name)] -= num_mesh
        localized[name] = edges

    torch.save(grid_lat_lon, output_dir / "grid_lat_lon.pt")
    torch.save(mesh_lat_lon, output_dir / "mesh_lat_lon.pt")
    torch.save([e.to(torch.int64) for e in m2m], output_dir / "m2m_edge_index.pt")
    for name, edges in localized.items():
        torch.save(edges, output_dir / name)

    is_hierarchical = len(mesh_lat_lon) > 1
    rebased: list[str] = []
    if is_hierarchical:
        counts = [int(m.shape[0]) for m in mesh_lat_lon]
        for name in _HIERARCHY_FILES:
            edges = _load(bsf_graph_dir, name)
            if edges is None:
                raise FileNotFoundError(
                    f"{bsf_graph_dir}/{name} missing, but mesh_features has "
                    f"{len(mesh_lat_lon)} levels -- a hierarchical upload requires it."
                )
            if not isinstance(edges, list):
                edges = [edges]
            edges, moved = localize_hierarchy(edges, counts, name)
            rebased.extend(moved)
            torch.save(edges, output_dir / name)

    for name in _PASSTHROUGH:
        feats = _load(bsf_graph_dir, name)
        if feats is not None:
            torch.save(feats, output_dir / name)

    return {
        "output_dir": str(output_dir),
        "flavour": flavour,
        "format": "hierarchical" if is_hierarchical else "single-scale",
        "levels": len(mesh_lat_lon),
        "mesh_nodes_per_level": [int(m.shape[0]) for m in mesh_lat_lon],
        "grid_nodes": num_grid,
        "lat_range": [float(grid_lat_lon[:, 0].min()), float(grid_lat_lon[:, 0].max())],
        "lon_range": [float(grid_lat_lon[:, 1].min()), float(grid_lat_lon[:, 1].max())],
        **diagnostics,
        **snap_info,
        **({"rebased_to_level_local": rebased} if rebased else {}),
    }


