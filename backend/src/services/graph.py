from pathlib import Path

import logging
import torch
import json

from services.crud import GraphCRUD
from services.bsf_converter import localize_hierarchy

logger = logging.getLogger(__name__)

# Graphs base directory — 4 levels up from this file, then data/graphs/
# Local:  <repo>/backend/src/services/graph.py → parents[3] = <repo>
# Docker: /usr/src/app/services/graph.py       → parents[3] = /usr
#         volume mounts ./data at /usr/data (see docker-compose.yml)
_GRAPHS_BASE       = Path(__file__).parents[3] / "data" / "graphs"
_DEFAULT_GRAPH_DIR = _GRAPHS_BASE / "global_hierarchical"


# ---------------------------------------------------------------------------
# Caches (all keyed by resolved folder path)
#
#   _raw_cache        — raw tensors from .pt files, never recomputed
#   _statistics_cache — degree + length tensors derived from raw, never recomputed
#
# Filtered responses are NOT cached: thresholds vary per request, but the
# mask operations on cached tensors are cheap.
# ---------------------------------------------------------------------------

_raw_cache:        dict[str, dict] = {}
_statistics_cache: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Graph folder discovery
# ---------------------------------------------------------------------------

def list_graph_folders() -> list[str]:
    """Returns the names of available graph folders under _GRAPHS_BASE."""
    if not _GRAPHS_BASE.exists():
        return []
    return sorted(p.name for p in _GRAPHS_BASE.iterdir() if p.is_dir())


# ---------------------------------------------------------------------------
# Raw tensor loader
# ---------------------------------------------------------------------------

def grid_spacing_deg(grid_lat_lon: torch.Tensor) -> float:
    """
    Characteristic spacing of the grid, in degrees of latitude.

    The frontend sizes nodes in world units, so a size tuned for one graph blankets a denser
    one: at 60 km a node is a third of the gap on the 1.5 deg ERA5 grid but twice the gap on
    a 0.25 deg ocean grid. Reporting the spacing lets the client scale instead of hardcoding.

    Two cases, because both kinds of graph land here:

    Regular lat/lon grid (ERA5, global ocean) — the gap between consecutive distinct
    latitudes is the same everywhere, so the median gap IS the step, exactly. Confirmed by
    requiring nearly all gaps to agree with it; measured 100% agreement on both.

    Irregular in lat/lon (limited-area graphs, regular in projected metres instead) — no
    lattice step exists. Only 19% of gaps agree on the Baltic graph. Estimated instead from
    the mean area per node over the covered box. Do not use the minimum gap here: a handful
    of near-coincident latitudes drag it to the rounding floor and yield metre-wide nodes.

    Must be computed here rather than client-side: the frontend only ever sees the nodes
    inside the current viewport, so it cannot see the true global spacing.
    """
    lat = torch.unique(torch.round(grid_lat_lon[:, 0].double() * 1000) / 1000)
    if lat.numel() >= 3:
        gaps = torch.diff(lat)
        gaps = gaps[gaps > 1e-6]
        if gaps.numel() > 0:
            step = float(gaps.median())
            uniform = float(((gaps - step).abs() <= 0.05 * step).double().mean())
            if uniform >= 0.9:
                return round(step, 6)

    # Irregular: spread the covered lat/lon box over the node count.
    n = max(1, grid_lat_lon.shape[0])
    lat_span = float(grid_lat_lon[:, 0].max() - grid_lat_lon[:, 0].min()) or 180.0
    lon_span = float(grid_lat_lon[:, 1].max() - grid_lat_lon[:, 1].min()) or 360.0
    return round((lat_span * lon_span / n) ** 0.5, 6)


def _localize_or_keep(
    edges_per_step: list[torch.Tensor], counts: list[int], name: str
) -> list[torch.Tensor]:
    """
    `localize_hierarchy`, degraded to a no-op when the graph fits neither index convention.

    The converter is right to raise there: it is writing a bundle to disk, and a misread
    index must not be persisted. On load the trade-off is the opposite one. This tool exists
    to inspect graphs that are structurally wrong, so refusing the whole load would take the
    six working subgraphs down with the bad one. Leaving the tensors untouched reproduces
    exactly the prior behaviour for that case -- the offending mesh_up/mesh_down subgraph
    errors when it is queried, everything else renders.
    """
    try:
        localized, rebased = localize_hierarchy(edges_per_step, counts, name)
    except (ValueError, IndexError) as e:
        logger.warning("load_graph: %s left in its original index space -- %s", name, e)
        return edges_per_step
    if rebased:
        logger.info("load_graph: rebased %s to level-local indices", ", ".join(rebased))
    return localized


def load_graph(folder: Path | str | None = None) -> dict:
    """
    Loads all .pt files and caches raw tensors per folder path.

    Detects hierarchical graphs at load time: mesh_lat_lon.pt is a list of
    >1 tensors (one per level). Single-scale graphs wrap mesh tensors in a
    single-element list — unwrap() handles that without affecting hierarchical.

    Builds a **subgraph registry** keyed by subgraph key, e.g. "g2m", "m2m_0",
    "mesh_up_1". Each entry stores:
        { src, edge, tgt, label, plot, edge_index, src_lat_lon, tgt_lat_lon }

    Single-scale registry keys: ["g2m", "m2m", "m2g"]
    Hierarchical registry keys (4-level): 12 entries — see EXPLORE_HIERARCHY.md

    Returns dict with:
        "meta"         — graph metadata including is_hierarchical, node_types, etc.
        "registry"     — subgraph registry dict
        "grid_lat_lon" — [N_grid, 2] tensor (always)
        "mesh_lat_lon" — single tensor (single-scale) or list of tensors (hierarchical)
        + flat index/feats keys for single-scale backward compat (route T2 removes them)
    """
    global _raw_cache

    if folder is None:
        folder = _DEFAULT_GRAPH_DIR
    else:
        folder = Path(folder)
        if not folder.is_absolute():
            folder = _GRAPHS_BASE / folder
    folder = folder.resolve()
    cache_key = str(folder)

    if cache_key in _raw_cache:
        return _raw_cache[cache_key]

    if not folder.exists():
        raise FileNotFoundError(
            f"Graph directory not found at {folder}. "
            "Set the GRAPH_DIR environment variable to point to a valid graph folder."
        )

    def load(filename):
        path = folder / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        return torch.load(path, weights_only=True)

    def unwrap(t):
        """Unwraps single-element [tensor] lists. Never call on multi-element lists."""
        return t[0] if isinstance(t, (list, tuple)) else t

    def norm_lon(t: torch.Tensor) -> torch.Tensor:
        """Normalize longitude column from [0, 360] → [-180, 180]."""
        out = t.clone().float()
        out[:, 1] = torch.where(out[:, 1] > 180, out[:, 1] - 360, out[:, 1])
        return out

    grid_lat_lon = norm_lon(unwrap(load("grid_lat_lon.pt")))   # [N_grid, 2]

    raw_mesh = load("mesh_lat_lon.pt")
    is_hierarchical = isinstance(raw_mesh, list) and len(raw_mesh) > 1

    if is_hierarchical:
        n_levels  = len(raw_mesh)
        mesh_lls  = [norm_lon(t) for t in raw_mesh]   # norm_lon applied per level

        # All hierarchical edge tensors are lists matching their level counts.
        # Do NOT unwrap — they are intentionally multi-element.
        m2m_indices  = load("m2m_edge_index.pt")       # list of n_levels
        up_indices   = load("mesh_up_edge_index.pt")   # list of n_levels - 1
        down_indices = load("mesh_down_edge_index.pt") # list of n_levels - 1
        g2m_index    = unwrap(load("g2m_edge_index.pt"))
        m2g_index    = unwrap(load("m2g_edge_index.pt"))

        # Normalize the mesh_up/mesh_down index convention before anything reads it.
        #
        # Two conventions exist upstream. create_graph.py's hierarchical path numbers each
        # side from 0 in its own level's space, which is what the registry, the bbox filters
        # and get_graph_metrics all assume. The K-means cluster path (Njord's
        # global_cluster_analysis, baltic_cluster) instead writes each step in the COMBINED
        # space of the pair it connects, [level i | level i+1], so the coarse row is offset
        # by the finer level's node count. Passed through raw, that row indexes past the end
        # of the coarse level's lat/lon tensor: an IndexError, surfacing as a 500 on
        # /edges/mesh_{up,down}_i and /graph_metrics/mesh_{up,down}_i.
        #
        # convert_bsf_to_neural_lam already localizes this for raw ocean-format uploads, but
        # a graph uploaded as a ready-made neural-lam bundle never passes through the
        # converter, so it has to happen here as well. Doing it in both places is safe --
        # an already-local row matches the first convention and is returned untouched.
        mesh_counts  = [t.shape[0] for t in mesh_lls]
        up_indices   = _localize_or_keep(up_indices,   mesh_counts, "mesh_up_edge_index.pt")
        down_indices = _localize_or_keep(down_indices, mesh_counts, "mesh_down_edge_index.pt")

        node_types  = ["grid"] + [f"mesh_{i}" for i in range(n_levels)]
        node_counts = {"grid": grid_lat_lon.shape[0]}
        node_counts.update({f"mesh_{i}": mesh_lls[i].shape[0] for i in range(n_levels)})

        registry: dict[str, dict] = {}

        registry["g2m"] = {
            "src": "grid", "edge": "g2m", "tgt": "mesh_0",
            "label": "G2M", "plot": "histogram",
            "edge_index":  g2m_index,
            "src_lat_lon": grid_lat_lon,
            "tgt_lat_lon": mesh_lls[0],
        }

        for i, ei in enumerate(m2m_indices):
            registry[f"m2m_{i}"] = {
                "src": f"mesh_{i}", "edge": "m2m", "tgt": f"mesh_{i}",
                "label": f"M2M L{i}", "plot": "lollipop",
                "edge_index":  ei,
                "src_lat_lon": mesh_lls[i],
                "tgt_lat_lon": mesh_lls[i],
            }

        # mesh_up_i: mesh_i → mesh_{i+1}, indices local to each level's space
        for i, ei in enumerate(up_indices):
            registry[f"mesh_up_{i}"] = {
                "src": f"mesh_{i}", "edge": "mesh_up", "tgt": f"mesh_{i+1}",
                "label": f"↑ L{i}→{i+1}", "plot": "lollipop",
                "edge_index":  ei,
                "src_lat_lon": mesh_lls[i],
                "tgt_lat_lon": mesh_lls[i + 1],
            }

        # mesh_down_i: mesh_{i+1} → mesh_i, indices local to each level's space
        for i, ei in enumerate(down_indices):
            registry[f"mesh_down_{i}"] = {
                "src": f"mesh_{i+1}", "edge": "mesh_down", "tgt": f"mesh_{i}",
                "label": f"↓ L{i+1}→{i}", "plot": "lollipop",
                "edge_index":  ei,
                "src_lat_lon": mesh_lls[i + 1],
                "tgt_lat_lon": mesh_lls[i],
            }

        registry["m2g"] = {
            "src": "mesh_0", "edge": "m2g", "tgt": "grid",
            "label": "M2G", "plot": "histogram",
            "edge_index":  m2g_index,
            "src_lat_lon": mesh_lls[0],
            "tgt_lat_lon": grid_lat_lon,
        }

        n_mesh_total = sum(mesh_lls[i].shape[0] for i in range(n_levels))
        _raw_cache[cache_key] = {
            "meta": {
                "spec_version": "0.3.0",
                "is_hierarchical": True,
                "n_levels": n_levels,
                "n_grid": grid_lat_lon.shape[0],
                "grid_spacing_deg": grid_spacing_deg(grid_lat_lon),
                "n_mesh": n_mesh_total,
                "node_types": node_types,
                "node_counts": node_counts,
                "folder": str(folder),
            },
            "grid_lat_lon": grid_lat_lon,
            "mesh_lat_lon": mesh_lls,   # list of tensors — route updated in T2
            "registry": registry,
        }

    else:
        # Single-scale: mesh tensors are wrapped in single-element lists
        mesh_lat_lon = norm_lon(unwrap(raw_mesh))
        g2m_index    = unwrap(load("g2m_edge_index.pt"))
        m2m_index    = unwrap(load("m2m_edge_index.pt"))
        m2g_index    = unwrap(load("m2g_edge_index.pt"))
        g2m_feats    = unwrap(load("g2m_features.pt"))
        m2m_feats    = unwrap(load("m2m_features.pt"))
        m2g_feats    = unwrap(load("m2g_features.pt"))

        node_types  = ["grid", "mesh"]
        node_counts = {
            "grid": grid_lat_lon.shape[0],
            "mesh": mesh_lat_lon.shape[0],
        }

        registry = {
            "g2m": {
                "src": "grid", "edge": "g2m", "tgt": "mesh",
                "label": "G2M", "plot": "histogram",
                "edge_index":  g2m_index,
                "src_lat_lon": grid_lat_lon,
                "tgt_lat_lon": mesh_lat_lon,
            },
            "m2m": {
                "src": "mesh", "edge": "m2m", "tgt": "mesh",
                "label": "M2M", "plot": "lollipop",
                "edge_index":  m2m_index,
                "src_lat_lon": mesh_lat_lon,
                "tgt_lat_lon": mesh_lat_lon,
            },
            "m2g": {
                "src": "mesh", "edge": "m2g", "tgt": "grid",
                "label": "M2G", "plot": "histogram",
                "edge_index":  m2g_index,
                "src_lat_lon": mesh_lat_lon,
                "tgt_lat_lon": grid_lat_lon,
            },
        }

        _raw_cache[cache_key] = {
            "meta": {
                "spec_version": "0.3.0",
                "is_hierarchical": False,
                "n_levels": 1,
                "n_grid": grid_lat_lon.shape[0],
                "grid_spacing_deg": grid_spacing_deg(grid_lat_lon),
                "n_mesh": mesh_lat_lon.shape[0],
                "node_types": node_types,
                "node_counts": node_counts,
                "n_edges_g2m": g2m_index.shape[1],
                "n_edges_m2m": m2m_index.shape[1],
                "n_edges_m2g": m2g_index.shape[1],
                "folder": str(folder),
            },
            "grid_lat_lon": grid_lat_lon,
            "mesh_lat_lon": mesh_lat_lon,   # single tensor — route uses this
            "registry": registry,
            # Flat keys kept for route layer backward compat (route unchanged in T1)
            "g2m_index": g2m_index, "g2m_feats": g2m_feats,
            "m2m_index": m2m_index, "m2m_feats": m2m_feats,
            "m2g_index": m2g_index, "m2g_feats": m2g_feats,
        }

    return _raw_cache[cache_key]


# ---------------------------------------------------------------------------
# Tensor-level graph statistics computation (degree + length)
# ---------------------------------------------------------------------------

def _degree_tensor(indices: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """
    Computes degree via torch.bincount — a single vectorized C call, no Python loop.

    Args:
        indices:  1D int tensor of node IDs.
                  Pass edge_index[0] for out-degree, edge_index[1] for in-degree.
        n_nodes:  number of nodes on that side (sets the minimum output length).

    Returns:
        Tensor [n_nodes] — degree of each node.
    """
    return torch.bincount(indices, minlength=n_nodes)


def _haversine_tensor(
    edge_index: torch.Tensor,
    src_lat_lon: torch.Tensor,
    tgt_lat_lon: torch.Tensor,
) -> torch.Tensor:
    """
    Computes haversine distance (km) for all edges in one batched tensor pass.

    Args:
        edge_index:  [2, E] — row 0 = source indices, row 1 = target indices.
        src_lat_lon: [N_src, 2] — lat/lon of source nodes.
        tgt_lat_lon: [N_tgt, 2] — lat/lon of target nodes.

    Returns:
        Tensor [E] — great-circle distance in km for each edge.
    """
    src  = src_lat_lon[edge_index[0]]
    tgt  = tgt_lat_lon[edge_index[1]]
    lat1 = torch.deg2rad(src[:, 0])
    lat2 = torch.deg2rad(tgt[:, 0])
    dlat = lat2 - lat1
    dlon = torch.deg2rad(tgt[:, 1] - src[:, 1])
    a    = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    return 2 * 6371.0 * torch.asin(torch.clamp(torch.sqrt(a), max=1.0))


def _two_std_bounds(values: torch.Tensor) -> tuple[float, float]:
    """Returns (mean - 2*std, mean + 2*std) as the normal range for a tensor."""
    v = values.float()
    return (v.mean() - 2 * v.std()).item(), (v.mean() + 2 * v.std()).item()



# ---------------------------------------------------------------------------
# Metric tensor cache  (degree + length, computed once per folder+subgraph_key)
# ---------------------------------------------------------------------------

def get_graph_metrics(subgraph_key: str, folder: Path | str | None = None) -> dict:
    """
    Returns (and caches) the metric tensors for the requested subgraph.

    Args:
        subgraph_key: registry key, e.g. "g2m", "m2m", "m2m_0", "mesh_up_1".
                      Single-scale keys "g2m"/"m2m"/"m2g" remain valid.

    Returns:
        {
            "src_deg":     Tensor [N_src] — out-degree of source nodes,
            "tgt_deg":     Tensor [N_tgt] — in-degree of target nodes
                           (identical to src_deg for symmetric subgraphs),
            "lengths":     Tensor [E] — haversine km per edge,
            "edge_index":  Tensor [2, E],
            "src_lat_lon": Tensor [N_src, 2],
            "tgt_lat_lon": Tensor [N_tgt, 2],
        }
    """
    raw = load_graph(folder)
    cache_key = raw["meta"]["folder"]
    if cache_key in _statistics_cache and subgraph_key in _statistics_cache[cache_key]:
        return _statistics_cache[cache_key][subgraph_key]

    reg = raw["registry"]
    if subgraph_key not in reg:
        raise ValueError(
            f"Unknown subgraph key '{subgraph_key}'. "
            f"Available: {sorted(reg.keys())}"
        )

    entry  = reg[subgraph_key]
    ei     = entry["edge_index"]
    src_ll = entry["src_lat_lon"]
    tgt_ll = entry["tgt_lat_lon"]

    src_deg = _degree_tensor(ei[0], src_ll.shape[0])
    # Symmetric subgraphs (m2m_*): src and tgt are the same node set — reuse tensor.
    tgt_deg = src_deg if entry["src"] == entry["tgt"] else _degree_tensor(ei[1], tgt_ll.shape[0])

    lengths = _haversine_tensor(ei, src_ll, tgt_ll)

    metrics = {
        "src_deg":     src_deg,
        "tgt_deg":     tgt_deg,
        "lengths":     lengths,
        # Ascending copy, cached alongside the unsorted one.
        #
        # Every consumer of the length array on the client is a threshold query: how many
        # edges fall below the slider's low mark, how many above its high mark, how many land
        # in each histogram bin, and whether the values form discrete levels. On unsorted data
        # each of those is a full O(E) scan, and they all rerun on every slider input event --
        # ~8M element visits per tick on the ocean m2g subgraph (2,011,528 edges). Sorted,
        # they are all binary searches.
        #
        # Sorting here rather than on the client is deliberate: torch.sort costs ~0.1 s once
        # per subgraph, and sorted float32 also gzips ~4.4x better than unsorted (measured on
        # ocean m2g: 1.78 MB vs 5.61 MB at compresslevel=1), because neighbouring values share
        # their high-order bytes. The client gets a smaller payload AND cheaper queries.
        "lengths_sorted": torch.sort(lengths).values.contiguous(),
        "edge_index":  ei,
        "src_lat_lon": src_ll,
        "tgt_lat_lon": tgt_ll,
    }

    _statistics_cache.setdefault(cache_key, {})[subgraph_key] = metrics
    return metrics


# ---------------------------------------------------------------------------
# Tensor → JSON-serializable list  (modular serialization)
# ---------------------------------------------------------------------------

def nodes_to_list(
    lat_lon: torch.Tensor,
    node_type: str,
    degree: torch.Tensor | None = None,
    ids: torch.Tensor | None = None,
) -> list[dict]:
    """
    Converts a [N, 2] lat/lon tensor to a list of node dicts.
    Bulk-converts tensors with .tolist() before assembling dicts (one C call each).

    Args:
        lat_lon:   [N, 2] tensor of lat/lon coordinates.
        node_type: "grid" or "mesh".
        degree:    optional [N] int tensor — adds a "degree" field.
        ids:       optional [N] int tensor of original node IDs.
                   Pass this when lat_lon is a masked subset so original IDs are preserved.
    """
    prefix   = node_type[0]
    coords   = lat_lon.tolist()
    degs     = degree.tolist() if degree is not None else None
    node_ids = ids.tolist()    if ids    is not None else list(range(len(coords)))

    if degs is not None:
        return [
            {"id": node_ids[i], "type": node_type, "label": f"{prefix}{node_ids[i]}",
             "lat": c[0], "lon": c[1], "degree": degs[i]}
            for i, c in enumerate(coords)
        ]
    return [
        {"id": node_ids[i], "type": node_type, "label": f"{prefix}{node_ids[i]}",
         "lat": c[0], "lon": c[1]}
        for i, c in enumerate(coords)
    ]


def edges_to_list(
    edge_index: torch.Tensor,
    feats: torch.Tensor | None,
    src_lat_lon: torch.Tensor,
    tgt_lat_lon: torch.Tensor,
    length: torch.Tensor | None = None,
) -> list[dict]:
    """
    Converts a [2, E] edge index to a list of edge dicts.
    Embeds src/tgt coordinates so the frontend can render edges without a node lookup,
    allowing edges to include endpoints outside the viewport bbox.

    Args:
        edge_index:  [2, E] — source/target node indices.
        feats:       [E, F] edge feature matrix, or None to omit the "features" field.
        src_lat_lon: [N_src, 2] — lat/lon of all source nodes (indexed by edge_index[0]).
        tgt_lat_lon: [N_tgt, 2] — lat/lon of all target nodes (indexed by edge_index[1]).
        length:      optional [E] float tensor — adds a "length" field (km).
    """
    sources    = edge_index[0].tolist()
    targets    = edge_index[1].tolist()
    src_coords = src_lat_lon[edge_index[0]].tolist()   # [E, 2]
    tgt_coords = tgt_lat_lon[edge_index[1]].tolist()   # [E, 2]
    feats_list = feats.tolist() if feats is not None else None
    feat_keys  = [f"f{j}" for j in range(feats.shape[1])] if feats is not None else []
    lengths    = length.tolist() if length is not None else None

    if feats_list is not None and lengths is not None:
        return [
            {"source": sources[i], "target": targets[i],
             "src_lat": src_coords[i][0], "src_lon": src_coords[i][1],
             "tgt_lat": tgt_coords[i][0], "tgt_lon": tgt_coords[i][1],
             "features": dict(zip(feat_keys, feats_list[i])),
             "length": lengths[i]}
            for i in range(len(sources))
        ]
    if feats_list is not None:
        return [
            {"source": sources[i], "target": targets[i],
             "src_lat": src_coords[i][0], "src_lon": src_coords[i][1],
             "tgt_lat": tgt_coords[i][0], "tgt_lon": tgt_coords[i][1],
             "features": dict(zip(feat_keys, feats_list[i]))}
            for i in range(len(sources))
        ]
    if lengths is not None:
        return [
            {"source": sources[i], "target": targets[i],
             "src_lat": src_coords[i][0], "src_lon": src_coords[i][1],
             "tgt_lat": tgt_coords[i][0], "tgt_lon": tgt_coords[i][1],
             "length": lengths[i]}
            for i in range(len(sources))
        ]
    return [
        {"source": sources[i], "target": targets[i],
         "src_lat": src_coords[i][0], "src_lon": src_coords[i][1],
         "tgt_lat": tgt_coords[i][0], "tgt_lon": tgt_coords[i][1]}
        for i in range(len(sources))
    ]
