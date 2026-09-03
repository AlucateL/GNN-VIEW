"""
Spatial bounding-box utilities for viewport-based graph filtering.

The frontend sends the current map viewport as (min_lat, max_lat, min_lon, max_lon)
and these utilities filter nodes/edges to that rectangle.

R-tree indices are built once per graph folder and cached in memory.
"""
import torch
from rtree import index as rtree_index

_rtree_cache: dict[str, dict] = {}


def _build_rtree(lat_lon: torch.Tensor) -> rtree_index.Index:
    idx = rtree_index.Index()
    for i, (lat, lon) in enumerate(lat_lon.tolist()):
        idx.insert(i, (lon, lat, lon, lat))
    return idx


def get_or_build_rtrees(
    folder_key: str,
    grid_lat_lon: torch.Tensor,
    mesh_lat_lon: torch.Tensor,
) -> dict:
    """Returns cached R-tree indices (keyed by graph folder) for grid and mesh nodes."""
    if folder_key not in _rtree_cache:
        import time
        print(f"[rtree] Building spatial index for '{folder_key}' ({grid_lat_lon.shape[0]} grid + {mesh_lat_lon.shape[0]} mesh nodes)...", flush=True)
        t0 = time.perf_counter()
        trees = {
            "grid": _build_rtree(grid_lat_lon),
            "mesh": _build_rtree(mesh_lat_lon),
        }
        _rtree_cache.setdefault(folder_key, trees)
        print(f"[rtree] Index ready in {time.perf_counter() - t0:.2f}s (cached until restart)", flush=True)
    return _rtree_cache[folder_key]


def nodes_in_bbox(
    rtree: rtree_index.Index,
    lat_lon: torch.Tensor,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    prev_min_lat: float | None = None,
    prev_max_lat: float | None = None,
    prev_min_lon: float | None = None,
    prev_max_lon: float | None = None,
) -> torch.Tensor:
    """
    Returns a sorted 1D tensor of node IDs inside the new viewport bbox.

    If prev_* bounds are provided, performs a single tree query for the new bbox
    then filters out nodes that also fall inside the previous bbox (coordinate check
    on the small result set — no second tree traversal).

    rtree expects (min_lon, min_lat, max_lon, max_lat).
    """
    hits = list(rtree.intersection((min_lon, min_lat, max_lon, max_lat)))
    if prev_min_lat is not None:
        coords = lat_lon.tolist()
        hits = [
            i for i in hits
            if not (prev_min_lat <= coords[i][0] <= prev_max_lat and
                    prev_min_lon <= coords[i][1] <= prev_max_lon)
        ]
    return torch.tensor(sorted(hits), dtype=torch.long)


def node_mask_from_ids(node_ids: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Converts a 1D tensor of node IDs to a boolean membership mask of length n_nodes."""
    mask = torch.zeros(n_nodes, dtype=torch.bool)
    if node_ids.numel() > 0:
        mask[node_ids] = True
    return mask


def filter_edges_by_node_masks(
    edge_index: torch.Tensor,
    feats: torch.Tensor,
    src_mask: torch.Tensor,
    tgt_mask: torch.Tensor,
    src_mask_prev: torch.Tensor | None = None,
    tgt_mask_prev: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Keeps edges where at least one endpoint is within the new bbox (OR semantics —
    allows edges that cross the viewport boundary to be visible).

    If prev masks are provided, also excludes edges already sent in the previous
    viewport (both endpoints were visible then), so the caller can merge.

    Args:
        edge_index:    [2, E]
        feats:         [E, F]
        src_mask:      [N_src] bool — True if source node is in NEW bbox
        tgt_mask:      [N_tgt] bool — True if target node is in NEW bbox
        src_mask_prev: [N_src] bool — True if source was in PREV bbox (optional)
        tgt_mask_prev: [N_tgt] bool — True if target was in PREV bbox (optional)

    Returns:
        (filtered_edge_index [2, E'], filtered_feats [E', F])
    """
    keep = src_mask[edge_index[0]] | tgt_mask[edge_index[1]]
    if src_mask_prev is not None:
        already_sent = src_mask_prev[edge_index[0]] | tgt_mask_prev[edge_index[1]]
        keep = keep & ~already_sent
    return edge_index[:, keep], feats[keep]
