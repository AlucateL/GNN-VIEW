"""
Spatial bounding-box utilities — pure torch implementation.

Faster than R-tree for this dataset scale (~30k nodes).
No build step, no cache, no extra dependency.

Benchmark result (global_multiscale, 1000 runs):
    torch : ~0.23 ms/call
    rtree : ~0.28 ms/call  (+842 ms one-off build on cold start)

Longitude is matched modulo 360° so the filter follows deck.gl's
wrapLongitude rendering at zoom-out and across the antimeridian.
See BBOX_TORCH.md for the full algorithm.
"""
import math
import torch


def _lon_mask(lon: torch.Tensor, min_lon: float, max_lon: float) -> torch.Tensor:
    """
    Boolean mask of nodes whose longitude lies in [min_lon, max_lon],
    treating longitude as equivalent modulo 360°.

    Three branches:
      1. width ≥ 360°  → whole world, mask is all True.
      2. shifted single segment → fold (min_lon, max_lon) into the canonical
         [-180, 180) frame; if it doesn't cross +180, compare directly.
      3. antimeridian-crossing → shifted segment straddles +180, so the
         predicate becomes (lon ≥ min_lon) OR (lon ≤ max_lon − 360).

    The node data is never shifted — only the viewport bounds. Node lons
    stay canonical in [-180, 180].
    """
    if max_lon - min_lon >= 360:
        return torch.ones_like(lon, dtype=torch.bool)

    shift = math.floor((min_lon + 180) / 360) * 360
    min_lon -= shift
    max_lon -= shift

    if max_lon <= 180:
        return (lon >= min_lon) & (lon <= max_lon)
    return (lon >= min_lon) | (lon <= max_lon - 360)


def bbox_node_mask(
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
    Returns a boolean mask [N] — True for nodes inside the new viewport bbox.

    Longitudes are matched modulo 360° so the filter handles antimeridian-
    crossing viewports and wrap-rendered world copies at zoom-out. See
    _lon_mask for the branching logic.

    If prev_* bounds are provided, returns only nodes that entered the viewport
    (in new bbox but NOT in prev bbox), so the caller can merge rather than replace.
    The same wrap-aware predicate is applied to the prev bounds.

    Args:
        lat_lon: [N, 2] tensor of (lat, lon) coordinates.
        min_lat, max_lat, min_lon, max_lon: new viewport bounds.
        prev_*: previous viewport bounds; if all four given, returns delta only.
    """
    lat = lat_lon[:, 0]
    lon = lat_lon[:, 1]
    new_mask = (lat >= min_lat) & (lat <= max_lat) & _lon_mask(lon, min_lon, max_lon)
    if prev_min_lat is not None:
        prev_mask = (
            (lat >= prev_min_lat) & (lat <= prev_max_lat) &
            _lon_mask(lon, prev_min_lon, prev_max_lon)
        )
        return new_mask & ~prev_mask
    return new_mask


def filter_edges_by_node_masks(
    edge_index: torch.Tensor,
    feats: torch.Tensor | None,
    src_mask: torch.Tensor,
    tgt_mask: torch.Tensor,
    src_mask_prev: torch.Tensor | None = None,
    tgt_mask_prev: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Keeps edges where at least one endpoint is within the new bbox (OR semantics —
    allows edges that cross the viewport boundary to be visible).

    If prev masks are provided, also excludes edges that were already sent
    (at least one endpoint was in the previous bbox), so the caller can merge.

    Args:
        edge_index:    [2, E]
        feats:         [E, F] or None — when None, the second return value is also None
        src_mask:      [N_src] bool — True if source node is in NEW bbox
        tgt_mask:      [N_tgt] bool — True if target node is in NEW bbox
        src_mask_prev: [N_src] bool — True if source was in PREV bbox (optional)
        tgt_mask_prev: [N_tgt] bool — True if target was in PREV bbox (optional)

    Returns:
        (filtered_edge_index [2, E'], filtered_feats [E', F] or None)
    """
    keep = src_mask[edge_index[0]] | tgt_mask[edge_index[1]]
    if src_mask_prev is not None:
        already_sent = src_mask_prev[edge_index[0]] | tgt_mask_prev[edge_index[1]]
        keep = keep & ~already_sent
    filtered_feats = feats[keep] if feats is not None else None
    return edge_index[:, keep], filtered_feats