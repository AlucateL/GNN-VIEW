"""
Benchmark: torch bbox mask vs R-tree for viewport filtering.

Run from repo root:
    conda run -n cute_GNNs python benchmark_bbox.py
"""
import time
import torch
from rtree import index as rtree_index

# ---------------------------------------------------------------------------
# Load real data
# ---------------------------------------------------------------------------
BASE = "data/graphs/global_multiscale/"

def unwrap(t):
    return t[0] if isinstance(t, (list, tuple)) else t

grid_lat_lon = unwrap(torch.load(BASE + "grid_lat_lon.pt",   weights_only=True))  # [29040, 2]
mesh_lat_lon = unwrap(torch.load(BASE + "mesh_lat_lon.pt",   weights_only=True))  # [2562,  2]
g2m_index    = unwrap(torch.load(BASE + "g2m_edge_index.pt", weights_only=True))  # [2, 46158]
g2m_feats    = unwrap(torch.load(BASE + "g2m_features.pt",   weights_only=True))

print(f"grid nodes : {grid_lat_lon.shape[0]:,}")
print(f"mesh nodes : {mesh_lat_lon.shape[0]:,}")
print(f"g2m edges  : {g2m_index.shape[1]:,}")
print()

# ---------------------------------------------------------------------------
# Viewport bbox — roughly Europe
# ---------------------------------------------------------------------------
MIN_LAT, MAX_LAT = 35.0, 60.0
MIN_LON, MAX_LON = -10.0, 30.0

RUNS = 1_000  # repeat each op N times to get stable timings

# ---------------------------------------------------------------------------
# APPROACH A: torch boolean mask
# ---------------------------------------------------------------------------

def bbox_node_mask_torch(lat_lon, min_lat, max_lat, min_lon, max_lon):
    return (
        (lat_lon[:, 0] >= min_lat) & (lat_lon[:, 0] <= max_lat) &
        (lat_lon[:, 1] >= min_lon) & (lat_lon[:, 1] <= max_lon)
    )

def filter_edges_by_masks(edge_index, feats, src_mask, tgt_mask):
    keep = src_mask[edge_index[0]] & tgt_mask[edge_index[1]]
    return edge_index[:, keep], feats[keep]

# warm-up
_ = bbox_node_mask_torch(grid_lat_lon, MIN_LAT, MAX_LAT, MIN_LON, MAX_LON)

t0 = time.perf_counter()
for _ in range(RUNS):
    grid_mask = bbox_node_mask_torch(grid_lat_lon, MIN_LAT, MAX_LAT, MIN_LON, MAX_LON)
    mesh_mask = bbox_node_mask_torch(mesh_lat_lon, MIN_LAT, MAX_LAT, MIN_LON, MAX_LON)
    ei_t, ft_t = filter_edges_by_masks(g2m_index, g2m_feats, grid_mask, mesh_mask)
t_torch = (time.perf_counter() - t0) / RUNS * 1000  # ms per call

torch_grid_hits = grid_mask.sum().item()
torch_mesh_hits = mesh_mask.sum().item()
torch_edge_hits = ei_t.shape[1]

# ---------------------------------------------------------------------------
# APPROACH B: R-tree (build once, then query)
# ---------------------------------------------------------------------------

def build_rtree(lat_lon):
    idx = rtree_index.Index()
    for i, (lat, lon) in enumerate(lat_lon.tolist()):
        idx.insert(i, (lon, lat, lon, lat))
    return idx

def nodes_in_bbox_rtree(rtree, min_lat, max_lat, min_lon, max_lon):
    hits = list(rtree.intersection((min_lon, min_lat, max_lon, max_lat)))
    return torch.tensor(sorted(hits), dtype=torch.long)

def node_mask_from_ids(ids, n):
    mask = torch.zeros(n, dtype=torch.bool)
    if ids.numel() > 0:
        mask[ids] = True
    return mask

# build time (one-off cost)
t_build_start = time.perf_counter()
rtree_grid = build_rtree(grid_lat_lon)
rtree_mesh = build_rtree(mesh_lat_lon)
t_build = (time.perf_counter() - t_build_start) * 1000  # ms

# warm-up
_ = nodes_in_bbox_rtree(rtree_grid, MIN_LAT, MAX_LAT, MIN_LON, MAX_LON)

t0 = time.perf_counter()
for _ in range(RUNS):
    grid_ids   = nodes_in_bbox_rtree(rtree_grid, MIN_LAT, MAX_LAT, MIN_LON, MAX_LON)
    mesh_ids   = nodes_in_bbox_rtree(rtree_mesh, MIN_LAT, MAX_LAT, MIN_LON, MAX_LON)
    grid_mask_r = node_mask_from_ids(grid_ids, grid_lat_lon.shape[0])
    mesh_mask_r = node_mask_from_ids(mesh_ids, mesh_lat_lon.shape[0])
    ei_r, ft_r = filter_edges_by_masks(g2m_index, g2m_feats, grid_mask_r, mesh_mask_r)
t_rtree = (time.perf_counter() - t0) / RUNS * 1000  # ms per call

rtree_grid_hits = grid_ids.shape[0]
rtree_mesh_hits = mesh_ids.shape[0]
rtree_edge_hits = ei_r.shape[1]

# ---------------------------------------------------------------------------
# APPROACH C: multiple crops around violations (union of small bboxes)
# ---------------------------------------------------------------------------
# Simulate 50 violations scattered across the globe, each with a 5° crop

torch.manual_seed(42)
N_VIOLS = 50
CROP_DEG = 5.0
viol_lats = (torch.rand(N_VIOLS) * 120 - 60).tolist()   # [-60, 60]
viol_lons = (torch.rand(N_VIOLS) * 360 - 180).tolist()  # [-180, 180]

def multi_crop_torch(lat_lon, centers_lat, centers_lon, delta):
    combined = torch.zeros(lat_lon.shape[0], dtype=torch.bool)
    for clat, clon in zip(centers_lat, centers_lon):
        combined |= bbox_node_mask_torch(lat_lon, clat - delta, clat + delta, clon - delta, clon + delta)
    return combined

def multi_crop_rtree(rtree, n_nodes, centers_lat, centers_lon, delta):
    id_set = set()
    for clat, clon in zip(centers_lat, centers_lon):
        id_set.update(rtree.intersection((clon - delta, clat - delta, clon + delta, clat + delta)))
    ids = torch.tensor(sorted(id_set), dtype=torch.long)
    return node_mask_from_ids(ids, n_nodes)

RUNS_MULTI = 200

t0 = time.perf_counter()
for _ in range(RUNS_MULTI):
    gm = multi_crop_torch(grid_lat_lon, viol_lats, viol_lons, CROP_DEG)
    mm = multi_crop_torch(mesh_lat_lon, viol_lats, viol_lons, CROP_DEG)
t_multi_torch = (time.perf_counter() - t0) / RUNS_MULTI * 1000

t0 = time.perf_counter()
for _ in range(RUNS_MULTI):
    gm_r = multi_crop_rtree(rtree_grid, grid_lat_lon.shape[0], viol_lats, viol_lons, CROP_DEG)
    mm_r = multi_crop_rtree(rtree_mesh, mesh_lat_lon.shape[0], viol_lats, viol_lons, CROP_DEG)
t_multi_rtree = (time.perf_counter() - t0) / RUNS_MULTI * 1000

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
print("=" * 55)
print("SINGLE VIEWPORT QUERY  (avg over {:,} runs)".format(RUNS))
print("=" * 55)
print(f"  torch mask  : {t_torch:.4f} ms/call   → nodes in view: {torch_grid_hits:,}g  {torch_mesh_hits:,}m  edges: {torch_edge_hits:,}")
print(f"  rtree query : {t_rtree:.4f} ms/call   → nodes in view: {rtree_grid_hits:,}g  {rtree_mesh_hits:,}m  edges: {rtree_edge_hits:,}")
print(f"  rtree build : {t_build:.1f} ms  (one-off, then cached)")
print(f"  speedup     : {t_rtree/t_torch:.1f}x  {'(torch faster)' if t_torch < t_rtree else '(rtree faster)'}")
print()
print("=" * 55)
print("MULTI-CROP ({} violations, ±{}°)  (avg over {:,} runs)".format(N_VIOLS, CROP_DEG, RUNS_MULTI))
print("=" * 55)
print(f"  torch mask  : {t_multi_torch:.4f} ms/call")
print(f"  rtree query : {t_multi_rtree:.4f} ms/call")
print(f"  speedup     : {t_multi_rtree/t_multi_torch:.1f}x  {'(torch faster)' if t_multi_torch < t_multi_rtree else '(rtree faster)'}")
print()
print("Sanity check — same results?", bool((gm == gm_r).all() and (mm == mm_r).all()))
