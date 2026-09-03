"""
Tests for services.bbox_torch.

Runnable in two ways:
    pytest backend/tests/test_bbox_torch.py            # if pytest is installed
    python3 backend/tests/test_bbox_torch.py           # bare-assert fallback

Both modes share the same `test_*` functions.
"""
import sys
from pathlib import Path

import torch

# Make services.bbox_torch importable without installing the package.
# Mirrors the docker layout (/usr/src/app on PYTHONPATH).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from services.bbox_torch import (  # noqa: E402
    _lon_mask,
    bbox_node_mask,
    filter_edges_by_node_masks,
)


# Reusable sample of canonical node coordinates (lat ∈ [-90, 90], lon ∈ [-180, 180]).
NODES = torch.tensor(
    [
        [0.0,     0.0],   # 0  equator, prime meridian
        [45.0,   45.0],   # 1
        [-45.0, -45.0],   # 2
        [10.0,  179.0],   # 3  just west of antimeridian
        [10.0, -179.0],   # 4  just east of antimeridian
        [60.0,  170.0],   # 5
        [60.0, -170.0],   # 6
        [-80.0,   0.0],   # 7  far south
        [80.0,  100.0],   # 8  far north-east
    ],
    dtype=torch.float64,
)


# ─────────────────────────────────────────────────────────────────────────────
# _lon_mask
# ─────────────────────────────────────────────────────────────────────────────

def test_lon_mask_normal_segment():
    lon = NODES[:, 1]
    m = _lon_mask(lon, -30.0, 30.0)
    assert m.tolist() == [True, False, False, False, False, False, False, True, False]


def test_lon_mask_whole_world():
    lon = NODES[:, 1]
    m = _lon_mask(lon, -540.0, 540.0)
    assert m.all().item()


def test_lon_mask_whole_world_exact_360():
    lon = NODES[:, 1]
    m = _lon_mask(lon, -180.0, 180.0)
    assert m.all().item()


def test_lon_mask_antimeridian_no_shift():
    # viewport (170, 190) straddles +180 — picks up 179, -179, 170, -170
    lon = NODES[:, 1]
    m = _lon_mask(lon, 170.0, 190.0)
    expected = [False, False, False, True, True, True, True, False, False]
    assert m.tolist() == expected


def test_lon_mask_shift_east():
    # viewport (190, 200) is equivalent to (-170, -160) after shift
    lon = NODES[:, 1]
    m = _lon_mask(lon, 190.0, 200.0)
    # only node 6 at lon=-170 is on the boundary
    assert m.tolist() == [False, False, False, False, False, False, True, False, False]


def test_lon_mask_shift_west():
    # viewport (-200, -160) — shift = -360 → equivalent to (160, 200) → straddles +180
    lon = NODES[:, 1]
    m = _lon_mask(lon, -200.0, -160.0)
    # picks up 170, 179, -179, -170
    expected = [False, False, False, True, True, True, True, False, False]
    assert m.tolist() == expected


def test_lon_mask_far_east_shift():
    # viewport (300, 320) is equivalent to (-60, -40) after shift = 360
    lon = NODES[:, 1]
    m = _lon_mask(lon, 300.0, 320.0)
    # node 2 at lon=-45 is the only one inside the [-60, -40] window
    assert m.tolist() == [False, False, True, False, False, False, False, False, False]


# ─────────────────────────────────────────────────────────────────────────────
# bbox_node_mask
# ─────────────────────────────────────────────────────────────────────────────

def test_bbox_node_mask_lat_only_clips():
    # equator-only band keeps node 0 only (lat==0)
    m = bbox_node_mask(NODES, -1.0, 1.0, -180.0, 180.0)
    assert m.tolist() == [True, False, False, False, False, False, False, False, False]


def test_bbox_node_mask_high_lat_band_with_antimeridian():
    # lat in [55, 65] AND viewport across antimeridian
    m = bbox_node_mask(NODES, 55.0, 65.0, 170.0, 190.0)
    # nodes 5 (60, 170) and 6 (60, -170) qualify
    expected = [False, False, False, False, False, True, True, False, False]
    assert m.tolist() == expected


def test_bbox_node_mask_whole_world_at_zoomout():
    # max-zoom-out viewport — lon ranges >= 360, lat clamped to ±90
    m = bbox_node_mask(NODES, -90.0, 90.0, -540.0, 540.0)
    assert m.all().item()


def test_bbox_node_mask_delta_excludes_prev():
    # new viewport = (-30..30 lat, -90..90 lon) ; prev was (-30..30, -30..30)
    # delta should contain nodes inside new but not prev
    new_mask = bbox_node_mask(NODES, -30.0, 30.0, -90.0, 90.0)
    full_mask = bbox_node_mask(
        NODES, -30.0, 30.0, -90.0, 90.0,
        prev_min_lat=-30.0, prev_max_lat=30.0,
        prev_min_lon=-30.0, prev_max_lon=30.0,
    )
    # node 0 (0,0) was in prev — must be excluded from delta
    assert new_mask[0].item() is True
    assert full_mask[0].item() is False
    # whatever else is in new should now equal new ∩ ~prev
    prev_mask = bbox_node_mask(NODES, -30.0, 30.0, -30.0, 30.0)
    assert torch.equal(full_mask, new_mask & ~prev_mask)


def test_bbox_node_mask_delta_wrap_aware():
    # Prev: standard view in eastern hemisphere (170..180 lon, 55..65 lat) — covers node 5 only.
    # New:  pans east across antimeridian (170..190 lon, 55..65 lat) — covers nodes 5 and 6.
    # Delta should be just node 6 (entered viewport via wrap).
    full_mask = bbox_node_mask(
        NODES, 55.0, 65.0, 170.0, 190.0,
        prev_min_lat=55.0, prev_max_lat=65.0,
        prev_min_lon=170.0, prev_max_lon=180.0,
    )
    expected = [False, False, False, False, False, False, True, False, False]
    assert full_mask.tolist() == expected


# ─────────────────────────────────────────────────────────────────────────────
# filter_edges_by_node_masks
# ─────────────────────────────────────────────────────────────────────────────

def test_edge_filter_or_semantics_keeps_boundary_edges():
    # 3 edges over 4 nodes. src_mask=[T,F,F,F], tgt_mask=[T,F,F,F].
    #  0→1: src in       → kept
    #  2→0: tgt in       → kept
    #  3→2: neither in   → dropped
    edge_index = torch.tensor([[0, 2, 3], [1, 0, 2]], dtype=torch.long)
    src_mask = torch.tensor([True, False, False, False])
    tgt_mask = torch.tensor([True, False, False, False])
    out_ei, out_feats = filter_edges_by_node_masks(edge_index, None, src_mask, tgt_mask)
    assert out_ei.tolist() == [[0, 2], [1, 0]]
    assert out_feats is None


def test_edge_filter_feats_propagate():
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    feats = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    src_mask = torch.tensor([True, False, False])
    tgt_mask = torch.tensor([False, False, False])
    out_ei, out_feats = filter_edges_by_node_masks(edge_index, feats, src_mask, tgt_mask)
    assert out_ei.tolist() == [[0], [1]]
    assert out_feats.tolist() == [[1.0, 1.0]]


def test_edge_filter_delta_excludes_already_sent():
    # 3 edges: 0→1, 1→2, 2→3
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    # New viewport — all four nodes in
    src_mask = torch.tensor([True, True, True, True])
    tgt_mask = torch.tensor([True, True, True, True])
    # Prev — nodes 0, 1 were already visible; edge 0→1 already sent
    src_mask_prev = torch.tensor([True, True, False, False])
    tgt_mask_prev = torch.tensor([True, True, False, False])
    out_ei, _ = filter_edges_by_node_masks(
        edge_index, None, src_mask, tgt_mask, src_mask_prev, tgt_mask_prev
    )
    # Edge 0→1: both endpoints in prev → excluded.
    # Edge 1→2: endpoint 1 in prev → excluded.
    # Edge 2→3: neither in prev → kept.
    assert out_ei.tolist() == [[2], [3]]


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner — used when pytest is not installed.
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)