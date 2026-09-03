"""
Tests for mesh_up/mesh_down index localization in services.bsf_converter.

Runnable in two ways:
    pytest backend/tests/test_bsf_hierarchy_localize.py    # if pytest is installed
    python3 backend/tests/test_bsf_hierarchy_localize.py   # bare-assert fallback

Both modes share the same `test_*` functions.
"""
import sys
from pathlib import Path

import torch

# Make services.bsf_converter importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from services.bsf_converter import localize_hierarchy  # noqa: E402

# Two levels: 6 fine nodes, 2 coarse nodes. In the combined space the coarse nodes are
# numbered 6 and 7; level-local they are 0 and 1.
COUNTS = [6, 2, 1]


def test_combined_space_is_rebased():
    """The cluster-path convention: coarse row offset by the finer level's node count."""
    up = [torch.tensor([[0, 1, 2, 3, 4, 5], [6, 6, 6, 7, 7, 7]])]
    out, rebased = localize_hierarchy(up, COUNTS, "mesh_up_edge_index.pt")

    assert out[0][1].tolist() == [0, 0, 0, 1, 1, 1]
    assert out[0][0].tolist() == [0, 1, 2, 3, 4, 5], "fine row must be left alone"
    assert rebased == ["mesh_up_0"]


def test_level_local_is_untouched():
    """The create_graph.py convention: already local, must pass through unchanged."""
    up = [torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 0, 1, 1, 1]])]
    out, rebased = localize_hierarchy(up, COUNTS, "mesh_up_edge_index.pt")

    assert out[0].tolist() == up[0].tolist()
    assert rebased == []


def test_mesh_down_rebases_the_other_row():
    """mesh_down runs coarse -> fine, so the offset sits on row 0, not row 1."""
    down = [torch.tensor([[6, 6, 6, 7, 7, 7], [0, 1, 2, 3, 4, 5]])]
    out, rebased = localize_hierarchy(down, COUNTS, "mesh_down_edge_index.pt")

    assert out[0][0].tolist() == [0, 0, 0, 1, 1, 1]
    assert out[0][1].tolist() == [0, 1, 2, 3, 4, 5]
    assert rebased == ["mesh_down_0"]


def test_sparse_level_local_row_is_not_shifted():
    """
    The trap that rules out "just subtract the row minimum".

    Here coarse node 0 has no up-edge, so the row minimum is 1 while the indices are already
    level-local. Subtracting the minimum would silently point every edge at the wrong node;
    rebasing against the known offset leaves it correct.
    """
    up = [torch.tensor([[0, 1, 2, 3, 4, 5], [1, 1, 1, 1, 1, 1]])]
    out, rebased = localize_hierarchy(up, COUNTS, "mesh_up_edge_index.pt")

    assert out[0][1].tolist() == [1, 1, 1, 1, 1, 1]
    assert rebased == []


def test_unknown_convention_raises():
    """Neither local nor offset by the finer count -> fail loudly, not out of bounds later."""
    up = [torch.tensor([[0, 1, 2, 3, 4, 5], [40, 41, 42, 43, 44, 45]])]
    try:
        localize_hierarchy(up, COUNTS, "mesh_up_edge_index.pt")
    except ValueError as e:
        assert "index convention" in str(e)
    else:
        raise AssertionError("expected ValueError for an unrecognised index convention")


def test_multi_step_hierarchy():
    """Step i is offset by counts[i], not by a cumulative base across all levels."""
    counts = [6, 2, 1]
    up = [
        torch.tensor([[0, 1, 2, 3, 4, 5], [6, 6, 6, 7, 7, 7]]),  # offset 6  = counts[0]
        torch.tensor([[0, 1], [2, 2]]),                          # offset 2  = counts[1]
    ]
    out, rebased = localize_hierarchy(up, counts, "mesh_up_edge_index.pt")

    assert out[1][1].tolist() == [0, 0]
    assert rebased == ["mesh_up_0", "mesh_up_1"]


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