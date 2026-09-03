"""
Tests for state-column resolution across domains (services.inference).

`compute_prediction_errors` used to carry five hand-written index cases for the ERA5
variable x level layout. They are now one call to `resolve_state_columns` followed by a
per-node RMSE, so these tests pin two things:

  1. the ERA5 column selection is byte-identical to the pre-refactor arithmetic, and
  2. ocean caches -- whose features carry their depth in the name and have no level axis --
     resolve through the same entry point without borrowing ERA5's arithmetic.

Both layouts are dense and neither raises when indexed with the other's formula, so a
mix-up returns a different variable rather than an error. That is what is being guarded.

Runnable in two ways:
    pytest backend/tests/test_state_columns.py
    python3 backend/tests/test_state_columns.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from services.inference import resolve_state_columns, state_column_names  # noqa: E402

ATM = ["geopotential", "specific_humidity", "temperature", "u", "v", "w"]
SURF = ["2t", "10u", "10v", "msl", "tp"]
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
ERA5_CACHE = {"atm_vars": ATM, "surf_vars": SURF, "levels": LEVELS}

OCEAN_FEATURES = [
    "zos", "siconc", "sithick",
    "thetao_0m", "thetao_47m", "so_0m", "so_47m", "uo_0m", "vo_0m",
]
OCEAN_CACHE = {"_ocean": True, "state_features": OCEAN_FEATURES}

_N_LEVELS = len(LEVELS)
_N_ATM = len(ATM)
_D_STATE = _N_ATM * _N_LEVELS + len(SURF)


def _legacy_columns(variable, z_level):
    """The five index cases exactly as they were written before the refactor."""
    if variable in SURF:
        return [_N_ATM * _N_LEVELS + SURF.index(variable)]
    if variable in ATM and z_level != "total":
        return [ATM.index(variable) * _N_LEVELS + LEVELS.index(int(z_level))]
    if variable in ATM and z_level == "total":
        i = ATM.index(variable)
        return list(range(i * _N_LEVELS, (i + 1) * _N_LEVELS))
    if variable == "total" and z_level != "total":
        li = LEVELS.index(int(z_level))
        return [v * _N_LEVELS + li for v in range(_N_ATM)]
    if variable == "total" and z_level == "total":
        return list(range(_D_STATE))
    raise ValueError(variable, z_level)


def test_era5_matches_legacy_arithmetic():
    for variable, z_level in [
        ("total", "total"), ("total", "500"),
        ("temperature", "500"), ("temperature", "total"),
        ("geopotential", "1000"), ("w", "50"),
        ("2t", "total"), ("msl", "total"),
    ]:
        assert resolve_state_columns(ERA5_CACHE, variable, z_level) == _legacy_columns(
            variable, z_level
        ), f"{variable}@{z_level}"


def test_era5_column_names_cover_d_state():
    names = state_column_names(ERA5_CACHE)
    assert len(names) == _D_STATE
    assert names[0] == "geopotential@50"
    assert names[-1] == "tp"


def test_single_column_rmse_is_absolute_error():
    """
    The identity the collapsed aggregation relies on.

    Cases 1 and 5 returned |pred - truth| for one column; the unified path takes an RMSE
    over the selected columns. For a single column those are the same number, which is why
    the five branches could become one.
    """
    err = torch.randn(500, _D_STATE).abs()
    cols = resolve_state_columns(ERA5_CACHE, "2t", "total")
    assert torch.allclose((err[:, cols] ** 2).mean(dim=-1).sqrt(), err[:, cols[0]])


def test_surface_variable_ignores_level():
    """Surface variables have no level, so z_level must not shift the column."""
    assert resolve_state_columns(ERA5_CACHE, "msl", "total") == resolve_state_columns(
        ERA5_CACHE, "msl", "500"
    )


def test_total_at_one_level_excludes_surface():
    """Surface variables sit at no pressure level and must not be folded into one."""
    cols = resolve_state_columns(ERA5_CACHE, "total", "500")
    assert len(cols) == _N_ATM
    assert max(cols) < _N_ATM * _N_LEVELS


def test_ocean_resolves_by_name():
    assert resolve_state_columns(OCEAN_CACHE, "thetao_47m", None) == [4]
    assert resolve_state_columns(OCEAN_CACHE, "zos", "total") == [0]
    assert state_column_names(OCEAN_CACHE) == OCEAN_FEATURES


def test_ocean_total_is_every_feature():
    assert resolve_state_columns(OCEAN_CACHE, "total", "total") == list(
        range(len(OCEAN_FEATURES))
    )


def test_ocean_rejects_a_separate_level():
    """
    Depth belongs to the feature name. Accepting a level and ignoring it would return the
    surface field labelled as depth -- silently, since both are valid columns.
    """
    for variable in ("thetao_0m", "total"):
        try:
            resolve_state_columns(OCEAN_CACHE, variable, "47")
        except ValueError as e:
            assert "no separate level axis" in str(e)
        else:
            raise AssertionError(f"expected ValueError for {variable} with a level")


def test_ocean_unknown_feature_raises():
    try:
        resolve_state_columns(OCEAN_CACHE, "thetao", None)   # missing its depth suffix
    except ValueError as e:
        assert "Unknown ocean feature" in str(e)
    else:
        raise AssertionError("expected ValueError for an unqualified ocean feature")


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