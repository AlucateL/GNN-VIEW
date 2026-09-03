"""
Smoke test for /inference/normalization + /inference/prediction_field.

Verifies the frontend will get everything it needs to:
  - denormalize pred and truth to physical units
  - compute |pred - truth| client-side
  - reproduce the existing /prediction_errors output (within float rounding)
  - colour a map with a stable physical [min, max] scale

Run against a backend that has its ERA5 window initialised. Defaults to the
local docker-compose backend.

  python backend/tests/test_prediction_field.py
  python backend/tests/test_prediction_field.py --base-url http://localhost:8080 --pred-time 2020-08-27T12:00:00

Exit code is non-zero if any assertion fails so it can be wired into CI later.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import requests


N_GRID = 29040  # 240 × 121, the ERA5 1.5° global grid used by this backend
D_STATE = 83    # 6 atm × 13 levels + 5 surface


def hit(url: str, **params) -> dict:
    try:
        r = requests.get(url, params=params, timeout=120)
    except requests.exceptions.ConnectionError:
        raise SystemExit(
            f"FAIL — could not connect to {url}. Is the backend running?\n"
            f"       Start it with: docker compose up -d backend\n"
            f"       Verify with:   curl -s http://localhost:8080/health"
        )
    if not r.ok:
        raise SystemExit(f"FAIL {r.status_code} {r.url}\n{r.text[:500]}")
    return r.json()


def check(name: str, cond: bool, detail: str = "") -> None:
    mark = "OK  " if cond else "FAIL"
    print(f"  [{mark}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8080")
    ap.add_argument("--pred-time", default="2020-08-27T12:00:00")
    ap.add_argument("--variable", default="temperature")
    ap.add_argument("--z-level", default="500")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")

    # ── 1. /normalization ────────────────────────────────────────────────────
    print("\n/inference/normalization")
    norm = hit(f"{base}/inference/normalization")
    mean = np.asarray(norm["mean"], dtype=np.float64)
    std  = np.asarray(norm["std"],  dtype=np.float64)
    scales = norm["physical_scales"]

    check("mean is length d_state",  mean.shape == (D_STATE,), f"got {mean.shape}")
    check("std is length d_state",   std.shape  == (D_STATE,), f"got {std.shape}")
    check("std is strictly positive", float(std.min()) > 0,    f"min std = {float(std.min())}")
    check(
        "physical_scales contains target key",
        f"{args.variable}@{args.z_level}" in scales,
        f"keys sample: {list(scales)[:3]}",
    )
    s_min, s_max = scales[f"{args.variable}@{args.z_level}"]
    check("physical scale has min < max", s_min < s_max, f"[{s_min:.3f}, {s_max:.3f}]")

    # ── 2. /prediction_field ─────────────────────────────────────────────────
    print(f"\n/inference/prediction_field?variable={args.variable}&z_level={args.z_level}")
    field = hit(
        f"{base}/inference/prediction_field",
        pred_time=args.pred_time,
        variable=args.variable,
        z_level=args.z_level,
    )
    pred_norm  = np.asarray(field["pred"],  dtype=np.float64)
    truth_norm = np.asarray(field["truth"], dtype=np.float64)
    c_idx      = int(field["component_idx"])

    check("pred length is N_grid",  pred_norm.shape  == (N_GRID,), f"got {pred_norm.shape}")
    check("truth length is N_grid", truth_norm.shape == (N_GRID,), f"got {truth_norm.shape}")
    check("component_idx in range", 0 <= c_idx < D_STATE, f"got {c_idx}")
    check("pred has no NaNs",  not np.isnan(pred_norm).any())
    check("truth has no NaNs", not np.isnan(truth_norm).any())

    # ── 3. Frontend can denormalize ──────────────────────────────────────────
    print("\nFrontend-side denormalization")
    pred_phys  = pred_norm  * std[c_idx] + mean[c_idx]
    truth_phys = truth_norm * std[c_idx] + mean[c_idx]
    check(
        "denormalized pred falls within physical_scale (with 10% margin)",
        (pred_phys.min()  >= s_min - 0.1 * (s_max - s_min)) and
        (pred_phys.max()  <= s_max + 0.1 * (s_max - s_min)),
        f"pred range [{pred_phys.min():.3f}, {pred_phys.max():.3f}] vs scale [{s_min:.3f}, {s_max:.3f}]",
    )
    check(
        "denormalized truth falls within physical_scale",
        (truth_phys.min() >= s_min - 1e-3) and (truth_phys.max() <= s_max + 1e-3),
        f"truth range [{truth_phys.min():.3f}, {truth_phys.max():.3f}]",
    )

    # ── 4. Cross-check against existing /prediction_errors ───────────────────
    print("\nCross-check vs /inference/prediction_errors (case 1: single var + level)")
    errors_ref = hit(
        f"{base}/inference/prediction_errors",
        pred_time=args.pred_time,
        variable=args.variable,
        z_level=args.z_level,
    )
    errors_ref = np.asarray(errors_ref, dtype=np.float64)
    errors_local = np.abs(pred_norm - truth_norm)

    check("errors length match", errors_ref.shape == errors_local.shape)
    max_diff = float(np.max(np.abs(errors_ref - errors_local)))
    check(
        "frontend-computed |pred-truth| ~= /prediction_errors output",
        max_diff < 1e-5,
        f"max abs diff = {max_diff:.2e}",
    )

    # ── 5. /prediction_errors?with_field=true returns bundled response ───────
    print("\n/inference/prediction_errors?with_field=true (single-component)")
    bundled = hit(
        f"{base}/inference/prediction_errors",
        pred_time=args.pred_time,
        variable=args.variable,
        z_level=args.z_level,
        with_field="true",
    )
    check("bundled response is a dict", isinstance(bundled, dict), f"got {type(bundled).__name__}")
    check("bundled.errors length == N_grid", len(bundled["errors"]) == N_GRID)
    check("bundled.pred length == N_grid",   bundled["pred"]  is not None and len(bundled["pred"])  == N_GRID)
    # truth is intentionally absent: nothing in the frontend reads it, and it was a third
    # of the payload on the hot path. /prediction_field (checked above) still returns it.
    check("bundled carries no truth key",    "truth" not in bundled)
    check("bundled.component_idx matches",   bundled["component_idx"] == c_idx)

    bundled_err   = np.asarray(bundled["errors"], dtype=np.float64)
    bundled_pred  = np.asarray(bundled["pred"],   dtype=np.float64)
    check(
        "bundled.errors ~= original /prediction_errors",
        float(np.max(np.abs(bundled_err - errors_ref))) < 1e-9,
    )
    check(
        "bundled.pred ~= /prediction_field.pred",
        float(np.max(np.abs(bundled_pred - pred_norm))) < 1e-9,
    )

    # ── 6. Aggregated case returns null pred ────────────────────────────────
    print("\n/inference/prediction_errors?with_field=true (aggregated case: variable=total)")
    bundled_agg = hit(
        f"{base}/inference/prediction_errors",
        pred_time=args.pred_time,
        variable="total",
        z_level="total",
        with_field="true",
    )
    check("aggregated response is dict",      isinstance(bundled_agg, dict))
    check("aggregated.errors populated",      len(bundled_agg["errors"]) == N_GRID)
    check("aggregated.pred is null",          bundled_agg["pred"]  is None)
    check("aggregated carries no truth key",  "truth" not in bundled_agg)
    check("aggregated.component_idx is null", bundled_agg["component_idx"] is None)

    # ── 7. Backwards compatibility — default still returns list[float] ───────
    print("\n/inference/prediction_errors (no with_field flag) — backwards compat")
    legacy = hit(
        f"{base}/inference/prediction_errors",
        pred_time=args.pred_time,
        variable=args.variable,
        z_level=args.z_level,
    )
    check("legacy response is a list",   isinstance(legacy, list))
    check("legacy length == N_grid",     len(legacy) == N_GRID)

    # ── 8. Cache reuse — second call should be near-instant ──────────────────
    print("\nInference cache reuse — second prediction_field call (different level)")
    import time
    t0 = time.perf_counter()
    other_level = "850" if args.z_level != "850" else "500"
    hit(
        f"{base}/inference/prediction_field",
        pred_time=args.pred_time,
        variable=args.variable,
        z_level=other_level,
    )
    dt = time.perf_counter() - t0
    check(
        "second call is cached (< 2s)",
        dt < 2.0,
        f"took {dt*1000:.0f}ms (expected <2000ms — first call may exceed this)",
    )

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()