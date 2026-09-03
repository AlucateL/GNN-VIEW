#!/usr/bin/env python3
"""IG step-count accuracy sweep — script form of test_ig_steps.ipynb.

Runs where the model runs. Peak ~14 GiB VRAM: it competes with live requests, so run it
when nobody is using the deployment.
"""

# # How many Riemann steps does Integrated Gradients need?
# 
# `_IG_STEPS` is currently **4**, chosen for latency when the ocean model ran on CPU. On GPU the
# picture changed: compute is ~0.37 s per step against ~3 s of fixed overhead (zarr read, 8.7 MB
# of JSON, network), so step count barely moves wall time. This measures what accuracy we are
# buying, so the constant is set from evidence rather than habit.
# 
# **The metric is completeness.** IG's defining property is that attributions sum to the
# prediction shift:
# 
# $$\sum_i \mathrm{IG}_i(x) = f(x) - f(x')$$
# 
# with $x'$ the baseline (zeros in normalised space = the climatological mean). A finite Riemann
# sum only approximates this, and the residual is the approximation error — directly comparable
# across step counts, with no reference run needed.
# 
# A second view is **convergence**: how much the attribution map still moves between step counts.
# Completeness can look acceptable while individual attributions are still shifting.
# 
# > **Memory.** Runs a forward+backward over 675,219 nodes and 3.4M edges; peak ~14 GiB. Step
# > count does not change peak (steps are sequential, `batch_size=1`) — only runtime. Needs a
# > GPU with ~16 GiB or a machine with comparable RAM; an 8 GB laptop cannot run this.

import sys, time
from pathlib import Path
import numpy as np
import torch

for _root in [Path("/usr/src/app")] + [c / sub
        for c in (Path.cwd(), *Path.cwd().parents)
        for sub in (Path("."), Path("src"), Path("backend") / "src")]:
    if (_root / "services" / "ocean.py").exists():
        sys.path.insert(0, str(_root)); print("backend src:", _root); break
else:
    raise RuntimeError("could not locate services/ocean.py")

from services.ocean import prepare_OceanBench, load_ocean_model, get_ocean_step_inputs, OCEAN_DEVICE
from services.integrated_gradients import compute_integrated_gradients

DEVICE = OCEAN_DEVICE
print("device:", DEVICE)

# ## Setup
# 
# One target node and channel. Pick a node in open ocean — a coastal or shallow node has most of
# its input zeroed, so the prediction shift is tiny and completeness is dominated by noise.

cache = prepare_OceanBench(device=DEVICE)
model = load_ocean_model(device=DEVICE)

TARGET_VAR  = "thetao_47m"
TARGET_NODE = 366499
PRED_TIME   = cache["forecast_only_times"][0]

names = cache["state_features"]
target_idx = names.index(TARGET_VAR)
valid = cache["valid_mask"]
print(f"target      : {TARGET_VAR} (channel {target_idx}) at node {TARGET_NODE}")
print(f"pred_time   : {PRED_TIME}")
print(f"node valid? : {bool(valid[TARGET_NODE, target_idx])}   <- must be True for a meaningful shift")

# ## The quantity IG must reproduce
# 
# `f(x) − f(x′)`: the model output at the target, minus the same with both state inputs replaced
# by the baseline. Forcing is held at its real value in both — it is not an attributed input, so
# it is not part of the shift.

prev, prev_prev, forcing = get_ocean_step_inputs(cache, PRED_TIME, device=DEVICE)
prev, prev_prev, forcing = prev.detach(), prev_prev.detach(), forcing.detach()

with torch.no_grad():
    f_x  = model(prev, prev_prev, forcing)[0, TARGET_NODE, target_idx].item()
    f_xp = model(torch.zeros_like(prev), torch.zeros_like(prev_prev), forcing)[0, TARGET_NODE, target_idx].item()

shift = f_x - f_xp
print(f"f(x)        = {f_x:+.6f}")
print(f"f(x')       = {f_xp:+.6f}")
print(f"shift       = {shift:+.6f}   <- IG must sum to this")
assert abs(shift) > 1e-3, "prediction shift is ~0; pick a different node or channel"

# ## Sweep
# 
# `sources=None` returns every feature, which is required — completeness only holds over the
# *full* attribution, not a slice. Runs ascending so an out-of-memory failure still leaves the
# cheaper results.

STEP_COUNTS = [4, 8, 16, 32, 64]
results = {}

for k in STEP_COUNTS:
    torch.cuda.empty_cache() if DEVICE.startswith("cuda") else None
    t0 = time.time()
    out = compute_integrated_gradients(
        model=model, era5_cache=cache, pred_time=PRED_TIME,
        target_variable=TARGET_VAR, target_level=None, target_node=TARGET_NODE,
        sources=None, device=DEVICE, steps=k, batch_size=1,
    )
    elapsed = time.time() - t0

    total = sum(np.asarray(v, dtype=np.float64).sum()
                for blk in ("attributions_prev", "attributions_prev_prev")
                for v in out[blk].values())
    gap = total - shift
    results[k] = {
        "sum": total, "gap": gap, "rel_gap": abs(gap) / abs(shift), "secs": elapsed,
        "prev": {kk: np.asarray(vv, dtype=np.float64) for kk, vv in out["attributions_prev"].items()},
    }
    print(f"steps={k:>3}  sum={total:+.6f}  gap={gap:+.3e}  rel={100*abs(gap)/abs(shift):6.2f}%  {elapsed:5.1f}s")

# ## Completeness error vs cost
# 
# `rel_gap` is the fraction of the prediction shift the approximation fails to account for. It
# should fall roughly as 1/steps for the midpoint rule.

print(f"{'steps':>6} {'rel. completeness error':>24} {'runtime':>9} {'error x vs 64':>14}")
ref_err = results[max(STEP_COUNTS)]["rel_gap"]
for k in STEP_COUNTS:
    r = results[k]
    print(f"{k:>6} {100*r['rel_gap']:>23.3f}% {r['secs']:>8.1f}s {r['rel_gap']/ref_err:>13.1f}x")

print()
for a, b in zip(STEP_COUNTS, STEP_COUNTS[1:]):
    ratio = results[a]["rel_gap"] / max(results[b]["rel_gap"], 1e-12)
    print(f"  {a:>2} -> {b:>2}: error improves {ratio:.1f}x for {results[b]['secs']-results[a]['secs']:+.1f}s")

# ## Convergence of the map itself
# 
# Completeness is a single scalar and can look fine while individual attributions still move.
# This compares each run against the 64-step reference over the attributed field.

ref = results[max(STEP_COUNTS)]["prev"]
print(f"{'steps':>6} {'rel. L2 vs 64':>15} {'cosine':>9} {'max abs diff':>14}")
for k in STEP_COUNTS:
    a = np.concatenate([results[k]["prev"][key].ravel() for key in sorted(ref)])
    b = np.concatenate([ref[key].ravel() for key in sorted(ref)])
    l2 = np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)
    cos = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    print(f"{k:>6} {100*l2:>14.3f}% {cos:>9.5f} {np.abs(a-b).max():>14.3e}")

# ## Verdict
# 
# Read off two things:
# 
# 1. the smallest step count whose **completeness error** is acceptable (a few percent is the
#    usual bar for IG to be quotable), and
# 2. the point where **rel. L2 vs 64** stops improving materially — beyond it, extra steps buy
#    numerical precision nobody can see on a map.
# 
# Set `_IG_STEPS` in `routes/integrated_gradients.py` to that value. The cache key already
# includes it (`ig_k{steps}`), so changing it cannot serve stale results at the old count.
# 
# Note the runtime column understates the real cost: it excludes the ~3 s of per-request overhead
# (zarr read, 8.7 MB JSON, network) that dominates the user-visible latency.

best = min((k for k in STEP_COUNTS if results[k]["rel_gap"] < 0.05), default=max(STEP_COUNTS))
print(f"smallest step count with completeness error < 5%: {best}")
print(f"  its runtime: {results[best]['secs']:.1f}s vs {results[4]['secs']:.1f}s at the current 4")
print(f"  current setting (4) completeness error: {100*results[4]['rel_gap']:.2f}%")


# ── Are the multi-GPU / compiler packages ever actually used? ─────────────────
#
# torch's CUDA wheel hard-depends on nvidia-nccl (290 MB, multi-GPU collectives),
# nvidia-nvshmem (139 MB, cross-GPU memory) and triton (197 MB, the torch.compile kernel
# compiler). We run one GPU and a TorchScript model, so none should be reachable — and
# dropping ~600 MB may get the image layer under whatever Harbor rejects, which is what
# currently breaks build caching.
#
# Checked by observation rather than inference: if a module was never imported after a full
# forward + backward on both domains, removing it cannot break those paths. Note the
# updateStrategy is Recreate, so a bad image means downtime, not a silent rollback — worth
# being sure before touching the Dockerfile.
import sys

WATCH = ("triton", "nccl", "nvshmem")

def _loaded():
    return sorted({m.split(".")[0] for m in sys.modules if any(k in m.lower() for k in WATCH)})

print("\n" + "=" * 70)
print("after ocean load + forward + backward:", _loaded() or "none of", WATCH)

# ERA5 goes through a different loader (load_model, .pt/.ckpt) than the ocean path, so the
# sweep above never exercises it. Removing a package the weather models need would not show
# up otherwise.
try:
    from services.inference import PRESET_MODELS, load_model, preload_era5_tensors, FIELDS_ZARR_PATH, DEVICE
    from services.inference import get_era5_step_inputs, get_available_pred_timestamps
    key, path = next((k, p) for k, p in PRESET_MODELS.items() if p.exists())
    era5_model = load_model(path=path, device=DEVICE)
    era5_cache = preload_era5_tensors(FIELDS_ZARR_PATH, device=DEVICE, lazy=True)
    ts = get_available_pred_timestamps(era5_cache, as_str=True)
    p1, p2, fc = get_era5_step_inputs(era5_cache, ts[0], DEVICE)
    with torch.no_grad():
        era5_model(p1, p2, fc)
    print(f"after ERA5 load + forward ({key}):   ", _loaded() or "none of", WATCH)
except Exception as e:
    print("ERA5 probe skipped:", type(e).__name__, e)

print("=" * 70)
print("empty on both lines => safe to pip uninstall nvidia-nccl-cu12 nvidia-nvshmem-cu12 triton")
print("anything listed     => that package is genuinely in use; leave it")
