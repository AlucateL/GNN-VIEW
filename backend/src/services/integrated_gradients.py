"""
Integrated Gradients explainability for the GNN weather model.

Per-input-node attribution: how much does each chosen *source feature*
(variable, optional pressure level) at each grid node contribute to a
chosen *target feature* in the model output, summed over all output nodes?

Both `prev` (state at t-1) and `prev_prev` (state at t-2) are integrated
jointly along a baseline-to-input path; `forcing` is held fixed.
Baseline = zeros in normalized space (= climatological mean per feature).
"""

import time
import tracemalloc
from typing import Callable

import torch

from services.inference import (
    DEVICE,
    get_era5_step_inputs,
    run_inference,
)


class IGCancelled(Exception):
    """Raised when an IG compute is cancelled mid-flight (e.g. client disconnected).

    Lets the route abort early and skip caching a partial result.
    """
    pass


def _feature_key(variable: str, level: int | None) -> str:
    """Canonical name for a feature: 'temperature@500' or '2m_temperature'."""
    return f"{variable}@{level}" if level is not None else variable


def _resolve_feature_index(
    variable: str,
    level: int | None,
    atm_vars: list[str],
    surf_vars: list[str],
    levels: list[int],
) -> int:
    """Map (variable, level) to the column index used by _stack_state."""
    n_levels = len(levels)
    if variable in atm_vars:
        if level is None:
            raise ValueError(
                f"Atmospheric variable '{variable}' requires a pressure level "
                f"(one of {levels})."
            )
        if level not in levels:
            raise ValueError(f"Level {level} not in available levels {levels}.")
        return atm_vars.index(variable) * n_levels + levels.index(level)
    if variable in surf_vars:
        return len(atm_vars) * n_levels + surf_vars.index(variable)
    raise ValueError(
        f"Unknown variable '{variable}'. "
        f"Atmospheric: {atm_vars}. Surface: {surf_vars}."
    )


def _enumerate_all_features(
    atm_vars: list[str],
    surf_vars: list[str],
    levels: list[int],
) -> list[tuple[str, int | None]]:
    """All (variable, level) pairs in the canonical _stack_state order."""
    out: list[tuple[str, int | None]] = []
    for v in atm_vars:
        for lv in levels:
            out.append((v, lv))
    for v in surf_vars:
        out.append((v, None))
    return out


def _resolve_target(cache: dict, variable: str, level: int | None) -> int:
    """
    Map (variable, level) to a channel index, for either domain.

    ERA5 stacks atmospheric variables x pressure levels first and surface variables last, so
    the index is arithmetic. The ocean datastore instead carries 27 fully-qualified names in
    its own order (3 surface fields, then 4 variables x 6 depths), so the name is looked up
    directly -- computing an index from the ERA5 layout would silently select a different
    variable, since both orderings are dense and neither errors.
    """
    if cache.get("_ocean"):
        names = cache["state_features"]
        key = variable if level is None else f"{variable}_{level}m"
        if key not in names:
            raise ValueError(f"Unknown ocean feature '{key}'. Available: {names}")
        return names.index(key)
    return _resolve_feature_index(
        variable, level, cache["atm_vars"], cache["surf_vars"], cache["levels"]
    )


def _all_features(cache: dict) -> list[tuple[str, int | None]]:
    """Every (variable, level) pair in the cache's own channel order."""
    if cache.get("_ocean"):
        return [(n, None) for n in cache["state_features"]]
    return _enumerate_all_features(cache["atm_vars"], cache["surf_vars"], cache["levels"])


def _step_inputs(cache: dict, pred_time, device):
    """
    Build (prev, prev_prev, forcing) for either domain.

    IG and IxG call this rather than get_era5_step_inputs directly, so an ocean cache routes
    to the ocean builder. The returned tuple has the same shape contract either way; for the
    ocean the forcing block is the day-of-year and atmospheric windows concatenated, which
    the model adapter splits again.
    """
    if cache.get("_ocean"):
        from services.ocean import get_ocean_step_inputs  # noqa: PLC0415

        return get_ocean_step_inputs(cache, pred_time=pred_time, device=device)
    return get_era5_step_inputs(era5_cache=cache, pred_time=pred_time, device=device)


def _forward(model: torch.nn.Module, prev, prev_prev, forcing) -> torch.Tensor:
    """Dispatch to TorchScript / predict_step / __call__, returning the prediction tensor."""
    if isinstance(model, torch.jit.ScriptModule):
        out = model(prev, prev_prev, forcing)
    elif hasattr(model, "predict_step"):
        out = model.predict_step(prev, prev_prev, forcing)
    else:
        out = model(prev, prev_prev, forcing)
    return out[0] if isinstance(out, tuple) else out


def _mask_absent(sliced: "np.ndarray", era5_cache: dict, source_indices: "torch.Tensor") -> "np.ndarray":
    """NaN out cells the source channel has no data for. No-op outside the ocean domain.

    Ocean channels die off with depth: a cell shallower than 92 m carries nothing for so_92m,
    and its input was zeroed to the training mean. Attribution there is input x gradient = 0,
    which the diverging colormap renders as the centre of its scale -- white -- making "no data
    here" look identical to "no influence here". orjson writes NaN as JSON null, so the frontend
    can colour it as absent. Same treatment the error map and prediction field already get.
    """
    valid = era5_cache.get("valid_mask")
    if valid is None:
        return sliced
    m = valid[:, source_indices].detach().cpu().numpy()   # (N_grid, n_sources)
    out = sliced.astype("float32", copy=True)
    out[~m] = float("nan")
    return out


def compute_integrated_gradients(
    model: torch.nn.Module,
    era5_cache: dict,
    pred_time: str,
    target_variable: str,
    target_level: int | None = None,
    target_node: int | None = None,
    sources: list[tuple[str, int | None]] | None = None,
    steps: int = 32,
    batch_size: int | None = None,
    device: str = DEVICE,
    should_continue: Callable[[], bool] | None = None,
) -> dict:
    """
    Compute Integrated Gradients attributions for a single prediction step.

    Returns one attribution scalar per grid node *per requested source feature*,
    in normalized units.

    Target scalar:
      - target_node=None: sum of the target feature over all output grid nodes
        (global integral).
      - target_node=k:    pred[k, target_idx] — attributes how every input
        feature at every input node contributed to the prediction *at this
        specific output node*.

    Batching:
      Interpolation steps are processed in mini-batches of size `batch_size`
      (default: all steps in one forward pass).  Each mini-batch runs one
      combined forward + backward pass, then gradients are accumulated.
      batch_size=1 reproduces the original sequential behaviour.

    Args:
        model:           Loaded GNN.
        era5_cache:      Result of services.inference.preload_era5_tensors.
        pred_time:       Prediction timestamp (ISO string).
        target_variable: Output variable to explain.
        target_level:    Pressure level (hPa) — required iff target_variable is atmospheric.
        target_node:     Optional grid-node index in [0, N_grid). When set, the
                         target scalar is the prediction at this single node;
                         otherwise it's summed across all nodes.
        sources:         List of (variable, level) pairs to attribute. None or empty
                         list means "all features".
        steps:           Total Riemann-sum steps (midpoint rule). Controls IG accuracy.
        batch_size:      Steps per forward pass. None or >= steps → single batched call.
                         1 → sequential (original). Values in between → mini-batches.
        device:

    Returns dict with:
        attributions_prev:      {feature_key: list[float] of length N_grid}
        attributions_prev_prev: {feature_key: list[float] of length N_grid}
        sources: metadata for each returned feature (in result order)
        target_node: echoed back (None for global-sum mode).
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    # Clamp batch_size to [1, steps]
    if batch_size is None or batch_size >= steps:
        batch_size = steps
    batch_size = max(1, batch_size)

    # The scripted ocean model is batch-size-1 only. torch.jit.trace records shapes as
    # constants, and it was traced with a single example, so the node count 675219 is baked
    # into the clamping reparameterisation (ar_model.py:568 get_clamped_new_state). Any
    # other batch size raises a TorchScript shape error from deep inside the graph, which
    # gives no hint about the cause. Forcing 1 costs nothing in practice: a single ocean
    # backward already needs ~9 GB, so batching alpha steps would exhaust memory anyway.
    # Re-exporting with torch.jit.script instead of trace would lift this.
    if era5_cache.get("_ocean") and batch_size != 1:
        batch_size = 1

    target_idx = _resolve_target(era5_cache, target_variable, target_level)

    if not sources:
        source_specs = _all_features(era5_cache)
    else:
        source_specs = list(sources)

    source_meta = [
        {
            "variable": v,
            "level": lv,
            "feature_index": _resolve_target(era5_cache, v, lv),
            "key": _feature_key(v, lv),
        }
        for v, lv in source_specs
    ]
    source_indices = torch.tensor(
        [m["feature_index"] for m in source_meta], dtype=torch.long, device=device
    )

    prev, prev_prev, forcing = _step_inputs(era5_cache, pred_time, device)

    prev      = prev.detach()       # (1, N_grid, d_state)
    prev_prev = prev_prev.detach()
    forcing   = forcing.detach()    # (1, N_grid, 15)

    n_grid  = prev.shape[1]
    d_state = prev.shape[2]

    if target_node is not None and not 0 <= target_node < n_grid:
        raise ValueError(
            f"target_node={target_node} out of range; valid range is [0, {n_grid - 1}]"
        )

    # Baselines = zeros in normalized space (= climatological mean per feature).
    baseline_prev      = torch.zeros_like(prev)
    baseline_prev_prev = torch.zeros_like(prev_prev)
    delta_prev      = prev      - baseline_prev      # (1, N_grid, d_state)
    delta_prev_prev = prev_prev - baseline_prev_prev

    # Midpoint Riemann rule: alpha_k = (k - 0.5) / steps for k in 1..steps
    alphas = (torch.arange(steps, device=device, dtype=prev.dtype) + 0.5) / steps

    # Accumulators over all steps — shape (N_grid, d_state)
    accumulated_grad_prev      = torch.zeros(n_grid, d_state, dtype=prev.dtype, device=device)
    accumulated_grad_prev_prev = torch.zeros(n_grid, d_state, dtype=prev.dtype, device=device)

    model.eval()
    _mem_stage(device, "1 inputs built")
    # Release cached-but-unused blocks before the accumulation loop. IG allocates and frees a
    # full activation set per step, which fragments the pool: the failure this guards against
    # reported 3.45 GiB reserved-but-unallocated while a 2.16 GiB allocation failed, inside a
    # 16 GiB cap. Cheap (microseconds) next to a backward pass over 3.4M edges.
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    with torch.enable_grad():
        for start in range(0, steps, batch_size):
            # Cancellation point: between mini-batches the route can signal abort
            # (typically because the client disconnected). Raising here drops the
            # partial accumulator instead of returning incomplete attributions.
            if should_continue is not None and not should_continue():
                raise IGCancelled(f"cancelled after {start}/{steps} steps")
            chunk = alphas[start : start + batch_size]   # (chunk_size,)
            alpha_col = chunk[:, None, None]             # (chunk_size, 1, 1)

            # (chunk_size, N_grid, d_state) via broadcasting over (1, N_grid, d_state)
            x_prev = (baseline_prev + alpha_col * delta_prev).detach().requires_grad_(True)
            x_prev_prev = (baseline_prev_prev + alpha_col * delta_prev_prev).detach().requires_grad_(True)
            forcing_batch = forcing.expand(len(chunk), -1, -1).detach()

            pred = _forward(model, x_prev, x_prev_prev, forcing_batch)
            if start == 0:
                _mem_stage(device, "2 after forward (step 1)")
            # pred: (chunk_size, N_grid, d_state)

            if target_node is None:
                target_scalar = pred[:, :, target_idx].sum()
            else:
                target_scalar = pred[:, target_node, target_idx].sum()

            grads = torch.autograd.grad(target_scalar, [x_prev, x_prev_prev])
            # grads[i]: (chunk_size, N_grid, d_state) — sum over chunk to accumulate
            if start == 0:
                _mem_stage(device, "3 after backward (step 1)")
            accumulated_grad_prev      = accumulated_grad_prev      + grads[0].detach().sum(dim=0)
            accumulated_grad_prev_prev = accumulated_grad_prev_prev + grads[1].detach().sum(dim=0)

    # IG = delta * mean_grad  (N_grid, d_state)
    ig_prev      = delta_prev[0]      * (accumulated_grad_prev      / steps)
    ig_prev_prev = delta_prev_prev[0] * (accumulated_grad_prev_prev / steps)

    sliced_prev      = ig_prev     .index_select(dim=-1, index=source_indices).detach().cpu().numpy()
    sliced_prev_prev = ig_prev_prev.index_select(dim=-1, index=source_indices).detach().cpu().numpy()
    sliced_prev      = _mask_absent(sliced_prev,      era5_cache, source_indices)
    sliced_prev_prev = _mask_absent(sliced_prev_prev, era5_cache, source_indices)

    attributions_prev      = {meta["key"]: sliced_prev[:, i]      for i, meta in enumerate(source_meta)}
    attributions_prev_prev = {meta["key"]: sliced_prev_prev[:, i] for i, meta in enumerate(source_meta)}

    return {
        "attributions_prev":      attributions_prev,
        "attributions_prev_prev": attributions_prev_prev,
        "n_nodes": n_grid,
        "target": {
            "variable": target_variable,
            "level": target_level,
            "feature_index": target_idx,
            "node": target_node,
        },
        "sources": source_meta,
        "steps": steps,
        "batch_size": batch_size,
        "pred_time": pred_time,
    }


# ---------------------------------------------------------------------------
# Input × Gradient attribution
# ---------------------------------------------------------------------------

def _mem_stage(device: str, label: str) -> None:
    """Log allocator state at a named point in a pass, to stdout.

    A pass is the only place the memory actually goes, and every external probe reports one
    number for the whole thing -- enough to see that a pass cost 13 GiB or hit the ceiling,
    never enough to see *where*. These four points split it into inputs / forward / backward /
    slice, so a passing and a failing run can be compared stage by stage.

    No-op off CUDA. Cheap: two counter reads and a print, against a multi-second pass.
    """
    if not str(device).startswith("cuda"):
        return
    a = torch.cuda.memory_allocated() / 1024 ** 3
    r = torch.cuda.memory_reserved() / 1024 ** 3
    p = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"[mem] {label:<22} allocated={a:6.2f} GiB  reserved={r:6.2f} GiB  peak={p:6.2f} GiB",
          flush=True)


def compute_input_x_gradients(
    model: torch.nn.Module,
    era5_cache: dict,
    pred_time: str,
    target_variable: str,
    target_level: int | None = None,
    target_node: int | None = None,
    sources: list[tuple[str, int | None]] | None = None,
    device: str = DEVICE,
) -> dict:
    """
    Compute Input × Gradient attributions for a single prediction step.

    attr = input * ∂output/∂input  (evaluated at the actual input, not a baseline path).

    Faster than IG (one forward + one backward pass) but does not satisfy the
    completeness axiom — attributions do not sum to the prediction difference.
    Return format is identical to compute_integrated_gradients so the route,
    cache, and slice logic require no changes.

    Args:
        model, era5_cache, pred_time, target_variable, target_level,
        target_node, sources, device: same semantics as compute_integrated_gradients.
    """
    target_idx = _resolve_target(era5_cache, target_variable, target_level)

    source_specs = _all_features(era5_cache) if not sources else list(sources)
    source_meta  = [
        {
            "variable": v,
            "level": lv,
            "feature_index": _resolve_target(era5_cache, v, lv),
            "key": _feature_key(v, lv),
        }
        for v, lv in source_specs
    ]
    source_indices = torch.tensor(
        [m["feature_index"] for m in source_meta], dtype=torch.long, device=device
    )

    prev, prev_prev, forcing = _step_inputs(era5_cache, pred_time, device)

    n_grid = prev.shape[1]
    if target_node is not None and not 0 <= target_node < n_grid:
        raise ValueError(
            f"target_node={target_node} out of range; valid range is [0, {n_grid - 1}]"
        )

    x_prev      = prev.detach().requires_grad_(True)
    x_prev_prev = prev_prev.detach().requires_grad_(True)

    model.eval()
    # Same reclaim IG does. Without it IxG inherits whatever pool state the previous request
    # left: the caching allocator grows to fill its limit, so after a few passes reserved sits
    # near the cap with little of it live, and a fresh 14 GiB pass cannot find room.
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    _mem_stage(device, "1 inputs built")

    with torch.enable_grad():
        pred = _forward(model, x_prev, x_prev_prev, forcing.detach())
        _mem_stage(device, "2 after forward")
        if target_node is None:
            target_scalar = pred[0, :, target_idx].sum()
        else:
            target_scalar = pred[0, target_node, target_idx]
        grads = torch.autograd.grad(target_scalar, [x_prev, x_prev_prev])
    _mem_stage(device, "3 after backward")

    # attr = input * gradient, both evaluated at the actual input (no baseline)
    ixg_prev      = prev[0]      * grads[0][0]   # (N_grid, d_state)
    ixg_prev_prev = prev_prev[0] * grads[1][0]

    sliced_prev      = ixg_prev     .index_select(-1, source_indices).detach().cpu().numpy()
    sliced_prev_prev = ixg_prev_prev.index_select(-1, source_indices).detach().cpu().numpy()
    sliced_prev      = _mask_absent(sliced_prev,      era5_cache, source_indices)
    sliced_prev_prev = _mask_absent(sliced_prev_prev, era5_cache, source_indices)
    _mem_stage(device, "4 after slice")

    return {
        "attributions_prev":      {m["key"]: sliced_prev[:, i]      for i, m in enumerate(source_meta)},
        "attributions_prev_prev": {m["key"]: sliced_prev_prev[:, i] for i, m in enumerate(source_meta)},
        "n_nodes":  n_grid,
        "target":   {"variable": target_variable, "level": target_level, "feature_index": target_idx, "node": target_node},
        "sources":  source_meta,
        "steps":    1,
        "pred_time": pred_time,
    }


# ---------------------------------------------------------------------------
# Profiling helpers
# ---------------------------------------------------------------------------

def profile_batch(
    model: torch.nn.Module,
    era5_cache: dict,
    pred_time: str,
    target_variable: str,
    target_level: int | None = None,
    target_node: int | None = None,
    batch_sizes: list[int] | tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    steps: int = 32,
    device: str = DEVICE,
) -> list[dict]:
    """
    Profile batched IG for a range of mini-batch sizes.

    For each batch_size, runs compute_integrated_gradients with `steps` total
    Riemann steps and `batch_size` steps per forward pass, then records
    wall-clock time and peak RSS memory from tracemalloc.

    Args:
        model:           Loaded GNN.
        era5_cache:      Result of preload_era5_tensors.
        pred_time:       Prediction timestamp (ISO string).
        target_variable: Output variable to explain.
        target_level:    Pressure level (hPa) — required for atmospheric variables.
        target_node:     Grid-node index, or None for global-sum target.
        batch_sizes:     Mini-batch sizes to benchmark. Each must be in [1, steps].
        steps:           Total IG interpolation steps (controls accuracy, not batch size).
        device:

    Returns:
        List of dicts, one per batch_size:
          {batch_size, time_s, peak_ram_mb[, peak_vram_mb]}
    """
    results: list[dict] = []

    for bs in batch_sizes:
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        tracemalloc.start()
        t0 = time.perf_counter()

        compute_integrated_gradients(
            model=model,
            era5_cache=era5_cache,
            pred_time=pred_time,
            target_variable=target_variable,
            target_level=target_level,
            target_node=target_node,
            steps=steps,
            batch_size=bs,
            device=device,
        )

        elapsed = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        row: dict = {
            "batch_size": bs,
            "time_s": round(elapsed, 3),
            "peak_ram_mb": round(peak_bytes / 1e6, 2),
        }
        if device.startswith("cuda") and torch.cuda.is_available():
            row["peak_vram_mb"] = round(
                torch.cuda.max_memory_allocated(device) / 1e6, 2
            )

        label = f"  batch={bs:3d}: {elapsed:.3f}s  RAM={row['peak_ram_mb']:.1f} MB"
        if "peak_vram_mb" in row:
            label += f"  VRAM={row['peak_vram_mb']:.1f} MB"
        print(label)

        results.append(row)

    return results


def profile_inference_batch(
    model: torch.nn.Module,
    era5_cache: dict,
    pred_time: str,
    batch_sizes: list[int] | tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    device: str = DEVICE,
    warmup: int = 1,
) -> list[dict]:
    """
    Profile run_inference throughput for a range of batch sizes.

    Each batch is constructed by repeating the same ERA5 timestep `batch_size`
    times — this isolates model throughput from data-loading cost and is
    representative of real batched IG workloads.

    Args:
        model:       Loaded GNN.
        era5_cache:  Result of preload_era5_tensors.
        pred_time:   Prediction timestamp (ISO string).
        batch_sizes: Batch sizes to benchmark.
        device:
        warmup:      Number of dry-run forward passes before timing (warms JIT/CUDA).

    Returns:
        List of dicts, one per batch_size:
          {batch_size, time_s, peak_ram_mb[, peak_vram_mb]}
    """
    # Warm up with the smallest batch to amortize JIT compilation
    for _ in range(warmup):
        run_inference(
            model=model,
            era5_cache=era5_cache,
            pred_time=pred_time,
            device=device,
            batch_size=batch_sizes[0],
        )

    results: list[dict] = []

    for bs in batch_sizes:
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        tracemalloc.start()
        t0 = time.perf_counter()

        run_inference(
            model=model,
            era5_cache=era5_cache,
            pred_time=pred_time,
            device=device,
            batch_size=bs,
        )

        elapsed = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        row: dict = {
            "batch_size": bs,
            "time_s": round(elapsed, 3),
            "peak_ram_mb": round(peak_bytes / 1e6, 2),
        }
        if device.startswith("cuda") and torch.cuda.is_available():
            row["peak_vram_mb"] = round(
                torch.cuda.max_memory_allocated(device) / 1e6, 2
            )

        label = f"  batch={bs:3d}: {elapsed:.3f}s  RAM={row['peak_ram_mb']:.1f} MB"
        if "peak_vram_mb" in row:
            label += f"  VRAM={row['peak_vram_mb']:.1f} MB"
        print(label)

        results.append(row)

    return results
