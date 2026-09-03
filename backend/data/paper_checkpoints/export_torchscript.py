"""
Export a neural-lam checkpoint to a TorchScript .pt file.

Wraps the upstream export_torchscript.py logic with fixes needed when running
against paper checkpoints without the original training dataset:

  1. Hyperparameter reading: PL's save_hyperparameters() stores the args
     namespace under hp["args"], not flat in hp. The upstream script does
     hp.get("hidden_dim", ...) which always falls through to the default.

  2. Graph name resolution: checkpoints may store a graph name (e.g.
     "global_ms4") that doesn't match any local directory. Falls back to
     whatever graph exists under graphs/.

  3. TorchScript tracing: tracing a LightningModule as an nn.Module triggers
     PL's trainer property during module traversal. Fixed by tracing a plain
     function instead, which skips module traversal entirely.

Usage (run from project root):
    PYTHONPATH=models/neural-lam python data/paper_checkpoints/export_torchscript.py \
        --checkpoint data/paper_checkpoints/graphcast.ckpt \
        --model graphcast
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# Project root is two levels up from this script (data/paper_checkpoints/)
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
_DATA_ROOT    = _PROJECT_ROOT / "data"
_STATIC_FILES_PATH = _DATA_ROOT / "era5_datasets" / "static"
# Passed as args.dataset: os.path.join("data", <abs_path>, "static") → <abs_path>/static
_DEFAULT_DATASET = str(_STATIC_FILES_PATH.parent)

# Per-checkpoint inference overrides (flags not reliably stored in ckpt hparams).
# Canonical source: model_configs.json in the same directory as this script.
_MODEL_CONFIGS_PATH = _SCRIPT_DIR / "model_configs.json"

def _load_model_configs() -> dict:
    if _MODEL_CONFIGS_PATH.exists():
        with open(_MODEL_CONFIGS_PATH) as f:
            cfg = json.load(f)
        # Strip comment keys
        return {k: v for k, v in cfg.items() if not k.startswith("_")}
    return {}

_MODEL_CONFIGS = _load_model_configs()

from neural_lam import constants
from neural_lam.models.base_hi_graph_model import BaseHiGraphModel
from neural_lam.models.graph_efm import GraphEFM
from neural_lam.models.graph_fm import GraphFM
from neural_lam.models.graphcast import GraphCast

MODELS = {
    "graphcast": GraphCast,
    "graph_fm": GraphFM,
    "graph_efm": GraphEFM,
}


class ModelArgs:
    """Plain namespace for model constructor args — module-level so it's picklable."""
    pass


def _read_hparams(ckpt_path: str) -> object:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    # PL stores hparams under hp["args"] when save_hyperparameters() is called
    # with an argparse Namespace; flat dict otherwise.
    return hp.get("args", hp)


def _get(hp, key, default):
    if hasattr(hp, key):
        return getattr(hp, key)
    if isinstance(hp, dict):
        return hp.get(key, default)
    return default


def build_args(model_name: str, ckpt_path: str, dataset: str) -> ModelArgs:
    hp      = _read_hparams(ckpt_path)
    cfg     = _MODEL_CONFIGS.get(model_name, {})  # overrides from model_configs.json

    def _cfg_or_hp(key, default):
        """Return cfg[key] if present, else hp value, else default."""
        if key in cfg:
            return cfg[key]
        return _get(hp, key, default)

    args = ModelArgs()
    args.model                    = model_name
    args.dataset                  = dataset  # always local — ckpt value won't exist here
    args.graph                    = _cfg_or_hp("graph", "global_multiscale")
    args.hidden_dim               = _cfg_or_hp("hidden_dim", 64)
    args.hidden_layers            = _cfg_or_hp("hidden_layers", 1)
    args.processor_layers         = _cfg_or_hp("processor_layers", 4)
    args.prior_processor_layers   = _cfg_or_hp("prior_processor_layers", 2)
    args.encoder_processor_layers = _cfg_or_hp("encoder_processor_layers", 2)
    args.mesh_aggr                = _cfg_or_hp("mesh_aggr", "sum")
    args.ar_steps                 = _cfg_or_hp("ar_steps", 1)
    args.lr                       = _cfg_or_hp("lr", 1e-3)
    args.loss                     = _cfg_or_hp("loss", "mse")
    args.step_length              = _cfg_or_hp("step_length", 6)
    args.num_past_forcing_steps   = _cfg_or_hp("num_past_forcing_steps", 1)
    args.num_future_forcing_steps = _cfg_or_hp("num_future_forcing_steps", 1)
    args.eval_leads               = _cfg_or_hp("eval_leads", 1)
    args.vertical_propnets        = _cfg_or_hp("vertical_propnets", False)
    args.output_std               = _cfg_or_hp("output_std", False)
    args.ensemble_size            = _cfg_or_hp("ensemble_size", 1)
    args.eval                     = None
    args.n_example_pred           = 0
    args.batch_size               = 1
    args.restore_opt              = False
    args.load                     = None
    args.save_forecasts           = False
    args.n_workers                = 0
    args.seed                     = 42

    if cfg:
        print(f"  model_configs.json overrides applied: { {k: cfg[k] for k in cfg if k not in ('checkpoint', 'model_class')} }")
    return args


def _needs_hierarchical_graph(model_cls, original_graph_name: str) -> bool:
    """
    Return True when the model requires a hierarchical graph.

    BaseHiGraphModel subclasses (e.g. GraphFM) always need a hierarchical
    graph.  For dynamic models like GraphEFM the graph name stored in the
    checkpoint is the best signal: if it contains "hierarchical" or "hi_" it
    was trained on a hierarchical graph and must use one here too.
    """
    if issubclass(model_cls, BaseHiGraphModel):
        return True
    name = original_graph_name.lower()
    return "hierarchical" in name or name.startswith("hi_")


def resolve_graph(args: ModelArgs, model_cls) -> None:
    """
    Replace args.graph with a graph directory that actually exists locally.
    The checkpoint may store a training-time name (e.g. "global_ms4") that
    isn't present in this repo.

    Fallback order is model-aware: hierarchical models prefer
    global_hierarchical; flat models prefer global_multiscale.
    """
    if Path("graphs", args.graph).exists():
        return
    original = args.graph
    if _needs_hierarchical_graph(model_cls, original):
        fallbacks = ("global_hierarchical", "global_multiscale")
    else:
        fallbacks = ("global_multiscale", "global_hierarchical")
    for candidate in fallbacks:
        if Path("graphs", candidate).exists():
            print(f"Graph '{original}' not found locally — using '{candidate}'")
            args.graph = candidate
            return
    raise FileNotFoundError(
        f"No usable graph directory found under graphs/. "
        f"Tried '{original}', 'global_multiscale', 'global_hierarchical'."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--dataset", default=_DEFAULT_DATASET,
                        help="Absolute path to the dataset dir containing static/ "
                             f"(default: {_DEFAULT_DATASET})")
    parser.add_argument("--output", default=str(_SCRIPT_DIR / "graphcast_scripted.pt"))
    parser.add_argument("--batch-size", type=int, default=1)
    cfg = parser.parse_args()

    print(f"Loading checkpoint: {cfg.checkpoint}")
    args = build_args(cfg.model, cfg.checkpoint, cfg.dataset)
    model_cls = MODELS[cfg.model]
    resolve_graph(args, model_cls)
    print(f"  hidden_dim={args.hidden_dim}, graph={args.graph}, dataset={args.dataset}")

    model = model_cls.load_from_checkpoint(cfg.checkpoint, args=args, strict=True)
    model.eval()
    model.requires_grad_(False)  # parameters must not require grad to be traced as constants

    B = cfg.batch_size
    N_grid = model.num_grid_nodes
    d_state = constants.GRID_STATE_DIM
    grid_static_dim = model.grid_static_features.shape[1]
    d_forcing = model.grid_dim - 2 * d_state - grid_static_dim
    print(f"Grid nodes: {N_grid}, state dim: {d_state}, forcing dim: {d_forcing}")

    prev_state      = torch.zeros(B, N_grid, d_state)
    prev_prev_state = torch.zeros(B, N_grid, d_state)
    forcing         = torch.zeros(B, N_grid, d_forcing)

    # Trace a plain function rather than an nn.Module wrapper.
    # Tracing an nn.Module that contains a LightningModule triggers PL's
    # trainer property during module traversal, which raises if no Trainer
    # is attached. A function trace skips module traversal entirely.
    #
    # We also return only the prediction tensor (not the std which is None
    # when output_std=False) because TorchScript cannot trace None outputs.
    print("Tracing model...")
    try:
        with torch.no_grad():
            traced = torch.jit.trace(
                lambda ps, pps, f: model.predict_step(ps, pps, f)[0],
                (prev_state, prev_prev_state, forcing),
                strict=False,
            )
    except Exception as e:
        print(f"\nTrace failed: {e}")
        print("Fallback: saving state dict (requires neural_lam + args to reload).")
        torch.save({"state_dict": model.state_dict(), "args": args}, cfg.output)
        print(f"Saved state dict to {cfg.output}")
        sys.exit(0)

    with torch.no_grad():
        pred_orig   = model.predict_step(prev_state, prev_prev_state, forcing)[0]
        pred_traced = traced(prev_state, prev_prev_state, forcing)
    max_err = (pred_orig - pred_traced).abs().max().item()
    print(f"Trace verification max abs error: {max_err:.2e}")
    if max_err > 1e-4:
        print("WARNING: large numerical difference — check model for data-dependent branching")

    torch.jit.save(traced, cfg.output)
    print(f"\nSaved TorchScript model to: {cfg.output}")
    print("To run inference (no neural_lam needed):")
    print(f"    model = torch.jit.load('{cfg.output}')")
    print( "    pred = model(prev_state, prev_prev_state, forcing)  # (B, N_grid, d_state)")


if __name__ == "__main__":
    main()
