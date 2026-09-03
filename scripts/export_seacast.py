#!/usr/bin/env python3
"""
Export the SeaCast checkpoint to TorchScript.

Ocean counterpart of `data/paper_checkpoints/export_torchscript.py`, which handles the
weather models. Three things differ and each one matters:

  1. `predict_step` takes FIVE arguments here, not three:
         predict_step(prev_state, prev_prev_state, forcing, boundary_forcing,
                      atmosphere_forcing)
     `boundary_forcing` is None for the global model.

  2. The model classes come from the VENDORED BSF tree, which ships its own `neural_lam`
     that collides with `backend/neural-lam`. See VENDORED.md -- importing the wrong one
     loads the wrong model code silently. This script asserts which one it got.

  3. The checkpoint was pickled under zarr 2 and references `zarr.core.Array`,
     `zarr.hierarchy` and friends, none of which exist in zarr 3. Run this in `.venv-export`
     (zarr<3), never in the main venv.

WHY THIS NEEDS DATASTORES AT ALL
    Layer sizes come from the data: how many state features, how wide the windowed forcing
    is, how many static fields, how many mesh nodes and edges. The checkpoint already stores
    `config`, `datastore_atmosphere` and `datastore_boundary` (see `ar_model.py:42`, which
    saves everything except the four listed there), so only `datastore` and
    `statistics_datastore` have to be rebuilt here.

Usage:
    PYTHONPATH=vendor/baltic-sea-forecasting .venv-export/bin/python scripts/export_seacast.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
_VENDOR = _REPO / "vendor" / "baltic-sea-forecasting"
_CONFIGS = _REPO / "data" / "ocean_datastores"
_OCEAN_DATA = _REPO / "backend" / "data" / "ocean_datasets"
_CKPT_DIR = _REPO / "backend" / "data" / "paper_checkpoints"
_GRAPH_DIR = _REPO / "backend" / "data" / "graphs"

# MDPDatastore locates its zarr as <config dir>/<config basename>.zarr (mdp.py:53-55), so the
# stores have to be reachable under those exact names. Symlinked rather than copied -- these
# are hundreds of MB and the originals live where the backend reads them.
_ZARR_LINKS = {
    "analysis_ocean_mdp_config.zarr": _OCEAN_DATA / "analysis_ocean.zarr",
    "global_ocean_1_4_mdp_config.zarr": _CONFIGS / "slice_global_ocean_1_4_mdp_config.zarr",
    "analysis_ifs_forecasts_mdp_config.zarr": _OCEAN_DATA / "ifs_forcing.zarr",
}


def _link_zarrs() -> None:
    # neural-lam resolves graphs as <datastore root>/graphs/<name>, and the datastore root is
    # the config's directory (mdp.py:51). The graphs live with the backend data, so link.
    links = dict(_ZARR_LINKS)
    links["graphs"] = _GRAPH_DIR

    for name, target in links.items():
        link = _CONFIGS / name
        if link.is_symlink() or link.exists():
            continue
        if not target.exists():
            print(f"  ! {target} missing -- {name} will not resolve")
            continue
        link.symlink_to(target.resolve())
        print(f"  linked {name} -> {target.relative_to(_REPO)}")


def _assert_vendored_neural_lam() -> None:
    import neural_lam

    got = Path(neural_lam.__file__).resolve()
    if _VENDOR not in got.parents:
        raise SystemExit(
            f"Wrong neural_lam: {got}\n"
            f"Expected the vendored BSF tree under {_VENDOR}.\n"
            "Run with PYTHONPATH=vendor/baltic-sea-forecasting so it wins over the pip "
            "package and the backend submodule."
        )
    print(f"  neural_lam: {got.relative_to(_REPO)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=str(_CKPT_DIR / "seacast.ckpt"))
    ap.add_argument("--output", default=str(_CKPT_DIR / "seacast_scripted.pt"))
    ap.add_argument("--graph", default="global_hierarchical_analysis")
    ap.add_argument("--config", default=str(_CONFIGS / "analysis_ocean_mdp_config.yaml"))
    ap.add_argument("--stats-config", default=str(_CONFIGS / "global_ocean_1_4_mdp_config.yaml"))
    args = ap.parse_args()

    print("Environment")
    _assert_vendored_neural_lam()
    import zarr

    if int(zarr.__version__.split(".")[0]) >= 3:
        raise SystemExit(
            f"zarr {zarr.__version__} -- the checkpoint was pickled under zarr 2 and "
            "references classes that no longer exist. Use .venv-export."
        )
    print(f"  zarr: {zarr.__version__}")

    print("\nDatastores")
    _link_zarrs()
    from neural_lam.datastore.mdp import MDPDatastore

    datastore = MDPDatastore(config_path=args.config)
    stats_datastore = MDPDatastore(config_path=args.stats_config)

    print("\nModel")
    from neural_lam.models.graph_fm import GraphFM

    # config, datastore_atmosphere and datastore_boundary are restored from the checkpoint's
    # saved hyperparameters; only these two were excluded by save_hyperparameters.
    # weights_only=False: the checkpoint stores an argparse.Namespace and the saved
    # datastore objects, which torch >=2.6 refuses under the default weights_only=True.
    # Safe here -- this is our own file from a known source.
    model = GraphFM.load_from_checkpoint(
        args.checkpoint,
        datastore=datastore,
        statistics_datastore=stats_datastore,
        map_location="cpu",
        weights_only=False,
    )
    model.eval()
    model.requires_grad_(False)   # so weights trace as constants

    n_grid = model.num_grid_nodes
    d_state = model.grid_output_dim            # 28 = 27 state features + ice density
    d_static = model.grid_static_features.shape[1]
    d_atmo = model.atmosphere_dim
    # interior_input_dim = 2*d_state + d_static + d_forcing + d_atmo, so forcing falls out.
    # It is the only width the model does not expose directly.
    d_forcing = model.interior_input_dim - 2 * d_state - d_static - d_atmo

    print(f"  grid nodes         : {n_grid:,}")
    print(f"  state (incl density): {d_state}")
    print(f"  static             : {d_static}")
    print(f"  forcing            : {d_forcing}   (2 features x 3 window steps)")
    print(f"  atmosphere         : {d_atmo}  (8 features x 3 window steps)")
    print(f"  interior_input_dim : {model.interior_input_dim}"
          f"  = 2*{d_state} + {d_static} + {d_forcing} + {d_atmo}")

    print("\nTracing")
    b = 1
    prev = torch.zeros(b, n_grid, d_state)
    prev_prev = torch.zeros_like(prev)
    forcing = torch.zeros(b, n_grid, d_forcing)
    atmosphere = torch.zeros(b, n_grid, d_atmo)

    with torch.no_grad():
        traced = torch.jit.trace(
            lambda ps, pps, f, a: model.predict_step(ps, pps, f, None, a)[0],
            (prev, prev_prev, forcing, atmosphere),
            strict=False,
        )
        ref = model.predict_step(prev, prev_prev, forcing, None, atmosphere)[0]
        got = traced(prev, prev_prev, forcing, atmosphere)

    err = (ref - got).abs().max().item()
    print(f"  trace verification max abs error: {err:.2e}")
    if err > 1e-4:
        print("  WARNING: large difference -- check for data-dependent branching")

    torch.jit.save(traced, args.output)
    print(f"\nSaved: {args.output}")
    print("Load with torch.jit.load -- no neural_lam, zarr or BSF needed at runtime.")


if __name__ == "__main__":
    main()
