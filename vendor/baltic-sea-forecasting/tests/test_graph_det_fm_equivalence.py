"""
Guard that the graph_det and graph_fm backbones stay architecturally identical.

graph_det (GraphDET -> GraphDiff) and graph_fm (GraphFM -> BaseHiGraphModel ->
BaseGraphModel) are meant to be the *same* encode-process-decode model, with
the single difference that graph_det replaces every plain ``nn.LayerNorm`` by a
``ConditionalLayerNorm`` modulated by a noise-level embedding.

Historically the two were maintained as independent copies of the same code and
could silently drift apart. The shared building blocks now live in
``neural_lam.utils`` and ``neural_lam.interaction_net``, and this test asserts
that the assembled models still line up parameter for parameter.
"""

# Standard library
import argparse
from pathlib import Path

# Third-party
import pytest
import torch

# First-party
import neural_lam.train_model as train_model

# Config used by the paired training scripts
# (training_scripts/graph_fm/graph_fm_pretrain.sh and
#  training_scripts/seacast/seacast_pretrain_muon.sh), which are identical
# except for --model.
CONFIG_PATH = Path("data/global_ocean_1_density.yaml")
GRAPH_NAME = "global_cluster_1_deg_20_refinement_4_levels"

# The args shared by both training scripts
COMMON_ARGS = [
    "--config_path", str(CONFIG_PATH),
    "--graph", GRAPH_NAME,
    "--hidden_dim", "64",
    "--hidden_dim_grid", "32",
    "--hidden_dim_mesh_nodes", "32",
    "--hidden_dim_edge", "16",
    "--processor_layers", "2",
    "--input_steps", "2",
    "--batch_size", "1",
]

MAP_NOISE_PREFIX = "map_noise."
COND_NORM_INFIX = ".layer_norm."


def _build_arg_parser():
    """
    Get the real train_model argument parser, so that this test always uses the
    same defaults as an actual training run. The parser is built inside
    ``main()``, so capture it on its way to ``parse_args``.
    """
    captured = {}
    original_parse_args = argparse.ArgumentParser.parse_args

    class _Captured(Exception):
        pass

    def _capture(self, *args, **kwargs):
        captured["parser"] = self
        raise _Captured()

    argparse.ArgumentParser.parse_args = _capture
    try:
        train_model.main([])
    except _Captured:
        pass
    finally:
        argparse.ArgumentParser.parse_args = original_parse_args

    return captured["parser"]


def _param_shapes(model, strip_prefix=""):
    """Map parameter name -> shape, optionally stripping a module prefix."""
    shapes = {}
    for name, param in model.named_parameters():
        if strip_prefix and name.startswith(strip_prefix):
            name = name[len(strip_prefix) :]
        shapes[name] = tuple(param.shape)
    return shapes


def compare_backbones(model_fm, model_det):
    """
    Line up the graph_fm and graph_det parameters.

    Returns a dict with:
    shared: {name: shape} present in both, in graph_fm naming
    fm_only: {name: shape} only in graph_fm (expected: LayerNorm affine params)
    det_only: {name: shape} only in graph_det (expected: empty)
    mismatched: {name: (fm_shape, det_shape)} (expected: empty)
    cond_norm: {name: shape} graph_det's ConditionalLayerNorm params
    noise_embedding: {name: shape} graph_det's noise embedding stack
    """
    fm_params = _param_shapes(model_fm)
    # GraphDET wraps its backbone in self.model
    det_params = _param_shapes(model_det, strip_prefix="model.")

    noise_embedding = {
        k: v for k, v in det_params.items() if k.startswith(MAP_NOISE_PREFIX)
    }
    det_params = {
        k: v
        for k, v in det_params.items()
        if not k.startswith(MAP_NOISE_PREFIX)
    }

    cond_norm = {k: v for k, v in det_params.items() if COND_NORM_INFIX in k}
    # graph_det's Linear stacks live under an extra "mlp_layers." level;
    # drop it to get graph_fm's flat nn.Sequential naming.
    det_linear = {
        k.replace(".mlp_layers.", "."): v
        for k, v in det_params.items()
        if COND_NORM_INFIX not in k
    }

    shared = {k: v for k, v in fm_params.items() if k in det_linear}
    return {
        "shared": shared,
        "fm_only": {k: v for k, v in fm_params.items() if k not in det_linear},
        "det_only": {
            k: v for k, v in det_linear.items() if k not in fm_params
        },
        "mismatched": {
            k: (fm_params[k], det_linear[k])
            for k in shared
            if fm_params[k] != det_linear[k]
        },
        "cond_norm": cond_norm,
        "noise_embedding": noise_embedding,
    }


@pytest.fixture(scope="module")
def built_models():
    """
    Build one instance of each model that shares the GraphDiff/graph_fm
    prediction head, from the config the paired training scripts use.
    """
    if not CONFIG_PATH.exists():
        pytest.skip(f"{CONFIG_PATH} not available")

    parser = _build_arg_parser()

    (
        config,
        datastore,
        datastore_boundary,
        datastore_atmosphere,
        statistics_datastore,
        statistics_datastore_boundary,
        statistics_datastore_atmosphere,
    ) = train_model.load_config_and_datastores(config_path=str(CONFIG_PATH))

    graph_dir = Path(datastore.root_path) / "graphs" / GRAPH_NAME
    if not graph_dir.exists():
        pytest.skip(f"graph {GRAPH_NAME} not available at {graph_dir}")

    kwargs = dict(
        config=config,
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        datastore_atmosphere=datastore_atmosphere,
        statistics_datastore=statistics_datastore,
        statistics_datastore_boundary=statistics_datastore_boundary,
        statistics_datastore_atmosphere=statistics_datastore_atmosphere,
    )

    return {
        name: train_model.MODELS[name](
            parser.parse_args(COMMON_ARGS + ["--model", name]), **kwargs
        )
        for name in ("graph_fm", "graph_det", "graph_crps")
    }


@pytest.fixture(scope="module")
def models(built_models):
    """The (graph_fm, graph_det) pair, in that order."""
    return built_models["graph_fm"], built_models["graph_det"]


def test_backbones_have_matching_parameters(models):
    """
    Every graph_fm parameter that is not a LayerNorm affine weight must have a
    graph_det counterpart of exactly the same shape, and vice versa.
    """
    report = compare_backbones(*models)

    assert not report["mismatched"], (
        "graph_fm and graph_det have differently shaped parameters: "
        f"{report['mismatched']}"
    )
    assert not report["det_only"], (
        "graph_det has backbone parameters that graph_fm does not: "
        f"{sorted(report['det_only'])}"
    )
    assert report["shared"], "no parameters were matched up at all"


def test_only_difference_is_the_normalization(models):
    """
    The graph_fm parameters without a graph_det counterpart must all be plain
    LayerNorm affine parameters, and each must correspond to a
    ConditionalLayerNorm (scale_layer + offset_layer) in graph_det.
    """
    report = compare_backbones(*models)

    missing_cond_norm = []
    for name in report["fm_only"]:
        # "<module path>.<index>.weight" -> "<module path>"
        module_path = name.rsplit(".", 2)[0]
        scale = f"{module_path}{COND_NORM_INFIX}scale_layer.weight"
        if scale not in report["cond_norm"]:
            missing_cond_norm.append(name)

    assert not missing_cond_norm, (
        "graph_fm parameters with no graph_det counterpart and no matching "
        f"ConditionalLayerNorm: {missing_cond_norm}"
    )

    # Each LayerNorm site contributes weight+bias in graph_fm, and
    # scale_layer.{weight,bias} + offset_layer.{weight,bias} in graph_det.
    assert len(report["cond_norm"]) == 2 * len(report["fm_only"]), (
        f"{len(report['fm_only'])} LayerNorm affine params in graph_fm but "
        f"{len(report['cond_norm'])} ConditionalLayerNorm params in graph_det"
    )

    # The conditioning signal itself is graph_det's only other addition.
    assert report["noise_embedding"], (
        "graph_det has no noise embedding stack to drive its conditional "
        "layer norms"
    )


CLAMP_BUFFERS = (
    "clamp_lower_upper_idx",
    "clamp_lower_idx",
    "clamp_upper_idx",
)


def test_all_models_prepare_clamping(built_models):
    """
    Every model using the graph_fm prediction head must call
    prepare_clamping_params, otherwise get_clamped_new_state has no indices to
    work with and predictions silently escape the configured valid range
    (e.g. siconc outside [0, 1]).

    This is a guard for new models wired to the GraphDiff backbone: forgetting
    prepare_clamping_params is exactly the bug graph_det and graph_crps had.
    """
    for name, model in built_models.items():
        buffers = dict(model.named_buffers())
        for buffer_name in CLAMP_BUFFERS:
            assert buffer_name in buffers, (
                f"{name} is missing buffer '{buffer_name}' - it likely never "
                "calls prepare_clamping_params(config, datastore)"
            )


def test_clamping_indices_agree_across_models(built_models):
    """
    All these models share a config, so they must clamp exactly the same state
    variables in the same way.
    """
    reference_name = "graph_fm"
    reference = dict(built_models[reference_name].named_buffers())

    for name, model in built_models.items():
        if name == reference_name:
            continue
        buffers = dict(model.named_buffers())
        for buffer_name in CLAMP_BUFFERS:
            assert torch.equal(
                buffers[buffer_name], reference[buffer_name]
            ), (
                f"{name}.{buffer_name} differs from {reference_name}: "
                f"{buffers[buffer_name]} vs {reference[buffer_name]}"
            )
