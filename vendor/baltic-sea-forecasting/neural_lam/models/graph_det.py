# Standard library
from typing import Union

# Third-party
import matplotlib.pyplot as plt
import torch
import wandb

# First-party
from neural_lam import metrics
from neural_lam.models.ar_model import ARModel
from neural_lam.models.graph_diff import GraphDiff

# Local
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore


class GraphDET(ARModel):
    """
    SeaCast with learnable layernorm.

    Deterministic model that is architecturally identical to graph_fm
    (GraphFM), with the single difference that its backbone (GraphDiff) uses
    ConditionalLayerNorm, modulated by an embedding of a constant noise level,
    where graph_fm uses plain nn.LayerNorm. See
    tests/test_graph_det_fm_equivalence.py, which enforces this.
    """

    def __init__(
        self,
        args,
        config: NeuralLAMConfig,
        datastore: BaseDatastore,
        datastore_boundary: Union[BaseDatastore, None],
        datastore_atmosphere: Union[BaseDatastore, None],
        **kwargs,
    ):
        super().__init__(
            args,
            config,
            datastore,
            datastore_boundary,
            datastore_atmosphere,
            **kwargs,
        )

        if args.backbone_model == "graph_diff":
            print("Using GraphDiff")
            self.model = GraphDiff(
                args,
                config,
                datastore,
                datastore_boundary,
                datastore_atmosphere,
            )
        else:
            raise NotImplementedError(
                f"Unknown backbone model: {args.backbone_model}"
            )

        # Compute indices and define clamping functions, matching
        # BaseGraphModel (and thereby graph_fm)
        self.prepare_clamping_params(config, datastore)

    def predict_step(
        self,
        prev_state,
        prev_prev_state,
        forcing,
        boundary_forcing,
        atmosphere_forcing,
    ):
        """
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
        prev_state: (B, num_grid_nodes, feature_dim), X_t
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
        forcing: (B, num_grid_nodes, forcing_dim)
        boundary_forcing: (B, num_boundary_nodes, boundary_forcing_dim)
        atmosphere_forcing: (B, num_atmosphere_nodes, atmosphere_forcing_dim)


        Returns:
        next_state: (B, N_grid, d_state),
            predicted weather state X_{t+1} at time t+1
        pred_std: (B, N_grid, d_state) if output_std, else None
        """
        if prev_prev_state is not None:
            input_grid = torch.cat(
                (prev_state, prev_prev_state, forcing), dim=-1
            )
        else:
            input_grid = torch.cat((prev_state, forcing), dim=-1)
        # (B, N_grid, d_input)

        z = torch.ones(
            prev_state.shape[0],
            self.model.noise_dim,
            device=prev_state.device) * self.args.deterministic_noise_value

        net_output = self.model(
            input_grid,
            noise_level=z,
            cond=None,
            boundary_forcing=boundary_forcing,
            atmosphere_forcing=atmosphere_forcing,
        )  # (B, N_grid, d_grid_out)

        # NOTE: From here on this is identical to BaseGraphModel.predict_step,
        # so that graph_det and graph_fm differ only in the normalization
        # layers of the backbone. --pred_residual has no effect here, the
        # residual parametrization is always used.
        if self.output_std:
            pred_delta_mean, pred_std_raw = net_output.chunk(
                2, dim=-1
            )  # both (B, num_grid_nodes, d_f)
            # NOTE: The predicted std. is not scaled in any way here
            # linter for some reason does not think softplus is callable
            # pylint: disable-next=not-callable
            pred_std = torch.nn.functional.softplus(pred_std_raw)
        else:
            pred_delta_mean = net_output
            pred_std = None

        # Rescale with one-step difference statistics
        rescaled_delta_mean = pred_delta_mean * self.diff_std + self.diff_mean

        # Clamp values to valid range (also add the delta to the previous state)
        new_state = self.get_clamped_new_state(rescaled_delta_mean, prev_state)

        return new_state, pred_std
