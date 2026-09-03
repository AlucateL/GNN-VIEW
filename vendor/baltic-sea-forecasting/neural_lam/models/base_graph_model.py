# Standard library
from typing import Union

# Third-party
import torch

# Local
from .. import utils
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore
from ..interaction_net import FlexibleNet
from .ar_model import ARModel


class BaseGraphModel(ARModel):
    """
    Base (abstract) class for graph-based models building on
    the encode-process-decode idea.
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
            config=config,
            datastore=datastore,
            datastore_boundary=datastore_boundary,
            datastore_atmosphere=datastore_atmosphere,
            **kwargs,
        )

        # Load graph with static features
        # NOTE: (IMPORTANT!) mesh nodes MUST have the first
        # num_mesh_nodes indices,
        graph_dir_path = datastore.root_path / "graphs" / args.graph
        self.hierarchical, graph_ldict = utils.load_graph(
            graph_dir_path=graph_dir_path,
            datastore=datastore,
        )
        for name, attr_value in graph_ldict.items():
            # Make BufferLists module members and register tensors as buffers
            if isinstance(attr_value, torch.Tensor):
                self.register_buffer(name, attr_value, persistent=False)
            else:
                setattr(self, name, attr_value)

        # Determine grid hidden dim
        if args.hidden_dim_grid is None:
            hidden_dim_grid = args.hidden_dim
        else:
            hidden_dim_grid = args.hidden_dim_grid

        # Determine edge hidden dim
        if args.hidden_dim_edge is None:
            hidden_dim_edge = hidden_dim_grid
        else:
            hidden_dim_edge = args.hidden_dim_edge

        # Determine mesh node hidden dim
        if args.hidden_dim_mesh_nodes is None:
            self.hidden_dim_mesh_nodes = args.hidden_dim
        else:
            self.hidden_dim_mesh_nodes = args.hidden_dim_mesh_nodes

        print(
            f"Using hidden_dim_grid={hidden_dim_grid}, "
            f"hidden_dim_mesh_nodes={self.hidden_dim_mesh_nodes}, "
            f"hidden_dim_edge={hidden_dim_edge}, hidden_dim={args.hidden_dim}"
        )

        # grid_dim from data + static
        self.g2m_edges, g2m_dim = self.g2m_features.shape
        self.m2g_edges, m2g_dim = self.m2g_features.shape

        # Define sub-models
        # Feature embedders for grid
        self.mlp_blueprint_end = [args.hidden_dim] * (args.hidden_layers + 1)
        # For grid hidden dim
        self.grid_mlp_blueprint_end = [hidden_dim_grid] * (
            args.hidden_layers + 1
        )
        # For edge hidden dim
        self.edge_mlp_blueprint_end = [hidden_dim_edge] * (
            args.hidden_layers + 1
        )

        # Grid embedders output hidden_dim_grid
        self.interior_embedder = utils.make_mlp(
            [self.interior_input_dim] + self.grid_mlp_blueprint_end
        )
        if self.boundary_forced:
            self.boundary_embedder = utils.make_mlp(
                [self.boundary_dim] + self.grid_mlp_blueprint_end
            )
        if self.atmosphere_forced and self.use_atmosphere_g2m:
            self.atmosphere_embedder = utils.make_mlp(
                [self.atmosphere_dim] + self.grid_mlp_blueprint_end
            )

        self.pre_mesh_proj = utils.make_mlp(
            [hidden_dim_grid] + [args.hidden_dim]
        )
        self.post_mesh_proj = utils.make_mlp(
            [args.hidden_dim] + [hidden_dim_grid]
        )

        self.g2m_embedder = utils.make_mlp(
            [g2m_dim] + self.edge_mlp_blueprint_end
        )
        self.m2g_embedder = utils.make_mlp(
            [m2g_dim] + self.edge_mlp_blueprint_end
        )

        # GNNs
        # encoder
        self.g2m_gnn = FlexibleNet(
            edge_index=self.g2m_edge_index,
            send_node_dim=hidden_dim_grid,
            rec_node_dim=self.hidden_dim_mesh_nodes,
            edge_dim=hidden_dim_edge,
            hidden_layers=args.hidden_layers,
            num_rec=self.num_grid_connected_mesh_nodes,
            propagation=True,
            aggr="mean",
        )
        self.encoding_grid_mlp = utils.make_mlp(
            [hidden_dim_grid] + self.grid_mlp_blueprint_end
        )

        # decoder
        self.m2g_gnn = FlexibleNet(
            edge_index=self.m2g_edge_index,
            send_node_dim=hidden_dim_grid,
            rec_node_dim=hidden_dim_grid,
            edge_dim=hidden_dim_edge,
            hidden_layers=args.hidden_layers,
            num_rec=self.num_grid_nodes,
            propagation=False,
            aggr="sum",
        )

        # Output mapping (hidden_dim_grid -> output_dim)
        self.output_map = utils.make_mlp(
            [hidden_dim_grid] * (args.hidden_layers + 1)
            + [self.grid_output_dim],
            layer_norm=False,
        )  # No layer norm on this one

        # Compute indices and define clamping functions
        self.prepare_clamping_params(config, datastore)

    @property
    def num_grid_connected_mesh_nodes(self):
        """
        Get the total number of mesh nodes that have a connection to
        the grid (e.g. bottom level in a hierarchy)
        """
        raise NotImplementedError(
            "num_grid_connected_mesh_nodes not implemented"
        )

    def get_num_mesh(self):
        """
        Compute number of mesh nodes from loaded features,
        and number of mesh nodes that should be ignored in encoding/decoding
        """
        raise NotImplementedError("get_num_mesh not implemented")

    def embedd_mesh_nodes(self):
        """
        Embed static mesh features
        Returns tensor of shape (num_mesh_nodes, d_h)
        """
        raise NotImplementedError("embedd_mesh_nodes not implemented")

    def process_step(self, mesh_rep):
        """
        Process step of embedd-process-decode framework
        Processes the representation on the mesh, possible in multiple steps

        mesh_rep: has shape (B, num_mesh_nodes, d_h)
        Returns mesh_rep: (B, num_mesh_nodes, d_h)
        """
        raise NotImplementedError("process_step not implemented")

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
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1} (None if
            input_steps==1)
        forcing: (B, num_grid_nodes, forcing_dim)
        boundary_forcing: (B, num_boundary_nodes, boundary_forcing_dim)
        atmosphere_forcing: (B, num_atmosphere_nodes, atmosphere_forcing_dim)
        """
        batch_size = prev_state.shape[0]
        interior_input_list = [prev_state]
        if prev_prev_state is not None:
            interior_input_list.append(prev_prev_state)
        interior_input_list.extend(
            [
                forcing,
                self.expand_to_batch(self.grid_static_features, batch_size),
            ]
        )
        if self.concat_atmosphere:
            interior_input_list.append(atmosphere_forcing)
        interior_features = torch.cat(interior_input_list, dim=-1)
        # (B, num_interior_nodes, interior_input_dim)

        # Embed all features
        interior_emb = self.interior_embedder(
            interior_features
        )  # (B, num_interior_nodes, d_h)
        g2m_emb = self.g2m_embedder(self.g2m_features)  # (M_g2m, d_h)
        m2g_emb = self.m2g_embedder(self.m2g_features)  # (M_m2g, d_h)
        mesh_emb = self.embedd_mesh_nodes()
        grid_emb_list = [interior_emb]

        if self.boundary_forced:
            boundary_features = torch.cat(
                (
                    boundary_forcing,
                    self.expand_to_batch(
                        self.boundary_static_features, batch_size
                    ),
                ),
                dim=-1,
            )  # (B, num_boundary_nodes, interior_input_dim)
            boundary_emb = self.boundary_embedder(
                boundary_features
            )  # (B, num_boundary_nodes, d_h)
            grid_emb_list.append(boundary_emb)

        if self.atmosphere_forced and self.use_atmosphere_g2m:
            atmosphere_features = torch.cat(
                (
                    atmosphere_forcing,
                    self.expand_to_batch(
                        self.atmosphere_static_features, batch_size
                    ),
                ),
                dim=-1,
            )  # (B, num_atmosphere_nodes, interior_input_dim)

            atmosphere_emb = self.atmosphere_embedder(
                atmosphere_features
            )  # (B, num_atmosphere_nodes, d_h)
            grid_emb_list.append(atmosphere_emb)

        if len(grid_emb_list) == 1:
            # Only interior
            grid_emb = grid_emb_list[0]
        else:
            # NOTE: We here assume the order of grid node index is 1) interior,
            # 2) boundary, 3) atmosphere. This has to be followed also when
            # constructing g2m. Concatenates all existing embeddings.
            grid_emb = torch.cat(grid_emb_list, dim=1)
            # (B, num_total_grid_nodes, d_h)

        # Verify dimension matches expected total grid nodes
        assert grid_emb.shape[1] == self.num_total_grid_nodes, (
            f"grid_emb has {grid_emb.shape[1]} nodes but expected "
            f"{self.num_total_grid_nodes} (interior: {self.num_grid_nodes}, "
            f"boundary: {getattr(self, 'num_boundary_nodes', 0)}, "
            f"atmosphere: {getattr(self, 'num_atmosphere_nodes', 0)})"
        )
        # Verify g2m edge indices are within bounds
        max_grid_idx = self.g2m_edge_index[0].max().item()
        assert max_grid_idx < self.num_total_grid_nodes, (
            f"g2m_edge_index[0] has max index {max_grid_idx} "
            f"but grid_emb only has {self.num_total_grid_nodes} nodes"
        )

        # Map from grid to mesh
        mesh_emb_expanded = self.expand_to_batch(
            mesh_emb, batch_size
        )  # (B, num_mesh_nodes, d_h)
        g2m_emb_expanded = self.expand_to_batch(g2m_emb, batch_size)

        # This also splits representation into grid and mesh
        mesh_rep = self.g2m_gnn(
            grid_emb, mesh_emb_expanded, g2m_emb_expanded
        )  # (B, num_mesh_nodes, d_h)
        # Also MLP with residual for grid representation
        grid_rep = interior_emb + self.encoding_grid_mlp(
            interior_emb
        )  # (B, num_interior_nodes, d_h)

        # Project up mesh rep to hidden dim of graph
        mesh_rep = self.pre_mesh_proj(mesh_rep)  # g -> m

        # Run processor step
        mesh_rep = self.process_step(mesh_rep)  # m -> m

        # Project down mesh rep to hidden dim of grid
        mesh_rep = self.post_mesh_proj(mesh_rep)  # m -> g

        # Map back from mesh to grid
        m2g_emb_expanded = self.expand_to_batch(m2g_emb, batch_size)
        grid_rep = self.m2g_gnn(
            mesh_rep, grid_rep, m2g_emb_expanded
        )  # (B, num_interior_nodes, d_h)

        # Map to output dimension, only for grid
        net_output = self.output_map(
            grid_rep
        )  # (B, num_interior_nodes, d_grid_out)

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
