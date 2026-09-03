# Third-party
from torch import nn

# First-party
from neural_lam import utils
from neural_lam.interaction_net import FlexibleNet
from neural_lam.models.base_latent_encoder import BaseLatentEncoder


class GraphLatentEncoder(BaseLatentEncoder):
    """
    Encoder that maps from grid to mesh and defines a latent distribution
    on mesh
    """

    def __init__(
        self,
        latent_dim,
        g2m_edge_index,
        m2m_edge_index,
        hidden_dim,
        processor_layers,
        hidden_layers=1,
        output_dist="isotropic",
        hidden_dim_grid=None,
        hidden_dim_mesh_nodes=None,
        hidden_dim_edge=None,
        num_grid_con_mesh_nodes=None,
    ):
        super().__init__(
            latent_dim,
            output_dist,
        )

        hidden_dim_grid = (
            hidden_dim if hidden_dim_grid is None else hidden_dim_grid
        )
        hidden_dim_mesh_nodes = (
            hidden_dim
            if hidden_dim_mesh_nodes is None
            else hidden_dim_mesh_nodes
        )
        hidden_dim_edge = (
            hidden_dim_grid if hidden_dim_edge is None else hidden_dim_edge
        )
        if num_grid_con_mesh_nodes is None:
            num_grid_con_mesh_nodes = int(g2m_edge_index[1].max().item() + 1)

        # GNN from grid to mesh
        self.g2m_gnn = FlexibleNet(
            edge_index=g2m_edge_index,
            send_node_dim=hidden_dim_grid,
            rec_node_dim=hidden_dim_mesh_nodes,
            edge_dim=hidden_dim_edge,
            hidden_layers=hidden_layers,
            num_rec=num_grid_con_mesh_nodes,
            propagation=True,
            aggr="mean",
        )

        self.pre_mesh_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim_grid, hidden_dim),
        )

        # Processor layers on mesh
        self.processor = utils.make_gnn_seq(
            m2m_edge_index,
            processor_layers,
            hidden_layers,
            hidden_dim,
        )

        self.latent_param_map = utils.make_mlp(
            [hidden_dim] * (hidden_layers + 1) + [self.output_dim],
            layer_norm=False,
        )

    # pylint: disable-next=arguments-differ
    def compute_dist_params(self, grid_rep, graph_emb, **kwargs):
        """
        Compute parameters of distribution over latent variable using the
        grid representation

        grid_rep: (B, N_grid, d_h)
        graph_emb: dict with graph embedding vectors, entries at least
            mesh: (B, N_mesh, d_h)
            g2m: (B, M_g2m, d_h)
            m2m: (B, M_g2m, d_h)

        Returns:
        parameters: (B, num_mesh_nodes, d_output)
        """
        mesh_rep = self.g2m_gnn(
            grid_rep, graph_emb["mesh"], graph_emb["g2m"]
        )  # (B, N_mesh, hidden_dim_grid)
        mesh_rep = self.pre_mesh_proj(mesh_rep)  # (B, N_mesh, d_h)
        mesh_rep, _ = self.processor(mesh_rep, graph_emb["m2m"])
        return self.latent_param_map(mesh_rep)  # (B, N_mesh, d_output)
