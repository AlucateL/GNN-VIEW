# Third-party
from torch import nn

# First-party
from neural_lam import utils
from neural_lam.interaction_net import FlexibleNet
from neural_lam.models.base_graph_latent_decoder import BaseGraphLatentDecoder


class GraphLatentDecoder(BaseGraphLatentDecoder):
    """
    Decoder that maps grid input + latent variable on mesh to prediction on grid
    Uses non-hierarchical graph
    """

    def __init__(
        self,
        g2m_edge_index,
        m2m_edge_index,
        m2g_edge_index,
        hidden_dim,
        latent_dim,
        decode_dim,
        grid_state_dim,
        processor_layers,
        hidden_layers=1,
        output_std=True,
        hidden_dim_mesh_nodes=None,
        hidden_dim_edge=None,
        num_interior_nodes=None,
    ):
        super().__init__(
            hidden_dim,
            decode_dim,
            latent_dim,
            grid_state_dim,
            hidden_layers,
            output_std,
        )

        hidden_dim_grid = decode_dim
        hidden_dim_mesh_nodes = (
            hidden_dim
            if hidden_dim_mesh_nodes is None
            else hidden_dim_mesh_nodes
        )
        hidden_dim_edge = (
            hidden_dim_grid if hidden_dim_edge is None else hidden_dim_edge
        )
        if num_interior_nodes is None:
            num_interior_nodes = int(m2g_edge_index[1].max().item() + 1)

        # GNN from grid to mesh
        self.g2m_gnn = FlexibleNet(
            edge_index=g2m_edge_index,
            send_node_dim=hidden_dim_grid,
            rec_node_dim=hidden_dim,
            edge_dim=hidden_dim_edge,
            hidden_layers=hidden_layers,
            num_rec=int(g2m_edge_index[1].max().item() + 1),
            propagation=True,
            aggr="mean",
        )

        self.pre_processor_proj = nn.Sequential(
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

        self.post_mesh_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim_grid),
        )

        # GNN from mesh to grid
        self.m2g_gnn = FlexibleNet(
            edge_index=m2g_edge_index,
            send_node_dim=hidden_dim_grid,
            rec_node_dim=hidden_dim_grid,
            edge_dim=hidden_dim_edge,
            hidden_layers=hidden_layers,
            num_rec=num_interior_nodes,
            propagation=False,
            aggr="sum",
        )

    def combine_with_latent(
        self, original_grid_rep, latent_rep, residual_grid_rep, graph_emb
    ):
        """
        Combine the grid representation with representation of latent variable.
        The output should be on the grid again.

        original_grid_rep: (B, num_grid_nodes, d_h)
        latent_rep: (B, num_mesh_nodes, d_h)
        residual_grid_rep: (B, num_grid_nodes, d_h)

        Returns:
        grid_rep: (B, num_grid_nodes, d_h)
        """
        mesh_rep = self.g2m_gnn(
            original_grid_rep, latent_rep, graph_emb["g2m"]
        )  # (B, N_mesh, hidden_dim_grid)
        mesh_rep = self.pre_processor_proj(mesh_rep)  # (B, N_mesh, hidden_dim)
        mesh_rep, _ = self.processor(mesh_rep, graph_emb["m2m"])
        mesh_rep = self.post_mesh_proj(mesh_rep)  # (B, N_mesh, d_h_grid)
        grid_rep = self.m2g_gnn(
            mesh_rep, residual_grid_rep, graph_emb["m2g"]
        )  # (B, N_grid, d_h_grid)
        return grid_rep
