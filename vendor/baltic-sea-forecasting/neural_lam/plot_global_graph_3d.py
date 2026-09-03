# Standard library
import os
from argparse import ArgumentParser

# Third-party
import numpy as np
import plotly.graph_objects as go
import torch

# Local
from . import utils
from .config import load_config_and_datastores
from .graphs import vis

GRID_RADIUS = 1
MAX_EDGES_PLOT = 100_000


def main():
    """Plot global graph structure in 3D."""
    parser = ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to config file.",
    )
    parser.add_argument(
        "--graph_name",
        type=str,
        default="global_multiscale",
        help="Name of saved graph to plot.",
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Name of .html file to save interactive plot to.",
    )
    parser.add_argument(
        "--show_axis",
        action="store_true",
        help="If the axis should be displayed.",
    )
    # Geometry
    parser.add_argument(
        "--mesh_height",
        type=float,
        default=0.03,
        help="Height of mesh over grid (radius offset).",
    )
    parser.add_argument(
        "--mesh_level_dist",
        type=float,
        default=0.06,
        help="Distance between mesh levels (radius offset).",
    )
    parser.add_argument(
        "--edge_width",
        type=float,
        default=0.4,
        help="Width of g2m/m2g edges.",
    )
    parser.add_argument(
        "--mesh_edge_width",
        type=float,
        help="Width of mesh edges, if different than --edge_width.",
    )
    parser.add_argument(
        "--grid_node_size",
        type=float,
        default=1.0,
        help="Size of grid nodes.",
    )
    parser.add_argument(
        "--mesh_node_size",
        type=float,
        default=2.0,
        help="Size of mesh nodes.",
    )
    # Colors
    parser.add_argument(
        "--g2m_color",
        type=str,
        default="black",
        help="Color of g2m edges.",
    )
    parser.add_argument(
        "--m2g_color",
        type=str,
        default="black",
        help="Color of m2g edges.",
    )
    parser.add_argument(
        "--grid_color",
        type=str,
        default="dodgerblue",
        help="Color of grid nodes.",
    )
    parser.add_argument(
        "--mesh_color",
        type=str,
        default="orange",
        help="Color of mesh nodes and edges.",
    )
    # Earth
    parser.add_argument(
        "--texture_resolution",
        type=float,
        default=0.75,
        help="Resolution of texture on earth (1.0 = full).",
    )

    args = parser.parse_args()

    assert args.config_path is not None, "Specify --config_path"

    _, datastore, *_ = load_config_and_datastores(config_path=args.config_path)

    graph_dir_path = os.path.join(
        datastore.root_path, "graphs", args.graph_name
    )
    if not os.path.isdir(graph_dir_path):
        raise FileNotFoundError(
            f"Graph directory not found: {graph_dir_path}. "
            "Run create_global_graph first."
        )

    # Load graph (g2m, m2g, m2m, mesh_up/down, reindexed)
    hierarchical, graph_ldict = utils.load_graph(
        graph_dir_path=graph_dir_path,
        datastore=datastore,
    )

    # Interior grid: get_xy returns (lon, lat) = (x, y); use as-is for vis
    interior_mask = datastore.get_mask(surface=True, stacked=True, invert=False)
    xy = datastore.get_xy("state", stacked=True)
    grid_xy = np.asarray(xy[interior_mask], dtype=np.float32)

    # Mesh positions: (sin_lon, cos_lon, sin_lat, cos_lat) per level
    mesh_pos_list = torch.load(
        os.path.join(graph_dir_path, "mesh_features.pt"),
        map_location="cpu",
        weights_only=True,
    )

    def to_numpy(level_pos):
        return (
            level_pos.numpy()
            if isinstance(level_pos, torch.Tensor)
            else level_pos
        )

    def node_features_to_lon_lat(feats: np.ndarray) -> np.ndarray:
        """Invert node features to (lon, lat) in degrees."""
        sin_lon, cos_lon, sin_lat, cos_lat = (
            feats[:, 0],
            feats[:, 1],
            feats[:, 2],
            feats[:, 3],
        )
        lon_rad = np.arctan2(sin_lon, cos_lon)
        lat_rad = np.arctan2(sin_lat, cos_lat)
        lon_deg = np.float32(np.rad2deg(lon_rad))
        lat_deg = np.float32(np.rad2deg(lat_rad))
        return np.stack([lon_deg, lat_deg], axis=1)

    mesh_xy_level = []
    for p in mesh_pos_list:
        arr = np.asarray(to_numpy(p), dtype=np.float32)
        arr = node_features_to_lon_lat(arr[:, :4])
        mesh_xy_level.append(arr)

    # Edge indices (reindexed by load_graph: grid 0..N_grid-1, mesh 0..N_mesh-1)
    g2m_edge_index = graph_ldict["g2m_edge_index"].numpy()
    m2g_edge_index = graph_ldict["m2g_edge_index"].numpy()

    def subsample_edges(edge_index: np.ndarray, max_edges: int, label: str):
        """Subsample to max_edges for plotting; return edge_index."""
        n_edges = edge_index.shape[1]
        if n_edges <= max_edges:
            return edge_index
        rng = np.random.default_rng(42)
        idx = rng.choice(n_edges, size=max_edges, replace=False)
        sub = edge_index[:, idx]
        print(f"Subsampled {label}: {n_edges} -> {max_edges} edges")
        return sub

    g2m_edge_index = subsample_edges(g2m_edge_index, MAX_EDGES_PLOT, "g2m")
    m2g_edge_index = subsample_edges(m2g_edge_index, MAX_EDGES_PLOT, "m2g")

    mesh_edge_width = (
        args.edge_width
        if args.mesh_edge_width is None
        else args.mesh_edge_width
    )
    mesh_radius = GRID_RADIUS + args.mesh_height

    data_objs = []

    # Grid nodes (lon, lat) = (x, y)
    data_objs.append(
        vis.create_node_plot(
            grid_xy,
            "Grid Nodes",
            color=args.grid_color,
            radius=GRID_RADIUS,
            size=args.grid_node_size,
        )
    )

    # Mesh levels and edges
    if hierarchical:
        m2m_edge_index = [ei.numpy() for ei in graph_ldict["m2m_edge_index"]]
        mesh_up_edge_index = [
            ei.numpy() for ei in graph_ldict["mesh_up_edge_index"]
        ]
        mesh_down_edge_index = [
            ei.numpy() for ei in graph_ldict["mesh_down_edge_index"]
        ]

        for bot_level_i, intra_ei in enumerate(m2m_edge_index):
            top_level_i = bot_level_i + 1
            bot_pos = mesh_xy_level[bot_level_i]
            bot_radius = mesh_radius + bot_level_i * args.mesh_level_dist

            data_objs.append(
                vis.create_node_plot(
                    bot_pos,
                    f"Mesh level {bot_level_i} nodes",
                    color=args.mesh_color,
                    radius=bot_radius,
                    size=args.mesh_node_size,
                )
            )
            data_objs.append(
                vis.create_edge_plot(
                    intra_ei,
                    bot_pos,
                    bot_pos,
                    f"Mesh level {bot_level_i} edges",
                    color=args.mesh_color,
                    width=mesh_edge_width,
                    from_radius=bot_radius,
                    to_radius=bot_radius,
                )
            )

            if top_level_i < len(m2m_edge_index):
                up_ei = mesh_up_edge_index[bot_level_i]
                down_ei = mesh_down_edge_index[bot_level_i]
                top_pos = mesh_xy_level[top_level_i]
                top_radius = mesh_radius + top_level_i * args.mesh_level_dist
                data_objs.append(
                    vis.create_edge_plot(
                        up_ei,
                        bot_pos,
                        top_pos,
                        f"Mesh up {bot_level_i}->{top_level_i}",
                        color=args.mesh_color,
                        width=mesh_edge_width,
                        from_radius=bot_radius,
                        to_radius=top_radius,
                    )
                )
                data_objs.append(
                    vis.create_edge_plot(
                        down_ei,
                        top_pos,
                        bot_pos,
                        f"Mesh down {top_level_i}->{bot_level_i}",
                        color=args.mesh_color,
                        width=mesh_edge_width,
                        from_radius=top_radius,
                        to_radius=bot_radius,
                    )
                )

        grid_con_xy = mesh_xy_level[0]
    else:
        mesh_xy = mesh_xy_level[0]
        # Non-hierarchical: m2m_edge_index is a single tensor
        m2m_ei = graph_ldict["m2m_edge_index"]
        m2m_edge_index = (
            m2m_ei.numpy() if torch.is_tensor(m2m_ei) else m2m_ei[0].numpy()
        )
        data_objs.append(
            vis.create_node_plot(
                mesh_xy,
                "Mesh Nodes",
                radius=mesh_radius,
                color=args.mesh_color,
                size=args.mesh_node_size,
            )
        )
        data_objs.append(
            vis.create_edge_plot(
                m2m_edge_index,
                mesh_xy,
                mesh_xy,
                "Mesh Edges",
                from_radius=mesh_radius,
                to_radius=mesh_radius,
                color=args.mesh_color,
                width=mesh_edge_width,
            )
        )
        grid_con_xy = mesh_xy

    # G2M edges (grid -> mesh bottom)
    data_objs.append(
        vis.create_edge_plot(
            g2m_edge_index,
            grid_xy,
            grid_con_xy,
            "G2M Edges",
            color=args.g2m_color,
            width=args.edge_width,
            from_radius=GRID_RADIUS,
            to_radius=mesh_radius,
        )
    )

    # M2G edges (mesh bottom -> grid)
    data_objs.append(
        vis.create_edge_plot(
            m2g_edge_index,
            grid_con_xy,
            grid_xy,
            "M2G Edges",
            color=args.m2g_color,
            width=args.edge_width,
            from_radius=mesh_radius,
            to_radius=GRID_RADIUS,
        )
    )

    # Earth texture
    try:
        data_objs.append(
            vis.make_earth(
                radius=GRID_RADIUS,
                resolution_reduction=args.texture_resolution,
            )
        )
    except Exception as e:
        print(f"Earth texture skipped ({e}).")

    fig = go.Figure(data=data_objs)
    fig.update_layout(scene_aspectmode="data")
    fig.update_traces(connectgaps=False)

    if not args.show_axis:
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(
                yanchor="bottom",
                y=0.01,
                xanchor="right",
                x=0.99,
            ),
            scene={
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "zaxis": {"visible": False},
            },
        )

    if args.save:
        fig.write_html(args.save, include_plotlyjs="cdn")
    else:
        fig.show()


if __name__ == "__main__":
    main()
