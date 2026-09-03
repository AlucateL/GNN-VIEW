# Standard library
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

# Third-party
import numpy as np
import plotly.graph_objects as go

# Local
from . import utils
from .config import load_config_and_datastores
from .graphs import vis

GRID_RADIUS = 1


def main():
    """Plot graph structure in 3D using plotly."""
    parser = ArgumentParser(
        description="Plot graph",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the configuration for neural-lam",
    )
    parser.add_argument(
        "--graph_name",
        type=str,
        default="multiscale",
        help="Name of saved graph to plot",
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Name of .html file to save interactive plot to",
    )
    parser.add_argument(
        "--show_axis",
        action="store_true",
        help="If the axis should be displayed",
    )
    parser.add_argument(
        "--corner_filter_radius",
        type=float,
        help="Filter plotted objects to within given radius of interior corner",
    )
    # Geometry
    parser.add_argument(
        "--mesh_height",
        type=float,
        default=0.02,
        help="Height of mesh over grid",
    )
    parser.add_argument(
        "--mesh_level_dist",
        type=float,
        default=0.02,
        help="Distance between mesh levels",
    )
    parser.add_argument(
        "--edge_width",
        type=float,
        default=0.4,
        help="Width of edges",
    )
    parser.add_argument(
        "--mesh_edge_width",
        type=float,
        help="Width of mesh edges, if different than --edge_width",
    )
    parser.add_argument(
        "--grid_node_size",
        type=float,
        default=1.0,
        help="Size of grid nodes",
    )
    parser.add_argument(
        "--mesh_node_size",
        type=float,
        default=1.5,
        help="Size of mesh nodes",
    )
    # Colors
    parser.add_argument(
        "--g2m_color",
        type=str,
        default="black",
        help="Color of g2m edges",
    )
    parser.add_argument(
        "--m2g_color",
        type=str,
        default="black",
        help="Color of m2g edges",
    )
    parser.add_argument(
        "--grid_color",
        type=str,
        default="dodgerblue",
        help="Color of grid nodes (interior in case of LAM setup)",
    )
    parser.add_argument(
        "--boundary_grid_color",
        type=str,
        default="teal",
        help="Color of boundary grid nodes",
    )
    parser.add_argument(
        "--use_atmosphere_g2m",
        action="store_true",
        help="Include atmosphere grid nodes in the plot",
    )
    parser.add_argument(
        "--atmosphere_grid_color",
        type=str,
        default="mediumpurple",
        help="Color of atmospheric grid nodes",
    )
    parser.add_argument(
        "--mesh_color",
        type=str,
        default="orange",
        help="Color of mesh nodes and edges",
    )
    # Earth
    parser.add_argument(
        "--texture_resolution",
        type=float,
        default=0.5,
        help="Resolution of texture on earth, 1.0 is full resolution "
        "(high resolution can be slow)",
    )

    args = parser.parse_args()

    assert (
        args.config_path is not None
    ), "Specify your config with --config_path"

    _, datastore, datastore_boundary, datastore_atmosphere, *_ = (
        load_config_and_datastores(config_path=args.config_path)
    )

    boundary_forced = datastore_boundary is not None

    # Load graph data
    graph_dir_path = os.path.join(
        datastore.root_path, "graphs", args.graph_name
    )
    hierarchical, graph_ldict = utils.load_graph(
        graph_dir_path=graph_dir_path,
        datastore=datastore,
    )
    # Turn all to numpy
    (g2m_edge_index, m2g_edge_index) = (
        graph_ldict["g2m_edge_index"].numpy(),
        graph_ldict["m2g_edge_index"].numpy(),
    )

    # Load coordinates of grid nodes
    interior_mask = datastore.get_mask(surface=True, stacked=True, invert=False)
    xy_interior = datastore.get_xy("state", stacked=True)

    boundary_mask = datastore_boundary.get_mask(
        surface=True, stacked=True, invert=False
    )
    xy_boundary = datastore_boundary.get_xy("forcing", stacked=True)

    # get_xy returns (lon, lat) = (x, y)
    interior_lon_lat = xy_interior[interior_mask]
    boundary_lon_lat = xy_boundary[boundary_mask]

    atmosphere_lon_lat = None
    if args.use_atmosphere_g2m:
        atmosphere_mask = datastore_atmosphere.get_atmosphere_mask(
            stacked=True, invert=False
        )
        xy_atmosphere = datastore_atmosphere.get_xy("forcing", stacked=True)
        atmosphere_lon_lat = xy_atmosphere[atmosphere_mask]

    # Plotting is in 3d; grid positions (lon, lat)
    grid_lon_lat = np.concatenate(
        (interior_lon_lat, boundary_lon_lat)
        + ((atmosphere_lon_lat,) if atmosphere_lon_lat is not None else ()),
        axis=0,
    )  # (num_grid, 2)

    # Optionally create corner filter
    if args.corner_filter_radius is not None:
        # Define corner in terms of last point (lon, lat)
        corner = interior_lon_lat[-1]
        lon_corner, lat_corner = corner

        def corner_filter_func(pos_lon_lat):
            """
            pos_lon_lat is (N, 2) [longitude, latitude].
            Measure distance using haversine.
            """
            lon_pos = pos_lon_lat[:, 0]
            lat_pos = pos_lon_lat[:, 1]

            lon_rad_corner, lat_rad_corner, lon_rad_pos, lat_rad_pos = map(
                np.radians, [lon_corner, lat_corner, lon_pos, lat_pos]
            )

            dlon_rad = lon_rad_pos - lon_rad_corner
            dlat_rad = lat_rad_pos - lat_rad_corner

            hav_interm = (
                np.sin(dlat_rad / 2.0) ** 2
                + np.cos(lat_rad_corner)
                * np.cos(lat_rad_pos)
                * np.sin(dlon_rad / 2.0) ** 2
            )
            hav_rad = 2 * np.arcsin(np.sqrt(hav_interm))
            hav_m = 6378137.0 * hav_rad

            return hav_m <= args.corner_filter_radius

    else:
        corner_filter_func = None

    mesh_edge_width = (
        args.edge_width
        if args.mesh_edge_width is None
        else args.mesh_edge_width
    )

    # Add plotting objects to this list
    data_objs = []

    # Plot grid nodes
    if boundary_forced:
        # Create separate plot objects for interior and boundary
        data_objs.append(
            vis.create_node_plot(
                interior_lon_lat,
                "Interior grid Nodes",
                color=args.grid_color,
                radius=GRID_RADIUS,
                size=args.grid_node_size,
                pos_filter_func=corner_filter_func,
            )
        )
        data_objs.append(
            vis.create_node_plot(
                boundary_lon_lat,
                "Boundary grid Nodes",
                color=args.boundary_grid_color,
                radius=GRID_RADIUS,
                size=args.grid_node_size,
                pos_filter_func=corner_filter_func,
            )
        )
        if args.use_atmosphere_g2m:
            data_objs.append(
                vis.create_node_plot(
                    atmosphere_lon_lat,
                    "Atmospheric grid Nodes",
                    color=args.atmosphere_grid_color,
                    radius=GRID_RADIUS,
                    size=args.grid_node_size,
                    pos_filter_func=corner_filter_func,
                )
            )
    else:
        # All grid nodes together
        data_objs.append(
            vis.create_node_plot(
                grid_lon_lat,
                "Grid Nodes",
                color=args.grid_color,
                radius=GRID_RADIUS,
                size=args.grid_node_size,
                pos_filter_func=corner_filter_func,
            )
        )

    # Radius
    mesh_radius = GRID_RADIUS + args.mesh_height

    # Mesh positioning and edges to plot differ if we have a hierarchical graph
    if hierarchical:
        # Make edge_index to numpy
        def tensor_list_to_numpy(tensor_list):
            """Helper function to make list of tensors numpy arrays"""
            return [elem.numpy() for elem in tensor_list]

        m2m_edge_index = tensor_list_to_numpy(graph_ldict["m2m_edge_index"])
        mesh_lon_lat_level = tensor_list_to_numpy(graph_ldict["mesh_lon_lat"])
        mesh_up_edge_index = tensor_list_to_numpy(
            graph_ldict["mesh_up_edge_index"]
        )
        mesh_down_edge_index = tensor_list_to_numpy(
            graph_ldict["mesh_down_edge_index"]
        )

        # Iterate over levels, adding all nodes and edges
        for bot_level_i, intra_ei in enumerate(
            m2m_edge_index,
        ):
            # Extract position and radius
            top_level_i = bot_level_i + 1
            bot_pos = mesh_lon_lat_level[bot_level_i]
            bot_radius = mesh_radius + bot_level_i * args.mesh_level_dist

            # Mesh nodes at bottom level
            data_objs.append(
                vis.create_node_plot(
                    bot_pos,
                    f"Mesh level {bot_level_i} nodes",
                    color=args.mesh_color,
                    radius=bot_radius,
                    size=args.mesh_node_size,
                    pos_filter_func=corner_filter_func,
                )
            )
            # Intra-level edges at bottom level
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
                    pos_filter_func=corner_filter_func,
                )
            )

            # Do add include up/down edges for top level
            if top_level_i < len(m2m_edge_index):
                up_ei = mesh_up_edge_index[bot_level_i]
                down_ei = mesh_down_edge_index[bot_level_i]
                top_pos = mesh_lon_lat_level[top_level_i]
                top_radius = mesh_radius + (top_level_i) * args.mesh_level_dist

                # Up edges
                data_objs.append(
                    vis.create_edge_plot(
                        up_ei,
                        bot_pos,
                        top_pos,
                        f"Mesh up {bot_level_i}->{top_level_i} edges",
                        color=args.mesh_color,
                        width=mesh_edge_width,
                        from_radius=bot_radius,
                        to_radius=top_radius,
                        pos_filter_func=corner_filter_func,
                    )
                )
                # Down edges
                data_objs.append(
                    vis.create_edge_plot(
                        down_ei,
                        top_pos,
                        bot_pos,
                        f"Mesh down {top_level_i}->{bot_level_i} edges",
                        color=args.mesh_color,
                        width=mesh_edge_width,
                        from_radius=top_radius,
                        to_radius=bot_radius,
                        pos_filter_func=corner_filter_func,
                    )
                )

        # Connect g2m and m2g only to bottom level
        grid_con_lon_lat = mesh_lon_lat_level[0]
    else:
        # Non-hierarchical
        mesh_lon_lat = graph_ldict["mesh_lon_lat"][0].numpy()
        m2m_edge_index = graph_ldict["m2m_edge_index"].numpy()

        # Calculate degree-dependent node sizes
        num_mesh_nodes = mesh_lon_lat.shape[0]
        mesh_degrees = np.zeros(num_mesh_nodes, dtype=int)
        # Count edges where node is source (outgoing)
        np.add.at(mesh_degrees, m2m_edge_index[0], 1)
        # Count edges where node is target (incoming)
        np.add.at(mesh_degrees, m2m_edge_index[1], 1)
        # Use degrees to scale node sizes
        if mesh_degrees.max() > 0:
            # Normalize degrees to [0.5, 2.0] range and multiply by base size
            normalized_degrees = 0.5 + 1.5 * (mesh_degrees / mesh_degrees.max())
            mesh_node_size = normalized_degrees * args.mesh_node_size
        else:
            mesh_node_size = args.mesh_node_size

        data_objs.append(
            vis.create_node_plot(
                mesh_lon_lat,
                "Mesh Nodes",
                radius=mesh_radius,
                color=args.mesh_color,
                size=mesh_node_size,
                pos_filter_func=corner_filter_func,
            )
        )
        data_objs.append(
            vis.create_edge_plot(
                m2m_edge_index,
                mesh_lon_lat,
                mesh_lon_lat,
                "Mesh Edges",
                from_radius=mesh_radius,
                to_radius=mesh_radius,
                color=args.mesh_color,
                width=mesh_edge_width,
                pos_filter_func=corner_filter_func,
            )
        )

        grid_con_lon_lat = mesh_lon_lat

    # Plot G2M
    data_objs.append(
        vis.create_edge_plot(
            g2m_edge_index,
            grid_lon_lat,
            grid_con_lon_lat,
            "G2M Edges",
            color=args.g2m_color,
            width=args.edge_width,
            from_radius=GRID_RADIUS,
            to_radius=mesh_radius,
            pos_filter_func=corner_filter_func,
        )
    )

    # Plot M2G
    data_objs.append(
        vis.create_edge_plot(
            m2g_edge_index,
            grid_con_lon_lat,
            grid_lon_lat,
            "M2G Edges",
            color=args.m2g_color,
            width=args.edge_width,
            from_radius=mesh_radius,
            to_radius=GRID_RADIUS,
            pos_filter_func=corner_filter_func,
        )
    )

    # Plot earth
    data_objs.append(
        vis.make_earth(radius=1, resolution_reduction=args.texture_resolution)
    )

    fig = go.Figure(data=data_objs)

    fig.update_layout(scene_aspectmode="data")
    fig.update_traces(connectgaps=False)

    if not args.show_axis:
        # Hide axis
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
