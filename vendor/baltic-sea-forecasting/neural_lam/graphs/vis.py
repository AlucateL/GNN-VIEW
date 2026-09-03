# Standard library
import os

# Third-party
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch_geometric as pyg
from PIL import Image

# Local
from . import utils as gutils

# https://community.plotly.com/t/whats-the-efficient-way-to-create-3d-scatter-plot-for-millions-of-points/60965/4
NODE_PLOT_LIMIT = (
    100_000  # Limit on number of points to plot before subsampling
)


def plot_graph(
    graph,
    title=None,
    graph_dir_path=None,
    reindex_edges=True,
    order_by_degree=False,
):
    fig, axis = plt.subplots(figsize=(8, 8), dpi=200)  # W,H
    edge_index = graph.edge_index
    pos = graph.pos

    # Fix for re-indexed edge indices only containing mesh nodes at
    # higher levels in hierarchy
    if reindex_edges and edge_index.numel() > 0:
        edge_index = edge_index - edge_index.min()

    if pyg.utils.is_undirected(edge_index):
        # Keep only 1 direction of edge_index
        edge_index = edge_index[:, edge_index[0] < edge_index[1]]  # (2, M/2)

    # Move all to cpu and numpy, compute (in)-degrees
    degrees = (
        pyg.utils.degree(edge_index[1], num_nodes=pos.shape[0]).cpu().numpy()
    )
    edge_index = edge_index.cpu().numpy()
    pos = pos.cpu().numpy()

    # Plot edges
    from_pos = pos[edge_index[0]]  # (M/2, 2)
    to_pos = pos[edge_index[1]]  # (M/2, 2)
    edge_lines = np.stack((from_pos, to_pos), axis=1)
    lc = matplotlib.collections.LineCollection(
        edge_lines, lw=0.3, colors="black", zorder=1, rasterized=True
    )
    axis.add_collection(lc)

    # Plot nodes
    if order_by_degree:
        order = np.argsort(degrees)
        pos = pos[order]
        degrees = degrees[order]
    node_scatter = axis.scatter(
        pos[:, 0],
        pos[:, 1],
        c=degrees,
        s=2,
        marker="o",
        zorder=2,
        cmap="viridis",
        clim=None,
        rasterized=True,
    )

    plt.colorbar(node_scatter, aspect=50)

    if title is not None:
        axis.set_title(title)

    if graph_dir_path is not None:
        plt.savefig(os.path.join(graph_dir_path, f"{title}.png"))
    plt.close(fig)


def plot_node_values(
    graph,
    title=None,
    graph_dir_path=None,
    node_values=None,
    value_name="Voronoi area",
):
    fig, axis = plt.subplots(figsize=(8, 8), dpi=200)
    edge_index = graph.edge_index
    pos = graph.pos

    if edge_index.numel() > 0:
        edge_index = edge_index - edge_index.min()
    if pyg.utils.is_undirected(edge_index):
        edge_index = edge_index[:, edge_index[0] < edge_index[1]]

    edge_index = edge_index.cpu().numpy()
    pos = pos.cpu().numpy()

    values = np.ravel(
        node_values.cpu().numpy()
        if hasattr(node_values, "cpu")
        else np.asarray(node_values)
    )

    edge_lines = np.stack((pos[edge_index[0]], pos[edge_index[1]]), axis=1)
    axis.add_collection(
        matplotlib.collections.LineCollection(
            edge_lines, lw=0.3, colors="black", zorder=1, rasterized=True
        )
    )

    s = axis.scatter(
        pos[:, 0],
        pos[:, 1],
        c=values,
        s=2,
        cmap="viridis",
        zorder=2,
        rasterized=True,
    )
    cbar = plt.colorbar(s, aspect=50)
    cbar.set_label(value_name)
    if title is not None:
        axis.set_title(title)
    if graph_dir_path is not None:
        plt.savefig(os.path.join(graph_dir_path, f"{title}.png"))
    plt.close(fig)


def make_earth(radius, resolution_reduction=1.0):
    """
    Plotly earth from
    https://community.plotly.com/t/applying-full-color-image-texture-to-create-an-interactive-earth-globe/60166

    radius: radius of earth in plot
    resolution: float, percentage of full resolution
    """
    earth_colorscale = [
        [0.0, "rgb(30, 59, 117)"],
        [0.1, "rgb(46, 68, 21)"],
        [0.2, "rgb(74, 96, 28)"],
        [0.3, "rgb(115,141,90)"],
        [0.4, "rgb(122, 126, 75)"],
        [0.6, "rgb(122, 126, 75)"],
        [0.7, "rgb(141,115,96)"],
        [0.8, "rgb(223, 197, 170)"],
        [0.9, "rgb(237,214,183)"],
        [1.0, "rgb(255, 255, 255)"],
    ]
    texture_path = "figures/earth_texture.jpeg"
    img = Image.open(texture_path)

    # Calculate new width to maintain aspect ratio
    new_width = int(img.width * resolution_reduction)
    new_height = int(img.height * resolution_reduction)

    # Resize image preserving aspect ratio
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    texture = np.asarray(img_resized).T

    N_lon = int(texture.shape[0])
    N_lat = int(texture.shape[1])
    theta = np.linspace(-np.pi, np.pi, N_lon)
    phi = np.linspace(0, np.pi, N_lat)

    # Set up coordinates for points on the sphere
    x0 = radius * np.outer(np.cos(theta), np.sin(phi))
    y0 = radius * np.outer(np.sin(theta), np.sin(phi))
    z0 = radius * np.outer(np.ones(N_lon), np.cos(phi))

    return go.Surface(
        x=x0,
        y=y0,
        z=z0,
        surfacecolor=texture,
        colorscale=earth_colorscale,
        name="Earth",
        showscale=False,
        showlegend=True,
    )


def create_edge_plot(
    edge_index,
    from_node_xy,
    to_node_xy,
    label,
    color="blue",
    width=1,
    from_radius=1,
    to_radius=1,
    pos_filter_func=None,
):
    """
    Create a plotly object showing edges.

    edge_index: (2, M)
    from_node_xy: (N, 2) [longitude, latitude] in degrees (x=lon, y=lat)
    to_node_xy: (N, 2) [longitude, latitude] in degrees
    label: str, label of plot object
    """
    from_node_cart = gutils.node_lon_lat_to_cart(from_node_xy) * from_radius
    to_node_cart = gutils.node_lon_lat_to_cart(to_node_xy) * to_radius

    edge_start = from_node_cart[edge_index[0]]  # (M, 2)
    edge_end = to_node_cart[edge_index[1]]  # (M, 2)

    if pos_filter_func is not None:
        # Filter edges
        edge_start_xy = from_node_xy[edge_index[0]]  # (M, 2)
        edge_end_xy = to_node_xy[edge_index[1]]  # (M, 2)

        edge_mask = np.logical_and(
            pos_filter_func(edge_start_xy),
            pos_filter_func(edge_end_xy),
        )
        edge_start = edge_start[edge_mask]
        edge_end = edge_end[edge_mask]

    n_edges = edge_start.shape[0]

    x_edges = np.stack(
        (edge_start[:, 0], edge_end[:, 0], np.full(n_edges, None)), axis=1
    ).flatten()
    y_edges = np.stack(
        (edge_start[:, 1], edge_end[:, 1], np.full(n_edges, None)), axis=1
    ).flatten()
    z_edges = np.stack(
        (edge_start[:, 2], edge_end[:, 2], np.full(n_edges, None)), axis=1
    ).flatten()

    return go.Scatter3d(
        x=x_edges,
        y=y_edges,
        z=z_edges,
        mode="lines",
        line={"color": color, "width": width},
        name=label,
    )


def create_node_plot(
    node_xy, label, color="blue", size=1, radius=1, pos_filter_func=None
):
    """
    Create a plotly object showing nodes.

    node_xy: (N, 2) [longitude, latitude] in degrees (x=lon, y=lat)
    label: str, label of plot object
    """
    node_pos = gutils.node_lon_lat_to_cart(node_xy) * radius
    if pos_filter_func is not None:
        # Filter nodes before plotting
        node_pos = node_pos[pos_filter_func(node_xy)]

    # Plotly 3d can not render large amounts of points in some browsers, so
    # for very large node sets we need to somehow subsample it before plotting.
    # This is a simple solution
    num_nodes = node_pos.shape[0]
    subsample = num_nodes > NODE_PLOT_LIMIT
    if subsample:
        # Figure out how much to subsample by
        subsampling_factor = int(num_nodes / NODE_PLOT_LIMIT)
        node_pos = node_pos[::subsampling_factor]  # Simple subsampling

    return go.Scatter3d(
        x=node_pos[:, 0],
        y=node_pos[:, 1],
        z=node_pos[:, 2],
        mode="markers",
        marker={"color": color, "size": size},
        name=f"{label} (subsampled)" if subsample else label,
    )


def plot_disconnected_nodes(
    pos,
    is_mesh,
    is_any_grid,
    disc_grid_indices,
    disc_mesh_indices,
    graph_dir_path,
    title="g2m_disconnected",
):
    """
    Plot disconnected nodes in g2m graph.

    Parameters
    ----------
    pos : np.ndarray
        Node positions, shape (N, 2)
    is_mesh : np.ndarray
        Boolean mask for mesh nodes
    is_any_grid : np.ndarray
        Boolean mask for all grid nodes
    disc_grid_indices : np.ndarray
        Indices of disconnected grid nodes
    disc_mesh_indices : np.ndarray
        Indices of disconnected mesh nodes
    graph_dir_path : str
        Path to save plot
    title : str
        Title and filename for plot
    """
    fig, axis = plt.subplots(figsize=(8, 8), dpi=200)

    # Plot all grid nodes
    axis.scatter(
        pos[is_any_grid, 0],
        pos[is_any_grid, 1],
        s=1,
        label="all grids",
        color="lightblue",
    )

    # Plot all mesh nodes
    axis.scatter(
        pos[is_mesh, 0],
        pos[is_mesh, 1],
        s=1,
        label="mesh",
        color="green",
    )

    # Plot disconnected grid nodes
    if len(disc_grid_indices) > 0:
        axis.scatter(
            pos[disc_grid_indices, 0],
            pos[disc_grid_indices, 1],
            s=6,
            label="disconnected grid",
            color="red",
        )

    # Plot disconnected mesh nodes
    if len(disc_mesh_indices) > 0:
        axis.scatter(
            pos[disc_mesh_indices, 0],
            pos[disc_mesh_indices, 1],
            s=8,
            label="disconnected mesh",
            color="darkred",
        )

    axis.legend(loc="upper left")
    axis.set_title(title)

    plt.savefig(os.path.join(graph_dir_path, f"{title}.png"))
    plt.close()
