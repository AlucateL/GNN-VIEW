# Standard library
from typing import Optional

# Third-party
import numpy as np
import scipy.spatial
import torch
import torch_geometric as pyg

# Local
from . import utils as gutils
from .spherical_kmeans import SphericalKMeans


def _cart_to_lon_lat_matching_utils(cart: np.ndarray) -> np.ndarray:
    """Convert 3D Cartesian to (lon, lat) in degrees."""
    r = np.linalg.norm(cart, axis=1, keepdims=True)
    cart = cart / (r + 1e-12)
    # z = sin(lat), x = cos(lat)*cos(lon), y = cos(lat)*sin(lon)
    lat_rad = np.arcsin(np.clip(cart[:, 2], -1, 1))
    lon_rad = np.arctan2(cart[:, 1], cart[:, 0])
    lon = np.rad2deg(lon_rad)
    lat = np.rad2deg(lat_rad)
    return np.stack([lon, lat], axis=1).astype(np.float32)


def _knn_edges_sphere(mesh_3d: np.ndarray, k: int) -> np.ndarray:
    """Build (2, E) edge index from k-NN on unit sphere (chord distance).
    Each node connects to its k nearest neighbors; edges are undirected
    (both directions). Guarantees every node has at least one edge.
    """
    n = mesh_3d.shape[0]
    k = min(k, n - 1)
    if k < 1:
        return np.zeros((2, 0), dtype=np.int64)
    kdt = scipy.spatial.cKDTree(mesh_3d)
    dists, nn = kdt.query(mesh_3d, k=k + 1)  # k+1 to exclude self
    # nn shape (n, k+1); column 0 is self (dist 0)
    edges = set()
    for i in range(n):
        for j in nn[i, 1:]:  # skip self
            edges.add((min(i, int(j)), max(i, int(j))))
    edges = np.array(list(edges), dtype=np.int64).T
    edges_both = np.concatenate([edges, edges[[1, 0]]], axis=1)
    return edges_both


def _spherical_delaunay_edges(mesh_3d: np.ndarray) -> np.ndarray:
    """Build (2, E) edge index from spherical Delaunay triangulation.

    Uses ConvexHull of points on the unit sphere: its surface facets are
    the Delaunay triangles on the sphere. mesh_3d: (N, 3) unit-norm points.
    Returns directed edges (both directions) for each undirected edge.
    """
    n = mesh_3d.shape[0]
    if n < 3:
        return np.zeros((2, 0), dtype=np.int64)
    hull = scipy.spatial.ConvexHull(mesh_3d)
    # In 3D, simplices are the triangular faces of the hull (n_facets, 3)
    edges = set()
    for tri in hull.simplices:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        edges.add((min(i, j), max(i, j)))
        edges.add((min(j, k), max(j, k)))
        edges.add((min(k, i), max(k, i)))
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    edges_arr = np.array(list(edges), dtype=np.int64).T  # (2, E_undir)
    edges_both = np.concatenate([edges_arr, edges_arr[[1, 0]]], axis=1)
    return edges_both


def build_graph_from_mesh_pos_sphere(
    mesh_xy: np.ndarray,
    k_nn: int = 5,
) -> pyg.data.Data:
    """Build mesh graph from node positions on sphere using k-NN.

    mesh_xy: (N, 2) [longitude, latitude] in degrees (x=lon, y=lat).
    Edges from k-NN in 3D (chord distance on unit sphere) so every node
    has same-level neighbors (avoids isolated nodes from ConvexHull interior).
    """
    mesh_3d = gutils.node_lon_lat_to_cart(mesh_xy)
    edge_index = _knn_edges_sphere(mesh_3d, k=k_nn)
    pos = torch.tensor(mesh_xy, dtype=torch.float32)
    graph = pyg.data.Data(
        pos=pos,
        edge_index=torch.from_numpy(edge_index).long(),
    )
    return graph


def build_graph_from_mesh_pos_sphere_delaunay(
    mesh_xy: np.ndarray,
) -> pyg.data.Data:
    """Build mesh graph from node positions on sphere using spherical Delaunay.

    Uses ConvexHull of 3D points on the unit sphere; surface facets are the
    Delaunay triangles. mesh_xy: (N, 2) [longitude, latitude] in degrees.
    """
    mesh_3d = gutils.node_lon_lat_to_cart(mesh_xy)
    edge_index = _spherical_delaunay_edges(mesh_3d)
    pos = torch.tensor(mesh_xy, dtype=torch.float32)
    graph = pyg.data.Data(
        pos=pos,
        edge_index=torch.from_numpy(edge_index).long(),
    )
    return graph


def build_cluster_mesh_graph_global(
    sea_xy: np.ndarray,
    land_xy: np.ndarray,
    mesh_refinement_factor: float = 9,
    grid_to_first_mesh_refinement: float = 9,
    limit_mesh_levels: Optional[int] = None,
    mesh_plot_function=None,
    random_state: int = 42,
    max_chord_len: float = 0.1,
):
    """Build hierarchical cluster mesh over the globe (sea points only).

    Uses KMeans in 3D Cartesian (unit sphere) so clusters respect spherical
    geometry. Same-level mesh edges from spherical Delaunay (ConvexHull of
    points on sphere); Edges crossing land are then filtered.

    Parameters
    ----------
    sea_xy : (N_sea, 2) [longitude, latitude] in degrees (interior grid points)
    land_xy : (N_land, 2) [longitude, latitude] in degrees (land grid points)
    mesh_refinement_factor : factor between levels
    grid_to_first_mesh_refinement : ratio grid nodes / first-level mesh nodes
    limit_mesh_levels : max number of levels (default: from formula)
    mesh_plot_function : optional callback(level_graph, title)
    random_state : for KMeans
    max_chord_len : max edge chord length on unit sphere for land filter (~0.1
        corresponds to ~5.7° great-circle arc)

    Returns
    -------
    mesh_pos : list of (N_i, 3) tensors, 3D Cartesian per level
    bottom_mesh : pyg Data (pos 3D, edge_index, len, vdiff) for finest level
    save_graphs : dict with "m2m", "mesh_up", "mesh_down"
    """
    n_sea = sea_xy.shape[0]
    possible_mesh_levels = int(
        np.floor(
            np.log(n_sea / grid_to_first_mesh_refinement)
            / np.log(mesh_refinement_factor)
        )
    )
    if limit_mesh_levels is None:
        num_mesh_levels = possible_mesh_levels
    else:
        num_mesh_levels = min(possible_mesh_levels, limit_mesh_levels)
    num_mesh_levels = max(1, num_mesh_levels)

    # Equal-area weights; must be strictly positive for k-means++ (cos(90°)
    # can be tiny negative in float, and zeros make probabilities invalid)
    lat_weights = np.cos(np.deg2rad(sea_xy[:, 1].astype(np.float32)))
    lat_weights = np.maximum(lat_weights, 1e-10)
    lat_weights = np.where(np.isfinite(lat_weights), lat_weights, 1.0)
    sea_3d = gutils.node_lon_lat_to_cart(sea_xy)

    mesh_level_graphs = []
    mesh_up_graphs = []
    mesh_down_graphs = []

    for level_i in range(num_mesh_levels):
        print(f"Running KMeans for global cluster level {level_i}...")
        if level_i == 0:
            prev_level_pos = sea_3d
            prev_level_weights = lat_weights
            num_clusters = int(np.round(n_sea / grid_to_first_mesh_refinement))
            num_clusters = max(4, num_clusters)  # ConvexHull needs >= 4 in 3D
        else:
            prev_level_pos = mesh_level_graphs[-1].pos.numpy()  # (N, 3)
            prev_level_weights = None
            num_clusters = int(
                np.round(prev_level_pos.shape[0] / mesh_refinement_factor)
            )
            num_clusters = max(2, num_clusters)

        kmeans = SphericalKMeans(
            n_clusters=num_clusters,
            init="k-means++",
            n_init=1,
            random_state=random_state,
        )
        kmeans.fit(prev_level_pos, sample_weight=prev_level_weights)
        centers_3d = kmeans.cluster_centers_
        r = np.linalg.norm(centers_3d, axis=1, keepdims=True)
        centers_3d = centers_3d / (r + 1e-12)
        level_lon_lat = _cart_to_lon_lat_matching_utils(centers_3d)

        level_graph = build_graph_from_mesh_pos_sphere_delaunay(level_lon_lat)
        mesh_cart = gutils.node_lon_lat_to_cart(level_graph.pos.numpy())
        edge_index_np = level_graph.edge_index.numpy()
        # Scale max chord length by level
        max_chord_len_level = max_chord_len * (
            mesh_refinement_factor ** (0.5 * level_i)
        )
        mesh_cart_f, edge_index_f = gutils.filter_global_edges_land(
            mesh_cart,
            edge_index_np,
            sea_xy,
            land_xy,
            max_chord_len=max_chord_len_level,
        )
        level_graph.pos = torch.from_numpy(mesh_cart_f.astype(np.float32))
        level_graph.edge_index = torch.from_numpy(edge_index_f.astype(np.int64))
        gutils.add_edge_features_pyg(level_graph)
        mesh_level_graphs.append(level_graph)

        n_nodes = mesh_cart_f.shape[0]
        n_edges = edge_index_f.shape[1]
        if n_edges == 0:
            print(f"Mesh level {level_i}: {n_nodes} nodes, 0 edges")
        else:
            chord = np.linalg.norm(
                mesh_cart_f[edge_index_f[1]] - mesh_cart_f[edge_index_f[0]],
                axis=1,
            )
            dm_lvl = float(np.mean(chord))
            print(
                f"Mesh level {level_i}: {n_nodes} nodes, "
                f"mean edge length (chord) = {dm_lvl:.6f}"
            )

        if mesh_plot_function is not None:
            mesh_plot_function(level_graph, f"Mesh graph, level {level_i}")

        if level_i > 0:
            # Build up/down edges after filtering:
            # coarse->nearest fine, fine->nearest coarse (3D)
            coarse_pos = mesh_level_graphs[level_i - 1].pos.numpy()
            fine_pos = level_graph.pos.numpy()
            n_prev = coarse_pos.shape[0]
            kdt_fine = scipy.spatial.cKDTree(fine_pos)
            _, coarse_to_fine = kdt_fine.query(coarse_pos, k=1)
            coarse_to_fine = np.asarray(coarse_to_fine, dtype=np.int64).ravel()
            up_edge_index = torch.stack(
                (
                    torch.arange(n_prev, dtype=torch.long),
                    torch.from_numpy(coarse_to_fine) + n_prev,
                ),
                dim=0,
            )
            up_graph = pyg.data.Data(
                edge_index=up_edge_index,
                pos=torch.cat(
                    (
                        mesh_level_graphs[level_i - 1].pos,
                        level_graph.pos,
                    ),
                    dim=0,
                ),
            )
            gutils.add_edge_features_pyg(up_graph)
            mesh_up_graphs.append(up_graph)

            # Down = reverse of up (same edges, level 1 -> level 0)
            down_edge_index = torch.stack(
                (up_edge_index[1], up_edge_index[0]), dim=0
            )
            down_graph = pyg.data.Data(
                edge_index=down_edge_index,
                pos=up_graph.pos,
            )
            gutils.add_edge_features_pyg(down_graph)
            mesh_down_graphs.append(down_graph)

            if mesh_plot_function is not None:
                mesh_plot_function(
                    down_graph, f"Down graph, {level_i} -> {level_i - 1}"
                )
                mesh_plot_function(
                    up_graph, f"Up graph, {level_i - 1} -> {level_i}"
                )

    mesh_pos = [g.pos for g in mesh_level_graphs]
    save_graphs = {
        "m2m": mesh_level_graphs,
        "mesh_up": mesh_up_graphs,
        "mesh_down": mesh_down_graphs,
    }
    bottom_mesh = mesh_level_graphs[0]
    return mesh_pos, bottom_mesh, save_graphs
