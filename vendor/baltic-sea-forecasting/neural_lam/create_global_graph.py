# Standard library
import os
from argparse import ArgumentParser
from types import SimpleNamespace
from typing import Optional, Tuple

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
import torch
import torch_geometric as pyg
import zarr
from torch_geometric.utils import degree

# Local
from .config import load_config_and_datastores
from .datastore.base import BaseRegularGridDatastore
from .graphs import global_cluster_mesh, global_icosahedral_mesh, saving
from .graphs import utils as gutils
from .graphs import vis


def load_grid_from_zarr(dataset_path: str) -> np.ndarray:
    """Load grid from zarr. Expects 'longitude' and 'latitude' arrays.

    Handles 1D (n_points,) or 2D (n_lon, n_lat) arrays. Returns (n_points, 2)
    with columns [longitude, latitude] in degrees (x=lon, y=lat convention).
    """
    root = zarr.open(dataset_path, mode="r")
    lon = np.asarray(root["longitude"]).astype(np.float32)
    lat = np.asarray(root["latitude"]).astype(np.float32)
    if lon.ndim == 1 and lat.ndim == 1:
        if lon.shape[0] == lat.shape[0]:
            return np.stack([lon, lat], axis=1)
        # Assume 2D grid: lon (n_lon,), lat (n_lat,)
        lon_2d, lat_2d = np.meshgrid(lon, lat)
        return np.stack([lon_2d.ravel(), lat_2d.ravel()], axis=1)
    if lon.ndim == 2 and lat.ndim == 2:
        return np.stack([lon.ravel(), lat.ravel()], axis=1)
    raise ValueError(
        f"Unsupported longitude/latitude shapes: {lon.shape}, {lat.shape}"
    )


def _lon_lat_to_node_features(lon_lat_deg: np.ndarray) -> np.ndarray:
    """
    Returns (sin(lon), cos(lon), sin(lat), cos(lat)) as (N, 4).
    Convention: lon = column 0, lat = column 1.
    """
    lon_rad = np.deg2rad(lon_lat_deg[:, 0].astype(np.float64))
    lat_rad = np.deg2rad(lon_lat_deg[:, 1].astype(np.float64))
    return np.stack(
        [
            np.sin(lon_rad),
            np.cos(lon_rad),
            np.sin(lat_rad),
            np.cos(lat_rad),
        ],
        axis=1,
    ).astype(np.float32)


def load_grid_from_datastore(
    datastore: BaseRegularGridDatastore,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load interior and land grid for global graph.

    Returns
    -------
    sea_xy : (n_sea, 2) interior points, [longitude, latitude]
    land_xy : (n_land, 2) land points, [longitude, latitude]
    """
    interior_mask = datastore.get_mask(surface=True, stacked=True, invert=False)
    xy = datastore.get_xy("state", stacked=True)  # (lon, lat) = (x, y)
    sea_xy = xy[interior_mask].astype(np.float32)
    land_xy = xy[~interior_mask].astype(np.float32)
    return sea_xy, land_xy


def _cart_to_node_features(mesh_cart: np.ndarray) -> np.ndarray:
    """
    Convert (N, 3) Cartesian on unit sphere to
    (sin_lon, cos_lon, sin_lat, cos_lat).
    """
    r = np.linalg.norm(mesh_cart, axis=1, keepdims=True)
    mesh_cart = mesh_cart / (r + 1e-12)
    x, y, z = mesh_cart[:, 0], mesh_cart[:, 1], mesh_cart[:, 2]
    lon_rad = np.arctan2(y, x)
    lat_rad = np.arcsin(np.clip(z, -1.0, 1.0))
    return np.stack(
        [
            np.sin(lon_rad),
            np.cos(lon_rad),
            np.sin(lat_rad),
            np.cos(lat_rad),
        ],
        axis=1,
    ).astype(np.float32)


def _combine_icosahedral_multiscale_to_fine(
    mesh_levels: list[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Combine edges from multiple icosahedral levels onto finest node set.

    Parameters
    ----------
    mesh_levels : list of (cart, edge_index), ordered finest -> coarsest.

    Returns
    -------
    fine_cart : (N_fine, 3)
    combined_edge_index : (2, E_combined) directed edges on finest node indices
    """
    fine_cart, fine_ei = mesh_levels[0]
    if len(mesh_levels) == 1:
        return fine_cart, fine_ei

    fine_kdt = scipy.spatial.cKDTree(fine_cart)

    edge_set = {
        (int(src), int(dst))
        for src, dst in zip(fine_ei[0], fine_ei[1])
        if src != dst
    }

    for lvl_cart, lvl_ei in mesh_levels[1:]:
        _, lvl_to_fine = fine_kdt.query(lvl_cart, k=1)
        lvl_to_fine = np.asarray(lvl_to_fine, dtype=np.int64).ravel()

        src_mapped = lvl_to_fine[lvl_ei[0]]
        dst_mapped = lvl_to_fine[lvl_ei[1]]
        for src, dst in zip(src_mapped, dst_mapped):
            if src == dst:
                continue
            edge_set.add((int(src), int(dst)))

    if len(edge_set) == 0:
        combined_ei = np.zeros((2, 0), dtype=np.int64)
    else:
        combined_ei = np.array(list(edge_set), dtype=np.int64).T

    return fine_cart, combined_ei


def create_global_graph(
    graph_dir_path: str,
    grid_xy: np.ndarray,
    splits: int = 3,
    levels: Optional[int] = None,
    graph_type: str = "multiscale",
    g2m_radius: float = 0.67,
    m2g_k: int = 3,
    create_plot: bool = False,
    connect_disconnected: bool = False,
    sea_xy: Optional[np.ndarray] = None,
    land_xy: Optional[np.ndarray] = None,
    max_chord_len: float = 0.1,
    mesh_refinement_factor: float = 9,
    grid_to_first_mesh_refinement: float = 9,
):
    """Create global graph: icosahedral mesh + g2m + m2g.

    grid_xy, sea_xy, land_xy: (N, 2) [longitude, latitude] in degrees.

    Node convention for loader: mesh 0..N_mesh-1, grid N_mesh..N_mesh+N_grid-1.
    When sea_xy and land_xy are provided, m2g edges over land are filtered.
    """
    os.makedirs(graph_dir_path, exist_ok=True)
    print(f"Writing global graph to {graph_dir_path}")

    # Build mesh: icosahedral or cluster
    if graph_type == "cluster":
        if sea_xy is None or land_xy is None:
            raise ValueError("graph_type=cluster requires sea_xy and land_xy")
        mesh_pos_list, bottom_mesh, save_graphs_cluster = (
            global_cluster_mesh.build_cluster_mesh_graph_global(
                sea_xy,
                land_xy,
                mesh_refinement_factor=mesh_refinement_factor,
                grid_to_first_mesh_refinement=grid_to_first_mesh_refinement,
                limit_mesh_levels=levels,
                max_chord_len=max_chord_len,
            )
        )
        mesh_cart = bottom_mesh.pos.numpy()
        mesh_edge_index = bottom_mesh.edge_index.numpy()
        num_mesh = mesh_cart.shape[0]
        num_grid = grid_xy.shape[0]
        hierarchical_cluster = len(mesh_pos_list) > 1
    else:
        # Icosahedral (multiscale or hierarchical)
        if levels is None:
            levels = 1 if graph_type == "multiscale" else (splits + 1)
        mesh_levels = (
            global_icosahedral_mesh.get_hierarchy_of_triangular_meshes(
                splits=splits, levels=levels
            )
        )
        # get_hierarchy_of_triangular_meshes returns coarsest->finest.
        # Our hierarchical models assume level 0 is the bottom one
        mesh_levels = list(reversed(mesh_levels))  # finest->coarsest
        mesh_levels = [(c, e) for c, _, e in mesh_levels]  # (cart, ei) only

        # Diagnostics: mean chord per level
        for level_i, (lvl_cart, lvl_ei) in enumerate(mesh_levels):
            if lvl_ei.shape[1] == 0:
                print(
                    f"Mesh level {level_i}: {lvl_cart.shape[0]} nodes, 0 edges"
                )
                continue
            dm_lvl = float(
                np.mean(
                    global_icosahedral_mesh.mesh_edge_lengths_cart(
                        lvl_cart, lvl_ei
                    )
                )
            )
            print(
                f"Mesh level {level_i}: {lvl_cart.shape[0]} nodes, "
                f"mean edge length (chord) = {dm_lvl:.6f}"
            )

        mesh_levels_filtered = []
        if sea_xy is not None and land_xy is not None and land_xy.shape[0] > 0:
            for level_i, (lvl_cart, lvl_ei) in enumerate(mesh_levels):
                f_cart, f_ei = gutils.filter_global_edges_land(
                    lvl_cart, lvl_ei, sea_xy, land_xy
                )
                mesh_levels_filtered.append((f_cart, f_ei))
                print(
                    f"Filtered icosahedral mesh level {level_i} to sea-only: "
                    f"{f_cart.shape[0]} nodes"
                )
            mesh_levels = mesh_levels_filtered

        # Bottom (level 0) mesh is the finest. For multiscale,
        # combine edges from coarser levels onto this finest node set.
        if graph_type == "multiscale" and len(mesh_levels) > 1:
            mesh_cart, mesh_edge_index = (
                _combine_icosahedral_multiscale_to_fine(mesh_levels)
            )
            print(
                "Combined multiscale icosahedral mesh: "
                f"{mesh_cart.shape[0]} nodes, {mesh_edge_index.shape[1]} "
                f"directed m2m edges across {len(mesh_levels)} levels"
            )
        else:
            mesh_cart, mesh_edge_index = mesh_levels[0]
        num_mesh = mesh_cart.shape[0]
        num_grid = grid_xy.shape[0]
        hierarchical_cluster = False
        mesh_pos_list = None
        save_graphs_cluster = None

    num_m2m_edges = mesh_edge_index.shape[1]
    if num_m2m_edges == 0:
        raise ValueError("Mesh has 0 m2m edges after filter_edges_land.")
    dm = float(
        np.mean(
            global_icosahedral_mesh.mesh_edge_lengths_cart(
                mesh_cart, mesh_edge_index
            )
        )
    )
    print(f"Mesh: {num_mesh} nodes, mean edge length (chord) = {dm:.6f}")
    grid_cart = gutils.node_lon_lat_to_cart(grid_xy)
    mesh_xy = gutils.node_cart_to_lon_lat(mesh_cart)

    # G2M: grid -> mesh (radius query in Cartesian for chord distance on sphere;
    #      icosahedral mesh is native Cartesian; cluster is lon-lat but we pass
    #      mesh_cart so radius is correct globally)
    g2m_ei, g2m_len, g2m_vdiff = global_icosahedral_mesh.g2m_radius_query(
        grid_xy,
        mesh_cart,
        mesh_edge_index,
        radius_factor=g2m_radius,
    )
    g2m_edge_index_saved = np.stack(
        [g2m_ei[0] + num_mesh, g2m_ei[1]], axis=0
    ).astype(np.int64)
    g2m_edge_index_t = torch.from_numpy(g2m_edge_index_saved)
    g2m_len_t = torch.from_numpy(g2m_len)
    g2m_vdiff_t = torch.from_numpy(g2m_vdiff)

    # Connect disconnected g2m nodes (3D Cartesian for chord distance)
    if connect_disconnected:
        pos_g2m = np.concatenate([mesh_cart, grid_cart], axis=0)
        pyg_g2m = pyg.data.Data(
            pos=torch.from_numpy(pos_g2m.astype(np.float32)).float(),
            edge_index=g2m_edge_index_t.clone(),
        )
        gutils.add_edge_features_pyg(pyg_g2m)
        is_mesh = np.array([True] * num_mesh + [False] * num_grid)
        is_grid_interior = np.array([False] * num_mesh + [True] * num_grid)
        is_grid_boundary = np.zeros(num_mesh + num_grid, dtype=bool)
        is_grid_atm = np.zeros(num_mesh + num_grid, dtype=bool)
        kdt_m = scipy.spatial.KDTree(mesh_cart)
        gutils.connect_disconnected_g2m(
            pyg_g2m,
            is_mesh,
            is_grid_interior,
            is_grid_boundary,
            is_grid_atm,
            list(range(num_mesh)),
            mesh_cart,
            kdt_m,
            dm,
            g2m_radius,
            0.0,
            0.0,
        )
        g2m_edge_index_t = pyg_g2m.edge_index
        g2m_len_t = pyg_g2m.len
        g2m_vdiff_t = pyg_g2m.vdiff

    g2m_graph = SimpleNamespace(
        edge_index=g2m_edge_index_t,
        len=g2m_len_t,
        vdiff=g2m_vdiff_t,
    )
    saving.save_edges(g2m_graph, "g2m", graph_dir_path)

    # M2G: mesh -> grid (knn)
    m2g_ei, m2g_len, m2g_vdiff = global_icosahedral_mesh.m2g_knn(
        grid_xy, mesh_cart, k=m2g_k
    )
    m2g_edge_index_saved = np.stack(
        [m2g_ei[0], m2g_ei[1] + num_mesh], axis=0
    ).astype(np.int64)
    m2g_edge_index_t = torch.from_numpy(m2g_edge_index_saved)
    m2g_len_t = torch.from_numpy(m2g_len)
    m2g_vdiff_t = torch.from_numpy(m2g_vdiff)

    # Filter m2g edges over land (3D Cartesian: keep edge if midpoint over sea)
    if sea_xy is not None and land_xy is not None and land_xy.shape[0] > 0:
        pos_m2g_3d = np.concatenate([mesh_cart, grid_cart], axis=0)
        ei_np = m2g_edge_index_t.numpy()
        _, ei_filtered = gutils.filter_global_edges_land(
            pos_m2g_3d,
            ei_np,
            sea_xy,
            land_xy,
            max_chord_len=max_chord_len,
            edges_only=True,
        )
        pyg_m2g = pyg.data.Data(
            pos=torch.from_numpy(pos_m2g_3d.astype(np.float32)).float(),
            edge_index=torch.from_numpy(ei_filtered),
        )
        gutils.add_edge_features_pyg(pyg_m2g)
        m2g_edge_index_t = pyg_m2g.edge_index
        m2g_len_t = pyg_m2g.len
        m2g_vdiff_t = pyg_m2g.vdiff

    # Connect disconnected m2g grid nodes (3D Cartesian)
    if connect_disconnected:
        pos_m2g = np.concatenate([mesh_cart, grid_cart], axis=0)
        pyg_m2g = pyg.data.Data(
            pos=torch.from_numpy(pos_m2g.astype(np.float32)).float(),
            edge_index=m2g_edge_index_t.clone(),
        )
        gutils.add_edge_features_pyg(pyg_m2g)
        is_mesh = np.array([True] * num_mesh + [False] * num_grid)
        is_grid = np.array([False] * num_mesh + [True] * num_grid)
        kdt_m = scipy.spatial.KDTree(mesh_cart)
        xy_land = land_xy if land_xy is not None else np.empty((0, 2))
        gutils.connect_disconnected_m2g(
            pyg_m2g,
            is_mesh,
            is_grid,
            grid_cart,
            list(range(num_mesh)),
            mesh_cart,
            kdt_m,
            xy_land,
        )
        m2g_edge_index_t = pyg_m2g.edge_index
        m2g_len_t = pyg_m2g.len
        m2g_vdiff_t = pyg_m2g.vdiff
        m2g_edge_index_saved = pyg_m2g.edge_index.numpy()

    m2g_graph = SimpleNamespace(
        edge_index=m2g_edge_index_t,
        len=m2g_len_t,
        vdiff=m2g_vdiff_t,
    )
    saving.save_edges(m2g_graph, "m2g", graph_dir_path)

    # M2M and mesh features: cluster (single or hierarchical) or icosahedral
    save_graphs_icosahedral = None
    if graph_type == "cluster" and save_graphs_cluster is not None:
        m2m_graphs = save_graphs_cluster["m2m"]
        saving.save_edges_list(m2m_graphs, "m2m", graph_dir_path)
        # Node features cos(lat), sin(lat), sin(lon), cos(lon), Voronoi area
        mesh_features_list = []
        for g in m2m_graphs:
            cart = g.pos.numpy()
            node_feat = _cart_to_node_features(cart)
            voronoi_areas = gutils.compute_voronoi_areas_spherical(cart)
            features = np.concatenate(
                [node_feat, voronoi_areas.reshape(-1, 1)], axis=1
            )
            mesh_features_list.append(
                torch.from_numpy(features.astype(np.float32))
            )
        if hierarchical_cluster:
            mesh_up = save_graphs_cluster["mesh_up"]
            mesh_down = save_graphs_cluster["mesh_down"]
            saving.save_edges_list(mesh_up, "mesh_up", graph_dir_path)
            saving.save_edges_list(mesh_down, "mesh_down", graph_dir_path)
    else:
        # Icosahedral meshes
        if graph_type == "hierarchical":
            # mesh_levels is finest->coarsest, and level 0 is grid connected
            # bottom mesh (matches model expectations).
            m2m_graphs = []
            mesh_features_list = []
            for lvl_cart, lvl_ei in mesh_levels:
                lvl_ei_t = torch.from_numpy(lvl_ei.astype(np.int64))
                lvl_len = global_icosahedral_mesh.mesh_edge_lengths_cart(
                    lvl_cart, lvl_ei
                )
                if lvl_ei.shape[1] > 0:
                    src, dst = lvl_ei[0], lvl_ei[1]
                    # 3D Cartesian vdiff (receiver - sender) on unit sphere
                    lvl_vdiff_3d = lvl_cart[dst].astype(np.float32) - lvl_cart[
                        src
                    ].astype(np.float32)
                else:
                    lvl_vdiff_3d = np.zeros((0, 3), dtype=np.float32)

                m2m_graphs.append(
                    SimpleNamespace(
                        edge_index=lvl_ei_t,
                        len=torch.from_numpy(lvl_len),
                        vdiff=torch.from_numpy(lvl_vdiff_3d),
                    )
                )
                # Node features
                # cos(lat), sin(lat), sin(lon), cos(lon), Voronoi area
                lvl_node_feat = _cart_to_node_features(lvl_cart)
                voronoi_areas = gutils.compute_voronoi_areas_spherical(lvl_cart)
                features = np.concatenate(
                    [lvl_node_feat, voronoi_areas.reshape(-1, 1)], axis=1
                )
                mesh_features_list.append(
                    torch.from_numpy(features.astype(np.float32))
                )

            saving.save_edges_list(m2m_graphs, "m2m", graph_dir_path)

            # Inter-level (up/down) edges between adjacent levels
            mesh_up_graphs = []
            mesh_down_graphs = []
            for level_i in range(len(mesh_levels) - 1):
                fine_cart, _ = mesh_levels[level_i]
                coarse_cart, _ = mesh_levels[level_i + 1]
                if fine_cart.shape[0] == 0 or coarse_cart.shape[0] == 0:
                    raise ValueError(
                        "Empty mesh level encountered while building hierarchy."
                    )

                kdt_coarse = scipy.spatial.cKDTree(coarse_cart)
                _, nn = kdt_coarse.query(fine_cart, k=1)
                nn = nn.astype(np.int64).ravel()

                fine_idx = np.arange(fine_cart.shape[0], dtype=np.int64)
                up_ei = np.stack([fine_idx, nn], axis=0)  # fine -> coarse
                down_ei = np.stack([nn, fine_idx], axis=0)  # coarse -> fine

                # Chord length features
                up_len = np.linalg.norm(
                    coarse_cart[nn] - fine_cart, axis=1
                ).astype(np.float32)
                down_len = up_len.copy()

                # vdiff: 3D Cartesian (receiver - sender) on unit sphere
                up_vdiff_3d = coarse_cart[nn].astype(
                    np.float32
                ) - fine_cart.astype(np.float32)
                down_vdiff_3d = fine_cart.astype(np.float32) - coarse_cart[
                    nn
                ].astype(np.float32)

                mesh_up_graphs.append(
                    SimpleNamespace(
                        edge_index=torch.from_numpy(up_ei),
                        len=torch.from_numpy(up_len),
                        vdiff=torch.from_numpy(up_vdiff_3d),
                    )
                )
                mesh_down_graphs.append(
                    SimpleNamespace(
                        edge_index=torch.from_numpy(down_ei),
                        len=torch.from_numpy(down_len),
                        vdiff=torch.from_numpy(down_vdiff_3d),
                    )
                )

            saving.save_edges_list(mesh_up_graphs, "mesh_up", graph_dir_path)
            saving.save_edges_list(
                mesh_down_graphs, "mesh_down", graph_dir_path
            )
            save_graphs_icosahedral = {
                "m2m": [
                    pyg.data.Data(
                        pos=torch.from_numpy(cart.astype(np.float32)),
                        edge_index=torch.from_numpy(ei.astype(np.int64)),
                    )
                    for cart, ei in mesh_levels
                ],
                "mesh_up": mesh_up_graphs,
                "mesh_down": mesh_down_graphs,
            }
        else:
            mesh_edge_index_t = torch.from_numpy(
                mesh_edge_index.astype(np.int64)
            )
            m2m_len = global_icosahedral_mesh.mesh_edge_lengths_cart(
                mesh_cart, mesh_edge_index
            )
            m2m_src, m2m_dst = mesh_edge_index[0], mesh_edge_index[1]
            # 3D Cartesian vdiff (receiver - sender) on unit sphere
            m2m_vdiff_3d = mesh_cart[m2m_dst].astype(np.float32) - mesh_cart[
                m2m_src
            ].astype(np.float32)
            m2m_graph = SimpleNamespace(
                edge_index=mesh_edge_index_t,
                len=torch.from_numpy(m2m_len),
                vdiff=torch.from_numpy(m2m_vdiff_3d),
            )
            saving.save_edges_list([m2m_graph], "m2m", graph_dir_path)
            # Node features
            # cos(lat), sin(lat), sin(lon), cos(lon), Voronoi area
            mesh_node_feat = _cart_to_node_features(mesh_cart)
            voronoi_areas = gutils.compute_voronoi_areas_spherical(mesh_cart)
            # Concatenate: (sin_lon, cos_lon, sin_lat, cos_lat, voronoi_area)
            features = np.concatenate(
                [mesh_node_feat, voronoi_areas.reshape(-1, 1)], axis=1
            )
            mesh_features_list = [torch.from_numpy(features.astype(np.float32))]
    torch.save(
        mesh_features_list,
        os.path.join(graph_dir_path, "mesh_features.pt"),
    )

    if create_plot:
        print("Plotting global graph...")
        # grid_xy and mesh_xy are (lon, lat) throughout; plot x=lon, y=lat
        mesh_plot_xy = mesh_xy
        grid_plot_xy = grid_xy

        # Overview: x=lon, y=lat
        print("  Overview (mesh + grid)")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(
            mesh_plot_xy[:, 0],
            mesh_plot_xy[:, 1],
            s=1,
            c="orange",
            label="Mesh",
        )
        ax.scatter(
            grid_plot_xy[:, 0],
            grid_plot_xy[:, 1],
            s=0.5,
            c="blue",
            alpha=0.3,
            label="Grid",
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(loc="upper right")
        ax.set_title("Global graph: mesh and grid")
        fig.savefig(
            os.path.join(graph_dir_path, "global_graph_overview.png"),
            dpi=150,
        )
        plt.close(fig)

        # Combined pos (lon, lat) for g2m/m2g plots
        pos_combined = np.concatenate([mesh_plot_xy, grid_plot_xy], axis=0)
        is_mesh = np.array([True] * num_mesh + [False] * num_grid)
        is_any_grid = np.array([False] * num_mesh + [True] * num_grid)

        # Mesh level(s)
        if graph_type == "cluster" and save_graphs_cluster is not None:
            print("  Mesh levels (cluster)")
            m2m_graphs = save_graphs_cluster["m2m"]
            for level_i, g in enumerate(m2m_graphs):
                # Cluster mesh pos is 3D; convert to lon/lat for 2D plot
                level_xy = gutils.node_cart_to_lon_lat(g.pos.numpy())
                level_graph = pyg.data.Data(
                    pos=torch.from_numpy(level_xy).float(),
                    edge_index=g.edge_index,
                )
                vis.plot_graph(
                    level_graph,
                    f"Mesh graph, level {level_i}",
                    graph_dir_path,
                )
                voronoi_areas = mesh_features_list[level_i][:, 4].cpu().numpy()
                vis.plot_node_values(
                    level_graph,
                    f"Mesh Voronoi areas, level {level_i}",
                    graph_dir_path,
                    node_values=voronoi_areas,
                    value_name="Voronoi area (steradians)",
                )
            # Plot inter-level edges if present
            # (convert 3D pos to lon/lat for 2D plot)
            mesh_up_graphs_plot = save_graphs_cluster.get("mesh_up", [])
            mesh_down_graphs_plot = save_graphs_cluster.get("mesh_down", [])
            for level_i, g in enumerate(mesh_up_graphs_plot):
                g_xy = gutils.node_cart_to_lon_lat(g.pos.numpy())
                vis.plot_graph(
                    pyg.data.Data(
                        pos=torch.from_numpy(g_xy).float(),
                        edge_index=g.edge_index,
                    ),
                    f"Mesh up {level_i} to {level_i + 1}",
                    graph_dir_path,
                )
            for level_i, g in enumerate(mesh_down_graphs_plot):
                g_xy = gutils.node_cart_to_lon_lat(g.pos.numpy())
                vis.plot_graph(
                    pyg.data.Data(
                        pos=torch.from_numpy(g_xy).float(),
                        edge_index=g.edge_index,
                    ),
                    f"Mesh down {level_i + 1} -> {level_i}",
                    graph_dir_path,
                )
        elif graph_type == "hierarchical":
            # Icosahedral hierarchy: plot m2m graph for each level (0=bottom)
            print("  Mesh levels (icosahedral)")
            for level_i, (lvl_cart, lvl_ei) in enumerate(mesh_levels):
                lvl_xy = gutils.node_cart_to_lon_lat(lvl_cart)
                level_graph = pyg.data.Data(
                    pos=torch.from_numpy(lvl_xy).float(),
                    edge_index=torch.from_numpy(lvl_ei.astype(np.int64)),
                )
                vis.plot_graph(
                    level_graph,
                    f"Mesh graph, level {level_i}",
                    graph_dir_path,
                )
                voronoi_areas = mesh_features_list[level_i][:, 4].cpu().numpy()
                vis.plot_node_values(
                    level_graph,
                    f"Mesh Voronoi areas, level {level_i}",
                    graph_dir_path,
                    node_values=voronoi_areas,
                    value_name="Voronoi area (steradians)",
                )
            # Plot inter-level edges (fine level_i -> coarse level_i+1)
            for level_i in range(len(mesh_levels) - 1):
                fine_cart, _ = mesh_levels[level_i]
                coarse_cart, _ = mesh_levels[level_i + 1]
                kdt_coarse = scipy.spatial.cKDTree(coarse_cart)
                _, nn = kdt_coarse.query(fine_cart, k=1)
                nn = nn.astype(np.int64).ravel()

                n_fine = fine_cart.shape[0]
                fine_idx = np.arange(n_fine, dtype=np.int64)
                fine_xy = gutils.node_cart_to_lon_lat(fine_cart)
                coarse_xy = gutils.node_cart_to_lon_lat(coarse_cart)
                pos_updown = np.concatenate([fine_xy, coarse_xy], axis=0)

                up_ei_plot = np.stack([fine_idx, nn + n_fine], axis=0)
                up_graph = pyg.data.Data(
                    pos=torch.from_numpy(pos_updown).float(),
                    edge_index=torch.from_numpy(up_ei_plot),
                )
                vis.plot_graph(
                    up_graph,
                    f"Mesh up {level_i} to {level_i + 1}",
                    graph_dir_path,
                )

                down_ei_plot = np.stack([nn + n_fine, fine_idx], axis=0)
                down_graph = pyg.data.Data(
                    pos=torch.from_numpy(pos_updown).float(),
                    edge_index=torch.from_numpy(down_ei_plot),
                )
                vis.plot_graph(
                    down_graph,
                    f"Mesh down {level_i + 1} -> {level_i}",
                    graph_dir_path,
                )
        else:
            print("  Mesh level (multiscale)")
            mesh_level_graph = pyg.data.Data(
                pos=torch.from_numpy(mesh_plot_xy).float(),
                edge_index=mesh_edge_index_t,
            )
            vis.plot_graph(
                mesh_level_graph,
                "Mesh graph, level 0",
                graph_dir_path,
            )
            voronoi_areas = mesh_features_list[0][:, 4].cpu().numpy()
            vis.plot_node_values(
                mesh_level_graph,
                "Mesh Voronoi areas, level 0",
                graph_dir_path,
                node_values=voronoi_areas,
                value_name="Voronoi area (steradians)",
            )

        # G2M: build pyg with pos (lon, lat), compute disconnected
        print("  G2M (grid-to-mesh)")
        pyg_g2m = pyg.data.Data(
            pos=torch.from_numpy(pos_combined).float(),
            edge_index=g2m_graph.edge_index.clone(),
        )
        g2m_src = pyg_g2m.edge_index[0]
        g2m_dst = pyg_g2m.edge_index[1]
        g2m_outdeg = degree(g2m_src, num_nodes=pyg_g2m.num_nodes)
        g2m_indeg = degree(g2m_dst, num_nodes=pyg_g2m.num_nodes)
        grid_mask_t = torch.as_tensor(is_any_grid, device=pyg_g2m.pos.device)
        mesh_mask_t = torch.as_tensor(is_mesh, device=pyg_g2m.pos.device)
        disc_grid_np = (
            torch.where((g2m_outdeg == 0) & grid_mask_t)[0].cpu().numpy()
        )
        disc_mesh_np = (
            torch.where((g2m_indeg == 0) & mesh_mask_t)[0].cpu().numpy()
        )
        vis.plot_disconnected_nodes(
            pos_combined,
            is_mesh,
            is_any_grid,
            disc_grid_np,
            disc_mesh_np,
            graph_dir_path,
            title="g2m_disconnected",
        )
        vis.plot_graph(
            pyg_g2m, "Grid-to-mesh", graph_dir_path, order_by_degree=True
        )
        pyg_g2m_r = pyg_g2m.clone()
        pyg_g2m_r.edge_index = pyg_g2m.edge_index[[1, 0]]
        vis.plot_graph(pyg_g2m_r, "Grid-to-mesh-r", graph_dir_path)

        # M2G
        print("  M2G (mesh-to-grid)")
        pyg_m2g = pyg.data.Data(
            pos=torch.from_numpy(pos_combined).float(),
            edge_index=m2g_graph.edge_index.clone(),
        )
        m2g_dst = pyg_m2g.edge_index[1]
        m2g_indeg = degree(m2g_dst, num_nodes=pyg_m2g.num_nodes)
        m2g_disc_grid_np = (
            torch.where((m2g_indeg == 0) & grid_mask_t)[0].cpu().numpy()
        )
        vis.plot_disconnected_nodes(
            pos_combined,
            is_mesh,
            is_any_grid,
            m2g_disc_grid_np,
            np.array([], dtype=int),
            graph_dir_path,
            title="m2g_disconnected",
        )
        vis.plot_graph(
            pyg_m2g, "Mesh-to-grid", graph_dir_path, reindex_edges=False
        )
        pyg_m2g_r = pyg_m2g.clone()
        pyg_m2g_r.edge_index = pyg_m2g.edge_index[[1, 0]]
        vis.plot_graph(
            pyg_m2g_r,
            "Mesh-to-grid-r",
            graph_dir_path,
            reindex_edges=False,
            order_by_degree=True,
        )

    # Print graph statistics
    if save_graphs_cluster is not None:
        save_graphs = save_graphs_cluster
    elif save_graphs_icosahedral is not None:
        save_graphs = save_graphs_icosahedral
    else:
        # Multiscale or single-level icosahedral
        if graph_type == "multiscale":
            m2m_list = [
                pyg.data.Data(
                    pos=torch.from_numpy(mesh_cart.astype(np.float32)),
                    edge_index=torch.from_numpy(
                        mesh_edge_index.astype(np.int64)
                    ),
                )
            ]
        else:
            m2m_list = [
                pyg.data.Data(
                    pos=torch.from_numpy(cart.astype(np.float32)),
                    edge_index=torch.from_numpy(ei.astype(np.int64)),
                )
                for cart, ei in mesh_levels
            ]
        save_graphs = {"m2m": m2m_list, "mesh_up": [], "mesh_down": []}
    n_combined = num_mesh + num_grid
    pyg_g2m = pyg.data.Data(
        pos=torch.zeros(n_combined, 2),
        edge_index=g2m_graph.edge_index,
    )
    pyg_m2g = pyg.data.Data(
        pos=torch.zeros(n_combined, 2),
        edge_index=m2g_graph.edge_index,
    )
    gutils.print_graph_stats(save_graphs, pyg_g2m, pyg_m2g)


def cli(input_args=None):
    parser = ArgumentParser(description="Create global graph.")
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to neural-lam config; grid is loaded from datastore. "
        "Mutually exclusive with --dataset.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="global_multiscale",
        help="Graph name (saved under datastore root or cwd).",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=6,
        help="Icosahedral mesh subdivisions (0=base, 3 ~ 2562 nodes).",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="multiscale",
        choices=["multiscale", "hierarchical", "cluster"],
        help="Graph type: multiscale (single icosahedral), hierarchical "
        "(multi-level icosahedral), or cluster (KMeans over sea, requires "
        "--config_path).",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=2,
        help="Number of mesh levels (default: 2).",
    )
    parser.add_argument(
        "--g2m_radius",
        type=float,
        default=0.67,
        help="G2M connection radius as multiple of mean mesh edge length.",
    )
    parser.add_argument(
        "--m2g_k",
        type=int,
        default=3,
        help="Number of nearest mesh nodes per grid point for M2G.",
    )
    parser.add_argument(
        "--connect_disconnected",
        action="store_true",
        help="Connect disconnected grid/mesh nodes via nearest neighbor.",
    )
    parser.add_argument(
        "--max_chord_len",
        type=float,
        default=0.1,
        help="Max edge chord length on unit sphere for land filtering (~0.1 "
        "corresponds to ~5.7° great-circle arc).",
    )
    parser.add_argument(
        "--mesh_refinement_factor",
        type=float,
        default=4,
        help="(Cluster) factor between mesh nodes at consecutive levels.",
    )
    parser.add_argument(
        "--grid_to_first_mesh_refinement",
        type=float,
        default=23,
        help="(Cluster) ratio grid nodes / first-level mesh nodes.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save overview plot of mesh and grid.",
    )
    args = parser.parse_args(input_args)

    _, datastore, *_ = load_config_and_datastores(config_path=args.config_path)
    sea_xy, land_xy = load_grid_from_datastore(datastore)
    graph_dir_path = os.path.join(datastore.root_path, "graphs", args.name)

    create_global_graph(
        graph_dir_path=graph_dir_path,
        grid_xy=sea_xy,
        splits=args.splits,
        levels=args.levels,
        graph_type=args.type,
        g2m_radius=args.g2m_radius,
        m2g_k=args.m2g_k,
        create_plot=args.plot,
        connect_disconnected=args.connect_disconnected,
        sea_xy=sea_xy,
        land_xy=land_xy,
        max_chord_len=args.max_chord_len,
        mesh_refinement_factor=args.mesh_refinement_factor,
        grid_to_first_mesh_refinement=args.grid_to_first_mesh_refinement,
    )


if __name__ == "__main__":
    cli()
