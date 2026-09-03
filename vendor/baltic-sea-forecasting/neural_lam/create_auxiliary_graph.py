# Standard library
import os
import shutil
from argparse import ArgumentParser
from types import SimpleNamespace

# Third-party
import numpy as np
import scipy.spatial
import torch
import torch_geometric as pyg
from torch_geometric.utils import degree

# Local
from .config import load_config_and_datastores
from .create_global_graph import load_grid_from_datastore
from .graphs import global_icosahedral_mesh, saving
from .graphs import utils as gutils
from .graphs import vis


def _mesh_features_to_cart(mesh_features: torch.Tensor) -> np.ndarray:
    """
    Convert mesh_features (N, 5) [sin_lon, cos_lon, sin_lat, cos_lat, voronoi]
    to mesh_cart (N, 3).
    """
    feat = mesh_features.cpu().numpy()
    sin_lon, cos_lon = feat[:, 0], feat[:, 1]
    sin_lat = np.clip(feat[:, 2], -1.0, 1.0)
    lon_rad = np.arctan2(sin_lon, cos_lon)
    lat_rad = np.arcsin(sin_lat)
    mesh_xy_deg = np.stack(
        [np.rad2deg(lon_rad), np.rad2deg(lat_rad)], axis=1
    ).astype(np.float32)
    return gutils.node_lon_lat_to_cart(mesh_xy_deg)


def _mesh_features_to_lon_lat(mesh_features: torch.Tensor) -> np.ndarray:
    """Convert mesh_features to (N, 2) lon/lat in degrees for plotting."""
    feat = mesh_features.cpu().numpy()
    sin_lon, cos_lon = feat[:, 0], feat[:, 1]
    sin_lat = np.clip(feat[:, 2], -1.0, 1.0)
    lon_rad = np.arctan2(sin_lon, cos_lon)
    lat_rad = np.arcsin(sin_lat)
    return np.stack([np.rad2deg(lon_rad), np.rad2deg(lat_rad)], axis=1).astype(
        np.float32
    )


def _reindex_edge_index(ei: torch.Tensor) -> np.ndarray:
    """Make edge_index 0-based (both rows)."""
    ei = ei.cpu().numpy() if ei.dim() > 0 else np.array(ei)
    if ei.size == 0:
        return ei
    return (ei - ei.min()).astype(np.int64)


def create_auxiliary_graph(
    graph_dir_path_new: str,
    graph_dir_path_original: str,
    grid_xy: np.ndarray,
    sea_xy: np.ndarray,
    land_xy: np.ndarray,
    g2m_radius: float = 0.67,
    m2g_k: int = 3,
    connect_disconnected: bool = False,
    max_chord_len: float = 0.1,
):
    """
    Create an auxiliary graph that reuses the mesh from the original graph
    and connects it to a new (e.g. coarser) grid via new g2m and m2g edges.

    Works only with graphs produced by create_global_graph.py.
    Mesh files (m2m, mesh_up, mesh_down, mesh_features) are copied from the
    original; g2m and m2g are built for the new grid and saved.

    grid_xy, sea_xy, land_xy: (N, 2) [longitude, latitude] in degrees.
    """
    os.makedirs(graph_dir_path_new, exist_ok=True)
    print(f"Creating auxiliary graph at {graph_dir_path_new}")
    print(f"  Original graph: {graph_dir_path_original}")

    def load_pt(name):
        return torch.load(
            os.path.join(graph_dir_path_original, name),
            map_location="cpu",
            weights_only=True,
        )

    # Load mesh from original
    mesh_features_list = load_pt("mesh_features.pt")
    if not isinstance(mesh_features_list, list):
        mesh_features_list = [mesh_features_list]
    m2m_edge_index_list = load_pt("m2m_edge_index.pt")
    if not isinstance(m2m_edge_index_list, list):
        m2m_edge_index_list = [m2m_edge_index_list]

    mesh_cart = _mesh_features_to_cart(mesh_features_list[0])
    num_mesh = mesh_cart.shape[0]
    mesh_edge_index = _reindex_edge_index(m2m_edge_index_list[0])
    num_grid = grid_xy.shape[0]
    grid_cart = gutils.node_lon_lat_to_cart(grid_xy)

    # Copy mesh-related files to new graph dir
    for name in [
        "m2m_edge_index.pt",
        "m2m_features.pt",
        "mesh_features.pt",
    ]:
        src = os.path.join(graph_dir_path_original, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(graph_dir_path_new, name))
    for suffix in ["edge_index", "features"]:
        for prefix in ["mesh_up", "mesh_down"]:
            name = f"{prefix}_{suffix}.pt"
            src = os.path.join(graph_dir_path_original, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(graph_dir_path_new, name))

    dm = float(
        np.mean(
            global_icosahedral_mesh.mesh_edge_lengths_cart(
                mesh_cart, mesh_edge_index
            )
        )
    )
    print(f"Mesh: {num_mesh} nodes, mean edge length (chord) = {dm:.6f}")
    print(f"New grid: {num_grid} points")

    # G2M: grid -> mesh
    g2m_ei, _, _ = global_icosahedral_mesh.g2m_radius_query(
        grid_xy,
        mesh_cart,
        mesh_edge_index,
        radius_factor=g2m_radius,
    )
    g2m_edge_index_saved = np.stack(
        [g2m_ei[0] + num_mesh, g2m_ei[1]], axis=0
    ).astype(np.int64)
    g2m_edge_index_t = torch.from_numpy(g2m_edge_index_saved)

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

    # Save g2m edge features as (len, vdiff_3d) = 4 cols
    pos_3d_g2m = torch.from_numpy(
        np.concatenate([mesh_cart, grid_cart], axis=0).astype(np.float32)
    ).float()
    pyg_g2m_final = pyg.data.Data(pos=pos_3d_g2m, edge_index=g2m_edge_index_t)
    gutils.add_edge_features_pyg(pyg_g2m_final)
    g2m_graph = SimpleNamespace(
        edge_index=g2m_edge_index_t,
        len=pyg_g2m_final.len,
        vdiff=pyg_g2m_final.vdiff,
    )
    saving.save_edges(g2m_graph, "g2m", graph_dir_path_new)

    # M2G: mesh -> grid
    m2g_ei, _, _ = global_icosahedral_mesh.m2g_knn(grid_xy, mesh_cart, k=m2g_k)
    m2g_edge_index_saved = np.stack(
        [m2g_ei[0], m2g_ei[1] + num_mesh], axis=0
    ).astype(np.int64)
    m2g_edge_index_t = torch.from_numpy(m2g_edge_index_saved)

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
        ei_filtered = ei_filtered.astype(np.int64)
        pyg_m2g = pyg.data.Data(
            pos=torch.from_numpy(pos_m2g_3d.astype(np.float32)).float(),
            edge_index=torch.from_numpy(ei_filtered),
        )
        gutils.add_edge_features_pyg(pyg_m2g)
        m2g_edge_index_t = pyg_m2g.edge_index

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

    # Save m2g edge features as (len, vdiff_3d) = 4 cols
    pos_3d_m2g = torch.from_numpy(
        np.concatenate([mesh_cart, grid_cart], axis=0).astype(np.float32)
    ).float()
    pyg_m2g_final = pyg.data.Data(pos=pos_3d_m2g, edge_index=m2g_edge_index_t)
    gutils.add_edge_features_pyg(pyg_m2g_final)
    m2g_graph = SimpleNamespace(
        edge_index=m2g_edge_index_t,
        len=pyg_m2g_final.len,
        vdiff=pyg_m2g_final.vdiff,
    )
    saving.save_edges(m2g_graph, "m2g", graph_dir_path_new)

    # Build save_graphs for stats (and optional plot)
    m2m_graphs = []
    for level_i, ei_pt in enumerate(m2m_edge_index_list):
        ei = _reindex_edge_index(ei_pt)
        mesh_xy_lvl = (
            _mesh_features_to_lon_lat(mesh_features_list[level_i])
            if level_i < len(mesh_features_list)
            else _mesh_features_to_lon_lat(mesh_features_list[0])
        )
        m2m_graphs.append(
            pyg.data.Data(
                pos=torch.from_numpy(mesh_xy_lvl).float(),
                edge_index=torch.from_numpy(ei),
            )
        )
    save_graphs = {"m2m": m2m_graphs, "mesh_up": [], "mesh_down": []}
    mesh_up_list = []
    mesh_down_list = []
    for prefix in ["mesh_up", "mesh_down"]:
        try:
            ei_list = load_pt(f"{prefix}_edge_index.pt")
            feat_list = load_pt(f"{prefix}_features.pt")
            for e, f in zip(ei_list, feat_list):
                g = SimpleNamespace(
                    edge_index=e,
                    len=f[:, 0],
                    vdiff=f[:, 1:],
                )
                (
                    mesh_up_list if prefix == "mesh_up" else mesh_down_list
                ).append(g)
        except FileNotFoundError:
            pass
    if mesh_up_list:
        save_graphs["mesh_up"] = mesh_up_list
    if mesh_down_list:
        save_graphs["mesh_down"] = mesh_down_list

    pos_combined = np.concatenate(
        [_mesh_features_to_lon_lat(mesh_features_list[0]), grid_xy], axis=0
    )
    pyg_g2m = pyg.data.Data(
        pos=torch.from_numpy(pos_combined).float(),
        edge_index=g2m_graph.edge_index,
    )
    pyg_m2g = pyg.data.Data(
        pos=torch.from_numpy(pos_combined).float(),
        edge_index=m2g_graph.edge_index,
    )
    gutils.print_graph_stats(save_graphs, pyg_g2m, pyg_m2g)


def _plot_auxiliary_graph(graph_dir_path: str, grid_xy: np.ndarray):
    """Plot graph from saved pt files: mesh levels, up/down, g2m, m2g."""
    print("Plotting auxiliary graph...")

    def load_pt(name):
        return torch.load(
            os.path.join(graph_dir_path, name),
            map_location="cpu",
            weights_only=True,
        )

    mesh_features_list = load_pt("mesh_features.pt")
    if not isinstance(mesh_features_list, list):
        mesh_features_list = [mesh_features_list]
    mesh_xy = _mesh_features_to_lon_lat(mesh_features_list[0])
    num_mesh = mesh_xy.shape[0]
    num_grid = grid_xy.shape[0]
    pos_combined = np.concatenate([mesh_xy, grid_xy], axis=0).astype(np.float32)
    is_mesh = np.array([True] * num_mesh + [False] * num_grid)
    is_any_grid = np.array([False] * num_mesh + [True] * num_grid)

    m2m_edge_index_list = load_pt("m2m_edge_index.pt")
    if not isinstance(m2m_edge_index_list, list):
        m2m_edge_index_list = [m2m_edge_index_list]

    m2m_graphs = []
    # Mesh level(s)
    for level_i, ei_pt in enumerate(m2m_edge_index_list):
        ei = _reindex_edge_index(ei_pt)
        if level_i < len(mesh_features_list):
            lvl_xy = _mesh_features_to_lon_lat(mesh_features_list[level_i])
        else:
            lvl_xy = mesh_xy
        level_graph = pyg.data.Data(
            pos=torch.from_numpy(lvl_xy).float(),
            edge_index=torch.from_numpy(ei),
        )
        m2m_graphs.append(level_graph)
        vis.plot_graph(
            level_graph,
            f"Mesh graph, level {level_i}",
            graph_dir_path,
        )
        if (
            level_i < len(mesh_features_list)
            and mesh_features_list[level_i].shape[1] >= 5
        ):
            voronoi_areas = mesh_features_list[level_i][:, 4].cpu().numpy()
            vis.plot_node_values(
                level_graph,
                f"Mesh Voronoi areas, level {level_i}",
                graph_dir_path,
                node_values=voronoi_areas,
                value_name="Voronoi area (steradians)",
            )

    # Mesh up/down if present (hierarchical)
    for prefix, title_up in [
        ("mesh_up", "Mesh up"),
        ("mesh_down", "Mesh down"),
    ]:
        try:
            ei_list = load_pt(f"{prefix}_edge_index.pt")
            if not isinstance(ei_list, list):
                ei_list = [ei_list]
            for level_i, ei_pt in enumerate(ei_list):
                ei = (
                    ei_pt.numpy()
                    if isinstance(ei_pt, torch.Tensor)
                    else np.array(ei_pt)
                )
                if ei.size == 0:
                    continue
                fine_xy = _mesh_features_to_lon_lat(mesh_features_list[level_i])
                coarse_xy = _mesh_features_to_lon_lat(
                    mesh_features_list[level_i + 1]
                )
                pos_updown = np.concatenate([fine_xy, coarse_xy], axis=0)
                g = pyg.data.Data(
                    pos=torch.from_numpy(pos_updown).float(),
                    edge_index=torch.from_numpy(ei),
                )
                label = (
                    f"{title_up} {level_i} to {level_i + 1}"
                    if "up" in prefix
                    else f"{title_up} {level_i + 1} -> {level_i}"
                )
                vis.plot_graph(g, label, graph_dir_path, reindex_edges=False)
        except FileNotFoundError:
            pass

    # Build save_graphs and print statistics
    mesh_up_list = []
    mesh_down_list = []
    for prefix in ["mesh_up", "mesh_down"]:
        try:
            ei_list = load_pt(f"{prefix}_edge_index.pt")
            if not isinstance(ei_list, list):
                ei_list = [ei_list]
            for e in ei_list:
                g = pyg.data.Data(edge_index=e.clone())
                (
                    mesh_up_list if prefix == "mesh_up" else mesh_down_list
                ).append(g)
        except FileNotFoundError:
            pass
    save_graphs = {
        "m2m": m2m_graphs,
        "mesh_up": mesh_up_list,
        "mesh_down": mesh_down_list,
    }
    g2m_edge_index = load_pt("g2m_edge_index.pt")
    m2g_edge_index = load_pt("m2g_edge_index.pt")
    pyg_g2m = pyg.data.Data(
        pos=torch.from_numpy(pos_combined).float(),
        edge_index=g2m_edge_index.clone(),
    )
    pyg_m2g = pyg.data.Data(
        pos=torch.from_numpy(pos_combined).float(),
        edge_index=m2g_edge_index.clone(),
    )
    gutils.print_graph_stats(save_graphs, pyg_g2m, pyg_m2g)

    # G2M
    grid_mask_t = torch.as_tensor(is_any_grid, device=pyg_g2m.pos.device)
    mesh_mask_t = torch.as_tensor(is_mesh, device=pyg_g2m.pos.device)
    g2m_src = pyg_g2m.edge_index[0]
    g2m_dst = pyg_g2m.edge_index[1]
    g2m_outdeg = degree(g2m_src, num_nodes=pyg_g2m.num_nodes)
    g2m_indeg = degree(g2m_dst, num_nodes=pyg_g2m.num_nodes)
    disc_grid_np = torch.where((g2m_outdeg == 0) & grid_mask_t)[0].cpu().numpy()
    disc_mesh_np = torch.where((g2m_indeg == 0) & mesh_mask_t)[0].cpu().numpy()
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
    vis.plot_graph(pyg_m2g, "Mesh-to-grid", graph_dir_path, reindex_edges=False)
    pyg_m2g_r = pyg_m2g.clone()
    pyg_m2g_r.edge_index = pyg_m2g.edge_index[[1, 0]]
    vis.plot_graph(
        pyg_m2g_r,
        "Mesh-to-grid-r",
        graph_dir_path,
        reindex_edges=False,
        order_by_degree=True,
    )


def cli(input_args=None):
    parser = ArgumentParser(
        description="Create auxiliary graph: connect a new (e.g. coarser) grid "
        "to the mesh of an existing graph from create_global_graph.py."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        required=True,
        help="Path to config; new grid is loaded from this datastore.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name for the new auxiliary graph.",
    )
    parser.add_argument(
        "--name_original",
        type=str,
        required=True,
        help="Name of the original graph to connect to.",
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
        "--plot",
        action="store_true",
        help="Plot mesh levels, up/down, g2m, m2g graphs.",
    )
    args = parser.parse_args(input_args)

    _, datastore, *_ = load_config_and_datastores(config_path=args.config_path)
    sea_xy, land_xy = load_grid_from_datastore(datastore)
    root = datastore.root_path
    graph_dir_path_original = os.path.join(root, "graphs", args.name_original)
    graph_dir_path_new = os.path.join(root, "graphs", args.name)

    if not os.path.isdir(graph_dir_path_original):
        raise FileNotFoundError(
            f"Original graph directory not found: {graph_dir_path_original}"
        )

    if args.plot and os.path.isdir(graph_dir_path_new):
        _plot_auxiliary_graph(graph_dir_path_new, sea_xy)
        return

    create_auxiliary_graph(
        graph_dir_path_new=graph_dir_path_new,
        graph_dir_path_original=graph_dir_path_original,
        grid_xy=sea_xy,
        sea_xy=sea_xy,
        land_xy=land_xy,
        g2m_radius=args.g2m_radius,
        m2g_k=args.m2g_k,
        connect_disconnected=args.connect_disconnected,
        max_chord_len=args.max_chord_len,
    )

    if args.plot:
        _plot_auxiliary_graph(graph_dir_path_new, sea_xy)


if __name__ == "__main__":
    cli()
