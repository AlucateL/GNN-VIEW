# Standard library
from typing import List, Tuple

# Third-party
import numpy as np
import scipy.spatial

# Local
from . import utils as gutils


def _icosahedron_vertices() -> np.ndarray:
    """Vertices of unit icosahedron (12 points on sphere)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=np.float64,
    )
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    return verts


def _icosahedron_faces() -> np.ndarray:
    """Faces of icosahedron (20 triangles, 3 vertex indices each)."""
    return np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int64,
    )


def _cartesian_to_lon_lat(cart: np.ndarray) -> np.ndarray:
    """Convert (N, 3) Cartesian on unit sphere to (N, 2)
    [longitude, latitude] in degrees."""
    r = np.linalg.norm(cart, axis=1, keepdims=True)
    cart = cart / (r + 1e-12)
    # z = sin(lat), x = cos(lat)*cos(lon), y = cos(lat)*sin(lon)
    lat_rad = np.arcsin(np.clip(cart[:, 2], -1, 1))
    lon_rad = np.arctan2(cart[:, 1], cart[:, 0])
    # Return (lon, lat) = (x, y) convention
    lon_lat = np.stack(
        [np.rad2deg(lon_rad), np.rad2deg(lat_rad)], axis=1
    ).astype(np.float32)
    return lon_lat


def _subdivide_to_sphere(
    vertices: np.ndarray, faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Subdivide each triangle into 4, project new vertices to unit sphere.
    Shared edges get a single new vertex (deduplicated by edge key).
    """
    new_vertices = list(vertices)
    edge_to_idx = {}

    def get_midpoint(i: int, j: int) -> int:
        key = (min(i, j), max(i, j))
        if key not in edge_to_idx:
            mid = (vertices[i] + vertices[j]) / 2
            mid /= np.linalg.norm(mid)
            edge_to_idx[key] = len(new_vertices)
            new_vertices.append(mid)
        return edge_to_idx[key]

    new_faces = []
    for f in faces:
        a, b, c = f[0], f[1], f[2]
        m_ab = get_midpoint(a, b)
        m_bc = get_midpoint(b, c)
        m_ca = get_midpoint(c, a)
        new_faces.append([a, m_ab, m_ca])
        new_faces.append([m_ab, b, m_bc])
        new_faces.append([m_ca, m_bc, c])
        new_faces.append([m_ab, m_bc, m_ca])

    return np.array(new_vertices, dtype=np.float64), np.array(
        new_faces, dtype=np.int64
    )


def faces_to_edges(faces: np.ndarray) -> np.ndarray:
    """Convert (F, 3) faces to (2, E) edge index, undirected."""
    edges = set()
    for f in faces:
        a, b, c = f[0], f[1], f[2]
        edges.add((min(a, b), max(a, b)))
        edges.add((min(b, c), max(b, c)))
        edges.add((min(c, a), max(c, a)))
    edges = np.array(list(edges), dtype=np.int64).T  # (2, E)
    # Both directions for directed graph
    edges_both = np.concatenate([edges, edges[[1, 0]]], axis=1)
    return edges_both


def get_icosahedral_mesh_level(
    splits: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one level of icosahedral mesh on unit sphere.

    splits: number of 1-to-4 subdivisions (0 = base icosahedron).

    Returns
    -------
    vertices_cart : (N, 3)
    vertices_lon_lat : (N, 2) in degrees, [longitude, latitude]
    edge_index : (2, E)
    """
    vertices = _icosahedron_vertices()
    faces = _icosahedron_faces()
    for _ in range(splits):
        vertices, faces = _subdivide_to_sphere(vertices, faces)
    vertices_lon_lat = _cartesian_to_lon_lat(vertices)
    edge_index = faces_to_edges(faces)
    return vertices, vertices_lon_lat, edge_index


def get_hierarchy_of_triangular_meshes(
    splits: int,
    levels: int | None = None,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build hierarchy of icosahedral meshes (coarsest to finest).

    splits: number of subdivisions for the finest level.
    levels: number of levels to keep (1 = single level, finest). If None, use
        splits+1 levels (level 0..splits).

    Returns
    -------
    list of (vertices_cart, vertices_lon_lat, edge_index) for each level.
    """
    all_levels = []
    for s in range(splits + 1):
        vert_cart, vert_lon_lat, edge_idx = get_icosahedral_mesh_level(s)
        all_levels.append((vert_cart, vert_lon_lat, edge_idx))
    if levels is not None:
        # Keep last `levels` levels (finest)
        all_levels = all_levels[-levels:]
    return all_levels


def mesh_edge_lengths_cart(
    vertices_cart: np.ndarray, edge_index: np.ndarray
) -> np.ndarray:
    """Edge lengths (chord length on unit sphere) for mesh edges."""
    src, dst = edge_index[0], edge_index[1]
    vdiff = vertices_cart[dst] - vertices_cart[src]
    return np.linalg.norm(vdiff, axis=1).astype(np.float32)


def g2m_radius_query(
    grid_xy: np.ndarray,
    mesh_vertices_cart: np.ndarray,
    mesh_edge_index: np.ndarray,
    radius_factor: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid-to-mesh edges by radius query (chord distance on unit sphere).

    grid_xy: (N_grid, 2) [longitude, latitude] in degrees (x=lon, y=lat).

    Node convention: grid 0..N_grid-1, mesh N_grid..N_grid+N_mesh-1 in output.
    So g2m_edge_index[0] = grid indices (0..N_grid-1), [1] = mesh indices
    (0..N_mesh-1) in mesh_vertices_cart. Caller will shift for saved format.

    Returns
    -------
    g2m_edge_index : (2, M) with [0]=grid_idx, [1]=mesh_idx (0-based mesh)
    g2m_len : (M,) edge lengths (chord)
    g2m_vdiff : (M, 2) lon-lat difference in degrees (dlat, dlon)
    """
    grid_cart = gutils.node_lon_lat_to_cart(grid_xy)
    mesh_cart = mesh_vertices_cart
    dm = np.mean(mesh_edge_lengths_cart(mesh_vertices_cart, mesh_edge_index))
    radius = dm * radius_factor

    kdt = scipy.spatial.cKDTree(mesh_cart)
    g2m_src_list = []
    g2m_dst_list = []
    g2m_len_list = []
    g2m_vdiff_list = []

    for gi in range(grid_xy.shape[0]):
        neighs = kdt.query_ball_point(grid_cart[gi], radius)
        for mi in neighs:
            d = np.linalg.norm(mesh_cart[mi] - grid_cart[gi])
            # Edge feature: (dlat, dlon) in degrees
            g_lon, g_lat = np.deg2rad(grid_xy[gi, 0]), np.deg2rad(
                grid_xy[gi, 1]
            )
            m_lon_lat = _cartesian_to_lon_lat(mesh_cart[mi].reshape(1, 3))[0]
            m_lon, m_lat = np.deg2rad(m_lon_lat[0]), np.deg2rad(m_lon_lat[1])
            dlat = np.rad2deg(m_lat - g_lat)
            dlon = np.rad2deg(m_lon - g_lon)
            g2m_src_list.append(gi)
            g2m_dst_list.append(mi)
            g2m_len_list.append(d)
            g2m_vdiff_list.append([dlat, dlon])

    g2m_edge_index = np.stack(
        [np.array(g2m_src_list), np.array(g2m_dst_list)], axis=0
    )
    g2m_len = np.array(g2m_len_list, dtype=np.float32)
    g2m_vdiff = np.array(g2m_vdiff_list, dtype=np.float32)
    return g2m_edge_index, g2m_len, g2m_vdiff


def m2g_knn(
    grid_xy: np.ndarray,
    mesh_vertices_cart: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mesh-to-grid edges by k nearest mesh nodes per grid point.

    grid_xy: (N_grid, 2) [longitude, latitude] in degrees (x=lon, y=lat).

    Returns
    -------
    m2g_edge_index : (2, M) with [0]=mesh_idx, [1]=grid_idx (0-based)
    m2g_len : (M,) chord lengths
    m2g_vdiff : (M, 2) lon-lat diff in degrees (grid - mesh), (dlat, dlon)
    """
    n_grid = grid_xy.shape[0]
    grid_cart = gutils.node_lon_lat_to_cart(grid_xy)
    kdt = scipy.spatial.cKDTree(mesh_vertices_cart)
    dists, mesh_idx = kdt.query(grid_cart, k=k)
    if dists.ndim == 1:
        dists = dists.reshape(-1, 1)
        mesh_idx = mesh_idx.reshape(-1, 1)

    # Vectorized all mesh positions for (grid_i, neigh_j) and convert to lon/lat
    mesh_idx_flat = np.asarray(mesh_idx, dtype=np.intp)
    mesh_cart_flat = mesh_vertices_cart[mesh_idx_flat]  # (n_grid, k, 3)
    mesh_lon_lat = _cartesian_to_lon_lat(mesh_cart_flat.reshape(-1, 3)).reshape(
        n_grid, -1, 2
    )  # (n_grid, k, 2) [lon, lat] degrees

    # Edge lists: src = mesh idx, dst = grid idx (one per (grid, neighbor))
    m2g_src = mesh_idx_flat.ravel()
    m2g_dst = np.repeat(
        np.arange(n_grid, dtype=np.intp), mesh_idx_flat.shape[1]
    )
    m2g_len = np.asarray(dists.ravel(), dtype=np.float32)

    # vdiff (dlat, dlon) in degrees: grid - mesh
    grid_xy_expanded = grid_xy[:, None, :]  # (n_grid, 1, 2)
    dlat = (grid_xy_expanded[:, :, 1] - mesh_lon_lat[:, :, 1]).ravel()
    dlon = (grid_xy_expanded[:, :, 0] - mesh_lon_lat[:, :, 0]).ravel()
    m2g_vdiff = np.stack([dlat, dlon], axis=1).astype(np.float32)

    m2g_edge_index = np.stack([m2g_src, m2g_dst], axis=0)
    return m2g_edge_index, m2g_len, m2g_vdiff


def inter_mesh_connection(
    from_cart: np.ndarray,
    to_cart: np.ndarray,
    radius_factor: float = 1.1,
    from_edge_length: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Connect coarse (from) mesh to fine (to) mesh by radius query.

    Returns
    -------
    edge_index : (2, M) [from_idx, to_idx]
    edge_len : (M,)
    edge_vdiff : (M, 2) lon-lat diff in degrees
    """
    if from_edge_length is None:
        from_edge_length = 0.1  # fallback
    radius = from_edge_length * radius_factor
    kdt = scipy.spatial.cKDTree(to_cart)
    from_lon_lat = _cartesian_to_lon_lat(from_cart)
    to_lon_lat = _cartesian_to_lon_lat(to_cart)
    src_list, dst_list, len_list, vdiff_list = [], [], [], []
    for fi in range(from_cart.shape[0]):
        neighs = kdt.query_ball_point(from_cart[fi], radius)
        for ti in neighs:
            d = np.linalg.norm(to_cart[ti] - from_cart[fi])
            f_lon, f_lat = from_lon_lat[fi, 0], from_lon_lat[fi, 1]
            t_lon, t_lat = to_lon_lat[ti, 0], to_lon_lat[ti, 1]
            src_list.append(fi)
            dst_list.append(ti)
            len_list.append(float(d))
            vdiff_list.append([t_lat - f_lat, t_lon - f_lon])
    edge_index = np.stack([np.array(src_list), np.array(dst_list)], axis=0)
    edge_len = np.array(len_list, dtype=np.float32)
    edge_vdiff = np.array(vdiff_list, dtype=np.float32)
    return edge_index, edge_len, edge_vdiff
