# Third-party
import networkx
import numpy as np
import scipy
import torch
from torch_geometric.utils.convert import from_networkx

# Local
from . import utils as gutils


def mk_2d_graph(xy, nx, ny, threshold):
    # xy is (num_grid, 2)
    min_x, min_y = np.min(xy, axis=0)
    max_x, max_y = np.max(xy, axis=0)
    # Offset from edge
    offset_x = (max_x - min_x) / (2 * nx)
    offset_y = (max_y - min_y) / (2 * ny)
    x_coords = np.linspace(min_x + offset_x, max_x - offset_x, nx)
    y_coords = np.linspace(min_y + offset_y, max_y - offset_y, ny)

    xy_kdtree = scipy.spatial.KDTree(xy)

    # build base grid graph
    g = networkx.grid_2d_graph(nx, ny)

    for node in list(g.nodes):
        i, j = node
        node_pos = np.array((x_coords[i], y_coords[j]))

        # Find closest point in xy, check that we are over grid
        dist, _ = xy_kdtree.query(node_pos, k=1)
        if dist < threshold:
            g.nodes[node]["pos"] = node_pos
        else:
            g.remove_node(node)

    # add diagonal edges if both nodes exist
    for x in range(nx):
        for y in range(ny):
            if g.has_node((x, y)) and g.has_node((x + 1, y + 1)):
                g.add_edge((x, y), (x + 1, y + 1))
            if g.has_node((x + 1, y)) and g.has_node((x, y + 1)):
                g.add_edge((x + 1, y), (x, y + 1))

    # turn into directed graph
    dg = networkx.DiGraph(g)

    # add node data
    for u, v in g.edges():
        d = np.sqrt(np.sum((g.nodes[u]["pos"] - g.nodes[v]["pos"]) ** 2))
        dg.edges[u, v]["len"] = d
        dg.edges[u, v]["vdiff"] = g.nodes[u]["pos"] - g.nodes[v]["pos"]
        dg.add_edge(v, u)
        dg.edges[v, u]["len"] = d
        dg.edges[v, u]["vdiff"] = g.nodes[v]["pos"] - g.nodes[u]["pos"]

    # add self edge if needed
    for v, degree in list(dg.degree()):
        if degree <= 1:
            dg.add_edge(v, v, len=0, vdiff=np.array([0, 0]))

    return dg


def build_regular_mesh_graph(
    xy,
    mesh_node_distance,
    limit_mesh_levels,
    hierarchical,
    mesh_plot_function,
    mesh_refinement_factor=9,
):
    save_graphs = {}

    # Must be a squared number, convert to integer
    mrf_1d_float = np.sqrt(mesh_refinement_factor)
    mesh_refinement_factor_1d = mrf_1d_float.round().astype(int)
    assert np.isclose(mrf_1d_float, mesh_refinement_factor_1d), (
        "For regular mesh, the mesh refinement factor must be a squared number"
        " (e.g. 4, 9, ...) as we take the square root when laying out "
        "mesh nodes in each direction. "
        f"Got {mesh_refinement_factor} with square root {mrf_1d_float}."
    )

    # Below computation is copied from wmg
    # Compute the size along x and y direction of area to cover with graph
    # This is measured in the Cartesian coordnates of xy
    coord_extent = np.ptp(xy, axis=0)
    # Number of nodes that would fit on bottom level of hierarchy,
    # in both directions
    max_nodes_bottom = (coord_extent / mesh_node_distance).astype(int)

    # Find the number of mesh levels possible in x- and y-direction,
    # and the number of leaf nodes that would correspond to
    # max_nodes_bottom/(mesh_refinement_factor_1d^mesh_levels) = 1
    max_mesh_levels_float = np.log(max_nodes_bottom) / np.log(
        mesh_refinement_factor_1d
    )

    max_mesh_levels = max_mesh_levels_float.astype(int)  # (2,)
    nleaf = np.maximum(max_nodes_bottom, 1)
    nleaf = mesh_refinement_factor_1d**max_mesh_levels
    # leaves at the bottom in each direction, if using max_mesh_levels

    # As we can not instantiate different number of mesh levels in each
    # direction, create mesh levels corresponding to the minimum of the two
    mesh_levels_to_create = max_mesh_levels.min()

    if limit_mesh_levels:
        # Limit the levels in mesh graph
        mesh_levels_to_create = min(mesh_levels_to_create, limit_mesh_levels)

    print(f"mesh_levels: {mesh_levels_to_create}, nleaf: {nleaf}")

    # Compute filtering threshold from finest level to use for all levels
    nodes_x, nodes_y = (nleaf / (mesh_refinement_factor_1d**0)).astype(int)
    min_x, min_y = np.min(xy, axis=0)
    max_x, max_y = np.max(xy, axis=0)
    offset_x = (max_x - min_x) / (2 * nodes_x)
    offset_y = (max_y - min_y) / (2 * nodes_y)
    threshold = np.sqrt(offset_x**2 + offset_y**2)

    G = []
    for lev in range(mesh_levels_to_create):  # 0-index mesh levels
        # Compute number of nodes on level separate for each direction
        nodes_x, nodes_y = (nleaf / (mesh_refinement_factor_1d**lev)).astype(
            int
        )
        g = mk_2d_graph(xy, nodes_x, nodes_y, threshold)
        if mesh_plot_function is not None:
            mesh_plot_function(from_networkx(g), f"Mesh graph, level {lev}")

        G.append(g)

    if hierarchical:
        # Relabel nodes of each level with level index first
        G = [
            gutils.prepend_node_index(graph, level_i)
            for level_i, graph in enumerate(G)
        ]

        num_nodes_level = np.array([len(g_level.nodes) for g_level in G])
        # First node index in each level in the hierarchical graph
        first_index_level = np.concatenate(
            (np.zeros(1, dtype=int), np.cumsum(num_nodes_level[:-1]))
        )

        # Create inter-level mesh edges
        up_graphs = []
        down_graphs = []
        for from_level, to_level, G_from, G_to, start_index in zip(
            range(1, mesh_levels_to_create),
            range(0, mesh_levels_to_create - 1),
            G[1:],
            G[:-1],
            first_index_level[: mesh_levels_to_create - 1],
        ):
            # start out from graph at from level
            G_down = G_from.copy()
            G_down.clear_edges()
            G_down = networkx.DiGraph(G_down)

            # Add nodes of to level
            G_down.add_nodes_from(G_to.nodes(data=True))

            # build kd tree for mesh point pos
            # order in vm should be same as in vm_xy
            v_to_list = list(G_to.nodes)
            v_from_list = list(G_from.nodes)
            v_from_xy = np.array([xy for _, xy in G_from.nodes.data("pos")])
            kdt_m = scipy.spatial.KDTree(v_from_xy)

            # add edges between levels
            for v in v_to_list:
                # find 1(?) nearest neighbours (index to vm_xy)
                neigh_idx = kdt_m.query(G_down.nodes[v]["pos"], 1)[1]
                u = v_from_list[neigh_idx]

                # add edge from mesh to grid
                G_down.add_edge(u, v)
                d = np.sqrt(
                    np.sum(
                        (G_down.nodes[u]["pos"] - G_down.nodes[v]["pos"]) ** 2
                    )
                )
                G_down.edges[u, v]["len"] = d
                G_down.edges[u, v]["vdiff"] = (
                    G_down.nodes[u]["pos"] - G_down.nodes[v]["pos"]
                )

            # relabel nodes to integers (sorted)
            G_down_int = networkx.convert_node_labels_to_integers(
                G_down, first_label=start_index, ordering="sorted"
            )  # Issue with sorting here
            G_down_int = gutils.sort_nodes_internally(G_down_int)
            pyg_down = gutils.from_networkx_with_start_index(
                G_down_int, start_index
            )

            # Create up graph, invert downwards edges
            up_edges = torch.stack(
                (pyg_down.edge_index[1], pyg_down.edge_index[0]), dim=0
            )
            pyg_up = pyg_down.clone()
            pyg_up.edge_index = up_edges

            up_graphs.append(pyg_up)
            down_graphs.append(pyg_down)

            if mesh_plot_function is not None:
                mesh_plot_function(
                    pyg_down,
                    f"Down graph, {from_level} -> {to_level}",
                )
                mesh_plot_function(
                    pyg_up,
                    f"Up graph, {to_level} -> {from_level}",
                )

        # Save up and down edges
        save_graphs["mesh_up"] = up_graphs
        save_graphs["mesh_down"] = down_graphs

        # Extract intra-level edges for m2m
        m2m_graphs = [
            gutils.from_networkx_with_start_index(
                networkx.convert_node_labels_to_integers(
                    level_graph, first_label=start_index, ordering="sorted"
                ),
                start_index,
            )
            for level_graph, start_index in zip(G, first_index_level)
        ]

        mesh_pos = [graph.pos.to(torch.float32) for graph in m2m_graphs]

        # For use in g2m and m2g
        G_bottom_mesh = G[0]
    else:
        # Non-hierarchical graph that combines all resolutions into one mesh
        G_tot = G[0].copy()
        G_tot = networkx.DiGraph(G_tot)

        # Build KDTree for position matching
        # Collect all positions from finest level for nearest neighbor lookup
        fine_nodes = list(G_tot.nodes)
        fine_positions = np.array(
            [G_tot.nodes[node]["pos"] for node in fine_nodes]
        )
        fine_kdtree = scipy.spatial.KDTree(fine_positions)

        # For each coarser level, map nodes to finest level and compose
        for lev in range(1, len(G)):
            g_level = G[lev].copy()
            g_level = networkx.DiGraph(g_level)

            # Map each node in this level to the corresponding finest level node
            node_mapping = {}
            for node in g_level.nodes:
                node_pos = g_level.nodes[node]["pos"]
                # Find nearest node in finest level using KDTree
                dist, idx = fine_kdtree.query(node_pos, k=1)
                nearest_fine_node = fine_nodes[idx]
                node_mapping[node] = nearest_fine_node

            # Relabel coarser level nodes to match finest level nodes
            g_level_relabeled = networkx.relabel_nodes(
                g_level, node_mapping, copy=True
            )

            # Add edges from coarser level
            for u, v, edge_data in g_level_relabeled.edges(data=True):
                if not G_tot.has_edge(u, v):
                    G_tot.add_edge(u, v, **edge_data)

        # Relabel nodes to sorted integers
        G_int = networkx.convert_node_labels_to_integers(
            G_tot, first_label=0, ordering="sorted"
        )
        G_int = gutils.sort_nodes_internally(G_int)

        # Graph to use in g2m and m2g
        G_bottom_mesh = G_int

        # Export the nx graph to PyTorch geometric
        pyg_m2m = from_networkx(G_int)
        m2m_graphs = [pyg_m2m]
        mesh_pos = [pyg_m2m.pos.to(torch.float32)]

        if mesh_plot_function is not None:
            mesh_plot_function(pyg_m2m, "Mesh-to-mesh")

    save_graphs["m2m"] = m2m_graphs
    return mesh_pos, G_bottom_mesh, save_graphs
