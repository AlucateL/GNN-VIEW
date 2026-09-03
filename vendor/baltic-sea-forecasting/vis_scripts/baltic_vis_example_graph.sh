#!/bin/sh
# Make and plot example graph for model diagram

# Make graph
python -m neural_lam.create_graph\
    --config_path configs/graph_rework_example/baltic_graph_rework.yaml\
    --type cluster\
    --allow_disconnected\
    --levels 2\
    --name vis_cluster\
    --grid_to_first_mesh_refinement 100\
    --g2m_radius 0.5\

# Visualize
python -m neural_lam.plot_graph_3d\
    --config_path configs/graph_rework_example/baltic_graph_rework.yaml\
    --graph_name vis_cluster\
    --texture_resolution 1.0\
    --grid_node_size 4\
    --mesh_level_dist 0.02\
    --mesh_node_size 3\
    --mesh_edge_width 0.8\
