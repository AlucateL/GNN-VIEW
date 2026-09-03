#!/bin/sh
# Make and plot example global graph for model diagram

# Make graph
python -m neural_lam.create_global_graph\
    --config_path configs/global_data_small/global_ocean_1_4.yaml\
    --type cluster\
    --connect_disconnected\
    --levels 3\
    --g2m_radius 0.5\
    --name vis_cluster\
    --grid_to_first_mesh_refinement 400\
    --mesh_refinement_factor 3\

# Visualize
python -m neural_lam.plot_global_graph_3d\
    --config_path configs/global_data_small/global_ocean_1_4.yaml\
    --graph_name vis_cluster\
    --texture_resolution 1.0\
    --grid_node_size 4\
    --mesh_level_dist 0.3\
    --mesh_height 0.1\
    --mesh_node_size 5\
    --mesh_edge_width 1.8\
