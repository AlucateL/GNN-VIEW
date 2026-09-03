# Standard library
import os

# Third-party
import cartopy.crs as ccrs
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from tueplots import fonts

# First-party
from neural_lam.config import load_config_and_datastores

CONFIG_PATH = "configs/global_data_small/global_ocean_1_4.yaml"
GRAPH_DIR_TO_NAME = {
    "global_cluster_1_4_deg_20_refinement_3_levels": "cluster",
    "global_hierarchical_1_4_deg_6_splits_3_levels": "hierarchical",
}
SAVE_DIR = "plots/graph_2d_global"

# Coordinate boxes in lon/lat degrees: label ->
# (lon_min, lon_max, lat_min, lat_max, legend_loc)
COORD_BOXES = {
    "full_domain": (-180, 180, -90, 90, "lower left"),
    "baltic": (3, 31, 53, 66, "lower right"),
    "indonesia": (92, 154, -17, 19, "upper right"),
    "northern_red_sea": (28, 44, 10, 33, "lower left"),
    "gulf_of_california": (-117, -105, 22, 33, "upper right"),
}

MESH_LEVELS_TO_PLOT = [0, 2]

GRID_COLOR = "dodgerblue"
MESH_NODE_SIZE = {0: 60.0, 2: 140.0}
MESH_COLOR = "orange"
LEGEND_MARKER_SIZE = 24


plt.rcParams.update(fonts.neurips2024())

PLATE_CARREE = ccrs.PlateCarree()


def mesh_features_to_lon_lat(feats: np.ndarray) -> np.ndarray:
    """
    Convert global mesh node features (sin_lon, cos_lon,
    sin_lat, cos_lat, ...) to (lon, lat) degrees.
    """
    sin_lon, cos_lon, sin_lat, cos_lat = (
        feats[:, 0],
        feats[:, 1],
        feats[:, 2],
        feats[:, 3],
    )
    lon_deg = np.rad2deg(np.arctan2(sin_lon, cos_lon)).astype(np.float32)
    lat_deg = np.rad2deg(np.arctan2(sin_lat, cos_lat)).astype(np.float32)
    return np.stack([lon_deg, lat_deg], axis=1)


def main():
    """Plot global graph structure in 2D lon/lat using matplotlib/cartopy."""
    _, datastore, *_ = load_config_and_datastores(config_path=CONFIG_PATH)

    interior_mask = datastore.get_mask(
        surface=True, stacked=False, invert=False
    )  # (N_lon, N_lat)
    grid_xy = datastore.get_xy(
        "state", stacked=False
    )  # (N_lon, N_lat, 2): (lon, lat)
    grid_lon = grid_xy[:, :, 0]
    grid_lat = grid_xy[:, :, 1]
    grid_values = interior_mask.astype(float)
    grid_values[~interior_mask] = float("nan")

    grid_cmap = ListedColormap([GRID_COLOR])
    grid_legend = mlines.Line2D(
        [],
        [],
        color=GRID_COLOR,
        marker="s",
        linestyle="None",
        markersize=LEGEND_MARKER_SIZE,
        label="Sea surface grid",
    )
    mesh_legend = mlines.Line2D(
        [],
        [],
        color=MESH_COLOR,
        marker="o",
        linestyle="None",
        markersize=LEGEND_MARKER_SIZE,
        label="Mesh nodes",
    )

    os.makedirs(SAVE_DIR, exist_ok=True)

    for graph_name, graph_short_name in GRAPH_DIR_TO_NAME.items():
        graph_dir_path = os.path.join(datastore.root_path, "graphs", graph_name)
        mesh_pos_list = torch.load(
            os.path.join(graph_dir_path, "mesh_features.pt"),
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(mesh_pos_list, list):
            mesh_pos_list = [mesh_pos_list]

        n_levels = len(mesh_pos_list)
        mesh_lon_lat_list = [
            mesh_features_to_lon_lat(p.numpy()) for p in mesh_pos_list
        ]

        for box_label, (
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            legend_loc,
        ) in COORD_BOXES.items():
            for mesh_level in MESH_LEVELS_TO_PLOT:
                if mesh_level >= n_levels:
                    print(
                        f"Skipping {graph_name} mesh level {mesh_level}: "
                        f"graph only has {n_levels} level(s)."
                    )
                    continue

                mesh_lon_lat = mesh_lon_lat_list[mesh_level]

                fig, ax = plt.subplots(
                    figsize=(10, 8),
                    subplot_kw={"projection": PLATE_CARREE},
                )
                fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                ax.set_facecolor("forestgreen")
                ax.set_extent(
                    [lon_min, lon_max, lat_min, lat_max], crs=PLATE_CARREE
                )

                ax.pcolormesh(
                    grid_lon,
                    grid_lat,
                    grid_values,
                    cmap=grid_cmap,
                    vmin=0,
                    vmax=1,
                    edgecolors="none",
                    transform=PLATE_CARREE,
                    zorder=3,
                )
                ax.scatter(
                    mesh_lon_lat[:, 0],
                    mesh_lon_lat[:, 1],
                    s=MESH_NODE_SIZE[mesh_level],
                    c=MESH_COLOR,
                    transform=PLATE_CARREE,
                    zorder=4,
                    linewidths=0,
                )

                legend = ax.legend(
                    handles=[grid_legend, mesh_legend],
                    loc=legend_loc,
                    fontsize=30,
                )

                #  if box_label == "full_domain":
                #  plt.show()

                base_name = f"{graph_short_name}_{box_label}_level{mesh_level}"
                for ext in ("pdf", "png"):
                    fig.savefig(
                        os.path.join(SAVE_DIR, f"{base_name}_legend.{ext}"),
                        bbox_inches="tight",
                    )
                legend.remove()
                for ext in ("pdf", "png"):
                    fig.savefig(
                        os.path.join(SAVE_DIR, f"{base_name}.{ext}"),
                        bbox_inches="tight",
                    )
                print(f"Saved {base_name} (with and without legend)")
                plt.close(fig)


if __name__ == "__main__":
    main()
