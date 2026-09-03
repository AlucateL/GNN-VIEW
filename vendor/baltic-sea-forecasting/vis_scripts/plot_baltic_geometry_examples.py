# Standard library
import os

# Third-party
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap
from tueplots import fonts

# First-party
from neural_lam.config import load_config_and_datastores

CONFIG_PATH = "configs/graph_rework_example/baltic_graph_rework.yaml"
GRAPH_NAMES = ["cluster", "hierarchical"]
SAVE_DIR = "plots/graph_2d_baltic"

# Coordinate boxes in projected CRS (metres):
# label -> (x_min, x_max, y_min, y_max, legend_loc)
COORD_BOXES = {
    "full_domain": (-900_000, 650_000, -770_000, 750_000, "upper left"),
    "orust": (-525_000, -460_000, -201_000, -137_000, "upper right"),
    "braviken": (-227_000, -147_000, -197_000, -125_000, "lower left"),
    "aland": (-47_000, 113_000, -28_000, 83_000, "upper left"),
    "turku_archipelago": (60_000, 192_000, -24_000, 74_000, "upper right"),
}

MESH_LEVELS_TO_PLOT = [0, 2]

GRID_COLOR = "dodgerblue"
MESH_NODE_SIZE = {0: 60.0, 2: 160.0}
MESH_COLOR = "orange"
LEGEND_MARKER_SIZE = 24


plt.rcParams.update(fonts.neurips2024())


def main():
    """Plot graph structure in 2D proj coordinates using matplotlib/cartopy"""
    _, datastore, *_ = load_config_and_datastores(config_path=CONFIG_PATH)

    crs = datastore.coords_projection

    interior_mask = datastore.get_mask(
        surface=True, stacked=False, invert=False
    )  # (N_x, N_y)
    grid_xy = datastore.get_projected_xy(
        "state", stacked=False
    )  # (N_x, N_y, 2)
    grid_x = grid_xy[:, :, 0]
    grid_y = grid_xy[:, :, 1]
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

    for graph_name in GRAPH_NAMES:
        graph_dir_path = os.path.join(datastore.root_path, "graphs", graph_name)
        mesh_pos_list = torch.load(
            os.path.join(graph_dir_path, "mesh_features.pt"),
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(mesh_pos_list, list):
            mesh_pos_list = [mesh_pos_list]

        n_levels = len(mesh_pos_list)

        for box_label, (
            x_min,
            x_max,
            y_min,
            y_max,
            legend_loc,
        ) in COORD_BOXES.items():
            for mesh_level in MESH_LEVELS_TO_PLOT:
                if mesh_level >= n_levels:
                    print(
                        f"Skipping {graph_name} mesh level {mesh_level}: "
                        f"graph only has {n_levels} level(s)."
                    )
                    continue

                mesh_xy = mesh_pos_list[mesh_level].numpy()[:, :2]

                fig, ax = plt.subplots(
                    figsize=(10, 8),
                    subplot_kw={"projection": crs},
                )
                fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                ax.set_facecolor("forestgreen")
                ax.set_extent([x_min, x_max, y_min, y_max], crs=crs)

                ax.pcolormesh(
                    grid_x,
                    grid_y,
                    grid_values,
                    cmap=grid_cmap,
                    vmin=0,
                    vmax=1,
                    edgecolors="none",
                    transform=crs,
                    zorder=3,
                )
                ax.scatter(
                    mesh_xy[:, 0],
                    mesh_xy[:, 1],
                    s=MESH_NODE_SIZE[mesh_level],
                    c=MESH_COLOR,
                    transform=crs,
                    zorder=4,
                    linewidths=0,
                )

                legend = ax.legend(
                    handles=[grid_legend, mesh_legend],
                    loc=legend_loc,
                    # markerscale=5,
                    fontsize=30,
                )

                #  if box_label == "full_domain":
                #  plt.show()

                base_name = f"{graph_name}_{box_label}_level{mesh_level}"
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
