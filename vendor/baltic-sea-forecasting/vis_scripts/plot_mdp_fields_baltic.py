"""Plot all fields from Baltic MDP config zarr files in the native LCC projection."""

# Standard library
import os
import warnings

# Third-party
import cartopy.crs as ccrs
import cmocean
import matplotlib
import numpy as np
import xarray as xr
from pyproj import CRS, Transformer

matplotlib.use("Agg")
# Third-party
import matplotlib.pyplot as plt  # noqa: E402

# First-party
from neural_lam.config import load_config_and_datastores

MAIN_CONFIG = "configs/graph_rework_example/baltic_graph_rework.yaml"
BALTIC_SEA_ZARR = "configs/graph_rework_example/baltic_sea_mdp_config.zarr"
BALTIC_BND_ZARR = "configs/graph_rework_example/baltic_boundary_mdp_config.zarr"
SAVE_DIR = "plots/baltic_fields"

# Shape of the grids in projected space
SEA_SHAPE = (763, 738)  # (n_x, n_y) - lon outer, lat inner
BND_SHAPE = (49, 61)  # (n_lon, n_lat) in geographic space


def get_cmap_vrange(name, valid_data):
    if len(valid_data) == 0:
        return "viridis", 0, 1
    if any(
        x in name for x in ("uo_", "vo_", "sla", "zos", "mdt", "day_of_year")
    ):
        v = np.percentile(np.abs(valid_data), 98)
        return cmocean.cm.balance, -v, v
    elif name.startswith("sin_") or name.startswith("cos_"):
        v = np.percentile(np.abs(valid_data), 99)
        return cmocean.cm.balance, -v, v
    elif "thetao" in name:
        return (
            cmocean.cm.thermal,
            np.percentile(valid_data, 2),
            np.percentile(valid_data, 98),
        )
    elif name.startswith("so_"):
        return (
            cmocean.cm.haline,
            np.percentile(valid_data, 2),
            np.percentile(valid_data, 98),
        )
    elif "siconc" in name:
        return cmocean.cm.ice, 0.0, 1.0
    elif "sithick" in name:
        return cmocean.cm.ice, 0.0, np.percentile(valid_data, 99)
    elif "deptho" in name:
        return cmocean.cm.deep, 0.0, np.percentile(valid_data, 99)
    elif any(x in name for x in ("swh", "mwp", "mlotst", "coast_dist")):
        return "viridis", 0.0, np.percentile(valid_data, 99)
    else:
        return (
            "viridis",
            np.percentile(valid_data, 2),
            np.percentile(valid_data, 98),
        )


def save_field(x_2d, y_2d, data_2d, name, crs, extent):
    flat = data_2d.ravel()
    valid = flat[np.isfinite(flat)]
    cmap, vmin, vmax = get_cmap_vrange(name, valid)
    if vmin == vmax:
        vmax = vmin + 1e-6

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0, 0, 1, 1], projection=crs)

    ax.pcolormesh(
        x_2d,
        y_2d,
        data_2d,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=crs,
        rasterized=True,
    )

    ax.set_extent(extent, crs=crs)
    ax.set_axis_off()
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    os.makedirs(SAVE_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(
            os.path.join(SAVE_DIR, f"{name}.{ext}"),
            transparent=True,
            bbox_inches="tight",
            pad_inches=0,
            dpi=150,
        )
    plt.close(fig)
    print(f"Saved {name}")


def main():
    # Load main datastore for CRS and state projected XY
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, datastore, *_ = load_config_and_datastores(MAIN_CONFIG)

    crs = datastore.coords_projection

    # --- Baltic sea grid ---
    xy = datastore.get_projected_xy("state", stacked=False)  # (763, 738, 2)
    x_sea = xy[:, :, 0]  # (n_x, n_y) in metres
    y_sea = xy[:, :, 1]

    buf = 50_000  # 50 km border buffer
    sea_extent = [
        x_sea.min() - buf,
        x_sea.max() + buf,
        y_sea.min() - buf,
        y_sea.max() + buf,
    ]

    ds_sea = xr.open_zarr(BALTIC_SEA_ZARR)
    mask_sea = ds_sea["mask"].values  # (10, 563094)

    def flat_to_sea_2d(flat):
        return flat.reshape(SEA_SHAPE)

    state_features = list(ds_sea["state_feature"].values)
    state_arr = ds_sea["state"].values  # (10, n_time, 563094)
    mask_feat_list = list(ds_sea["mask_feature"].values)

    for i, feat in enumerate(state_features):
        data = state_arr[i, 0, :].astype(float)
        mi = mask_feat_list.index(feat)
        data[mask_sea[mi] == 0] = np.nan
        save_field(
            x_sea,
            y_sea,
            flat_to_sea_2d(data),
            f"sea_state_{feat}",
            crs,
            sea_extent,
        )

    for feat in ds_sea["forcing_feature"].values:
        data = (
            ds_sea["forcing"]
            .sel(forcing_feature=feat)
            .isel(time=0)
            .values.astype(float)
        )
        save_field(
            x_sea,
            y_sea,
            flat_to_sea_2d(data),
            f"sea_forcing_{feat}",
            crs,
            sea_extent,
        )

    for feat in ds_sea["static_feature"].values:
        data = ds_sea["static"].sel(static_feature=feat).values.astype(float)
        save_field(
            x_sea,
            y_sea,
            flat_to_sea_2d(data),
            f"sea_static_{feat}",
            crs,
            sea_extent,
        )

    for i, feat in enumerate(mask_feat_list):
        data = mask_sea[i].astype(float)
        save_field(
            x_sea,
            y_sea,
            flat_to_sea_2d(data),
            f"sea_mask_{feat}",
            crs,
            sea_extent,
        )

    # --- Baltic boundary grid ---
    ds_bnd = xr.open_zarr(BALTIC_BND_ZARR)
    lat_b = ds_bnd["latitude"].values
    lon_b = ds_bnd["longitude"].values

    # Transform geographic lat/lon to projected XY
    proj_crs = CRS.from_string(str(crs))
    transformer = Transformer.from_crs("EPSG:4326", proj_crs, always_xy=True)
    x_b_flat, y_b_flat = transformer.transform(lon_b, lat_b)
    x_bnd = x_b_flat.reshape(BND_SHAPE)
    y_bnd = y_b_flat.reshape(BND_SHAPE)

    bnd_extent = [
        x_b_flat.min() - buf,
        x_b_flat.max() + buf,
        y_b_flat.min() - buf,
        y_b_flat.max() + buf,
    ]

    mask_bnd = ds_bnd["mask"].values  # (10, 2989)
    bnd_mask_feat_list = list(ds_bnd["mask_feature"].values)

    def flat_to_bnd_2d(flat):
        return flat.reshape(BND_SHAPE)

    for feat in ds_bnd["forcing_feature"].values:
        data = (
            ds_bnd["forcing"]
            .sel(forcing_feature=feat)
            .isel(time=0)
            .values.astype(float)
        )
        mi = bnd_mask_feat_list.index(feat)
        data[mask_bnd[mi] == 0] = np.nan
        save_field(
            x_bnd,
            y_bnd,
            flat_to_bnd_2d(data),
            f"boundary_forcing_{feat}",
            crs,
            bnd_extent,
        )

    for feat in ds_bnd["static_feature"].values:
        data = ds_bnd["static"].sel(static_feature=feat).values.astype(float)
        save_field(
            x_bnd,
            y_bnd,
            flat_to_bnd_2d(data),
            f"boundary_static_{feat}",
            crs,
            bnd_extent,
        )

    for i, feat in enumerate(bnd_mask_feat_list):
        data = mask_bnd[i].astype(float)
        save_field(
            x_bnd,
            y_bnd,
            flat_to_bnd_2d(data),
            f"boundary_mask_{feat}",
            crs,
            bnd_extent,
        )


if __name__ == "__main__":
    main()
