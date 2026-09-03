"""Plot all fields from global ocean MDP config zarr in Robinson projection."""

# Standard library
import os

# Third-party
import cartopy.crs as ccrs
import cmocean
import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
# Third-party
import matplotlib.pyplot as plt  # noqa: E402

ZARR_PATH = "configs/global_data_small/global_ocean_1_4_mdp_config.zarr"
SAVE_DIR = "plots/global_fields"
N_LON = 1440
N_LAT = 680

ROBINSON = ccrs.Robinson()
PLATE_CARREE = ccrs.PlateCarree()


def get_cmap_vrange(name, valid_data):
    if len(valid_data) == 0:
        return "viridis", 0, 1
    if any(x in name for x in ("uo_", "vo_", "zos", "mdt", "day_of_year")):
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
    elif "coast_dist" in name:
        return "viridis", 0.0, np.percentile(valid_data, 99)
    else:
        return (
            "viridis",
            np.percentile(valid_data, 2),
            np.percentile(valid_data, 98),
        )


def save_field(lon_2d, lat_2d, data_2d, name):
    flat = data_2d.ravel()
    valid = flat[np.isfinite(flat)]
    cmap, vmin, vmax = get_cmap_vrange(name, valid)
    if vmin == vmax:
        vmax = vmin + 1e-6

    fig = plt.figure(figsize=(10, 5.4))
    ax = fig.add_axes([0, 0, 1, 1], projection=ROBINSON)

    ax.pcolormesh(
        lon_2d,
        lat_2d,
        data_2d,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=PLATE_CARREE,
        rasterized=True,
    )

    ax.set_global()
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


def flat_to_2d(flat_data):
    """Reshape flat (grid_index,) array to (N_LAT, N_LON) for pcolormesh."""
    return flat_data.reshape(N_LON, N_LAT).T


def main():
    ds = xr.open_zarr(ZARR_PATH)

    lat = ds["latitude"].values
    lon = ds["longitude"].values

    # Grid stored as (lon outer, lat inner): reshape to (N_LON, N_LAT),
    # then transpose to (N_LAT, N_LON) for pcolormesh row=lat, col=lon
    lat_2d = lat.reshape(N_LON, N_LAT).T
    lon_2d = lon.reshape(N_LON, N_LAT).T

    mask_arr = ds["mask"].values  # (27, 979200)

    for feat in ds["state_feature"].values:
        data = (
            ds["state"]
            .sel(state_feature=feat)
            .isel(time=0)
            .values.astype(float)
        )
        feat_idx = list(ds["mask_feature"].values).index(feat)
        data[mask_arr[feat_idx] == 0] = np.nan
        save_field(lon_2d, lat_2d, flat_to_2d(data), f"state_{feat}")

    for feat in ds["forcing_feature"].values:
        data = (
            ds["forcing"]
            .sel(forcing_feature=feat)
            .isel(time=0)
            .values.astype(float)
        )
        save_field(lon_2d, lat_2d, flat_to_2d(data), f"forcing_{feat}")

    for feat in ds["static_feature"].values:
        data = ds["static"].sel(static_feature=feat).values.astype(float)
        save_field(lon_2d, lat_2d, flat_to_2d(data), f"static_{feat}")

    for i, feat in enumerate(ds["mask_feature"].values):
        data = mask_arr[i].astype(float)
        save_field(lon_2d, lat_2d, flat_to_2d(data), f"mask_{feat}")


if __name__ == "__main__":
    main()
