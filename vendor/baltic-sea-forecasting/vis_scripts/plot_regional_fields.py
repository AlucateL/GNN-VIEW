"""
Plot regional field snapshots comparing analysis, Njord ensemble, and SeaCast.

For each region in COORD_BOXES, lead times +5 and +10 days, and each variable,
create a 6-panel figure:
  Row 0: Analysis (GT) | Njord Ens Mean | SeaCast              + field colorbar
  Row 1: Njord Member 0 | Diff GT-Njord | Diff GT-SeaCast      + diff colorbar

Run with the `baltic` conda environment.
"""

# Standard library
import os

# Third-party
import matplotlib
import numpy as np
import zarr

matplotlib.use("Agg")
# Third-party
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer

# First-party
from neural_lam.config import load_config_and_datastores

# ---------------------------------------------------------------------------
# Paths / settings
# ---------------------------------------------------------------------------
CONFIG_PATH = "configs/graph_rework_example/baltic_graph_rework.yaml"
FC_PATH = "data/njord_baltic_example_fc.zarr"
SC_PATH = "data/seacast_baltic_example_fc.zarr"
RA_PATH = "data/cmems_baltic_analysis_2024_plot.zarr"
OUT_DIR = "plots/regional_fields"

MEMBER_IDX = 0  # which ensemble member to show
LEAD_DAYS_PLOT = [5, 10]  # lead times to plot
MISSING = -999.0
FONTSIZE = 9
TITLE_FS = 9

# Coordinate boxes in Lambert Conformal Conic projection (metres)
COORD_BOXES = {
    "full_domain": (-900_000, 650_000, -770_000, 750_000),
    "orust": (-525_000, -460_000, -201_000, -137_000),
    "braviken": (-227_000, -147_000, -197_000, -125_000),
    "aland": (-47_000, 113_000, -28_000, 83_000),
    "turku_archipelago": (60_000, 192_000, -24_000, 74_000),
}

FEATURE_CFG = {
    "sla": dict(
        base_var="sla",
        depth=None,
        cmap=cmocean.cm.balance,
        units="m",
        long_name="Sea Level Anomaly",
        symmetric=True,
    ),
    "siconc": dict(
        base_var="siconc",
        depth=None,
        cmap=cmocean.cm.ice,
        units="",
        long_name="Sea Ice Concentration",
        symmetric=False,
        vmin=0.0,
        vmax=1.0,
    ),
    "sithick": dict(
        base_var="sithick",
        depth=None,
        cmap=cmocean.cm.ice,
        units="m",
        long_name="Sea Ice Thickness",
        symmetric=False,
    ),
}
for base_var, kw in [
    (
        "thetao",
        dict(
            cmap=cmocean.cm.thermal,
            units="°C",
            long_name="Potential Temperature",
            symmetric=False,
        ),
    ),
    (
        "so",
        dict(
            cmap=cmocean.cm.haline,
            units="PSU",
            long_name="Salinity",
            symmetric=False,
        ),
    ),
    (
        "uo",
        dict(
            cmap=cmocean.cm.balance,
            units="m/s",
            long_name="Eastward Current",
            symmetric=True,
        ),
    ),
    (
        "vo",
        dict(
            cmap=cmocean.cm.balance,
            units="m/s",
            long_name="Northward Current",
            symmetric=True,
        ),
    ),
]:
    for depth in [1, 9, 28, 47, 91]:
        FEATURE_CFG[f"{base_var}_{depth}m"] = dict(
            base_var=base_var, depth=depth, **kw
        )

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data ...", flush=True)
fc = zarr.open(FC_PATH)
sc = zarr.open(SC_PATH)
ra = zarr.open(RA_PATH)

features = [str(f) for f in fc["state_feature"][:]]
sc_features = [str(f) for f in sc["state_feature"][:]]
lead_days = fc["lead_time"][:]  # [1 2 3 ... 10]
n_lt = len(lead_days)

ra_depth = ra["depth"][:]
DEPTH_IDX = {
    d: int(np.argmin(np.abs(ra_depth - d))) for d in [1, 9, 28, 47, 91]
}

fc_lat = fc["latitude"][:]  # (738,)
fc_lon = fc["longitude"][:]  # (763,)
ra_lat = ra["latitude"][:]  # (738,)
ra_lon = ra["longitude"][:]  # (762,)

# ---------------------------------------------------------------------------
# Build projected-to-latlon transformer from the LCC CRS
# ---------------------------------------------------------------------------
print("Setting up CRS transformer ...", flush=True)
_, datastore, *_ = load_config_and_datastores(config_path=CONFIG_PATH)
lcc_crs = CRS.from_string(str(datastore.coords_projection))
transformer = Transformer.from_crs(lcc_crs, "EPSG:4326", always_xy=True)

# Projected XY of the FC grid: (n_lon=763, n_lat=738, 2)
grid_xy = datastore.get_projected_xy("state", stacked=False)
grid_x = grid_xy[:, :, 0]  # (763, 738)
grid_y = grid_xy[:, :, 1]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def box_latlon(x_min, x_max, y_min, y_max, pad=0.1):
    """Convert projected box corners to (lon_min, lon_max, lat_min, lat_max)."""
    xs = [x_min, x_min, x_max, x_max]
    ys = [y_min, y_max, y_min, y_max]
    lons, lats = transformer.transform(xs, ys)
    return min(lons) - pad, max(lons) + pad, min(lats) - pad, max(lats) + pad


def idx_range(arr, vmin, vmax):
    """First and last+1 indices where arr is within [vmin, vmax]."""
    mask = (arr >= vmin) & (arr <= vmax)
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return 0, len(arr)
    return idxs[0], idxs[-1] + 1


def load_forecast_lt(feat_idx, lt_idx):
    """Return (n_ens, n_lat, n_lon) for a single lead time."""
    # fc["prediction"]: (n_ens, n_lt, n_feat, n_lon, n_lat)
    data = fc["prediction"][:, lt_idx, feat_idx, :, :]  # (n_ens, n_lon, n_lat)
    return np.transpose(data.astype(float), (0, 2, 1))  # (n_ens, n_lat, n_lon)


def load_seacast_lt(feature, lt_idx):
    """Return (n_lat, n_lon) for a single lead time."""
    sc_feat_idx = sc_features.index(feature)
    data = sc["prediction"][lt_idx, sc_feat_idx, :, :]  # (n_lon, n_lat)
    return np.transpose(data.astype(float))  # (n_lat, n_lon)


def load_reanalysis_lt(base_var, depth, lt_idx):
    """Return (n_lat, n_lon_ra) for a single lead time, land=NaN."""
    if depth is None:
        data = ra[base_var][lt_idx, :, :]
    else:
        data = ra[base_var][lt_idx, DEPTH_IDX[depth], :, :]
    data = data.astype(float)
    data[np.abs(data - MISSING) < 1.0] = np.nan
    return data


def field_vrange(fc_slice, ra_slice, sc_slice, symmetric, cfg):
    if "vmin" in cfg:
        return cfg["vmin"], cfg["vmax"]
    merged = np.concatenate(
        [
            fc_slice[np.isfinite(fc_slice)].ravel(),
            ra_slice[np.isfinite(ra_slice)].ravel(),
            sc_slice[np.isfinite(sc_slice)].ravel(),
        ]
    )
    if len(merged) == 0:
        return 0, 1
    if symmetric:
        v = np.percentile(np.abs(merged), 99)
        return -v, v
    return np.percentile(merged, 1), np.percentile(merged, 99)


def diff_vrange(*diffs):
    merged = np.concatenate([d[np.isfinite(d)].ravel() for d in diffs])
    if len(merged) == 0:
        return -1, 1
    v = np.nanpercentile(np.abs(merged), 99)
    return -v, v


def add_map_features(ax):
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=2)
    ax.set_axis_off()


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------


def make_plot(
    feature, feat_idx, cfg, region_name, x_min, x_max, y_min, y_max, lead_day
):
    lt_idx = int(np.where(lead_days == lead_day)[0][0])

    lon_min, lon_max, lat_min, lat_max = box_latlon(x_min, x_max, y_min, y_max)

    # Array index slices for FC/SC (763 lon) and RA (762 lon)
    lat0, lat1 = idx_range(fc_lat, lat_min, lat_max)
    lon0_fc, lon1_fc = idx_range(fc_lon, lon_min, lon_max)
    lon0_ra, lon1_ra = idx_range(ra_lon, lon_min, lon_max)

    # Load and slice
    base_var = cfg["base_var"]
    depth = cfg["depth"]

    fc_data = load_forecast_lt(feat_idx, lt_idx)  # (n_ens, 738, 763)
    sc_data = load_seacast_lt(feature, lt_idx)  # (738, 763)
    ra_data = load_reanalysis_lt(base_var, depth, lt_idx)  # (738, 762)

    # Slices
    fc_s = fc_data[:, lat0:lat1, lon0_fc:lon1_fc]  # (n_ens, n_lat_r, n_lon_r)
    sc_s = sc_data[lat0:lat1, lon0_fc:lon1_fc]
    ra_s = ra_data[lat0:lat1, lon0_ra:lon1_ra]
    em_s = np.nanmean(fc_s, axis=0)  # (n_lat_r, n_lon_r)
    mb_s = fc_s[MEMBER_IDX]

    # Align RA with FC lon for difference (trim to min width)
    n_lon_fc = lon1_fc - lon0_fc
    n_lon_ra = lon1_ra - lon0_ra
    n_lon = min(n_lon_fc, n_lon_ra)
    em_aligned = em_s[:, :n_lon]
    sc_aligned = sc_s[:, :n_lon]
    ra_aligned = ra_s[:, :n_lon]
    diff_njord = ra_aligned - em_aligned
    diff_sc = ra_aligned - sc_aligned

    # Coordinate meshgrids for pcolormesh
    lat_r = fc_lat[lat0:lat1]
    lon_r_fc = fc_lon[lon0_fc:lon1_fc]
    lon_r_ra = ra_lon[lon0_ra : lon0_ra + n_lon]

    lon2d_fc, lat2d_fc = np.meshgrid(lon_r_fc, lat_r)
    lon2d_ra, lat2d_ra = np.meshgrid(lon_r_ra, lat_r)

    # Color ranges
    vmin, vmax = field_vrange(em_s, ra_s, sc_s, cfg["symmetric"], cfg)
    dv = diff_vrange(diff_njord, diff_sc)
    dv_min, dv_max = -max(abs(dv[0]), abs(dv[1])), max(abs(dv[0]), abs(dv[1]))

    cmap_field = cfg["cmap"]
    cmap_diff = cmocean.cm.balance
    unit_str = f" ({cfg['units']})" if cfg["units"] else ""
    depth_str = f" at {depth} m" if depth is not None else ""

    # Extent for cartopy axes
    extent = [lon_min, lon_max, lat_min, lat_max]
    proj = ccrs.PlateCarree()

    # -------------------------------------------------------------------
    # Figure layout: 2 rows x 4 cols (col 3 = colorbars)
    # -------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[1, 1, 1, 0.045],
        wspace=0.04,
        hspace=0.18,
        left=0.02,
        right=0.94,
        top=0.88,
        bottom=0.04,
    )

    # Row 0: GT, Njord mean, SeaCast, field cbar
    ax_gt = fig.add_subplot(gs[0, 0], projection=proj)
    ax_em = fig.add_subplot(gs[0, 1], projection=proj)
    ax_sc = fig.add_subplot(gs[0, 2], projection=proj)
    ax_cb1 = fig.add_subplot(gs[0, 3])

    # Row 1: Njord member, Diff Njord, Diff SC, diff cbar
    ax_mb = fig.add_subplot(gs[1, 0], projection=proj)
    ax_dn = fig.add_subplot(gs[1, 1], projection=proj)
    ax_ds = fig.add_subplot(gs[1, 2], projection=proj)
    ax_cb2 = fig.add_subplot(gs[1, 3])

    kw_pc = dict(transform=proj, zorder=3)

    for ax in [ax_gt, ax_em, ax_sc, ax_mb, ax_dn, ax_ds]:
        ax.set_extent(extent, crs=proj)
        add_map_features(ax)

    def pmesh(ax, lon2d, lat2d, data, **kwargs):
        return ax.pcolormesh(lon2d, lat2d, data, **kw_pc, **kwargs)

    im_gt = pmesh(
        ax_gt, lon2d_ra, lat2d_ra, ra_s, cmap=cmap_field, vmin=vmin, vmax=vmax
    )
    pmesh(
        ax_em, lon2d_fc, lat2d_fc, em_s, cmap=cmap_field, vmin=vmin, vmax=vmax
    )
    pmesh(
        ax_sc, lon2d_fc, lat2d_fc, sc_s, cmap=cmap_field, vmin=vmin, vmax=vmax
    )
    pmesh(
        ax_mb, lon2d_fc, lat2d_fc, mb_s, cmap=cmap_field, vmin=vmin, vmax=vmax
    )
    im_dn = pmesh(
        ax_dn,
        lon2d_ra,
        lat2d_ra,
        diff_njord,
        cmap=cmap_diff,
        vmin=dv_min,
        vmax=dv_max,
    )
    pmesh(
        ax_ds,
        lon2d_ra,
        lat2d_ra,
        diff_sc,
        cmap=cmap_diff,
        vmin=dv_min,
        vmax=dv_max,
    )

    # Colorbars
    cb1 = fig.colorbar(im_gt, cax=ax_cb1)
    cb1.set_label(f"{feature}{unit_str}", fontsize=FONTSIZE)
    cb1.ax.tick_params(labelsize=FONTSIZE - 1)

    cb2 = fig.colorbar(im_dn, cax=ax_cb2)
    cb2.set_label(f"GT − forecast{unit_str}", fontsize=FONTSIZE)
    cb2.ax.tick_params(labelsize=FONTSIZE - 1)

    # Titles
    ax_gt.set_title("Analysis (GT)", fontsize=TITLE_FS, pad=3)
    ax_em.set_title("Njord Ens. Mean", fontsize=TITLE_FS, pad=3)
    ax_sc.set_title("SeaCast", fontsize=TITLE_FS, pad=3)
    ax_mb.set_title(f"Njord Member {MEMBER_IDX + 1}", fontsize=TITLE_FS, pad=3)
    ax_dn.set_title("GT − Njord Mean", fontsize=TITLE_FS, pad=3)
    ax_ds.set_title("GT − SeaCast", fontsize=TITLE_FS, pad=3)

    fig.suptitle(
        f"{cfg['long_name']}{depth_str}  ·  {region_name.replace('_', ' ').title()}"
        f"  ·  Lead time +{lead_day} days",
        fontsize=TITLE_FS + 2,
        y=0.96,
    )

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    fname = f"{region_name}_{feature}_lt{lead_day:02d}d.png"
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}", flush=True)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    for feat_idx, feature in enumerate(features):
        if feature not in FEATURE_CFG:
            print(f"[skip] {feature} — no config")
            continue

        cfg = FEATURE_CFG[feature]
        print(f"\n{feature}", flush=True)

        for region_name, (x_min, x_max, y_min, y_max) in COORD_BOXES.items():
            for lead_day in LEAD_DAYS_PLOT:
                make_plot(
                    feature,
                    feat_idx,
                    cfg,
                    region_name,
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                    lead_day,
                )

    print(f"\nDone. Plots saved to {OUT_DIR}/")
