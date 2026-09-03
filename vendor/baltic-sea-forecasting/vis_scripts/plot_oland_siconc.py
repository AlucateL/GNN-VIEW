"""
Static plot of sea ice concentration around Öland island at 7-day lead time.
Row 0: Analysis | Neural-ROM Ens. Mean | SeaCast | (field colorbar) | (empty)
Row 1: Member 1 | Member 2 | Member 3  | Ens. Std. | (std colorbar)
Run with the `baltic` conda environment.
"""

# Standard library
import os

# Third-party
import matplotlib
import numpy as np
import zarr

matplotlib.use("Agg")
# Standard library
from datetime import datetime, timedelta

# Third-party
import cmocean
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

FC_PATH = "data/njord_baltic_example_fc.zarr"
SC_PATH = "data/seacast_baltic_example_fc.zarr"
RA_PATH = "data/cmems_baltic_analysis_2024_plot.zarr"
OUT_DIR = "plots"
LEAD_DAY = 9  # plot this lead time (days)
FONTSIZE = 14
TITLE_FS = 14

# Öland region bounding box (degrees)
LAT_MIN, LAT_MAX = 55.5, 57.5
LON_MIN, LON_MAX = 15.5, 18.0

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
fc = zarr.open(FC_PATH)
sc = zarr.open(SC_PATH)
ra = zarr.open(RA_PATH)

fc_lat = fc["latitude"][:]
fc_lon = fc["longitude"][:]
ra_lat = ra["latitude"][:]
ra_lon = ra["longitude"][:]

lead_days = fc["lead_time"][:]
lt_idx = int(np.argmin(np.abs(lead_days - LEAD_DAY)))
assert (
    lead_days[lt_idx] == LEAD_DAY
), f"Lead day {LEAD_DAY} not found; got {lead_days[lt_idx]}"

fc_features = [str(f) for f in fc["state_feature"][:]]
feat_idx = fc_features.index("siconc")

sc_features = [str(f) for f in sc["state_feature"][:]]
sc_feat_idx = sc_features.index("siconc")

init_s = int(fc["init_time"][()])
init_dt = datetime(1970, 1, 1) + timedelta(seconds=init_s)

# Crop indices (fc and sc share the same lat/lon grid)
fc_lat_idx = np.where((fc_lat >= LAT_MIN) & (fc_lat <= LAT_MAX))[0]
fc_lon_idx = np.where((fc_lon >= LON_MIN) & (fc_lon <= LON_MAX))[0]
ra_lat_idx = np.where((ra_lat >= LAT_MIN) & (ra_lat <= LAT_MAX))[0]
ra_lon_idx = np.where((ra_lon >= LON_MIN) & (ra_lon <= LON_MAX))[0]

fc_lat_sl = slice(fc_lat_idx[0], fc_lat_idx[-1] + 1)
fc_lon_sl = slice(fc_lon_idx[0], fc_lon_idx[-1] + 1)
ra_lat_sl = slice(ra_lat_idx[0], ra_lat_idx[-1] + 1)
ra_lon_sl = slice(ra_lon_idx[0], ra_lon_idx[-1] + 1)

# FC prediction: (n_ens, n_lt, n_feat, n_lon, n_lat) -> (n_ens, n_lat_c, n_lon_c)
fc_raw = fc["prediction"][:, lt_idx, feat_idx, :, :]  # (n_ens, n_lon, n_lat)
fc_raw = np.transpose(fc_raw.astype(float), (0, 2, 1))  # (n_ens, n_lat, n_lon)
fc_crop = fc_raw[:, fc_lat_sl, fc_lon_sl]

# SeaCast prediction: (n_lt, n_feat, n_lon, n_lat) -> (n_lat_c, n_lon_c)
sc_raw = sc["prediction"][lt_idx, sc_feat_idx, :, :].astype(
    float
)  # (n_lon, n_lat)
sc_crop = sc_raw.T[fc_lat_sl, fc_lon_sl]  # (n_lat_c, n_lon_c)

# RA siconc: (n_lt, n_lat, n_lon)
MISSING = -999.0
ra_raw = ra["siconc"][lt_idx, :, :].astype(float)
ra_raw[np.abs(ra_raw - MISSING) < 1.0] = np.nan
ra_crop = ra_raw[ra_lat_sl, ra_lon_sl]

ens_mean = np.nanmean(fc_crop, axis=0)
ens_std = np.nanstd(fc_crop, axis=0)

std_vmax = float(np.nanpercentile(ens_std[np.isfinite(ens_std)], 99))

# Extent for imshow
fc_extent = [
    fc_lon[fc_lon_idx[0]],
    fc_lon[fc_lon_idx[-1]],
    fc_lat[fc_lat_idx[0]],
    fc_lat[fc_lat_idx[-1]],
]
ra_extent = [
    ra_lon[ra_lon_idx[0]],
    ra_lon[ra_lon_idx[-1]],
    ra_lat[ra_lat_idx[0]],
    ra_lat[ra_lat_idx[-1]],
]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
DISPLAY_MEMBERS = [0, 1, 2]
n_members = len(DISPLAY_MEMBERS)

fig = plt.figure(figsize=(19, 9))
gs_outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.18)

# Row 0: Analysis | Neural-ROM Ens. Mean | SeaCast | field cbar | (empty)
gs_top = gridspec.GridSpecFromSubplotSpec(
    1,
    5,
    subplot_spec=gs_outer[0],
    width_ratios=[1, 1, 1, 0.055, 1],
    wspace=0.06,
)
ax_ra = fig.add_subplot(gs_top[0, 0])
ax_mean = fig.add_subplot(gs_top[0, 1])
ax_sc = fig.add_subplot(gs_top[0, 2])
ax_cbar_field = fig.add_subplot(gs_top[0, 3])
# gs_top[0, 4] intentionally left empty

# Row 1: Member 1 | Member 2 | Member 3 | Ens. Std. | std cbar
gs_bot = gridspec.GridSpecFromSubplotSpec(
    1,
    5,
    subplot_spec=gs_outer[1],
    width_ratios=[1, 1, 1, 1, 0.055],
    wspace=0.06,
)
ax_members = [fig.add_subplot(gs_bot[0, i]) for i in range(n_members)]
ax_std = fig.add_subplot(gs_bot[0, 3])
ax_cbar_std = fig.add_subplot(gs_bot[0, 4])

kw_ice = dict(
    origin="lower",
    aspect="auto",
    interpolation="nearest",
    cmap=cmocean.cm.ice,
    vmin=0.0,
    vmax=1.0,
)
kw_std = dict(
    origin="lower",
    aspect="auto",
    interpolation="nearest",
    cmap=cmocean.cm.amp,
    vmin=0.0,
    vmax=std_vmax,
)

im_ra = ax_ra.imshow(ra_crop, **kw_ice, extent=ra_extent)
im_mean = ax_mean.imshow(ens_mean, **kw_ice, extent=fc_extent)
im_sc = ax_sc.imshow(sc_crop, **kw_ice, extent=fc_extent)
im_std = ax_std.imshow(ens_std, **kw_std, extent=fc_extent)

im_members = [
    ax.imshow(fc_crop[m], **kw_ice, extent=fc_extent)
    for m, ax in zip(DISPLAY_MEMBERS, ax_members)
]

# Colorbars
cb_field = fig.colorbar(im_ra, cax=ax_cbar_field)
cb_field.set_label("Sea Ice Concentration", fontsize=FONTSIZE - 2)
cb_field.ax.tick_params(labelsize=FONTSIZE - 3)

cb_std = fig.colorbar(im_std, cax=ax_cbar_std)
cb_std.set_label("Ens. Std.", fontsize=FONTSIZE - 2)
cb_std.ax.tick_params(labelsize=FONTSIZE - 3)

for ax in [ax_ra, ax_mean, ax_sc, ax_std] + ax_members:
    ax.set_axis_off()

# Titles
ax_ra.set_title("Analysis", fontsize=TITLE_FS)
ax_mean.set_title("Neural-ROM Ens. Mean", fontsize=TITLE_FS)
ax_sc.set_title("SeaCast", fontsize=TITLE_FS)
ax_std.set_title("Ens. Std.", fontsize=TITLE_FS)
for m, ax in zip(DISPLAY_MEMBERS, ax_members):
    ax.set_title(f"Member {m + 1}", fontsize=TITLE_FS)

init_str = init_dt.strftime("%Y-%m-%dT%H")
fig.suptitle(
    f"Sea Ice Concentration around Öland — {init_str} + {LEAD_DAY} days",
    fontsize=TITLE_FS + 1,
    y=1.01,
)

out_path = os.path.join(OUT_DIR, f"oland_siconc_{LEAD_DAY}day.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
print(f"Saved {out_path}")
plt.close(fig)
