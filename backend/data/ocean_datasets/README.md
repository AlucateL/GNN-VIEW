# `ocean_datasets`

Ocean counterpart of `era5_datasets/`, same layout: `static/` for the per-variable constants
loaded at inference, `forcing/` for forcing artifacts, and sample zarr windows at the top
level.

## Provenance

| Artifact | Source | Notes |
| --- | --- | --- |
| `static/parameter_mean.pt`, `parameter_std.pt` | `global_ocean_1_4_mdp_config.zarr` (`state__train__mean` / `std`) | 27 values each, one per state feature |
| `static/diff_mean.pt`, `diff_std.pt` | same, `state__train__diff_mean` / `diff_std` | one-step difference statistics |
| `static/state_features.json` | same, `state_feature` coord | the channel order — see below |
| `static/forcing_mean.pt`, `forcing_std.pt` | `global_ocean_1_4_mdp_config.zarr` (`forcing__train__mean` / `std`) | 2 values, day-of-year sin/cos |
| `static/atmosphere_mean.pt`, `atmosphere_std.pt` | `slice_global_era5_1_4_mdp_config.zarr` (`forcing__train__mean` / `std`) | 8 values, the IFS atmospheric fields |

Note the asymmetry on the atmosphere: the **data** comes from the IFS store, its **statistics**
from the ERA5 one. That is what `analysis_ocean_forecasts.yaml` specifies —
`datastore_atmosphere: analysis_ifs_forecasts` but
`statistics_datastore_atmosphere: global_era5_1_4`. The IFS store carries no statistics of its
own, and the `analysis_*` stores carry all-NaN placeholders, which is why the config points at
`global_*` for every statistic. Use the plain `mean`/`std`, never `diff_mean`/`diff_std` —
`mdp.py:344` gates the diff stats behind `category == "state"`, so nothing reads a forcing
diff. They have the same shape and dtype, so substituting them raises nothing.

The statistics come from the **reanalysis** datastore. That is safe even though the graphs are
built on the *analysis* grid: the config
computes them with `dims: [grid_index, time]`, so they are reduced over space and time and
are 27 scalars per operation, not per-node fields. The grid the statistics were computed on
therefore cannot affect them.

## Channel order

`state_features.json` records the order the model was trained on:

```
zos, siconc, sithick,
thetao_{0,47,92,222,318,541}m,
so_{...}m, uo_{...}m, vo_{...}m
```

Keep it. A wrong order produces plausible-but-incorrect variables with no error anywhere --
the failure mode `VENDORED_BALTIC_SEA_FORECASTING.md` warns about.

## `ifs_forcing.zarr` — atmospheric forcing

A single-forecast slice of `analysis_ifs_forecasts_mdp_config.zarr`, covering the window
around Hurricane Milton.

| | |
| --- | --- |
| init | index 40 = 2024-10-08 (of 52 weekly inits) |
| leads | all 12, valid 2024-10-07 … 2024-10-18 |
| features | `sotemair`, `sowinu10`, `sowinv10`, `sosudosw`, `sosudolw`, `sowaprec`, `sod2m`, `somslpre` |
| size | 142 MB |

The store keeps the full declared shape `[8, 12, 52, 979200]`; only the inits that were
downloaded have chunks on disk. Zarr returns fill-value (NaN) for absent chunks rather than
raising, so **any other init reads as silently empty** -- `_present_inits` in `ocean.py`
therefore reads the chunk *filenames* rather than probing, since a probe pulls an 11.8 MB
chunk per init just to ask which exist.

An init spans only 12 days, so a longer window needs several: `prepare_OceanBench` gives each
timestep its own `(init, lead)` and, where weekly inits overlap by five days, takes the most
recent one. Only init 40 is committed; extend the window locally (see `.gitignore`) to
recompute a longer timeline.

Layout notes, since both bit us once:

- Chunks are `forcing/{feature}.{lead}.{init}.{grid}`. Feature and grid are single chunks,
  so those indices are always 0 -- one file is *all 8 variables for one day over the whole
  globe*, not a spatial tile.
- `lead_time` starts at **-1 day**, so lead 0 precedes the init: 12 leads = 2 input steps
  (`--input_steps 2`) plus a 10-day rollout. This matches the Njord paper
  (arXiv:2605.15470), which evaluates "52 forecasts initialized weekly (every Tuesday)
  throughout 2024" over "a 10-day forecast horizon". The `--ar_steps_eval 4` in
  `training_scripts/seacast/*.sh` is the shorter rollout used for validation *during*
  training, not the final evaluation.
- Zarr metadata lives in dotfiles (`.zarray`, `.zattrs`, `.zgroup`, `.zmetadata`). Google
  Drive strips the leading dot on download and Finder hides them by default -- transfer as a
  zip, or they go missing and the store will not open.

Verified on arrival: all 12 leads x 8 features finite, values physically plausible (winds
+/-20 m/s, precip to 16.5 mm/hr, MSLP down to 95,409 Pa), and **675,219 finite points of
979,200** -- matching the graph node count exactly, so forcing, state and graph share one
grid with no reindexing.

> **"Physically plausible" is necessary but not sufficient, and reading it as sufficient cost
> us a month of wrong numbers.** These values are in K, m/s, W/m^2 and Pa. The model was
> trained on them *standardised*, so MSLP ~ 1e5 passing a plausibility check is precisely the
> symptom: correct physics, four orders of magnitude outside the model's input distribution.
> `prepare_OceanBench` now standardises with `static/atmosphere_mean.pt` / `_std.pt` before
> windowing. Raw units scored 0.519 normalised RMSE against 0.245 for persistence -- twice as
> bad as predicting no change at all. Standardised and correctly interleaved: 0.175.

## Not here yet

- **Sample state window.** The 27 state features for the same dates, from the prepared
  0.25 degree datastore. Also `mask` and `static`, which are ~3 MB and ~27 MB, small enough
  that deriving them is not worth the risk of disagreeing with the upstream definitions.
- `scripts/download_ocean_window.py` verifies a store's land-sea mask against a graph before
  downloading, for whatever arrives.
- ~~**`grid_features.pt` / `grid_weights.pt`.**~~ **Not needed.** The 7 static features
  (`deptho`, `mdt`, `sin_lon`, `cos_lon`, `sin_lat`, `cos_lat`, `coast_dist`) are baked into
  the TorchScript export: `scripts/export_seacast.py` traces the instantiated model with
  `requires_grad_(False)`, so `model.grid_static_features` becomes a constant inside the
  graph. The scripted module's signature is `(prev, prev_prev, forcing, atmosphere)` -- no
  static input -- which is the proof it carries them internally.

  Kept as a note because the original concern still applies to any *re-export*: the static
  features must come from the **analysis** grid, and the `static` array in the reanalysis
  slice is `[7, 979200]` on the wrong grid, so it cannot be reused the way the statistics can.

## Grid alignment

Model output is one value per sea point and is paired with graph nodes positionally, so the
dataset's surface mask must select exactly as many points as the graph has grid nodes:

```
global_cluster_analysis        675,219
global_hierarchical_analysis   675,219
```

The reanalysis mask gives 676,736 -- a 1,517-node difference that would shift every value
after the first missing cell without raising anything. `download_ocean_window.py --graph`
asserts this before downloading.