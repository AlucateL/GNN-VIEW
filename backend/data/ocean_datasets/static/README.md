# `ocean_datasets/static`

Normalisation constants for the ocean state, in the same shape and filenames as
`era5_datasets/static/` so the loading code carries over.

| file | contents |
| --- | --- |
| `parameter_mean.pt`, `parameter_std.pt` | 27 values each, one per state feature |
| `diff_mean.pt`, `diff_std.pt` | one-step difference statistics, 27 each |
| `state_features.json` | the channel order |

## Why these come from the reanalysis datastore

Extracted from `global_ocean_1_4_mdp_config.zarr` -- the **reanalysis** statistics datastore --
not from `analysis_ocean.zarr`, which is where the state data comes from. That is deliberate,
and it is what neural-lam does:

```yaml
# analysis_ocean.yaml
datastore:             analysis_ocean_mdp_config.yaml       # the data
statistics_datastore:  global_ocean_1_4_mdp_config.yaml     # the constants
```

The two datastores sit on slightly different grids -- the reanalysis surface mask has 676,736
sea points against the analysis grid's 675,219, a difference of 1,517 nodes that cost an
afternoon to track down. It does not matter here, because these statistics are **not
per-node**. The config computes them with `dims: [grid_index, time]`, reducing over space and
time, so each is 27 scalars. The grid they were computed on has been averaged away.

The same reasoning is why the analysis store's own `*__train__*` arrays were not downloaded:
they exist, but they are not the ones the model normalises with.

`__train__` in those names means the statistics cover only the training split
(2023-01-01 to 2023-12-31), never val or test -- computing them over the full range would leak
the evaluation period into the scaling.

## Channel order

`state_features.json` is authoritative, and was verified against the analysis store's own
`state_feature` coordinate:

```
zos, siconc, sithick,
thetao_{0,47,92,222,318,541}m, so_{...}m, uo_{...}m, vo_{...}m
```

Note the Njord paper's Table 1 lists these as `uo, vo, so, thetao` -- a different order from the
data. Build any channel mapping from this file, not from the paper, or temperature and current
channels swap with no error raised.

## Not per-node, unlike ERA5's

`era5_datasets/static/` also holds `grid_features.pt` and `grid_weights.pt`, which are
per-grid-node. The ocean equivalents are the 7 static fields (`deptho`, `mdt`, `sin_lon`,
`cos_lon`, `sin_lat`, `cos_lat`, `coast_dist`) and they live in `analysis_ocean.zarr/static`,
on the analysis grid. They are model inputs and are expected to be baked into the TorchScript
export, as they are for the weather models -- see `export_torchscript.py`, which reads
`model.grid_static_features.shape[1]` to derive the input dimension.
