"""
The set of forecast domains the backend can serve, and how a model is routed to one.

A "domain" is the pairing of a dataset with the loader that can run a model against it.
It exists because the two are not interchangeable at any layer:

  loading   ERA5 checkpoints are TorchScript modules taking (prev, prev_prev, forcing).
            The SeaCast export takes a fourth atmosphere argument and must be wrapped by
            `ocean._OceanModelAdapter` before anything else in the backend can call it.
            `inference.load_model` dispatches on file *extension* only, so it cannot tell
            the two apart -- handed seacast_scripted.pt it returns a bare ScriptModule that
            fails with "forward() is missing value for argument 'a'".
  data      ERA5 state comes from `preload_era5_tensors`, ocean state from
            `ocean.prepare_OceanBench`. Different variables, different grids, different
            timestamps.
  device    The ocean graph has 675,219 nodes against ERA5's 29,040. A single backward pass
            overruns Apple MPS's allocation cap, so ocean defaults to CPU.

Kept in its own module so `models`, `routes` and `services` can all name a domain without
importing each other.

NOT a graph-compatibility check. Models do not consume the graph selected in the UI -- it is
traced into the TorchScript at export. The graph pairing constraint is positional and lives
one level down: the model emits one value per grid node and the frontend pairs output[i] with
grid_lat_lon[i], so what must match is the node *count and order*. Two graphs in the same
domain can still be built on different grids (the analysis and reanalysis ocean masks differ
by 1,517 nodes), which a domain label would wave through. See ocean_datasets/README.md.
"""

from __future__ import annotations

DOMAIN_ERA5 = "era5"
DOMAIN_OCEAN = "ocean"

VALID_DOMAINS = (DOMAIN_ERA5, DOMAIN_OCEAN)

DEFAULT_DOMAIN = DOMAIN_ERA5


def normalize_domain(value: str | None) -> str:
    """
    Coerce a stored or user-supplied domain to a known one.

    None and the empty string mean "not recorded", which is every model row written before
    the column existed; those are ERA5, the only domain that worked then. An unrecognised
    non-empty value is a caller error and raises rather than silently defaulting -- routing a
    model to the wrong loader is the failure this module exists to prevent.
    """
    if value is None or value == "":
        return DEFAULT_DOMAIN
    v = str(value).strip().lower()
    if v not in VALID_DOMAINS:
        raise ValueError(
            f"Unknown model domain '{value}'. Valid domains: {', '.join(VALID_DOMAINS)}."
        )
    return v
