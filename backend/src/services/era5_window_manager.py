from __future__ import annotations

import dataclasses
import logging
import os
import shutil
import threading
from typing import Any

import numpy as np
import xarray as xr

from services.domains import DOMAIN_ERA5, DOMAIN_OCEAN
from services.inference import (
    FIELDS_ZARR_PATH,
    _clear_era5_slice_caches,
    preload_era5_tensors,
)

logger = logging.getLogger(__name__)

# must match `_NEURAL_LAM_VARS` in `scripts/precompute_timeline.py`
NEURAL_LAM_VARS = [
    "geopotential", "specific_humidity", "temperature",
    "u_component_of_wind", "v_component_of_wind", "vertical_velocity",
    "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
    "mean_sea_level_pressure", "total_precipitation_6hr",
]

_GCS_URL = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
)
_CACHE_DIR = "/tmp/era5_cache"

_MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


class WindowAlreadyLoadingError(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class MonthBand:
    label: str
    start: str
    end: str
    count: int


@dataclasses.dataclass(frozen=True)
class _ActiveWindow:
    range_start: str
    range_end: str
    timestamps: tuple[str, ...]
    timestamps_set: frozenset[str]
    cache: dict[str, Any]


def _ts_to_iso(t) -> str:
    """Convert a numpy datetime64 of any unit to a seconds-precision ISO string."""
    return str(t.astype("datetime64[s]"))


def derive_month_bands(timestamps: list[str]) -> list[MonthBand]:
    """Group an ordered timestamp list into contiguous per-month bands."""
    groups: dict[tuple[int, int], list[str]] = {}
    order: list[tuple[int, int]] = []
    for ts in timestamps:
        year, month = int(ts[:4]), int(ts[5:7])
        key = (year, month)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(ts)
    return [
        MonthBand(
            label=f"{_MONTH_NAMES[m - 1]} {y}",
            start=groups[(y, m)][0],
            end=groups[(y, m)][-1],
            count=len(groups[(y, m)]),
        )
        for y, m in order
    ]


class _ERA5WindowManager:

    def __init__(self) -> None:
        # Keyed by domain. ERA5 is populated by initialize() at startup; ocean is built on
        # first use by ensure_domain(), because its cache is ~1.7 GB resident for the 12
        # shipped timesteps and most sessions never touch it.
        self._windows: dict[str, _ActiveWindow] = {}
        self._timeline_timestamps: dict[str, list[str]] = {}
        self._month_bands: dict[str, list[MonthBand]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, timeline: dict) -> None:
        """Called once from the FastAPI startup event. No GCS contact. ERA5 only."""
        try:
            self._timeline_timestamps[DOMAIN_ERA5] = [
                str(np.datetime64(t, "s")) for t in timeline["timestamps"]
            ]
            self._month_bands[DOMAIN_ERA5] = derive_month_bands(
                self._timeline_timestamps[DOMAIN_ERA5]
            )

            # Step 1: read seed zarr time coordinate (metadata only, milliseconds)
            seed_ds = xr.open_zarr(str(FIELDS_ZARR_PATH), consolidated=True)
            all_ts = [_ts_to_iso(t) for t in seed_ds.time.values]
            range_start = all_ts[0]
            range_end = all_ts[-1]

            # Step 2: preload ERA5 tensors from seed zarr
            cache = preload_era5_tensors(
                FIELDS_ZARR_PATH,
                time_start=range_start,
                n_steps=len(all_ts),
                lazy=True,
            )

            # Step 3: store active window; timestamps = time[2:] (prediction-valid only)
            pred_ts = tuple(all_ts[2:])
            self._windows[DOMAIN_ERA5] = _ActiveWindow(
                range_start=range_start,
                range_end=range_end,
                timestamps=pred_ts,
                timestamps_set=frozenset(pred_ts),
                cache=cache,
            )
        except Exception:
            logger.exception(
                "[ERA5WindowManager] initialization failed — dependent endpoints will return 503"
            )

    def ensure_domain(self, domain: str) -> None:
        """
        Make `domain` usable, building its window if this is the first request to need it.

        ERA5 is built eagerly at startup and this is a no-op for it. Ocean is built here
        rather than at startup: `prepare_OceanBench` is eager by design (the window is days,
        not an archive), which costs ~1.7 GB of resident tensors -- 143 MB per timestep, so
        roughly 20-30 steps is the practical ceiling before a container needs real headroom.
        Paying that only when an ocean model is actually selected keeps weather-only sessions
        at their current footprint.

        Raises on failure so the caller can turn it into a 503 naming the domain, rather than
        the request falling through to whatever cache happens to be loaded.
        """
        if domain in self._windows:
            return
        if domain != DOMAIN_OCEAN:
            raise RuntimeError(f"No builder for domain '{domain}'.")

        with self._lock:
            if domain in self._windows:   # another request built it while we waited
                return
            # Imported here, not at module scope: services.ocean pulls in the ocean stack,
            # and a weather-only deployment should not pay for it at import time.
            from services.ocean import OCEAN_DEVICE, prepare_OceanBench  # noqa: PLC0415

            # Must match the device the request path uses, or every ocean inference fails on a
            # device mismatch between these constants and the per-request tensors.
            cache = prepare_OceanBench(device=OCEAN_DEVICE)
            all_ts = [_ts_to_iso(t) for t in cache["time"]]
            # Predictions start at index 2 (the first two steps condition the model), but the
            # selectable set is narrower: only steps whose whole forcing window is genuine
            # forecast (lead >= 1). A step forced partly by analysis is not comparable to the
            # rest, and offering it would let a user scrub to a point the timeline does not
            # score -- and whose live value would differ from any timeline computed against a
            # window with more inits. Excluding it here keeps that mismatch unreachable
            # regardless of how much data a given deployment ships.
            pred_ts = tuple(_ts_to_iso(t) for t in cache["forecast_only_times"])
            if not pred_ts:
                raise RuntimeError(
                    f"Ocean window has {len(all_ts)} timesteps with data but none whose "
                    "forcing window is entirely forecast (lead >= 1). Needs at least 3 "
                    "consecutive steps covered by leads >= 1 -- download another IFS init."
                )
            # Unlike ERA5 -- whose timeline spans the full test set while only a window is
            # resident -- the ocean timeline is exactly the loaded window, so it is derived
            # from the cache rather than from a separate timeline file.
            self._timeline_timestamps[domain] = list(pred_ts)
            self._month_bands[domain] = derive_month_bands(list(pred_ts))
            self._windows[domain] = _ActiveWindow(
                range_start=all_ts[0],
                range_end=all_ts[-1],
                timestamps=pred_ts,
                timestamps_set=frozenset(pred_ts),
                cache=cache,
            )
            logger.info(
                "[WindowManager] built '%s' window: %d prediction timesteps (%s .. %s)",
                domain, len(pred_ts), pred_ts[0], pred_ts[-1],
            )

    # ------------------------------------------------------------------
    # Window loading
    # ------------------------------------------------------------------

    def load_window(self, start: str, end: str) -> dict:
        """
        Swap the resident ERA5 window. ERA5-only: it fetches from GCS, and the ocean window
        is whatever `prepare_OceanBench` found on disk -- there is nothing to page in.
        """
        acquired = self._lock.acquire(blocking=False)
        if not acquired:
            raise WindowAlreadyLoadingError("A window load is already in progress")
        try:
            # Short-circuit: requested window is already active
            current = self._windows.get(DOMAIN_ERA5)
            if current is not None and current.range_start == start and current.range_end == end:
                return {
                    "range_start": current.range_start,
                    "range_end": current.range_end,
                    "timestamps": list(current.timestamps),
                }

            # Validate start/end against the full timeline
            ts_list = self._timeline_timestamps.get(DOMAIN_ERA5, [])
            logger.debug(
                "[load_window] start=%r end=%r list_len=%d list_sample=%r",
                start, end, len(ts_list), ts_list[:3] if ts_list else [],
            )
            try:
                start_idx = ts_list.index(start)
            except ValueError:
                raise ValueError(f"start timestamp '{start}' not found in timeline (sample: {ts_list[:3]})")
            try:
                end_idx = ts_list.index(end)
            except ValueError:
                raise ValueError(f"end timestamp '{end}' not found in timeline")
            if start_idx > end_idx:
                raise ValueError(f"start '{start}' comes after end '{end}' in timeline")

            n_steps = end_idx - start_idx + 1

            # Build safe local zarr path (replace ':' with '-' for filesystem safety)
            safe_start = start.replace(":", "-")
            safe_end = end.replace(":", "-")
            local_path = os.path.join(_CACHE_DIR, f"{safe_start}_{safe_end}.zarr")
            os.makedirs(_CACHE_DIR, exist_ok=True)

            # Download from GCS only on first access; reuse local zarr on subsequent clicks.
            # Write to a temp path first so a failed download never leaves a poisoned cache entry.
            if not os.path.exists(local_path):
                tmp_path = local_path + ".tmp"
                if os.path.exists(tmp_path):
                    shutil.rmtree(tmp_path)
                try:
                    gcs_ds = xr.open_zarr(
                        _GCS_URL,
                        storage_options={"token": "anon"},
                        consolidated=True,
                    )
                    (
                        gcs_ds[NEURAL_LAM_VARS]
                        .sel(time=slice(start, end))
                        .to_zarr(tmp_path, mode="w", consolidated=True, zarr_format=2, safe_chunks=False)
                    )
                    os.rename(tmp_path, local_path)
                except Exception:
                    if os.path.exists(tmp_path):
                        shutil.rmtree(tmp_path)
                    raise

            # Preload tensors from local zarr
            cache = preload_era5_tensors(
                local_path,
                time_start=start,
                n_steps=n_steps,
                lazy=True,
            )

            # Re-open the local zarr to read the time coordinate (cache dict is opaque)
            local_ds = xr.open_zarr(local_path, consolidated=True)
            all_ts = [_ts_to_iso(t) for t in local_ds.time.values]
            pred_ts = tuple(all_ts[2:])

            # range_start/range_end verbatim from request so band-highlight comparison holds
            new_window = _ActiveWindow(
                range_start=start,
                range_end=end,
                timestamps=pred_ts,
                timestamps_set=frozenset(pred_ts),
                cache=cache,
            )
            # Single atomic dict assignment (GIL-safe; readers capture a frozen object)
            self._windows[DOMAIN_ERA5] = new_window
            # Guard: the idx-keyed ERA5 slice LRUs (S8) assume a stable window. If the
            # window ever actually changes at runtime, a stale idx would otherwise
            # silently serve data from the old window.
            _clear_era5_slice_caches()

            return {
                "range_start": start,
                "range_end": end,
                "timestamps": list(pred_ts),
            }
        finally:
            self._lock.release()

    # ------------------------------------------------------------------
    # Read accessors (lock-free; safe because _window is a frozen object)
    # ------------------------------------------------------------------

    def get_current_cache(self, domain: str = DOMAIN_ERA5) -> dict[str, Any]:
        return self.get_current_window(domain).cache

    def get_current_window(self, domain: str = DOMAIN_ERA5) -> _ActiveWindow:
        window = self._windows.get(domain)
        if window is None:
            raise RuntimeError(f"WindowManager has no '{domain}' window loaded")
        return window

    def is_in_window(self, pred_time: str, domain: str = DOMAIN_ERA5) -> bool:
        window = self._windows.get(domain)
        return window is not None and pred_time in window.timestamps_set

    def is_initialized(self, domain: str = DOMAIN_ERA5) -> bool:
        """
        Whether `domain` has a window ready to serve.

        Ocean is built lazily, so a False here is not necessarily a failure -- call
        ensure_domain() first. The default keeps every existing ERA5 call site unchanged.
        """
        return domain in self._windows

    def get_all_cached_timestamps(self) -> list[str]:
        """Scan _CACHE_DIR for all local zarr files and return every prediction-valid timestamp."""
        if not os.path.exists(_CACHE_DIR):
            return []
        all_ts: set[str] = set()
        for entry in os.scandir(_CACHE_DIR):
            if not entry.is_dir() or not entry.name.endswith(".zarr"):
                continue
            try:
                ds = xr.open_zarr(entry.path, consolidated=True)
                ts = [_ts_to_iso(t) for t in ds.time.values]
                for t in ts[2:]:  # skip the 2 context timesteps, same as load_window
                    all_ts.add(t)
            except Exception:
                logger.warning("[get_all_cached_timestamps] skipping %s", entry.path)
        return sorted(all_ts)

    @property
    def month_bands(self) -> list[MonthBand]:
        """ERA5 month bands. Kept for callers that predate multi-domain support."""
        return self.month_bands_for(DOMAIN_ERA5)

    def month_bands_for(self, domain: str) -> list[MonthBand]:
        """
        Month bands for one domain's timeline.

        Returns [] for a domain with no bands rather than raising: bands are decoration, and
        a missing one must not take down the timeline endpoint. Note the ocean window is ~10
        daily steps, so its bands collapse to a single month -- whether that is the right
        presentation for a short series is still open, but serving ERA5's 2020 bands against
        ocean timestamps would be wrong regardless.
        """
        return self._month_bands.get(domain, [])


manager = _ERA5WindowManager()
