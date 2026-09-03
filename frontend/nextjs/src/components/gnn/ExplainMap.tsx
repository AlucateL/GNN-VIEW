"use client";

import React, { useEffect, useRef } from "react";
import { PolygonLayer } from "@deck.gl/layers";
import {
  HEATMAP_STOPS, HEATMAP_NO_DATA,
  PREDICTION_STOPS, PREDICTION_NO_DATA,
  DIVERGING_WIND_STOPS, DIVERGING_WIND_NO_DATA,
  THERMAL_FIELD_STOPS, THERMAL_FIELD_NO_DATA,
  DENSE_FIELD_STOPS, DENSE_FIELD_NO_DATA,
  HALINE_FIELD_STOPS, HALINE_FIELD_NO_DATA,
  RAIN_FIELD_STOPS, RAIN_FIELD_NO_DATA,
  ICE_FIELD_STOPS, ICE_FIELD_NO_DATA,
} from '../../lib/colors';
import { LRUCache } from '../../lib/lru';

// ── Types ─────────────────────────────────────────────────────────────────────

type GridData = { lat: number[]; lon: number[]; grid_spacing_deg?: number };

type ScaleRanges = {
  single_component: number;
  variable_total: number;
  total_by_level: number;
  global_total: number;
};

type HeatmapCell = {
  polygon: [number, number][];
  nodeIndex: number;
};

type NormalizationStats = {
  mean: number[];
  std: number[];
  physical_scales: Record<string, [number, number]>;
};

// `/prediction_errors` response.
//
// `pred` and `component_idx` are absent unless the request asked for the field, which only
// happens in Prediction display mode (see buildFieldUrl). Even then they come back null for
// aggregated selections (variable=total or z_level=total), where a per-node "average
// prediction" has no physical meaning. Undefined and null are equivalent to every reader
// here — both mean "no prediction slice to draw" — which is why the guards below test
// `== null` rather than `=== null`.
//
// The backend no longer bundles the matching `truth` slice: nothing on this side ever read
// it, and it was a third of the payload on the hot path. `/prediction_field` still serves it
// for callers that want to recompute |pred - truth| themselves.
type PredictionErrorsWithField = {
  /** A null entry means the backend has no data for that node in the selected channel —
   *  the water is shallower than the channel's depth, so there is nothing to score. Distinct
   *  from 0, which is a real (perfect) prediction. JSON null is how orjson writes NaN. */
  errors: (number | null)[];
  /** Same null convention as `errors`: no data at that node for this channel. */
  pred?: (number | null)[] | null;
  component_idx?: number | null;
};

// Surface variables — must mirror SURF_VARS in ExplainRightPanel / GNNExplorer.
// Used to drop z_level from URLs in prediction mode.
const SURF_VARS = [
  "2m_temperature",
  "10m_u_component_of_wind",
  "10m_v_component_of_wind",
  "mean_sea_level_pressure",
  "total_precipitation_6hr",
] as const;

/**
 * Half-width of a heatmap cell, in degrees.
 *
 * Derived from the grid the backend reports rather than fixed: a cell must cover its own
 * grid box, and the grids differ by 6x (ERA5 1.5°, ocean 0.25°). A constant tuned for ERA5
 * makes ocean cells overlap six deep, painting a smear instead of a field.
 *
 * Falls back to ERA5's step when the backend omits the spacing, which keeps older backends
 * rendering exactly as before.
 */
const ERA5_SPACING_DEG = 1.5;
const cellHalfDeg = (grid: GridData | null): number =>
  ((grid?.grid_spacing_deg ?? ERA5_SPACING_DEG) / 2);

// ── Module-level render state ─────────────────────────────────────────────────
// Survives re-renders; read by layer builders called from GNNExplorer.

// Cell geometry — computed once per grid load (polar cells |lat|>85.05 excluded)
let cellGeometry: HeatmapCell[] | null = null;
let nodeIndexToCell: Map<number, HeatmapCell> | null = null;
let lastGridRef: GridData | null = null;

// Render state — mutated by async fetch callbacks
let currentErrors: (number | null)[] | null = null;
let currentAlpha = 1.0;
let currentScaleMax = 0;
let currentScaleMin = 0;
let currentLogScale = false;
// Which kind of data `currentErrors` currently holds — NOT which mode the UI is in.
// The two diverge while a mode switch is in flight: Error mode does not fetch `pred`, so
// toggling to Prediction has to go back to the wire, and until it lands the array in hand is
// still normalized errors. Colouring those with the prediction ramp (a physical range like
// 219-281 K) clamps every cell to the bottom of the scale and flashes a uniformly cold map.
// Painting by the data's own mode keeps the old error map on screen, dimmed by currentAlpha,
// which is what every other in-flight fetch already does.
let currentDataMode: "error" | "prediction" = "error";
let currentV1ForColor = "";
let heatmapEverLoadedFlag = false;
let cachedScaleRanges: ScaleRanges | null = null;

// Recently fetched /prediction_errors responses, keyed on `${t}|${v1}|${z1}`
// (see fieldCacheKey). An LRU (not a single slot) so flipping between several
// timestamps — the core Explain comparison workflow — stays instant after each has
// been visited once, instead of re-hitting the backend.
//
// An entry does not always serve both display modes: one fetched in Error mode has no
// `pred` slice, so a switch to Prediction is a cache miss for that key and refetches
// with the field. The refetch overwrites the entry, so the upgrade happens once per key
// and every later toggle at that key is free. See servesMode.
const FIELD_CACHE_CAPACITY = 16;
const fieldCache = new LRUCache<PredictionErrorsWithField>(FIELD_CACHE_CAPACITY);

// A cached entry can always paint the error map — `errors` is in every response. Painting
// the prediction map additionally needs `pred`, which is only present when the entry was
// fetched in Prediction mode.
const servesMode = (
  entry: PredictionErrorsWithField,
  mode: "error" | "prediction",
): boolean => mode === "error" || entry.pred != null;

// Drops all cached field responses. The cache key is (t, v1, z1) — it does NOT
// encode the active model, so the parent must call this on model switch (the
// only event that changes what the same key maps to) to avoid serving stale
// data for the newly-selected model.
export function clearFieldCache(): void {
  fieldCache.clear();
}

// ── Colormap helpers ──────────────────────────────────────────────────────────

function heatmapColor(
  value: number,
  max: number,
  alpha: number,
  logScale: boolean = false,
): [number, number, number, number] {
  if (max <= 0) return [...HEATMAP_NO_DATA, Math.round(180 * alpha)] as [number, number, number, number];
  const t = logScale
    ? Math.max(0, Math.min(1, Math.log1p(value) / Math.log1p(max)))
    : Math.max(0, Math.min(1, value / max));
  let r = HEATMAP_STOPS[HEATMAP_STOPS.length - 1][1];
  let g = HEATMAP_STOPS[HEATMAP_STOPS.length - 1][2];
  let b = HEATMAP_STOPS[HEATMAP_STOPS.length - 1][3];
  for (let i = 0; i < HEATMAP_STOPS.length - 1; i++) {
    const [t0, r0, g0, b0] = HEATMAP_STOPS[i];
    const [t1, r1, g1, b1] = HEATMAP_STOPS[i + 1];
    if (t <= t1) {
      const f = (t - t0) / (t1 - t0);
      r = Math.round(r0 + f * (r1 - r0));
      g = Math.round(g0 + f * (g1 - g0));
      b = Math.round(b0 + f * (b1 - b0));
      break;
    }
  }
  return [r, g, b, Math.round(210 * alpha)];
}

// Variables whose raw field is signed and needs a diverging colormap centered on 0.
const DIVERGING_PRED_VARS = new Set([
  "u_component_of_wind",
  "v_component_of_wind",
  "vertical_velocity",
  "10m_u_component_of_wind",
  "10m_v_component_of_wind",
]);

// Ocean anomaly/velocity channels — diverging around zero, same treatment as wind
// components. Must mirror isOceanDivergingVar in ExplainRightPanel.
function isOceanDivergingVar(v1: string): boolean {
  return v1 === "zos" || v1.startsWith("uo_") || v1.startsWith("vo_");
}

function interpolateStops(
  t: number,
  stops: [number, number, number, number][],
): [number, number, number] {
  let r = stops[stops.length - 1][1];
  let g = stops[stops.length - 1][2];
  let b = stops[stops.length - 1][3];
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, r0, g0, b0] = stops[i];
    const [t1, r1, g1, b1] = stops[i + 1];
    if (t <= t1) {
      const f = (t - t0) / (t1 - t0);
      r = Math.round(r0 + f * (r1 - r0));
      g = Math.round(g0 + f * (g1 - g0));
      b = Math.round(b0 + f * (b1 - b0));
      break;
    }
  }
  return [r, g, b];
}

/** No-data tint for a variable's colormap family, used where the backend sent null: the water
 *  is shallower than the selected channel's depth, so there is no value to colour. All families
 *  currently resolve to the same grey in colors.json, but they are kept separate so a palette
 *  can differentiate them without touching this dispatch. */
function noDataColorForVar(v1: string, alpha: number): [number, number, number, number] {
  const aND = Math.round(180 * alpha);
  if (DIVERGING_PRED_VARS.has(v1) || isOceanDivergingVar(v1)) {
    return [...DIVERGING_WIND_NO_DATA, aND] as [number, number, number, number];
  }
  if (v1 === "total_precipitation_6hr") {
    return [...RAIN_FIELD_NO_DATA, aND] as [number, number, number, number];
  }
  if (v1 === "temperature" || v1 === "2m_temperature" || v1.startsWith("thetao_")) {
    return [...THERMAL_FIELD_NO_DATA, aND] as [number, number, number, number];
  }
  if (v1 === "specific_humidity" || v1.startsWith("so_")) {
    return [...HALINE_FIELD_NO_DATA, aND] as [number, number, number, number];
  }
  if (v1 === "geopotential" || v1 === "mean_sea_level_pressure") {
    return [...DENSE_FIELD_NO_DATA, aND] as [number, number, number, number];
  }
  if (v1 === "siconc" || v1 === "sithick") {
    return [...ICE_FIELD_NO_DATA, aND] as [number, number, number, number];
  }
  return [...PREDICTION_NO_DATA, aND] as [number, number, number, number];
}

// Per-variable prediction colormap dispatcher.
// Diverging vars: symmetric [-A, +A] with A = max(|min|, |max|); neutral tint at exactly 0.
// Precipitation: sqrt scale, zero → transparent.
// All others: sequential [min, max] with the appropriate perceptually-uniform family.
function predictionColorForVar(
  value: number,
  v1: string,
  min: number,
  max: number,
  alpha: number,
): [number, number, number, number] {
  const a210 = Math.round(210 * alpha);
  const aND  = Math.round(180 * alpha);

  if (DIVERGING_PRED_VARS.has(v1) || isOceanDivergingVar(v1)) {
    const A = Math.max(Math.abs(min), Math.abs(max));
    if (A <= 0) return [...DIVERGING_WIND_NO_DATA, aND] as [number, number, number, number];
    const t = 0.5 + 0.5 * Math.max(-1, Math.min(1, value / A));
    return [...interpolateStops(t, DIVERGING_WIND_STOPS), a210] as [number, number, number, number];
  }

  if (v1 === "total_precipitation_6hr") {
    if (value <= 0) return [255, 255, 255, 0];
    const sqrtMax = Math.sqrt(Math.max(0, max));
    if (sqrtMax <= 0) return [...RAIN_FIELD_NO_DATA, aND] as [number, number, number, number];
    const t = Math.min(1, Math.sqrt(value) / sqrtMax);
    return [...interpolateStops(t, RAIN_FIELD_STOPS), a210] as [number, number, number, number];
  }

  const span = max - min;
  if (span <= 0) return [...PREDICTION_NO_DATA, aND] as [number, number, number, number];
  const t = Math.max(0, Math.min(1, (value - min) / span));

  if (v1 === "temperature" || v1 === "2m_temperature" || v1.startsWith("thetao_")) {
    return [...interpolateStops(t, THERMAL_FIELD_STOPS), a210] as [number, number, number, number];
  }
  if (v1 === "specific_humidity" || v1.startsWith("so_")) {
    return [...interpolateStops(t, HALINE_FIELD_STOPS), a210] as [number, number, number, number];
  }
  // geopotential, mean_sea_level_pressure, and any future sequential variable
  if (v1 === "geopotential" || v1 === "mean_sea_level_pressure") {
    return [...interpolateStops(t, DENSE_FIELD_STOPS), a210] as [number, number, number, number];
  }
  if (v1 === "siconc" || v1 === "sithick") {
    return [...interpolateStops(t, ICE_FIELD_STOPS), a210] as [number, number, number, number];
  }

  // Fallback: legacy sequential cool→warm for anything not yet classified
  let r = PREDICTION_STOPS[PREDICTION_STOPS.length - 1][1];
  let g = PREDICTION_STOPS[PREDICTION_STOPS.length - 1][2];
  let b = PREDICTION_STOPS[PREDICTION_STOPS.length - 1][3];
  for (let i = 0; i < PREDICTION_STOPS.length - 1; i++) {
    const [t0, r0, g0, b0] = PREDICTION_STOPS[i];
    const [t1, r1, g1, b1] = PREDICTION_STOPS[i + 1];
    if (t <= t1) {
      const f = (t - t0) / (t1 - t0);
      r = Math.round(r0 + f * (r1 - r0));
      g = Math.round(g0 + f * (g1 - g0));
      b = Math.round(b0 + f * (b1 - b0));
      break;
    }
  }
  return [r, g, b, a210];
}

function selectScaleMax(ranges: ScaleRanges, v1: string, z1: string | null): number {
  if (v1 === "total" && z1 === "sum") return ranges.global_total;
  if (v1 === "total") return ranges.total_by_level;
  if (z1 === "sum") return ranges.variable_total;
  return ranges.single_component;
}

// `with_field` is requested only in Prediction display mode.
//
// It roughly doubles the response — `pred` is a second full-length array — and Error mode,
// the default and where users spend most of their time, never reads it. Asking for it
// unconditionally made every variable and level switch pay for a slice that was usually
// discarded: ~450 ms per switch in production, against ~150 ms for errors alone. The cost of
// gating is one refetch the first time the user toggles to Prediction at a given (t, v1, z1);
// the field cache absorbs every toggle after that.
function buildFieldUrl(
  apiBase: string,
  modelId: string,
  v1: string,
  z1: string | null,
  t: string,
  wantField: boolean,
): string {
  const url = new URL(`${apiBase}/inference/${encodeURIComponent(modelId)}/prediction_errors`);
  url.searchParams.set("pred_time", t);
  url.searchParams.set("variable", v1);
  if (z1 === "sum") {
    url.searchParams.set("z_level", "total");
  } else if (z1 !== null) {
    url.searchParams.set("z_level", z1);
  }
  if (wantField) url.searchParams.set("with_field", "true");
  return url.toString();
}

const fieldCacheKey = (t: string, v1: string, z1: string | null): string =>
  `${t}|${v1}|${z1 ?? ""}`;

// `/prediction_errors` has two wire shapes, and which one you get depends on `with_field`:
// with it, an object; without it, a bare `list[float]` — the original pre-with_field response,
// kept for backwards compatibility and asserted by test_prediction_field.py. Error mode now
// takes the second path, so normalise here rather than at three call sites. Missing this is
// what blanked the heatmap: `resp.errors` on a bare array is undefined, so currentErrors never
// got set and buildHeatmapLayer returned null.
const normalizeFieldResponse = (
  data: PredictionErrorsWithField | (number | null)[],
): PredictionErrorsWithField => (Array.isArray(data) ? { errors: data } : data);

// Translate a response into the module-level render state used by buildHeatmapLayer.
// Called from the fetch effect on both the cache-hit and the network path.
function applyFieldResponse(
  resp: PredictionErrorsWithField,
  v1: string,
  z1: string | null,
  mode: "error" | "prediction",
  normStats: NormalizationStats | null,
  onColorbarUpdate?: (errors: (number | null)[] | null, max: number, phase: "skeleton" | "ready", min?: number) => void,
): void {
  if (mode === "prediction") {
    // Prediction display needs a single-component case + normalization stats. If
    // either is missing, fall back to nothing (parent UI prevents this by hiding
    // the toggle/aggregated buttons; this branch is a guard, not a code path users
    // can reach).
    if (resp.pred == null || resp.component_idx == null || !normStats) {
      currentErrors = null;
      currentScaleMin = 0;
      currentScaleMax = 0;
      heatmapEverLoadedFlag = false;
      onColorbarUpdate?.(null, 0, "skeleton", 0);
      return;
    }
    // Surface vars and ocean state channels (which carry no pressure level — z1 is
    // always null for them) key physical_scales by the bare variable name.
    const isSurface = (SURF_VARS as readonly string[]).includes(v1) || z1 === null;
    const key = isSurface ? v1 : `${v1}@${z1}`;
    const scale = normStats.physical_scales[key];
    if (!scale) {
      console.warn("[ExplainMap] no physical scale for", key);
      return;
    }
    const c = resp.component_idx;
    const m = normStats.mean[c];
    const s = normStats.std[c];
    const physical = new Array<number>(resp.pred.length);
    for (let i = 0; i < resp.pred.length; i++) {
      const v = resp.pred[i];
      // null must not fall through the arithmetic: it coerces to 0, and 0 in normalized space
      // denormalizes to exactly the training mean — the very value this masking exists to hide.
      physical[i] = v === null ? NaN : v * s + m;
    }
    currentErrors = physical;
    currentScaleMin = scale[0];
    currentScaleMax = scale[1];
    currentV1ForColor = v1;
    currentAlpha = 1.0;
    currentDataMode = "prediction";
    heatmapEverLoadedFlag = true;
    onColorbarUpdate?.(physical, scale[1], "ready", scale[0]);
  } else {
    // Error display: use the errors array directly. scaleMax comes from cached error
    // scale ranges (range fetch lives on its own effect); we forward what we have and
    // let the GNNExplorer's per-variable vs shared logic pick the final display max.
    currentErrors = resp.errors;
    currentScaleMin = 0;
    currentV1ForColor = v1;
    currentDataMode = "error";
    const rangeMax = cachedScaleRanges ? selectScaleMax(cachedScaleRanges, v1, z1) : currentScaleMax;
    currentScaleMax = rangeMax;
    currentAlpha = 1.0;
    heatmapEverLoadedFlag = true;
    onColorbarUpdate?.(resp.errors, rangeMax, cachedScaleRanges ? "ready" : "skeleton", 0);
  }
}

// ── Layer builders — exported so GNNExplorer can compose them into its layer stack ──
// Note: the deck canvas has pointer-events:none, so all mouse events go through
// Leaflet → deckInstance.pickObject() → manual dispatch in GNNExplorer's map handlers.
// Layer-level onClick/onHover are intentionally absent here.

export function buildHeatmapLayer(): PolygonLayer<HeatmapCell> | null {
  console.log("[ExplainMap] buildHeatmapLayer:", { cells: cellGeometry?.length ?? null, errors: currentErrors?.length ?? null, everLoaded: heatmapEverLoadedFlag, scaleMax: currentScaleMax, mode: currentDataMode });
  if (!cellGeometry || !currentErrors || !heatmapEverLoadedFlag) return null;
  // The grid and the error array arrive from separate requests, so they can disagree: when
  // /prediction_errors fails (an ocean forward pass OOM-killing the backend, say) after
  // /grid has already swapped in a new grid, the previous model's errors are still in hand.
  // Drawing them indexes the first errors.length nodes and leaves the rest undefined —
  // on the ocean grid that paints a convincing 4% band at the date line and nothing else.
  // Render nothing instead: an empty map is obviously broken, a partial one is not.
  if (currentErrors.length !== lastGridRef?.lat.length) {
    console.warn(
      `[ExplainMap] errors/grid mismatch: ${currentErrors.length} values for ` +
      `${lastGridRef?.lat.length ?? 0} nodes — skipping render until they agree.`
    );
    return null;
  }
  const errSnap = currentErrors;
  const maxSnap = currentScaleMax;
  const minSnap = currentScaleMin;
  const alphaSnap = currentAlpha;
  const logSnap = currentLogScale;
  const modeSnap = currentDataMode;
  const v1Snap = currentV1ForColor;
  return new PolygonLayer<HeatmapCell>({
    id: "explain-heatmap",
    data: cellGeometry,
    getPolygon: (d) => d.polygon,
    getFillColor: (d) => {
      const v = errSnap[d.nodeIndex];
      // null (JSON for the backend's NaN) means there is no data at this node for the selected
      // channel — below the seafloor at that depth. Coercing it to 0, as this did, painted
      // entire shelf seas as perfectly predicted.
      if (v === null || v === undefined || !Number.isFinite(v)) {
        return modeSnap === "prediction"
          ? noDataColorForVar(v1Snap, alphaSnap)
          : ([...HEATMAP_NO_DATA, Math.round(180 * alphaSnap)] as [number, number, number, number]);
      }
      return modeSnap === "prediction"
        ? predictionColorForVar(v, v1Snap, minSnap, maxSnap, alphaSnap)
        : heatmapColor(v, maxSnap, alphaSnap, logSnap);
    },
    getLineWidth: 0,
    lineWidthMinPixels: 0,
    filled: true,
    stroked: false,
    pickable: true,
    wrapLongitude: true,
    updateTriggers: {
      getFillColor: [errSnap, maxSnap, minSnap, alphaSnap, logSnap, modeSnap, v1Snap],
    },
  });
}

// ── Selection ring — thick black PolygonLayer stroke over the selected cell ───

export function buildSelectionRingLayer(
  selectedNodeIndex: number | null,
): PolygonLayer<HeatmapCell> | null {
  if (selectedNodeIndex === null || !nodeIndexToCell) return null;
  const cell = nodeIndexToCell.get(selectedNodeIndex);
  if (!cell) return null;
  return new PolygonLayer<HeatmapCell>({
    id: "explain-selection-ring",
    data: [cell],
    getPolygon: (d) => d.polygon,
    getFillColor: [0, 0, 0, 0],
    getLineColor: [0, 0, 0, 255],
    getLineWidth: 3,
    lineWidthUnits: "pixels",
    lineWidthMinPixels: 3,
    filled: false,
    stroked: true,
    pickable: false,
    wrapLongitude: true,
    updateTriggers: { data: selectedNodeIndex },
  });
}

// ── Hover ring — thin gray PolygonLayer stroke over the hovered cell ──────────

export function buildHoverRingLayer(
  hoveredNodeIndex: number | null,
): PolygonLayer<HeatmapCell> | null {
  if (hoveredNodeIndex === null || !nodeIndexToCell) return null;
  const cell = nodeIndexToCell.get(hoveredNodeIndex);
  if (!cell) return null;
  return new PolygonLayer<HeatmapCell>({
    id: "explain-hover-ring",
    data: [cell],
    getPolygon: (d) => d.polygon,
    getFillColor: [0, 0, 0, 0],
    getLineColor: [107, 114, 128, 180],
    getLineWidth: 1,
    lineWidthUnits: "pixels",
    lineWidthMinPixels: 1,
    filled: false,
    stroked: true,
    pickable: false,
    wrapLongitude: true,
    updateTriggers: { data: hoveredNodeIndex },
  });
}

// ── Props ─────────────────────────────────────────────────────────────────────

export type ExplainMapProps = {
  v1: string;
  z1: string | null;
  t: string;
  grid: GridData | null;
  apiBaseUrl: string;
  /** DB model_id. Must be in every URL: the un-prefixed routes fall back to the active
   *  ERA5 preset, so an ocean timestamp would be rejected against the ERA5 window. */
  modelId: string;
  /** Timestamps the active model's window can serve, from /loaded_window. Fetches are held
   *  until `t` appears here, because `modelId` and `t` are published by different code paths
   *  (refreshExplainModelOptions vs loadExplainModelRuntime) and are briefly mismatched on
   *  mount — the default model id is the ERA5 preset, so an already-restored ocean timestamp
   *  gets scored against the ERA5 window and 400s. Null means the window is not known yet. */
  availableTimestamps: Set<string> | null;
  refetchTrigger?: number;
  logScale?: boolean;
  colorbarMax?: number;
  displayMode?: "error" | "prediction";
  normStats?: NormalizationStats | null;
  onRebuildLayers: () => void;
  // Fires whenever the colorbar data changes — parent renders the colorbar in the right panel.
  // For prediction mode, `min` is the lower physical bound of the colorbar; null/undefined in error mode.
  onColorbarUpdate?: (errors: (number | null)[] | null, max: number, phase: "skeleton" | "ready", min?: number) => void;
};

// ── Component — pure side-effects: fetches heatmap data and notifies parent ──
// Colorbar is rendered by ExplainLeftPanel (receives data via onColorbarUpdate).

export default function ExplainMap({
  v1,
  z1,
  t,
  grid,
  apiBaseUrl,
  modelId,
  availableTimestamps,
  refetchTrigger,
  logScale,
  colorbarMax,
  displayMode = "error",
  normStats,
  onRebuildLayers,
  onColorbarUpdate,
}: ExplainMapProps): React.ReactElement {
  // Refs for async callbacks — always reflect the latest prop values
  const currentV1Ref = useRef(v1);
  const currentZ1Ref = useRef(z1);
  const displayModeRef = useRef(displayMode);
  const normStatsRef = useRef(normStats);
  currentV1Ref.current = v1;
  currentZ1Ref.current = z1;
  displayModeRef.current = displayMode;
  normStatsRef.current = normStats;

  // No separate display-mode effect: displayMode is a dependency of the heatmap fetch
  // effect below, which re-runs on toggle and either re-applies from cache or refetches.
  // applyFieldResponse publishes currentDataMode, so the layer builders follow the data.

  const heatmapAbortRef = useRef<AbortController | null>(null);
  const rangeAbortRef = useRef<AbortController | null>(null);

  // ── Build cell geometry when grid changes (polar cells |lat|>85.05 filtered) ─
  useEffect(() => {
    if (!grid) { cellGeometry = null; nodeIndexToCell = null; lastGridRef = null; return; }
    if (grid === lastGridRef) return;
    lastGridRef = grid;
    const cells: HeatmapCell[] = [];
    const lookup = new Map<number, HeatmapCell>();
    // Sized from the grid the backend reports, not a constant: ERA5 is 1.5 deg and the
    // ocean grid 0.25 deg, so a fixed half-width draws ocean cells six times too wide.
    const half = cellHalfDeg(grid);
    for (let i = 0; i < grid.lat.length; i++) {
      const lat = grid.lat[i];
      if (Math.abs(lat) > 85.05) continue;
      const latS = Math.max(-90, lat - half);
      const latN = Math.min( 90, lat + half);
      // No antimeridian clamp. Cells on lon -180 do reach -180.125, but wrapLongitude makes
      // deck.gl cut them against the mercator bounds and draw the remainder on the +180
      // side. Clamping instead DELETES that half, leaving a visible seam down the date line
      // — half a cell wide, so 0.75 deg on ERA5 and 0.125 deg on the ocean grid.
      const cell: HeatmapCell = {
        polygon: [
          [grid.lon[i] - half, latS],
          [grid.lon[i] + half, latS],
          [grid.lon[i] + half, latN],
          [grid.lon[i] - half, latN],
        ] as [number, number][],
        nodeIndex: i,
      };
      cells.push(cell);
      lookup.set(i, cell);
    }
    cellGeometry = cells;
    nodeIndexToCell = lookup;
    onRebuildLayers();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [grid]);

  // ── Log scale toggle — updates module-level flag and triggers layer rebuild ─
  useEffect(() => {
    currentLogScale = logScale ?? false;
    onRebuildLayers();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [logScale]);

  // ── Colorbar max update — updates scale and triggers layer rebuild ─────────
  useEffect(() => {
    if (colorbarMax !== undefined && colorbarMax > 0) {
      currentScaleMax = colorbarMax;
      onRebuildLayers();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [colorbarMax]);

  // ── Scale range fetch — fires on timestep change only ─────────────────────
  useEffect(() => {
    if (!t || !modelId) return;
    // Hold until (modelId, t) is a matched pair — see availableTimestamps in the props.
    if (!availableTimestamps?.has(t)) return;

    const ctrl = new AbortController();
    rangeAbortRef.current?.abort();
    rangeAbortRef.current = ctrl;

    fetch(
      `${apiBaseUrl}/inference/${encodeURIComponent(modelId)}/prediction_errors/range?pred_time=${encodeURIComponent(t)}`,
      { signal: ctrl.signal },
    )
      .then((r) => r.ok ? r.json() : r.json().then((b: unknown) => Promise.reject(`HTTP ${r.status}: ${JSON.stringify(b)}`)))
      .then((data: ScaleRanges) => {
        if (ctrl.signal.aborted) return;
        cachedScaleRanges = data;
        const max = selectScaleMax(data, currentV1Ref.current, currentZ1Ref.current);
        currentScaleMax = max;
        // Don't rebuild here — the heatmap fetch will rebuild when its data arrives.
        // Just notify GNNExplorer so it can apply the shared-colorbar toggle and feed
        // the correct display scale back through the colorbarMax prop.
        onColorbarUpdate?.(currentErrors, max, "ready");
      })
      .catch((err) => { if (!ctrl.signal.aborted) console.error("[ExplainMap] range fetch failed:", err); });

    return () => ctrl.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [t, apiBaseUrl, modelId, availableTimestamps, refetchTrigger]);

  // ── Recompute scale max from cached ranges when V1/z1 change ──────────────
  // Skipped in prediction mode — physical scale comes from normStats, not error ranges.
  useEffect(() => {
    if (displayMode === "prediction") return;
    if (!cachedScaleRanges) return;
    const max = selectScaleMax(cachedScaleRanges, v1, z1);
    currentScaleMax = max;
    // Don't rebuild here — the heatmap fetch will rebuild with the new data.
    onColorbarUpdate?.(currentErrors, max, "ready", 0);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [v1, z1, displayMode]);

  // ── Heatmap data fetch — fires when V1, z1, t, grid or displayMode changes ─
  // displayMode is a dependency because the response shape depends on it: Error mode asks
  // for errors alone, Prediction mode additionally for `pred`. This effect owns applying
  // the data for both the cache-hit and the network path.
  useEffect(() => {
    if (!t || !grid || !modelId) return;
    // Same matched-pair gate as the range fetch above. Checked before the cache lookup so a
    // mismatched pair cannot seed the cache under a key the other model would then read.
    if (!availableTimestamps?.has(t)) return;

    const key = fieldCacheKey(t, v1, z1);

    // Cache hit — (t, v1, z1) already in memory *and* the entry carries what this mode
    // needs. Re-apply and rebuild layers without a network round-trip. This is the path that
    // makes flipping between visited timestamps instant, and toggling Error/Prediction
    // instant once the key has been fetched in Prediction mode at least once.
    const cached = fieldCache.get(key);
    if (cached && servesMode(cached, displayMode)) {
      applyFieldResponse(cached, v1, z1, displayModeRef.current, normStatsRef.current ?? null, onColorbarUpdate);
      onRebuildLayers();
      return;
    }

    const ctrl = new AbortController();
    heatmapAbortRef.current?.abort();
    heatmapAbortRef.current = ctrl;

    if (heatmapEverLoadedFlag) {
      currentAlpha = 0.6;
      onRebuildLayers();
    }

    const url = buildFieldUrl(apiBaseUrl, modelId, v1, z1, t, displayMode === "prediction");
    console.log("[ExplainMap] field fetch →", url);
    onColorbarUpdate?.(currentErrors, currentScaleMax, "skeleton", currentScaleMin);

    fetch(url, { signal: ctrl.signal })
      .then((r) => r.ok ? r.json() : r.json().then((b: unknown) => Promise.reject(`HTTP ${r.status}: ${JSON.stringify(b)}`)))
      .then((raw: PredictionErrorsWithField | (number | null)[]) => {
        if (ctrl.signal.aborted) return;
        const data = normalizeFieldResponse(raw);
        // Overwrites any errors-only entry under this key, so the Error -> Prediction
        // upgrade costs one fetch per key rather than one per toggle.
        fieldCache.set(key, data);
        applyFieldResponse(data, v1, z1, displayModeRef.current, normStatsRef.current ?? null, onColorbarUpdate);
        onRebuildLayers();
      })
      .catch((err) => {
        if (ctrl.signal.aborted) return;
        console.error("[ExplainMap] field fetch failed:", err);
        if (heatmapEverLoadedFlag) {
          currentAlpha = 1.0;
          onRebuildLayers();
        }
      });

    return () => ctrl.abort();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [v1, z1, t, grid, displayMode, apiBaseUrl, modelId, availableTimestamps, refetchTrigger, normStats]);

  return <></>;
}
