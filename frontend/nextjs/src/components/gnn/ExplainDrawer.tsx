"use client";

import React, { useEffect, useRef, useState } from "react";
import { stepLabel } from "../../lib/varLabels";
import HintButton from "./HintButton";
import { HINT_IXG_METHOD, HINT_IG_METHOD } from "../../lib/hints";
import { ATTR_COLD, ATTR_COLD_MID, ATTR_HOT_MID, ATTR_HOT, HEATMAP_NO_DATA } from '../../lib/colors';
export { ATTR_COLD, ATTR_HOT };
import L from "leaflet";
import { Deck } from "@deck.gl/core";
import { GeoJsonLayer, PolygonLayer } from "@deck.gl/layers";
import ExplainRankingSidebar from "./ExplainRankingSidebar";

// ── Types (mirrored from GNNExplorer.tsx) ─────────────────────────────────────

type GridData = {
  lat: number[];
  lon: number[];
  /** Spacing of this model's grid, from GET /grid. Absent on older responses, in which case
   *  cellHalfDeg falls back to the ERA5 value. */
  grid_spacing_deg?: number;
};

type SourceRankingEntry = {
  variable: string;
  level: number | null;
  score: number;
  score_prev_prev: number;
  score_prev: number;
  percentage?: number;
};

export type AttributionResponse = {
  attributions_prev_prev: Record<string, number[]>;
  attributions_prev: Record<string, number[]>;
  target: { variable: string; level: number | null; node: number };
  sources: string[];
  n_nodes: number;
  method: string;
  steps: number | null;
  pred_time: string;
  source_ranking: SourceRankingEntry[];
  visible_indices: number[];
};

// ── Constants ─────────────────────────────────────────────────────────────────

const ATM_VARS = [
  "geopotential", "specific_humidity", "temperature",
  "u_component_of_wind", "v_component_of_wind", "vertical_velocity",
] as const;

const SURF_VARS = [
  "2m_temperature",
  "10m_u_component_of_wind",
  "10m_v_component_of_wind",
  "mean_sea_level_pressure",
  "total_precipitation_6hr",
] as const;

// Ocean state channels are named "<var>_<depth>m" (e.g. "thetao_47m"); zos/siconc/sithick
// carry no depth suffix. Mirrors the map in ExplainRightPanel.
const OCEAN_VAR_LABELS: Record<string, string> = {
  zos: "Sea Surface Height",
  siconc: "Sea Ice Concentration",
  sithick: "Sea Ice Thickness",
  thetao: "Temperature",
  so: "Salinity",
  uo: "U Velocity",
  vo: "V Velocity",
};

export const PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000] as const;

// A variable has a pressure-level axis only if it is one of the ERA5 atmospheric variables.
// Everything else is addressed by name alone: ERA5 surface fields, and every ocean feature,
// which carries its depth in the name (thetao_47m) and has no separate level axis.
// Defined as "not atmospheric" rather than "in SURF_VARS" so ocean features fall on the
// correct side — this one definition drives both the attribution key format (`v` vs `v@level`)
// and whether the level selector is shown, and ocean needs the same answer for both.
const hasLevelAxis = (v: string): boolean =>
  (ATM_VARS as readonly string[]).includes(v);

const isSurfaceVar = (v: string): boolean => !hasLevelAxis(v);

// Sums attribution arrays across all 13 pressure levels for a given variable.
// Returns null if any key is missing (treat as no data).
function sumAttrArrays(map: Record<string, number[]>, v2: string): number[] | null {
  let result: number[] | null = null;
  for (const lvl of PRESSURE_LEVELS) {
    const arr = map[`${v2}@${lvl}`];
    if (!arr) return null;
    if (!result) {
      result = [...arr];
    } else {
      for (let i = 0; i < arr.length; i++) result[i] += arr[i];
    }
  }
  return result;
}

// Appends the correct source= params to a URL.
// When z is "sum", appends 13 repeated params (one per pressure level).
// "sum" is never sent as a literal token to the API.
export function buildPanelSourceParams(url: URL, v: string, z: string | null): void {
  if (isSurfaceVar(v)) {
    url.searchParams.append("source", v);
  } else if (z === "sum") {
    for (const lvl of PRESSURE_LEVELS) {
      url.searchParams.append("source", `${v}@${lvl}`);
    }
  } else if (z !== null) {
    url.searchParams.append("source", `${v}@${z}`);
  }
  // z === null for an atmospheric var is a caller bug; omit rather than send "@null"
}

// Resolves the attribution array for one panel/timestamp slice, handling the "sum" case.
function resolveAttrArr(
  data: AttributionResponse | null | undefined,
  ts: "attributions_prev_prev" | "attributions_prev",
  v2: string,
  z2: string | null,
  key: string,
): number[] | undefined {
  if (!data) return undefined;
  if (z2 === "sum") return sumAttrArrays(data[ts], v2) ?? undefined;
  return data[ts][key];
}

const VAR_LABELS: Record<string, string> = {
  geopotential: "Geopotential",
  specific_humidity: "Sp. Humidity",
  temperature: "Temperature",
  u_component_of_wind: "U Wind",
  v_component_of_wind: "V Wind",
  vertical_velocity: "Vert. Velocity",
  "2m_temperature": "2m Temperature",
  "10m_u_component_of_wind": "10m U Wind",
  "10m_v_component_of_wind": "10m V Wind",
  mean_sea_level_pressure: "MSLP",
  total_precipitation_6hr: "Precip. 6hr",
};

const varLabel = (v: string): string => {
  if (VAR_LABELS[v]) return VAR_LABELS[v];
  const m = /^([a-z]+)_(\d+)m$/.exec(v);
  if (m && OCEAN_VAR_LABELS[m[1]]) return `${OCEAN_VAR_LABELS[m[1]]} (${m[2]}m)`;
  if (OCEAN_VAR_LABELS[v]) return OCEAN_VAR_LABELS[v];
  return v.replace(/_/g, " ");
};

// ── Attribution colormap endpoints — re-exported from src/lib/colors.ts ──────

// ── Mini-map cell geometry (module-level cache, same pattern as ExplainMap) ───

type AttrCell = {
  polygon: [number, number][];
  nodeIndex: number;
};

// Half a grid cell, in degrees, derived from the grid the backend reports. Hardcoding ERA5's
// 1.5° here drew the ocean's 0.25° cells six times too large in each direction, so attribution
// cells overhung the coastline by several cells while the error map — which already derives
// this from grid_spacing_deg — lined up exactly. Same helper, same source of truth.
const ERA5_SPACING_DEG = 1.5;
const cellHalfDeg = (grid: GridData | null): number =>
  ((grid?.grid_spacing_deg ?? ERA5_SPACING_DEG) / 2);

let miniCellGeom: AttrCell[] | null = null;
let miniCellGeomGrid: GridData | null = null;
let miniNodeIndexToCell: Map<number, AttrCell> | null = null;

function ensureMiniCellGeom(grid: GridData): AttrCell[] {
  if (miniCellGeom && miniCellGeomGrid === grid) return miniCellGeom;
  miniCellGeomGrid = grid;
  const half = cellHalfDeg(grid);
  const cells: AttrCell[] = [];
  const lookup = new Map<number, AttrCell>();
  for (let i = 0; i < grid.lat.length; i++) {
    const lat = grid.lat[i];
    if (Math.abs(lat) > 85.05) continue;
    const latS = Math.max(-90, lat - half);
    const latN = Math.min(90, lat + half);
    const cell: AttrCell = {
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
  miniCellGeom = cells;
  miniNodeIndexToCell = lookup;
  return miniCellGeom;
}

// ── Diverging colormap: teal → white → rose (matches UI palette) ──────────────

const ATTR_STOPS: [number, [number, number, number]][] = [
  [-1.0, [ 58, 172, 190]], // #3AACBE teal (most negative)
  [-0.5, [136, 208, 222]], // #88D0DE light teal
  [ 0.0, [255, 255, 255]], // #ffffff white — zero attribution
  [ 0.5, [220, 140, 168]], // #DC8CA8 light rose
  [ 1.0, [200,  88, 120]], // #C85878 deep rose (most positive)
];

function attrColor(value: number | null, scaleMax: number): [number, number, number, number] {
  // The centre of a diverging scale is white, so a zeroed input rendered identically to a
  // genuine zero attribution — "no data here" looked like "no influence here", which is the
  // opposite of what the map should say. Absent cells now arrive as null and get the same
  // no-data grey the error map uses.
  if (value === null || !Number.isFinite(value)) {
    return [...HEATMAP_NO_DATA, 180] as [number, number, number, number];
  }
  if (scaleMax === 0) return [255, 255, 255, 200];
  const t = Math.max(-1, Math.min(1, value / scaleMax));
  let r = ATTR_STOPS[ATTR_STOPS.length - 1][1][0];
  let g = ATTR_STOPS[ATTR_STOPS.length - 1][1][1];
  let b = ATTR_STOPS[ATTR_STOPS.length - 1][1][2];
  for (let i = 0; i < ATTR_STOPS.length - 1; i++) {
    const [t0, [r0, g0, b0]] = ATTR_STOPS[i];
    const [t1, [r1, g1, b1]] = ATTR_STOPS[i + 1];
    if (t <= t1) {
      const f = (t - t0) / (t1 - t0);
      r = Math.round(r0 + f * (r1 - r0));
      g = Math.round(g0 + f * (g1 - g0));
      b = Math.round(b0 + f * (b1 - b0));
      break;
    }
  }
  return [r, g, b, 220];
}

// Symmetric scale max across both panels, excluding the target node's self-attribution.
// Accepts the two arrays to compare directly (both may be null/undefined).
function computeSharedScale(
  arrA: (number | null)[] | undefined | null,
  arrB: (number | null)[] | undefined | null,
  targetNode: number | null,
): number {
  let max = 0;
  const scan = (arr: (number | null)[] | undefined | null): void => {
    if (!arr) return;
    for (let i = 0; i < arr.length; i++) {
      if (targetNode !== null && i === targetNode) continue;
      const v = arr[i];
      // null (JSON for the backend's NaN) means the channel has no data at this cell — not a
      // value of zero. Excluded from the scale so absent cells cannot influence the colormap.
      if (v === null || !Number.isFinite(v)) continue;
      const abs = Math.abs(v);
      if (abs > max) max = abs;
    }
  };
  scan(arrA);
  scan(arrB);
  return max > 0 ? max : 1;
}

// Grand total of absolute attribution mass across every feature and both timesteps,
// derived from source_ranking which the backend computes on the full (un-sliced)
// attribution tensor. Same denominator the ranking sidebar uses for its per-feature
// percentages — so a colorbar tick and a ranking row entry are directly comparable
// (the cell's individual share will always be ≤ its variable's aggregated share).
function computeGrandTotal(data: AttributionResponse | null): number {
  if (!data || !data.source_ranking) return 0;
  let total = 0;
  for (const entry of data.source_ranking) total += entry.score;
  return total;
}

// Format a share-of-total value (already in percent, e.g. 0.42) with adaptive
// precision. Now that the denominator is the grand total across every feature
// and both timesteps, a single cell's share is typically much smaller than a
// variable's share — we keep more decimals on the way down so the value is still
// readable, falling back to scientific notation only when even four decimals
// would round to zero.
function formatSharePct(p: number): string {
  if (!isFinite(p) || p === 0) return "0%";
  const abs = Math.abs(p);
  if (abs >= 10)    return p.toFixed(1) + "%";
  if (abs >= 1)     return p.toFixed(2) + "%";
  if (abs >= 0.1)   return p.toFixed(3) + "%";
  if (abs >= 0.001) return p.toFixed(4) + "%";
  return p.toExponential(1) + "%";
}

// ── Saved xai configuration chips ───────────────────────────────────────────

export type SavedExplainConfig = {
  id: string;
  v1: string;
  z1: string | null;
  node: number;
  t: string;
  label: string;
};

// ── Props ─────────────────────────────────────────────────────────────────────

export type ExplainDrawerProps = {
  open: boolean;
  headerOnly: boolean;
  node: number | null;
  v1: string;
  z1: string | null;
  t: string;
  grid: GridData | null;
  attributionData: AttributionResponse | null;
  attributionDataB: AttributionResponse | null;
  loadingA: boolean;
  loadingB: boolean;
  errorA: string | null;
  errorB: string | null;
  compareMode: "timestamps" | "variables";
  method: "ixg" | "ig";
  pA_v2: string | null;
  pA_z2: string | null;
  pB_v2: string | null;
  pB_z2: string | null;
  z2Locked: boolean;
  /** Flat ocean feature list from /variables (state_vars) when the active model is an ocean
   *  model; null for ERA5. Ocean features have no pressure-level axis — depth is part of the
   *  name — so they replace the Atmospheric/Surface grouping entirely. */
  oceanVars: string[] | null;
  /** Hours between consecutive window timestamps, measured from the loaded window. The model
   *  consumes `prev` and `prev_prev`, so the two input steps are t−1 and t−2 of these — 6h and
   *  12h for ERA5, but 1d and 2d for the ocean. Labels were hardcoded to the ERA5 cadence and
   *  therefore misreported the ocean inputs by a factor of four. */
  stepHours: number;
  coastlineGeoJson: unknown;
  onHide: () => void;
  onExpand: () => void;
  onCompareModeChange: (mode: "timestamps" | "variables") => void;
  onMethodChange: (method: "ixg" | "ig") => void;
  onPAV2Change: (v: string) => void;
  onPAZ2Change: (z: string) => void;
  onPBV2Change: (v: string) => void;
  onPBZ2Change: (z: string) => void;
  onZ2LockChange: (locked: boolean) => void;
  onComputeTarget: (v1: string, z1: string | null) => void;
  savedConfigs: SavedExplainConfig[];
  onSaveConfig: () => void;
  onRestoreConfig: (cfg: SavedExplainConfig) => void;
  onTransitionEnd: () => void;
  onDrawerResize?: () => void;
  openRankingSidebar?: boolean;
  closeSidebarForTour?: boolean;
};

// ── Helpers ───────────────────────────────────────────────────────────────────

const formatCoord = (val: number, isLat: boolean): string => {
  const abs = Math.abs(val).toFixed(1);
  const dir = isLat ? (val >= 0 ? "N" : "S") : (val >= 0 ? "E" : "W");
  return `${abs}°${dir}`;
};

function useLatest<T>(value: T): React.MutableRefObject<T> {
  const ref = useRef(value);
  useEffect(() => {
    ref.current = value;
  }, [value]);
  return ref;
}

// Header button label for the IG method (user-facing wording; full method description is in the HintButton tooltip).
const igHeaderLabel = (): string => {
  return "Exact Feature Attribution (~10-15s)";
};

// Footer method indicator — parsed from method field's k{N} suffix (§6.6).
const footerMethodLabel = (
  data: AttributionResponse | null,
  method: "ixg" | "ig",
): string => {
  const m = data?.method ?? (method === "ixg" ? "ixg" : "ig");
  if (m === "ixg") return "IxG (fast)";
  const kMatch = m.match(/ig_k(\d+)/);
  if (kMatch) {
    const k = Number(kMatch[1]);
    return `IG R-steps=${k} (${k >= 32 ? "complete" : "approx."})`;
  }
  return m;
};

// ── Component ─────────────────────────────────────────────────────────────────

export default function ExplainDrawer({
  open,
  headerOnly,
  node,
  v1,
  z1,
  grid,
  attributionData,
  attributionDataB,
  loadingA,
  loadingB,
  errorA,
  errorB,
  compareMode,
  method,
  pA_v2,
  pA_z2,
  pB_v2,
  pB_z2,
  z2Locked,
  oceanVars,
  stepHours,
  coastlineGeoJson,
  onHide,
  onExpand,
  onCompareModeChange,
  onMethodChange,
  onPAV2Change,
  onPAZ2Change,
  onPBV2Change,
  onPBZ2Change,
  onZ2LockChange,
  onComputeTarget,
  savedConfigs,
  onSaveConfig,
  onRestoreConfig,
  onTransitionEnd,
  onDrawerResize,
  openRankingSidebar,
  closeSidebarForTour,
}: ExplainDrawerProps): React.ReactElement {
  // ── Variables mode timestep toggle (local — not URL-serialized) ──────────
  const [tsToggle, setTsToggle] = useState<"t-12h" | "t-6h">("t-6h");
  // ── Pending (uncommitted) target var + level — draft state for the header's
  // "Compute" flow. Re-syncs from the applied v1/z1 whenever they change
  // externally (e.g. via the left panel), so a stale draft can never be
  // applied by mistake. Node selection never routes through this draft —
  // node clicks on the map always refetch against the *applied* target. ────
  const [pendingV1, setPendingV1] = useState(v1);
  const [pendingZ1, setPendingZ1] = useState(z1);
  // Adjust-state-during-render pattern (not an effect) so the draft resets to
  // the applied target in the same commit the prop changes in, with no extra
  // render pass.
  const [appliedV1Seen, setAppliedV1Seen] = useState(v1);
  const [appliedZ1Seen, setAppliedZ1Seen] = useState(z1);
  if (v1 !== appliedV1Seen || z1 !== appliedZ1Seen) {
    setAppliedV1Seen(v1);
    setAppliedZ1Seen(z1);
    setPendingV1(v1);
    setPendingZ1(z1);
  }
  const targetDirty = pendingV1 !== v1 || pendingZ1 !== z1;
  // ── Ranking sidebar open/closed (local — not URL-serialized) ─────────────
  const [sidebarOpen, setSidebarOpen] = useState(false);
  // ── DOM + instance refs ──────────────────────────────────────────────────
  const drawerRef = useRef<HTMLDivElement>(null);
  const mapAContainerRef = useRef<HTMLDivElement>(null);
  const mapBContainerRef = useRef<HTMLDivElement>(null);
  const mapARef = useRef<L.Map | null>(null);
  const mapBRef = useRef<L.Map | null>(null);
  const deckARef = useRef<Deck | null>(null);
  const deckBRef = useRef<Deck | null>(null);
  const deckAReadyRef = useRef(false);
  const deckBReadyRef = useRef(false);
  // Semaphore preventing mutual update loops between linked viewports.
  const isSyncingRef = useRef(false);

  // ── Slowness indicator — triggers "Taking longer than expected" at 3s ────
  const [slowA, setSlowA] = useState(false);
  const [slowB, setSlowB] = useState(false);

  // ── Prop refs — let the stable init-effect always read latest values ──────
  const onTransitionEndRef = useLatest(onTransitionEnd);
  const coastlineRef = useLatest(coastlineGeoJson);
  const nodeRef = useLatest(node);
  const gridRef = useLatest(grid);
  const attrDataRef = useLatest(attributionData);
  const attrDataBRef = useLatest(attributionDataB);
  const pAV2Ref = useLatest(pA_v2);
  const pAZ2Ref = useLatest(pA_z2);
  const pBV2Ref = useLatest(pB_v2);
  const pBZ2Ref = useLatest(pB_z2);
  const v1Ref = useLatest(v1);
  const z1Ref = useLatest(z1);
  const compareModeRef = useLatest(compareMode);
  const tsToggleRef = useLatest<"t-12h" | "t-6h">(tsToggle);

  // ── Drag-resize refs (R3.7) ──────────────────────────────────────────────
  const dragStartY = useRef(0);
  const dragStartVh = useRef(70);
  const lastInvalidateRef = useRef(0);

  // ── Layer rebuild — always reads latest props via refs ───────────────────
  // Updated on every render; called imperatively from effects and Deck onLoad.
  const rebuildLayersRef = useRef<() => void>(() => {});

  const rebuildLayers = (): void => {
    const attrDataA = attrDataRef.current;
    const attrDataB = attrDataBRef.current;
    const grid = gridRef.current;
    const mode = compareModeRef.current;
    const ts = tsToggleRef.current;

    const pAv2 = pAV2Ref.current ?? v1Ref.current;
    const pAz2 = pAZ2Ref.current ?? z1Ref.current;
    // When z2 = "sum", use a sentinel key that won't match any response entry —
    // the actual array is resolved via sumAttrArrays below.
    const pAKey = isSurfaceVar(pAv2) ? pAv2 : pAz2 === "sum" ? `${pAv2}@sum` : pAz2 ? `${pAv2}@${pAz2}` : pAv2;

    const pBv2 = pBV2Ref.current ?? v1Ref.current;
    const pBz2 = pBZ2Ref.current ?? z1Ref.current;
    const pBKey = isSurfaceVar(pBv2) ? pBv2 : pBz2 === "sum" ? `${pBv2}@sum` : pBz2 ? `${pBv2}@${pBz2}` : pBv2;

    // Resolve which array each panel shows
    let arrA: number[] | undefined;
    let arrB: number[] | undefined;
    if (mode === "timestamps") {
      arrA = resolveAttrArr(attrDataA, "attributions_prev_prev", pAv2, pAz2, pAKey);
      arrB = resolveAttrArr(attrDataA, "attributions_prev", pAv2, pAz2, pAKey);
    } else {
      // Variables mode: each panel draws from its own response; timestep toggle selects the timestamp
      if (ts === "t-12h") {
        arrA = resolveAttrArr(attrDataA, "attributions_prev_prev", pAv2, pAz2, pAKey);
        arrB = resolveAttrArr(attrDataB, "attributions_prev_prev", pBv2, pBz2, pBKey);
      } else {
        arrA = resolveAttrArr(attrDataA, "attributions_prev", pAv2, pAz2, pAKey);
        arrB = resolveAttrArr(attrDataB, "attributions_prev", pBv2, pBz2, pBKey);
      }
    }
  const targetNode = attrDataA?.target.node ?? attrDataB?.target.node ?? null;
  const scaleMax = computeSharedScale(arrA, arrB, targetNode);
    const cells = grid ? ensureMiniCellGeom(grid) : null;

    const panelData = [
      { deck: deckARef.current, ready: deckAReadyRef.current, id: "A", attrArr: arrA ?? null },
      { deck: deckBRef.current, ready: deckBReadyRef.current, id: "B", attrArr: arrB ?? null },
    ];

    for (const { deck, ready, id, attrArr } of panelData) {
      if (!deck || !ready) continue;
      const layers: unknown[] = [];

      // Attribution heatmap — cells is polar-filtered so cells.length < attrArr.length (N_grid)
      if (cells && attrArr && cells.length > 0) {
        layers.push(
          new PolygonLayer<AttrCell>({
            id: `mini-attr-${id}`,
            data: cells,
            getPolygon: (d) => d.polygon,
            getFillColor: (d) => attrColor(attrArr[d.nodeIndex], scaleMax),
            getLineWidth: 0,
            lineWidthMinPixels: 0,
            filled: true,
            stroked: false,
            pickable: false,
            wrapLongitude: true,
            updateTriggers: { getFillColor: [attrArr, scaleMax] },
          }),
        );
      }

      const cGeoJson = coastlineRef.current;
      if (cGeoJson) {
        layers.push(
          new GeoJsonLayer({
            id: `mini-coast-${id}`,
            data: cGeoJson as GeoJSON.FeatureCollection,
            stroked: true,
            filled: false,
            lineWidthMinPixels: 1,
            getLineColor: [148, 163, 184, 255], // --text-dim
            pickable: false,
            parameters: { depthTest: false },
          }),
        );
      }

      const n = nodeRef.current;
      if (n !== null && miniNodeIndexToCell) {
        const targetCell = miniNodeIndexToCell.get(n);
        if (targetCell) {
          layers.push(
            new PolygonLayer<AttrCell>({
              id: `mini-ring-${id}`,
              data: [targetCell],
              getPolygon: (d) => d.polygon,
              getFillColor: [0, 0, 0, 0],
              getLineColor: [0, 0, 0, 255],
              getLineWidth: 2,
              lineWidthUnits: "pixels",
              lineWidthMinPixels: 2,
              filled: false,
              stroked: true,
              pickable: false,
              wrapLongitude: true,
              updateTriggers: { data: n },
            }),
          );
        }
      }

      deck.setProps({ layers: layers as never[] });
    }
  };

  useEffect(() => {
    rebuildLayersRef.current = rebuildLayers;
  });

  // ── Init both mini-maps on mount ─────────────────────────────────────────
  // The drawer is always in the DOM (CSS transform hides it), so Leaflet
  // initialises into a container that already has its correct computed size.
  useEffect(() => {
    const containerA = mapAContainerRef.current;
    const containerB = mapBContainerRef.current;
    if (!containerA || !containerB) return;

    // ── Leaflet maps ───────────────────────────────────────────────────────
    const mapOpts: L.MapOptions = {
      zoomControl: false,
      attributionControl: false,
      preferCanvas: false,
      minZoom: 1,
      maxZoom: 8,
    };
    const mapA = L.map(containerA, mapOpts);
    const mapB = L.map(containerB, mapOpts);
    mapA.setView([20, 0], 1);
    mapB.setView([20, 0], 1);
    mapARef.current = mapA;
    mapBRef.current = mapB;

    // ── Deck canvases (pointer-events:none — same pattern as main map) ─────
    const makeCanvas = (container: HTMLDivElement, lmap: L.Map): HTMLCanvasElement => {
      const sz = lmap.getSize();
      const canvas = document.createElement("canvas");
      canvas.style.cssText = "position:absolute;top:0;left:0;pointer-events:none;z-index:500;";
      canvas.width  = sz.x; canvas.height  = sz.y;
      canvas.style.width  = `${sz.x}px`;
      canvas.style.height = `${sz.y}px`;
      container.appendChild(canvas);
      return canvas;
    };
    const canvasA = makeCanvas(containerA, mapA);
    const canvasB = makeCanvas(containerB, mapB);

    // ── Deck view sync helper ──────────────────────────────────────────────
    const syncDeck = (lmap: L.Map, deck: Deck): void => {
      const c = lmap.getCenter();
      // deck.gl uses 512px tiles; Leaflet reports zoom for 256px tiles → subtract 1
      deck.setProps({
        viewState: {
          longitude: c.lng,
          latitude: c.lat,
          zoom: lmap.getZoom() - 1,
          pitch: 0,
          bearing: 0,
          transitionDuration: 0,
        },
      });
      deck.redraw();
    };

    // ── Deck instances ─────────────────────────────────────────────────────
    const makeDeck = (
      canvas: HTMLCanvasElement,
      lmap: L.Map,
      readyRef: React.MutableRefObject<boolean>,
      deckRef: React.MutableRefObject<Deck | null>,
    ): Deck => {
      const sz = lmap.getSize();
      // deck is captured by the onLoad closure; onLoad fires async so deck is
      // fully initialized by the time the callback runs.
      const deck = new Deck({
        canvas,
        width: sz.x,
        height: sz.y,
        controller: false,
        deviceProps: { type: "webgl" },
        initialViewState: { longitude: 0, latitude: 20, zoom: 0, pitch: 0, bearing: 0 },
        layers: [],
        onLoad: () => {
          deckRef.current = deck;
          readyRef.current = true;
          syncDeck(lmap, deck);
          rebuildLayersRef.current();
        },
      });
      return deck;
    };

    const deckA = makeDeck(canvasA, mapA, deckAReadyRef, deckARef);
    const deckB = makeDeck(canvasB, mapB, deckBReadyRef, deckBRef);

    // ── Sync deck views on every Leaflet move ──────────────────────────────
    mapA.on("move", () => { if (deckAReadyRef.current) syncDeck(mapA, deckA); });
    mapB.on("move", () => { if (deckBReadyRef.current) syncDeck(mapB, deckB); });

    // ── Linked viewport sync via semaphore ─────────────────────────────────
    mapA.on("moveend", () => {
      if (isSyncingRef.current) return;
      isSyncingRef.current = true;
      mapB.setView(mapA.getCenter(), mapA.getZoom(), { animate: false });
      isSyncingRef.current = false;
    });
    mapB.on("moveend", () => {
      if (isSyncingRef.current) return;
      isSyncingRef.current = true;
      mapA.setView(mapB.getCenter(), mapB.getZoom(), { animate: false });
      isSyncingRef.current = false;
    });

    // ── ResizeObserver: keep deck canvases matched to their containers ──────
    const makeResizeObserver = (
      container: HTMLDivElement,
      canvas: HTMLCanvasElement,
      lmap: L.Map,
      deckRef: React.MutableRefObject<Deck | null>,
    ): ResizeObserver => {
      const ro = new ResizeObserver(() => {
        const w = container.clientWidth;
        const h = container.clientHeight;
        if (w === 0 || h === 0) return;
        canvas.width = w; canvas.height = h;
        canvas.style.width = `${w}px`; canvas.style.height = `${h}px`;
        deckRef.current?.setProps({ width: w, height: h });
        lmap.invalidateSize();
      });
      ro.observe(container);
      return ro;
    };
    const roA = makeResizeObserver(containerA, canvasA, mapA, deckARef);
    const roB = makeResizeObserver(containerB, canvasB, mapB, deckBRef);

    // ── Drawer transitionend → invalidateSize on main + mini-maps ─────────
    // Fires on both open and close; filtered to the transform property to avoid
    // spurious calls from child element transitions.
    const drawerEl = drawerRef.current;
    const onTransition = (e: TransitionEvent): void => {
      if (e.propertyName !== "transform" || e.target !== drawerEl) return;
      onTransitionEndRef.current();
      mapA.invalidateSize();
      mapB.invalidateSize();
    };
    drawerEl?.addEventListener("transitionend", onTransition as EventListener);

    return () => {
      roA.disconnect();
      roB.disconnect();
      drawerEl?.removeEventListener("transitionend", onTransition as EventListener);
      mapA.remove();
      mapB.remove();
      deckA.finalize();
      deckB.finalize();
    };
  }, [onTransitionEndRef]); // runs once on mount

  // ── Rebuild layers when any layer-relevant prop changes ──────────────────
  useEffect(() => {
    rebuildLayersRef.current();
  }, [coastlineGeoJson, node, grid, attributionData, attributionDataB, pA_v2, pA_z2, pB_v2, pB_z2, compareMode, tsToggle]);

  // ── Reset to global view each time the drawer opens ─────────────────────
  useEffect(() => {
    if (open) {
      mapARef.current?.setView([20, 0], 1, { animate: false });
      mapBRef.current?.setView([20, 0], 1, { animate: false });
    }
  }, [open]);

  // ── Slowness timers ──────────────────────────────────────────────────────
  useEffect(() => {
    const id = setTimeout(() => setSlowA(loadingA), loadingA ? 3000 : 0);
    return () => clearTimeout(id);
  }, [loadingA]);

  useEffect(() => {
    const id = setTimeout(() => setSlowB(loadingB), loadingB ? 3000 : 0);
    return () => clearTimeout(id);
  }, [loadingB]);

  const sidebarVisible = method === "ig" || (sidebarOpen && !closeSidebarForTour) || openRankingSidebar;

  // Escape: cascade — sidebar first, then drawer
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (sidebarVisible) { setSidebarOpen(false); return; }
      if (open) { onHide(); }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [sidebarVisible, open, onHide]);

  // ── Derived display values ───────────────────────────────────────────────

  const pAv2Display = pA_v2 ?? v1;
  const pAz2Display = pA_z2 ?? z1;
  const pBv2Display = pB_v2 ?? v1;
  const pBz2Display = pB_z2 ?? z1;

  // When z2 = "sum", use a sentinel key for ranking sidebar match (never matches a real entry).
  // Option list shared by every variable select in this drawer (target, and the source
  // selectors in both compare modes). Ocean models expose a flat feature list, so there is
  // nothing to group by and no level axis to offer.
  const varOptions = (): React.ReactElement =>
    oceanVars && oceanVars.length > 0 ? (
      <optgroup label="Ocean">
        {oceanVars.map((v) => (
          <option key={v} value={v}>{varLabel(v)}</option>
        ))}
      </optgroup>
    ) : (
      <>
        <optgroup label="Atmospheric">
          {(ATM_VARS as readonly string[]).map((v) => (
            <option key={v} value={v}>{varLabel(v)}</option>
          ))}
        </optgroup>
        <optgroup label="Surface">
          {(SURF_VARS as readonly string[]).map((v) => (
            <option key={v} value={v}>{varLabel(v)}</option>
          ))}
        </optgroup>
      </>
    );

  const pAKey = isSurfaceVar(pAv2Display) ? pAv2Display : pAz2Display === "sum" ? `${pAv2Display}@sum` : pAz2Display ? `${pAv2Display}@${pAz2Display}` : pAv2Display;
  const pBKey = isSurfaceVar(pBv2Display) ? pBv2Display : pBz2Display === "sum" ? `${pBv2Display}@sum` : pBz2Display ? `${pBv2Display}@${pBz2Display}` : pBv2Display;

  // Resolve which arrays are currently visible (mirrors rebuildLayers logic for colorbar)
  const displayArrA = compareMode === "timestamps"
    ? resolveAttrArr(attributionData, "attributions_prev_prev", pAv2Display, pAz2Display, pAKey)
    : tsToggle === "t-12h"
      ? resolveAttrArr(attributionData, "attributions_prev_prev", pAv2Display, pAz2Display, pAKey)
      : resolveAttrArr(attributionData, "attributions_prev", pAv2Display, pAz2Display, pAKey);
  const displayArrB = compareMode === "timestamps"
    ? resolveAttrArr(attributionData, "attributions_prev", pAv2Display, pAz2Display, pAKey)
    : tsToggle === "t-12h"
      ? resolveAttrArr(attributionDataB, "attributions_prev_prev", pBv2Display, pBz2Display, pBKey)
      : resolveAttrArr(attributionDataB, "attributions_prev", pBv2Display, pBz2Display, pBKey);

  const targetNode = attributionData?.target.node ?? attributionDataB?.target.node ?? null;
  // Symmetric scale max across both panels (excluding self-attribution) — colour
  // intensity is value / scaleMax. Grand total comes from source_ranking so the
  // colorbar's % matches the ranking sidebar's % on the same denominator.
  const attrScaleMax = computeSharedScale(displayArrA, displayArrB, targetNode);
  const attrGrandTotal = computeGrandTotal(attributionData ?? attributionDataB);
  const attrSharePct = attrGrandTotal > 0 ? (attrScaleMax / attrGrandTotal) * 100 : 0;

  const coordLabel =
    node !== null && grid && node < grid.lat.length
      ? `${formatCoord(grid.lat[node], true)}, ${formatCoord(grid.lon[node], false)}`
      : "No node selected";

  const showIgBaseline = attributionData?.method !== "ixg" && method !== "ixg";

  // ── Render ───────────────────────────────────────────────────────────────

  // ── Drag-resize handlers (R3.7) ─────────────────────────────────────────
  const onDragPointerDown = (e: React.PointerEvent<HTMLDivElement>): void => {
    e.currentTarget.setPointerCapture(e.pointerId);
    const appEl = document.getElementById("app");
    const cssVal = appEl?.style.getPropertyValue("--drawer-height-vh") ?? "";
    dragStartY.current = e.clientY;
    dragStartVh.current = parseFloat(cssVal) || 80;
  };

  const onDragPointerMove = (e: React.PointerEvent<HTMLDivElement>): void => {
    if (!e.currentTarget.hasPointerCapture(e.pointerId)) return;
    const deltaY = dragStartY.current - e.clientY;
    const newVh = Math.max(25, Math.min(90, dragStartVh.current + (deltaY / window.innerHeight) * 100));
    document.getElementById("app")?.style.setProperty("--drawer-height-vh", `${newVh}vh`);
    const now = Date.now();
    if (now - lastInvalidateRef.current > 100) {
      lastInvalidateRef.current = now;
      mapARef.current?.invalidateSize();
      mapBRef.current?.invalidateSize();
      onDrawerResize?.();
    }
  };

  const onDragPointerUp = (e: React.PointerEvent<HTMLDivElement>): void => {
    e.currentTarget.releasePointerCapture(e.pointerId);
  };

  return (
    <div ref={drawerRef} id="explain-drawer-root" className={`explain-drawer${open ? " open" : ""}`}>

      {/* ── Drag-resize handle (R3.7) — top edge of drawer ──────────────── */}
      <div
        className="ex-drawer-drag-bar"
        onPointerDown={onDragPointerDown}
        onPointerMove={onDragPointerMove}
        onPointerUp={onDragPointerUp}
      />

      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div
        className={`ex-drawer-hdr${headerOnly ? " ex-drawer-hdr--clickable" : ""}`}
        onClick={headerOnly ? onExpand : undefined}
      >
        <div className="ex-drawer-hdr-c">
          {/* Compare mode toggle */}
          <span className="ex-drawer-hdr-pill-label">Compare:</span>
          <div
            id="ex-drawer-compare"
            className="ex-drawer-toggle ex-drawer-toggle--pill"
            role="radiogroup"
            aria-label="Comparison mode"
          >
            <span
              className={`ex-drawer-toggle-pill-thumb${compareMode === "variables" ? " ex-drawer-toggle-pill-thumb--right" : ""}`}
              aria-hidden="true"
            />
            <button
              type="button"
              role="radio"
              title="Compare attributions across two different input timepoints"
              aria-checked={compareMode === "timestamps"}
              className={`ex-drawer-toggle-btn${compareMode === "timestamps" ? " active" : ""}`}
              onClick={() => onCompareModeChange("timestamps")}
            >
              Input Timepoints
            </button>
            <button
              type="button"
              role="radio"
              title="Compare attributions for two different source variables side by side"
              aria-checked={compareMode === "variables"}
              className={`ex-drawer-toggle-btn${compareMode === "variables" ? " active" : ""}`}
              onClick={() => onCompareModeChange("variables")}
            >
              Source Variables
            </button>
          </div>

          {/* Method toggle */}
          <span className="ex-drawer-hdr-pill-label">Method:</span>
          <div
            className="ex-drawer-toggle ex-drawer-toggle--pill"
            role="radiogroup"
            aria-label="Attribution method"
          >
            <span
              className={`ex-drawer-toggle-pill-thumb${method === "ig" ? " ex-drawer-toggle-pill-thumb--right" : ""}`}
              aria-hidden="true"
            />
            <button
              type="button"
              role="radio"
              aria-checked={method === "ixg"}
              className={`ex-drawer-toggle-btn${method === "ixg" ? " active" : ""}`}
              onClick={() => onMethodChange("ixg")}
            >
              Fast Overview (~2s)
              <HintButton placement="top" delayMs={200} content={HINT_IXG_METHOD} />
            </button>
            <button
              type="button"
              role="radio"
              aria-checked={method === "ig"}
              className={`ex-drawer-toggle-btn${method === "ig" ? " active" : ""}`}
              onClick={() => onMethodChange("ig")}
            >
              {igHeaderLabel()}
              <HintButton placement="top" delayMs={200} content={HINT_IG_METHOD} />
            </button>
          </div>
        </div>

        <div className="ex-drawer-hdr-r">
          {headerOnly && (
            <span className="ex-drawer-hdr-expand" aria-hidden="true">▲</span>
          )}
          <button
            type="button"
            className="ex-drawer-hide-btn"
            title={headerOnly ? "Close the attribution drawer" : "Minimise the attribution drawer"}
            onClick={(e) => { e.stopPropagation(); onHide(); }}
          >
            {headerOnly ? "Close ✕" : "Hide ↓"}
          </button>
        </div>
      </div>

      {/* ── Context bar ──────────────────────────────────────────────────
          Pulled out of the header so the Compare/Method pills aren't squeezed
          by long target variable names. Always visible (including collapsed
          state) so the target variable name is never hidden. */}
      <div className="ex-drawer-context">
        <span className="ex-drawer-hdr-key">Target</span>
        <select
          className="ex-target-select"
          value={pendingV1}
          onChange={(e) => {
            const nextV1 = e.target.value;
            setPendingV1(nextV1);
            if (isSurfaceVar(nextV1)) {
              setPendingZ1(null);
            } else if (pendingZ1 === null) {
              setPendingZ1("500");
            }
          }}
          title="Target variable — press Compute to apply"
        >
          {varOptions()}
        </select>
        {!isSurfaceVar(pendingV1) && (
          <select
            className="ex-target-select ex-target-select--level"
            value={pendingZ1 ?? "500"}
            onChange={(e) => setPendingZ1(e.target.value)}
            title="Target pressure level — press Compute to apply"
          >
            {(PRESSURE_LEVELS as readonly number[]).map((lvl) => (
              <option key={lvl} value={String(lvl)}>{lvl} hPa</option>
            ))}
          </select>
        )}
        <button
          type="button"
          className="ex-target-compute-btn"
          disabled={!targetDirty}
          title={targetDirty ? "Apply the selected target and recompute attribution" : "Select a different variable or level to enable"}
          onClick={() => onComputeTarget(pendingV1, isSurfaceVar(pendingV1) ? null : pendingZ1)}
        >
          Compute
        </button>
        <span className="ex-drawer-hdr-sep">|</span>
        <span className="ex-drawer-hdr-key">Node</span>
        <span className="ex-drawer-hdr-val">{coordLabel}</span>
        <button
          type="button"
          className="ex-target-flag-btn"
          disabled={node === null}
          title="Save this (target, node, timestep) configuration as a chip"
          onClick={onSaveConfig}
        >
          ⚑ Flag
        </button>
      </div>

      {savedConfigs.length > 0 && (
        <div className="ex-saved-configs-row">
          <span className="ex-drawer-hdr-key">Saved</span>
          <div className="ex-saved-configs-chips">
            {savedConfigs.map((cfg) => (
              <button
                key={cfg.id}
                type="button"
                className="ex-saved-config-chip"
                title={`${varLabel(cfg.v1)}${cfg.z1 && cfg.z1 !== "sum" ? ` @ ${cfg.z1} hPa` : ""} · node ${cfg.node} · ${cfg.t}`}
                onClick={() => onRestoreConfig(cfg)}
              >
                {cfg.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* ── Source controls (F7) ─────────────────────────────────────────── */}
      {compareMode === "timestamps" ? (
        /* Timestamps mode: one shared source above both panels */
        <div className="ex-drawer-source-row">
          <span className="ex-drawer-source-label">Source</span>
          <div className="ex-src-controls">
            <select
              className="ex-v2-select"
              value={pAv2Display}
              onChange={(e) => onPAV2Change(e.target.value)}
            >
              {varOptions()}
            </select>
            {!isSurfaceVar(pAv2Display) && (
              <div className="ex-z2-chips">
                <button
                  key="sum"
                  type="button"
                  title="Sum attributions across all pressure levels"
                  className={`ex-z2-chip${pAz2Display === "sum" ? " active" : ""}`}
                  onClick={() => onPAZ2Change("sum")}
                >
                  Σ
                </button>
                {(PRESSURE_LEVELS as readonly number[]).map((lvl) => (
                  <button
                    key={lvl}
                    type="button"
                    title={`${lvl} hPa pressure level`}
                    className={`ex-z2-chip${pAz2Display === String(lvl) ? " active" : ""}`}
                    onClick={() => onPAZ2Change(String(lvl))}
                  >
                    {lvl}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      ) : (
        /* Variables mode: timestep toggle + per-panel source selectors */
        <div className="ex-drawer-vars-ctrl">
          <div className="ex-ts-toggle-row">
            <span className="ex-drawer-source-label">Input timestep</span>
            <div className="ex-drawer-toggle" role="group">
              <button
                type="button"
                title={`Show attributions from the input ${stepLabel(stepHours, 2)} before forecast time`}
                className={`ex-drawer-toggle-btn ex-drawer-toggle-btn--t12${tsToggle === "t-12h" ? " active" : ""}`}
                onClick={() => { setTsToggle("t-12h"); rebuildLayersRef.current(); }}
              >
                {stepLabel(stepHours, 2)}
              </button>
              <button
                type="button"
                title={`Show attributions from the input ${stepLabel(stepHours, 1)} before forecast time`}
                className={`ex-drawer-toggle-btn ex-drawer-toggle-btn--t6${tsToggle === "t-6h" ? " active" : ""}`}
                onClick={() => { setTsToggle("t-6h"); rebuildLayersRef.current(); }}
              >
                {stepLabel(stepHours, 1)}
              </button>
            </div>
          </div>
          <div className="ex-vars-panel-row">
            {/* Panel A source */}
            <div className="ex-vars-panel-src">
              <span className="ex-panel-chip ex-panel-chip--a">SOURCE A</span>
              <div className="ex-src-controls">
                <select
                  className="ex-v2-select"
                  value={pAv2Display}
                  onChange={(e) => onPAV2Change(e.target.value)}
                >
                  {varOptions()}
                </select>
                {!isSurfaceVar(pAv2Display) && (
                  <div className="ex-z2-chips">
                    <button
                      key="sum"
                      type="button"
                      title="Sum attributions across all pressure levels"
                      className={`ex-z2-chip${pAz2Display === "sum" ? " active" : ""}`}
                      onClick={() => onPAZ2Change("sum")}
                    >
                      Σ
                    </button>
                    {(PRESSURE_LEVELS as readonly number[]).map((lvl) => (
                      <button
                        key={lvl}
                        type="button"
                        title={`${lvl} hPa pressure level`}
                        className={`ex-z2-chip${pAz2Display === String(lvl) ? " active" : ""}`}
                        onClick={() => onPAZ2Change(String(lvl))}
                      >
                        {lvl}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
            {/* z2 lock between the two panel rows */}
            <button
              type="button"
              className={`ex-z2-lock-btn${z2Locked ? " active" : ""}`}
              title={z2Locked ? "Unlink pressure levels" : "Link pressure levels"}
              onClick={() => onZ2LockChange(!z2Locked)}
            >
              {z2Locked ? "⇔ Linked" : "⇔ Link z"}
            </button>
            {/* Panel B source */}
            <div className="ex-vars-panel-src">
              <span className="ex-panel-chip ex-panel-chip--b">SOURCE B</span>
              <div className="ex-src-controls">
                <select
                  className="ex-v2-select"
                  value={pBv2Display}
                  onChange={(e) => onPBV2Change(e.target.value)}
                >
                  {varOptions()}
                </select>
                {!isSurfaceVar(pBv2Display) && (
                  <div className="ex-z2-chips">
                    <button
                      key="sum"
                      type="button"
                      title="Sum attributions across all pressure levels"
                      className={`ex-z2-chip${pBz2Display === "sum" ? " active" : ""}`}
                      onClick={() => onPBZ2Change("sum")}
                    >
                      Σ
                    </button>
                    {(PRESSURE_LEVELS as readonly number[]).map((lvl) => (
                      <button
                        key={lvl}
                        type="button"
                        title={`${lvl} hPa pressure level`}
                        className={`ex-z2-chip${pBz2Display === String(lvl) ? " active" : ""}`}
                        onClick={() => onPBZ2Change(String(lvl))}
                      >
                        {lvl}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Two panels side by side ──────────────────────────────────────── */}
      <div className="ex-drawer-panels">

        {/* Panel A — t-12h in timestamps mode, selected ts in variables mode */}
        <div className="ex-drawer-panel">
          <div className="ex-panel-hdr">
            {compareMode === "timestamps"
              ? <span className="ex-panel-chip ex-panel-chip--t12">{stepLabel(stepHours, 2)}</span>
              : <><span className="ex-panel-chip ex-panel-chip--a">SOURCE A</span><span className={`ex-panel-chip ${tsToggle === "t-12h" ? "ex-panel-chip--t12" : "ex-panel-chip--t6"}`}>{tsToggle === "t-12h" ? stepLabel(stepHours, 2) : stepLabel(stepHours, 1)}</span>{varLabel(pAv2Display)}{!isSurfaceVar(pAv2Display) && ` @ ${pAz2Display} hPa`}</>}
          </div>
          <div className="ex-mini-map-wrap" ref={mapAContainerRef}>
            {/* Shimmer — node selected, fetch in flight */}
            {loadingA && (
              <div className="ex-panel-overlay ex-panel-overlay--shimmer">
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "8px" }}>
                  <div className="ex-spinner ex-spinner--sm" />
                  {slowA ? (
                    <span className="ex-panel-overlay-text">Taking longer than expected…</span>
                  ) : method === "ig" ? (
                    <span className="ex-panel-overlay-text">Computing IG attribution…</span>
                  ) : null}
                </div>
              </div>
            )}
            {/* Error state */}
            {!loadingA && errorA && (
              <div className="ex-panel-overlay ex-panel-overlay--error">
                <span className="ex-panel-overlay-text">{errorA}</span>
              </div>
            )}
            {/* Empty state — drawer open, no node selected */}
            {!loadingA && !errorA && node === null && (
              <div className="ex-panel-overlay ex-panel-overlay--empty">
                No node selected — click a node on the map to load attribution.
              </div>
            )}
          </div>
        </div>

        {/* Panel B — t-6h in timestamps mode, selected ts + source in variables mode */}
        <div className="ex-drawer-panel">
          <div className="ex-panel-hdr">
            {compareMode === "timestamps"
              ? <span className="ex-panel-chip ex-panel-chip--t6">{stepLabel(stepHours, 1)}</span>
              : <><span className="ex-panel-chip ex-panel-chip--b">SOURCE B</span><span className={`ex-panel-chip ${tsToggle === "t-12h" ? "ex-panel-chip--t12" : "ex-panel-chip--t6"}`}>{tsToggle === "t-12h" ? stepLabel(stepHours, 2) : stepLabel(stepHours, 1)}</span>{varLabel(pBv2Display)}{!isSurfaceVar(pBv2Display) && ` @ ${pBz2Display} hPa`}</>}
          </div>
          <div className="ex-mini-map-wrap" ref={mapBContainerRef}>
            {loadingB && (
              <div className="ex-panel-overlay ex-panel-overlay--shimmer">
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "8px" }}>
                  <div className="ex-spinner ex-spinner--sm" />
                  {slowB ? (
                    <span className="ex-panel-overlay-text">Taking longer than expected…</span>
                  ) : method === "ig" ? (
                    <span className="ex-panel-overlay-text">Computing IG attribution…</span>
                  ) : null}
                </div>
              </div>
            )}
            {!loadingB && errorB && (
              <div className="ex-panel-overlay ex-panel-overlay--error">
                <span className="ex-panel-overlay-text">{errorB}</span>
              </div>
            )}
            {!loadingB && !errorB && node === null && (
              <div className="ex-panel-overlay ex-panel-overlay--empty">
                No node selected — click a node on the map to load attribution.
              </div>
            )}
          </div>
        </div>

        {/* ── Feature ranking toggle + sidebar ─────────────────────────── */}
        <div id="ex-ranking-wrap" style={{ display: "flex", flexShrink: 0 }}>
        <button
          id="ex-ranking-toggle"
          type="button"
          className="ex-ranking-toggle"
          onClick={() => setSidebarOpen((v) => !v)}
          title={sidebarVisible ? "Close feature ranking" : "Open feature ranking"}
        >
          <span style={{
            writingMode: "vertical-rl",
            transform: "rotate(180deg)",
            fontSize: "9px",
            fontVariant: "small-caps",
            letterSpacing: "0.1em",
            color: "var(--text-muted)",
            userSelect: "none",
          }}>RANKING</span>
          <span style={{ fontSize: "11px" }}>{sidebarOpen ? "‹" : "›"}</span>
        </button>

        {/* ── Feature ranking sidebar ─────────────────────────────────── */}
        {sidebarVisible && (
          <ExplainRankingSidebar
            stepHours={stepHours}
            key={compareMode}
            sourceRanking={(attributionData ?? attributionDataB)?.source_ranking ?? null}
            pAKey={pAKey}
            pBKey={pBKey}
            compareMode={compareMode}
            method={method}
            tsToggle={tsToggle}
            loading={loadingA || loadingB}
          />
        )}
        </div>{/* /ex-ranking-wrap */}

      </div>

      {/* ── Attribution colorbar ────────────────────────────────────────── */}
      <div className="ex-attr-cb-row">
        {/* Keep stale colorbar visible during re-fetch; skeleton only on first load */}
        {attributionData ? (
          <div className="ex-attr-cb">
            {/* Ticks show the share of total absolute attribution mass carried by
                the most-extreme cell visible in the drawer. Symmetric around 0:
                colour intensity reflects each cell's share, ±(max share) at the
                extremes. Values are typically a few % for spatially concentrated
                attributions and fractions of % for diffuse ones. */}
            <span className="ex-attr-cb-tick">−{formatSharePct(attrSharePct)}</span>
            <div
              className="ex-attr-cb-bar"
              style={{ background: `linear-gradient(to right, ${ATTR_COLD}, ${ATTR_COLD_MID}, #ffffff, ${ATTR_HOT_MID}, ${ATTR_HOT})` }}
            />
            <span className="ex-attr-cb-tick">+{formatSharePct(attrSharePct)}</span>
          </div>
        ) : (loadingA || loadingB) ? (
          <div className="ex-attr-cb-skeleton" />
        ) : null}
        <div className="ex-attr-cb-meta">
          <span className="ex-attr-cb-label">
            Attribution (norm.)
            {(loadingA || loadingB) && <span className="ex-spinner ex-spinner--xs" />}
            {showIgBaseline && (
              <span
                className="ex-attr-cb-ig"
                title="Baseline = climatological mean (zero in normalised space). Positive attribution: this input, being above climatology, supported the correct prediction. Negative: being above climatology pushed the prediction away from truth."
              >
                {" "}ⓘ
              </span>
            )}
          </span>
          <span className="ex-attr-cb-sep">·</span>
          <span className="ex-attr-cb-note">
            State inputs only ({stepLabel(stepHours, 2)}, {stepLabel(stepHours, 1)}) · forcing variables held fixed
          </span>
        </div>
      </div>

      {/* ── Footer ──────────────────────────────────────────────────────── */}
      <div className="ex-drawer-footer">
        <span className="ex-drawer-method-ind">
          Method: {footerMethodLabel(attributionData, method)}
        </span>
      </div>

    </div>
  );
}
