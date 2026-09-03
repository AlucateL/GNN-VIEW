"use client";

import "leaflet/dist/leaflet.css";
import "./GNNExplorer.css";
import React, { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent } from "react";
import { NODE_GRID, NODE_MESH, EDGE_ENCODER, EDGE_DECODER, EDGE_MESH, VIOL_ABOVE, VIOL_BELOW, VIOL_NEUTRAL, TEXT, TEXT_DIM, MINIMAP_ACCENT } from '../../lib/colors';
import { ONBOARD_COOKIE, EXPLORE_COOKIE, INSPECT_COOKIE, EXPLAIN_COOKIE, ONBOARD_STEPS, EXPLORE_STEPS, INSPECT_STEPS, EXPLAIN_STEPS } from '../../lib/tourSteps';
import { HINT_SCALE_FADE, HINT_SRC_DEGREE, HINT_TGT_DEGREE, HINT_EDGE_LENGTH } from '../../lib/hints';
import { GCS_FETCH_ENABLED, UPLOADS_ENABLED } from '../../lib/config';
import { StackedSidebar, strokeForLevel, nodeLevelBoost, GlyphSquare, GlyphCircle, GlyphArrow, GlyphLine } from "./LeftPanel";
import ExplainDrawer, { buildPanelSourceParams } from "./ExplainDrawer";
import ExplainLeftPanel from "./ExplainLeftPanel";
import ExplainRightPanel from "./ExplainRightPanel";
import ExplainMap, { buildHeatmapLayer, buildHoverRingLayer, buildSelectionRingLayer, clearFieldCache } from "./ExplainMap";
import { LRUCache } from "../../lib/lru";
import ExplainTimeline, { type MonthBand } from "./ExplainTimeline";
import HintButton from "./HintButton";
import L from "leaflet";
import * as d3 from "d3";
import { Deck, MapView } from "@deck.gl/core";
import { GeoJsonLayer, IconLayer, LineLayer, ScatterplotLayer } from "@deck.gl/layers";

type SubgraphKey = string;
type ViolGraphType = SubgraphKey;
type NodeType = string;
type SubgraphDef = { key: string; src: string; edge: string; tgt: string; label: string; plot: "lollipop" | "histogram"; n_edges?: number };

type NodeDatum = {
  id: number;
  lat: number;
  lon: number;
  type: NodeType;
};

type ApiEdge = {
  source: number;
  target: number;
  src_lat: number;
  src_lon: number;
  tgt_lat: number;
  tgt_lon: number;
  features?: { f0?: number };
};

type EdgeDatum = {
  src: { id: number; lat: number; lon: number };
  tgt: { id: number; lat: number; lon: number };
  source: number;
  target: number;
  len: number;
};

type STType = {
  showGrid: boolean;
  showMesh: boolean;
  showG2M: boolean;
  showM2M: boolean;
  showM2G: boolean;
  selected: NodeDatum | null;
  violation: boolean;
  activeInspectSubgraph: SubgraphKey;
  srcDegMin: number;
  srcDegMax: number;
  tgtDegMin: number;
  tgtDegMax: number;
  lenSliderMax: number;
  srcDegLogScaleX: boolean;
  srcDegLogScaleY: boolean;
  tgtDegLogScaleX: boolean;
  tgtDegLogScaleY: boolean;
  lenLogScaleX: boolean;
  lenLogScaleY: boolean;
  zoomFade: boolean;
  activeSubgraphs: Set<SubgraphKey>;
  activeNodeTypes: Set<NodeType>;
};

type MapTypeConfig = {
  url: string;
  opts: L.TileLayerOptions;
  attr: string;
  filter: string;
};

type ThemeMode = "dark" | "light" | "auto";


type PerSubgraphThresholds = {
  srcDegLo: number; srcDegHi: number;
  tgtDegLo: number; tgtDegHi: number;
  lengthLo: number; lengthHi: number;
};

type GraphStatsMeta = {
  n_grid: number;
  grid_spacing_deg: number;   // 0 until the backend reports it
  n_mesh: number;
  n_edges_g2m: number;
  n_edges_m2m: number;
  n_edges_m2g: number;
  is_hierarchical: boolean;
  n_levels: number;
  node_types: string[];
  node_counts: Record<string, number>;
  subgraphs: SubgraphDef[];
};

type GraphRecord = {
  graph_id: string;
  name: string;
  description?: string | null;
  folder_path: string;
  created_at?: string | null;
  updated_at?: string | null;
};

type GNNExplorerProps = {
  apiBaseUrl: string;
  cartoApiKey?: string;
};

// ── Explain mode types ──────────────────────────────────────────────────────

type ScaleRanges = {
  single_component: number;
  variable_total: number;
  total_by_level: number;
  global_total: number;
};

type VariablesMeta = {
  atm_vars: string[];
  surf_vars: string[];
  levels: number[];
  is_ocean?: boolean;
  state_vars?: string[];
  /** Whether the backend can afford an attribution pass for this model. Absent on older
   *  backends, which is treated as available — only an explicit `false` blocks. */
  attribution_available?: boolean;
};

type ModelInfo = {
  model_id?: string;
  name: string;
  path: string;
  key?: string;
  loaded_at?: string;
};

type ExplainModelRecord = {
  model_id: string;
  name: string;
  description?: string | null;
  model_path: string;
  /** Dataset the model runs against ("era5" | "ocean"). Backend defaults it to "era5". */
  domain: string;
  created_at?: string | null;
  updated_at?: string | null;
};

type GridData = {
  lat: number[];
  lon: number[];
  /** Grid step in degrees; heatmap cells are sized from it. */
  grid_spacing_deg?: number;
};

type TimelineData = {
  range_start: string;
  range_end: string;
  timestamps: string[];
  mean_error: number[];
  /** null on artifacts written before these series existed — their chips are disabled. */
  peak_error?: number[] | null;
  error_concentration?: number[] | null;
  month_bands?: MonthBand[];
  /** Subset of `timestamps` the active window can actually run inference for. The RMSE curve
   *  is precomputed over a wider range than the shipped data chunks cover — the ocean image
   *  carries 31 scored dates but only 8 loadable ones — so selection must come from here,
   *  never from `timestamps`, or the user can click a date that 404s on every field fetch. */
  covered_timestamps?: string[];
};

/** Per-timestep top-3 peak components, from /prediction_errors/timeline/components.
 *  Absent (404) for models whose timeline artifact predates peak_top3. */
type ComponentPeaks = {
  timestamps: string[];
  /** Indices into column_names; -1 pads a domain with fewer than three components. */
  peak_top3: number[][];
  column_names: string[];
};

/** Which variables / levels to mark at one timestep, by rank 1..3. */
type PeakMarks = {
  /** variable name -> best (lowest) rank among its components */
  varRank: Record<string, number>;
  /** "variable@level" -> rank */
  levelRank: Record<string, number>;
};

/** Decode one timestep's top-3 into variable and level marks.
 *
 *  Column names carry their own layout: ERA5 atmospheric columns are "var@level",
 *  ERA5 surface variables and every ocean channel ("thetao_47m") are bare names that
 *  have no level. Splitting on "@" is therefore the whole decoding step, and it marks
 *  no level for exactly the columns that have none. */
function decodePeakMarks(peaks: ComponentPeaks | null, t: string | null): PeakMarks {
  const empty: PeakMarks = { varRank: {}, levelRank: {} };
  if (!peaks || !t) return empty;
  const i = peaks.timestamps.indexOf(t);
  if (i < 0) return empty;
  const row = peaks.peak_top3[i];
  if (!row) return empty;

  const marks: PeakMarks = { varRank: {}, levelRank: {} };
  row.forEach((colIdx, k) => {
    if (colIdx < 0 || colIdx >= peaks.column_names.length) return;   // -1 padding
    const rank = k + 1;
    const name = peaks.column_names[colIdx];
    const at = name.indexOf("@");
    const variable = at >= 0 ? name.slice(0, at) : name;
    // A variable keeps its best rank: two of its levels can both be in the top 3.
    if (!(variable in marks.varRank) || rank < marks.varRank[variable]) {
      marks.varRank[variable] = rank;
    }
    if (at >= 0 && !(name in marks.levelRank)) marks.levelRank[name] = rank;
  });
  return marks;
}

type LoadedWindow = {
  range_start: string;
  range_end: string;
  timestamps: string[];
};

type SourceRankingEntry = {
  variable: string;
  level: number | null;
  score: number;
  score_prev_prev: number;
  score_prev: number;
  percentage?: number;
};

type AttributionResponse = {
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
  cache_status?: "mem-hit" | "disk-hit" | "compute" | "mem-miss";
};

type ExSTType = {
  t: string;
  v1: string;
  z1: string | null;
  node: number | null;
  drawerOpen: boolean;
  drawerHeaderOnly: boolean;
  compareMode: "timestamps" | "variables";
  pA_v2: string | null;
  pA_z2: string | null;
  pB_v2: string | null;
  pB_z2: string | null;
  z2Locked: boolean;
  method: "ixg" | "ig";
  heatmapErrors: number[] | null;
  scaleRanges: ScaleRanges | null;
  attributionData: AttributionResponse | null;
  attributionDataB: AttributionResponse | null;
  variablesMeta: VariablesMeta | null;
  modelInfo: ModelInfo | null;
  grid: GridData | null;
  timelineData: TimelineData | null;
  heatmapEverLoaded: boolean;
  sidebarOpen: boolean;
  z1MemoryByVar: Record<string, string>;
  availableTimestamps: Set<string> | null;
};

const LEN_PLOT_COOKIE = "gnn_len_plot";


const saveLenPlotCookie = (): void => {
  const bins = (document.getElementById("len-bins") as HTMLInputElement | null)?.value ?? "15";
  const val = `logx=${ST.lenLogScaleX ? 1 : 0};logy=${ST.lenLogScaleY ? 1 : 0};bins=${bins}`;
  document.cookie = `${LEN_PLOT_COOKIE}=${val}; max-age=31536000; path=/`;
};

const loadLenPlotCookie = (): { logx: boolean; logy: boolean; bins: number } => {
  const entry = document.cookie.split("; ").find((c) => c.startsWith(`${LEN_PLOT_COOKIE}=`));
  if (!entry) return { logx: false, logy: false, bins: 15 };
  const pairs = Object.fromEntries(entry.split("=").slice(1).join("=").split(";").map((p) => p.split("=")));
  return {
    logx: pairs.logx === "1",
    logy: pairs.logy === "1",
    bins: Math.max(2, Math.min(60, parseInt(pairs.bins ?? "15", 10) || 15)),
  };
};

const DARK_MAP_URL = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png";
const DARK_MAP_FILTER = "saturate(60%) brightness(0.88)";
let cartoApiKey = "";

const resolveMapUrl = (url: string): string => {
  if (!cartoApiKey || !url.includes("basemaps.cartocdn.com")) return url;
  return `${url}?key=${encodeURIComponent(cartoApiKey)}`;
};

let API = "http://localhost:8080";
const GRAPH_ID = "1";
const DEFAULT_EXPLAIN_MODEL_ID = "1";
const FETCH_DEBOUNCE_MS = 400;
const SINGLE_SCALE_UPLOAD_FILES = [
  "grid_lat_lon.pt",
  "mesh_lat_lon.pt",
  "g2m_edge_index.pt",
  "g2m_features.pt",
  "m2m_edge_index.pt",
  "m2m_features.pt",
  "m2g_edge_index.pt",
  "m2g_features.pt",
] as const;

const HIERARCHICAL_UPLOAD_FILES = [
  "grid_lat_lon.pt",
  "mesh_lat_lon.pt",
  "g2m_edge_index.pt",
  "m2m_edge_index.pt",
  "m2g_edge_index.pt",
  "mesh_up_edge_index.pt",
  "mesh_down_edge_index.pt",
] as const;

const ANEMOI_UPLOAD_FILES = [
  "graph.pt",
] as const;

// Ocean graphs (Njord / baltic-sea-forecasting) ship no position files at all — the backend
// reconstructs grid_lat_lon/mesh_lat_lon from mesh_features and the g2m/m2g edge vdiff.
// So this schema is defined by what such a folder *does* contain; the absence of
// grid_lat_lon.pt is what identifies it server-side.
const OCEAN_UPLOAD_FILES = [
  "mesh_features.pt",
  "g2m_edge_index.pt",
  "g2m_features.pt",
  "m2g_edge_index.pt",
  "m2g_features.pt",
  "m2m_edge_index.pt",
] as const;

const UPLOAD_FILE_SCHEMAS = [SINGLE_SCALE_UPLOAD_FILES, HIERARCHICAL_UPLOAD_FILES, ANEMOI_UPLOAD_FILES, OCEAN_UPLOAD_FILES] as const;


const MAP_TYPES: Record<string, MapTypeConfig> = {
  map: {
    url: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    opts: { subdomains: "abcd", maxZoom: 19 },
    attr: "&copy; <a href=\"https://osm.org\">OSM</a> &copy; <a href=\"https://carto.com\">CARTO</a>",
    filter: "saturate(35%) brightness(1.04)",
  },
  satellite: {
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    opts: { maxZoom: 19 },
    attr: "&copy; Esri, Maxar, Earthstar Geographics",
    filter: "saturate(60%) brightness(0.93) opacity(72%)",
  },
  terrain: {
    url: "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
    opts: { subdomains: "abc", maxZoom: 17 },
    attr: "&copy; <a href=\"https://opentopomap.org\">OpenTopoMap</a> &copy; OSM",
    filter: "saturate(45%) brightness(1.0) opacity(72%)",
  },
};

let gridNodes: NodeDatum[] = [];
let meshNodes: NodeDatum[] = [];
let gridById: Record<number, NodeDatum> = {};
let meshById: Record<number, NodeDatum> = {};
let edgesStore: Record<SubgraphKey, EdgeDatum[]> = {};
let edgesCache: Record<SubgraphKey, Record<string, EdgeDatum>> = {};
let cachedGridById: Record<number, NodeDatum> = {};
let cachedMeshByKey: Record<string, NodeDatum> = {};
let fetchedBboxes: Array<{ min_lat: number; max_lat: number; min_lon: number; max_lon: number }> = [];
let prevBbox: ReturnType<typeof getBbox> | null = null;
let fetchGen = 0;
let activeGraphId = GRAPH_ID;
let activeExplainModelId = DEFAULT_EXPLAIN_MODEL_ID;

let ST: STType = {
  showGrid: true,
  showMesh: true,
  showG2M: true,
  showM2M: true,
  showM2G: true,
  selected: null,
  violation: false,
  activeInspectSubgraph: "g2m",
  srcDegMin: 0,
  srcDegMax: 30,
  tgtDegMin: 0,
  tgtDegMax: 30,
  lenSliderMax: 600,
  srcDegLogScaleX: false,
  srcDegLogScaleY: false,
  tgtDegLogScaleX: false,
  tgtDegLogScaleY: false,
  lenLogScaleX: false,
  lenLogScaleY: false,
  zoomFade: true,
  activeSubgraphs: new Set(),
  activeNodeTypes: new Set(),
};

const _setMetaLevelsBridge: { set?: (L: number) => void } = {};
const _setMetaNodeCountsBridge: { set?: (nc: Record<string, number>) => void } = {};
const _setActiveInspectSubgraphBridge: { set?: (v: SubgraphKey) => void } = {};
const _setCurrentAppModeBridge: { set?: (m: string) => void } = {};

// ── Explain mode state ──────────────────────────────────────────────────────

const ExST: ExSTType = {
  t: "",
  v1: "total",
  z1: "sum",
  node: null,
  drawerOpen: false,
  drawerHeaderOnly: false,
  compareMode: "timestamps",
  pA_v2: null,
  pA_z2: null,
  pB_v2: null,
  pB_z2: null,
  z2Locked: false,
  method: "ixg",
  heatmapErrors: null,
  scaleRanges: null,
  attributionData: null,
  attributionDataB: null,
  variablesMeta: null,
  modelInfo: null,
  grid: null,
  timelineData: null,
  heatmapEverLoaded: false,
  sidebarOpen: false,
  z1MemoryByVar: {},
  availableTimestamps: new Set<string>(),
};

let currentMode: "explore" | "inspect" | "explain" = "explore";
// Snapshot of the explore-mode panel state, saved whenever handlePanelStateChange runs
// outside inspect mode. Restored when switching back from inspect to explore so that
// renderLayers() sees the correct active sets instead of the stale inspect-mode ones.
let _savedExploreSubgraphs: Set<SubgraphKey> | null = null;
let _savedExploreNodeTypes: Set<NodeType> | null = null;
let explainInitialized = false;
// Set by bindUIHandlers so the useEffect URL restore can call setAppMode without coupling to its closure.
let _setAppMode: ((mode: "explore" | "inspect" | "explain") => void) | null = null;
let _openExplainTourBridge: (() => void) | null = null;
let _openInspectTourBridge: (() => void) | null = null;
// Bridge: assigned from the component useEffect so setAppMode (module-level) can trigger a re-render.
// Object wrapper avoids TypeScript's let-narrowing-to-never across async callback boundaries.
const _explainModelInfoBridge: { set: ((m: ModelInfo | null) => void) | null } = { set: null };
const _explainGridBridge: { set: ((g: GridData | null) => void) | null } = { set: null };
const _explainTimelineBridge: { set: ((d: TimelineData | null) => void) | null } = { set: null };
const _explainTBridge: { set: ((t: string) => void) | null } = { set: null };
const _explainAvailableTimestampsBridge: { set: ((s: Set<string> | null) => void) | null } = { set: null };
const _explainCoastlineBridge: { set: ((g: unknown) => void) | null } = { set: null };
const _explainDrawerBridge: { set: ((open: boolean) => void) | null } = { set: null };
const _explainDrawerHeaderOnlyBridge: { set: ((v: boolean) => void) | null } = { set: null };
const _loadExplainModelRuntimeBridge: { run: ((modelId: string) => Promise<void>) | null } = { run: null };

let savedExploreMapView: { center: L.LatLng; zoom: number } | null = null;

const thresholdCache: Partial<Record<SubgraphKey, PerSubgraphThresholds>> = {};

// Full-graph metric arrays for the active inspect subgraph, as typed arrays decoded from the
// binary /graph_metrics container. Typed rather than number[] because the ocean subgraphs make
// these large — m2g carries 675,219 degrees and 2,011,528 lengths — and every consumer either
// indexes them by node id or runs a threshold query over them.
//
// fullGraphLengths is ASCENDING (the backend sorts it). Do not reorder it: countLE/countGE and
// binCountsSorted rely on that, and it is what keeps slider drags off the O(E) path.
type DegreeArray = Uint16Array | Uint32Array;
let fullGraphSrcDeg: DegreeArray = new Uint16Array(0);
let fullGraphTgtDeg: DegreeArray = new Uint16Array(0);
let fullGraphLengths: Float32Array = new Float32Array(0);
// Scalars the backend reports alongside the arrays, so nothing has to scan for a min or max.
let fullGraphBounds = {
  srcDegMin: 0, srcDegMax: 0,
  tgtDegMin: 0, tgtDegMax: 0,
  lengthMin: 0, lengthMax: 0,
};
let graphMeta: GraphStatsMeta = { n_grid: 0, grid_spacing_deg: 0, n_mesh: 0, n_edges_g2m: 0, n_edges_m2m: 0, n_edges_m2g: 0, is_hierarchical: false, n_levels: 1, node_types: [], node_counts: {}, subgraphs: [] };
let violTotals = { srcLo: 0, srcHi: 0, tgtLo: 0, tgtHi: 0, edgeLo: 0, edgeHi: 0 };
// True only after fetchGraphMetrics has completed for ST.activeInspectSubgraph.
// Guards the early renderLayers() in setViolGraphType from applying stale thresholds.
let violMetricsReady = false;
let graphCatalog: GraphRecord[] = [];

let leafMap: L.Map;
let miniMap: L.Map;
let viewRect: L.Rectangle;
let viewRectWrap: L.Rectangle;
let activeTileLayer: L.TileLayer;
let activeMiniTileLayer: L.TileLayer;
let currentMapType = "map";
let currentThemeMode: ThemeMode = "auto";
let systemDarkMq = null as MediaQueryList | null;

let deckCanvas: HTMLCanvasElement;
let deckInstance: Deck<MapView[]> | null = null;
let deckReady = false;
let isZoomAnimating = false;
// rAF loop that mirrors Leaflet's live zoom transform onto the deck canvas.
let zoomRafId = 0;
// Leaflet proxy scale captured at zoom-animation start; the deck canvas scale
// is the current proxy scale divided by this baseline.
let zoomStartProxyScale = 1;
let fetchTimer: ReturnType<typeof setTimeout> | null = null;

let coastlineGeoJsonClipped: unknown | null = null;
let coastlineFetched = false;

// Plain loop instead of Math.max(...arr): spreading tens of thousands of args hits
// JS engine argument-count limits (JavaScriptCore ~65k), and this array grows with
// grid resolution (29,040 today, 1,038,240 at 0.25° ERA5).
// Skips null/NaN: a null entry means the node has no data for the selected channel, not a
// value of zero, and must not participate in the colorbar's upper bound. `> m` already
// excludes NaN, but null would coerce to 0 and quietly win an all-negative array.
function maxOfArray(arr: (number | null)[]): number {
  let m = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (v !== null && v > m) m = v;
  }
  return m;
}

function clipCoastlineToMercator(geojson: unknown): unknown {
  const MAX_LAT = 85.05;

  function clipLine(coords: number[][]): number[][][] {
    const segments: number[][][] = [];
    let seg: number[][] = [];
    for (const c of coords) {
      if (Math.abs(c[1]) <= MAX_LAT) {
        seg.push(c);
      } else {
        if (seg.length > 1) segments.push(seg);
        seg = [];
      }
    }
    if (seg.length > 1) segments.push(seg);
    return segments;
  }

  const fc = geojson as { type: string; features: { type: string; geometry: { type: string; coordinates: unknown } }[] };
  const newFeatures: unknown[] = [];
  for (const f of fc.features) {
    if (f.geometry.type === "LineString") {
      const segs = clipLine(f.geometry.coordinates as number[][]);
      for (const s of segs)
        newFeatures.push({ ...f, geometry: { type: "LineString", coordinates: s } });
    } else if (f.geometry.type === "MultiLineString") {
      const allSegs: number[][][] = [];
      for (const line of f.geometry.coordinates as number[][][])
        allSegs.push(...clipLine(line));
      if (allSegs.length) newFeatures.push({ ...f, geometry: { type: "MultiLineString", coordinates: allSegs } });
    } else {
      newFeatures.push(f);
    }
  }
  return { ...fc, features: newFeatures };
}

// Attribution fetch AbortControllers — A = Panel A (t-12h), B = Panel B (t-6h).
// Both are aborted on any new node click. In "Compare variables" mode (F5/F6),
// panel-specific V2/z2 changes abort only the corresponding controller.
let attributionAbortA: AbortController | null = null;
let attributionAbortB: AbortController | null = null;

// Recently fetched attribution responses, keyed on the full request URL (which
// encodes pred_time, target node/variable/level, method, and source params — a
// complete cache key). An LRU so flipping between already-visited timestamps (or
// re-selecting a node/source) is instant instead of re-running the integrated-
// gradients round-trip. The URL does NOT encode the active model, so this cache
// is cleared on model switch (see handleExplainModelSelect). Capacity is smaller
// than the field cache because each entry can be larger (many source arrays over
// visible nodes).
const ATTR_CACHE_CAPACITY = 8;
const attrCache = new LRUCache<AttributionResponse>(ATTR_CACHE_CAPACITY);

// Hovered grid-cell index in Explain mode — set by moveHandler, read by rebuildExplainLayers.
let currentExplainHoveredNode: number | null = null;

// Bridge so the module-level map handlers can call the React component's node-click handler.
// Updated on every render (not just mount) so it always holds the latest closure.
const _explainNodeClickBridge: { handler: ((idx: number) => void) | null } = { handler: null };

let GRID_ICON_ATLAS: HTMLCanvasElement | null = null;
const GRID_ICON_MAPPING: Record<string, { x: number; y: number; width: number; height: number; anchorX: number; anchorY: number }> = {};
let ARROW_ICON_ATLAS: HTMLCanvasElement | null = null;
const ARROW_ICON_MAPPING: Record<string, { x: number; y: number; width: number; height: number; anchorX: number; anchorY: number }> = {};
const SPRITE_SIZE = 256;
const NODE_BORDER_DIVISOR = 24; // lower = thicker border, higher = thinner
const NODE_ICON_ALPHA = 220;     // shared opacity for both grid and mesh icon layers (0-255)
// Node size is expressed in world units, so a fixed value that reads well on one grid
// blankets a denser one: 60 km is about a third of the gap on the 1.5° ERA5 grid, but twice
// the gap on a 0.25° ocean grid, so neighbours merge into a solid sheet. Size therefore
// scales with the graph's own grid spacing, reported by the backend as grid_spacing_deg.
//
// Calibration: at REFERENCE_SPACING_DEG the result is exactly NODE_SIZE_M, so the ERA5
// graphs render identically to before this became dynamic.
const NODE_SIZE_M = 60000;
const REFERENCE_SPACING_DEG = 1.5;   // ERA5 240×121 grid — the size above was tuned for it
const nodeSizeM = (): number =>
  NODE_SIZE_M * ((graphMeta.grid_spacing_deg || REFERENCE_SPACING_DEG) / REFERENCE_SPACING_DEG);
const NODE_MAX_PX = 18 * (typeof window !== "undefined" ? window.devicePixelRatio : 1);
const EQUAL_AREA_NODES = true; // scale mesh circle up by 2/√π so it matches square visual weight
const MESH_SIZE_SCALE = EQUAL_AREA_NODES ? 2 / Math.sqrt(Math.PI) : 1.0;
const MERC_MAX = 85.05113;

// ── Explain mode constants ──────────────────────────────────────────────────
const ATM_VARS = ["geopotential", "specific_humidity", "temperature", "u_component_of_wind", "v_component_of_wind", "vertical_velocity"] as const;
const SURF_VARS = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure", "total_precipitation_6hr"] as const;
const PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000] as const;

// ── Saved xai configuration chips — a "flagged" (target v1@z1, node, t) snapshot
// the user can jump back to. In-memory only; FIFO-evicted beyond this cap.
const MAX_SAVED_EXPLAIN_CONFIGS = 6;
type SavedExplainConfig = {
  id: string;
  v1: string;
  z1: string | null;
  node: number;
  t: string;
  label: string;
};
const EXPLAIN_VAR_SHORT: Record<string, string> = {
  geopotential: "Geopot.", specific_humidity: "Sp.Hum.", temperature: "Temp.",
  u_component_of_wind: "U-Wind", v_component_of_wind: "V-Wind", vertical_velocity: "Vert.Vel.",
  "2m_temperature": "2m Temp", "10m_u_component_of_wind": "10m U", "10m_v_component_of_wind": "10m V",
  mean_sea_level_pressure: "MSLP", total_precipitation_6hr: "Precip.",
};
const explainVarShort = (v: string): string => EXPLAIN_VAR_SHORT[v] ?? v.replace(/_/g, " ");
const formatExplainCoord = (val: number, isLat: boolean): string => {
  const abs = Math.abs(val).toFixed(1);
  const dir = isLat ? (val >= 0 ? "N" : "S") : (val >= 0 ? "E" : "W");
  return `${abs}${dir}`;
};
const formatExplainConfigTime = (iso: string): string => {
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" }) +
    " " + d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", hour12: false, timeZone: "UTC" }) + "z";
};
const buildSavedExplainLabel = (
  v1: string, z1: string | null, node: number, t: string, grid: { lat: number[]; lon: number[] } | null,
): string => {
  const varPart = (SURF_VARS as readonly string[]).includes(v1) || z1 === null
    ? explainVarShort(v1)
    : `${explainVarShort(v1)}@${z1}`;
  const lat = grid?.lat[node];
  const lon = grid?.lon[node];
  const nodePart = lat !== undefined && lon !== undefined
    ? `${formatExplainCoord(lat, true)},${formatExplainCoord(lon, false)}`
    : `n${node}`;
  return `${varPart} · ${nodePart} · ${formatExplainConfigTime(t)}`;
};
const N_GRID = 29040; // TODO: fetch this from backend route later on
const VALID_EXPLAIN_VARS = new Set<string>([...ATM_VARS, ...SURF_VARS, "total"]);
const VALID_COMPARE_MODES = new Set(["timestamps", "variables"]);
const VALID_METHODS = new Set(["ixg", "ig"]);

// Edge alpha: 0–255. Adjust these to taste.
const EDGE_ALPHA_NORMAL = 80;      // normal view — all edges
const EDGE_ALPHA_VIOL_NEUTRAL = 45;   // violation mode — non-flagged (grey) edges

// Scale fade: edges shorter than (viewport width × THRESHOLD_FRAC) fade toward FLOOR × their normal alpha.
// Raise THRESHOLD_FRAC to start fading later (less aggressive). Lower FLOOR toward 0 for stronger fade.
// POWER > 1 makes short edges fade much harder while barely affecting edges near the threshold.
const ZOOM_FADE_THRESHOLD_FRAC = 0.10;  // edge must span this fraction of viewport to be fully opaque
const ZOOM_FADE_MAX_ALPHA = 140;        // alpha for edges at/above threshold (decoupled from EDGE_ALPHA_NORMAL)
const ZOOM_FADE_FLOOR = 0.10;          // minimum alpha multiplier (× ZOOM_FADE_MAX_ALPHA) for shortest edges
const ZOOM_FADE_POWER = 1;             // 1 = linear, 2 = quadratic, 3 = cubic

const getEl = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element with id '${id}'`);
  return el as T;
};

const hexToRgba = (hex: string, a?: number): [number, number, number, number] => {
  const r = Number.parseInt(hex.slice(1, 3), 16);
  const g = Number.parseInt(hex.slice(3, 5), 16);
  const b = Number.parseInt(hex.slice(5, 7), 16);
  return [r, g, b, a ?? 255];
};

const hexToGrayRgba = (hex: string, a: number): [number, number, number, number] => {
  const [r, g, b] = hexToRgba(hex);
  const lum = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
  return [lum, lum, lum, a];
};

// ── Explain mode helpers ────────────────────────────────────────────────────

const isSurfaceVar = (varName: string): boolean =>
  (SURF_VARS as readonly string[]).includes(varName);

// Returns true if ExST.attributionData matches the current ExST target selection.
// Used by handleOpenAttribution to avoid redundant re-fetches.
const isAttrCurrent = (): boolean => {
  const d = ExST.attributionData;
  if (!d) return false;
  const z1Level = ExST.z1 !== null && ExST.z1 !== "sum" ? Number(ExST.z1) : null;
  const methodMatches = ExST.method === "ixg" ? d.method === "ixg" : d.method.startsWith("ig_k");
  return (
    d.pred_time === ExST.t &&
    d.target.node === ExST.node &&
    d.target.variable === ExST.v1 &&
    d.target.level === z1Level &&
    methodMatches
  );
};

// Returns true if ExST.attributionDataB matches the current ExST target selection (Variables mode Panel B).
const isAttrBCurrent = (): boolean => {
  const d = ExST.attributionDataB;
  if (!d) return false;
  const z1Level = ExST.z1 !== null && ExST.z1 !== "sum" ? Number(ExST.z1) : null;
  const methodMatches = ExST.method === "ixg" ? d.method === "ixg" : d.method.startsWith("ig_k");
  return (
    d.pred_time === ExST.t &&
    d.target.node === ExST.node &&
    d.target.variable === ExST.v1 &&
    d.target.level === z1Level &&
    methodMatches
  );
};

let _toastShown = false;
const showToast = (message: string, oneTime = false): void => {
  if (oneTime && _toastShown) return;
  if (oneTime) _toastShown = true;
  const existing = document.getElementById("explain-toast");
  if (existing) existing.remove();
  const el = document.createElement("div");
  el.id = "explain-toast";
  el.textContent = message;
  el.className = "explain-toast";
  document.getElementById("app")?.appendChild(el);
  // trigger reflow so the fade-out transition fires
  el.getBoundingClientRect();
  el.classList.add("explain-toast--visible");
  setTimeout(() => {
    el.classList.remove("explain-toast--visible");
    el.addEventListener("transitionend", () => el.remove(), { once: true });
  }, 3000);
};

// ── Explain URL serialization ───────────────────────────────────────────────

const encodeExplainState = (): URLSearchParams => {
  const p = new URLSearchParams();
  p.set("mode", "explain");
  if (ExST.t) p.set("t", ExST.t);
  p.set("v1", ExST.v1);
  // z1: omit entirely when v1 is a surface variable
  if (!isSurfaceVar(ExST.v1) && ExST.z1 !== null) p.set("z1", ExST.z1);
  if (ExST.node !== null) p.set("node", String(ExST.node));
  p.set("compare", ExST.compareMode);
  if (ExST.pA_v2) {
    p.set("pA_v2", ExST.pA_v2);
    // pA_z2: omit when pA_v2 is surface
    if (!isSurfaceVar(ExST.pA_v2) && ExST.pA_z2 !== null) p.set("pA_z2", ExST.pA_z2);
  }
  if (ExST.pB_v2) {
    p.set("pB_v2", ExST.pB_v2);
    // pB_z2: omit when pB_v2 is surface
    if (!isSurfaceVar(ExST.pB_v2) && ExST.pB_z2 !== null) p.set("pB_z2", ExST.pB_z2);
  }
  p.set("z2_locked", ExST.z2Locked ? "1" : "0");
  p.set("method", ExST.method);
  if (ExST.drawerHeaderOnly) p.set("drawer", "header");
  return p;
};

type ExplainDecodeResult = {
  t: string;
  v1: string;
  z1: string | null;
  node: number | null;
  compareMode: "timestamps" | "variables";
  pA_v2: string | null;
  pA_z2: string | null;
  pB_v2: string | null;
  pB_z2: string | null;
  z2Locked: boolean;
  method: "ixg" | "ig";
  drawerHeaderOnly: boolean;
  hadInvalidParams: boolean;
};

const decodeExplainState = (params: URLSearchParams): ExplainDecodeResult => {
  let hadInvalidParams = false;

  const rawV1 = params.get("v1") ?? "total";
  const v1 = VALID_EXPLAIN_VARS.has(rawV1) ? rawV1 : (() => { hadInvalidParams = true; return "total"; })();

  // z1: ignored for surface variables; must be a valid hPa string, "sum", or absent
  let z1: string | null = null;
  if (!isSurfaceVar(v1)) {
    const rawZ1 = params.get("z1");
    if (rawZ1 === null) {
      z1 = "500";
    } else if (rawZ1 === "sum") {
      z1 = "sum";
    } else {
      const lvl = Number(rawZ1);
      z1 = (PRESSURE_LEVELS as readonly number[]).includes(lvl)
        ? rawZ1
        : (() => { hadInvalidParams = true; return "500"; })();
    }
  }

  const rawNode = params.get("node");
  let node: number | null = null;
  if (rawNode !== null) {
    const n = Number.parseInt(rawNode, 10);
    node = Number.isFinite(n) && n >= 0 && n < N_GRID ? n : (() => { hadInvalidParams = true; return null; })();
  }

  const rawCompare = params.get("compare") ?? "timestamps";
  const compareMode = VALID_COMPARE_MODES.has(rawCompare)
    ? (rawCompare as "timestamps" | "variables")
    : (() => { hadInvalidParams = true; return "timestamps" as const; })();

  const rawMethod = params.get("method") ?? "ixg";
  const method = VALID_METHODS.has(rawMethod)
    ? (rawMethod as "ixg" | "ig")
    : (() => { hadInvalidParams = true; return "ixg" as const; })();

  const decodeSourceVar = (key: string): string | null => {
    const raw = params.get(key);
    if (!raw) return null;
    // source vars can't be "total" — only specific variables
    const valid = VALID_EXPLAIN_VARS.has(raw) && raw !== "total";
    if (!valid) { hadInvalidParams = true; return null; }
    return raw;
  };

  const decodeSourceLevel = (varName: string | null, key: string): string | null => {
    if (!varName || isSurfaceVar(varName)) return null;
    const raw = params.get(key);
    if (raw === null) return null;
    if (raw === "sum") return "sum";
    const lvl = Number(raw);
    if ((PRESSURE_LEVELS as readonly number[]).includes(lvl)) return raw;
    hadInvalidParams = true;
    return null;
  };

  const pA_v2 = decodeSourceVar("pA_v2");
  const pA_z2 = decodeSourceLevel(pA_v2, "pA_z2");
  const pB_v2 = decodeSourceVar("pB_v2");
  const pB_z2 = decodeSourceLevel(pB_v2, "pB_z2");

  const rawZ2Locked = params.get("z2_locked");
  const z2Locked = rawZ2Locked === "1";

  const drawerHeaderOnly = params.get("drawer") === "header";

  const t = params.get("t") ?? "";

  return { t, v1, z1, node, compareMode, pA_v2, pA_z2, pB_v2, pB_z2, z2Locked, method, drawerHeaderOnly, hadInvalidParams };
};

// Called on settle events (mouseup, change) only — never on continuous input drag.
const writeExplainUrl = (): void => {
  if (currentMode !== "explain") return;
  const params = encodeExplainState();
  history.replaceState(null, "", "?" + params.toString());
};



const buildIcons = (): void => {

  // Build grid icon atlas on a canvas — synchronous, no async SVG loading.
  // deck.gl accepts HTMLCanvasElement directly as iconAtlas.
  const s = SPRITE_SIZE;
  const sw = Math.round(s / NODE_BORDER_DIVISOR);
  const gridIcons = [
    { name: "grid_normal", fill: NODE_GRID,     stroke: TEXT },
    { name: "grid_grey",   fill: VIOL_NEUTRAL,  stroke: TEXT },
    { name: "grid_hi",     fill: VIOL_ABOVE,    stroke: TEXT },
    { name: "grid_lo",     fill: VIOL_BELOW,    stroke: TEXT },
  ];
  const canvas = document.createElement("canvas");
  canvas.width = s * gridIcons.length;
  canvas.height = s;
  const ctx = canvas.getContext("2d")!;
  gridIcons.forEach(({ name, fill, stroke }, i) => {
    const x = i * s;
    ctx.fillStyle = fill;
    ctx.fillRect(x + sw, sw, s - sw * 2, s - sw * 2);
    ctx.strokeStyle = stroke;
    ctx.lineWidth = sw;
    ctx.strokeRect(x + sw / 2, sw / 2, s - sw, s - sw);
    GRID_ICON_MAPPING[name] = { x, y: 0, width: s, height: s, anchorX: s / 2, anchorY: s / 2 };
  });
  GRID_ICON_ATLAS = canvas;

  // Arrow icon atlas — same synchronous canvas approach.
  // anchorX is SPRITE_SIZE (tip at right edge) to place the arrowhead at the edge endpoint.
  const arrowIcons = [
    { name: "arrow_g2m",  fill: EDGE_ENCODER },
    { name: "arrow_m2g",  fill: EDGE_DECODER },
    { name: "arrow_m2m",  fill: EDGE_MESH },
    { name: "arrow_grey", fill: VIOL_NEUTRAL },
    { name: "arrow_hi",   fill: VIOL_ABOVE },
    { name: "arrow_lo",   fill: VIOL_BELOW },
  ];
  const aCanvas = document.createElement("canvas");
  aCanvas.width = s * arrowIcons.length;
  aCanvas.height = s;
  const aCtx = aCanvas.getContext("2d")!;
  arrowIcons.forEach(({ name, fill }, i) => {
    const x = i * s;
    aCtx.beginPath();
    aCtx.moveTo(x + s, s / 2);
    aCtx.lineTo(x, 0);
    aCtx.lineTo(x + Math.round(s * 0.28), s / 2);
    aCtx.lineTo(x, s);
    aCtx.closePath();
    aCtx.fillStyle = fill;
    aCtx.fill();
    ARROW_ICON_MAPPING[name] = { x, y: 0, width: s, height: s, anchorX: s, anchorY: s / 2 };
  });
  ARROW_ICON_ATLAS = aCanvas;
};

const clampLat = (lat: number): number => (lat > MERC_MAX ? MERC_MAX : lat < -MERC_MAX ? -MERC_MAX : lat);

// Mercator Y in "degrees" (same unit as longitude, so [lon, mercYdeg] is an isotropic space)
const mercYdeg = (lat: number): number =>
  Math.log(Math.tan(Math.PI / 4 + clampLat(lat) * Math.PI / 360)) * (180 / Math.PI);
const mercYdegInv = (my: number): number =>
  (Math.atan(Math.exp(my * Math.PI / 180)) - Math.PI / 4) * (360 / Math.PI);

const squareBorderDist = (angleDeg: number): number => {
  const h = 3.5;
  const rad = (angleDeg * Math.PI) / 180;
  const cx = Math.abs(Math.cos(rad));
  const cy = Math.abs(Math.sin(rad));
  return Math.min(cx > 1e-10 ? h / cx : 1e9, cy > 1e-10 ? h / cy : 1e9);
};

// All geometry done in Mercator [lon, mercYdeg] space, which is isotropic (equal scale in x/y).
// Node has getSize = NODE_SIZE_M * cos(lat) meters. In Mercator space that cancels to NODE_SIZE_M / 111320
// degrees — constant at all latitudes. So the border offset is also a constant.
const nodeBorderMerc = (): number => nodeSizeM() / 2 / 111320;
const adjustedEndpoint = (srcLonLat: [number, number], tgtLonLat: [number, number], tgtIsSquare: boolean, tgtRadiusScale = 1.0): [number, number] => {
  const sx = srcLonLat[0];
  const sy = mercYdeg(srcLonLat[1]);
  const tx = tgtLonLat[0];
  const ty = mercYdeg(tgtLonLat[1]);
  const dx = tx - sx;
  const dy = ty - sy;
  const angleDeg = (Math.atan2(dy, dx) * 180) / Math.PI;
  const borderMerc = tgtIsSquare
    ? (squareBorderDist(angleDeg + 180) / 3.5) * nodeBorderMerc()
    : nodeBorderMerc() * tgtRadiusScale;
  const len = Math.hypot(dx, dy);
  // 2×borderMerc: below this the arrowhead can't fit between src and tgt boundary
  if (len <= 2 * borderMerc) return tgtLonLat;
  const scale = (len - borderMerc) / len;
  return [sx + dx * scale, mercYdegInv(sy + dy * scale)];
};

const getBbox = (): { min_lat: number; max_lat: number; min_lon: number; max_lon: number } => {
  const b = leafMap.getBounds();
  const min_lat = b.getSouth() <= -(MERC_MAX - 1) ? -90 : b.getSouth();
  const max_lat = b.getNorth() >= (MERC_MAX - 1) ? 90 : b.getNorth();
  return { min_lat, max_lat, min_lon: b.getWest(), max_lon: b.getEast() };
};

const bboxParams = (
  bbox: ReturnType<typeof getBbox>,
  prev: ReturnType<typeof getBbox> | null,
): URLSearchParams => {
  const p: Record<string, number> = {
    min_lat: bbox.min_lat,
    max_lat: bbox.max_lat,
    min_lon: bbox.min_lon,
    max_lon: bbox.max_lon,
  };
  if (prev) {
    p.prev_min_lat = prev.min_lat;
    p.prev_max_lat = prev.max_lat;
    p.prev_min_lon = prev.min_lon;
    p.prev_max_lon = prev.max_lon;
  }
  const stringParams: Record<string, string> = {};
  Object.entries(p).forEach(([k, v]) => { stringParams[k] = String(v); });
  return new URLSearchParams(stringParams);
};

// Wrap-aware lon predicate — mirrors backend services/bbox_torch.py:_lon_mask.
// Three branches: width ≥ 360 → all pass; shifted single segment; antimeridian-crossing.
const lonInRange = (lon: number, minLon: number, maxLon: number): boolean => {
  if (maxLon - minLon >= 360) return true;
  const shift = Math.floor((minLon + 180) / 360) * 360;
  const lo = minLon - shift;
  const hi = maxLon - shift;
  if (hi <= 180) return lon >= lo && lon <= hi;
  return lon >= lo || lon <= hi - 360;
};

const inBbox = (node: { lat: number; lon: number }, bbox: ReturnType<typeof getBbox>): boolean =>
  node.lat >= bbox.min_lat && node.lat <= bbox.max_lat && lonInRange(node.lon, bbox.min_lon, bbox.max_lon);

const edgeKey = (e: EdgeDatum): string => `${e.source}:${e.target}`;

const lonSegments = (minLon: number, maxLon: number): Array<[number, number]> => {
  if (minLon <= maxLon) return [[minLon, maxLon]];
  return [[minLon, 180], [-180, maxLon]];
};

const lonSegmentContains = (outer: [number, number], inner: [number, number]): boolean =>
  outer[0] <= inner[0] && outer[1] >= inner[1];

const lonRangeContains = (outerMin: number, outerMax: number, innerMin: number, innerMax: number): boolean => {
  const outerSegs = lonSegments(outerMin, outerMax);
  const innerSegs = lonSegments(innerMin, innerMax);
  return innerSegs.every((seg) => outerSegs.some((outer) => lonSegmentContains(outer, seg)));
};

const bboxContains = (
  outer: { min_lat: number; max_lat: number; min_lon: number; max_lon: number },
  inner: { min_lat: number; max_lat: number; min_lon: number; max_lon: number },
): boolean =>
  outer.min_lat <= inner.min_lat &&
  outer.max_lat >= inner.max_lat &&
  lonRangeContains(outer.min_lon, outer.max_lon, inner.min_lon, inner.max_lon);

const isBboxCoveredByCache = (bbox: ReturnType<typeof getBbox>): boolean =>
  fetchedBboxes.some((cached) => bboxContains(cached, bbox));

// Bumped whenever the node/edge caches are written or cleared. Pairs with the bbox below so a
// rescan is skipped only when neither the viewport nor the cache contents have moved.
let cacheEpoch = 0;
const bumpCacheEpoch = (): void => { cacheEpoch++; };
let _lastRefresh: { bbox: ReturnType<typeof getBbox>; epoch: number } | null = null;

const sameBbox = (
  a: ReturnType<typeof getBbox>, b: ReturnType<typeof getBbox>,
): boolean =>
  a.min_lat === b.min_lat && a.max_lat === b.max_lat &&
  a.min_lon === b.min_lon && a.max_lon === b.max_lon;

// Materialises the visible node/edge sets out of the full caches.
//
// This is O(everything cached), not O(visible): Object.values over each cache allocates an
// array of every entry before filtering it. Because every edge must remain visible at every
// zoom, those caches converge on the whole graph — 675k grid nodes and 2M m2g edges — so this
// is the most expensive thing on the pan path, and it is on the "cache hit" fast path too.
//
// It cannot be made cheaper without a spatial index (see LARGE_GRAPH_PERFORMANCE.md), but it
// can be skipped: the result is a pure function of (bbox, cache contents), so a repeat call
// with both unchanged returns the arrays already in hand. That also preserves array identity,
// which is what keeps the longestFirst and computeDegrees memos warm across redraws.
const refreshVisibleDataFromCache = (bbox: ReturnType<typeof getBbox>): void => {
  if (_lastRefresh && _lastRefresh.epoch === cacheEpoch && sameBbox(_lastRefresh.bbox, bbox)) return;
  _lastRefresh = { bbox, epoch: cacheEpoch };
  visibleDataToken++;
  gridNodes = Object.values(cachedGridById).filter((n) => inBbox(n, bbox));
  meshNodes = Object.values(cachedMeshByKey).filter((n) => inBbox(n, bbox));

  gridById = {};
  meshById = {};
  gridNodes.forEach((n) => { gridById[n.id] = n; });
  meshNodes.forEach((n) => { meshById[n.id] = n; });

  for (const key of Object.keys(edgesCache)) {
    edgesStore[key] = Object.values(edgesCache[key]).filter((e) => inBbox(e.src, bbox) || inBbox(e.tgt, bbox));
  }
};

const haversineM = (aLat: number, aLon: number, bLat: number, bLon: number): number => {
  const R = 6371000;
  const toRad = (d: number) => (d * Math.PI) / 180;
  const dφ = toRad(bLat - aLat);
  const dλ = toRad(bLon - aLon);
  const s = Math.sin(dφ / 2) ** 2 + Math.cos(toRad(aLat)) * Math.cos(toRad(bLat)) * Math.sin(dλ / 2) ** 2;
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(s)));
};

const buildEdges = (apiEdges: ApiEdge[]): EdgeDatum[] =>
  apiEdges.map((e) => ({
    src: { id: e.source, lat: e.src_lat, lon: e.src_lon },
    tgt: { id: e.target, lat: e.tgt_lat, lon: e.tgt_lon },
    source: e.source,
    target: e.target,
    len: haversineM(e.src_lat, e.src_lon, e.tgt_lat, e.tgt_lon),
  }));

// Longest-first draw order for M2M edges, memoized per subgraph.
//
// The order only matters so that long edges are drawn under short ones, and it depends solely
// on the edge set — but it sat inside buildLayers, which reruns on every pan, zoom, selection
// and slider move. On the ocean graph's 166k-edge m2m_0 that was a copy plus a comparator
// sort of 166k objects per render. Keyed on the array identity: refreshVisibleDataFromCache
// assigns a fresh array whenever the visible set changes, so identity is exactly the right
// invalidation signal.
const _sortedEdgeCache = new Map<string, { src: EdgeDatum[]; sorted: EdgeDatum[] }>();

const longestFirst = (key: string, edges: EdgeDatum[]): EdgeDatum[] => {
  const hit = _sortedEdgeCache.get(key);
  if (hit && hit.src === edges) return hit.sorted;
  const sorted = [...edges].sort((a, b) => b.len - a.len);
  _sortedEdgeCache.set(key, { src: edges, sorted });
  return sorted;
};

const getMeshNeighborIds = (selectedId: number): Set<number> => {
  const neighbors = new Set<number>();
  for (const key of Object.keys(edgesStore)) {
    if (key !== "m2m" && !key.startsWith("m2m_")) continue;
    for (const edge of edgesStore[key]) {
      if (edge.source === selectedId) neighbors.add(edge.target);
      else if (edge.target === selectedId) neighbors.add(edge.source);
    }
  }
  return neighbors;
};

const getConnectedGridNodeIds = (selectedId: number): Set<number> => {
  const connected = new Set<number>();
  for (const edge of edgesStore["g2m"] ?? []) {
    if (edge.target === selectedId) connected.add(edge.source);
  }
  for (const edge of edgesStore["m2g"] ?? []) {
    if (edge.source === selectedId) connected.add(edge.target);
  }
  return connected;
};

const getConnectedMeshNodeIds = (selectedId: number): Set<number> => {
  const connected = new Set<number>();
  for (const edge of edgesStore["g2m"] ?? []) {
    if (edge.source === selectedId) connected.add(edge.target);
  }
  for (const edge of edgesStore["m2g"] ?? []) {
    if (edge.target === selectedId) connected.add(edge.source);
  }
  return connected;
};

const setLoading = (on: boolean): void => {
  getEl<HTMLDivElement>("loading-badge").classList.toggle("active", on);
};

// eslint-disable-next-line @typescript-eslint/no-unused-vars
const setLoadingText = (_text: string): void => { /* spinner-only — no text */ };

// Degree map over the visible edges, memoized.
//
// Rebuilt by walking every edge in every active subgraph, and called from showNodeTooltip and
// updateInfoPanel — so once per hover and once per selection. With the ocean graph that is a
// multi-hundred-thousand-edge walk per mouse move. The result depends only on the visible edge
// sets and the active subgraph selection, both of which change far less often than the cursor.
let _degreeCache: { token: number; subgraphs: string; deg: Record<number, number> } | null = null;
// Bumped by refreshVisibleDataFromCache — the single place the visible edge/node sets change.
let visibleDataToken = 0;

const computeDegrees = (): Record<number, number> => {
  const subgraphs = [...ST.activeSubgraphs].sort().join(",");
  if (_degreeCache && _degreeCache.token === visibleDataToken && _degreeCache.subgraphs === subgraphs) {
    return _degreeCache.deg;
  }
  const deg = computeDegreesUncached();
  _degreeCache = { token: visibleDataToken, subgraphs, deg };
  return deg;
};

const computeDegreesUncached = (): Record<number, number> => {
  const deg: Record<number, number> = {};
  gridNodes.forEach((n) => { deg[n.id] = 0; });
  meshNodes.forEach((n) => { deg[n.id] = 0; });
  const inc = (id: number): void => { if (id in deg) deg[id] += 1; };
  for (const key of Object.keys(edgesStore)) {
    if (!ST.activeSubgraphs.has(key)) continue;
    // Symmetric subgraphs (e.g. M2M) store both A→B and B→A. Only count the
    // source side so each node is counted once per unique neighbor, not twice.
    const sg = graphMeta.subgraphs.find((s) => s.key === key);
    const isSymmetric = sg ? sg.src === sg.tgt : false;
    edgesStore[key].forEach((e) => {
      inc(e.source);
      if (!isSymmetric) inc(e.target);
    });
  }
  return deg;
};

const resizeDeck = (): void => {
  const sz = leafMap.getSize();
  const w = sz.x;
  const h = sz.y;
  deckCanvas.style.width = `${w}px`;
  deckCanvas.style.height = `${h}px`;
  if (deckInstance) deckInstance.setProps({ width: w, height: h });
};

const inferGraphNameFromFiles = (files: File[]): string => {
  const first = files.find((file) => file.webkitRelativePath) ?? files[0];
  const rel = first?.webkitRelativePath ?? "";
  if (rel.includes("/")) {
    return rel.split("/")[0] || "uploaded_graph";
  }
  return first?.name ? first.name.replace(/\.pt$/i, "") : "uploaded_graph";
};

const readUploadError = async (response: Response): Promise<string> => {
  const body = await response.text().catch(() => "");
  if (!body) return `Request failed with status ${response.status}`;
  try {
    const payload = JSON.parse(body) as { detail?: unknown };
    if (typeof payload.detail === "string") return payload.detail;
    if (Array.isArray(payload.detail)) return payload.detail.join(", ");
    if (payload.detail != null) return JSON.stringify(payload.detail);
  } catch {
    // Use the raw body text when the response is not JSON.
  }
  return body;
};

const normalizeGraphRecord = (raw: unknown): GraphRecord | null => {
  if (!raw || typeof raw !== "object") return null;
  const item = raw as Record<string, unknown>;
  const graphIdValue = item.graph_id;
  const graphId =
    typeof graphIdValue === "string"
      ? graphIdValue
      : typeof graphIdValue === "number" || typeof graphIdValue === "bigint"
        ? String(graphIdValue)
        : "";
  if (!graphId) return null;

  return {
    graph_id: graphId,
    name: typeof item.name === "string" && item.name.trim() ? item.name : graphId,
    description: typeof item.description === "string" ? item.description : null,
    folder_path: typeof item.folder_path === "string" ? item.folder_path : "",
    created_at: typeof item.created_at === "string" ? item.created_at : null,
    updated_at: typeof item.updated_at === "string" ? item.updated_at : null,
  };
};

const normalizeExplainModelRecord = (raw: unknown): ExplainModelRecord | null => {
  if (!raw || typeof raw !== "object") return null;
  const item = raw as Record<string, unknown>;
  const modelIdValue = item.model_id;
  const modelId =
    typeof modelIdValue === "string"
      ? modelIdValue
      : typeof modelIdValue === "number" || typeof modelIdValue === "bigint"
        ? String(modelIdValue)
        : "";
  if (!modelId) return null;

  return {
    model_id: modelId,
    name: typeof item.name === "string" && item.name.trim() ? item.name : modelId,
    description: typeof item.description === "string" ? item.description : null,
    model_path: typeof item.model_path === "string" ? item.model_path : "",
    // Older backends do not send a domain; treat those as era5, which is what they served.
    domain: typeof item.domain === "string" && item.domain ? item.domain : "era5",
    created_at: typeof item.created_at === "string" ? item.created_at : null,
    updated_at: typeof item.updated_at === "string" ? item.updated_at : null,
  };
};

const fetchGraphCatalog = async (): Promise<GraphRecord[]> => {
  const response = await fetch(`${API}/graphs`);
  if (!response.ok) {
    throw new Error(await readUploadError(response));
  }

  const payload = (await response.json()) as unknown;
  const rows = Array.isArray(payload)
    ? payload
    : payload && typeof payload === "object" && Array.isArray((payload as { graphs?: unknown[] }).graphs)
      ? (payload as { graphs: unknown[] }).graphs
      : [];

  return rows
    .map((item) => normalizeGraphRecord(item))
    .filter((item): item is GraphRecord => item != null)
    .sort((a, b) => {
      const aT = Date.parse(a.updated_at ?? a.created_at ?? "") || 0;
      const bT = Date.parse(b.updated_at ?? b.created_at ?? "") || 0;
      if (aT !== bT) return bT - aT;
      return a.name.localeCompare(b.name);
    });
};

const fetchModelCatalog = async (): Promise<ExplainModelRecord[]> => {
  const response = await fetch(`${API}/models`);
  if (!response.ok) {
    throw new Error(await readUploadError(response));
  }

  const payload = (await response.json()) as unknown;
  const rows = Array.isArray(payload)
    ? payload
    : payload && typeof payload === "object" && Array.isArray((payload as { models?: unknown[] }).models)
      ? (payload as { models: unknown[] }).models
      : [];

  return rows
    .map((item) => normalizeExplainModelRecord(item))
    .filter((item): item is ExplainModelRecord => item != null)
    .sort((a, b) => {
      const aT = Date.parse(a.updated_at ?? a.created_at ?? "") || 0;
      const bT = Date.parse(b.updated_at ?? b.created_at ?? "") || 0;
      if (aT !== bT) return bT - aT;
      return a.name.localeCompare(b.name);
    });
};

const resetGraphData = (): void => {
  gridNodes = [];
  meshNodes = [];
  gridById = {};
  meshById = {};
  edgesStore = {};
  edgesCache = {};
  cachedGridById = {};
  cachedMeshByKey = {};
  fetchedBboxes = [];
  bumpCacheEpoch();
  prevBbox = null;
  fetchGen = 0;
  if (fetchTimer) {
    clearTimeout(fetchTimer);
    fetchTimer = null;
  }

  fullGraphSrcDeg = new Uint16Array(0);
  fullGraphTgtDeg = new Uint16Array(0);
  fullGraphLengths = new Float32Array(0);
  fullGraphBounds = { srcDegMin: 0, srcDegMax: 0, tgtDegMin: 0, tgtDegMax: 0, lengthMin: 0, lengthMax: 0 };
  derivedMetrics = null;
  graphMeta = { n_grid: 0, grid_spacing_deg: 0, n_mesh: 0, n_edges_g2m: 0, n_edges_m2m: 0, n_edges_m2g: 0, is_hierarchical: false, n_levels: 1, node_types: [], node_counts: {}, subgraphs: [] };
  violTotals = { srcLo: 0, srcHi: 0, tgtLo: 0, tgtHi: 0, edgeLo: 0, edgeHi: 0 };
  violMetricsReady = false;
  // Thresholds belong to the graph they were set on — drop the in-memory copies so the
  // next graph hydrates from its own persisted entry (or from its metric bounds).
  for (const k of Object.keys(thresholdCache)) delete thresholdCache[k as SubgraphKey];
  ST.selected = null;
};

const reloadActiveGraphData = async (): Promise<void> => {
  resetGraphData();
  renderLayers();
  updateGraphStats();
  updateHeaderCounts();
  updateInfoPanel();

  setLoadingText("Loading...");
  await fetchGraphMeta();
  // After loading new graph meta, ensure activeInspectSubgraph is valid for this graph.
  // A stale key from the previous graph would cause the violations endpoint to 404.
  const validKeys = new Set(graphMeta.subgraphs.map((s) => s.key));
  if (!validKeys.has(ST.activeInspectSubgraph)) {
    ST.activeInspectSubgraph = graphMeta.subgraphs.find((s) => s.edge === "g2m")?.key ?? graphMeta.subgraphs[0]?.key ?? "g2m";
    // Keep the sidebar's controlled value in sync — it drives the radio selection on inspect entry.
    _setActiveInspectSubgraphBridge.set?.(ST.activeInspectSubgraph);
  }
  if (ST.violation) {
    await fetchGraphMetrics(ST.activeInspectSubgraph);
  }
  await fetchAndRender();
};

const uploadGraphFolder = async (files: File[]): Promise<GraphRecord> => {
  // Separate folder uploads (webkitRelativePath) from direct file uploads
  const folderFiles = files.filter((file) => {
    const isPt = file.name.toLowerCase().endsWith(".pt");
    const isRootFile = !file.webkitRelativePath || file.webkitRelativePath.split("/").length <= 2;
    return isPt && isRootFile && file.webkitRelativePath;  // Must have path for folder upload
  });

  const directFiles = files.filter((file) => {
    const isPt = file.name.toLowerCase().endsWith(".pt");
    return isPt && !file.webkitRelativePath;  // Direct file selection, no path
  });

  // Determine which upload type this is
  const uploadedFiles = folderFiles.length > 0 ? folderFiles : directFiles;
  
  if (!uploadedFiles.length) {
    throw new Error("Select either a folder containing .pt files or a single graph.pt file.");
  }

  const fileNames = new Set(uploadedFiles.map((file) => file.name));
  const matchesSchema = UPLOAD_FILE_SCHEMAS.some((schema) => 
    schema.every((name) => fileNames.has(name))
  );
  
  if (!matchesSchema) {
    throw new Error(
      "Unsupported format. Upload either:\n" +
      "1. An Anemoi graph (single graph.pt file), or\n" +
      "2. A neural-lam graph folder with required .pt files (grid_lat_lon.pt, mesh_lat_lon.pt, g2m/m2m/m2g edge files, etc.), or\n" +
      "3. An ocean graph folder (mesh_features.pt plus the g2m/m2g/m2m edge index and feature files) — positions are reconstructed server-side"
    );
  }

  const graphName = inferGraphNameFromFiles(uploadedFiles);
  const formData = new FormData();
  formData.append("name", graphName);
  formData.append("description", 
    uploadedFiles[0].name === "graph.pt" 
      ? "Anemoi graph uploaded"
      : `Uploaded from local folder ${graphName}`
  );
  
  uploadedFiles.forEach((file) => {
    formData.append("files", file, file.name);
  });

  setLoadingText("Uploading...");
  setLoading(true);

  const response = await fetch(`${API}/graphs`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await readUploadError(response));
  }

  const createdRaw = (await response.json()) as unknown;
  const created = normalizeGraphRecord(createdRaw);
  if (!created) {
    throw new Error("Upload succeeded but returned malformed graph metadata.");
  }
  activeGraphId = created.graph_id;
  await reloadActiveGraphData();
  return created;
};

// Reads the live scale factor of Leaflet's internal zoom-animation proxy.
// Leaflet drives its entire zoom animation by CSS-transitioning a single
// `.leaflet-proxy` element (see L.Map._createAnimProxy). Sampling that element's
// computed scale each frame gives the true, in-progress interpolated zoom with
// no risk of a drifting parallel animation. Returns null if the proxy can't be
// read (e.g. animation disabled), signalling callers to skip mirroring.
const readLeafletProxyScale = (): number | null => {
  const proxy = leafMap.getContainer().querySelector(".leaflet-proxy") as HTMLElement | null;
  if (!proxy) return null;
  const t = getComputedStyle(proxy).transform;
  if (!t || t === "none") return 1;
  try {
    const m = new DOMMatrixReadOnly(t);
    return Math.hypot(m.a, m.b);
  } catch {
    return null;
  }
};

const syncDeckView = (): void => {
  if (!deckInstance || !deckReady || isZoomAnimating) return;
  const c = leafMap.getCenter();
  // deck.gl uses 512px tiles; Leaflet reports zoom for 256px tiles → always subtract 1
  const zoom = leafMap.getZoom() - 1;
  deckInstance.setProps({
    viewState: {
      map: {
        longitude: c.lng,
        latitude: c.lat,
        zoom,
        pitch: 0,
        bearing: 0,
        transitionDuration: 0,
      },
    },
  });
  deckInstance.redraw();
};

const buildLayers = (): unknown[] => {
  const isViol = ST.violation && violMetricsReady;
  const srcDeg = getSrcDegThresholds();
  const tgtDeg = getTgtDegThresholds();
  const lenThresh = getLenThresholds();
  // e.len is in meters (haversineM uses R=6371000); lenThresh is in km — convert once here.
  const lenThreshM = { lo: lenThresh.lo * 1000, hi: lenThresh.hi * 1000 };
  const sg = graphMeta.subgraphs.find((s) => s.key === ST.activeInspectSubgraph);

  // Determine which degree array and thresholds apply to each node type.
  const gridIsSource = sg ? sg.src === "grid" : true;
  const gridDegArr   = gridIsSource ? fullGraphSrcDeg : fullGraphTgtDeg;
  const gridDegLo    = gridIsSource ? srcDeg.lo : tgtDeg.lo;
  const gridDegHi    = gridIsSource ? srcDeg.hi : tgtDeg.hi;
  const meshIsSource = sg ? sg.src !== "grid" : false;
  const meshDegArr   = meshIsSource ? fullGraphSrcDeg : fullGraphTgtDeg;
  const meshDegLo    = meshIsSource ? srcDeg.lo : tgtDeg.lo;
  const meshDegHi    = meshIsSource ? srcDeg.hi : tgtDeg.hi;

  const violHash = [isViol, srcDeg.lo, srcDeg.hi, tgtDeg.lo, tgtDeg.hi, lenThresh.lo, lenThresh.hi].join(":");

  const isMeshSelected = ST.selected !== null && ST.selected.id in meshById;
  const isGridSelected = ST.selected !== null && ST.selected.id in gridById;
  const meshNeighborIds = isMeshSelected ? getMeshNeighborIds(ST.selected!.id) : null;
  const selectedMeshId = isMeshSelected ? ST.selected!.id : null;
  const selectedGridId = isGridSelected ? ST.selected!.id : null;
  const selectionHash = selectedMeshId !== null ? `m${selectedMeshId}` : selectedGridId !== null ? `g${selectedGridId}` : "none";

  const connectedGridIds = isMeshSelected ? getConnectedGridNodeIds(ST.selected!.id) : null;
  const connectedMeshIds = isGridSelected ? getConnectedMeshNodeIds(ST.selected!.id) : null;

  const meshHighlighted = (id: number): boolean => {
    if (meshNeighborIds !== null) return id === selectedMeshId || meshNeighborIds.has(id);
    if (connectedMeshIds !== null) return connectedMeshIds.has(id);
    return true;
  };
  const gridHighlighted = (id: number): boolean => {
    if (connectedGridIds !== null) return connectedGridIds.has(id);
    if (selectedGridId !== null) return id === selectedGridId;
    return true;
  };

  const layers: unknown[] = [];

  const b = leafMap.getBounds();
  const c = leafMap.getCenter();
  // Antimeridian-safe width; lat clamped at 60° so polar viewports don't collapse the threshold.
  let dLon = b.getEast() - b.getWest();
  if (dLon <= 0) dLon += 360;
  dLon = Math.min(360, dLon);
  const windowWidthM = dLon * 111320 * Math.cos(Math.min(60, Math.abs(c.lat)) * Math.PI / 180);

  const zoomFadeFactor = (len: number): number => {
    if (!ST.zoomFade || windowWidthM <= 0) return 1;
    const threshold = windowWidthM * ZOOM_FADE_THRESHOLD_FRAC;
    if (len >= threshold) return 1;
    return ZOOM_FADE_FLOOR + (1 - ZOOM_FADE_FLOOR) * Math.pow(len / threshold, ZOOM_FADE_POWER);
  };

  const eColor = (e: EdgeDatum, normalHex: string): [number, number, number, number] => {
    const fade = zoomFadeFactor(e.len); // 1.0 when zoom fade is off
    const baseAlpha = ST.zoomFade ? ZOOM_FADE_MAX_ALPHA : EDGE_ALPHA_NORMAL;
    if (isViol && e.len >= lenThreshM.hi) return hexToRgba(VIOL_ABOVE, 230);
    if (isViol && e.len <= lenThreshM.lo) return hexToRgba(VIOL_BELOW, 230);
    const greyAlpha = isViol ? (ST.zoomFade ? ZOOM_FADE_MAX_ALPHA : EDGE_ALPHA_VIOL_NEUTRAL) : baseAlpha;
    return hexToRgba(isViol ? VIOL_NEUTRAL : normalHex, Math.round(greyAlpha * fade));
  };

  const arrowIcon = (e: EdgeDatum, normalIcon: string): string => {
    if (!isViol) return normalIcon;
    if (e.len >= lenThreshM.hi) return "arrow_hi";
    if (e.len <= lenThreshM.lo) return "arrow_lo";
    return "arrow_grey";
  };

  // M2M: both endpoints are mesh ids — check either side.
  const meshEdgeDimmed = (e: EdgeDatum): boolean => {
    if (selectedMeshId !== null) return e.source !== selectedMeshId && e.target !== selectedMeshId;
    if (selectedGridId !== null) return true; // M2M edges are outside a grid node's 1-hop
    return false;
  };
  // G2M: source = grid id, target = mesh id.
  const g2mEdgeDimmed = (e: EdgeDatum): boolean => {
    if (selectedMeshId !== null) return e.target !== selectedMeshId;
    if (selectedGridId !== null) return e.source !== selectedGridId;
    return false;
  };
  // M2G: source = mesh id, target = grid id.
  const m2gEdgeDimmed = (e: EdgeDatum): boolean => {
    if (selectedMeshId !== null) return e.source !== selectedMeshId;
    if (selectedGridId !== null) return e.target !== selectedGridId;
    return false;
  };

  const grayEdgeColor = (color: [number, number, number, number]): [number, number, number, number] => {
    const lum = Math.round(0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]);
    return [lum, lum, lum, color[3]];
  };

  const edgeAngle = (e: EdgeDatum): number =>
    (Math.atan2(mercYdeg(e.tgt.lat) - mercYdeg(e.src.lat), e.tgt.lon - e.src.lon) * 180) / Math.PI;
  const srcPos = (e: EdgeDatum): [number, number] => [e.src.lon, clampLat(e.src.lat)];

  const addDirectedEdges = (
    id: string,
    data: EdgeDatum[],
    normalColor: string,
    normalArrow: string,
    tgtIsSquare: boolean,
    baseWidth = 1.8,
    tgtRadiusScale = 1.0,
    arrowSizeScale = 1.0,
    dimFn?: (e: EdgeDatum) => boolean,
  ): void => {
    if (!data.length) return;
    const tgtPos = (e: EdgeDatum): [number, number] =>
      adjustedEndpoint(srcPos(e), [e.tgt.lon, clampLat(e.tgt.lat)], tgtIsSquare, tgtRadiusScale);
    // Only show arrowheads on edges long enough for the head to clear the target node boundary.
    // For circle targets: minimum is 2×borderMerc so the arrow body doesn't bleed outside the node icon.
    const minLenMerc = tgtIsSquare ? 0 : 2 * nodeBorderMerc() * tgtRadiusScale;
    const arrowData = minLenMerc > 0
      ? data.filter((e) => {
          const dx = e.tgt.lon - e.src.lon;
          const dy = mercYdeg(clampLat(e.tgt.lat)) - mercYdeg(clampLat(e.src.lat));
          return Math.hypot(dx, dy) > minLenMerc;
        })
      : data;
    layers.push(
      new LineLayer<EdgeDatum>({
        id: `${id}-edges`,
        data,
        getSourcePosition: srcPos,
        getTargetPosition: tgtPos,
        getColor: (e) => {
          const color = eColor(e, normalColor);
          if (dimFn?.(e)) return grayEdgeColor(color);
          if (selectedMeshId !== null || selectedGridId !== null) return [color[0], color[1], color[2], 255] as [number, number, number, number];
          return color;
        },
        getWidth: (e) => (!isViol || e.len <= lenThreshM.lo || e.len >= lenThreshM.hi ? baseWidth : baseWidth * 0.55),
        widthUnits: "pixels",
        wrapLongitude: true,
        updateTriggers: { getColor: [violHash, windowWidthM, ST.zoomFade, selectionHash], getWidth: violHash },
      }),
    );
    layers.push(
      new IconLayer<EdgeDatum>({
        id: `${id}-arrows`,
        data: arrowData,
        iconAtlas: ARROW_ICON_ATLAS! as unknown as string,
        iconMapping: ARROW_ICON_MAPPING,
        getPosition: tgtPos,
        getIcon: (e) => dimFn?.(e) ? "arrow_grey" : arrowIcon(e, normalArrow),
        getAngle: edgeAngle,
        getSize: (e) => nodeSizeM() * 0.7 * arrowSizeScale * Math.cos(clampLat(e.tgt.lat) * Math.PI / 180),
        sizeUnits: "meters",
        sizeMinPixels: 0,
        getColor: (e) => {
          const [,,, a] = eColor(e, normalColor);
          const alpha = ((selectedMeshId !== null || selectedGridId !== null) && !dimFn?.(e)) ? 255 : a;
          return [255, 255, 255, alpha];
        },
        wrapLongitude: true,
        updateTriggers: { getIcon: [violHash, selectionHash], getColor: [violHash, windowWidthM, ST.zoomFade, selectionHash] },
      }),
    );
  };

  // Layer order: G2M (bottom) → mesh/bridge edges → M2G (top)
  if (ST.showG2M && ST.activeSubgraphs.has("g2m")) {
    addDirectedEdges("g2m", edgesStore["g2m"] ?? [], EDGE_ENCODER, "arrow_g2m", false, 1.8, 1.0, 1.0, g2mEdgeDimmed);
  }
  for (const key of Object.keys(edgesStore).sort()) {
    if (key === "g2m" || key === "m2g") continue;
    if (!ST.activeSubgraphs.has(key)) continue;
    const edges = edgesStore[key];
    if (!edges?.length) continue;
    if (key.startsWith("mesh_up_") || key.startsWith("mesh_down_")) {
      // Directed inter-level edges — finer level index is the suffix (e.g. "mesh_up_1" → lvl 1)
      const finerLvl = parseInt(key.split("_")[2] ?? "0", 10);
      const baseW = strokeForLevel(finerLvl, graphMeta.n_levels);
      // Target level: mesh_up goes src=finer→tgt=coarser(+1); mesh_down goes src=coarser→tgt=finer
      const tgtLvl = key.startsWith("mesh_up_") ? finerLvl + 1 : finerLvl;
      const tgtRadiusScale = MESH_SIZE_SCALE * nodeLevelBoost(`mesh_${tgtLvl}`, graphMeta.n_levels);
      addDirectedEdges(key, edges, EDGE_MESH, "arrow_m2m", false, baseW, tgtRadiusScale, tgtRadiusScale, meshEdgeDimmed);
    } else {
      // M2M intra-level edges — level index is the suffix (e.g. "m2m_2" → lvl 2)
      const lvl = parseInt(key.split("_")[1] ?? "0", 10);
      const baseW = key.startsWith("m2m_") ? strokeForLevel(lvl, graphMeta.n_levels) : 1.5;
      const sorted = longestFirst(key, edges);
      layers.push(
        new LineLayer<EdgeDatum>({
          id: `edges-${key}`,
          data: sorted,
          getSourcePosition: srcPos,
          getTargetPosition: (e) => [e.tgt.lon, clampLat(e.tgt.lat)],
          getColor: (e) => {
            const color = eColor(e, EDGE_MESH);
            if (meshEdgeDimmed(e)) return grayEdgeColor(color);
            if (selectedMeshId !== null) return [color[0], color[1], color[2], 255] as [number, number, number, number];
            return color;
          },
          getWidth: (e) => (!isViol || e.len <= lenThreshM.lo || e.len >= lenThreshM.hi ? baseW : baseW * 0.55),
          widthUnits: "pixels",
          wrapLongitude: true,
          updateTriggers: { getColor: [violHash, windowWidthM, ST.zoomFade, selectionHash], getWidth: violHash },
        }),
      );
    }
  }
  if (ST.showM2G && ST.activeSubgraphs.has("m2g")) {
    addDirectedEdges("m2g", edgesStore["m2g"] ?? [], EDGE_DECODER, "arrow_m2g", true, 1.8, 1.0, 1.0, m2gEdgeDimmed);
  }

  if (ST.showGrid && gridNodes.length) {
    layers.push(
      new IconLayer<NodeDatum>({
        id: "grid-nodes",
        data: gridNodes,
        iconAtlas: GRID_ICON_ATLAS! as unknown as string,
        iconMapping: GRID_ICON_MAPPING,
        getPosition: (n) => [n.lon, clampLat(n.lat)],
        getIcon: (n) => {
          if (!isViol) return "grid_normal";
          const d = gridDegArr[n.id] ?? 0;
          if (d >= gridDegHi) return "grid_hi";
          if (d <= gridDegLo) return "grid_lo";
          return "grid_grey";
        },
        getColor: (n) => gridHighlighted(n.id)
          ? [255, 255, 255, isViol ? 230 : NODE_ICON_ALPHA] as [number, number, number, number]
          : [160, 160, 160, 80] as [number, number, number, number],
        getSize: (n) => nodeSizeM() * Math.cos(clampLat(n.lat) * Math.PI / 180),
        sizeUnits: "meters",
        sizeMinPixels: 0,
        sizeMaxPixels: NODE_MAX_PX * 2,
        pickable: true,
        wrapLongitude: true,
        updateTriggers: { getIcon: violHash, getColor: [violHash, selectionHash] },
      }),
    );
  }

  if (ST.showMesh && meshNodes.length) {
    // In inspect mode for hierarchical graphs derive the relevant node types directly
    // from the active subgraph definition — ST.activeNodeTypes can be stale during
    // async mode transitions, causing the filter to produce an empty array.
    let meshTypeFilter: Set<string>;
    if (graphMeta.is_hierarchical) {
      if (ST.violation) {
        const sg = graphMeta.subgraphs.find((s) => s.key === ST.activeInspectSubgraph);
        meshTypeFilter = sg ? new Set([sg.src, sg.tgt]) : ST.activeNodeTypes;
      } else {
        meshTypeFilter = ST.activeNodeTypes;
      }
    } else {
      meshTypeFilter = new Set<string>(); // unused — non-hierarchical skips filter
    }
    const visibleMesh = graphMeta.is_hierarchical
      ? meshNodes.filter((n) => meshTypeFilter.has(n.type))
      : meshNodes;

    const meshNodeFillColor = (id: number): [number, number, number, number] =>
      meshHighlighted(id)
        ? hexToRgba(NODE_MESH, NODE_ICON_ALPHA) as [number, number, number, number]
        : hexToGrayRgba(NODE_MESH, 150);

    const meshNodeLineColor = (id: number): [number, number, number, number] =>
      meshHighlighted(id)
        ? hexToRgba(TEXT, NODE_ICON_ALPHA) as [number, number, number, number]
        : hexToGrayRgba(TEXT, 150);

    if (!isViol) {
      layers.push(
        new ScatterplotLayer<NodeDatum>({
          id: "mesh-nodes",
          data: visibleMesh,
          getPosition: (n) => [n.lon, clampLat(n.lat)],
          getRadius: (n) => nodeSizeM() * MESH_SIZE_SCALE * 0.5 * Math.cos(clampLat(n.lat) * Math.PI / 180) * nodeLevelBoost(n.type, graphMeta.n_levels),
          radiusUnits: "meters",
          radiusMinPixels: 0,
          radiusMaxPixels: NODE_MAX_PX,
          stroked: true,
          filled: true,
          getLineWidth: (n) => nodeSizeM() * MESH_SIZE_SCALE * (1 / NODE_BORDER_DIVISOR) * Math.cos(clampLat(n.lat) * Math.PI / 180) * nodeLevelBoost(n.type, graphMeta.n_levels),
          lineWidthUnits: "meters",
          getFillColor: (n) => meshNodeFillColor(n.id),
          getLineColor: (n) => meshNodeLineColor(n.id),
          pickable: true,
          wrapLongitude: true,
          updateTriggers: { getFillColor: selectionHash, getLineColor: selectionHash },
        }),
      );
    } else {
      layers.push(
        new ScatterplotLayer<NodeDatum>({
          id: "mesh-nodes-inspect",
          data: visibleMesh,
          getPosition: (n) => [n.lon, clampLat(n.lat)],
          getRadius: (n) => nodeSizeM() * MESH_SIZE_SCALE * 0.5 * Math.cos(clampLat(n.lat) * Math.PI / 180) * nodeLevelBoost(n.type, graphMeta.n_levels),
          radiusUnits: "meters",
          radiusMinPixels: 0,
          radiusMaxPixels: NODE_MAX_PX,
          stroked: true,
          filled: true,
          getLineWidth: (n) => nodeSizeM() * MESH_SIZE_SCALE * (1 / NODE_BORDER_DIVISOR) * Math.cos(clampLat(n.lat) * Math.PI / 180) * nodeLevelBoost(n.type, graphMeta.n_levels),
          lineWidthUnits: "meters",
          getLineColor: (n) => meshNodeLineColor(n.id),
          getFillColor: (n) => {
            if (!meshHighlighted(n.id)) return hexToGrayRgba(NODE_MESH, 150);
            const d = meshDegArr[n.id] ?? 0;
            if (d >= meshDegHi) return hexToRgba(VIOL_ABOVE, 230) as [number, number, number, number];
            if (d <= meshDegLo) return hexToRgba(VIOL_BELOW, 230) as [number, number, number, number];
            return hexToRgba(VIOL_NEUTRAL, 180) as [number, number, number, number];
          },
          pickable: true,
          wrapLongitude: true,
          updateTriggers: { getFillColor: [violHash, selectionHash], getLineColor: selectionHash },
        }),
      );
    }
  }


  return layers;
};

const buildCoastlineLayer = (): GeoJsonLayer | null => {
  if (!coastlineGeoJsonClipped) return null;
  const stroke = getComputedStyle(document.documentElement)
    .getPropertyValue("--text-dim").trim() || "#94a3b8";
  return new GeoJsonLayer({
    id: "explain-coastline",
    data: coastlineGeoJsonClipped as GeoJSON.FeatureCollection,
    stroked: true,
    filled: false,
    lineWidthMinPixels: 1,
    getLineColor: hexToRgba(stroke, 255),
    pickable: false,
    parameters: { depthTest: false },
  });
};

const rebuildExplainLayers = (): void => {
  if (!deckInstance || !deckReady) { console.log("[Explain] rebuildExplainLayers skipped — deck not ready", { deckInstance: !!deckInstance, deckReady }); return; }
  if (currentMode !== "explain") return;
  const layers: unknown[] = [];
  const heatmap = buildHeatmapLayer();
  if (heatmap) layers.push(heatmap);
  const coastline = buildCoastlineLayer();
  if (coastline) layers.push(coastline);
  const hoverRing = buildHoverRingLayer(currentExplainHoveredNode);
  if (hoverRing) layers.push(hoverRing);
  const ring = buildSelectionRingLayer(ExST.node);
  if (ring) layers.push(ring);
  console.log("[Explain] rebuildExplainLayers → layer ids:", layers.map((l: unknown) => (l as { id: string }).id));
  deckInstance.setProps({ layers: layers as never[] });
};

/** @param force draw even while violation thresholds are still loading. Only fetchGraphMetrics
 *  should pass this, once it knows whether thresholds arrived. */
const renderLayers = (force = false): void => {
  if (!deckInstance || !deckReady) return;
  // Switching subgraphs clears violMetricsReady and refetches. In that gap buildLayers falls
  // back to explore colouring, which is indistinguishable from a genuine "nothing violates"
  // result — the most misleading thing this view can show, and it fires on every switch
  // because pan/zoom and the geometry fetch both render before the metrics land. Hold the
  // previous frame instead; fetchGraphMetrics forces a draw as soon as it settles.
  if (!force && currentMode !== "explain" && ST.violation && !violMetricsReady) return;
  if (currentMode === "explain") {
    rebuildExplainLayers();
  } else {
    deckInstance.setProps({ layers: buildLayers() as never[] });
  }
};

const statRow = (icon: string, name: string, inView: number, total: number): string =>
  `<div class="stat-row">${icon}<span class="stat-name">${name}</span>` +
  `<span class="stat-val">${inView.toLocaleString()} in view / ${total > 0 ? total.toLocaleString() : "–"} total</span></div>`;

const updateGraphStats = (): void => {
  let html = "";
  if (graphMeta.subgraphs.length > 0) {
    // Hierarchical graph: per-level mesh nodes (coarsest first), then edges, then grid
    const L = graphMeta.n_levels;
    const levelFromKey = (key: string): number => {
      const m = key.match(/_(\d+)$/);
      return m ? parseInt(m[1], 10) : 0;
    };
    const nodeIconSize = (lvl: number): number => Math.round(7 + 6 * lvl / Math.max(1, L - 1));
    if (L <= 1) {
      html += statRow(svgCircle("var(--mesh-color)", 10), "Mesh nodes", meshNodes.length, graphMeta.n_mesh);
    } else {
      for (let lvl = L - 1; lvl >= 0; lvl--) {
        const type = `mesh_${lvl}`;
        const inView = meshNodes.filter((n) => n.type === type).length;
        const total = graphMeta.node_counts[type] ?? 0;
        html += statRow(svgCircle("var(--mesh-color)", nodeIconSize(lvl)), `Mesh node L${lvl}`, inView, total);
      }
    }
    // Ordered m2m → mesh_up → mesh_down → g2m → m2g; within each group coarsest first
    const order = (key: string): number => {
      if (key.startsWith("m2m_") || key === "m2m") return 0;
      if (key.startsWith("mesh_up_"))   return 1;
      if (key.startsWith("mesh_down_")) return 2;
      if (key === "g2m") return 3;
      if (key === "m2g") return 4;
      return 5;
    };
    const sorted = [...graphMeta.subgraphs].sort((a, b) => {
      const od = order(a.key) - order(b.key);
      if (od !== 0) return od;
      return levelFromKey(b.key) - levelFromKey(a.key);  // coarsest first within group
    });
    for (const sg of sorted) {
      const inView = edgesStore[sg.key]?.length ?? 0;
      const lvl = levelFromKey(sg.key);
      const w = strokeForLevel(lvl, L);
      const icon = sg.key === "g2m"               ? svgArrow("var(--g2m-color)") :
                   sg.key === "m2g"               ? svgArrow("var(--m2g-color)") :
                   sg.key.startsWith("mesh_up_")  ? svgArrow("var(--mesh-color)", w) :
                   sg.key.startsWith("mesh_down_")? svgArrow("var(--mesh-color)", w) :
                                                    svgLine("var(--mesh-color)", w);
      html += statRow(icon, sg.label, inView, sg.n_edges ?? 0);
    }
    html += statRow(svgSquare("var(--grid-color)"), "Grid nodes", gridNodes.length, graphMeta.n_grid);
  } else {
    // Non-hierarchical (L=1): single M2M level uses mid-range weight
    const w1 = strokeForLevel(0, 1);
    html =
      statRow(svgLine("var(--mesh-color)", w1), "M2M edges", edgesStore["m2m"]?.length ?? 0, graphMeta.n_edges_m2m) +
      statRow(svgArrow("var(--g2m-color)"),     "G2M edges", edgesStore["g2m"]?.length ?? 0, graphMeta.n_edges_g2m) +
      statRow(svgArrow("var(--m2g-color)"),     "M2G edges", edgesStore["m2g"]?.length ?? 0, graphMeta.n_edges_m2g) +
      statRow(svgSquare("var(--grid-color)"),   "Grid nodes", gridNodes.length, graphMeta.n_grid);
  }
  getEl<HTMLDivElement>("stats-grid").innerHTML = html;
};

const updateHeaderCounts = (): void => {
  const n = graphMeta.n_grid + graphMeta.n_mesh;
  const e = graphMeta.is_hierarchical
    ? graphMeta.subgraphs.reduce((sum, sg) => sum + (sg.n_edges ?? 0), 0)
    : graphMeta.n_edges_g2m + graphMeta.n_edges_m2m + graphMeta.n_edges_m2g;
  getEl<HTMLSpanElement>("hdr-node-count").textContent = n > 0 ? n.toLocaleString() : "–";
  getEl<HTMLSpanElement>("hdr-edge-count").textContent = e > 0 ? e.toLocaleString() : "–";
};

const updateGraphNameDisplay = (): void => {
  const el = document.getElementById("graph-name-display");
  if (!el) return;
  const graph = graphCatalog.find((g) => g.graph_id === activeGraphId);
  if (!graph) { el.textContent = ""; return; }
  const parts = graph.folder_path.replace(/\\/g, "/").split("/").filter(Boolean);
  const folderName = parts[parts.length - 1] || graph.name || "";
  el.textContent = folderName ? `Graph name: ${folderName}` : "";
};

const updateInfoPanel = (): void => {
  const titleEl = getEl<HTMLDivElement>("info-title");
  const subEl = getEl<HTMLDivElement>("info-sub");
  const detSec = getEl<HTMLDivElement>("info-details");
  if (!ST.selected) {
    titleEl.textContent = "Element Info";
    subEl.textContent = "Click a node to inspect";
    detSec.style.display = "none";
    return;
  }
  const n = ST.selected;
  const deg = computeDegrees()[n.id] ?? 0;
  titleEl.textContent = n.type === "grid" ? "Grid Node" : "Mesh Node";
  subEl.textContent = `ID: ${n.id}`;
  detSec.style.display = "block";
  detSec.innerHTML =
    `<div class=\"det-row\"><span class=\"dk\">ID</span><span class=\"dv\">${n.id}</span></div>` +
    `<div class=\"det-row\"><span class=\"dk\">Type</span><span class=\"dv\">${n.type}</span></div>` +
    `<div class=\"det-row\"><span class=\"dk\">Lat / Lon</span><span class=\"dv\">${n.lat.toFixed(3)}°, ${n.lon.toFixed(3)}°</span></div>` +
    `<div class=\"det-row\"><span class=\"dk\">Degree</span><span class=\"dv\">${deg}</span></div>`;
};

// Unified icon SVG strings — single source of truth used in both sidebar and violation stats.
const svgSquare = (color: string): string =>
  `<svg width="10" height="10" viewBox="0 0 10 10" style="flex-shrink:0"><rect x="0.5" y="0.5" width="9" height="9" rx="1" fill="${color}" stroke="rgba(0,0,0,0.3)" stroke-width="1"/></svg>`;

const svgCircle = (color: string, size = 10): string => {
  const r = (size - 1) / 2;
  return `<svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" style="flex-shrink:0;align-self:center"><circle cx="${size/2}" cy="${size/2}" r="${r}" fill="${color}" stroke="rgba(0,0,0,0.3)" stroke-width="1"/></svg>`;
};

const svgArrow = (color: string, weight = 1.5): string =>
  `<svg width="16" height="10" viewBox="0 0 16 10" style="flex-shrink:0"><line x1="1" y1="5" x2="11" y2="5" stroke="${color}" stroke-width="${weight}" stroke-linecap="round"/><polyline points="11,2 15,5 11,8" fill="none" stroke="${color}" stroke-width="${weight}" stroke-linejoin="round" stroke-linecap="round"/></svg>`;

const svgLine = (color: string, weight = 1.5): string =>
  `<svg width="16" height="10" viewBox="0 0 16 10" style="flex-shrink:0"><line x1="1" y1="5" x2="15" y2="5" stroke="${color}" stroke-width="${weight}" stroke-linecap="round"/></svg>`;

const violLegendIcon = (shape: "square" | "circle" | "arrow" | "line", color: string): string => {
  const svg =
    shape === "square" ? svgSquare(color) :
    shape === "circle" ? svgCircle(color) :
    shape === "arrow"  ? svgArrow(color) :
                         svgLine(color);
  return `<span style="flex-shrink:0;margin-right:3px;display:inline-flex;align-items:center">${svg}</span>`;
};

const violLine = (color: string, label: string, n: number, N: number, shape: "square" | "circle" | "arrow" | "line" | ""): string => {
  const ratio = N > 0 ? ` <span style=\"color:var(--text-dim);font-weight:400\">/ ${N}</span>` : "";
  const icon = shape
    ? violLegendIcon(shape as "square" | "circle" | "arrow" | "line", color)
    : `<span class=\"vl-dot\" style=\"background:${color}\"></span>`;
  return (
    `<div class=\"viol-line\">` +
    icon +
    `<span class=\"vl-n\" style=\"color:${color}\">${n}</span>${ratio}` +
    `<span class=\"vl-label\">${label}</span>` +
    `</div>`
  );
};

const updateViolationStats = (): void => {
  const sg = graphMeta.subgraphs.find((s) => s.key === ST.activeInspectSubgraph);
  if (!sg) return;

  const srcN = graphMeta.node_counts[sg.src] ?? 0;
  const srcShape = sg.src === "grid" ? "square" : "circle";
  getEl<HTMLDivElement>("src-deg-violation-count").innerHTML =
    violLine(VIOL_BELOW, "under threshold", violTotals.srcLo, srcN, srcShape) +
    violLine(VIOL_ABOVE, "over threshold", violTotals.srcHi, srcN, srcShape);

  const tgtEl = document.getElementById("tgt-deg-violation-count");
  if (tgtEl) {
    if (sg.src !== sg.tgt) {
      const tgtN = graphMeta.node_counts[sg.tgt] ?? 0;
      const tgtShape = sg.tgt === "grid" ? "square" : "circle";
      tgtEl.innerHTML =
        violLine(VIOL_BELOW, "under threshold", violTotals.tgtLo, tgtN, tgtShape) +
        violLine(VIOL_ABOVE, "over threshold", violTotals.tgtHi, tgtN, tgtShape);
    } else {
      tgtEl.innerHTML = "";
    }
  }

  const edgeSh = sg.edge === "m2m" ? "line" : "arrow";
  getEl<HTMLDivElement>("len-violation-count").innerHTML =
    violLine(VIOL_BELOW, "under threshold", violTotals.edgeLo, 0, edgeSh) +
    violLine(VIOL_ABOVE, "over threshold", violTotals.edgeHi, 0, edgeSh);
};

// Corrected thumb center as % of slider element width.
// Reads slider.min from the element to account for non-zero min (e.g. log mode).
const thumbCenterPct = (value: number, maxV: number, sliderEl: HTMLInputElement | null): number => {
  const minV = sliderEl ? (parseFloat(sliderEl.min) || 0) : 0;
  const range = maxV - minV;
  if (!range) return 0;
  const W = sliderEl?.offsetWidth ?? 0;
  if (!W) return ((value - minV) / range) * 100;
  const r = 7;
  return (r + ((value - minV) / range) * (W - 2 * r)) / W * 100;
};

const updateThumbLabel = (id: string, pct: number, val: number | string): void => {
  const el = getEl<HTMLSpanElement>(id);
  el.textContent = String(val);
  el.style.left = `${pct}%`;
};

const updateTrackGradient = (trackId: string, p1: number, p2: number): void => {
  const el = getEl<HTMLDivElement>(trackId);
  el.style.background =
    `linear-gradient(to right,${VIOL_BELOW} 0%,${VIOL_BELOW} ${p1}%,#d1d5db ${p1}%,#d1d5db ${p2}%,${VIOL_ABOVE} ${p2}%,${VIOL_ABOVE} 100%)`;
};


// ── Threshold queries over the sorted length array ───────────────────────────
//
// Every length-derived number the Inspect panel shows is a threshold query, and all of them
// rerun on each slider `input` event. Against an unsorted array each is a full scan: on ocean
// m2g (2,011,528 edges) one slider tick cost two filters, a d3.bin, and a min/max pass — about
// 8M element visits, synchronously, per tick. The backend now returns the lengths ascending,
// which turns every one of them into a binary search.

/** First index whose value is >= x (== count of values strictly below x). */
const lowerBound = (a: ArrayLike<number>, x: number): number => {
  let lo = 0, hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (a[mid] < x) lo = mid + 1; else hi = mid;
  }
  return lo;
};

/** First index whose value is > x (== count of values <= x). */
const upperBound = (a: ArrayLike<number>, x: number): number => {
  let lo = 0, hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (a[mid] <= x) lo = mid + 1; else hi = mid;
  }
  return lo;
};

/** Count of values <= x in an ascending array. Exact, not estimated. */
const countLE = (sorted: ArrayLike<number>, x: number): number => upperBound(sorted, x);
/** Count of values >= x in an ascending array. Exact, not estimated. */
const countGE = (sorted: ArrayLike<number>, x: number): number => sorted.length - lowerBound(sorted, x);

/** Per-bin counts for an ascending array, given ascending bin edges of length nBins + 1.
 *  Values outside [edges[0], edges[nBins]] are excluded, matching d3.bin's domain clamping. */
const binCountsSorted = (sorted: ArrayLike<number>, edges: number[]): number[] => {
  const out: number[] = [];
  for (let i = 0; i < edges.length - 1; i++) {
    // Half-open [lo, hi) everywhere except the last bin, which closes — d3.bin does the same,
    // so the maximum value lands in the top bar rather than being dropped.
    const lo = lowerBound(sorted, edges[i]);
    const hi = i === edges.length - 2 ? upperBound(sorted, edges[i + 1]) : lowerBound(sorted, edges[i + 1]);
    out.push(Math.max(0, hi - lo));
  }
  return out;
};

/** Predicate count over a typed array — the fallback when the memo is not yet populated. */
const countMatching = (data: ArrayLike<number>, pred: (v: number) => boolean): number => {
  let n = 0;
  for (let i = 0; i < data.length; i++) if (pred(data[i])) n++;
  return n;
};

/** Per-bin counts from a value -> count table, given ascending bin edges. Same half-open
 *  convention as binCountsSorted, so a degree histogram and a length histogram bin alike. */
const binCountsFromEntries = (entries: { v: number; n: number }[], edges: number[]): number[] => {
  const nBins = edges.length - 1;
  const out = new Array<number>(nBins).fill(0);
  for (const e of entries) {
    if (e.v < edges[0] || e.v > edges[nBins]) continue;
    // Last index whose edge is <= v; clamped so the maximum lands in the top bar rather than
    // spilling past the end (the last bin closes on the right, as in binCountsSorted).
    const bin = Math.min(nBins - 1, upperBound(edges, e.v) - 1);
    out[bin] += e.n;
  }
  return out;
};

/** Exact value -> count entries for an integer array, ascending by value.
 *  Degrees are small integers, so this is a single pass and a sort of the distinct values. */
const degreeEntries = (data: ArrayLike<number>): { v: number; n: number }[] => {
  const counts = new Map<number, number>();
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    counts.set(v, (counts.get(v) ?? 0) + 1);
  }
  return [...counts.entries()].map(([v, n]) => ({ v, n })).sort((a, b) => a.v - b.v);
};

/** Cumulative counts for a degree entry table, so threshold queries are O(distinct) rather
 *  than O(N) — degrees have at most a few hundred distinct values even on the ocean graph. */
const entriesCountLE = (entries: { v: number; n: number }[], x: number): number => {
  let total = 0;
  for (const e of entries) { if (e.v <= x) total += e.n; else break; }
  return total;
};
const entriesCountGE = (entries: { v: number; n: number }[], x: number): number => {
  let total = 0;
  for (let i = entries.length - 1; i >= 0; i--) {
    if (entries[i].v >= x) total += entries[i].n; else break;
  }
  return total;
};

// ── Per-subgraph derived metrics, computed once per fetch ────────────────────
//
// These depend only on the metric arrays, never on the slider positions, but every consumer
// of them sits behind a slider handler. Memoizing on the subgraph key keeps them off the
// drag path; fetchGraphMetrics is the only thing that invalidates them.
type DerivedMetrics = {
  key: SubgraphKey;
  srcEntries: { v: number; n: number }[];
  tgtEntries: { v: number; n: number }[];
  lengthShape: { discrete: boolean; clusters: { v: number; n: number }[] };
};
let derivedMetrics: DerivedMetrics | null = null;

const getDerivedMetrics = (): DerivedMetrics | null => {
  if (derivedMetrics && derivedMetrics.key === ST.activeInspectSubgraph) return derivedMetrics;
  return null;
};

/** Padded chart bounds for a metric. Takes the extremes rather than the array: the backend
 *  reports min/max in the metrics header, so nothing here needs to scan 2M values. */
const degBoundsFromExtremes = (
  count: number, lo: number, hi: number, logMode: boolean,
): { min: number; max: number } => {
  if (!count) return { min: 0, max: 30 };
  if (logMode) {
    const lPad = Math.max(0.15, 0.05 * (Math.log(Math.max(1, hi)) - Math.log(Math.max(1, lo))));
    return {
      min: Math.max(1, Math.exp(Math.log(Math.max(1, lo)) - lPad)),
      max: Math.exp(Math.log(Math.max(1, hi)) + lPad),
    };
  }
  const pad = Math.max(1, Math.round(0.05 * (hi - lo)));
  return { min: Math.max(0, lo - pad), max: hi + pad };
};

/** Degree-side flavour of degBoundsFromExtremes, reading the extremes the backend supplied
 *  rather than scanning the (up to 675k-entry) degree array. */
const degBoundsGlobal = (side: "src" | "tgt", logMode: boolean): { min: number; max: number } => {
  const arr = side === "src" ? fullGraphSrcDeg : fullGraphTgtDeg;
  const lo = side === "src" ? fullGraphBounds.srcDegMin : fullGraphBounds.tgtDegMin;
  const hi = side === "src" ? fullGraphBounds.srcDegMax : fullGraphBounds.tgtDegMax;
  return degBoundsFromExtremes(arr.length, lo, hi, logMode);
};

const setDegSliderMode = (
  minSl: HTMLInputElement,
  maxSl: HTMLInputElement,
  sliderMin: number,
  sliderMax: number,
  lo: number,
  hi: number,
  logOn: boolean,
): void => {
  if (logOn) {
    const logMin = Math.log(Math.max(1, sliderMin));
    const logMax = Math.log(Math.max(1, sliderMax));
    minSl.min = logMin.toFixed(4); minSl.max = logMax.toFixed(4); minSl.step = "0.01";
    maxSl.min = logMin.toFixed(4); maxSl.max = logMax.toFixed(4); maxSl.step = "0.01";
    minSl.value = Math.log(Math.max(1, lo)).toFixed(4);
    maxSl.value = Math.log(Math.max(1, hi)).toFixed(4);
  } else {
    minSl.min = String(sliderMin); minSl.max = String(sliderMax); minSl.step = "1";
    maxSl.min = String(sliderMin); maxSl.max = String(sliderMax); maxSl.step = "1";
    minSl.value = String(lo); maxSl.value = String(hi);
  }
};

const getSrcDegThresholds = (): { lo: number; hi: number } => {
  const minSl = getEl<HTMLInputElement>("src-deg-min-sl");
  const maxSl = getEl<HTMLInputElement>("src-deg-max-sl");
  if (ST.srcDegLogScaleX) {
    return { lo: Math.round(Math.exp(parseFloat(minSl.value))), hi: Math.round(Math.exp(parseFloat(maxSl.value))) };
  }
  return { lo: parseInt(minSl.value, 10), hi: parseInt(maxSl.value, 10) };
};

const getTgtDegThresholds = (): { lo: number; hi: number } => {
  const minSl = getEl<HTMLInputElement>("tgt-deg-min-sl");
  const maxSl = getEl<HTMLInputElement>("tgt-deg-max-sl");
  if (ST.tgtDegLogScaleX) {
    return { lo: Math.round(Math.exp(parseFloat(minSl.value))), hi: Math.round(Math.exp(parseFloat(maxSl.value))) };
  }
  return { lo: parseInt(minSl.value, 10), hi: parseInt(maxSl.value, 10) };
};

const getLenThresholds = (): { lo: number; hi: number } => {
  const minSl = getEl<HTMLInputElement>("len-min-sl");
  const maxSl = getEl<HTMLInputElement>("len-max-sl");
  if (ST.lenLogScaleX) {
    return { lo: Math.round(Math.exp(parseFloat(minSl.value))), hi: Math.round(Math.exp(parseFloat(maxSl.value))) };
  }
  return { lo: parseFloat(minSl.value), hi: parseFloat(maxSl.value) };
};

// Above this many levels a stem plot is unreadable and the data is not meaningfully discrete.
const MAX_DISCRETE_LEVELS = 24;
// Degrees are integers, so a stem per value is natural — but only while there are few enough
// values to draw. Ocean m2g source degree spans [6, 4694] with 381 distinct values; 381 stems
// on a 240px axis is a smear, not a frequency plot.
const MAX_DISCRETE_DEGREES = 40;
// Degree plots have no bins control of their own (unlike edge length), so the binned fallback
// uses a fixed count.
const DEGREE_HIST_BINS = 30;

/** Number of distinct values, or null once it exceeds `cap`. Early-exits, so the wide cases
 *  this exists to detect cost almost nothing even on 675k-node degree arrays. */
const distinctAtMost = (data: ArrayLike<number>, cap: number): number | null => {
  const seen = new Set<number>();
  for (let i = 0; i < data.length; i++) {
    seen.add(data[i]);
    if (seen.size > cap) return null;
  }
  return seen.size;
};
// Enough to characterise a distribution's shape; keeps the decision off the critical path for
// the million-edge g2m/m2g subgraphs.
const SHAPE_SAMPLE_MAX = 200_000;
// Resolution of the occupancy test below. Fine enough that two levels have to sit within
// span/512 of each other before they merge, coarse enough that the counting pass is trivial.
const SHAPE_BINS = 512;
// Fraction of the sample trimmed off each end before the occupancy test. Guards the bin grid
// against a stray extreme value: the grid spans the observed range, so one far-out length
// would spread it thin enough to leave most bins empty and make continuous data look
// clustered. Used for occupancy only — see the note in analyzeLengths for why the level scan
// must not share it.
const SHAPE_TRIM = 0.005;
// Occupied-bin ratio above which the lengths are treated as continuous. Measured separation on
// the shipped graphs is wide in both directions: 0.05-0.18 for meshes with real levels
// (ocean m2m_0 the highest at 0.178), 0.85-1.00 for everything continuous (the K-means
// clustered m2m levels included, at 0.85+). Anything in between is genuinely ambiguous.
const OCCUPANCY_MAX = 0.35;

/** Stride-sample of at most `cap` entries from an already-ascending array.
 *  No sort: `data` arrives sorted from the backend, and a stride sample of a sorted array is
 *  itself sorted. This used to sort a 200k Float64Array on every slider tick. */
const sortedSample = (data: ArrayLike<number>, cap: number): Float64Array => {
  const n = data.length;
  if (n <= cap) return Float64Array.from(data as ArrayLike<number>);
  const stride = Math.ceil(n / cap);
  const out = new Float64Array(Math.ceil(n / stride));
  for (let i = 0, j = 0; i < n; i += stride, j++) out[j] = data[i];
  return out;
};

/** How much of a fine uniform grid the values actually touch, relative to what a continuous
 *  distribution of the same size would touch. Input must be ascending.
 *
 *  Lengths that sit at discrete levels leave almost every bin empty; continuous lengths fill
 *  essentially all of them. Normalising by the continuous expectation `1 - e^(-n/B)` is what
 *  makes the number comparable across subgraph sizes — the coarsest mesh level has a few
 *  hundred edges, where even a continuous spread could only reach a fraction of 512 bins, so
 *  a raw occupied-bin count would call it discrete for the wrong reason.
 *
 *  Scale-free by construction: the grid spans the observed range, so nothing here depends on
 *  absolute km. A global mesh and a limited-area mesh differ by an order of magnitude in edge
 *  length, and any fixed threshold would be wrong for one of them. */
const occupancyRatio = (s: Float64Array): number => {
  const n = s.length;
  const span = s[n - 1] - s[0];
  if (span <= 1e-9) return 0;
  const step = span / SHAPE_BINS;
  const seen = new Uint8Array(SHAPE_BINS);
  let filled = 0;
  for (let i = 0; i < n; i++) {
    let k = Math.floor((s[i] - s[0]) / step);
    if (k >= SHAPE_BINS) k = SHAPE_BINS - 1;
    if (!seen[k]) { seen[k] = 1; filled++; }
  }
  const expected = 1 - Math.exp(-n / SHAPE_BINS);
  return filled / SHAPE_BINS / Math.max(expected, 1e-9);
};

/** The ascending slice with `SHAPE_TRIM` of the values removed from each end. */
const trimmed = (s: Float64Array): Float64Array => {
  const n = s.length;
  const lo = Math.floor(n * SHAPE_TRIM);
  const hi = Math.ceil(n * (1 - SHAPE_TRIM));
  return hi - lo >= 2 ? s.subarray(lo, hi) : s;
};

/** Decide whether edge lengths fall into discrete levels (stems) or form a continuous
 *  distribution (binned histogram), and return the levels if they do.
 *
 *  The backend's per-subgraph `plot` field assumes every M2M-style mesh has a handful of exact
 *  levels. That holds for a regular icosahedral mesh but not for a K-means clustered one,
 *  where the lengths are effectively continuous and the stems degenerate into a forest of
 *  near-touching spikes that reads as noise. Deciding from the lengths themselves keeps both
 *  mesh families correct.
 *
 *  Two separate questions, deliberately answered by two different measures: whether the data
 *  is clustered at all (occupancy, below) and where the levels sit (the gap scan). Running
 *  only the gap scan and judging by how many groups it produced conflates them — on the
 *  clustered global mesh it yields 14-17 groups, under the drawable cap, which is why the
 *  cluster case has to be rejected before the levels are extracted rather than after. */
const analyzeLengths = (
  data: ArrayLike<number>,
): { discrete: boolean; clusters: { v: number; n: number }[] } => {
  if (!data.length) return { discrete: false, clusters: [] };

  // Decide the shape from a sample. m2g on the ocean graph has 2M edges, and sorting all of
  // them cost ~1s per subgraph switch — synchronous, so the degree plots drawn immediately
  // above never got painted either and the whole panel looked empty. A stride sample settles
  // "discrete or not" just as well, because a real level has many members either way.
  const sample = sortedSample(data, SHAPE_SAMPLE_MAX);
  if (sample[sample.length - 1] - sample[0] <= 1e-9) {
    return { discrete: true, clusters: [{ v: sample[0], n: data.length }] };
  }

  // Measured on the trimmed core so a lone extreme length cannot stretch the bin grid thin
  // enough to make a continuous spread read as clustered.
  const core = trimmed(sample);
  if (core[core.length - 1] - core[0] <= 1e-9) {
    return { discrete: true, clusters: [{ v: core[0], n: data.length }] };
  }
  if (occupancyRatio(core) > OCCUPANCY_MAX) return { discrete: false, clusters: [] };

  // Exact pass, only on the discrete branch: the counts are the stem heights, so they have to
  // come from every edge, not a sample. Already ascending, so this is a scan, not a sort.
  //
  // Full span here, not the trimmed core the occupancy test uses. The two measures defend
  // against opposite things and must not share an input: trimming stops one stray value from
  // faking discreteness, but a quarter of every mesh_up/mesh_down subgraph is zero-length
  // self-edges, far too many for a percentile trim to remove. All it removes there is the top
  // of a legitimate tail — on ocean up_0 it cuts the range from 535 km to 132 km, shrinking
  // the gap fourfold and splintering 13 real levels into 26.
  const full = data;
  const gap = (full[full.length - 1] - full[0]) * 0.01;
  const clusters: { v: number; n: number }[] = [];
  let start = 0;
  for (let i = 1; i <= full.length; i++) {
    if (i === full.length || full[i] - full[i - 1] > gap) {
      const len = i - start;
      const half = start + (len >> 1);
      // Exact, not rounded to whole km. `v` is the stem's x position, and rounding moved every
      // stem off its true length — harmless on a global mesh whose levels sit km apart, but on
      // a fine mesh two levels under 1 km apart collapse onto one x and draw on top of
      // each other. The axis formatter handles display, so nothing needs a rounded value.
      clusters.push({ v: len % 2 === 1 ? full[half] : (full[half - 1] + full[half]) / 2, n: len });
      start = i;
    }
  }
  // The occupancy test says the data clusters; this says the clusters are drawable. A mesh with
  // more levels than stems that fit on a 240px axis is better served by the binned histogram.
  if (clusters.length > MAX_DISCRETE_LEVELS) return { discrete: false, clusters: [] };
  return { discrete: true, clusters };
};

// Lollipop frequency plot for integer-valued data (node degrees) or pre-clustered entries.
const buildDegreeFreqPlot = (
  containerId: string,
  data: ArrayLike<number>,
  domainMax: number,
  threshLo: number | null,
  threshHi: number | null,
  logScaleX: boolean,
  logScaleY: boolean,
  xMinDomain?: number,
  prebuiltEntries?: { v: number; n: number }[],
): void => {
  const container = getEl<HTMLDivElement>(containerId);
  container.innerHTML = "";
  if (!prebuiltEntries && (!data || !data.length)) return;
  if (prebuiltEntries && !prebuiltEntries.length) return;

  const W = 240; const H = 70;
  const mg = { top: 4, right: 8, bottom: 18, left: 28 };
  const w = W - mg.left - mg.right;
  const h = H - mg.top - mg.bottom;

  const svg = d3.select(container).append("svg").attr("viewBox", `0 0 ${W} ${H}`);
  const g = svg.append("g").attr("transform", `translate(${mg.left},${mg.top})`);

  let entries: { v: number; n: number }[];
  if (prebuiltEntries) {
    entries = [...prebuiltEntries].sort((a, b) => a.v - b.v);
  } else {
    entries = degreeEntries(data);
  }

  const xMax = domainMax || ((d3.max(entries, (d) => d.v) ?? 0) + 1);
  const yMax = d3.max(entries, (d) => d.n) ?? 1;
  const xMin0 = xMinDomain !== undefined ? xMinDomain : (logScaleX ? 1 : 0);

  const x = logScaleX
    ? d3.scaleLog().domain([xMin0, xMax]).range([0, w]).clamp(true)
    : d3.scaleLinear().domain([xMin0, xMax]).range([0, w]);
  const y = logScaleY
    ? d3.scaleLog().domain([1, Math.max(1, yMax)]).range([h, 0]).clamp(true)
    : d3.scaleLinear().domain([0, yMax]).range([h, 0]);

  const nodeColor = (v: number): string => {
    if (threshLo == null) return "#9ca3af";
    if (v <= threshLo) return VIOL_BELOW;
    if (v >= (threshHi ?? Infinity)) return VIOL_ABOVE;
    return VIOL_NEUTRAL;
  };

  g.selectAll("line.lp").data(entries).join("line")
    .attr("class", "lp")
    .attr("x1", (d) => x(d.v))
    .attr("x2", (d) => x(d.v))
    .attr("y1", h)
    .attr("y2", (d) => y(Math.max(1, d.n)))
    .attr("stroke", (d) => nodeColor(d.v))
    .attr("stroke-width", 1.5)
    .attr("opacity", 0.8);

  g.selectAll("circle.lp").data(entries).join("circle")
    .attr("class", "lp")
    .attr("cx", (d) => x(d.v))
    .attr("cy", (d) => y(Math.max(1, d.n)))
    .attr("r", 2.5)
    .attr("fill", "white")
    .attr("stroke", (d) => nodeColor(d.v))
    .attr("stroke-width", 1.2)
    .attr("opacity", 0.9);

  if (threshLo != null && threshHi != null) {
    const xLo = (x as d3.ScaleLinear<number,number> | d3.ScaleLogarithmic<number,number>)(Math.max(xMin0, threshLo));
    const xHi = (x as d3.ScaleLinear<number,number> | d3.ScaleLogarithmic<number,number>)(Math.max(xMin0, threshHi));
    g.append("line").attr("x1", xLo).attr("x2", xLo).attr("y1", 0).attr("y2", h)
      .attr("stroke", VIOL_BELOW).attr("stroke-width", 1.5).attr("stroke-dasharray", "3,2").attr("opacity", 0.9);
    g.append("line").attr("x1", xHi).attr("x2", xHi).attr("y1", 0).attr("y2", h)
      .attr("stroke", VIOL_ABOVE).attr("stroke-width", 1.5).attr("stroke-dasharray", "3,2").attr("opacity", 0.9);
  }

  const axisX = logScaleX ? d3.axisBottom(x as d3.ScaleLogarithmic<number,number>).ticks(5, "d") : d3.axisBottom(x as d3.ScaleLinear<number,number>).ticks(5);
  const axisY = logScaleY ? d3.axisLeft(y as d3.ScaleLogarithmic<number,number>).ticks(3, "s") : d3.axisLeft(y as d3.ScaleLinear<number,number>).ticks(3, "s");
  g.append("g").attr("transform", `translate(0,${h})`).call(axisX.tickSize(3))
    .selectAll("text").attr("font-size", "8px").attr("fill", TEXT_DIM);
  g.append("g").call(axisY.tickSize(3))
    .selectAll("text").attr("font-size", "8px").attr("fill", TEXT_DIM);
  g.selectAll(".domain").attr("stroke", "#e2e8f0");
  g.selectAll(".tick line").attr("stroke", "#e2e8f0");
};

// Histogram for continuous data (edge lengths).
const buildHistogram = (
  containerId: string,
  data: ArrayLike<number>,
  domainMax: number,
  color: string,
  threshLo: number | null,
  threshHi: number | null,
  numBins: number,
  logScaleX: boolean,
  logScaleY: boolean,
  xMinDomain?: number,
  /** How to turn ascending bin edges into per-bin counts. Supplied by the caller because it
   *  knows which cheap representation it holds: `binCountsSorted` over the ascending length
   *  array, or `binCountsFromEntries` over a degree frequency table. Defaults to d3.bin,
   *  which walks every value and is only acceptable on small arrays. */
  binner?: (edges: number[]) => number[],
  /** Largest value in `data`, when the caller already knows it — avoids a scan for the
   *  fallback domain. Only consulted when `domainMax` is falsy. */
  dataMax?: number,
): void => {
  const container = getEl<HTMLDivElement>(containerId);
  container.innerHTML = "";
  if (!data.length) return;

  const W = 240; const H = 70;
  const mg = { top: 4, right: 8, bottom: 18, left: 28 };
  const w = W - mg.left - mg.right;
  const h = H - mg.top - mg.bottom;

  const svg = d3.select(container).append("svg").attr("viewBox", `0 0 ${W} ${H}`);
  const g = svg.append("g").attr("transform", `translate(${mg.left},${mg.top})`);

  const xMax = domainMax || ((dataMax ?? 0) + 1);
  const nBins = Math.max(2, numBins || 15);

  let x: d3.ScaleLinear<number, number> | d3.ScaleLogarithmic<number, number>;
  let threshArray: number[];
  let binDomain: [number, number];

  if (logScaleX) {
    const xMin = Math.max(1, xMinDomain ?? 1);
    x = d3.scaleLog().domain([xMin, xMax]).range([0, w]).clamp(true);
    threshArray = Array.from({ length: nBins - 1 }, (_, i) =>
      Math.exp(Math.log(xMin) + (Math.log(xMax) - Math.log(xMin)) * (i + 1) / nBins));
    binDomain = [xMin, xMax];
  } else {
    const xMin = xMinDomain ?? 0;
    x = d3.scaleLinear().domain([xMin, xMax]).range([0, w]);
    threshArray = Array.from({ length: nBins - 1 }, (_, i) => xMin + (i + 1) * (xMax - xMin) / nBins);
    binDomain = [xMin, xMax];
  }

  // Counted via the caller's `binner` rather than d3.bin, which walks every value. On ocean
  // m2g that is 2,011,528 visits per redraw — and this redraws on every slider `input` event.
  const edgesArr = [binDomain[0], ...threshArray, binDomain[1]];
  const counts = binner
    ? binner(edgesArr)
    : d3.bin<number, number>().domain(binDomain).thresholds(threshArray)(Array.from(data)).map((b) => b.length);
  const bins = counts.map((length, i) => ({ x0: edgesArr[i], x1: edgesArr[i + 1], length }));
  const yMax = d3.max(bins, (d) => d.length) ?? 1;
  const y = logScaleY
    ? d3.scaleLog().domain([1, Math.max(1, yMax)]).range([h, 0]).clamp(true)
    : d3.scaleLinear().domain([0, yMax]).range([h, 0]);

  g.selectAll("rect").data(bins).join("rect")
    .attr("x", (d) => x(Math.max(binDomain[0], d.x0 ?? 0)) + 0.5)
    .attr("y", (d) => y(Math.max(logScaleY ? 1 : 0, d.length)))
    .attr("width", (d) => Math.max(0, x(Math.max(binDomain[0], d.x1 ?? 0)) - x(Math.max(binDomain[0], d.x0 ?? 0)) - 1))
    .attr("height", (d) => h - y(Math.max(logScaleY ? 1 : 0, d.length)))
    .attr("fill", (d) => {
      if (threshLo == null || threshHi == null) return color;
      const mid = ((d.x0 ?? 0) + (d.x1 ?? 0)) / 2;
      if (mid <= threshLo) return VIOL_BELOW;
      if (mid >= threshHi) return VIOL_ABOVE;
      return VIOL_NEUTRAL;
    })
    .attr("opacity", 0.75)
    .attr("rx", 1);

  if (threshLo != null && threshHi != null) {
    const xLo = x(Math.max(binDomain[0], threshLo));
    const xHi = x(Math.max(binDomain[0], threshHi));
    g.append("line").attr("x1", xLo).attr("x2", xLo).attr("y1", 0).attr("y2", h)
      .attr("stroke", VIOL_BELOW).attr("stroke-width", 1.5).attr("stroke-dasharray", "3,2").attr("opacity", 0.9);
    g.append("line").attr("x1", xHi).attr("x2", xHi).attr("y1", 0).attr("y2", h)
      .attr("stroke", VIOL_ABOVE).attr("stroke-width", 1.5).attr("stroke-dasharray", "3,2").attr("opacity", 0.9);
  }

  const axisX = logScaleX
    ? d3.axisBottom(x as d3.ScaleLogarithmic<number,number>).ticks(5, "d")
    : d3.axisBottom(x as d3.ScaleLinear<number,number>).ticks(5);
  g.append("g").attr("transform", `translate(0,${h})`).call(axisX.tickSize(3))
    .selectAll("text").attr("font-size", "8px").attr("fill", TEXT_DIM);
  g.append("g").call(d3.axisLeft(y).ticks(3).tickSize(3))
    .selectAll("text").attr("font-size", "8px").attr("fill", TEXT_DIM);
  g.selectAll(".domain").attr("stroke", "#e2e8f0");
  g.selectAll(".tick line").attr("stroke", "#e2e8f0");
};

const buildViolationHistograms = (): void => {
  const lenBinsEl = document.getElementById("len-bins") as HTMLInputElement | null;
  const lenBins = parseInt(lenBinsEl?.value ?? "", 10) || 15;
  const sg = graphMeta.subgraphs.find((s) => s.key === ST.activeInspectSubgraph);
  if (!sg) return;

  const isSymmetric = sg.src === sg.tgt;
  const edgeWrap = document.getElementById("edge-len-section");
  if (edgeWrap) edgeWrap.style.display = "";

  const label = sg.label.toUpperCase();
  const srcTitle = document.getElementById("src-deg-section-title");
  if (srcTitle) srcTitle.textContent = `SOURCE NODE DEGREE (${label})`;
  const tgtTitle = document.getElementById("tgt-deg-section-title");
  if (tgtTitle) tgtTitle.textContent = `TARGET NODE DEGREE (${label})`;
  const tgtSec = document.getElementById("tgt-deg-section");
  if (tgtSec) tgtSec.style.display = isSymmetric ? "none" : "";
  const tgtDivider = document.getElementById("divider-tgt-deg");
  if (tgtDivider) tgtDivider.style.display = isSymmetric ? "none" : "";
  const symNote = document.getElementById("deg-sym-note");
  if (symNote) symNote.style.display = isSymmetric ? "" : "none";

  // Degrees are integers, but a stem per value only reads as a frequency plot while there are
  // few enough values to draw. Ocean m2g source degree spans [6, 4694] across 381 distinct
  // values; that is a binned distribution, not a set of levels.
  //
  // Both branches read the memoized frequency table rather than the 675k-element degree array:
  // the discrete branch uses it as the stems directly, the binned branch bins it. Neither
  // walks the raw array, which matters because this whole function reruns on every slider tick.
  const derived = getDerivedMetrics();
  const degPlot = (
    id: string, data: DegreeArray, entries: { v: number; n: number }[] | null,
    max: number, min: number,
    lo: number, hi: number, logX: boolean, logY: boolean,
  ): void => {
    const distinct = entries ? entries.length : distinctAtMost(data, MAX_DISCRETE_DEGREES);
    if (distinct !== null && distinct <= MAX_DISCRETE_DEGREES) {
      buildDegreeFreqPlot(id, data, max || 30, lo, hi, logX, logY, min || 0, entries ?? undefined);
    } else {
      // Same neutral bar colour the edge-length histogram uses, so the two read as one family.
      buildHistogram(
        id, data, max || 30, "#9ca3af", lo, hi, DEGREE_HIST_BINS, logX, logY, min || 0,
        entries ? (edges) => binCountsFromEntries(entries, edges) : undefined,
      );
    }
  };

  const srcDeg = getSrcDegThresholds();
  degPlot(
    "viol-degree-hist-src", fullGraphSrcDeg, derived?.srcEntries ?? null,
    ST.srcDegMax, ST.srcDegMin,
    srcDeg.lo, srcDeg.hi, ST.srcDegLogScaleX, ST.srcDegLogScaleY,
  );

  if (!isSymmetric) {
    const tgtDeg = getTgtDegThresholds();
    degPlot(
      "viol-degree-hist-tgt", fullGraphTgtDeg, derived?.tgtEntries ?? null,
      ST.tgtDegMax, ST.tgtDegMin,
      tgtDeg.lo, tgtDeg.hi, ST.tgtDegLogScaleX, ST.tgtDegLogScaleY,
    );
  }

  const lenBinsWrap = document.querySelector<HTMLElement>(".hist-bins-wrap");
  // Decided from the lengths, not from sg.plot. The backend marks every M2M-family subgraph
  // as a lollipop, which is right for a regular mesh but wrong for a clustered one — there the
  // lengths are effectively continuous and the stems become unreadable.
  // Memoized in fetchGraphMetrics — it depends only on the length array, never on the slider
  // positions, and used to re-sample and re-scan 200k values on every tick.
  const { discrete: useLollipop, clusters } = derived?.lengthShape ?? analyzeLengths(fullGraphLengths);
  if (lenBinsWrap) lenBinsWrap.style.display = useLollipop ? "none" : "";

  const lenT = getLenThresholds();
  if (useLollipop) {
    const b = lenChartBounds(ST.lenLogScaleX);
    buildDegreeFreqPlot(
      "viol-edge-hist", [], b.max,
      lenT.lo, lenT.hi, ST.lenLogScaleX, ST.lenLogScaleY, b.min,
      clusters,
    );
  } else {
    const b = lenChartBounds(ST.lenLogScaleX);
    buildHistogram(
      "viol-edge-hist", fullGraphLengths, b.max, "#9ca3af",
      lenT.lo, lenT.hi, lenBins, ST.lenLogScaleX, ST.lenLogScaleY, b.min,
      (edges) => binCountsSorted(fullGraphLengths, edges),
      fullGraphBounds.lengthMax,
    );
  }
};

// Internal: update track gradient + thumb labels for one degree slider pair.
const updateDegSliderPair = (
  prefix: string,
  minSl: HTMLInputElement,
  maxSl: HTMLInputElement,
  sliderMax: number,
  logMode: boolean,
  triggerFetch = true,
): void => {
  let slVal: number, shVal: number, p1: number, p2: number;
  let loActual: number, hiActual: number;
  let loLbl: string, hiLbl: string;

  if (logMode) {
    slVal = parseFloat(minSl.value); shVal = parseFloat(maxSl.value);
    loActual = Math.max(1, Math.round(Math.exp(slVal)));
    hiActual = Math.max(1, Math.round(Math.exp(shVal)));
    slVal = Math.log(loActual); shVal = Math.log(hiActual);
    if (slVal > shVal) { slVal = shVal; loActual = hiActual; }
    minSl.value = slVal.toFixed(4); maxSl.value = shVal.toFixed(4);
    p1 = thumbCenterPct(slVal, sliderMax, minSl);
    p2 = thumbCenterPct(shVal, sliderMax, maxSl);
    loLbl = String(loActual); hiLbl = String(hiActual);
  } else {
    loActual = parseInt(minSl.value, 10); hiActual = parseInt(maxSl.value, 10);
    if (loActual > hiActual) { loActual = hiActual; minSl.value = String(loActual); }
    p1 = thumbCenterPct(loActual, sliderMax, minSl);
    p2 = thumbCenterPct(hiActual, sliderMax, maxSl);
    loLbl = String(loActual); hiLbl = String(hiActual);
  }

  getEl<HTMLSpanElement>(`${prefix}-deg-min-lbl-bot`).textContent = `Min: ${loActual}`;
  getEl<HTMLSpanElement>(`${prefix}-deg-max-lbl-bot`).textContent = `Max: ${hiActual}`;
  updateTrackGradient(`${prefix}-deg-track`, p1, p2);
  updateThumbLabel(`${prefix}-deg-lbl-min`, p1, loLbl);
  updateThumbLabel(`${prefix}-deg-lbl-max`, p2, hiLbl);

  buildViolationHistograms();
  if (triggerFetch && ST.violation) {
    computeViolationTotals();
    updateViolationStats();
    renderLayers();
    commitThresholdState();
  }
};

const updateSrcDegSliders = (triggerFetch = true): void => {
  const minSl = getEl<HTMLInputElement>("src-deg-min-sl");
  const maxSl = getEl<HTMLInputElement>("src-deg-max-sl");
  const slMax = ST.srcDegLogScaleX ? Math.log(Math.max(1, ST.srcDegMax || 30)) : (ST.srcDegMax || 30);
  updateDegSliderPair("src", minSl, maxSl, slMax, ST.srcDegLogScaleX, triggerFetch);
};

const updateTgtDegSliders = (triggerFetch = true): void => {
  const minSl = getEl<HTMLInputElement>("tgt-deg-min-sl");
  const maxSl = getEl<HTMLInputElement>("tgt-deg-max-sl");
  const slMax = ST.tgtDegLogScaleX ? Math.log(Math.max(1, ST.tgtDegMax || 30)) : (ST.tgtDegMax || 30);
  updateDegSliderPair("tgt", minSl, maxSl, slMax, ST.tgtDegLogScaleX, triggerFetch);
};

const updateLenSliders = (triggerFetch = true): void => {
  const lenMinSl = getEl<HTMLInputElement>("len-min-sl");
  const lenMaxSl = getEl<HTMLInputElement>("len-max-sl");
  const slMax = parseFloat(lenMaxSl.max) || (ST.lenSliderMax || 600);
  let lo: number, hi: number, slLo: number, slHi: number;

  if (ST.lenLogScaleX) {
    slLo = parseFloat(lenMinSl.value); slHi = parseFloat(lenMaxSl.value);
    lo = Math.max(1, Math.round(Math.exp(slLo)));
    hi = Math.max(1, Math.round(Math.exp(slHi)));
    slLo = Math.log(lo); slHi = Math.log(hi);
    if (slLo > slHi) { slLo = slHi; lo = hi; }
    lenMinSl.value = slLo.toFixed(4); lenMaxSl.value = slHi.toFixed(4);
  } else {
    lo = parseFloat(lenMinSl.value); hi = parseFloat(lenMaxSl.value);
    if (lo > hi) { lo = hi; lenMinSl.value = String(lo); }
    slLo = lo; slHi = hi;
  }

  getEl<HTMLSpanElement>("len-min-lbl-bot").textContent = `Min: ${lo} km`;
  getEl<HTMLSpanElement>("len-max-lbl-bot").textContent = `Max: ${hi} km`;
  const p1 = thumbCenterPct(slLo, slMax, lenMinSl);
  const p2 = thumbCenterPct(slHi, slMax, lenMaxSl);
  updateTrackGradient("len-track", p1, p2);
  updateThumbLabel("len-lbl-min", p1, lo);
  updateThumbLabel("len-lbl-max", p2, hi);

  buildViolationHistograms();
  if (triggerFetch && ST.violation) {
    computeViolationTotals();
    updateViolationStats();
    renderLayers();
    commitThresholdState();
  }
};

const lenChartBounds = (logMode: boolean): { min: number; max: number } => {
  const b = degBoundsFromExtremes(
    fullGraphLengths.length, fullGraphBounds.lengthMin, fullGraphBounds.lengthMax, logMode,
  );
  if (logMode) return b;
  return { min: Math.floor(b.min), max: Math.ceil(b.max) };
};

// Thresholds are stored per graph *and* per subgraph — the same subgraph key means
// different degree/length ranges in a different graph.
const thresholdStorageKey = (key: SubgraphKey): string => `gnn_viol_thresholds:${activeGraphId}:${key}`;

// Read the live slider values.
const captureThresholdState = (): PerSubgraphThresholds => {
  const src = getSrcDegThresholds();
  const tgt = getTgtDegThresholds();
  const len = getLenThresholds();
  return {
    srcDegLo: src.lo, srcDegHi: src.hi,
    tgtDegLo: tgt.lo, tgtDegHi: tgt.hi,
    lengthLo: len.lo, lengthHi: len.hi,
  };
};

const persistThresholdState = (key: SubgraphKey): void => {
  const t = thresholdCache[key];
  if (!t) return;
  try {
    localStorage.setItem(thresholdStorageKey(key), JSON.stringify(t));
  } catch { /* ignore */ }
};

const saveThresholdState = (key: SubgraphKey): void => {
  thresholdCache[key] = captureThresholdState();
  persistThresholdState(key);
};

// Commit the active subgraph's thresholds after a user edit, so they survive leaving
// inspect mode and (via localStorage) a reload. Slider drags fire an event per pixel,
// so the in-memory cache updates immediately while the storage write is debounced.
let thresholdPersistTimer: ReturnType<typeof setTimeout> | null = null;
const commitThresholdState = (): void => {
  if (!violMetricsReady) return; // sliders don't hold this subgraph's values yet
  const key = ST.activeInspectSubgraph;
  const t = captureThresholdState();
  thresholdCache[key] = t;
  // Resolve the storage key now: a pending write must land on the graph/subgraph that
  // was active when the edit happened, even if either changes before it fires.
  const storageKey = thresholdStorageKey(key);
  if (thresholdPersistTimer) clearTimeout(thresholdPersistTimer);
  thresholdPersistTimer = setTimeout(() => {
    thresholdPersistTimer = null;
    try { localStorage.setItem(storageKey, JSON.stringify(t)); } catch { /* ignore */ }
  }, 300);
};

// Restore thresholds persisted by a previous session. Returns null when nothing valid
// is stored, so the caller falls back to the metric bounds.
const loadThresholdState = (key: SubgraphKey): PerSubgraphThresholds | null => {
  let raw: string | null = null;
  try {
    raw = localStorage.getItem(thresholdStorageKey(key));
  } catch { return null; }
  if (!raw) return null;
  try {
    const p = JSON.parse(raw) as Partial<Record<keyof PerSubgraphThresholds, unknown>>;
    const fields: (keyof PerSubgraphThresholds)[] = [
      "srcDegLo", "srcDegHi", "tgtDegLo", "tgtDegHi", "lengthLo", "lengthHi",
    ];
    const t = {} as PerSubgraphThresholds;
    for (const f of fields) {
      const v = p[f];
      if (typeof v !== "number" || !Number.isFinite(v)) return null;
      t[f] = v;
    }
    return t;
  } catch {
    return null;
  }
};

// Clamp restored thresholds into the current metric bounds — a stored value can fall
// outside them if the graph data changed since it was saved.
const clampThresholdState = (
  t: PerSubgraphThresholds,
  srcB: { min: number; max: number },
  tgtB: { min: number; max: number },
  lenB: { min: number; max: number },
): PerSubgraphThresholds => {
  const fit = (v: number, b: { min: number; max: number }) => Math.min(b.max, Math.max(b.min, v));
  const srcLo = fit(t.srcDegLo, srcB), srcHi = fit(t.srcDegHi, srcB);
  const tgtLo = fit(t.tgtDegLo, tgtB), tgtHi = fit(t.tgtDegHi, tgtB);
  const lenLo = fit(t.lengthLo, lenB), lenHi = fit(t.lengthHi, lenB);
  return {
    srcDegLo: Math.min(srcLo, srcHi), srcDegHi: srcHi,
    tgtDegLo: Math.min(tgtLo, tgtHi), tgtDegHi: tgtHi,
    lengthLo: Math.min(lenLo, lenHi), lengthHi: lenHi,
  };
};

const applyThresholdState = (t: PerSubgraphThresholds): void => {
  const srcB = degBoundsGlobal("src", ST.srcDegLogScaleX);
  ST.srcDegMin = srcB.min; ST.srcDegMax = srcB.max;
  const tgtB = degBoundsGlobal("tgt", ST.tgtDegLogScaleX);
  ST.tgtDegMin = tgtB.min; ST.tgtDegMax = tgtB.max;

  setDegSliderMode(
    getEl("src-deg-min-sl"), getEl("src-deg-max-sl"),
    ST.srcDegMin, ST.srcDegMax, t.srcDegLo, t.srcDegHi, ST.srcDegLogScaleX,
  );
  setDegSliderMode(
    getEl("tgt-deg-min-sl"), getEl("tgt-deg-max-sl"),
    ST.tgtDegMin, ST.tgtDegMax, t.tgtDegLo, t.tgtDegHi, ST.tgtDegLogScaleX,
  );
  const lenB = lenChartBounds(ST.lenLogScaleX);
  setDegSliderMode(
    getEl("len-min-sl"), getEl("len-max-sl"),
    lenB.min, lenB.max, t.lengthLo, t.lengthHi, ST.lenLogScaleX,
  );

  updateSrcDegSliders(false);
  updateTgtDegSliders(false);
  updateLenSliders(false);
};

const toggleSrcDegLogScaleX = (): void => {
  const prev = getSrcDegThresholds();
  ST.srcDegLogScaleX = !ST.srcDegLogScaleX;
  const b = degBoundsGlobal("src", ST.srcDegLogScaleX);
  ST.srcDegMin = b.min; ST.srcDegMax = b.max;
  const btn = document.getElementById("src-deg-logx-btn");
  if (btn) btn.classList.toggle("active", ST.srcDegLogScaleX);
  const minSl = getEl<HTMLInputElement>("src-deg-min-sl");
  const maxSl = getEl<HTMLInputElement>("src-deg-max-sl");
  setDegSliderMode(minSl, maxSl, ST.srcDegMin, ST.srcDegMax, prev.lo, prev.hi, ST.srcDegLogScaleX);
  updateSrcDegSliders(false);
};

const toggleSrcDegLogScaleY = (): void => {
  ST.srcDegLogScaleY = !ST.srcDegLogScaleY;
  const btn = document.getElementById("src-deg-logy-btn");
  if (btn) btn.classList.toggle("active", ST.srcDegLogScaleY);
  buildViolationHistograms();
};

const toggleTgtDegLogScaleX = (): void => {
  const prev = getTgtDegThresholds();
  ST.tgtDegLogScaleX = !ST.tgtDegLogScaleX;
  const b = degBoundsGlobal("tgt", ST.tgtDegLogScaleX);
  ST.tgtDegMin = b.min; ST.tgtDegMax = b.max;
  const btn = document.getElementById("tgt-deg-logx-btn");
  if (btn) btn.classList.toggle("active", ST.tgtDegLogScaleX);
  const minSl = getEl<HTMLInputElement>("tgt-deg-min-sl");
  const maxSl = getEl<HTMLInputElement>("tgt-deg-max-sl");
  setDegSliderMode(minSl, maxSl, ST.tgtDegMin, ST.tgtDegMax, prev.lo, prev.hi, ST.tgtDegLogScaleX);
  updateTgtDegSliders(false);
};

const toggleTgtDegLogScaleY = (): void => {
  ST.tgtDegLogScaleY = !ST.tgtDegLogScaleY;
  const btn = document.getElementById("tgt-deg-logy-btn");
  if (btn) btn.classList.toggle("active", ST.tgtDegLogScaleY);
  buildViolationHistograms();
};

const toggleLenLogScaleX = (): void => {
  const prevLen = getLenThresholds();
  ST.lenLogScaleX = !ST.lenLogScaleX;
  const btn = document.getElementById("len-logx-btn");
  if (btn) btn.classList.toggle("active", ST.lenLogScaleX);
  const lenMinSl = getEl<HTMLInputElement>("len-min-sl");
  const lenMaxSl = getEl<HTMLInputElement>("len-max-sl");
  const b = lenChartBounds(ST.lenLogScaleX);
  setDegSliderMode(lenMinSl, lenMaxSl, b.min, b.max, prevLen.lo, prevLen.hi, ST.lenLogScaleX);
  updateLenSliders(false);
  saveLenPlotCookie();
};

const toggleLenLogScaleY = (): void => {
  ST.lenLogScaleY = !ST.lenLogScaleY;
  const btn = document.getElementById("len-logy-btn");
  if (btn) btn.classList.toggle("active", ST.lenLogScaleY);
  buildViolationHistograms();
  saveLenPlotCookie();
};

const setViolGraphType = (type: SubgraphKey): void => {
  // Only save when actually switching — not when re-entering the same subgraph
  // (entering inspect mode calls this with the same key; DOM sliders hold HTML defaults at that point).
  // Also skip when the outgoing subgraph's metrics never landed: the sliders then still hold
  // another subgraph's (or the HTML default) values, and caching those would poison its thresholds.
  if (type !== ST.activeInspectSubgraph) {
    if (violMetricsReady) saveThresholdState(ST.activeInspectSubgraph);
    violMetricsReady = false; // suppress violation colors until fetchGraphMetrics completes for new subgraph
  }
  ST.activeInspectSubgraph = type;
  _setActiveInspectSubgraphBridge.set?.(type);

  const sg = graphMeta.subgraphs.find((s) => s.key === type);
  if (sg) {
    ST.activeSubgraphs = new Set([type]);
    const activeNodeSet = new Set<string>([sg.src, sg.tgt]);
    ST.activeNodeTypes = activeNodeSet;
    ST.showGrid = activeNodeSet.has("grid");
    ST.showMesh = [...activeNodeSet].some((k) => k === "mesh" || k.startsWith("mesh_"));
    ST.showG2M = type === "g2m";
    ST.showM2M = type === "m2m" || type.startsWith("m2m_");
    ST.showM2G = type === "m2g";
  }

  // Refresh meshNodes from cache before rendering so the first renderLayers() call
  // sees current data rather than whatever was left from the previous mode/subgraph.
  if (Object.keys(cachedMeshByKey).length > 0) {
    refreshVisibleDataFromCache(prevBbox ?? getBbox());
  }
  // No-op in violation mode until fetchGraphMetrics settles — see the guard in renderLayers.
  renderLayers();
  prevBbox = null;
  scheduleFetch();
  void fetchGraphMetrics(type);
};

/** Header of the binary /graph_metrics container. See get_graph_metrics_route in
 *  backend/src/routes/graph.py for the byte layout. */
type MetricsArraySpec = { offset: number; count: number; dtype: "uint16" | "uint32" | "float32" };
type MetricsHeader = {
  arrays: { src_degrees: MetricsArraySpec; tgt_degrees: MetricsArraySpec; lengths: MetricsArraySpec };
  degree_lo: number; degree_hi: number;
  tgt_degree_lo: number; tgt_degree_hi: number;
  length_lo: number; length_hi: number;
  src_degree_min: number; src_degree_max: number;
  tgt_degree_min: number; tgt_degree_max: number;
  length_min: number; length_max: number;
};

/** A view over `buf`, copied only when it has to be.
 *
 *  A TypedArray constructor throws `RangeError: start offset ... should be a multiple of N`
 *  when the byte offset is not element-aligned, and the throw propagates out of
 *  fetchGraphMetrics — which is why an unaligned subgraph used to leave the Inspect panel
 *  showing the previous subgraph's distributions and the map in explore colours.
 *
 *  The backend now pads every array onto a 4-byte offset, so the aligned branch is the one
 *  that runs. The copy stays as a fallback: a frontend talking to an older backend should be
 *  a little slower, not blank. */
const alignedView = <T extends Uint16Array | Uint32Array | Float32Array>(
  Ctor: {
    new (buf: ArrayBuffer, byteOffset?: number, length?: number): T;
    BYTES_PER_ELEMENT: number;
  },
  buf: ArrayBuffer,
  byteOffset: number,
  count: number,
): T => {
  const bpe = Ctor.BYTES_PER_ELEMENT;
  if (byteOffset % bpe === 0) return new Ctor(buf, byteOffset, count);
  console.warn(
    `graph_metrics: array at byte ${byteOffset} is not ${bpe}-byte aligned — copying. ` +
    "The backend should be padding this; see get_graph_metrics_route.",
  );
  return new Ctor(buf.slice(byteOffset, byteOffset + count * bpe));
};

/** Splits the container into its header and typed-array views.
 *  The views alias `buf` rather than copying it, so the whole response costs one allocation. */
const decodeMetricsBinary = (buf: ArrayBuffer): { header: MetricsHeader; arrays: {
  srcDeg: DegreeArray; tgtDeg: DegreeArray; lengths: Float32Array;
} } => {
  const headerLen = new DataView(buf).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, headerLen))) as MetricsHeader;
  const base = 4 + headerLen;
  const degView = (spec: MetricsArraySpec): DegreeArray =>
    spec.dtype === "uint32"
      ? alignedView(Uint32Array, buf, base + spec.offset, spec.count)
      : alignedView(Uint16Array, buf, base + spec.offset, spec.count);
  return {
    header,
    arrays: {
      srcDeg:  degView(header.arrays.src_degrees),
      tgtDeg:  degView(header.arrays.tgt_degrees),
      lengths: alignedView(
        Float32Array, buf, base + header.arrays.lengths.offset, header.arrays.lengths.count,
      ),
    },
  };
};

const fetchGraphMetrics = async (graphType: SubgraphKey): Promise<void> => {
  try {
    const r = await fetch(`${API}/graphs/${activeGraphId}/graph_metrics/${graphType}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const { header, arrays } = decodeMetricsBinary(await r.arrayBuffer());

    // header.degree_lo/_hi and length_lo/_hi (the 2-sigma suggestions) are deliberately not
    // read here: the slider defaults come from the padded chart bounds below, which is what
    // this function has always used. They stay on the wire for callers that want them.

    // Entering inspect mode fires two subgraph selections in quick succession (the mode-tab
    // handler, then the sidebar switching to radio mode), so two fetches can be in flight.
    // Discard any response that is no longer for the active subgraph — otherwise it overwrites
    // the metric arrays and sliders with another subgraph's data.
    if (graphType !== ST.activeInspectSubgraph) return;

    fullGraphSrcDeg = arrays.srcDeg;
    fullGraphTgtDeg = arrays.tgtDeg;
    fullGraphLengths = arrays.lengths;   // ascending — see the DegreeArray declaration
    fullGraphBounds = {
      srcDegMin: header.src_degree_min, srcDegMax: header.src_degree_max,
      tgtDegMin: header.tgt_degree_min, tgtDegMax: header.tgt_degree_max,
      lengthMin: header.length_min,     lengthMax: header.length_max,
    };

    // Everything that depends only on the arrays is computed here, once, so that no slider
    // handler ever walks them again.
    derivedMetrics = {
      key: graphType,
      srcEntries:  degreeEntries(fullGraphSrcDeg),
      tgtEntries:  degreeEntries(fullGraphTgtDeg),
      lengthShape: analyzeLengths(fullGraphLengths),
    };

    const srcB = degBoundsGlobal("src", ST.srcDegLogScaleX);
    ST.srcDegMin = srcB.min; ST.srcDegMax = srcB.max;
    const tgtB = degBoundsGlobal("tgt", ST.tgtDegLogScaleX);
    ST.tgtDegMin = tgtB.min; ST.tgtDegMax = tgtB.max;
    ST.lenSliderMax = (() => {
      if (!fullGraphLengths.length) return 600;
      const hi = fullGraphBounds.lengthMax;
      const lo = fullGraphBounds.lengthMin;
      return hi + Math.max(1, Math.round(0.05 * (hi - lo)));
    })();

    const lenB = lenChartBounds(ST.lenLogScaleX);
    const serverDefaults: PerSubgraphThresholds = {
      srcDegLo: srcB.min,
      srcDegHi: srcB.max,
      tgtDegLo: tgtB.min,
      tgtDegHi: tgtB.max,
      lengthLo: lenB.min,
      lengthHi: lenB.max,
    };
    const stored = thresholdCache[graphType] ?? loadThresholdState(graphType);
    const threshold: PerSubgraphThresholds = stored
      ? clampThresholdState(stored, srcB, tgtB, lenB)
      : { ...serverDefaults };
    thresholdCache[graphType] = threshold;
    applyThresholdState(threshold);
    violMetricsReady = true;
    buildViolationHistograms();
    if (ST.violation) {
      computeViolationTotals();
      updateViolationStats();
      renderLayers();
    }
  } catch (err) {
    console.warn("fetchGraphMetrics failed (backend offline?):", err);
    // renderLayers suppresses itself while violMetricsReady is false, so without force the
    // previous subgraph would stay on screen indefinitely. Colours fall back to explore mode,
    // which is honest here: with no thresholds there is genuinely nothing to classify against.
    if (graphType === ST.activeInspectSubgraph) renderLayers(true);
  }
};

// Exact counts, but from the sorted lengths and the degree frequency tables rather than six
// full scans. On ocean m2g that is ~5.4M element visits per slider tick replaced by a handful
// of binary searches; the numbers themselves are unchanged.
const computeViolationTotals = (): void => {
  if (!fullGraphSrcDeg.length) return;
  const s = getSrcDegThresholds();
  const t = getTgtDegThresholds();
  const l = getLenThresholds();
  const d = getDerivedMetrics();

  if (d) {
    violTotals.srcLo = entriesCountLE(d.srcEntries, s.lo);
    violTotals.srcHi = entriesCountGE(d.srcEntries, s.hi);
    violTotals.tgtLo = entriesCountLE(d.tgtEntries, t.lo);
    violTotals.tgtHi = entriesCountGE(d.tgtEntries, t.hi);
  } else {
    // Metrics for another subgraph are still in flight — fall back to a direct pass so the
    // panel stays correct rather than stale.
    violTotals.srcLo = countMatching(fullGraphSrcDeg, (v) => v <= s.lo);
    violTotals.srcHi = countMatching(fullGraphSrcDeg, (v) => v >= s.hi);
    violTotals.tgtLo = countMatching(fullGraphTgtDeg, (v) => v <= t.lo);
    violTotals.tgtHi = countMatching(fullGraphTgtDeg, (v) => v >= t.hi);
  }

  violTotals.edgeLo = countLE(fullGraphLengths, l.lo);
  violTotals.edgeHi = countGE(fullGraphLengths, l.hi);
};

const updatePill = (id: string, on: boolean): void => {
  const el = getEl<HTMLDivElement>(id);
  el.classList.toggle("on", on);
  el.classList.toggle("off", !on);
};

const applyTheme = (dark: boolean): void => {
  const app = document.getElementById("app");
  if (!app) return;
  app.classList.toggle("dark", dark);
  // Swap base map tiles between light/dark CARTO when on the map style
  if (currentMapType === "map") {
    const url = dark ? DARK_MAP_URL : MAP_TYPES.map.url;
    const filter = dark ? DARK_MAP_FILTER : MAP_TYPES.map.filter;
    if (leafMap && activeTileLayer) {
      leafMap.removeLayer(activeTileLayer);
      activeTileLayer = L.tileLayer(resolveMapUrl(url), { attribution: MAP_TYPES.map.attr, ...MAP_TYPES.map.opts }).addTo(leafMap);
      activeTileLayer.bringToBack();
    }
    if (miniMap && activeMiniTileLayer) {
      miniMap.removeLayer(activeMiniTileLayer);
      activeMiniTileLayer = L.tileLayer(resolveMapUrl(url), { ...MAP_TYPES.map.opts, }).addTo(miniMap);
      activeMiniTileLayer.bringToBack();
    }
    document.querySelectorAll<HTMLElement>("#map .leaflet-tile-pane, #minimap-map .leaflet-tile-pane")
      .forEach((p) => { p.style.filter = filter; });
  }
};

const setThemeMode = (mode: ThemeMode): void => {
  currentThemeMode = mode;
  document.querySelectorAll<HTMLButtonElement>(".theme-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.theme === mode);
  });
  applyTheme(mode === "dark" || (mode === "auto" && (systemDarkMq !== null ? systemDarkMq.matches : false)));
};

const setMapType = (type: string): void => {
  if (type === currentMapType) return;
  const cfg = MAP_TYPES[type];
  leafMap.removeLayer(activeTileLayer);
  activeTileLayer = L.tileLayer(resolveMapUrl(cfg.url), { attribution: cfg.attr, ...cfg.opts }).addTo(leafMap);
  activeTileLayer.bringToBack();

  miniMap.removeLayer(activeMiniTileLayer);
  activeMiniTileLayer = L.tileLayer(resolveMapUrl(cfg.url), { ...cfg.opts, }).addTo(miniMap);
  activeMiniTileLayer.bringToBack();

  document.querySelectorAll<HTMLElement>("#map .leaflet-tile-pane, #minimap-map .leaflet-tile-pane").forEach((pane) => {
    pane.style.filter = cfg.filter;
  });

  activeTileLayer.once("load", () => {
    const pane = document.querySelector<HTMLElement>("#map .leaflet-tile-pane");
    if (pane) pane.style.filter = cfg.filter;
  });

  document.querySelectorAll<HTMLButtonElement>(".mt-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.type === type);
  });

  currentMapType = type;
};

const placeTip = (cx: number, cy: number): void => {
  const mw = getEl<HTMLDivElement>("map-wrap");
  const tip = getEl<HTMLDivElement>("tip");
  const r = mw.getBoundingClientRect();
  let tx = cx - r.left + 14;
  let ty = cy - r.top + 14;
  if (tx + 210 > mw.clientWidth) tx = cx - r.left - 220;
  if (ty + 140 > mw.clientHeight) ty = cy - r.top - 150;
  tip.style.left = `${tx}px`;
  tip.style.top = `${ty}px`;
  tip.style.display = "block";
};

const showNodeTooltip = (cx: number, cy: number, n: NodeDatum): void => {
  const deg = computeDegrees()[n.id] ?? 0;
  const tooltipSg = graphMeta.subgraphs.find((s) => s.key === ST.activeInspectSubgraph);
  const isSrcNode = tooltipSg ? n.type === tooltipSg.src : true;
  const fullDeg = (isSrcNode ? fullGraphSrcDeg : fullGraphTgtDeg)[n.id] ?? 0;
  const thresholds = isSrcNode ? getSrcDegThresholds() : getTgtDegThresholds();
  const isHi = ST.violation && fullDeg >= thresholds.hi;
  const isLo = ST.violation && fullDeg <= thresholds.lo;
  const title = getEl<HTMLDivElement>("tip-title");
  const body = getEl<HTMLDivElement>("tip-body");

  title.textContent = `${n.type === "grid" ? "Grid" : "Mesh"} Node ${n.id}`;
  title.style.color = n.type === "grid" ? NODE_GRID : NODE_MESH;

  body.innerHTML =
    `<div class=\"tip-row\"><span>Position</span><span class=\"tip-val\">${n.lat.toFixed(2)}°N, ${n.lon.toFixed(2)}°E</span></div>` +
    `<div class=\"tip-row\"><span>Degree</span><span class=\"tip-val\">${deg}</span></div>` +
    (isHi ? '<div class="tip-row"><span style="color:var(--viol-hi)">⚠ Over threshold</span></div>' : "") +
    (isLo ? '<div class="tip-row"><span style="color:var(--viol-lo)">⚠ Under threshold</span></div>' : "");

  placeTip(cx, cy);
};

const fetchAndRender = async (): Promise<void> => {
  fetchGen++;
  const myGen = fetchGen;
  setLoading(true);
  try {
    const bbox = getBbox();

    if (isBboxCoveredByCache(bbox)) {
      refreshVisibleDataFromCache(bbox);
      prevBbox = bbox;
      if (fetchGen !== myGen) { setLoading(false); return; }
      updateHeaderCounts();
      updateGraphStats();
      setLoading(false);
      renderLayers();
      if (ST.violation) { computeViolationTotals(); updateViolationStats(); }
      return;
    }

    const prev = prevBbox;
    const params = bboxParams(bbox, prev).toString();

    // Set by the catch handlers below. A bbox whose fetch was only partly served must not
    // be recorded in fetchedBboxes: that list is what isBboxCoveredByCache consults, so
    // marking it would pin the gap in place until the user pans somewhere new. Leaving it
    // out costs one repeated fetch and lets a transient failure heal on the next view change.
    let fetchIncomplete = false;

    // A failing subgraph yields no edges for that subgraph, not a failed render.
    //
    // Both promise groups below are awaited through Promise.all, which rejects as soon as
    // any one member does, so without this an error on a single /edges call aborts the whole
    // fetch: nodes included, every other subgraph included, and the map draws nothing. That
    // is the wrong failure shape for a tool whose subject is graphs with structural defects
    // -- the one subgraph the backend cannot serve is exactly the one worth looking at the
    // rest of the graph around. An error body is also not an array, so buildEdges would
    // throw on it a step later regardless of the HTTP status.
    const fetchEdge = (key: ViolGraphType): Promise<EdgeDatum[]> => {
      return fetch(`${API}/graphs/${activeGraphId}/edges/${key}?${params}`)
        .then((r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.json() as Promise<ApiEdge[]>;
        })
        .then(buildEdges)
        .catch((err) => {
          console.warn(`edges/${key} failed — subgraph omitted from this fetch:`, err);
          fetchIncomplete = true;
          return [] as EdgeDatum[];
        });
    };

    const nodeTypes = graphMeta.node_types.length > 0 ? graphMeta.node_types : ["grid", "mesh"];
    const subgraphKeys = graphMeta.subgraphs.length > 0
      ? graphMeta.subgraphs.map((sg) => sg.key)
      : ["g2m", "m2m", "m2g"];

    const nodePromises = nodeTypes.map((nt) =>
      fetch(`${API}/graphs/${activeGraphId}/nodes/${nt}?${params}`)
        .then((r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.json() as Promise<NodeDatum[]>;
        })
        .then((nodes) => ({ nt, nodes }))
        // Same reasoning as fetchEdge: one unservable node type must not blank the map.
        .catch((err) => {
          console.warn(`nodes/${nt} failed — node type omitted from this fetch:`, err);
          fetchIncomplete = true;
          return { nt, nodes: [] as NodeDatum[] };
        }),
    );
    const edgePromises = subgraphKeys.map((key) =>
      fetchEdge(key).then((edges) => ({ key, edges })),
    );

    const [nodeResults, edgeResults] = await Promise.all([
      Promise.all(nodePromises),
      Promise.all(edgePromises),
    ]);

    nodeResults.forEach(({ nt, nodes }) => {
      nodes.forEach((n) => {
        const node = { ...n, type: nt };
        if (nt === "grid") {
          cachedGridById[node.id] = node;
        } else {
          cachedMeshByKey[`${nt}:${node.id}`] = node;
        }
      });
    });
    edgeResults.forEach(({ key, edges }) => {
      if (!edgesCache[key]) edgesCache[key] = {};
      edges.forEach((e) => { edgesCache[key][edgeKey(e)] = e; });
    });
    if (!fetchIncomplete) fetchedBboxes.push(bbox);
    bumpCacheEpoch();

    refreshVisibleDataFromCache(bbox);

    if (fetchGen !== myGen) { setLoading(false); return; }
    prevBbox = bbox;
    updateHeaderCounts();
    updateGraphStats();
  } catch (err) {
    console.error("fetchAndRender:", err);
    setLoading(false);
    return;
  }
  setLoading(false);
  renderLayers();
  if (ST.violation) { computeViolationTotals(); updateViolationStats(); }
};

const scheduleFetch = (): void => {
  if (currentMode === "explain") return;
  if (fetchTimer) clearTimeout(fetchTimer);
  fetchTimer = setTimeout(() => {
    void fetchAndRender();
  }, FETCH_DEBOUNCE_MS);
};

const fetchGraphMeta = async (): Promise<void> => {
  // Unlike the per-subgraph fetches in fetchAndRender, a failure here is whole-graph and is
  // rethrown rather than swallowed.
  //
  // Those handlers are deliberately tolerant: one unservable subgraph must still let the rest
  // of the graph draw, because a structurally broken subgraph is exactly what this tool exists
  // to look at. Meta is different — it carries node_types and the subgraph keys, so losing it
  // leaves fetchAndRender with nothing to request and the map draws nothing at all. Swallowed,
  // that is indistinguishable from an empty viewport, which is how a preset whose folder never
  // made it onto the deployment PVC presents: the picker lists it (seed_db keeps preset rows
  // even when the folder is absent), selecting it returns 503 on every route, and the user sees
  // a blank map with no explanation. Throwing puts the backend's message in front of them.
  const r = await fetch(`${API}/graphs/${activeGraphId}/meta`);
  if (!r.ok) {
    throw new Error(await readUploadError(r));
  }
  try {
    const data = (await r.json()) as Partial<GraphStatsMeta>;
    graphMeta.n_grid = data.n_grid ?? 0;
    graphMeta.grid_spacing_deg = data.grid_spacing_deg ?? 0;
    graphMeta.n_mesh = data.n_mesh ?? 0;
    graphMeta.n_edges_g2m = data.n_edges_g2m ?? 0;
    graphMeta.n_edges_m2m = data.n_edges_m2m ?? 0;
    graphMeta.n_edges_m2g = data.n_edges_m2g ?? 0;
    graphMeta.is_hierarchical = data.is_hierarchical ?? false;
    graphMeta.n_levels = data.n_levels ?? 1;
    graphMeta.node_types = data.node_types ?? [];
    graphMeta.node_counts = data.node_counts ?? {};
    graphMeta.subgraphs = data.subgraphs ?? [];
    updateHeaderCounts();
    // Initialize activeSubgraphs from server metadata (show all by default)
    if (graphMeta.subgraphs.length > 0) {
      ST.activeSubgraphs = new Set(graphMeta.subgraphs.map((sg) => sg.key));
    } else {
      ST.activeSubgraphs = new Set(["g2m", "m2m", "m2g"]);
    }
    if (graphMeta.node_types.length > 0) {
      ST.activeNodeTypes = new Set(graphMeta.node_types);
    } else {
      ST.activeNodeTypes = new Set(["grid", "mesh"]);
    }
    ST.showG2M = ST.activeSubgraphs.has("g2m");
    ST.showM2M = ST.activeSubgraphs.has("m2m") || [...ST.activeSubgraphs].some((k) => k.startsWith("m2m_"));
    ST.showM2G = ST.activeSubgraphs.has("m2g");
    ST.showGrid = ST.activeNodeTypes.has("grid");
    ST.showMesh = [...ST.activeNodeTypes].some((k) => k === "mesh" || k.startsWith("mesh_"));
    _savedExploreSubgraphs = new Set(ST.activeSubgraphs);
    _savedExploreNodeTypes = new Set(ST.activeNodeTypes);
    _setMetaLevelsBridge.set?.(graphMeta.n_levels);
    _setMetaNodeCountsBridge.set?.(graphMeta.node_counts);
  } catch (err) {
    console.warn("fetchGraphMeta failed:", err);
  }
};

const initDeck = (): void => {
  const c = leafMap.getCenter();
  const sz = leafMap.getSize();
  deckInstance = new Deck({
    canvas: deckCanvas,
    width: sz.x,
    height: sz.y,
    controller: false,
    deviceProps: { type: "webgl" },
    views: [new MapView({ id: "map", repeat: true })],
    initialViewState: {
      map: {
        longitude: c.lng,
        latitude: c.lat,
        zoom: leafMap.getZoom() - 1,
        pitch: 0,
        bearing: 0,
      },
    },
    layers: [],
    onLoad: () => {
      resizeDeck();
      deckReady = true;
      syncDeckView();
      renderLayers();
    },
  });
};

const boot = async (): Promise<void> => {
  initDeck();
  syncDeckView();
  await fetchGraphMeta();
  await fetchAndRender();
};

const bindUIHandlers = (): (() => void) => {
  const listeners: Array<() => void> = [];

  const on = <K extends keyof HTMLElementEventMap>(
    el: HTMLElement,
    type: K,
    fn: (ev: HTMLElementEventMap[K]) => void,
  ): void => {
    const handler = fn as EventListener;
    el.addEventListener(type, handler);
    listeners.push(() => el.removeEventListener(type, handler));
  };

  const setAppMode = (mode: "explore" | "inspect" | "explain") => {
    const wasExplain = currentMode === "explain";
    currentMode = mode;

    // ── mode tab active state ──────────────────────────────
    getEl("mode-explore").classList.toggle("active", mode === "explore");
    getEl("mode-inspect").classList.toggle("active", mode === "inspect");
    getEl("mode-explain").classList.toggle("active", mode === "explain");
    getEl("explain-help-btn").style.display = mode === "explain" ? "" : "none";
    getEl("inspect-help-btn").style.display = mode === "inspect" ? "" : "none";
    getEl("help-btn").style.display = mode === "explore" ? "" : "none";

    // ── mode class on #app (used for per-mode accent colour + explain layout) ──
    const appEl = document.getElementById("app");
    appEl?.classList.toggle("explore", mode === "explore");
    appEl?.classList.toggle("inspect", mode === "inspect");
    appEl?.classList.toggle("explain", mode === "explain");

    if (mode === "inspect") {
      if (!document.cookie.includes(INSPECT_COOKIE)) {
        setTimeout(() => _openInspectTourBridge?.(), 400);
      }
    }

    if (mode === "explain") {
      // Save map view for restoration on explain exit (session-only, no cookie)
      if (!wasExplain) {
        savedExploreMapView = { center: leafMap.getCenter(), zoom: leafMap.getZoom() };
        // Show explain tour on first ever visit to explain mode
        if (!document.cookie.includes(EXPLAIN_COOKIE)) {
          setTimeout(() => _openExplainTourBridge?.(), 400);
        }
      }
      leafMap.setView([20, 0], 2, { animate: false });
      // Ensure deck canvas matches the (possibly resized) map and viewport matches
      // the new view before layers are built.
      resizeDeck();
      syncDeckView();

      // Fetch coastline once; rebuild explain layers when it arrives
      if (!coastlineFetched) {
        coastlineFetched = true;
        fetch("/geojson/coastline-50m.geojson")
          .then((r) => r.json())
          .then((g: unknown) => { coastlineGeoJsonClipped = clipCoastlineToMercator(g); _explainCoastlineBridge.set?.(coastlineGeoJsonClipped); rebuildExplainLayers(); })
          .catch(() => console.log("[Explain] coastline GeoJSON missing — continuing without"));
      }

      // Reset drawer on explain re-entry from another mode
      if (!wasExplain) {
        ExST.drawerOpen = false;
        _explainDrawerBridge.set?.(false);
      }

      // Leaving GNN view: freeze violation state so re-entry restores correctly
      ST.selected = null;
      updateInfoPanel();
      rebuildExplainLayers();

      // Fire startup fetches on first Explain entry
      if (!explainInitialized) {
        explainInitialized = true;
        void _loadExplainModelRuntimeBridge.run?.(activeExplainModelId).catch(() => {});
      }
    } else {
      // Returning to GNN view
      if (wasExplain) {
        document.getElementById("app")?.classList.remove("explain-drawer-open", "explain-drawer-header");
        // Close drawer and clear header-only state on mode exit (R3.3)
        ExST.drawerOpen = false;
        ExST.drawerHeaderOnly = false;
        _explainDrawerBridge.set?.(false);
        _explainDrawerHeaderOnlyBridge.set?.(false);
        // Restore map view saved on explain entry
        if (savedExploreMapView) {
          leafMap.setView(savedExploreMapView.center, savedExploreMapView.zoom, { animate: false });
          savedExploreMapView = null;
        }
        // Reset animation flag in case a zoom was in progress when mode switched
        isZoomAnimating = false;
        if (zoomRafId) { cancelAnimationFrame(zoomRafId); zoomRafId = 0; }
        deckCanvas.style.transform = "";
        deckCanvas.style.transformOrigin = "";
        // Force layout refresh so map wrapping and tile positions reset correctly
        leafMap.invalidateSize();
        resizeDeck();
        syncDeckView();
        history.replaceState(null, "", window.location.pathname);
        ST.violation = mode === "inspect";
        getEl("pnl-graph").style.display = ST.violation ? "none" : "block";
        getEl("pnl-viol").style.display = ST.violation ? "block" : "none";
        if (ST.violation) {
          const validKeys = new Set(graphMeta.subgraphs.map((s) => s.key));
          if (!ST.activeInspectSubgraph || !validKeys.has(ST.activeInspectSubgraph)) {
            ST.activeInspectSubgraph = graphMeta.subgraphs.find((s) => s.edge === "g2m")?.key ?? graphMeta.subgraphs[0]?.key ?? "g2m";
          }
          setViolGraphType(ST.activeInspectSubgraph);
        } else {
          if (_savedExploreSubgraphs) {
            ST.activeSubgraphs = new Set(_savedExploreSubgraphs);
            ST.activeNodeTypes = new Set(_savedExploreNodeTypes ?? []);
            ST.showGrid = ST.activeNodeTypes.has("grid");
            ST.showMesh = [...ST.activeNodeTypes].some((k) => k === "mesh" || k.startsWith("mesh_"));
            ST.showG2M = ST.activeSubgraphs.has("g2m");
            ST.showM2M = [...ST.activeSubgraphs].some((k) => k === "m2m" || k.startsWith("m2m_"));
            ST.showM2G = ST.activeSubgraphs.has("m2g");
          }
          renderLayers();
          prevBbox = null;
          scheduleFetch();
        }
      } else {
        // Normal explore/inspect transition (existing behaviour)
        ST.violation = mode === "inspect";
        getEl("pnl-graph").style.display = ST.violation ? "none" : "block";
        getEl("pnl-viol").style.display = ST.violation ? "block" : "none";
        ST.selected = null;
        updateInfoPanel();
        if (ST.violation) {
          const validKeys = new Set(graphMeta.subgraphs.map((s) => s.key));
          if (!ST.activeInspectSubgraph || !validKeys.has(ST.activeInspectSubgraph)) {
            ST.activeInspectSubgraph = graphMeta.subgraphs.find((s) => s.edge === "g2m")?.key ?? graphMeta.subgraphs[0]?.key ?? "g2m";
          }
          setViolGraphType(ST.activeInspectSubgraph);
        } else {
          if (_savedExploreSubgraphs) {
            ST.activeSubgraphs = new Set(_savedExploreSubgraphs);
            ST.activeNodeTypes = new Set(_savedExploreNodeTypes ?? []);
            ST.showGrid = ST.activeNodeTypes.has("grid");
            ST.showMesh = [...ST.activeNodeTypes].some((k) => k === "mesh" || k.startsWith("mesh_"));
            ST.showG2M = ST.activeSubgraphs.has("g2m");
            ST.showM2M = [...ST.activeSubgraphs].some((k) => k === "m2m" || k.startsWith("m2m_"));
            ST.showM2G = ST.activeSubgraphs.has("m2g");
          }
          renderLayers();
          prevBbox = null;
          scheduleFetch();
        }
      }
    }
    _setCurrentAppModeBridge.set?.(mode);
  };
  _setAppMode = setAppMode;
  on(getEl("mode-explore"), "click", () => setAppMode("explore"));
  on(getEl("mode-inspect"), "click", () => setAppMode("inspect"));
  on(getEl("mode-explain"), "click", () => setAppMode("explain"));

  on(getEl("src-deg-min-sl"), "input", () => updateSrcDegSliders());
  on(getEl("src-deg-max-sl"), "input", () => updateSrcDegSliders());
  on(getEl("tgt-deg-min-sl"), "input", () => updateTgtDegSliders());
  on(getEl("tgt-deg-max-sl"), "input", () => updateTgtDegSliders());
  on(getEl("len-min-sl"), "input", () => updateLenSliders());
  on(getEl("len-max-sl"), "input", () => updateLenSliders());
  // Bins input removed from JSX; reading falls back to default 15. Wiring kept
  // conditional so re-adding the input later restores the handler automatically.
  const lenBinsEl = document.getElementById("len-bins") as HTMLInputElement | null;
  if (lenBinsEl) on(lenBinsEl, "input", () => { buildViolationHistograms(); saveLenPlotCookie(); });

  on(getEl("src-deg-logx-btn"), "click", () => toggleSrcDegLogScaleX());
  on(getEl("src-deg-logy-btn"), "click", () => toggleSrcDegLogScaleY());
  on(getEl("tgt-deg-logx-btn"), "click", () => toggleTgtDegLogScaleX());
  on(getEl("tgt-deg-logy-btn"), "click", () => toggleTgtDegLogScaleY());
  on(getEl("len-logx-btn"), "click", () => toggleLenLogScaleX());
  on(getEl("len-logy-btn"), "click", () => toggleLenLogScaleY());

  const bindClick = (id: string, fn: () => void): void => {
    on(getEl(id), "click", () => fn());
  };

  bindClick("toggle-scale-fade", () => {
    ST.zoomFade = !ST.zoomFade;
    updatePill("pe-scale-fade", ST.zoomFade);
    renderLayers();
  });

  document.querySelectorAll<HTMLButtonElement>(".mt-btn").forEach((btn) => {
    on(btn, "click", () => {
      const type = btn.dataset.type;
      if (type) setMapType(type);
    });
  });

  document.querySelectorAll<HTMLButtonElement>(".theme-btn[data-theme]").forEach((btn) => {
    on(btn, "click", () => {
      const mode = btn.dataset.theme as ThemeMode;
      if (mode) setThemeMode(mode);
    });
  });

  return () => {
    listeners.forEach((off) => off());
  };
};

const bindMapHandlers = (): (() => void) => {
  const mapWrap = getEl<HTMLDivElement>("map-wrap");
  const tip = getEl<HTMLDivElement>("tip");

  const onNodeHover = (info: { object: NodeDatum | null; x: number; y: number }): void => {
    if (info.object) {
      const r = mapWrap.getBoundingClientRect();
      showNodeTooltip(r.left + info.x, r.top + info.y, info.object);
      mapWrap.style.cursor = "pointer";
    } else {
      tip.style.display = "none";
      mapWrap.style.cursor = "";
    }
  };

  const onNodeClick = (info: { object: NodeDatum | null }): void => {
    if (info.object) {
      ST.selected = ST.selected && ST.selected.id === info.object.id ? null : info.object;
      updateInfoPanel();
      renderLayers();
    }
  };

  const clickHandler = (e: L.LeafletMouseEvent): void => {
    if (!deckInstance || !deckReady) return;
    if (currentMode === "explain") {
      const info = deckInstance.pickObject({ x: e.containerPoint.x, y: e.containerPoint.y, radius: 1 });
      const nodeIndex = (info?.object as { nodeIndex?: number } | null)?.nodeIndex;
      if (nodeIndex !== undefined) _explainNodeClickBridge.handler?.(nodeIndex);
      return;
    }
    const info = deckInstance.pickObject({ x: e.containerPoint.x, y: e.containerPoint.y, radius: 5 }) as {
      object?: NodeDatum;
    } | null;
    if (info?.object) {
      onNodeClick({ object: info.object });
    } else {
      ST.selected = null;
      updateInfoPanel();
      renderLayers();
    }
  };

  // Hover picking is coalesced to one pick per animation frame.
  //
  // deckInstance.pickObject is not a lookup — it re-renders every pickable layer into the
  // picking framebuffer and then blocks on readPixels to get the result back off the GPU. At
  // ERA5 size that is invisible; with the ocean graph's 675,219 pickable grid icons and
  // millions of edge segments it is tens of milliseconds, and mousemove fires far faster than
  // that. The events then queue up and the cursor visibly lags the pointer.
  //
  // Only the newest pointer position can matter for a hover, so intermediate events are
  // dropped rather than queued: store the latest point, pick once per frame. Picking is also
  // skipped mid-zoom-animation, where the deck view and Leaflet's transform disagree anyway
  // (the same condition syncDeckView already guards on).
  let pendingHover: { x: number; y: number } | null = null;
  let hoverRaf: number | null = null;

  const runHoverPick = (): void => {
    hoverRaf = null;
    const pt = pendingHover;
    pendingHover = null;
    if (!pt || !deckInstance || !deckReady) return;
    const info = deckInstance.pickObject({ x: pt.x, y: pt.y, radius: 5 }) as {
      object?: NodeDatum;
    } | null;
    onNodeHover({ object: info?.object ?? null, x: pt.x, y: pt.y });
  };

  const cancelHoverPick = (): void => {
    if (hoverRaf !== null) cancelAnimationFrame(hoverRaf);
    hoverRaf = null;
    pendingHover = null;
  };

  const moveHandler = (e: L.LeafletMouseEvent): void => {
    if (!deckInstance || !deckReady) return;
    if (currentMode === "explain") {
      // Explain mode is deliberately left on the immediate path — out of scope here.
      const info = deckInstance.pickObject({ x: e.containerPoint.x, y: e.containerPoint.y, radius: 1 });
      const nodeIndex = (info?.object as { nodeIndex?: number } | null)?.nodeIndex ?? null;
      if (nodeIndex !== currentExplainHoveredNode) {
        currentExplainHoveredNode = nodeIndex;
        rebuildExplainLayers();
      }
      return;
    }
    if (isZoomAnimating) { cancelHoverPick(); return; }
    pendingHover = { x: e.containerPoint.x, y: e.containerPoint.y };
    if (hoverRaf === null) hoverRaf = requestAnimationFrame(runHoverPick);
  };

  const outHandler = (): void => {
    cancelHoverPick();
    tip.style.display = "none";
    mapWrap.style.cursor = "";
    if (currentMode === "explain" && currentExplainHoveredNode !== null) {
      currentExplainHoveredNode = null;
      rebuildExplainLayers();
    }
  };

  const scheduleHandler = (): void => scheduleFetch();

  const updateMinimap = (): void => {
    const b = leafMap.getBounds();
    const s = Math.max(b.getSouth(), -85);
    const n = Math.min(b.getNorth(), 85);
    let w = b.getWest();
    let e = b.getEast();
    // Normalise west to [-180, 180)
    while (w > 180) { w -= 360; e -= 360; }
    while (w < -180) { w += 360; e += 360; }

    if (e - w >= 360) {
      // Full world visible
      viewRect.setBounds([[s, -180], [n, 180]]);
      viewRectWrap.setBounds([[s, -180], [n, 180]]);
    } else if (e > 180) {
      // Wraps past right edge → split into right fragment + left re-entry
      viewRect.setBounds([[s, w], [n, 180]]);
      viewRectWrap.setBounds([[s, -180], [n, e - 360]]);
    } else if (w < -180) {
      // Wraps past left edge → split into left fragment + right re-entry
      viewRect.setBounds([[s, -180], [n, e]]);
      viewRectWrap.setBounds([[s, w + 360], [n, 180]]);
    } else {
      viewRect.setBounds([[s, w], [n, e]]);
      viewRectWrap.setBounds([[s, w], [n, e]]);
    }
  };

  // Leaflet's zoomanim event only carries { center, zoom } — there is no scale.
  // We keep the deck overlay glued to the tiles by scaling the deck canvas about
  // the zoom's fixed screen anchor, sampling the true interpolated scale from
  // Leaflet's proxy element every frame (readLeafletProxyScale). The anchor is
  // computed once at animation start, when getZoom()/the projection still
  // reflect the source view (Leaflet applies the target via _move *after* the
  // event fires).
  const zoomanimHandler = (event: L.LeafletEvent): void => {
    if (isZoomAnimating) return; // one setup per animation (continuous zoom refires)
    isZoomAnimating = true;

    const e = event as L.LeafletEvent & { zoom: number; center: L.LatLng };
    const fromZoom = leafMap.getZoom();
    const s = Math.pow(2, e.zoom - fromZoom); // final zoom scale factor
    const start = readLeafletProxyScale();
    zoomStartProxyScale = start && start > 0 ? start : 1;

    // Fixed screen anchor p: the point that stays put through the zoom, derived
    // so the target centre lands at the viewport centre under scale-about-p.
    if (Math.abs(s - 1) > 1e-6) {
      const sz = leafMap.getSize();
      const C = leafMap.latLngToContainerPoint(e.center);
      const px = (s * C.x - sz.x / 2) / (s - 1);
      const py = (s * C.y - sz.y / 2) / (s - 1);
      deckCanvas.style.transformOrigin = `${px}px ${py}px`;
    }

    const tick = (): void => {
      const now = readLeafletProxyScale();
      if (now != null && zoomStartProxyScale > 0) {
        // now/start is the deck bitmap's scale relative to its rendered zoom.
        deckCanvas.style.transform = `scale(${now / zoomStartProxyScale})`;
      }
      zoomRafId = requestAnimationFrame(tick);
    };
    zoomRafId = requestAnimationFrame(tick);
  };

  const zoomendHandler = (): void => {
    isZoomAnimating = false;
    if (zoomRafId) { cancelAnimationFrame(zoomRafId); zoomRafId = 0; }
    deckCanvas.style.transform = "";
    deckCanvas.style.transformOrigin = "";
    syncDeckView();
    renderLayers();
  };

  leafMap.on("click", clickHandler);
  leafMap.on("mousemove", moveHandler);
  leafMap.on("mouseout", outHandler);
  leafMap.on("move", syncDeckView);
  leafMap.on("moveend", scheduleHandler);
  leafMap.on("zoomend", scheduleHandler);
  leafMap.on("zoomanim", zoomanimHandler);
  leafMap.on("zoomend", zoomendHandler);
  leafMap.on("move", updateMinimap);
  leafMap.on("zoom", updateMinimap);
  leafMap.on("moveend", updateMinimap);
  leafMap.on("zoomend", updateMinimap);
  updateMinimap();

  const resizeHandler = (): void => {
    if (!deckInstance) return;
    resizeDeck();
    syncDeckView();
    renderLayers();
  };

  window.addEventListener("resize", resizeHandler);

  systemDarkMq = window.matchMedia("(prefers-color-scheme: dark)");
  const onSystemTheme = (): void => {
    if (currentThemeMode === "auto") applyTheme(systemDarkMq !== null ? systemDarkMq.matches : false);
  };
  systemDarkMq.addEventListener("change", onSystemTheme);
  setThemeMode(currentThemeMode);

  return () => {
    leafMap.off("click", clickHandler);
    cancelHoverPick();
    leafMap.off("mousemove", moveHandler);
    leafMap.off("mouseout", outHandler);
    leafMap.off("move", syncDeckView);
    leafMap.off("moveend", scheduleHandler);
    leafMap.off("zoomend", scheduleHandler);
    leafMap.off("zoomanim", zoomanimHandler);
    leafMap.off("zoomend", zoomendHandler);
    leafMap.off("move", updateMinimap);
    leafMap.off("zoom", updateMinimap);
    leafMap.off("moveend", updateMinimap);
    leafMap.off("zoomend", updateMinimap);
    if (zoomRafId) { cancelAnimationFrame(zoomRafId); zoomRafId = 0; }
    window.removeEventListener("resize", resizeHandler);
    systemDarkMq?.removeEventListener("change", onSystemTheme);
  };
};

const bindPanelInteractions = (): (() => void) => {
  const cleanup: Array<() => void> = [];

  const initPanel = (panelId: string, _dragHandleId: string, resizeHandleId: string, minW: number, resizeFromLeft = false, onChange?: () => void): void => {
    const panel = getEl<HTMLDivElement>(panelId);
    const resizeHandle = getEl<HTMLDivElement>(resizeHandleId);

    let rStartX = 0, rStartY = 0, rStartW = 0, rStartH = 0, rStartL = 0;
    const onResizeMove = (e: PointerEvent): void => {
      if (resizeFromLeft) {
        const newW = Math.max(minW, rStartW - (e.clientX - rStartX));
        panel.style.left = `${rStartL + rStartW - newW}px`;
        panel.style.width = `${newW}px`;
      } else {
        panel.style.width = `${Math.max(minW, rStartW + e.clientX - rStartX)}px`;
      }
      panel.style.height = `${Math.max(120, rStartH + e.clientY - rStartY)}px`;
      onChange?.();
    };
    const onResizeUp = (): void => {
      document.removeEventListener("pointermove", onResizeMove);
      document.removeEventListener("pointerup", onResizeUp);
      document.body.style.userSelect = "";
    };
    const onResizeDown = (e: PointerEvent): void => {
      e.preventDefault(); e.stopPropagation();
      rStartX = e.clientX; rStartY = e.clientY;
      const r = panel.getBoundingClientRect();
      rStartW = r.width; rStartH = r.height; rStartL = r.left;
      document.body.style.userSelect = "none";
      document.addEventListener("pointermove", onResizeMove);
      document.addEventListener("pointerup", onResizeUp);
    };
    resizeHandle.addEventListener("pointerdown", onResizeDown);
    cleanup.push(() => resizeHandle.removeEventListener("pointerdown", onResizeDown));
  };


  const minimapEl = document.getElementById("minimap-container") as HTMLDivElement | null;
  if (minimapEl) {
    let mStartX = 0, mStartY = 0, mStartL = 0, mStartT = 0;
    const onMinimapMove = (e: PointerEvent): void => {
      minimapEl.style.left = `${Math.max(0, mStartL + e.clientX - mStartX)}px`;
      minimapEl.style.top = `${Math.max(0, mStartT + e.clientY - mStartY)}px`;
    };
    const onMinimapUp = (): void => {
      document.removeEventListener("pointermove", onMinimapMove);
      document.removeEventListener("pointerup", onMinimapUp);
      document.body.style.userSelect = "";
    };
    const onMinimapDown = (e: PointerEvent): void => {
      e.preventDefault();

      const r = minimapEl.getBoundingClientRect();
      mStartX = e.clientX; mStartY = e.clientY;
      mStartL = r.left; mStartT = r.top;
      minimapEl.style.right = "auto";
      minimapEl.style.left = `${r.left}px`;
      minimapEl.style.top = `${r.top}px`;
      document.body.style.userSelect = "none";
      document.addEventListener("pointermove", onMinimapMove);
      document.addEventListener("pointerup", onMinimapUp);
    };
    minimapEl.addEventListener("pointerdown", onMinimapDown);
    cleanup.push(() => minimapEl.removeEventListener("pointerdown", onMinimapDown));
  }

  initPanel("panel-left", "panel-left-header", "panel-left-resize", 340);
  initPanel("panel-right", "panel-right-header", "panel-right-resize", 240, true);
  initPanel("explain-panel-right", "explain-panel-right-header", "explain-panel-right-resize", 260, true);

  return () => cleanup.forEach((fn) => fn());
};

const resetState = (): void => {
  activeGraphId = GRAPH_ID;
  currentThemeMode = "auto";

  systemDarkMq = null;
  gridNodes = [];
  meshNodes = [];
  gridById = {};
  meshById = {};
  edgesStore = {};
  edgesCache = {};
  cachedGridById = {};
  cachedMeshByKey = {};
  fetchedBboxes = [];
  bumpCacheEpoch();
  prevBbox = null;

  ST = {
    showGrid: true,
    showMesh: true,
    showG2M: true,
    showM2M: true,
    showM2G: true,
    selected: null,
    violation: false,
    activeInspectSubgraph: "g2m",
    srcDegMin: 0,
    srcDegMax: 30,
    tgtDegMin: 0,
    tgtDegMax: 30,
    lenSliderMax: 600,
    srcDegLogScaleX: false,
    srcDegLogScaleY: false,
    tgtDegLogScaleX: false,
    tgtDegLogScaleY: false,
    lenLogScaleX: false,
    lenLogScaleY: false,
    zoomFade: true,
    activeSubgraphs: new Set(),
    activeNodeTypes: new Set(),
  };

  fullGraphSrcDeg = new Uint16Array(0);
  fullGraphTgtDeg = new Uint16Array(0);
  fullGraphLengths = new Float32Array(0);
  fullGraphBounds = { srcDegMin: 0, srcDegMax: 0, tgtDegMin: 0, tgtDegMax: 0, lengthMin: 0, lengthMax: 0 };
  derivedMetrics = null;
  graphMeta = { n_grid: 0, grid_spacing_deg: 0, n_mesh: 0, n_edges_g2m: 0, n_edges_m2m: 0, n_edges_m2g: 0, is_hierarchical: false, n_levels: 1, node_types: [], node_counts: {}, subgraphs: [] };
  violTotals = { srcLo: 0, srcHi: 0, tgtLo: 0, tgtHi: 0, edgeLo: 0, edgeHi: 0 };
  violMetricsReady = false;
  currentMapType = "map";

  fetchGen = 0;
  deckReady = false;

  if (fetchTimer) {
    clearTimeout(fetchTimer);
    fetchTimer = null;
  }
};

const levelFromType = (s: string): number => { const m = s.match(/_(\d+)$/); return m ? parseInt(m[1], 10) : 0; };

const NodeGlyph = ({ type, levels = 1 }: { type: string; levels?: number }) => {
  if (type === "grid") return <GlyphSquare color="var(--grid-color)" />;
  const lvl = levelFromType(type);
  const size = levels > 1 ? Math.round(7 + 6 * lvl / Math.max(1, levels - 1)) : 10;
  return <GlyphCircle color="var(--mesh-color)" size={size} />;
};

const EdgeGlyph = ({ edge, levels = 1 }: { edge: string; levels?: number }) => {
  const lvl = levelFromType(edge);
  const weight = levels > 1 ? strokeForLevel(lvl, levels) : 1.5;
  if (edge === "m2m" || edge.startsWith("m2m_")) return <GlyphLine color="var(--mesh-color)" weight={weight} />;
  if (edge.startsWith("mesh_up") || edge.startsWith("mesh_down")) return <GlyphArrow color="var(--mesh-color)" weight={weight} />;
  if (edge === "g2m") return <GlyphArrow color="var(--g2m-color)" />;
  return <GlyphArrow color="var(--m2g-color)" />;
};

const SubgraphChip = React.memo(({ sg, active, onSelect, levels = 1 }: { sg: SubgraphDef; active: boolean; onSelect: (k: SubgraphKey) => void; levels?: number }) => (
  <button className={`gt-chip${active ? " active" : ""}`} type="button" onClick={() => onSelect(sg.key)}>
    <span className="gt-chip-icons">
      <NodeGlyph type={sg.src} levels={levels} />
      <EdgeGlyph edge={sg.edge} levels={levels} />
      <NodeGlyph type={sg.tgt} levels={levels} />
    </span>
    <span className="gt-chip-label">{sg.label}</span>
  </button>
));
SubgraphChip.displayName = "SubgraphChip";

const EDGE_SECTION_LABEL: Record<string, string> = { g2m: "ENCODER", m2m: "PROCESSOR", m2g: "DECODER" };

const SubgraphSelector = React.memo(({ subgraphs, isHierarchical, active, onSelect, levels = 1 }: {
  subgraphs: SubgraphDef[];
  isHierarchical: boolean;
  active: SubgraphKey;
  onSelect: (k: SubgraphKey) => void;
  levels?: number;
}) => {
  if (!subgraphs.length) return null;

  if (!isHierarchical) {
    return (
      <div className="gt-tabs gt-tabs-flat">
        {subgraphs.map((sg) => (
          <div key={sg.key} className="gt-tabs-flat-item">
            <div className="gt-tabs-section-label">{EDGE_SECTION_LABEL[sg.edge] ?? ""}</div>
            <SubgraphChip sg={sg} active={sg.key === active} onSelect={onSelect} levels={levels} />
          </div>
        ))}
      </div>
    );
  }

  const encoder = subgraphs.filter((sg) => sg.edge === "g2m");
  const decoder = subgraphs.filter((sg) => sg.edge === "m2g");
  const processor = subgraphs.filter((sg) => sg.edge !== "g2m" && sg.edge !== "m2g");

  return (
    <div className="gt-tabs gt-tabs-hierarchical">
      <div className="gt-tabs-section gt-tabs-processor">
        <div className="gt-tabs-section-label">PROCESSOR</div>
        <div className="gt-tabs-chips">
          {processor.map((sg) => <SubgraphChip key={sg.key} sg={sg} active={sg.key === active} onSelect={onSelect} levels={levels} />)}
        </div>
      </div>
      <div className="gt-tabs-enc-dec-row">
        <div className="gt-tabs-section">
          <div className="gt-tabs-section-label">ENCODER</div>
          <div className="gt-tabs-chips">
            {encoder.map((sg) => <SubgraphChip key={sg.key} sg={sg} active={sg.key === active} onSelect={onSelect} levels={levels} />)}
          </div>
        </div>
        <div className="gt-tabs-section">
          <div className="gt-tabs-section-label">DECODER</div>
          <div className="gt-tabs-chips">
            {decoder.map((sg) => <SubgraphChip key={sg.key} sg={sg} active={sg.key === active} onSelect={onSelect} levels={levels} />)}
          </div>
        </div>
      </div>
    </div>
  );
});
SubgraphSelector.displayName = "SubgraphSelector";

export default function GNNExplorer({ apiBaseUrl, cartoApiKey: cartoApiKeyProp }: GNNExplorerProps) {
  const uploadInputRef = useRef<HTMLInputElement | null>(null);
  const modelUploadInputRef = useRef<HTMLInputElement | null>(null);

  // ── LeftPanel / StackedSidebar React state ─────────────────────────────────
  const [metaLevels, setMetaLevels] = useState(1);
  const [metaNodeCounts, setMetaNodeCounts] = useState<Record<string, number>>({});
  const [activeInspectSubgraph, setActiveInspectSubgraph] = useState<SubgraphKey>("g2m");
  const [currentAppMode, setCurrentAppMode] = useState<string>("explore");

  const handlePanelStateChange = useCallback(
    (uiSubs: Set<string>, uiNodes: Set<string>) => {
      const apiSubs = new Set<SubgraphKey>();
      for (const key of uiSubs) {
        if (key === "g2m") apiSubs.add("g2m");
        else if (key === "m2g") apiSubs.add("m2g");
        else if (key.startsWith("m2m-")) {
          const i = key.slice(4);
          apiSubs.add(graphMeta.is_hierarchical ? `m2m_${i}` : "m2m");
        } else if (key.startsWith("up-")) {
          apiSubs.add(`mesh_up_${key.slice(3)}`);
        } else if (key.startsWith("dn-")) {
          apiSubs.add(`mesh_down_${key.slice(3)}`);
        }
      }

      // In inspect mode the sidebar is radio — route through setViolGraphType.
      // Always return here so sidebar mode-transition resets (e.g. defaults firing
      // with empty apiSubs) never overwrite ST.activeNodeTypes with stale values.
      if (currentMode === "inspect") {
        if (apiSubs.size > 0) {
          const [firstSub] = apiSubs;
          setViolGraphType(firstSub);
        }
        return;
      }

      const apiNodes = new Set<NodeType>();
      for (const key of uiNodes) {
        if (key === "grid") apiNodes.add("grid");
        else if (key.startsWith("mesh-")) {
          const i = key.slice(5);
          apiNodes.add(graphMeta.is_hierarchical ? `mesh_${i}` : "mesh");
        }
      }
      ST.activeSubgraphs = apiSubs;
      ST.activeNodeTypes = apiNodes;
      ST.showGrid = apiNodes.has("grid");
      ST.showMesh = [...apiNodes].some((k) => k === "mesh" || k.startsWith("mesh_"));
      ST.showG2M = apiSubs.has("g2m");
      ST.showM2M = [...apiSubs].some((k) => k === "m2m" || k.startsWith("m2m_"));
      ST.showM2G = apiSubs.has("m2g");
      _savedExploreSubgraphs = new Set(apiSubs);
      _savedExploreNodeTypes = new Set(apiNodes);
      renderLayers();
      if (ST.violation) buildViolationHistograms();
    },
    [],
  );

  const [isUploading, setIsUploading] = useState(false);
  const [isSwitchingGraph, setIsSwitchingGraph] = useState(false);
  const [graphOptions, setGraphOptions] = useState<GraphRecord[]>([]);
  const [selectedGraphId, setSelectedGraphId] = useState(GRAPH_ID);
  const [explainModelOptions, setExplainModelOptions] = useState<ExplainModelRecord[]>([]);
  const [selectedExplainModelId, setSelectedExplainModelId] = useState(DEFAULT_EXPLAIN_MODEL_ID);
  const [showIntro, setShowIntro] = useState(false);
  const [introStep, setIntroStep] = useState(0);
  const [spotlightRect, setSpotlightRect] = useState<{ top: number; left: number; width: number; height: number } | null>(null);
  const [showExplainTour, setShowExplainTour] = useState(false);
  const [explainTourStep, setExplainTourStep] = useState(0);
  const [explainSpotlightRect, setExplainSpotlightRect] = useState<{ top: number; left: number; width: number; height: number } | null>(null);
  const [showExploreTour, setShowExploreTour] = useState(false);
  const [exploreTourStep, setExploreTourStep] = useState(0);
  const [exploreSpotlightRect, setExploreSpotlightRect] = useState<{ top: number; left: number; width: number; height: number } | null>(null);
  const [showInspectTour, setShowInspectTour] = useState(false);
  const [inspectTourStep, setInspectTourStep] = useState(0);
  const [inspectSpotlightRect, setInspectSpotlightRect] = useState<{ top: number; left: number; width: number; height: number } | null>(null);

  // ── Explain mode React state (drives JSX re-renders in F1–F5 sub-components) ──
  const [rightPanelCollapsed, setRightPanelCollapsed] = useState(false);
  const [leftPanelCollapsed, setLeftPanelCollapsed] = useState(false);
  const [explainRightPanelCollapsed, setExplainRightPanelCollapsed] = useState(false);

  // ExST remains the authoritative runtime object; these mirror it for rendering.
  const [explainV1, setExplainV1] = useState<string>(ExST.v1);
  const [explainZ1, setExplainZ1] = useState<string | null>(ExST.z1);
  const [explainNode, setExplainNode] = useState<number | null>(ExST.node);
  const [explainDrawerOpen, setExplainDrawerOpen] = useState<boolean>(ExST.drawerOpen);
  const [explainDrawerHeaderOnly, setExplainDrawerHeaderOnly] = useState<boolean>(ExST.drawerHeaderOnly);
  const [explainCompareMode, setExplainCompareMode] = useState<"timestamps" | "variables">(ExST.compareMode);
  const [explainMethod, setExplainMethod] = useState<"ixg" | "ig">(ExST.method);
  const [explainT, setExplainT] = useState<string>(ExST.t);
  // Setter only: model_info is still fetched and cached on ExST for the URL/debug paths,
  // but the left panel now renders from the /models catalogue instead.
  const [, setExplainModelInfo] = useState<ModelInfo | null>(null);
  const [explainModelSwitching, setExplainModelSwitching] = useState<string | null>(null);
  const [explainSwitchPending, setExplainSwitchPending] = useState<boolean>(false);
  const [explainRefetchTrigger, setExplainRefetchTrigger] = useState(0);
  const [explainGrid, setExplainGrid] = useState<GridData | null>(ExST.grid);
  const [explainVariablesMeta, setExplainVariablesMeta] = useState<VariablesMeta | null>(ExST.variablesMeta);

  // ── Model upload modal ──────────────────────────────────────────────────────
  const [modelUploadOpen, setModelUploadOpen] = useState(false);
  const [modelUploadDomain, setModelUploadDomain] = useState<"era5" | "ocean">("era5");
  const [modelUploadName, setModelUploadName] = useState("");
  const [modelUploadFile, setModelUploadFile] = useState<File | null>(null);
  const [modelUploadSubmitting, setModelUploadSubmitting] = useState(false);
  const [modelUploadError, setModelUploadError] = useState<string | null>(null);
  const [explainTimelineData, setExplainTimelineData] = useState<TimelineData | null>(ExST.timelineData);
  const [explainComponentPeaks, setExplainComponentPeaks] = useState<ComponentPeaks | null>(null);

  // Which variables/levels carry the largest component errors at the displayed timestep.
  // Empty when the model's artifact has no peak data, which the panel renders as no marks.
  const explainPeakMarks = useMemo(
    () => decodePeakMarks(explainComponentPeaks, explainT),
    [explainComponentPeaks, explainT],
  );
  const [explainAvailableTimestamps, setExplainAvailableTimestamps] = useState<Set<string> | null>(ExST.availableTimestamps);
  // Median gap between consecutive window timestamps, in hours. Median rather than first-two,
  // so a single irregular step (a gap in the shipped data) cannot skew it. Falls back to the
  // ERA5 cadence when the window is not loaded yet.
  const explainStepHours = React.useMemo(() => {
    const ts = [...(explainAvailableTimestamps ?? [])].sort();
    if (ts.length < 2) return 6;
    const gaps = ts.slice(1).map((t, i) => (Date.parse(t) - Date.parse(ts[i])) / 3_600_000);
    gaps.sort((a, b) => a - b);
    return gaps[Math.floor(gaps.length / 2)] || 6;
  }, [explainAvailableTimestamps]);

  // ── GCS window state ────────────────────────────────────────────────────────
  const [windowLoading, setWindowLoading] = useState(false);
  const [loadingBand, setLoadingBand] = useState<string | null>(null);
  const [era5InitError, setEra5InitError] = useState(false);
  const bandClickAbortRef = useRef<AbortController | null>(null);
  const [downloadedBandStarts, setDownloadedBandStarts] = useState<Set<string>>(new Set());
  const [allDownloadedTimestamps, setAllDownloadedTimestamps] = useState<Set<string>>(new Set());

  // ── Colorbar state — fed by ExplainMap via onColorbarUpdate, consumed by ExplainLeftPanel ──
  // null entries = no data for that node in the selected channel (below the seafloor at that
  // depth), as distinct from a zero error. Consumers must skip them, not coerce to 0.
  const [explainColorbarErrors, setExplainColorbarErrors] = useState<(number | null)[] | null>(null);
  const [explainColorbarMax, setExplainColorbarMax] = useState<number>(0);
  const [explainColorbarMin, setExplainColorbarMin] = useState<number>(0); // pred mode: lower bound; defaults to 0 (error mode)
  const [explainColorbarBackendMax, setExplainColorbarBackendMax] = useState<number>(0); // Store the shared max from backend
  const [explainColorbarPhase, setExplainColorbarPhase] = useState<"skeleton" | "ready">("skeleton");
  const [explainLogScale, setExplainLogScale] = useState(false);
  const [explainSharedColorbar, setExplainSharedColorbar] = useState(false);

  // ── Saved xai configurations — in-memory only, cleared on reload alongside
  // the IG cache. FIFO eviction beyond MAX_SAVED_EXPLAIN_CONFIGS. ─────────────
  const [explainSavedConfigs, setExplainSavedConfigs] = useState<SavedExplainConfig[]>([]);

  // ── Display mode + normalization stats (fetched once for prediction mode) ──
  const [explainDisplayMode, setExplainDisplayMode] = useState<"error" | "prediction">("error");
  type NormalizationStats = {
    mean: number[];
    std: number[];
    physical_scales: Record<string, [number, number]>;
  };
  const [explainNormStats, setExplainNormStats] = useState<NormalizationStats | null>(null);

  // Fetch normalization once on mount. Tiny payload (~few KB) — needed for the
  // Prediction display mode so the frontend can denormalize and pick a fixed
  // physical [min, max] colorbar per (variable, level).
  // Refetched per model, not once on mount: the stats are per-domain (ERA5 has 83
  // components, ocean 27), and the un-prefixed route resolves to the active ERA5 preset.
  useEffect(() => {
    let cancelled = false;
    if (!selectedExplainModelId) return;
    fetch(`${API}/inference/${encodeURIComponent(selectedExplainModelId)}/normalization`)
      .then((r) => (r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`)))
      .then((data: NormalizationStats) => { if (!cancelled) setExplainNormStats(data); })
      .catch((err) => console.warn("[Explain] normalization fetch failed:", err));
    return () => { cancelled = true; };
  }, [selectedExplainModelId]);

  // ── F5: attribution drawer state ────────────────────────────────────────────
  const [explainAttributionData, setExplainAttributionData] = useState<AttributionResponse | null>(null);
  const [explainAttrLoadingA, setExplainAttrLoadingA] = useState(false);
  const [explainAttrLoadingB, setExplainAttrLoadingB] = useState(false);
  const [explainAttrErrorA, setExplainAttrErrorA] = useState<string | null>(null);
  const [explainAttrErrorB, setExplainAttrErrorB] = useState<string | null>(null);
  const [explainCoastlineGeoJson, setExplainCoastlineGeoJson] = useState<unknown>(null);
  const [explainPAV2, setExplainPAV2] = useState<string | null>(null);
  const [explainPAZ2, setExplainPAZ2] = useState<string | null>(null);
  // ── F7: Variables mode Panel B state ───────────────────────────────────────
  const [explainAttributionDataB, setExplainAttributionDataB] = useState<AttributionResponse | null>(null);
  const [explainPBV2, setExplainPBV2] = useState<string | null>(null);
  const [explainPBZ2, setExplainPBZ2] = useState<string | null>(null);
  const [explainZ2Locked, setExplainZ2Locked] = useState<boolean>(false);

  // ── F5/F7: attribution fetch ────────────────────────────────────────────────
  // Helpers that build the shared target params on the URL.
  const _buildAttrBaseUrl = (): URL => {
    // The model id must be in the path. The un-prefixed /integrated_gradients/attributions
    // route falls back to the active ERA5 preset, so an ocean target like "thetao_0m" is
    // resolved against the ERA5 variable list and 400s — the same failure the Explain map
    // fetches had before they were given the id.
    const url = new URL(
      `${API}/integrated_gradients/${encodeURIComponent(activeExplainModelId)}/attributions`,
    );
    url.searchParams.set("pred_time", ExST.t);
    url.searchParams.set("target_node", String(ExST.node!));
    url.searchParams.set("target_variable", ExST.v1);
    if (!isSurfaceVar(ExST.v1) && ExST.z1 !== null) {
      url.searchParams.set("target_level", ExST.z1);
    }
    url.searchParams.set("method", ExST.method);
    return url;
  };

  // Builds the source query-parameter value for a given panel (v2 + z2 pair).
  const _buildPanelSource = (v2: string | null, z2: string | null): string => {
    const v = v2 ?? ExST.v1;
    const z = z2 ?? ExST.z1;
    return isSurfaceVar(v) ? v : `${v}@${z}`;
  };

  // Fires Panel A (and also Panel B loading in Timestamps mode — one shared response).
  // Callers must have already written ExST.pA_v2/pA_z2 before calling.
  // Returns a Promise that resolves once the fetch chain settles, so callers can
  // chain a follow-up request (e.g. fireAttrB) — useful for serializing same-target
  // requests so they hit the backend cache instead of duplicating compute.
  // The backend decides whether attribution is affordable for the active model and reports it
  // as `attribution_available` on /variables — an ocean pass is a 9-30 GB backward pass, which
  // a deployment without the memory (or without CUDA, where an over-allocation is catchable)
  // cannot absorb. Asking the backend rather than hardcoding it means a workstation with room
  // to spare gets the feature, and a constrained deployment turns it off with an env var,
  // without a frontend change either way.
  const OCEAN_ATTR_DISABLED_MSG =
    "Attribution is unavailable for this model on the current deployment: one pass needs far more memory than is available here. It is enabled automatically where the backend has the capacity.";
  const attrBlocked = (): boolean => ExST.variablesMeta?.attribution_available === false;

  const fireAttrA = (): Promise<void> => {
    if (ExST.node === null || !ExST.t) return Promise.resolve();
    if (ExST.v1 === "total" || ExST.z1 === "sum") return Promise.resolve();
    if (attrBlocked()) {
      setExplainAttrLoadingA(false);
      setExplainAttrErrorA(OCEAN_ATTR_DISABLED_MSG);
      if (ExST.compareMode === "timestamps") {
        setExplainAttrLoadingB(false);
        setExplainAttrErrorB(OCEAN_ATTR_DISABLED_MSG);
      }
      return Promise.resolve();
    }

    attributionAbortA?.abort();
    attributionAbortA = null;
    setExplainAttrLoadingA(true);
    setExplainAttrErrorA(null);
    if (ExST.compareMode === "timestamps") {
      // Both panels share one fetch
      attributionAbortB?.abort();
      attributionAbortB = null;
      setExplainAttrLoadingB(true);
      setExplainAttrErrorB(null);
    }

    const v2 = ExST.pA_v2 ?? ExST.v1;
    const z2 = isSurfaceVar(v2) ? null : (ExST.pA_z2 ?? ExST.z1 ?? "50");
    // Canonical source string used for dedup comparison (never sent to API directly).
    const sourceParam = isSurfaceVar(v2) ? v2 : `${v2}@${z2}`;

    // Snapshot critical state so .then() can detect stale responses.
    const snapNode     = ExST.node;
    const snapT        = ExST.t;
    const snapMethod   = ExST.method;
    const snapMode     = ExST.compareMode;
    // Capture Panel B source for dedup: if B still uses the same source as A when
    // A resolves, reuse A's response for B instead of waiting on a duplicate request.
    const snapPBSource = _buildPanelSource(ExST.pB_v2, ExST.pB_z2);

    const url = _buildAttrBaseUrl();
    // When z2 = "sum", appends 13 repeated source= params (one per pressure level).
    buildPanelSourceParams(url, v2, z2);

    const ctrl = new AbortController();
    attributionAbortA = ctrl;
    if (ExST.compareMode === "timestamps") attributionAbortB = ctrl;

    console.log(`[attr A] FIRED ${snapMethod} target=${ExST.v1}${ExST.z1 && ExST.z1 !== "sum" ? "@" + ExST.z1 : ""} src=${sourceParam}`);
    const tStart = performance.now();
    // Serve from the front-end LRU when this exact request (URL) was fetched
    // before; otherwise fetch and populate the cache. The rest of the chain
    // (staleness guards, Panel B dedup, loading state) is identical either way.
    const urlStrA = url.toString();
    const cachedA = attrCache.get(urlStrA);
    const acquireA: Promise<AttributionResponse> = cachedA
      ? Promise.resolve(cachedA)
      : fetch(urlStrA, { signal: ctrl.signal }).then(async (r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          const buf = await r.arrayBuffer();
          const data = JSON.parse(new TextDecoder().decode(buf)) as AttributionResponse;
          attrCache.set(urlStrA, data);
          console.log(
            `[attr A] ${snapMethod} ${(performance.now() - tStart).toFixed(0)}ms ` +
            `cache=${data.cache_status ?? "?"} ` +
            `bytes=${buf.byteLength} ` +
            `target=${ExST.v1}${ExST.z1 && ExST.z1 !== "sum" ? "@" + ExST.z1 : ""} ` +
            `src=${sourceParam} ` +
            `keys=${Object.keys(data.attributions_prev).join(",")}`,
          );
          return data;
        });
    if (cachedA) console.log(`[attr A] front-cache hit src=${sourceParam}`);
    return acquireA
      .then((data) => {
        // Discard if node, timestep, or method changed while the request was in-flight.
        // Clear loading if no newer fetch took over (otherwise the spinner sticks forever).
        if (ExST.node !== snapNode || ExST.t !== snapT || ExST.method !== snapMethod) {
          if (attributionAbortA === ctrl) { setExplainAttrLoadingA(false); setExplainAttrLoadingB(false); }
          return;
        }
        ExST.attributionData = data;
        setExplainAttributionData(data);
        setExplainAttrLoadingA(false);
        if (snapMode === "timestamps") {
          setExplainAttrLoadingB(false);
        } else if (snapMode === "variables") {
          // Dedup: if Panel B source hasn't changed and still equals Panel A source,
          // reuse this response — no second IG computation needed.
          if (_buildPanelSource(ExST.pB_v2, ExST.pB_z2) === snapPBSource && snapPBSource === sourceParam) {
            ExST.attributionDataB = data;
            setExplainAttributionDataB(data);
            setExplainAttrLoadingB(false);
          }
        }
      })
      .catch((err: unknown) => {
        if (err instanceof Error && err.name === "AbortError") return;
        if (ExST.node !== snapNode || ExST.t !== snapT || ExST.method !== snapMethod) {
          if (attributionAbortA === ctrl) { setExplainAttrLoadingA(false); setExplainAttrLoadingB(false); }
          return;
        }
        const msg = err instanceof Error ? err.message : "Attribution failed";
        setExplainAttrLoadingA(false);
        setExplainAttrErrorA(msg);
        if (snapMode === "timestamps") { setExplainAttrLoadingB(false); setExplainAttrErrorB(msg); }
      });
  };

  // Fires Panel B independently (Variables mode only).
  // Callers must have already written ExST.pB_v2/pB_z2 before calling.
  const fireAttrB = (): void => {
    if (ExST.node === null || !ExST.t) return;
    if (ExST.v1 === "total" || ExST.z1 === "sum") return;
    if (attrBlocked()) {
      setExplainAttrLoadingB(false);
      setExplainAttrErrorB(OCEAN_ATTR_DISABLED_MSG);
      return;
    }

    attributionAbortB?.abort();
    attributionAbortB = null;
    setExplainAttrLoadingB(true);
    setExplainAttrErrorB(null);

    const v2 = ExST.pB_v2 ?? ExST.v1;
    const z2 = isSurfaceVar(v2) ? null : (ExST.pB_z2 ?? ExST.z1 ?? "50");
    const sourceParam = isSurfaceVar(v2) ? v2 : `${v2}@${z2}`;

    // Snapshot critical state so .then() can detect stale responses.
    const snapNode   = ExST.node;
    const snapT      = ExST.t;
    const snapMethod = ExST.method;
    const snapMode   = ExST.compareMode;

    const url = _buildAttrBaseUrl();
    // When z2 = "sum", appends 13 repeated source= params (one per pressure level).
    buildPanelSourceParams(url, v2, z2);

    const ctrl = new AbortController();
    attributionAbortB = ctrl;

    console.log(`[attr B] FIRED ${snapMethod} target=${ExST.v1}${ExST.z1 && ExST.z1 !== "sum" ? "@" + ExST.z1 : ""} src=${sourceParam}`);
    const tStart = performance.now();
    const urlStrB = url.toString();
    const cachedB = attrCache.get(urlStrB);
    const acquireB: Promise<AttributionResponse> = cachedB
      ? Promise.resolve(cachedB)
      : fetch(urlStrB, { signal: ctrl.signal }).then(async (r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          const buf = await r.arrayBuffer();
          const data = JSON.parse(new TextDecoder().decode(buf)) as AttributionResponse;
          attrCache.set(urlStrB, data);
          console.log(
            `[attr B] ${snapMethod} ${(performance.now() - tStart).toFixed(0)}ms ` +
            `cache=${data.cache_status ?? "?"} ` +
            `bytes=${buf.byteLength} ` +
            `target=${ExST.v1}${ExST.z1 && ExST.z1 !== "sum" ? "@" + ExST.z1 : ""} ` +
            `src=${sourceParam} ` +
            `keys=${Object.keys(data.attributions_prev).join(",")}`,
          );
          return data;
        });
    if (cachedB) console.log(`[attr B] front-cache hit src=${sourceParam}`);
    acquireB
      .then((data) => {
        // Discard if node, timestep, method, or compare mode changed while in-flight.
        // Clear loading if no newer fetch took over (otherwise the spinner sticks forever).
        if (ExST.node !== snapNode || ExST.t !== snapT || ExST.method !== snapMethod || ExST.compareMode !== snapMode) {
          if (attributionAbortB === ctrl) setExplainAttrLoadingB(false);
          return;
        }
        ExST.attributionDataB = data;
        setExplainAttributionDataB(data);
        setExplainAttrLoadingB(false);
      })
      .catch((err: unknown) => {
        if (err instanceof Error && err.name === "AbortError") return;
        if (ExST.node !== snapNode || ExST.t !== snapT || ExST.method !== snapMethod || ExST.compareMode !== snapMode) {
          if (attributionAbortB === ctrl) setExplainAttrLoadingB(false);
          return;
        }
        const msg = err instanceof Error ? err.message : "Attribution failed";
        setExplainAttrLoadingB(false);
        setExplainAttrErrorB(msg);
      });
  };

  // Fires the appropriate fetches for the current compareMode.
  // In Variables mode deduplicates when Panel A and Panel B use the same source:
  // a single request fires and its response is reused for both panels.
  const fireAttributionFetch = (): void => {
    if (ExST.compareMode === "variables") {
      const pASource = _buildPanelSource(ExST.pA_v2, ExST.pA_z2);
      const pBSource = _buildPanelSource(ExST.pB_v2, ExST.pB_z2);
      if (pASource === pBSource) {
        // Same source: abort any stale Panel B request, put Panel B into loading,
        // and let fireAttrA's .then() copy its response to Panel B.
        attributionAbortB?.abort();
        attributionAbortB = null;
        setExplainAttrLoadingB(true);
        setExplainAttrErrorB(null);
        fireAttrA();
      } else {
        // Different sources, but same backend cache key (target). Serialize A → B so
        // B hits the cache instead of duplicating the cold compute. Panel B's spinner
        // still appears immediately; only its fetch is deferred until A's response lands.
        setExplainAttrLoadingB(true);
        setExplainAttrErrorB(null);
        attributionAbortB?.abort();
        attributionAbortB = null;
        fireAttrA().then(() => fireAttrB());
      }
    } else {
      fireAttrA(); // timestamps mode: fireAttrA handles both panels
    }
  };

  // ── Model upload ─────────────────────────────────────────────────────────────
  // Ocean checkpoints ship as TorchScript exports only (SeaCast has no Lightning
  // checkpoint variant); weather models may be either. Mirrors
  // routes/model.py:ALLOWED_EXTENSIONS_BY_TYPE — keep both in sync.
  // Ocean checkpoints ship as TorchScript only; ERA5 models may also be Lightning
  // checkpoints. Mirrors ALLOWED_EXTENSIONS_BY_DOMAIN in routes/model.py.
  const MODEL_UPLOAD_ACCEPT: Record<"era5" | "ocean", string> = {
    era5: ".pt,.ckpt",
    ocean: ".pt",
  };

  const openModelUploadModal = (): void => {
    setModelUploadDomain("era5");
    setModelUploadName("");
    setModelUploadFile(null);
    setModelUploadError(null);
    setModelUploadOpen(true);
  };

  const closeModelUploadModal = (): void => {
    if (modelUploadSubmitting) return;
    setModelUploadOpen(false);
  };

  const handleModelUploadFileChange = (event: ChangeEvent<HTMLInputElement>): void => {
    const file = event.currentTarget.files?.[0] ?? null;
    setModelUploadFile(file);
    if (file && !modelUploadName) {
      setModelUploadName(file.name.replace(/\.(pt|ckpt)$/i, ""));
    }
  };

  const handleModelUploadSubmit = async (): Promise<void> => {
    if (!modelUploadFile) {
      setModelUploadError("Choose a checkpoint file to upload.");
      return;
    }
    const suffix = modelUploadFile.name.toLowerCase().slice(modelUploadFile.name.lastIndexOf("."));
    const allowed = MODEL_UPLOAD_ACCEPT[modelUploadDomain].split(",");
    if (!allowed.includes(suffix)) {
      setModelUploadError(
        modelUploadDomain === "ocean"
          ? "Ocean checkpoints are TorchScript-only — select a .pt file."
          : `Weather checkpoints must be one of ${allowed.join(", ")}.`
      );
      return;
    }
    setModelUploadSubmitting(true);
    setModelUploadError(null);
    try {
      const formData = new FormData();
      formData.append("name", modelUploadName || modelUploadFile.name);
      formData.append("domain", modelUploadDomain);
      formData.append("file", modelUploadFile, modelUploadFile.name);
      const response = await fetch(`${API}/models`, { method: "POST", body: formData });
      if (!response.ok) {
        const detail = await response.json().catch(() => null);
        throw new Error(detail?.detail ?? `Upload failed: ${response.status}`);
      }
      setModelUploadOpen(false);
      // The preset dropdown only lists the fixed presets above, not the DB model
      // catalog — the upload is registered and loadable by model_id, but isn't
      // auto-added to that selector yet.
      window.alert(
        `Model "${modelUploadName || modelUploadFile.name}" uploaded and registered. ` +
        "It isn't in the preset dropdown yet — switch to it by model_id, or wire it into " +
        "the catalog dropdown."
      );
    } catch (err) {
      setModelUploadError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setModelUploadSubmitting(false);
    }
  };

  // ── Model selection ────────────────────────────────────────────────────────

  // Selects by DB model_id rather than by preset key. POST /inference/switch_model only
  // accepts the two ERA5 preset keys and would reject any ocean model, and the endpoints it
  // sets up are resolved without a model_id, which the backend short-circuits to the ERA5
  // domain. Every /inference/{model_id}/... route resolves the model's own domain, cache and
  // device, so going through model_id is what makes non-ERA5 models work at all.
  const handleExplainModelSelect = (modelId: string): void => {
    if (explainModelSwitching || !modelId) return;
    if (modelId === activeExplainModelId) return;
    setExplainModelSwitching(modelId);
    setExplainSwitchPending(true);
    ExST.timelineData = null;
    setExplainTimelineData(null);
    setExplainComponentPeaks(null);

    // Invalidate the front-end caches: their keys (field: (t,v1,z1); attr: request URL)
    // don't encode the model, so entries from the previous model would otherwise be served
    // for the new one.
    clearFieldCache();
    attrCache.clear();

    void loadExplainModelRuntime(modelId)
      .then(() => {
        // Reset the target. The variable set is per-domain — ERA5 has atmospheric variables
        // across pressure levels, the ocean 27 flat features — so a selection carried across
        // a switch can name something the new model does not predict, and the drawer's node
        // index refers to a grid that may no longer exist.
        const isOcean = !!ExST.variablesMeta?.is_ocean;
        ExST.v1 = "total";
        setExplainV1("total");
        ExST.z1 = isOcean ? null : "sum";
        setExplainZ1(ExST.z1);
        ExST.node = null;
        setExplainNode(null);
        ExST.drawerOpen = false;
        setExplainDrawerOpen(false);
        writeExplainUrl();
        // Trigger ExplainMap to re-fetch prediction_errors and range for the new model.
        setExplainRefetchTrigger((n) => n + 1);
        if (ExST.drawerOpen && ExST.node !== null) {
          fireAttrA();
          if (ExST.compareMode === "variables") fireAttrB();
        }
      })
      .catch((err) => { console.error("[model switch]", err); })
      .finally(() => { setExplainModelSwitching(null); });
  };

  // ── Display mode toggle (Error ↔ Prediction) ──────────────────────────────
  // Switching to Prediction redirects v1 from "total" → first atm var, and z1 from
  // "sum" → "500", because aggregated cases have no physical meaning for predictions.
  const handleExplainDisplayModeChange = (newMode: "error" | "prediction"): void => {
    if (newMode === explainDisplayMode) return;
    setExplainDisplayMode(newMode);
    if (newMode === "prediction") {
      const isOcean = !!ExST.variablesMeta?.is_ocean;
      if (ExST.v1 === "total") {
        const fallbackV1 = isOcean ? (ExST.variablesMeta?.state_vars?.[0] ?? "total") : "temperature";
        ExST.v1 = fallbackV1;
        setExplainV1(fallbackV1);
        if (!isOcean && (ExST.z1 === null || ExST.z1 === "sum")) {
          ExST.z1 = "500";
          setExplainZ1("500");
        }
      } else if (!isOcean && ExST.z1 === "sum") {
        ExST.z1 = "500";
        setExplainZ1("500");
      }
      writeExplainUrl();
    }
  };

  // ── F1: ExplainLeftPanel callbacks ─────────────────────────────────────────

  const handleExplainV1Change = (newV1: string): void => {
    ExST.v1 = newV1;
    setExplainV1(newV1);

    if ((SURF_VARS as readonly string[]).includes(newV1) || ExST.variablesMeta?.is_ocean) {
      // Surface variables (and every ocean state channel — no pressure levels): no z1
      ExST.z1 = null;
      setExplainZ1(null);
    } else if (newV1 === "total") {
      // "total": keep current z1 unchanged
    } else {
      // Atmospheric variable: keep current z1 if set, otherwise default to 500
      if (ExST.z1 === null) {
        ExST.z1 = "500";
        setExplainZ1("500");
      }
    }

    // Auto-close drawer when attribution is now blocked
    if (newV1 === "total" && ExST.drawerOpen) {
      ExST.drawerOpen = false;
      setExplainDrawerOpen(false);
    }
    // Target variable change invalidates the cached attribution. Clear so the
    // drawer mini-maps + ranking sidebar show loading state during refetch instead
    // of briefly displaying the previous variable's attribution.
    ExST.attributionData = null;
    ExST.attributionDataB = null;
    setExplainAttributionData(null);
    setExplainAttributionDataB(null);
    writeExplainUrl();
    // Re-fetch if drawer is still open with a valid target (source follows target)
    if (ExST.drawerOpen && ExST.node !== null) {
      ExST.pA_v2 = ExST.v1;
      ExST.pA_z2 = ExST.z1;
      setExplainPAV2(ExST.v1);
      setExplainPAZ2(ExST.z1);
      fireAttributionFetch();
    }
  };

  const handleExplainZ1Change = (newZ1: string): void => {
    ExST.z1 = newZ1;
    setExplainZ1(newZ1);
    // Auto-close drawer when Σ levels selected
    if (newZ1 === "sum" && ExST.drawerOpen) {
      ExST.drawerOpen = false;
      setExplainDrawerOpen(false);
    }
    // Target level change invalidates the cached attribution. Clear so the
    // ranking sidebar shows loading state during refetch.
    ExST.attributionData = null;
    ExST.attributionDataB = null;
    setExplainAttributionData(null);
    setExplainAttributionDataB(null);
    writeExplainUrl();
    // Re-fetch if drawer is still open with a specific level (source follows target)
    if (ExST.drawerOpen && ExST.node !== null) {
      ExST.pA_v2 = ExST.v1;
      ExST.pA_z2 = ExST.z1;
      setExplainPAV2(ExST.v1);
      setExplainPAZ2(ExST.z1);
      fireAttributionFetch();
    }
  };

  // Combined (v1, z1) target change fired by the drawer's "Compute" button — a
  // single atomic update + single attribution refetch, rather than the two
  // separate fetches that handleExplainV1Change + handleExplainZ1Change would
  // trigger back to back. Only reachable with a specific variable + specific
  // level (the drawer's target selector never offers "total" / "sum").
  const handleExplainComputeTarget = (newV1: string, newZ1: string | null): void => {
    if (newV1 === ExST.v1 && newZ1 === ExST.z1) return;
    ExST.v1 = newV1;
    setExplainV1(newV1);
    ExST.z1 = newZ1;
    setExplainZ1(newZ1);
    ExST.attributionData = null;
    ExST.attributionDataB = null;
    setExplainAttributionData(null);
    setExplainAttributionDataB(null);
    writeExplainUrl();
    // Node selection is unaffected — always refetch against the current
    // (applied) node, never a pending/uncommitted one, so a node change on
    // the main map is never blocked by an in-progress target edit here.
    if (ExST.drawerOpen && ExST.node !== null) {
      ExST.pA_v2 = newV1;
      ExST.pA_z2 = newZ1;
      setExplainPAV2(newV1);
      setExplainPAZ2(newZ1);
      if (ExST.compareMode === "variables") {
        ExST.pB_v2 = newV1;
        ExST.pB_z2 = newZ1;
        setExplainPBV2(newV1);
        setExplainPBZ2(newZ1);
      }
      fireAttributionFetch();
    }
  };

  // "Flag" the current (target v1@z1, node, t) as a saved config chip. In-memory
  // only, FIFO-evicted beyond MAX_SAVED_EXPLAIN_CONFIGS, cleared on reload
  // alongside the IG cache and everything else in ExST.
  const handleExplainSaveConfig = (): void => {
    if (ExST.node === null || ExST.v1 === "total" || ExST.z1 === "sum") return;
    const { v1, z1, node, t } = ExST;
    const label = buildSavedExplainLabel(v1, z1, node, t, explainGrid);
    const id = `${v1}|${z1}|${node}|${t}|${Date.now()}`;
    setExplainSavedConfigs((prev) => {
      const next = [...prev, { id, v1, z1, node, t, label }];
      return next.length > MAX_SAVED_EXPLAIN_CONFIGS
        ? next.slice(next.length - MAX_SAVED_EXPLAIN_CONFIGS)
        : next;
    });
  };

  // Restore a saved config: jumps timestep, target, and node in one shot, then
  // fires a single attribution refetch against the restored state.
  const handleExplainRestoreConfig = (cfg: SavedExplainConfig): void => {
    if (ExST.t !== cfg.t) {
      ExST.t = cfg.t;
      setExplainT(cfg.t);
    }
    ExST.v1 = cfg.v1;
    setExplainV1(cfg.v1);
    ExST.z1 = cfg.z1;
    setExplainZ1(cfg.z1);
    ExST.node = cfg.node;
    setExplainNode(cfg.node);

    ExST.attributionData = null;
    ExST.attributionDataB = null;
    setExplainAttributionData(null);
    setExplainAttributionDataB(null);

    if (!ExST.drawerOpen) {
      ExST.drawerOpen = true;
      setExplainDrawerOpen(true);
    }
    if (ExST.drawerHeaderOnly) {
      ExST.drawerHeaderOnly = false;
      setExplainDrawerHeaderOnly(false);
    }
    ExST.pA_v2 = cfg.v1;
    ExST.pA_z2 = cfg.z1;
    setExplainPAV2(cfg.v1);
    setExplainPAZ2(cfg.z1);
    if (ExST.compareMode === "variables") {
      ExST.pB_v2 = cfg.v1;
      ExST.pB_z2 = cfg.z1;
      setExplainPBV2(cfg.v1);
      setExplainPBZ2(cfg.z1);
    }
    writeExplainUrl();
    rebuildExplainLayers();
    fireAttributionFetch();
  };

  const handleExplainTChange = (newT: string): void => {
    ExST.t = newT;
    setExplainT(newT);
    // Timestep change invalidates the cached attribution (cache key includes pred_time).
    // Clear it so the drawer mini-maps shimmer and the ranking sidebar shows its
    // loading spinner while the new data is fetched, rather than briefly displaying
    // the previous timestep's attribution as if it were current.
    ExST.attributionData = null;
    ExST.attributionDataB = null;
    setExplainAttributionData(null);
    setExplainAttributionDataB(null);
    writeExplainUrl();
    // Re-fetch with new timestep when drawer is open
    if (ExST.drawerOpen && ExST.node !== null) {
      fireAttributionFetch();
    }
  };

  const handleBandClick = (band: MonthBand): void => {
    if (!GCS_FETCH_ENABLED) return;
    // Cancel any in-flight request; navigation intent is always "show me THIS month"
    bandClickAbortRef.current?.abort();
    const ctrl = new AbortController();
    bandClickAbortRef.current = ctrl;
    setWindowLoading(true);
    setLoadingBand(band.start);

    const run = async () => {
      for (let attempt = 0; attempt < 36; attempt++) {
        if (ctrl.signal.aborted) return;
        try {
          const r = await fetch(`${API}/inference/${activeExplainModelId}/load_window`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ start: band.start, end: band.end }),
            signal: ctrl.signal,
          });
          if (ctrl.signal.aborted) return;
          if (r.ok) {
            const data: LoadedWindow = await r.json();
            if (ctrl.signal.aborted) return;
            const s = new Set(data.timestamps);
            ExST.availableTimestamps = s;
            _explainAvailableTimestampsBridge.set?.(s);
            setAllDownloadedTimestamps(prev => { const n = new Set(prev); data.timestamps.forEach(t => n.add(t)); return n; });
            setDownloadedBandStarts(prev => new Set([...prev, band.start]));
            setWindowLoading(false);
            setLoadingBand(null);
            if (data.timestamps.length > 0) {
              handleExplainTChange(data.timestamps[0]);
            }
            return;
          } else if (r.status === 409) {
            // Backend lock held by previous download; retry after 5s
            if (attempt < 35) {
              await new Promise<void>(res => setTimeout(res, 5000));
            } else {
              setWindowLoading(false);
              setLoadingBand(null);
              console.error("load_window: max 409 retries exceeded");
            }
          } else if (r.status === 503) {
            setWindowLoading(false);
            setLoadingBand(null);
            console.error("load_window: manager not initialized (503)");
            return;
          } else if (r.status === 400) {
            setWindowLoading(false);
            setLoadingBand(null);
            console.error("load_window: band start/end not in timeline (400)");
            return;
          } else if (r.status === 404) {
            // Unreachable in normal flow per spec — availableTimestamps guards navigation
            console.warn("load_window: 404, ignoring");
            setWindowLoading(false);
            setLoadingBand(null);
            return;
          } else {
            setWindowLoading(false);
            setLoadingBand(null);
            console.error("load_window: unexpected status", r.status);
            return;
          }
        } catch (err) {
          if (ctrl.signal.aborted) return;
          setWindowLoading(false);
          setLoadingBand(null);
          console.error("load_window: network error", err);
          return;
        }
      }
    };
    void run();
  };

  const handleExplainNodeClear = (): void => {
    ExST.node = null;
    setExplainNode(null);
    // Abort in-flight fetches
    attributionAbortA?.abort();
    attributionAbortA = null;
    attributionAbortB?.abort();
    attributionAbortB = null;
    // Clear attribution state
    ExST.attributionData = null;
    ExST.attributionDataB = null;
    setExplainAttributionData(null);
    setExplainAttributionDataB(null);
    setExplainAttrLoadingA(false);
    setExplainAttrLoadingB(false);
    setExplainAttrErrorA(null);
    setExplainAttrErrorB(null);
    ExST.pA_v2 = null;
    ExST.pA_z2 = null;
    ExST.pB_v2 = null;
    ExST.pB_z2 = null;
    setExplainPAV2(null);
    setExplainPAZ2(null);
    setExplainPBV2(null);
    setExplainPBZ2(null);
    if (ExST.drawerOpen) {
      ExST.drawerOpen = false;
      setExplainDrawerOpen(false);
    }
    writeExplainUrl();
    rebuildExplainLayers();
  };

  const handleOpenAttribution = (): void => {
    if (!ExST.drawerOpen) {
      ExST.drawerOpen = true;
      setExplainDrawerOpen(true);
    }
    if (ExST.drawerHeaderOnly) {
      ExST.drawerHeaderOnly = false;
      setExplainDrawerHeaderOnly(false);
    }
    if (ExST.node === null) return;
    if (!isAttrCurrent()) {
      ExST.pA_v2 = ExST.v1;
      ExST.pA_z2 = ExST.z1;
      setExplainPAV2(ExST.v1);
      setExplainPAZ2(ExST.z1);
      fireAttrA();
    }
    if (ExST.compareMode === "variables" && !isAttrBCurrent()) {
      const initV2 = ExST.pB_v2 ?? ExST.v1;
      const initZ2 = ExST.pB_z2 ?? ExST.z1;
      ExST.pB_v2 = initV2;
      ExST.pB_z2 = initZ2;
      setExplainPBV2(initV2);
      setExplainPBZ2(initZ2);
      fireAttrB();
    }
  };

  const handleExplainNodeClick = (nodeIndex: number): void => {
    ExST.node = nodeIndex;
    setExplainNode(nodeIndex);

    const blocked = ExST.v1 === "total" || ExST.z1 === "sum";
    if (!blocked) {
      // Open drawer on first node click; if already open it will re-fetch in F5
      if (!ExST.drawerOpen) {
        ExST.drawerOpen = true;
        setExplainDrawerOpen(true);
      }
      // Set source defaults to current target, then fire the attribution fetch.
      // fireAttributionFetch handles aborting any in-flight requests.
      ExST.pA_v2 = ExST.v1;
      ExST.pA_z2 = ExST.z1;
      setExplainPAV2(ExST.v1);
      setExplainPAZ2(ExST.z1);
      if (ExST.compareMode === "variables") {
        ExST.pB_v2 = ExST.v1;
        ExST.pB_z2 = ExST.z1;
        setExplainPBV2(ExST.v1);
        setExplainPBZ2(ExST.z1);
      }
      fireAttributionFetch();
    }

    // Rebuild layers so the selection ring moves to the new cell immediately
    rebuildExplainLayers();
    writeExplainUrl();
  };

  // Keep bridge in sync on every render so clickHandler always calls the latest closure
  _explainNodeClickBridge.handler = handleExplainNodeClick;

  // ── F5: drawer shell handlers ───────────────────────────────────────────────

  const handleExplainHide = (): void => {
    if (ExST.drawerHeaderOnly) {
      // Second press (while header-only): fully close
      ExST.drawerHeaderOnly = false;
      ExST.drawerOpen = false;
      setExplainDrawerHeaderOnly(false);
      setExplainDrawerOpen(false);
    } else {
      // First press: collapse to header-only strip
      ExST.drawerHeaderOnly = true;
      setExplainDrawerHeaderOnly(true);
    }
    writeExplainUrl();
  };

  const handleExplainExpand = (): void => {
    if (!ExST.drawerHeaderOnly) return;
    ExST.drawerHeaderOnly = false;
    setExplainDrawerHeaderOnly(false);
    writeExplainUrl();
  };

  const handleExplainMethodChange = (newMethod: "ixg" | "ig"): void => {
    if (newMethod === ExST.method) return;
    ExST.method = newMethod;
    setExplainMethod(newMethod);
    // Clear both panels' cached responses so the map cells, colorbar, and ranking
    // sidebar don't show stale data from the previous method while the new one computes.
    ExST.attributionData = null;
    ExST.attributionDataB = null;
    setExplainAttributionData(null);
    setExplainAttributionDataB(null);
    writeExplainUrl();
    if (ExST.drawerOpen && ExST.node !== null) {
      fireAttributionFetch();
    }
  };

  const handleExplainCompareModeChange = (newMode: "timestamps" | "variables"): void => {
    if (newMode === ExST.compareMode) return;
    ExST.compareMode = newMode;
    setExplainCompareMode(newMode);
    writeExplainUrl();
    if (!ExST.drawerOpen || ExST.node === null) return;
    if (newMode === "variables") {
      // Default Panel B to same source as Panel A on first entry
      const initV2 = ExST.pA_v2 ?? ExST.v1;
      const initZ2 = ExST.pA_z2 ?? ExST.z1;
      ExST.pB_v2 = initV2;
      ExST.pB_z2 = initZ2;
      setExplainPBV2(initV2);
      setExplainPBZ2(initZ2);
      // Panel B starts with the same source as Panel A. Panel A's cached response
      // already contains that source key — reuse it directly, no second request needed.
      if (ExST.attributionData !== null) {
        ExST.attributionDataB = ExST.attributionData;
        setExplainAttributionDataB(ExST.attributionData);
      } else {
        fireAttrB();
      }
    } else {
      // Back to timestamps — clear stale Panel B data, Panel A re-fetches
      ExST.attributionDataB = null;
      setExplainAttributionDataB(null);
      fireAttrA();
    }
  };

  // ── F7: per-panel source variable handlers ─────────────────────────────────

  const handleExplainPAV2Change = (newV2: string): void => {
    ExST.pA_v2 = newV2;
    ExST.pA_z2 = isSurfaceVar(newV2) ? null : (ExST.pA_z2 ?? ExST.z1 ?? "50");
    setExplainPAV2(newV2);
    setExplainPAZ2(ExST.pA_z2);
    writeExplainUrl();
    if (ExST.drawerOpen && ExST.node !== null) fireAttrA();
  };

  const handleExplainPAZ2Change = (newZ2: string): void => {
    ExST.pA_z2 = newZ2;
    setExplainPAZ2(newZ2);
    if (ExST.z2Locked) {
      ExST.pB_z2 = newZ2;
      setExplainPBZ2(newZ2);
    }
    writeExplainUrl();
    if (ExST.drawerOpen && ExST.node !== null) {
      fireAttrA();
      if (ExST.z2Locked && ExST.compareMode === "variables") fireAttrB();
    }
  };

  const handleExplainPBV2Change = (newV2: string): void => {
    ExST.pB_v2 = newV2;
    ExST.pB_z2 = isSurfaceVar(newV2) ? null : (ExST.pB_z2 ?? ExST.z1 ?? "50");
    setExplainPBV2(newV2);
    setExplainPBZ2(ExST.pB_z2);
    writeExplainUrl();
    if (ExST.drawerOpen && ExST.node !== null) fireAttrB();
  };

  const handleExplainPBZ2Change = (newZ2: string): void => {
    ExST.pB_z2 = newZ2;
    setExplainPBZ2(newZ2);
    if (ExST.z2Locked) {
      ExST.pA_z2 = newZ2;
      setExplainPAZ2(newZ2);
    }
    writeExplainUrl();
    if (ExST.drawerOpen && ExST.node !== null) {
      fireAttrB();
      if (ExST.z2Locked && ExST.compareMode === "variables") fireAttrA();
    }
  };

  const handleExplainZ2LockChange = (locked: boolean): void => {
    ExST.z2Locked = locked;
    setExplainZ2Locked(locked);
    if (locked) {
      // Sync Panel B z2 to Panel A on lock
      const syncZ2 = ExST.pA_z2 ?? ExST.z1;
      if (ExST.pB_z2 !== syncZ2) {
        ExST.pB_z2 = syncZ2;
        setExplainPBZ2(syncZ2);
        writeExplainUrl();
        if (ExST.drawerOpen && ExST.node !== null && ExST.compareMode === "variables") fireAttrB();
        return;
      }
    }
    writeExplainUrl();
  };

  const handleDrawerTransitionEnd = (): void => {
    // Called by ExplainDrawer at its CSS transform transitionend.
    // The main Leaflet map needs invalidateSize after its container height settles.
    leafMap?.invalidateSize();
  };

  useEffect(() => {
    if (!showIntro) { setSpotlightRect(null); return; }
    const targetId = ONBOARD_STEPS[introStep].targetId;
    if (!targetId) { setSpotlightRect(null); return; }
    const update = () => {
      const el = document.getElementById(targetId);
      if (el) { const r = el.getBoundingClientRect(); setSpotlightRect({ top: r.top, left: r.left, width: r.width, height: r.height }); }
    };
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, [showIntro, introStep]);

  const dismissIntro = (): void => {
    document.cookie = `${ONBOARD_COOKIE}; max-age=31536000; path=/`;
    setShowIntro(false);
    setIntroStep(0);
  };

  useEffect(() => {
    if (!showExploreTour) { setExploreSpotlightRect(null); return; }
    const targetId = EXPLORE_STEPS[exploreTourStep]?.targetId;
    if (!targetId) { setExploreSpotlightRect(null); return; }
    const update = () => {
      const el = document.getElementById(targetId);
      if (el) { const r = el.getBoundingClientRect(); setExploreSpotlightRect({ top: r.top, left: r.left, width: r.width, height: r.height }); }
    };
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, [showExploreTour, exploreTourStep]);

  const dismissExploreTour = (): void => {
    document.cookie = `${EXPLORE_COOKIE}; max-age=31536000; path=/`;
    setShowExploreTour(false);
    setExploreTourStep(0);
  };
  const openExploreTour = (): void => { setExploreTourStep(0); setShowExploreTour(true); };

  useEffect(() => {
    if (!showExplainTour) { setExplainSpotlightRect(null); return; }
    const stepTargetId = EXPLAIN_STEPS[explainTourStep]?.targetId;
    // Steps 8–(N-2) show the attribution drawer. Steps 0–7 and the last step close it.
    const needsDrawer = !["explain-help-btn", null, undefined].includes(stepTargetId) &&
      ["explain-drawer-root", "ex-ranking-wrap", "ex-drawer-compare"].includes(stepTargetId ?? "");
    if (needsDrawer) {
      if (!explainDrawerOpen) {
        ExST.drawerOpen = true;
        setExplainDrawerOpen(true);
      }
      if (explainDrawerHeaderOnly) {
        ExST.drawerHeaderOnly = false;
        setExplainDrawerHeaderOnly(false);
      }
    } else {
      if (explainDrawerOpen) {
        ExST.drawerOpen = false;
        setExplainDrawerOpen(false);
      }
      if (explainDrawerHeaderOnly) {
        ExST.drawerHeaderOnly = false;
        setExplainDrawerHeaderOnly(false);
      }
    }
    const targetId = EXPLAIN_STEPS[explainTourStep].targetId;
    if (!targetId) { setExplainSpotlightRect(null); return; }
    const update = () => {
      const el = document.getElementById(targetId);
      if (el) { const r = el.getBoundingClientRect(); setExplainSpotlightRect({ top: r.top, left: r.left, width: r.width, height: r.height }); }
    };
    // Small delay so drawer open animation completes before measuring
    const t = setTimeout(() => { update(); window.addEventListener("resize", update); }, 320);
    return () => { clearTimeout(t); window.removeEventListener("resize", update); };
  }, [showExplainTour, explainTourStep, explainDrawerOpen, explainDrawerHeaderOnly]);

  const dismissExplainTour = (): void => {
    document.cookie = `${EXPLAIN_COOKIE}; max-age=31536000; path=/`;
    setShowExplainTour(false);
    setExplainTourStep(0);
  };
  const openExplainTour = (): void => { setExplainTourStep(0); setShowExplainTour(true); };

  useEffect(() => {
    if (!showInspectTour) { setInspectSpotlightRect(null); return; }
    const targetId = INSPECT_STEPS[inspectTourStep]?.targetId;
    if (!targetId) { setInspectSpotlightRect(null); return; }
    const measure = () => {
      const el = document.getElementById(targetId);
      if (el) {
        const r = el.getBoundingClientRect();
        if (r.width > 0 && r.height > 0)
          setInspectSpotlightRect({ top: r.top, left: r.left, width: r.width, height: r.height });
        else
          setInspectSpotlightRect(null);
      }
    };
    if (targetId === "edge-len-section") {
      const el = document.getElementById(targetId);
      el?.scrollIntoView({ behavior: "smooth", block: "end" });
      const t = setTimeout(measure, 350);
      window.addEventListener("resize", measure);
      return () => { clearTimeout(t); window.removeEventListener("resize", measure); };
    }
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [showInspectTour, inspectTourStep]);

  const dismissInspectTour = (): void => {
    document.cookie = `${INSPECT_COOKIE}; max-age=31536000; path=/`;
    setInspectSpotlightRect(null);
    setShowInspectTour(false);
    setInspectTourStep(0);
  };
  const openInspectTour = (): void => { setInspectTourStep(0); setShowInspectTour(true); };

  useEffect(() => {
    if (!document.cookie.includes(ONBOARD_COOKIE)) setShowIntro(true);
    const lp = loadLenPlotCookie();
    const binsEl = document.getElementById("len-bins") as HTMLInputElement | null;
    if (binsEl) binsEl.value = String(lp.bins);
  }, []);

  // Sync explain-drawer-open / explain-drawer-header classes on #app so CSS layout rules can react
  useEffect(() => {
    const app = document.getElementById("app");
    if (!app) return;
    const isFullOpen = explainDrawerOpen && !explainDrawerHeaderOnly;
    const isHeaderOnly = explainDrawerOpen && explainDrawerHeaderOnly;
    app.classList.toggle("explain-drawer-open", isFullOpen);
    app.classList.toggle("explain-drawer-header", isHeaderOnly);
  }, [explainDrawerOpen, explainDrawerHeaderOnly]);

  // Recalculate colorbar max when shared colorbar toggle changes.
  // Don't call rebuildExplainLayers() here — ExplainMap's own colorbarMax effect
  // will fire once the new prop propagates and call onRebuildLayers() with the
  // correct currentScaleMax. A direct call here would race against that update.
  useEffect(() => {
    if (explainColorbarErrors && explainColorbarErrors.length > 0) {
      if (explainSharedColorbar) {
        setExplainColorbarMax(explainColorbarBackendMax);
      } else {
        setExplainColorbarMax(maxOfArray(explainColorbarErrors));
      }
    }
  }, [explainSharedColorbar, explainColorbarErrors, explainColorbarBackendMax]);

  API = apiBaseUrl.replace(/\/+$/, "") || "http://localhost:8080";
  cartoApiKey = cartoApiKeyProp?.trim() ?? "";

  const formatGraphLastModified = (graph: GraphRecord): string => {
    const stamp = graph.updated_at ?? graph.created_at;
    if (!stamp) return "n/a";
    const parsed = new Date(stamp);
    if (Number.isNaN(parsed.getTime())) return "n/a";
    return new Intl.DateTimeFormat("en-CH", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    }).format(parsed);
  };

  const refreshGraphOptions = async (preferredGraphId?: string): Promise<string | null> => {
    const graphs = await fetchGraphCatalog();
    graphCatalog = graphs;
    setGraphOptions(graphs);
    if (!graphs.length) return null;

    const preferred = preferredGraphId ?? activeGraphId;
    const hasPreferred = graphs.some((g) => g.graph_id === preferred);
    const target = hasPreferred ? preferred : graphs[0].graph_id;
    activeGraphId = target;
    setSelectedGraphId(target);
    updateGraphNameDisplay();
    return target;
  };

  const refreshExplainModelOptions = async (preferredModelId?: string): Promise<string | null> => {
    const models = await fetchModelCatalog();
    setExplainModelOptions(models);
    if (!models.length) return null;

    const preferred = preferredModelId ?? activeExplainModelId;
    const hasPreferred = models.some((m) => m.model_id === preferred);
    const target = hasPreferred ? preferred : models[0].model_id;
    activeExplainModelId = target;
    setSelectedExplainModelId(target);
    return target;
  };

  const loadExplainModelRuntime = async (modelId: string): Promise<void> => {
    ExST.timelineData = null;
    setExplainTimelineData(null);
    setExplainComponentPeaks(null);
    const [modelInfoRes, variablesRes, gridRes, timelineRes, componentsRes, loadedWindowRes, cachedRes] = await Promise.all([
      fetch(`${API}/inference/${modelId}/model_info`).then((r) => r.ok ? r.json() : null),
      fetch(`${API}/inference/${modelId}/variables`).then((r) => r.ok ? r.json() : null),
      fetch(`${API}/inference/${modelId}/grid`).then((r) => r.ok ? r.json() : null),
      fetch(`${API}/inference/${modelId}/prediction_errors/timeline`).then((r) => r.ok ? r.json() : null),
      // 404s for models whose artifact predates peak_top3 — the panel just shows no marks.
      fetch(`${API}/inference/${modelId}/prediction_errors/timeline/components`).then((r) => r.ok ? r.json() : null),
      fetch(`${API}/inference/${modelId}/loaded_window`).then((r) => r.ok ? r.json() : null),
      fetch(`${API}/inference/${modelId}/cached_timestamps`).then((r) => r.ok ? r.json() : null),
    ]);

    const modelInfoData = modelInfoRes as ModelInfo | null;
    const variablesData = variablesRes as VariablesMeta | null;
    const gridData = gridRes as GridData | null;
    const timelineData = timelineRes as TimelineData | null;
    const componentPeaks = componentsRes as ComponentPeaks | null;
    const loadedWindowData = loadedWindowRes as LoadedWindow | null;
    const cachedData = cachedRes as { timestamps?: string[] } | null;

    if (modelInfoData) {
      ExST.modelInfo = modelInfoData;
      setExplainModelInfo(modelInfoData);
    }
    if (variablesData) {
      ExST.variablesMeta = variablesData;
      setExplainVariablesMeta(variablesData);
    }
    if (gridData) {
      ExST.grid = gridData;
      setExplainGrid(gridData);
    }
    if (timelineData) {
      ExST.timelineData = timelineData;
      setExplainTimelineData(timelineData);
    }
    setExplainComponentPeaks(componentPeaks);

    // `covered_timestamps` is the same set /loaded_window returns, carried on the timeline
    // response. Falling back to it matters because availability drives which points are
    // clickable: with neither source, ExplainTimeline treats every scored date as available
    // and a click lands outside the window.
    const windowTimestamps =
      loadedWindowData?.timestamps ?? timelineData?.covered_timestamps ?? null;

    if (windowTimestamps) {
      const available = new Set(windowTimestamps);
      ExST.availableTimestamps = available;
      setExplainAvailableTimestamps(available);

      const merged = new Set(windowTimestamps);
      if (cachedData?.timestamps) {
        cachedData.timestamps.forEach((ts) => merged.add(ts));
      }
      setAllDownloadedTimestamps(merged);

      const fallbackT = windowTimestamps[0] ?? "";
      const nextT = (ExST.t && available.has(ExST.t)) ? ExST.t : fallbackT;
      ExST.t = nextT;
      setExplainT(nextT);
    }

    // Publish the model id HERE, in the same tick as the timestamp, so the two always change
    // together. Setting it earlier (when the user clicks) leaves a render where the new
    // model is paired with the previous domain's timestamp, and the fetches 404 against a
    // window that does not contain it -- in whichever direction the switch went.
    activeExplainModelId = modelId;
    setSelectedExplainModelId(modelId);

    ExST.attributionData = null;
    ExST.attributionDataB = null;
    setExplainAttributionData(null);
    setExplainAttributionDataB(null);
    setExplainAttrErrorA(null);
    setExplainAttrErrorB(null);
    setExplainNode(null);
    ExST.node = null;
    rebuildExplainLayers();
  };


  useEffect(() => {
    if (!UPLOADS_ENABLED) return;
    if (!uploadInputRef.current) return;
    // Try to enable folder selection with webkitdirectory for better neural-lam UX
    // On browsers/OS without folder picker support, falls back to file selection
    uploadInputRef.current.setAttribute("webkitdirectory", "");
    uploadInputRef.current.setAttribute("directory", "");
    uploadInputRef.current.setAttribute("multiple", "");
  }, []);

  // ── On mount: fetch the active loaded window from the manager ────────────────
  // Populates availableTimestamps immediately (seed zarr is always ready at startup).
  // On 503 (manager init failed), retries every 5s up to 6 times then shows a banner.
  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      for (let attempt = 0; attempt < 6; attempt++) {
        if (cancelled) return;
        try {
          const r = await fetch(`${API}/inference/${activeExplainModelId}/loaded_window`);
          if (cancelled) return;
          if (r.ok) {
            const data: LoadedWindow = await r.json();
            if (cancelled) return;
            const s = new Set(data.timestamps);
            ExST.availableTimestamps = s;
            _explainAvailableTimestampsBridge.set?.(s);
            // Seed allDownloadedTimestamps with the active window, then merge all local-cache zarrs
            const accumulated = new Set(data.timestamps);
            try {
              const rc = await fetch(`${API}/inference/${activeExplainModelId}/cached_timestamps`);
              if (!cancelled && rc.ok) {
                const cd: { timestamps: string[] } = await rc.json();
                cd.timestamps.forEach(t => accumulated.add(t));
              }
            } catch { /* non-fatal */ }
            if (!cancelled) setAllDownloadedTimestamps(accumulated);
            if (!ExST.t && data.timestamps.length > 0) {
              ExST.t = data.timestamps[0];
              _explainTBridge.set?.(ExST.t);
            }
            return;
          } else if (r.status === 503) {
            if (attempt < 5) {
              await new Promise<void>(res => setTimeout(res, 5000));
            } else {
              if (!cancelled) setEra5InitError(true);
            }
          } else {
            if (!cancelled) setEra5InitError(true);
            return;
          }
        } catch {
          if (!cancelled) setEra5InitError(true);
          return;
        }
      }
    };
    void run();
    return () => { cancelled = true; };
  }, []);

  const openUploadPicker = (): void => {
    if (!UPLOADS_ENABLED) { window.alert("Graph uploads are disabled by configuration."); return; }
    if (isUploading || isSwitchingGraph) return;
    uploadInputRef.current?.click();
  };

  const handleUploadChange = async (event: ChangeEvent<HTMLInputElement>): Promise<void> => {
    if (!UPLOADS_ENABLED) { event.currentTarget.value = ""; return; }
    const selectedFiles = Array.from(event.currentTarget.files ?? []);
    event.currentTarget.value = "";
    if (!selectedFiles.length || isUploading || isSwitchingGraph) return;

    setIsUploading(true);
    try {
      const created = await uploadGraphFolder(selectedFiles);
      await refreshGraphOptions(created.graph_id);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setLoadingText("Upload failed");
      setLoading(false);
      window.alert(`Unable to upload graph: ${message}`);
    } finally {
      setIsUploading(false);
      setLoading(false);
      setLoadingText("Loading...");
    }
  };

  const handleGraphSelectChange = async (event: ChangeEvent<HTMLSelectElement>): Promise<void> => {
    const nextGraphId = event.currentTarget.value;
    if (!nextGraphId || nextGraphId === activeGraphId || isUploading || isSwitchingGraph) return;

    const previousGraphId = activeGraphId;
    setSelectedGraphId(nextGraphId);
    setIsSwitchingGraph(true);
    setLoadingText("Loading...");
    setLoading(true);

    try {
      activeGraphId = nextGraphId;
      updateGraphNameDisplay();
      await reloadActiveGraphData();
    } catch (err) {
      activeGraphId = previousGraphId;
      setSelectedGraphId(previousGraphId);
      updateGraphNameDisplay();
      const message = err instanceof Error ? err.message : String(err);
      // reloadActiveGraphData clears the previous graph's tensors before it fetches, so simply
      // restoring the id would leave the picker pointing at a graph whose data is no longer on
      // screen. Reload it. A second failure means the previous graph is unservable too; report
      // the original failure and leave the map empty rather than looping.
      try {
        await reloadActiveGraphData();
      } catch (restoreErr) {
        console.warn("Unable to restore the previous graph after a failed switch:", restoreErr);
      }
      window.alert(`Unable to switch graph: ${message}`);
    } finally {
      setIsSwitchingGraph(false);
      setLoading(false);
      setLoadingText("Loading...");
    }
  };

  useEffect(() => {
    resetState();
    buildIcons();

    const mapEl = getEl<HTMLDivElement>("map");
    const miniMapEl = getEl<HTMLDivElement>("minimap-map");

    leafMap = L.map(mapEl, { center: [20, 0], zoom: 4, minZoom: 1, maxZoom: 8, maxBounds: [[-85.05, -100000], [85.05, 100000]], maxBoundsViscosity: 1.0, zoomControl: false, attributionControl: true });
    activeTileLayer = L.tileLayer(resolveMapUrl(MAP_TYPES.map.url), { attribution: MAP_TYPES.map.attr, ...MAP_TYPES.map.opts }).addTo(leafMap);
    document.getElementById("zoom-in-btn")?.addEventListener("click", () => leafMap.zoomIn());
    document.getElementById("zoom-out-btn")?.addEventListener("click", () => leafMap.zoomOut());

    // Keep minZoom in sync with container height so the world edges never
    // leave the viewport (no white margins at max zoom-out).
    let mapMinZoomActive = true;
    const updateMapMinZoom = (): void => {
      if (!mapMinZoomActive) return;
      const h = mapEl.clientHeight;
      if (h === 0) return;
      const minZ = Math.log2(h / 256);
      leafMap.setMinZoom(minZ);
      if (leafMap.getZoom() < minZ) leafMap.setZoom(minZ, { animate: false });
    };
    const mapSizeObserver = new ResizeObserver(updateMapMinZoom);
    mapSizeObserver.observe(mapEl);
    updateMapMinZoom();

    miniMap = L.map(miniMapEl, {
      center: [0, 0],
      zoom: 0,
      minZoom: 0,
      maxZoom: 0,
      zoomControl: false,
      attributionControl: false,
      dragging: false,
      touchZoom: false,
      scrollWheelZoom: false,
      doubleClickZoom: false,
      boxZoom: false,
      keyboard: false,
    });
    activeMiniTileLayer = L.tileLayer(resolveMapUrl(MAP_TYPES.map.url), { ...MAP_TYPES.map.opts }).addTo(miniMap);
    const rectStyle = { color: MINIMAP_ACCENT, weight: 2, fillColor: MINIMAP_ACCENT, fillOpacity: 0.12, interactive: false };
    viewRect = L.rectangle([[0, 0], [0, 0]], rectStyle).addTo(miniMap);
    viewRectWrap = L.rectangle([[0, 0], [0, 0]], rectStyle).addTo(miniMap);

    deckCanvas = document.createElement("canvas");
    deckCanvas.style.cssText = "position:absolute;top:0;left:0;pointer-events:none;z-index:500;";
    getEl<HTMLDivElement>("map-wrap").appendChild(deckCanvas);

    updateGraphStats();
    updateSrcDegSliders(false);
    updateTgtDegSliders(false);
    updateLenSliders(false);
    updateInfoPanel();

    // Wire the model-info bridge so setAppMode (module-level) can trigger React re-render
    _explainModelInfoBridge.set = setExplainModelInfo;
    _explainGridBridge.set = setExplainGrid;
    _explainTBridge.set = setExplainT;
    _explainTimelineBridge.set = setExplainTimelineData;
    _explainAvailableTimestampsBridge.set = setExplainAvailableTimestamps;
    _explainCoastlineBridge.set = setExplainCoastlineGeoJson;
    _explainDrawerBridge.set = setExplainDrawerOpen;
    _explainDrawerHeaderOnlyBridge.set = setExplainDrawerHeaderOnly;
    _loadExplainModelRuntimeBridge.run = loadExplainModelRuntime;
    _openExplainTourBridge = openExplainTour;
    _openInspectTourBridge = openInspectTour;
    _setMetaLevelsBridge.set = setMetaLevels;
    _setMetaNodeCountsBridge.set = setMetaNodeCounts;
    _setActiveInspectSubgraphBridge.set = setActiveInspectSubgraph;
    _setCurrentAppModeBridge.set = setCurrentAppMode;

    const offUI = bindUIHandlers();
    const offMap = bindMapHandlers();
    const offPanels = bindPanelInteractions();

    // ── URL restore on mount ───────────────────────────────
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get("mode") === "explain") {
      const decoded = decodeExplainState(urlParams);
      // Apply decoded state to ExST
      ExST.t = decoded.t;
      ExST.v1 = decoded.v1;
      ExST.z1 = decoded.z1;
      ExST.node = decoded.node;
      ExST.compareMode = decoded.compareMode;
      ExST.pA_v2 = decoded.pA_v2;
      ExST.pA_z2 = decoded.pA_z2;
      ExST.pB_v2 = decoded.pB_v2;
      ExST.pB_z2 = decoded.pB_z2;
      ExST.z2Locked = decoded.z2Locked;
      ExST.method = decoded.method;
      ExST.drawerHeaderOnly = decoded.drawerHeaderOnly;
      // Mirror to React state so controlled components render correctly
      setExplainV1(decoded.v1);
      setExplainZ1(decoded.z1);
      setExplainNode(decoded.node);
      setExplainCompareMode(decoded.compareMode);
      setExplainMethod(decoded.method);
      setExplainT(decoded.t);
      if (decoded.pA_v2) { setExplainPAV2(decoded.pA_v2); setExplainPAZ2(decoded.pA_z2); }
      if (decoded.pB_v2) { setExplainPBV2(decoded.pB_v2); setExplainPBZ2(decoded.pB_z2); }
      setExplainZ2Locked(decoded.z2Locked);
      if (decoded.drawerHeaderOnly) setExplainDrawerHeaderOnly(true);
      // Restore z1 memory for the decoded variable
      if (!isSurfaceVar(decoded.v1) && decoded.z1 !== null) {
        ExST.z1MemoryByVar[decoded.v1] = decoded.z1;
      }
      if (decoded.hadInvalidParams) {
        showToast("Some URL parameters were invalid and have been reset to defaults.", true);
      }
      _setAppMode?.("explain");
    }

    leafMap.whenReady(() => {
      setTimeout(() => {
        void (async () => {
          try {
            await refreshGraphOptions(activeGraphId);
            await refreshExplainModelOptions(activeExplainModelId);
          } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            console.warn("Unable to load graph/model catalog:", message);
          }
          try {
            await boot();
          } catch (err) {
            // fetchGraphMeta now throws when the selected graph cannot be served. On startup
            // there is no previous selection to fall back to, so surface it and leave the map
            // empty — silently swallowing it is the blank-map failure this guard exists for.
            const message = err instanceof Error ? err.message : String(err);
            console.warn("Unable to load the initial graph:", message);
            showToast(`Unable to load graph: ${message}`);
          }
        })();
      }, 200);
    });

    return () => {
      _loadExplainModelRuntimeBridge.run = null;
      offUI();
      offMap();
      offPanels();
      mapMinZoomActive = false;
      mapSizeObserver.disconnect();
      if (deckInstance) {
        deckInstance.finalize();
        deckInstance = null;
      }
      if (leafMap) leafMap.remove();
      if (miniMap) miniMap.remove();
      if (deckCanvas && deckCanvas.parentNode) deckCanvas.parentNode.removeChild(deckCanvas);
    };
  }, []);

  return (
    <div id="app">
      {/* Full-screen map */}
      <div id="map-wrap">
        <div id="map" />
        <div id="minimap-container">
          <div id="map-type-switcher">
            <button className="mt-btn active" data-type="map" type="button" title="Street map">
              <span className="mt-preview mt-preview-map" />
              <span className="mt-label">Map</span>
            </button>
            <button className="mt-btn" data-type="satellite" type="button" title="Satellite imagery">
              <span className="mt-preview mt-preview-sat" />
              <span className="mt-label">Satellite</span>
            </button>
            <button className="mt-btn" data-type="terrain" type="button" title="Terrain map">
              <span className="mt-preview mt-preview-terrain" />
              <span className="mt-label">Terrain</span>
            </button>
          </div>
          <div id="minimap-clip">
            <div id="minimap-map" />
          </div>
          <div id="zoom-ctrl">
            <button className="zoom-btn" id="zoom-in-btn" type="button" title="Zoom in">+</button>
            <button className="zoom-btn" id="zoom-out-btn" type="button" title="Zoom out">−</button>
          </div>
        </div>

        <div id="tip">
          <div className="tip-title" id="tip-title" />
          <div id="tip-body" />
        </div>

        {/* Colorbar — position:absolute within #map-wrap; hidden outside explain mode via CSS */}
        <ExplainMap
          v1={explainV1}
          z1={explainZ1}
          t={explainT}
          grid={explainGrid}
          apiBaseUrl={API}
          modelId={selectedExplainModelId}
          availableTimestamps={explainAvailableTimestamps}
          refetchTrigger={explainRefetchTrigger}
          logScale={explainLogScale}
          colorbarMax={explainColorbarMax}
          displayMode={explainDisplayMode}
          normStats={explainNormStats}
          onRebuildLayers={rebuildExplainLayers}
          onColorbarUpdate={(errors, max, phase, min) => {
            setExplainColorbarErrors(errors);
            // Store the backend max (shared scale) always
            setExplainColorbarBackendMax(max);
            // In prediction mode, the [min, max] is the physical range — use as-is so
            // the colorbar matches what's painted on the map. In error mode keep the
            // existing per-variable vs shared logic.
            if (explainDisplayMode === "prediction") {
              setExplainColorbarMin(min ?? 0);
              setExplainColorbarMax(max);
            } else {
              setExplainColorbarMin(0);
              const displayMax = explainSharedColorbar ? max : (errors && errors.length > 0 ? maxOfArray(errors) : max);
              setExplainColorbarMax(displayMax);
            }
            setExplainColorbarPhase(phase);
            // Clear pending model-switch spinner once backend reports ready
            if (phase === "ready") setExplainSwitchPending(false);
          }}
        />
        {currentAppMode === "explain" && explainColorbarPhase === "skeleton" && (
          <div className="heatmap-loading-overlay">
            <div className="ex-spinner ex-spinner--sm" />
            <span className="heatmap-loading-label">Heatmap is loading…</span>
          </div>
        )}
      </div>

      {/* Explain mode timeline bar */}
      <div id="explain-timeline">
        {era5InitError ? (
          <div style={{ padding: "8px 12px", border: "1px solid var(--border)", color: "var(--text-muted)", fontSize: 11 }}>
            ERA5 data unavailable — restart the server.
          </div>
        ) : (
          <ExplainTimeline
            key={explainTimelineData ? `${explainTimelineData.range_start}|${explainTimelineData.range_end}|${explainTimelineData.timestamps.length}` : "loading"}
            timelineData={explainTimelineData}
            availableTimestamps={explainAvailableTimestamps}
            currentT={explainT}
            onTimestepChange={handleExplainTChange}
            monthBands={explainTimelineData?.month_bands ?? []}
            downloadedBandStarts={downloadedBandStarts}
            allDownloadedTimestamps={allDownloadedTimestamps}
            windowLoading={windowLoading}
            loadingBand={loadingBand}
            onBandClick={handleBandClick}
          />
        )}
      </div>

      {/* Top-center mode nav */}
      <div id="mode-nav">
        <button className="mode-tab active" id="mode-explore" type="button" title="Browse and filter graph structure">Explore</button>
        <div className="mode-nav-sep" />
        <button className="mode-tab" id="mode-inspect" type="button" title="Check degree and edge-length violations">Inspect</button>
        <div className="mode-nav-sep" />
        <button className="mode-tab" id="mode-explain" type="button" title="Analyse prediction errors and input attributions">Explain</button>
      </div>

      {/* Left floating panel */}
      <div
        id="panel-left"
        className={[
          (isSwitchingGraph || explainModelSwitching || explainSwitchPending || explainColorbarPhase === "skeleton" || explainAttrLoadingA || explainAttrLoadingB) ? "loading-dim" : "",
          leftPanelCollapsed ? "panel--collapsed" : "",
        ].filter(Boolean).join(" ")}
        onClick={explainDrawerOpen && !explainDrawerHeaderOnly ? handleExplainHide : undefined}
        title={explainDrawerOpen && !explainDrawerHeaderOnly ? "Click to collapse attribution drawer" : undefined}
      >
        <button
          className="panel-edge-tab"
          type="button"
          title={leftPanelCollapsed ? "Expand panel" : "Collapse panel"}
          onClick={(e) => { e.stopPropagation(); setLeftPanelCollapsed((v) => !v); }}
        >
          <span className={`chevron${leftPanelCollapsed ? " chevron--right" : " chevron--left"}`}>{"<"}</span>
        </button>
        <div className="panel-header" id="panel-left-header">
          <div className="panel-header-top">
            <div className="logo">
              <span className="logo-icon">
                <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">
                  <line x1="10" y1="2"  x2="18" y2="7"  stroke={EDGE_MESH} strokeWidth="1.1" strokeLinecap="round"/>
                  <line x1="18" y1="7"  x2="18" y2="13" stroke={EDGE_MESH} strokeWidth="1.1" strokeLinecap="round"/>
                  <line x1="18" y1="13" x2="10" y2="18" stroke={EDGE_MESH} strokeWidth="1.1" strokeLinecap="round"/>
                  <line x1="10" y1="18" x2="2"  y2="13" stroke={EDGE_MESH} strokeWidth="1.1" strokeLinecap="round"/>
                  <line x1="2"  y1="13" x2="2"  y2="7"  stroke={EDGE_MESH} strokeWidth="1.1" strokeLinecap="round"/>
                  <line x1="2"  y1="7"  x2="10" y2="2"  stroke={EDGE_MESH} strokeWidth="1.1" strokeLinecap="round"/>
                  <line x1="10" y1="10" x2="10" y2="2"  stroke={EDGE_DECODER} strokeWidth="1.1" strokeLinecap="round"/>
                  <line x1="10" y1="10" x2="18" y2="7"  stroke={EDGE_DECODER} strokeWidth="1.1" strokeLinecap="round"/>
                  <line x1="10" y1="10" x2="18" y2="13" stroke={EDGE_DECODER} strokeWidth="1.1" strokeLinecap="round"/>
                  <line x1="10" y1="10" x2="10" y2="18" stroke={EDGE_DECODER} strokeWidth="1.1" strokeLinecap="round"/>
                  <line x1="10" y1="10" x2="2"  y2="13" stroke={EDGE_DECODER} strokeWidth="1.1" strokeLinecap="round"/>
                  <line x1="10" y1="10" x2="2"  y2="7"  stroke={EDGE_DECODER} strokeWidth="1.1" strokeLinecap="round"/>
                  <circle cx="10" cy="2"  r="2" fill={NODE_MESH}/>
                  <circle cx="18" cy="7"  r="2" fill={NODE_MESH}/>
                  <circle cx="18" cy="13" r="2" fill={NODE_MESH}/>
                  <circle cx="10" cy="18" r="2" fill={NODE_MESH}/>
                  <circle cx="2"  cy="13" r="2" fill={NODE_MESH}/>
                  <circle cx="2"  cy="7"  r="2" fill={NODE_MESH}/>
                  <circle cx="10" cy="10" r="2.2" fill={NODE_GRID}/>
                </svg>
              </span>
              <span className="logo-text">
                <span className="logo-title"><span className="mesh-col">V.I.E.W.</span></span>
              </span>
              <span id="loading-badge" className={(isSwitchingGraph || explainModelSwitching || explainSwitchPending || explainColorbarPhase === "skeleton" || explainAttrLoadingA || explainAttrLoadingB) ? "active" : ""} />
            </div>
            <div className="theme-switcher" id="theme-switcher">
              <span id="theme-buttons" style={{ display: "contents" }}>
                <button className="theme-btn" data-theme="light" type="button" title="Light mode">☀</button>
                <button className="theme-btn active" data-theme="auto" type="button" title="Follow system">◑</button>
                <button className="theme-btn" data-theme="dark" type="button" title="Dark mode">☾</button>
              </span>
              <button className="theme-btn help-btn" id="explain-help-btn" type="button" title="Explain mode guide" style={{ display: "none" }} onClick={openExplainTour}>?</button>
              <button className="theme-btn help-btn" id="inspect-help-btn" type="button" title="Inspect mode guide" style={{ display: "none" }} onClick={openInspectTour}>?</button>
              <button className="theme-btn help-btn" id="help-btn" type="button" title="Explore mode guide" onClick={openExploreTour}>?</button>
            </div>
          </div>
          <div className="logo-sub"><strong>V</strong>isual <strong>I</strong>nspection and <strong>E</strong>xplanation of graph <strong>W</strong>eather models</div>
        </div>

        <div className="sb-sec" id="upload-sec">
          {UPLOADS_ENABLED ? (
            <>
              <div className="h-upload-row">
                <button className="upload-btn" type="button" title="Upload a graph file (.npz)" onClick={openUploadPicker} disabled={isUploading || isSwitchingGraph}>
                  {isUploading ? "Uploading…" : "Upload GNN"}
                </button>
                <input
                  ref={uploadInputRef}
                  className="upload-input"
                  type="file"
                  multiple
                  onChange={handleUploadChange}
                  aria-label="Upload a graph folder"
                />
                <select
                  className="graph-select"
                  value={graphOptions.length === 0 ? "" : selectedGraphId}
                  onChange={(event) => { void handleGraphSelectChange(event); }}
                  disabled={isUploading || isSwitchingGraph || graphOptions.length === 0}
                  aria-label="Select graph"
                >
                  {graphOptions.length === 0 ? (
                    <option value="">No graphs</option>
                  ) : (
                    graphOptions.map((graph) => {
                      const isPreset = graph.folder_path.startsWith("/usr/data/graphs/");
                      const label = isPreset ? graph.name : `${graph.name} | ${graph.graph_id} | ${formatGraphLastModified(graph)}`;
                      return (
                        <option key={graph.graph_id} value={graph.graph_id}>{label}</option>
                      );
                    })
                  )}
                </select>
              </div>
              <div className="graph-name-row" id="graph-name-display" />
              <div className="h-meta">
                <span><strong id="hdr-node-count">–</strong> nodes &nbsp;·&nbsp; <strong id="hdr-edge-count">–</strong> edges</span>
              </div>
            </>
          ) : (
            <>
              <div className="h-upload-row">
                <button className="upload-btn" type="button" title="Graph uploads disabled" disabled style={{ cursor: "not-allowed" }}>
                  Upload GNN
                </button>
                <select
                  className="graph-select"
                  value={graphOptions.length === 0 ? "" : selectedGraphId}
                  onChange={(event) => { void handleGraphSelectChange(event); }}
                  disabled={isSwitchingGraph || graphOptions.length === 0}
                  aria-label="Select graph"
                >
                  {graphOptions.length === 0 ? (
                    <option value="">No graphs</option>
                  ) : (
                    graphOptions.map((graph) => {
                      const isPreset = graph.folder_path.startsWith("/usr/data/graphs/");
                      const label = isPreset ? graph.name : `${graph.name} | ${graph.graph_id} | ${formatGraphLastModified(graph)}`;
                      return (
                        <option key={graph.graph_id} value={graph.graph_id}>{label}</option>
                      );
                    })
                  )}
                </select>
              </div>
              <div className="graph-name-row" id="graph-name-display" />
              <div className="h-meta">
                <span><strong id="hdr-node-count">–</strong> nodes &nbsp;·&nbsp; <strong id="hdr-edge-count">–</strong> edges</span>
              </div>
            </>
          )}
        </div>

        <div className="sidebar-main" id="panel-legend-sec">
          <StackedSidebar
            L={metaLevels}
            mode={currentAppMode === "inspect" ? "radio" : "check"}
            nodeCounts={metaNodeCounts}
            value={currentAppMode === "inspect" ? activeInspectSubgraph : undefined}
            onStateChange={handlePanelStateChange}
          />
        </div>

        {/* Explain mode left panel — Model section only */}
        <div id="explain-panel-content">
          <ExplainLeftPanel
            modelSwitching={explainModelSwitching}
            models={explainModelOptions}
            selectedModelId={selectedExplainModelId}
            onModelSelect={handleExplainModelSelect}
            onModelUpload={openModelUploadModal}
          />
        </div>

        <div className="resize-handle" id="panel-left-resize" />
      </div>

      {/* Explain mode right panel — Colorbar + Prediction Target + Node readout + Attribution */}
      <div
        id="explain-panel-right"
        className={explainRightPanelCollapsed ? "panel--collapsed" : undefined}
        onClick={!explainRightPanelCollapsed && explainDrawerOpen && !explainDrawerHeaderOnly ? handleExplainHide : undefined}
        title={!explainRightPanelCollapsed && explainDrawerOpen && !explainDrawerHeaderOnly ? "Click to collapse attribution drawer" : undefined}
      >
        <ExplainRightPanel
          peakMarks={explainPeakMarks}
          v1={explainV1}
          z1={explainZ1}
          node={explainNode}
          drawerOpen={explainDrawerOpen}
          errors={explainColorbarErrors}
          colorbarMax={explainColorbarMax}
          colorbarMin={explainColorbarMin}
          colorbarPhase={explainColorbarPhase}
          grid={ExST.grid}
          displayMode={explainDisplayMode}
          onDisplayModeChange={handleExplainDisplayModeChange}
          onV1Change={handleExplainV1Change}
          onZ1Change={handleExplainZ1Change}
          onNodeClear={handleExplainNodeClear}
          onOpenAttribution={handleOpenAttribution}
          onLogScaleChange={(v) => setExplainLogScale(v)}
          sharedColorbar={explainSharedColorbar}
          onSharedColorbarChange={(v) => setExplainSharedColorbar(v)}
          onOpenTour={openExplainTour}
          collapsed={explainRightPanelCollapsed}
          onToggleCollapse={() => setExplainRightPanelCollapsed((v) => !v)}
          oceanVars={explainVariablesMeta?.is_ocean ? explainVariablesMeta.state_vars ?? null : null}
        />
        <div className="resize-handle-left" id="explain-panel-right-resize" />
      </div>

      {/* Right floating panel */}
      <div id="panel-right" className={rightPanelCollapsed ? "panel--collapsed" : undefined}>
        <button
          className="panel-edge-tab"
          id="panel-right-header"
          type="button"
          title={rightPanelCollapsed ? "Expand panel" : "Collapse panel"}
          onClick={() => setRightPanelCollapsed((v) => !v)}
        >
          <span className={`chevron${rightPanelCollapsed ? " chevron--left" : " chevron--right"}`}>{"<"}</span>
        </button>
        <div id="panel-right-inner">
          <div id="pnl-graph">
            <div className="rs-sec">
              <div className="rs-title" id="info-title">Element Info</div>
              <div className="rs-sub" id="info-sub">Click a node or edge to inspect</div>
            </div>
            <div id="info-details" className="rs-sec" style={{ display: "none" }} />
            <div className="rs-sec" id="graph-stats-sec">
              <div className="sb-hd">Viewport Statistics</div>
              <div className="stats-grid" id="stats-grid" />
            </div>
            <div className="rs-sec">
              <div className="toggle-row" id="toggle-scale-fade">
                <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
                  Scale fade
                  <HintButton placement="right" content={HINT_SCALE_FADE} />
                </span>
                <span style={{ fontSize: "11px", color: "var(--text-dim)", marginRight: "auto" }}>short edges fade at zoom</span>
                <div className="pill on" id="pe-scale-fade" />
              </div>
            </div>
          </div>

          <div id="pnl-viol" style={{ display: "none" }}>
            <div className="rs-sec">
              <div className="rs-title">Graph Inspection</div>
              <div className="rs-sub" style={{ marginTop: 4 }}>
                <span style={{ display: "block" }}>· Adjust thresholds below to highlight nodes and edges outside the valid range</span>
              </div>
            </div>

            <hr className="inspect-sec-divider" />

            {/* SOURCE + TARGET NODE DEGREE */}
            <div id="inspect-deg-wrap">
            <div className="inspect-plot-sec" id="inspect-deg-section">
              <div className="inspect-plot-header" onClick={(e) => {
                const body = (e.currentTarget as HTMLElement).nextElementSibling as HTMLElement | null;
                const chev = (e.currentTarget as HTMLElement).querySelector(".inspect-plot-chevron") as HTMLElement | null;
                if (!body) return;
                const open = body.style.display !== "none";
                body.style.display = open ? "none" : "";
                if (chev) chev.style.transform = open ? "rotate(0deg)" : "rotate(90deg)";
              }}>
                <span className="inspect-plot-title" id="src-deg-section-title">SOURCE NODE DEGREE</span>
                <span onClick={(e) => e.stopPropagation()}>
                  <HintButton placement="left" content={HINT_SRC_DEGREE} />
                </span>
                <span className="inspect-plot-chevron" style={{ transform: "rotate(90deg)" }}>›</span>
              </div>
              <div className="inspect-plot-body">
                <em className="inspect-sym-note" id="deg-sym-note" style={{ display: "none" }}>Source and target nodes have the same degree distribution for intra-level edges.</em>
                <div style={{ display: "flex", justifyContent: "flex-end", gap: "5px", marginBottom: "3px" }}>
                  <button className="log-toggle" id="src-deg-logx-btn" type="button" title="Toggle logarithmic x axis">log x</button>
                  <button className="log-toggle" id="src-deg-logy-btn" type="button" title="Toggle logarithmic y axis">log y</button>
                </div>
                <div id="deg-hist-src-wrap">
                  <div className="histogram-container" id="viol-degree-hist-src" />
                  <div className="thresh-wrap">
                    <div className="slider-axis-wrap">
                      <div className="dual-slider">
                        <div className="slider-track" id="src-deg-track" />
                        <span className="thumb-lbl" id="src-deg-lbl-min" style={{ left: "3%" }}>1</span>
                        <span className="thumb-lbl" id="src-deg-lbl-max" style={{ left: "67%" }}>20</span>
                        <input type="range" min={0} max={30} step={1} defaultValue={1} id="src-deg-min-sl" />
                        <input type="range" min={0} max={30} step={1} defaultValue={20} id="src-deg-max-sl" />
                      </div>
                    </div>
                    <div className="thresh-labels">
                      <span id="src-deg-min-lbl-bot">Min: 1</span>
                      <span id="src-deg-max-lbl-bot">Max: 20</span>
                    </div>
                  </div>
                  <div className="viol-lines" id="src-deg-violation-count" />
                </div>
              </div>
            </div>

            <hr className="inspect-sec-divider" id="divider-tgt-deg" />

            {/* TARGET NODE DEGREE */}
            <div className="inspect-plot-sec" id="tgt-deg-section">
              <div className="inspect-plot-header" onClick={(e) => {
                const body = (e.currentTarget as HTMLElement).nextElementSibling as HTMLElement | null;
                const chev = (e.currentTarget as HTMLElement).querySelector(".inspect-plot-chevron") as HTMLElement | null;
                if (!body) return;
                const open = body.style.display !== "none";
                body.style.display = open ? "none" : "";
                if (chev) chev.style.transform = open ? "rotate(0deg)" : "rotate(90deg)";
              }}>
                <span className="inspect-plot-title" id="tgt-deg-section-title">TARGET NODE DEGREE</span>
                <span onClick={(e) => e.stopPropagation()}>
                  <HintButton placement="left" content={HINT_TGT_DEGREE} />
                </span>
                <span className="inspect-plot-chevron" style={{ transform: "rotate(90deg)" }}>›</span>
              </div>
              <div className="inspect-plot-body">
                <div style={{ display: "flex", justifyContent: "flex-end", gap: "5px", marginBottom: "3px" }}>
                  <button className="log-toggle" id="tgt-deg-logx-btn" type="button" title="Toggle logarithmic x axis">log x</button>
                  <button className="log-toggle" id="tgt-deg-logy-btn" type="button" title="Toggle logarithmic y axis">log y</button>
                </div>
                <div id="deg-hist-tgt-wrap">
                  <div className="histogram-container" id="viol-degree-hist-tgt" />
                  <div className="thresh-wrap">
                    <div className="slider-axis-wrap">
                      <div className="dual-slider">
                        <div className="slider-track" id="tgt-deg-track" />
                        <span className="thumb-lbl" id="tgt-deg-lbl-min" style={{ left: "3%" }}>1</span>
                        <span className="thumb-lbl" id="tgt-deg-lbl-max" style={{ left: "67%" }}>20</span>
                        <input type="range" min={0} max={30} step={1} defaultValue={1} id="tgt-deg-min-sl" />
                        <input type="range" min={0} max={30} step={1} defaultValue={20} id="tgt-deg-max-sl" />
                      </div>
                    </div>
                    <div className="thresh-labels">
                      <span id="tgt-deg-min-lbl-bot">Min: 1</span>
                      <span id="tgt-deg-max-lbl-bot">Max: 20</span>
                    </div>
                  </div>
                  <div className="viol-lines" id="tgt-deg-violation-count" />
                </div>
              </div>
            </div>
            </div>{/* end inspect-deg-wrap */}

            <hr className="inspect-sec-divider" />

            {/* EDGE LENGTH */}
            <div className="inspect-plot-sec" id="edge-len-section">
              <div className="inspect-plot-header" onClick={(e) => {
                const body = (e.currentTarget as HTMLElement).nextElementSibling as HTMLElement | null;
                const chev = (e.currentTarget as HTMLElement).querySelector(".inspect-plot-chevron") as HTMLElement | null;
                if (!body) return;
                const open = body.style.display !== "none";
                body.style.display = open ? "none" : "";
                if (chev) chev.style.transform = open ? "rotate(0deg)" : "rotate(90deg)";
              }}>
                <span className="inspect-plot-title">EDGE LENGTH (KM)</span>
                <span onClick={(e) => e.stopPropagation()}>
                  <HintButton placement="left" content={HINT_EDGE_LENGTH} />
                </span>
                <span className="inspect-plot-chevron" style={{ transform: "rotate(90deg)" }}>›</span>
              </div>
              <div className="inspect-plot-body">
                <div style={{ display: "flex", justifyContent: "flex-end", alignItems: "center", gap: "6px", marginBottom: "3px" }}>
                  <button className="log-toggle" id="len-logx-btn" type="button" title="Toggle logarithmic x axis">log x</button>
                  <button className="log-toggle" id="len-logy-btn" type="button" title="Toggle logarithmic y axis">log y</button>
                  {/* Bins selector — only meaningful for the continuous histogram. Always
                      mounted and shown/hidden by buildViolationHistograms via style.display,
                      which already knows the measured shape. Gating it on React state instead
                      would mean calling setState from inside the d3 render path. */}
                  <div className="hist-bins-wrap">
                    bins <input className="hist-bins-input" id="len-bins" type="number" min={2} max={60} defaultValue={15} />
                  </div>
                </div>
                <div className="histogram-container" id="viol-edge-hist" />
                <div className="thresh-wrap">
                  <div className="slider-axis-wrap">
                    <div className="dual-slider">
                      <div className="slider-track" id="len-track" />
                      <span className="thumb-lbl" id="len-lbl-min" style={{ left: "3%" }}>20</span>
                      <span className="thumb-lbl" id="len-lbl-max" style={{ left: "67%" }}>400</span>
                      <input type="range" min={0} max={600} step={5} defaultValue={20} id="len-min-sl" />
                      <input type="range" min={0} max={600} step={5} defaultValue={400} id="len-max-sl" />
                    </div>
                  </div>
                  <div className="thresh-labels">
                    <span id="len-min-lbl-bot">Min: 20 km</span>
                    <span id="len-max-lbl-bot">Max: 400 km</span>
                  </div>
                </div>
                <div className="viol-lines" id="len-violation-count" />
              </div>
            </div>
          </div>
        </div>
        <div className="resize-handle-left" id="panel-right-resize" />
      </div>

      {/* Attribution drawer — always in DOM, slides up/down via CSS transform */}
      <ExplainDrawer
        open={explainDrawerOpen}
        headerOnly={explainDrawerHeaderOnly}
        // Measured from the loaded window rather than assumed: ERA5 steps 6 hourly, the ocean
        // daily, and the drawer labels its two input steps from this.
        stepHours={explainStepHours}
        oceanVars={explainVariablesMeta?.is_ocean ? explainVariablesMeta.state_vars ?? null : null}
        node={explainNode}
        v1={explainV1}
        z1={explainZ1}
        t={explainT}
        grid={explainGrid}
        attributionData={explainAttributionData}
        attributionDataB={explainAttributionDataB}
        loadingA={explainAttrLoadingA}
        loadingB={explainAttrLoadingB}
        errorA={explainAttrErrorA}
        errorB={explainAttrErrorB}
        compareMode={explainCompareMode}
        method={explainMethod}
        pA_v2={explainPAV2}
        pA_z2={explainPAZ2}
        pB_v2={explainPBV2}
        pB_z2={explainPBZ2}
        z2Locked={explainZ2Locked}
        coastlineGeoJson={explainCoastlineGeoJson}
        onHide={handleExplainHide}
        onExpand={handleExplainExpand}
        onCompareModeChange={handleExplainCompareModeChange}
        onMethodChange={handleExplainMethodChange}
        onPAV2Change={handleExplainPAV2Change}
        onPAZ2Change={handleExplainPAZ2Change}
        onPBV2Change={handleExplainPBV2Change}
        onPBZ2Change={handleExplainPBZ2Change}
        onZ2LockChange={handleExplainZ2LockChange}
        onComputeTarget={handleExplainComputeTarget}
        savedConfigs={explainSavedConfigs}
        onSaveConfig={handleExplainSaveConfig}
        onRestoreConfig={handleExplainRestoreConfig}
        onTransitionEnd={handleDrawerTransitionEnd}
        onDrawerResize={() => leafMap.invalidateSize()}
        openRankingSidebar={showExplainTour && EXPLAIN_STEPS[explainTourStep]?.targetId === "ex-ranking-wrap"}
        closeSidebarForTour={showExplainTour && EXPLAIN_STEPS[explainTourStep]?.targetId === "explain-help-btn"}
      />

      {showIntro && (
        <>
          {spotlightRect
            ? <div className="tour-dismiss" onClick={dismissIntro}
                style={ONBOARD_STEPS[introStep].mapInteractive ? { pointerEvents: "none" } : undefined} />
            : <div className="intro-backdrop" onClick={dismissIntro} />}
          {spotlightRect && (
            <div className="tour-spotlight" style={{ top: spotlightRect.top - 8, left: spotlightRect.left - 8, width: spotlightRect.width + 16, height: spotlightRect.height + 16 }} />
          )}
          <div className="intro-modal-wrap" onClick={dismissIntro}>
            <div className="intro-modal" onClick={(e) => e.stopPropagation()}>
              <div className="intro-step-dots">
                {ONBOARD_STEPS.map((_, i) => (
                  <button key={i} type="button" className={`intro-dot${i === introStep ? " active" : ""}`} onClick={() => setIntroStep(i)} aria-label={`Step ${i + 1}`} />
                ))}
              </div>
              <div className="intro-icon">{ONBOARD_STEPS[introStep].icon}</div>
              <div className="intro-title">{ONBOARD_STEPS[introStep].title}</div>
              <div className="intro-body" style={{ whiteSpace: "pre-line" }}>{ONBOARD_STEPS[introStep].body}</div>
              <div className="intro-actions">
                <button className="intro-btn intro-btn-skip" type="button" onClick={dismissIntro}>Close guide</button>
                <div style={{ display: "flex", gap: "8px" }}>
                  {introStep > 0 && <button className="intro-btn intro-btn-prev" type="button" onClick={() => setIntroStep((s) => s - 1)}>Back</button>}
                  {introStep < ONBOARD_STEPS.length - 1
                    ? <button className="intro-btn intro-btn-next" type="button" onClick={() => setIntroStep((s) => s + 1)}>Next</button>
                    : <button className="intro-btn intro-btn-done" type="button" onClick={() => { dismissIntro(); openExploreTour(); }}>Start Explore tour</button>}
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {showExplainTour && (
        <>
          {explainSpotlightRect
            ? <div className="tour-dismiss" onClick={dismissExplainTour}
                style={EXPLAIN_STEPS[explainTourStep].mapInteractive ? { pointerEvents: "none" } : undefined} />
            : <div className="intro-backdrop" onClick={dismissExplainTour} />}
          {explainSpotlightRect && (
            <div className="tour-spotlight" style={{ top: explainSpotlightRect.top - 8, left: explainSpotlightRect.left - 8, width: explainSpotlightRect.width + 16, height: explainSpotlightRect.height + 16 }} />
          )}
          <div className="intro-modal-wrap" onClick={dismissExplainTour}>
            <div className="intro-modal" onClick={(e) => e.stopPropagation()}>
              <div className="intro-step-dots">
                {EXPLAIN_STEPS.map((_, i) => (
                  <button key={i} type="button" className={`intro-dot${i === explainTourStep ? " active" : ""}`} onClick={() => setExplainTourStep(i)} aria-label={`Step ${i + 1}`} />
                ))}
              </div>
              <div className="intro-icon">{EXPLAIN_STEPS[explainTourStep].icon}</div>
              <div className="intro-title">{EXPLAIN_STEPS[explainTourStep].title}</div>
              <div className="intro-body" style={{ whiteSpace: "pre-line" }}>{EXPLAIN_STEPS[explainTourStep].body}</div>
              <div className="intro-actions">
                <button className="intro-btn intro-btn-skip" type="button" onClick={dismissExplainTour}>Close guide</button>
                <div style={{ display: "flex", gap: "8px" }}>
                  {explainTourStep > 0 && <button className="intro-btn intro-btn-prev" type="button" onClick={() => setExplainTourStep((s) => s - 1)}>Back</button>}
                  {explainTourStep < EXPLAIN_STEPS.length - 1
                    ? <button className="intro-btn intro-btn-next" type="button" onClick={() => setExplainTourStep((s) => s + 1)}>Next</button>
                    : <button className="intro-btn intro-btn-done" type="button" onClick={dismissExplainTour}>Start exploring</button>}
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {showInspectTour && (
        <>
          {inspectSpotlightRect
            ? <div className="tour-dismiss" onClick={dismissInspectTour} />
            : <div className="intro-backdrop" onClick={dismissInspectTour} />}
          {inspectSpotlightRect && (
            <div className="tour-spotlight" style={{ top: inspectSpotlightRect.top - 8, left: inspectSpotlightRect.left - 8, width: inspectSpotlightRect.width + 16, height: inspectSpotlightRect.height + 16 }} />
          )}
          <div className="intro-modal-wrap" onClick={dismissInspectTour}>
            <div className="intro-modal" onClick={(e) => e.stopPropagation()}>
              <div className="intro-step-dots">
                {INSPECT_STEPS.map((_, i) => (
                  <button key={i} type="button" className={`intro-dot${i === inspectTourStep ? " active" : ""}`} onClick={() => setInspectTourStep(i)} aria-label={`Step ${i + 1}`} />
                ))}
              </div>
              <div className="intro-icon">{INSPECT_STEPS[inspectTourStep].icon}</div>
              <div className="intro-title">{INSPECT_STEPS[inspectTourStep].title}</div>
              <div className="intro-body" style={{ whiteSpace: "pre-line" }}>{INSPECT_STEPS[inspectTourStep].body}</div>
              <div className="intro-actions">
                <button className="intro-btn intro-btn-skip" type="button" onClick={dismissInspectTour}>Close guide</button>
                <div style={{ display: "flex", gap: "8px" }}>
                  {inspectTourStep > 0 && <button className="intro-btn intro-btn-prev" type="button" onClick={() => setInspectTourStep((s) => s - 1)}>Back</button>}
                  {inspectTourStep < INSPECT_STEPS.length - 1
                    ? <button className="intro-btn intro-btn-next" type="button" onClick={() => setInspectTourStep((s) => s + 1)}>Next</button>
                    : <button className="intro-btn intro-btn-done" type="button" onClick={dismissInspectTour}>Done</button>}
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {showExploreTour && (
        <>
          {exploreSpotlightRect
            ? <div className="tour-dismiss" onClick={dismissExploreTour}
                style={EXPLORE_STEPS[exploreTourStep].mapInteractive ? { pointerEvents: "none" } : undefined} />
            : <div className="intro-backdrop" onClick={dismissExploreTour} />}
          {exploreSpotlightRect && (
            <div className="tour-spotlight" style={{ top: exploreSpotlightRect.top - 8, left: exploreSpotlightRect.left - 8, width: exploreSpotlightRect.width + 16, height: exploreSpotlightRect.height + 16 }} />
          )}
          <div className="intro-modal-wrap" onClick={dismissExploreTour}>
            <div className="intro-modal" onClick={(e) => e.stopPropagation()}>
              <div className="intro-step-dots">
                {EXPLORE_STEPS.map((_, i) => (
                  <button key={i} type="button" className={`intro-dot${i === exploreTourStep ? " active" : ""}`} onClick={() => setExploreTourStep(i)} aria-label={`Step ${i + 1}`} />
                ))}
              </div>
              <div className="intro-icon">{EXPLORE_STEPS[exploreTourStep].icon}</div>
              <div className="intro-title">{EXPLORE_STEPS[exploreTourStep].title}</div>
              <div className="intro-body" style={{ whiteSpace: "pre-line" }}>{EXPLORE_STEPS[exploreTourStep].body}</div>
              <div className="intro-actions">
                <button className="intro-btn intro-btn-skip" type="button" onClick={dismissExploreTour}>Close guide</button>
                <div style={{ display: "flex", gap: "8px" }}>
                  {exploreTourStep > 0 && <button className="intro-btn intro-btn-prev" type="button" onClick={() => setExploreTourStep((s) => s - 1)}>Back</button>}
                  {exploreTourStep < EXPLORE_STEPS.length - 1
                    ? <button className="intro-btn intro-btn-next" type="button" onClick={() => setExploreTourStep((s) => s + 1)}>Next</button>
                    : <button className="intro-btn intro-btn-done" type="button" onClick={dismissExploreTour}>Start exploring</button>}
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {modelUploadOpen && (
        <div className="intro-modal-wrap" onClick={closeModelUploadModal}>
          <div className="intro-modal model-upload-modal" onClick={(e) => e.stopPropagation()}>
            <div className="intro-title">Upload Model</div>
            <div className="model-upload-field">
              <label className="model-upload-label" htmlFor="model-upload-type">Type</label>
              <select
                id="model-upload-type"
                className="graph-select"
                value={modelUploadDomain}
                onChange={(e) => {
                  setModelUploadDomain(e.target.value as "era5" | "ocean");
                  setModelUploadError(null);
                }}
              >
                <option value="era5">Weather (ERA5)</option>
                <option value="ocean">Ocean (OceanBench)</option>
              </select>
            </div>
            {modelUploadDomain === "ocean" && (
              <p className="ex-total-note">Ocean checkpoints are TorchScript-only (.pt) — SeaCast has no Lightning checkpoint export.</p>
            )}
            <div className="model-upload-field">
              <label className="model-upload-label" htmlFor="model-upload-name">Name</label>
              <input
                id="model-upload-name"
                className="hist-bins-input model-upload-name-input"
                type="text"
                value={modelUploadName}
                onChange={(e) => setModelUploadName(e.target.value)}
                placeholder="Display name"
              />
            </div>
            <div className="model-upload-field">
              <button
                type="button"
                className="upload-btn"
                onClick={() => modelUploadInputRef.current?.click()}
              >
                {modelUploadFile ? modelUploadFile.name : "Choose file…"}
              </button>
              <input
                ref={modelUploadInputRef}
                className="upload-input"
                type="file"
                accept={MODEL_UPLOAD_ACCEPT[modelUploadDomain]}
                onChange={handleModelUploadFileChange}
                aria-label="Select a model checkpoint file"
              />
            </div>
            {modelUploadError && <p className="ex-total-note" style={{ color: "var(--viol-hi)" }}>{modelUploadError}</p>}
            <div className="intro-actions">
              <button className="intro-btn intro-btn-skip" type="button" onClick={closeModelUploadModal} disabled={modelUploadSubmitting}>
                Cancel
              </button>
              <button
                className="intro-btn intro-btn-done"
                type="button"
                onClick={() => { void handleModelUploadSubmit(); }}
                disabled={modelUploadSubmitting || !modelUploadFile}
              >
                {modelUploadSubmitting ? "Uploading…" : "Upload"}
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
