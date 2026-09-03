"use client";

import React, { useMemo, useState } from "react";
import HintButton from "./HintButton";
import { HINT_COLORBAR, HINT_COLORBAR_PREDICTION, HINT_SHARE_COLORBAR, HINT_TOTAL_ERROR } from "../../lib/hints";
import { palette } from "../../lib/colors";

// ── Types ────────────────────────────────────────────────────────────────────

type GridData = {
  lat: number[];
  lon: number[];
};

// ── Constants ────────────────────────────────────────────────────────────────

const ATM_VARS = [
  "geopotential",
  "specific_humidity",
  "temperature",
  "u_component_of_wind",
  "v_component_of_wind",
  "vertical_velocity",
] as const;

const SURF_VARS = [
  "2m_temperature",
  "10m_u_component_of_wind",
  "10m_v_component_of_wind",
  "mean_sea_level_pressure",
  "total_precipitation_6hr",
] as const;

const PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000] as const;

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

// Ocean state channels are named "<var>_<depth>m" (e.g. "thetao_47m"); zos/siconc/sithick
// carry no depth suffix. Prettify the depth suffix, leave everything else as-is.
const OCEAN_VAR_LABELS: Record<string, string> = {
  zos: "Sea Surface Height",
  siconc: "Sea Ice Concentration",
  sithick: "Sea Ice Thickness",
  thetao: "Temperature",
  so: "Salinity",
  uo: "U Velocity",
  vo: "V Velocity",
};

const varLabel = (v: string): string => {
  if (VAR_LABELS[v]) return VAR_LABELS[v];
  const m = /^([a-z]+)_(\d+)m$/.exec(v);
  if (m && OCEAN_VAR_LABELS[m[1]]) return `${OCEAN_VAR_LABELS[m[1]]} (${m[2]}m)`;
  if (OCEAN_VAR_LABELS[v]) return OCEAN_VAR_LABELS[v];
  return v;
};

// ── Per-variable colormap helpers ────────────────────────────────────────────

const DIVERGING_PRED_VARS = new Set([
  "u_component_of_wind",
  "v_component_of_wind",
  "vertical_velocity",
  "10m_u_component_of_wind",
  "10m_v_component_of_wind",
]);

// Ocean anomaly/velocity channels — diverging around zero, same as wind components.
function isOceanDivergingVar(v1: string): boolean {
  return v1 === "zos" || v1.startsWith("uo_") || v1.startsWith("vo_");
}

function isDivergingPred(v1: string): boolean {
  return DIVERGING_PRED_VARS.has(v1) || isOceanDivergingVar(v1);
}

type GradStop = { offset: string; stopColor: string };

function getPredGradientStops(v1: string): GradStop[] {
  if (isDivergingPred(v1)) {
    return [
      { offset: "0%",   stopColor: palette.divergingWind.stop0 },
      { offset: "25%",  stopColor: palette.divergingWind.stop1 },
      { offset: "50%",  stopColor: palette.divergingWind.stop2 },
      { offset: "75%",  stopColor: palette.divergingWind.stop3 },
      { offset: "100%", stopColor: palette.divergingWind.stop4 },
    ];
  }
  if (v1 === "temperature" || v1 === "2m_temperature" || v1.startsWith("thetao_")) {
    return [
      { offset:   "0%", stopColor: palette.thermalField.stop0  },
      { offset:   "9%", stopColor: palette.thermalField.stop1  },
      { offset:  "18%", stopColor: palette.thermalField.stop2  },
      { offset:  "27%", stopColor: palette.thermalField.stop3  },
      { offset:  "36%", stopColor: palette.thermalField.stop4  },
      { offset:  "45%", stopColor: palette.thermalField.stop5  },
      { offset:  "55%", stopColor: palette.thermalField.stop6  },
      { offset:  "64%", stopColor: palette.thermalField.stop7  },
      { offset:  "73%", stopColor: palette.thermalField.stop8  },
      { offset:  "82%", stopColor: palette.thermalField.stop9  },
      { offset:  "91%", stopColor: palette.thermalField.stop10 },
      { offset: "100%", stopColor: palette.thermalField.stop11 },
    ];
  }
  if (v1 === "specific_humidity" || v1.startsWith("so_")) {
    return [
      { offset: "0%",   stopColor: palette.halineField.stop0 },
      { offset: "25%",  stopColor: palette.halineField.stop1 },
      { offset: "50%",  stopColor: palette.halineField.stop2 },
      { offset: "75%",  stopColor: palette.halineField.stop3 },
      { offset: "100%", stopColor: palette.halineField.stop4 },
    ];
  }
  if (v1 === "total_precipitation_6hr") {
    return [
      { offset: "0%",   stopColor: palette.rainField.stop0 },
      { offset: "25%",  stopColor: palette.rainField.stop1 },
      { offset: "50%",  stopColor: palette.rainField.stop2 },
      { offset: "75%",  stopColor: palette.rainField.stop3 },
      { offset: "100%", stopColor: palette.rainField.stop4 },
    ];
  }
  if (v1 === "siconc" || v1 === "sithick") {
    return [
      { offset: "0%",   stopColor: palette.iceField.stop0 },
      { offset: "25%",  stopColor: palette.iceField.stop1 },
      { offset: "50%",  stopColor: palette.iceField.stop2 },
      { offset: "75%",  stopColor: palette.iceField.stop3 },
      { offset: "100%", stopColor: palette.iceField.stop4 },
    ];
  }
  // geopotential, MSLP, and fallback
  return [
    { offset: "0%",   stopColor: palette.denseField.stop0 },
    { offset: "25%",  stopColor: palette.denseField.stop1 },
    { offset: "50%",  stopColor: palette.denseField.stop2 },
    { offset: "75%",  stopColor: palette.denseField.stop3 },
    { offset: "100%", stopColor: palette.denseField.stop4 },
  ];
}

// ── Colorbar helpers ─────────────────────────────────────────────────────────

function colorbarModeLabel(v1: string, z1: string | null, sharedColorbar: boolean, isOcean = false): string {
  const vLabel = varLabel(v1);

  if (isOcean) {
    if (v1 === "total") return "Colorbar: RMSE across all ocean state channels";
    return sharedColorbar
      ? "Colorbar shared across all ocean state channels"
      : `Colorbar for ${vLabel}`;
  }

  if (sharedColorbar) {
    // SHARED MODE
    if (v1 === "total" && z1 === "sum") return "Colorbar: RMSE across all variables & pressure levels";
    if (v1 === "total") return `Colorbar shared across all pressure levels (with variables summed)`;
    if (z1 === "sum") return `Colorbar shared across all variables (with pressure levels summed)`;
    return "Colorbar shared across all variables & pressure levels";
  } else {
    // PER-VARIABLE MODE
    if (v1 === "total" && z1 === "sum") return "Colorbar: RMSE across all variables & pressure levels";
    if (v1 === "total") return `Colorbar for variables summed at ${z1 ?? "?"} hPa`;
    if (z1 === "sum") return `Colorbar for ${vLabel} (pressure levels summed)`;
    return `Colorbar for ${vLabel} at ${z1 ?? "?"} hPa`;
  }
}

// Returns normalised [0,1] histogram heights, optionally log-scaled on x and y.
// When `min` is provided (prediction mode), values are binned across [min, max] in
// physical units. Otherwise (error mode) values are binned across [0, max].
function computeHistogram(
  // null = no data at that node for the selected channel. Skipped entirely: coercing it to 0
  // would pile every masked node into the lowest bin and invent a spike that isn't there.
  values: (number | null)[],
  max: number,
  nBins: number,
  logY: boolean,
  logX: boolean = false,
  min: number = 0,
): number[] {
  const span = max - min;
  if (span <= 0 || nBins < 1) return new Array(nBins).fill(0);
  const counts = new Array(nBins).fill(0);

  for (const v of values) {
    if (v === null || !Number.isFinite(v)) continue;
    let bin: number;
    if (logX) {
      // Logarithmic binning only meaningful on positive ranges (errors). For predictions
      // the caller should not enable logX since values can be negative; we still guard.
      const vLog = Math.log1p(Math.max(0, v - min));
      const spanLog = Math.log1p(span);
      bin = Math.min(nBins - 1, Math.floor((vLog / spanLog) * nBins));
    } else {
      bin = Math.min(nBins - 1, Math.floor(((v - min) / span) * nBins));
    }
    if (bin >= 0 && bin < nBins) counts[bin]++;
  }

  const transformed = logY ? counts.map((v) => Math.log1p(v)) : counts;
  const peak = Math.max(...transformed);
  return peak > 0 ? transformed.map((v) => v / peak) : transformed;
}

// ── Props ────────────────────────────────────────────────────────────────────

export type ExplainRightPanelProps = {
  v1: string;
  z1: string | null;
  node: number | null;
  drawerOpen: boolean;
  /** null entries are nodes with no data for the selected channel (the water is shallower
   *  than its depth), not zero-error nodes. */
  errors: (number | null)[] | null;
  colorbarMax: number;
  colorbarMin?: number;            // pred mode: lower bound of physical range; defaults to 0 (error mode)
  colorbarPhase: "skeleton" | "ready";
  grid: GridData | null;
  displayMode: "error" | "prediction";
  onDisplayModeChange: (mode: "error" | "prediction") => void;
  onV1Change: (newV1: string, newZ1: string | null) => void;
  onZ1Change: (newZ1: string) => void;
  onNodeClear: () => void;
  onOpenAttribution: () => void;
  onLogScaleChange?: (active: boolean) => void;
  sharedColorbar: boolean;
  onSharedColorbarChange: (active: boolean) => void;
  onOpenTour?: () => void;
  collapsed?: boolean;
  onToggleCollapse?: () => void;
  // Ocean models have no atmospheric/surface split or pressure levels — every state
  // channel (e.g. "thetao_47m") is a flat, independently-selectable variable. When set,
  // this list replaces the Atmospheric/Surface groups with a single "Ocean State" group.
  oceanVars?: string[] | null;
  /** Variables / levels holding the largest component errors at the displayed timestep,
   *  by rank 1..3. Empty when the model's timeline artifact carries no peak data. Keys are
   *  raw variable names and "variable@level"; ocean channels carry their depth in the name
   *  and so only ever appear in varRank. */
  peakMarks?: { varRank: Record<string, number>; levelRank: Record<string, number> } | null;
};

// ── Small circle indicator ───────────────────────────────────────────────────

function Radio({ active, rank }: { active: boolean; rank?: number }): React.ReactElement {
  // rank 1..3 tints the dot with the peak-error hue; anything else renders unchanged.
  const err = rank && rank >= 1 && rank <= 3 ? ` ex-radio--err ex-radio--err${rank}` : "";
  return <span className={`ex-radio${active ? " active" : ""}${err}`} aria-hidden="true" />;
}

/** Suffix for a marked control's tooltip, so the colour is not the only signal. */
function peakTitle(rank: number | undefined): string {
  return rank ? `  ·  #${rank} largest error at this timestep` : "";
}

// ── Component ────────────────────────────────────────────────────────────────

export default function ExplainRightPanel({
  v1,
  z1,
  node,
  drawerOpen,
  errors,
  colorbarMax,
  colorbarMin = 0,
  colorbarPhase,
  grid,
  displayMode,
  onDisplayModeChange,
  onV1Change,
  onZ1Change,
  onNodeClear,
  onOpenAttribution,
  onLogScaleChange,
  sharedColorbar,
  onSharedColorbarChange,
  collapsed,
  onToggleCollapse,
  oceanVars,
  peakMarks,
}: ExplainRightPanelProps): React.ReactElement {
  const [logY, setLogY] = useState(true);
  const [logX, setLogX] = useState(false);
  const [nBins, setNBins] = useState(100);

  const isOcean = !!oceanVars && oceanVars.length > 0;

  // Peak-error marking. Absent marks resolve to undefined, which Radio ignores.
  const varRank   = peakMarks?.varRank   ?? {};
  const levelRank = peakMarks?.levelRank ?? {};
  const isAtm = !isOcean && (ATM_VARS as readonly string[]).includes(v1);
  const isTotal = v1 === "total";
  const isPrediction = displayMode === "prediction";
  // In prediction mode, aggregated cases (Total Error, Σ levels) are hidden because
  // averaging predictions across levels/variables has no physical meaning. The button
  // hide is reinforced by the parent which auto-redirects v1/z1 on mode switch.
  const showTotalRow = !isPrediction;
  const showSumLevels = !isPrediction;
  // Ocean channels have no pressure levels — never show the Level column for them.
  const showZ1Col = !isOcean && (isAtm || isTotal);
  const attrBlocked = isTotal || z1 === "sum";

  const hasNode = node !== null;
  let nodeLat: number | null = null;
  let nodeLon: number | null = null;
  let nodeRmse: number | null = null;
  if (hasNode && grid && node !== null) {
    nodeLat = grid.lat[node] ?? null;
    nodeLon = grid.lon[node] ?? null;
    nodeRmse = errors ? (errors[node] ?? null) : null;
  }

  const formatCoord = (val: number, isLat: boolean): string => {
    const abs = Math.abs(val).toFixed(2);
    const dir = isLat ? (val >= 0 ? "N" : "S") : (val >= 0 ? "E" : "W");
    return `${abs}°${dir}`;
  };

  // Masked nodes in the current field. Drives the no-data legend below the colorbar, which is
  // only shown when there are any — ERA5 and surface ocean variables have none, and an
  // unexplained grey swatch there would be noise.
  const noDataCount = useMemo(
    () => (errors ? errors.reduce<number>((n, v) => (v === null || !Number.isFinite(v) ? n + 1 : n), 0) : 0),
    [errors],
  );

  const histBars = useMemo(() => {
    if (!errors || errors.length === 0) return null;
    if (colorbarMax - colorbarMin <= 0) return null;
    return computeHistogram(errors, colorbarMax, nBins, logY, logX, colorbarMin);
  }, [errors, colorbarMax, colorbarMin, nBins, logY, logX]);

  const modeLabel = colorbarModeLabel(v1, z1, sharedColorbar, isOcean);

  return (
    <>
      <button
        className="panel-edge-tab"
        id="explain-panel-right-header"
        type="button"
        title={collapsed ? "Expand panel" : "Collapse panel"}
        onClick={onToggleCollapse}
      >
        <span className={`chevron${collapsed ? " chevron--left" : " chevron--right"}`}>{"<"}</span>
      </button>

      {/* ── Compact target/node summary — visible only when the attribution drawer
            is open (CSS-controlled). Keeps the target readable above the greyed-out
            right panel without duplicating selectors users can't reach. ───────── */}
      <div id="ex-target-summary-sec" className="ex-target-summary">
        <span className="ex-drawer-hdr-key">Target</span>
        <span className="ex-drawer-hdr-val">
          {isTotal
            ? "Total Error"
            : (SURF_VARS as readonly string[]).includes(v1)
              ? varLabel(v1)
              : z1 && z1 !== "sum"
                ? `${varLabel(v1)} @ ${z1} hPa`
                : varLabel(v1)}
        </span>
        <span className="ex-drawer-hdr-sep">|</span>
        <span className="ex-drawer-hdr-key">Node</span>
        <span className="ex-drawer-hdr-val">
          {hasNode && nodeLat !== null && nodeLon !== null
            ? `${formatCoord(nodeLat, true)}, ${formatCoord(nodeLon, false)}`
            : "No node selected"}
        </span>
      </div>

      {/* ── Header: Explain Predictions ──────────────────────────── */}
      <div id="ex-header-sec" className="ex-sec ex-header-sec">
        <div className="ex-header-row">
          <div className="rs-title">Explain Predictions</div>
        </div>
        <p className="rs-sub">Select a prediction target variable (from below) and then click a node on the map you want to inspect to open the attribution analysis</p>
        {/* Mode toggle: error heatmap vs. raw prediction display.
            Pill style matches the Compare:/Method: toggles in the attribution drawer. */}
        <div
          className="ex-drawer-toggle ex-drawer-toggle--pill"
          role="radiogroup"
          aria-label="Map display mode"
          style={{ marginTop: 8 }}
        >
          <span
            className={`ex-drawer-toggle-pill-thumb${isPrediction ? " ex-drawer-toggle-pill-thumb--right" : ""}`}
            aria-hidden="true"
          />
          <button
            type="button"
            role="radio"
            aria-checked={!isPrediction}
            title="Show absolute error per node (|pred − truth|)"
            className={`ex-drawer-toggle-btn${!isPrediction ? " active" : ""}`}
            onClick={() => onDisplayModeChange("error")}
          >
            Error
          </button>
          <button
            type="button"
            role="radio"
            aria-checked={isPrediction}
            title="Show the model's prediction in physical units"
            className={`ex-drawer-toggle-btn${isPrediction ? " active" : ""}`}
            onClick={() => onDisplayModeChange("prediction")}
          >
            Prediction
          </button>
        </div>
      </div>

      {/* ── Colorbar ─────────────────────────────────────────── */}
      <div id="ex-colorbar-sec" className="ex-sec ex-colorbar-sec" aria-hidden="true">
        <div className="ex-colorbar-hd-row">
          <span className="ex-colorbar-hd" style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
            {isPrediction ? "Prediction (physical)" : "RMSE (norm.)"}
            <HintButton placement="right" content={isPrediction ? HINT_COLORBAR_PREDICTION : HINT_COLORBAR} />
          </span>
          {colorbarPhase === "ready" && (
            <div className="ex-colorbar-controls">
              {/* log scales don't apply to physical predictions (which can be negative) */}
              {!isPrediction && (
                <>
                  <button
                    type="button"
                    title="Toggle logarithmic y axis"
                    className={`log-toggle${logY ? " active" : ""}`}
                    onClick={() => setLogY((v) => !v)}
                  >
                    log y
                  </button>
                  <button
                    type="button"
                    title="Toggle logarithmic x axis"
                    className={`log-toggle${logX ? " active" : ""}`}
                    onClick={() => {
                      const next = !logX;
                      setLogX(next);
                      onLogScaleChange?.(next);
                    }}
                  >
                    log x
                  </button>
                </>
              )}
              <div className="hist-bins-wrap">
                bins
                <input
                  className="hist-bins-input"
                  type="number"
                  min={5}
                  max={100}
                  value={nBins}
                  onChange={(e) => {
                    const v = Math.max(5, Math.min(100, parseInt(e.target.value, 10) || 30));
                    setNBins(v);
                  }}
                />
              </div>
            </div>
          )}
        </div>
        {colorbarPhase === "skeleton" ? (
          <div className="ex-colorbar-skel">
            <div className="ex-spinner ex-spinner--sm" />
          </div>
        ) : (
          <>
            <svg
              className="ex-colorbar-svg"
              viewBox="0 0 1000 60"
              preserveAspectRatio="none"
            >
              <defs>
                <linearGradient id="ex-lb-grad" x1="0" y1="0" x2="1" y2="0">
                  {isPrediction ? (
                    getPredGradientStops(v1).map((s) => (
                      <stop key={s.offset} offset={s.offset} stopColor={s.stopColor} />
                    ))
                  ) : (
                    <>
                      <stop offset="0%"    stopColor={palette.errorHeatmap.stop0} />
                      <stop offset="1%"    stopColor={palette.errorHeatmap.stop1} />
                      <stop offset="33.3%" stopColor={palette.errorHeatmap.stop2} />
                      <stop offset="66.7%" stopColor={palette.errorHeatmap.stop3} />
                      <stop offset="83.3%" stopColor={palette.errorHeatmap.stop4} />
                      <stop offset="100%"  stopColor={palette.errorHeatmap.stop5} />
                    </>
                  )}
                </linearGradient>
              </defs>
              <rect x="0" y="48" width="1000" height="12" rx="2" fill="url(#ex-lb-grad)" />
              {histBars && histBars.map((h, i) => {
                const barW = 1000 / nBins;
                const barH = h * 44;
                return (
                  <rect
                    key={i}
                    x={i * barW}
                    y={46 - barH}
                    width={barW}
                    height={barH}
                    fill="var(--text)"
                    fillOpacity={0.2}
                  />
                );
              })}
            </svg>
            <div className="ex-colorbar-ticks">
              {isPrediction && isDivergingPred(v1) ? (() => {
                const A = Math.max(Math.abs(colorbarMin), Math.abs(colorbarMax));
                const fmt = (v: number) => Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(2);
                return (
                  <>
                    <span>{fmt(-A)}</span>
                    <span>0</span>
                    <span>{fmt(A)}</span>
                  </>
                );
              })() : (
                <>
                  <span>
                    {isPrediction
                      ? colorbarMin.toFixed(2)
                      : (logX ? "1" : "0")}
                  </span>
                  <span>
                    {colorbarMax > colorbarMin
                      ? isPrediction
                        ? colorbarMax.toFixed(2)
                        : logX
                          ? `log(${colorbarMax.toFixed(3)})`
                          : colorbarMax.toFixed(3)
                      : "–"}
                  </span>
                </>
              )}
            </div>
            {/* Explains the grey on the map: those nodes are not zero-error, they have no
                value at all because the water is shallower than the selected depth. Without
                this the grey reads as a rendering failure. */}
            {noDataCount > 0 && (
              <div className="ex-colorbar-nodata">
                <span
                  className="ex-colorbar-nodata-swatch"
                  style={{ background: palette.errorHeatmap.noData }}
                />
                <span>no data · shallower than this depth ({noDataCount.toLocaleString()} nodes)</span>
              </div>
            )}
          </>
        )}
        {/* "Share colorbar" toggles per-variable vs shared error scale. Predictions
            already use a fixed physical scale per variable, so the toggle is meaningless
            and hidden. */}
        {!isPrediction && (
          <div className="ex-colorbar-toggle-row">
            <label className="ex-colorbar-checkbox-label">
              <input
                type="checkbox"
                className="ex-colorbar-checkbox"
                checked={sharedColorbar}
                onChange={(e) => onSharedColorbarChange(e.target.checked)}
              />
              <span className="ex-checkbox-box" />
              <span className="ex-checkbox-text">Share colorbar</span>
              <HintButton placement="top" content={HINT_SHARE_COLORBAR} />
            </label>
          </div>
        )}
        <div className="ex-colorbar-mode">
          {isPrediction
            ? `${varLabel(v1)}${z1 && z1 !== "sum" ? ` @ ${z1} hPa` : ""} — physical units`
            : modeLabel}
        </div>
      </div>

      <div className="ex-panel-body">
      {/* ── Prediction target section ────────────────────────────── */}
      <div id="ex-target-sec" className="ex-sec ex-target-sec">
        <div className="ex-sec-hd">Prediction Target</div>

        {/* Selected Node — moved here from its own section so all "what am I
            targeting?" controls live under one header. */}
        <div id="ex-node-sec" className="ex-subsec ex-subsec--selected-node">
          <div className="ex-sec-hd-row">
            <div className="ex-subsec-hd">Selected Node</div>
            {hasNode && (
              <button type="button" className="ex-node-clear" title="Deselect the current grid node" onClick={onNodeClear}>
                Clear
              </button>
            )}
          </div>
          {!hasNode ? (
            <div className="ex-node-empty">No node selected</div>
          ) : (
            <div className="ex-node-readout">
              <div className="ex-node-rmse">
                <span className="ex-node-rmse-label">Coordinates</span>
                {nodeLat !== null && nodeLon !== null && (
                  <span className="ex-node-rmse-val">
                    {formatCoord(nodeLat, true)}, {formatCoord(nodeLon, false)}
                  </span>
                )}
              </div>
              {nodeRmse !== null && (
                <div className="ex-node-rmse">
                  <span className="ex-node-rmse-label">
                    {isPrediction ? "Prediction (physical)" : "RMSE (normalised)"}
                  </span>
                  <span className="ex-node-rmse-val">{nodeRmse.toFixed(4)}</span>
                </div>
              )}
            </div>
          )}
        </div>

        <div id="ex-selected-variable-sec" className="ex-subsec ex-subsec--selected-variable">
          <div className="ex-subsec-hd">Selected Variable</div>

          {/* Total Error — pinned at top. Hidden in prediction mode because
              averaging predictions across all variables/levels has no physical meaning. */}
          {showTotalRow && (
            <>
              <button
                type="button"
                className={`ex-v1-item ex-v1-total-row${isTotal ? " active" : ""}`}
                onClick={() => { if (!isTotal) onV1Change("total", null); }}
              >
                <Radio active={isTotal} />
                <span className="ex-v1-item-label" style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
                  Total Error
                  <HintButton placement="right" content={HINT_TOTAL_ERROR} />
                </span>
              </button>
              {isTotal && (
                <p className="ex-total-note">
                  Navigation only — select a variable to inspect attribution.
                </p>
              )}
            </>
          )}

        <div className="ex-var-z1-layout">
          {/* Left: variable list */}
          <div className="ex-v1-col">
            {isOcean ? (
              <>
                <div className="ex-v1-group-hd">Ocean State</div>
                {oceanVars!.map((vname) => (
                  <button
                    key={vname}
                    type="button"
                    className={`ex-v1-item${v1 === vname ? " active" : ""}`}
                    onClick={() => { if (v1 !== vname) onV1Change(vname, null); }}
                    title={`${vname}${peakTitle(varRank[vname])}`}
                  >
                    <Radio active={v1 === vname} rank={varRank[vname]} />
                    <span className="ex-v1-item-label">{varLabel(vname)}</span>
                  </button>
                ))}
              </>
            ) : (
              <>
                <div className="ex-v1-group-hd">Atmospheric</div>
                {(ATM_VARS as readonly string[]).map((vname) => (
                  <button
                    key={vname}
                    type="button"
                    className={`ex-v1-item${v1 === vname ? " active" : ""}`}
                    onClick={() => { if (v1 !== vname) onV1Change(vname, null); }}
                    title={`${VAR_LABELS[vname] ?? vname}${peakTitle(varRank[vname])}`}
                  >
                    <Radio active={v1 === vname} rank={varRank[vname]} />
                    <span className="ex-v1-item-label">{varLabel(vname)}</span>
                  </button>
                ))}

                <div className="ex-v1-group-hd">Surface</div>
                {(SURF_VARS as readonly string[]).map((vname) => (
                  <button
                    key={vname}
                    type="button"
                    className={`ex-v1-item${v1 === vname ? " active" : ""}`}
                    onClick={() => { if (v1 !== vname) onV1Change(vname, null); }}
                    title={`${VAR_LABELS[vname] ?? vname}${peakTitle(varRank[vname])}`}
                  >
                    <Radio active={v1 === vname} rank={varRank[vname]} />
                    <span className="ex-v1-item-label">{varLabel(vname)}</span>
                  </button>
                ))}
              </>
            )}
          </div>

          {/* Right: z1 level chips */}
          {showZ1Col ? (
            <div className="ex-z1-col">
              <div className="ex-v1-group-hd">Level</div>
              <div className="ex-z1-stack">
                {/* Σ levels — hidden in prediction mode (level aggregation isn't physical). */}
                {showSumLevels && (
                  <button
                    type="button"
                    className={`ex-z1-chip${z1 === "sum" ? " active" : ""}`}
                    onClick={() => { if (z1 !== "sum") onZ1Change("sum"); }}
                    title="Sum errors across all pressure levels"
                  >
                    <Radio active={z1 === "sum"} />
                    <span>Σ levels</span>
                  </button>
                )}
                {(PRESSURE_LEVELS as readonly number[]).map((lvl) => {
                  const key = String(lvl);
                  // Marked by rank only. Per-level error magnitudes are not stored — the
                  // artifact keeps the top-3 components, not the full (T, d_state) matrix —
                  // so there is no intensity to shade the rest of the profile with.
                  const rank = levelRank[`${v1}@${key}`];
                  return (
                    <button
                      key={key}
                      type="button"
                      className={`ex-z1-chip${z1 === key ? " active" : ""}`}
                      onClick={() => { if (z1 !== key) onZ1Change(key); }}
                      title={rank ? `${lvl} hPa${peakTitle(rank)}` : undefined}
                    >
                      <Radio active={z1 === key} rank={rank} />
                      <span>{lvl} hPa</span>
                    </button>
                  );
                })}
              </div>
            </div>
          ) : !isOcean ? (
            <div className="ex-z1-col ex-z1-col--surface">
              <div className="ex-v1-group-hd">Level</div>
              <div className="ex-surface-label">Surface</div>
            </div>
          ) : null}
        </div>

          {z1 === "sum" && !isTotal && (
            <p className="ex-total-note" style={{ marginTop: 6 }}>
              Navigation only — select a level to inspect attribution.
            </p>
          )}
        </div>{/* /ex-subsec--selected-variable */}
      </div>

      {/* ── Footer: attribution button ────────────────────────────── */}
      <div id="ex-panel-footer" className="ex-panel-footer">
        {attrBlocked ? (
          <button
            type="button"
            className="ex-attr-btn ex-attr-btn--disabled"
            disabled
            title={
              isTotal
                ? "Select a specific variable to inspect attribution."
                : "Select a specific pressure level to inspect attribution."
            }
          >
            Open attribution
          </button>
        ) : (
          <button
            type="button"
            title="Open the feature attribution analysis for the selected node"
            className={`ex-attr-btn${drawerOpen ? " ex-attr-btn--open" : ""}`}
            onClick={onOpenAttribution}
          >
            {drawerOpen ? "Attribution opened" : "Open attribution"}
          </button>
        )}
      </div>
      </div>
    </>
  );
}
