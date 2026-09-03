"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { TIMELINE_AVAILABLE } from '../../lib/colors';
import { GCS_FETCH_ENABLED } from '../../lib/config';
import HintButton from "./HintButton";
import { HINT_TIMELINE_OVERVIEW } from "../../lib/hints";

// ── Types ──────────────────────────────────────────────────────────────────────

export type MonthBand = {
  label: string;
  start: string;
  end: string;
  count: number;
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
};

/** Which error series the chart plots. Exclusive: the three sit on unrelated
 *  magnitudes and units (normalised RMSE, normalised max, a raw component count),
 *  so overlaying them would need a shared axis that flattens all three. */
type SeriesKey = "mean" | "peak" | "concentration";

type SeriesSpec = {
  key: SeriesKey;
  /** Chip label */
  chip: string;
  /** Panel heading when this series is active over a multi-month range */
  label: string;
  /** Chip tooltip */
  hint: string;
};

const SERIES: readonly SeriesSpec[] = [
  {
    key: "mean",
    chip: "Mean error",
    label: "TEST SET MEAN ERROR (NORM.)",
    hint: "RMSE over every grid node and state variable.",
  },
  {
    key: "peak",
    chip: "Peak error",
    label: "TEST SET PEAK ERROR (NORM.)",
    hint: "Largest single component error anywhere on the grid, before any averaging.",
  },
  {
    key: "concentration",
    chip: "Concentration",
    label: "TEST SET ERROR CONCENTRATION (COMPONENTS)",
    hint: "How many state components peak on the same grid node — a unit-free count of how coherent the error is.",
  },
] as const;

type Props = {
  timelineData: TimelineData | null;
  /** Live set of pred_time values the heatmap endpoint will accept. null = still loading (optimistic). */
  availableTimestamps: Set<string> | null;
  currentT: string;
  onTimestepChange: (t: string) => void;
  // Month band props — wired by Task D
  monthBands?: MonthBand[];
  /** band.start keys for every band loaded into local zarr cache */
  downloadedBandStarts?: Set<string>;
  /** all timestamps ever downloaded — used to colour the overview RMSE line */
  allDownloadedTimestamps?: Set<string>;
  windowLoading?: boolean;
  loadingBand?: string | null;
  onBandClick?: (band: MonthBand) => void;
};

// ── Helpers ────────────────────────────────────────────────────────────────────

/**
 * Parse a numpy datetime64[ns] string as UTC.
 * Without this, browsers treat date-time strings without a timezone suffix as
 * local time (ES spec §20.4.1.15), causing a display shift equal to the local
 * UTC offset. We strip sub-millisecond precision and append Z to force UTC.
 * "2020-06-01T12:00:00.000000000" → new Date("2020-06-01T12:00:00.000Z")
 */
function parseUTC(iso: string): Date {
  return new Date(iso.slice(0, 23) + "Z");
}

function daysBetween(a: string, b: string): number {
  return Math.abs(parseUTC(b).getTime() - parseUTC(a).getTime()) / 86_400_000;
}

/** "Jun 1, 2020 · 00:00 UTC" */
function fmtLabel(iso: string): string {
  const d = parseUTC(iso);
  if (isNaN(d.getTime())) return iso;
  const month = d.toLocaleString("en-US", { month: "short", timeZone: "UTC" });
  const hh = String(d.getUTCHours()).padStart(2, "0");
  const mm = String(d.getUTCMinutes()).padStart(2, "0");
  return `${month} ${d.getUTCDate()}, ${d.getUTCFullYear()} · ${hh}:${mm} UTC`;
}

/** "Jun 1, 2020" — compact label */
function fmtShort(iso: string): string {
  const d = parseUTC(iso);
  if (isNaN(d.getTime())) return iso;
  const month = d.toLocaleString("en-US", { month: "short", timeZone: "UTC" });
  return `${month} ${d.getUTCDate()}, ${d.getUTCFullYear()}`;
}

/** "YYYY-MM-DD HH:MM" — input field */
function fmtInput(iso: string): string {
  const d = parseUTC(iso);
  if (isNaN(d.getTime())) return "";
  return d.toISOString().slice(0, 16).replace("T", " ");
}

function nearestIndex(timestamps: string[], iso: string): number {
  const target = Date.parse(iso);
  if (isNaN(target)) return 0;
  let best = 0;
  let bestDelta = Infinity;
  for (let i = 0; i < timestamps.length; i++) {
    const delta = Math.abs(Date.parse(timestamps[i]) - target);
    if (delta < bestDelta) { bestDelta = delta; best = i; }
  }
  return best;
}

/** Build per-segment data for availability-coloured sparklines.
 *  Each segment is a run of consecutive available or unavailable points.
 *  Null availableTimestamps → treat all as available (optimistic render). */
function sparkSegments(
  values: number[],
  timestamps: string[],
  availableTimestamps: Set<string> | null,
  w: number,
  h: number,
): Array<{ points: string; available: boolean }> {
  if (values.length < 2) return [];
  const max = Math.max(...values);
  const min = Math.min(...values);
  const range = max - min || 1;
  const pts = values.map((v, i) => ({
    x: (i / (values.length - 1)) * w,
    y: h - ((v - min) / range) * (h - 2) - 1,
    available: !availableTimestamps || availableTimestamps.has(timestamps[i]),
  }));

  if (!availableTimestamps) {
    return [{ points: pts.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" "), available: true }];
  }

  const segments: Array<{ points: string; available: boolean }> = [];
  let i = 0;
  while (i < pts.length) {
    const curAvail = pts[i].available;
    let j = i;
    while (j < pts.length && pts[j].available === curAvail) j++;
    // include one overlap point on each boundary for visual continuity
    const start = i > 0 ? i - 1 : i;
    const end = j < pts.length ? j + 1 : j;
    segments.push({
      points: pts.slice(start, end).map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" "),
      available: curAvail,
    });
    i = j;
  }
  return segments;
}

/** Default coarse window: min(n, max(30, ceil(n * 0.10))). */
function defaultWindowSize(n: number): number {
  return Math.min(n, Math.max(30, Math.ceil(n * 0.10)));
}

function DownloadIcon(): React.ReactElement {
  return (
    <svg width="9" height="9" viewBox="0 0 9 9" fill="none" aria-hidden="true" style={{ display: "block" }}>
      <path d="M4.5 0.5v5M2 4l2.5 3L7 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
      <line x1="1" y1="8.5" x2="8" y2="8.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
    </svg>
  );
}

// ── Component ──────────────────────────────────────────────────────────────────

export default function ExplainTimeline({
  timelineData,
  availableTimestamps,
  currentT,
  onTimestepChange,
  monthBands = [],
  downloadedBandStarts,
  allDownloadedTimestamps,
  windowLoading = false,
  loadingBand = null,
  onBandClick = () => {},
}: Props): React.ReactElement {
  const initialWindowEnd = timelineData ? defaultWindowSize(timelineData.timestamps.length) : 0;
  const [windowStart, setWindowStart] = useState(0);
  const [windowEnd, setWindowEnd] = useState(initialWindowEnd);

  // Controlled date input
  const [inputValue, setInputValue] = useState(() => fmtInput(currentT));
  const [inputFocused, setInputFocused] = useState(false);

  // Drag state for overview window
  const overviewRef = useRef<SVGSVGElement>(null);
  const dragRef = useRef<{ startX: number; startWin: number } | null>(null);
  // Persists through pointerup so onClick can check it (onClick fires after pointerup)
  const didDragRef = useRef(false);

  const [hoveredBand, setHoveredBand] = useState<string | null>(null);

  const n = timelineData?.timestamps.length ?? 0;
  const windowSize = windowEnd > windowStart ? windowEnd - windowStart : 0;

  // Use allDownloadedTimestamps (accumulates every download) when available; fall back to
  // availableTimestamps (current window only). This ensures the detail panel stays consistent
  // with the overview — all ever-downloaded timestamps are coloured and clickable.
  const effectiveTs = allDownloadedTimestamps ?? availableTimestamps;

  // Keep a ref to window state so the auto-scroll effect can read it without stale closures
  const windowStateRef = useRef({ windowStart, windowEnd, windowSize, n });
  useEffect(() => {
    windowStateRef.current = { windowStart, windowEnd, windowSize, n };
  }, [windowStart, windowEnd, windowSize, n]);

  // When currentT changes externally (e.g. after a band load), scroll the window to show it
  useEffect(() => {
    if (!timelineData || !currentT) return;
    const { windowStart: ws, windowEnd: we, windowSize: wsize, n: nn } = windowStateRef.current;
    const idx = timelineData.timestamps.indexOf(currentT);
    if (idx < 0 || (idx >= ws && idx < we)) return;
    const newStart = Math.max(0, Math.min(nn - wsize, idx));
    setWindowStart(newStart);
    setWindowEnd(Math.min(nn, newStart + wsize));
  }, [currentT, timelineData]); // intentionally excludes windowStart/windowEnd — read via ref

  // Ordered list of available timestamps within the full timeline (preserves timeline order).
  // null effectiveTs = still loading → treat all timeline ticks as available (optimistic).
  const availableList = useMemo(() => {
    if (!timelineData) return [];
    if (!effectiveTs) return timelineData.timestamps;
    return timelineData.timestamps.filter(t => effectiveTs.has(t));
  }, [timelineData, effectiveTs]);

  // Current position in the available list — used for step-button disabled state.
  const currentAvailIdx = useMemo(() => availableList.indexOf(currentT), [availableList, currentT]);

  // Which error series is plotted. Falls back to the mean below when the loaded
  // model has no data for the chosen chip.
  const [seriesKey, setSeriesKey] = useState<SeriesKey>("mean");

  // All series but the mean are absent on timeline files predating them, in which case
  // their chips are disabled.
  const seriesByKey = useMemo((): Record<SeriesKey, number[] | null> => ({
    mean: timelineData?.mean_error ?? null,
    peak: timelineData?.peak_error ?? null,
    concentration: timelineData?.error_concentration ?? null,
  }), [timelineData]);

  // Derived, not stored: switching to a model whose artifact lacks the selected series
  // shows the mean rather than blanking the panel, and switching back restores the choice.
  const activeKey: SeriesKey = seriesByKey[seriesKey] ? seriesKey : "mean";
  const activeSeries = seriesByKey[activeKey] ?? [];
  const activeSpec = SERIES.find((sp) => sp.key === activeKey) ?? SERIES[0];

  // Sliced data for the detail panel
  const fineTimestamps = useMemo(() => {
    if (!timelineData || windowEnd <= windowStart) return [];
    return timelineData.timestamps.slice(windowStart, windowEnd);
  }, [timelineData, windowStart, windowEnd]);

  const fineSeries = useMemo(() => {
    if (!timelineData || windowEnd <= windowStart) return [];
    return activeSeries.slice(windowStart, windowEnd);
  }, [timelineData, activeSeries, windowStart, windowEnd]);

  // selectedIdx: position of currentT in the full timeline array (for cursor/ring placement).
  // Uses indexOf (ISO string equality) per spec §5.3 — never Date.parse.
  const selectedIdx = useMemo(() => {
    if (!timelineData || !currentT) return -1;
    return timelineData.timestamps.indexOf(currentT);
  }, [timelineData, currentT]);

  const selectedFineIdx =
    selectedIdx >= windowStart && selectedIdx < windowEnd ? selectedIdx - windowStart : -1;

  // ── Step navigation ────────────────────────────────────────────────────────
  // Steps through the available-only list, skipping unavailable timestamps per spec §5.2.
  // Step sizes follow the data's own cadence. ERA5 is 6-hourly, the ocean timeline daily,
  // so the old fixed "6h" / "1d" (1 step / 4 steps) labelled an ocean day as "6h" and a
  // four-day jump as "1d". Median gap is used rather than the first, so one irregular gap
  // (a missing timestep) cannot skew it.
  const { stepLabel, bigStepLabel, bigStepCount } = useMemo(() => {
    const ts = timelineData?.timestamps ?? [];
    if (ts.length < 2) return { stepLabel: "1", bigStepLabel: "", bigStepCount: 0 };
    const gaps: number[] = [];
    for (let i = 1; i < ts.length; i++) gaps.push(Date.parse(ts[i]) - Date.parse(ts[i - 1]));
    gaps.sort((a, b) => a - b);
    const hours = gaps[Math.floor(gaps.length / 2)] / 3_600_000;
    const fmt = (h: number) => (h < 24 ? `${Math.round(h)}h` : h < 168 ? `${Math.round(h / 24)}d` : `${Math.round(h / 168)}w`);
    // The larger jump is the next natural unit up: a day for sub-daily data, a week for daily.
    const bigHours = hours < 24 ? 24 : 168;
    const count = Math.max(2, Math.round(bigHours / hours));
    return { stepLabel: fmt(hours), bigStepLabel: fmt(bigHours), bigStepCount: count };
  }, [timelineData]);

  const stepTo = useCallback((delta: number) => {
    if (!timelineData || availableList.length === 0) return;
    let currentIdx = availableList.indexOf(currentT);
    if (currentIdx < 0) {
      // currentT not in available list — snap to nearest available before stepping
      const target = Date.parse(currentT);
      currentIdx = 0;
      let best = Infinity;
      for (let i = 0; i < availableList.length; i++) {
        const d = Math.abs(Date.parse(availableList[i]) - target);
        if (d < best) { best = d; currentIdx = i; }
      }
    }
    const nextIdx = Math.max(0, Math.min(availableList.length - 1, currentIdx + delta));
    const newT = availableList[nextIdx];
    onTimestepChange(newT);
    // Scroll window to keep selection visible (using position in full timeline array)
    const fullIdx = timelineData.timestamps.indexOf(newT);
    if (fullIdx >= 0) {
      if (fullIdx < windowStart) {
        const s = Math.max(0, fullIdx);
        setWindowStart(s);
        setWindowEnd(Math.min(n, s + windowSize));
      } else if (fullIdx >= windowEnd) {
        const e = Math.min(n, fullIdx + 1);
        setWindowStart(Math.max(0, e - windowSize));
        setWindowEnd(e);
      }
    }
  }, [timelineData, availableList, currentT, n, windowStart, windowEnd, windowSize, onTimestepChange]);

  // ── Overview drag ──────────────────────────────────────────────────────────
  const handleOverviewPointerDown = useCallback((e: React.PointerEvent<SVGSVGElement>) => {
    if (n === 0) return;
    (e.target as Element).setPointerCapture?.(e.pointerId);
    dragRef.current = { startX: e.clientX, startWin: windowStart };
    didDragRef.current = false; // reset at the start of each gesture
  }, [n, windowStart]);

  const handleOverviewPointerMove = useCallback((e: React.PointerEvent<SVGSVGElement>) => {
    if (!dragRef.current || !overviewRef.current || n === 0) return;
    const svgW = overviewRef.current.getBoundingClientRect().width;
    const dx = e.clientX - dragRef.current.startX;
    if (Math.abs(dx) > 2) didDragRef.current = true;
    const stepsDx = Math.round((dx / svgW) * n);
    const newStart = Math.max(0, Math.min(n - windowSize, dragRef.current.startWin + stepsDx));
    setWindowStart(newStart);
    setWindowEnd(Math.min(n, newStart + windowSize));
  }, [n, windowSize]);

  const handleOverviewPointerUp = useCallback(() => {
    dragRef.current = null;
    // didDragRef is intentionally NOT cleared here — onClick fires after pointerup
    // and needs to read the final drag state.
  }, []);

  // Click on overview to re-centre window (only when not dragging)
  const handleOverviewClick = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (didDragRef.current || !overviewRef.current || n === 0) return;
    const rect = overviewRef.current.getBoundingClientRect();
    const frac = (e.clientX - rect.left) / rect.width;
    const clickIdx = Math.round(frac * n);
    const half = Math.floor(windowSize / 2);
    const newStart = Math.max(0, Math.min(n - windowSize, clickIdx - half));
    setWindowStart(newStart);
    setWindowEnd(Math.min(n, newStart + windowSize));
  }, [n, windowSize]);

  // ── Detail click ───────────────────────────────────────────────────────────
  // Selects the nearest available tick in the window. No-op if no available ticks.
  const handleDetailClick = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (!timelineData || fineTimestamps.length === 0) return;
    // Collect available ticks in the current window
    const availableFine = fineTimestamps.filter(t =>
      !effectiveTs || effectiveTs.has(t)
    );
    if (availableFine.length === 0) return;
    const rect = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const fi = Math.round(frac * (fineTimestamps.length - 1));
    const clickedT = fineTimestamps[fi];
    const target = Date.parse(clickedT);
    let best = availableFine[0];
    let bestDelta = Infinity;
    for (const t of availableFine) {
      const d = Math.abs(Date.parse(t) - target);
      if (d < bestDelta) { bestDelta = d; best = t; }
    }
    onTimestepChange(best);
  }, [timelineData, fineTimestamps, effectiveTs, onTimestepChange]);

  // ── Input commit ───────────────────────────────────────────────────────────
  // Snaps to the nearest available timestamp. Falls back to all timestamps if none loaded.
  const commitInput = useCallback(() => {
    if (!timelineData) return;
    const parsed = Date.parse(inputValue);
    if (isNaN(parsed)) { setInputValue(fmtInput(currentT)); return; }
    const snapList = effectiveTs
      ? timelineData.timestamps.filter(t => effectiveTs.has(t))
      : timelineData.timestamps;
    if (snapList.length === 0) { setInputValue(fmtInput(currentT)); return; }
    const idx = nearestIndex(snapList, new Date(parsed).toISOString());
    const newT = snapList[idx];
    onTimestepChange(newT);
    setInputValue(fmtInput(newT));
    const fullIdx = timelineData.timestamps.indexOf(newT);
    if (fullIdx >= 0) {
      const half = Math.floor(windowSize / 2);
      const newStart = Math.max(0, Math.min(n - windowSize, fullIdx - half));
      setWindowStart(newStart);
      setWindowEnd(Math.min(n, newStart + windowSize));
    }
  }, [timelineData, effectiveTs, inputValue, currentT, onTimestepChange, windowSize, n]);

  // ── Loading state ──────────────────────────────────────────────────────────
  if (!timelineData || n === 0) {
    return (
      <div className="etl-root etl-loading" aria-label="Timeline loading">
        <div className="ex-spinner ex-spinner--xs" />
        <span className="etl-placeholder-label">Loading timeline…</span>
      </div>
    );
  }

  // ── SVG geometry constants ─────────────────────────────────────────────────
  const W = 1000;
  const H = 28; // viewBox height — stretched to fill container via preserveAspectRatio="none"

  // Overview window rect in SVG coords
  const winX = n > 0 ? (windowStart / n) * W : 0;
  const winW = n > 0 ? Math.max((windowSize / n) * W, 3) : 0;

  // Detail cursor position (fraction along detail panel)
  const cursorFrac =
    selectedFineIdx >= 0 && fineTimestamps.length > 1
      ? selectedFineIdx / (fineTimestamps.length - 1)
      : selectedFineIdx === 0 ? 0 : -1;

  const overviewLabel =
    daysBetween(timelineData.range_start, timelineData.range_end) > 30
      ? activeSpec.label
      : "DATASET OVERVIEW";

  const detailRangeLabel =
    fineTimestamps.length > 1
      ? `${fmtShort(fineTimestamps[0])} – ${fmtShort(fineTimestamps[fineTimestamps.length - 1])}`
      : fineTimestamps.length === 1 ? fmtShort(fineTimestamps[0]) : "—";

  const displayedInputValue = inputFocused ? inputValue : fmtInput(currentT);

  return (
    <div className="etl-root">

      {/* ── Charts row (coarse left · detail right) ── */}
      <div className="etl-charts-row">

        {/* Left: overview sparkline */}
        <div className="etl-panel etl-panel--overview">
          <div className="etl-panel-header">
            <span className="etl-section-label" style={{ display: "inline-flex", alignItems: "center", gap: 2 }}>
              {overviewLabel}
              <HintButton placement="top" content={HINT_TIMELINE_OVERVIEW} />
            </span>
            <div className="etl-series-chips" role="radiogroup" aria-label="Error series">
              {SERIES.map((sp) => {
                // The mean is always present; the others only once the artifact carries them.
                const has = sp.key === "mean" || !!seriesByKey[sp.key];
                return (
                  <button
                    key={sp.key}
                    type="button"
                    role="radio"
                    aria-checked={activeKey === sp.key}
                    disabled={!has}
                    className={`etl-series-chip${activeKey === sp.key ? " active" : ""}`}
                    onClick={() => setSeriesKey(sp.key)}
                    title={has ? sp.hint : `${sp.hint} Not available for this model — re-run scripts/precompute_timeline.py.`}
                  >
                    {sp.chip}
                  </button>
                );
              })}
            </div>
          </div>
          <svg
            ref={overviewRef}
            className="etl-sparkline etl-sparkline--overview"
            viewBox={`0 0 ${W} ${H}`}
            preserveAspectRatio="none"
            onPointerDown={handleOverviewPointerDown}
            onPointerMove={handleOverviewPointerMove}
            onPointerUp={handleOverviewPointerUp}
            onPointerCancel={handleOverviewPointerUp}
            onClick={handleOverviewClick}
            aria-label="RMSE overview — drag window to pan"
          >
            {/* Month band highlight rects — pointer-events off, clicks handled by label row */}
            {n > 1 && monthBands.map((band) => {
              const startIdx = timelineData.timestamps.indexOf(band.start);
              const endIdx = timelineData.timestamps.indexOf(band.end);
              if (startIdx < 0 || endIdx < 0) return null;
              const tickInterval = W / (n - 1);
              const xStart = startIdx * tickInterval;
              const xEnd = endIdx * tickInterval + tickInterval;
              const bandW = xEnd - xStart;
              const isHovered = hoveredBand === band.start;
              const fillOpacity = isHovered ? 0.10 : 0;
              return (
                <React.Fragment key={band.start}>
                  <rect
                    x={xStart} y={0} width={bandW} height={H}
                    fill="var(--text-dim)"
                    fillOpacity={fillOpacity}
                    pointerEvents="none"
                  />
                  <line
                    x1={xStart} y1={0} x2={xStart} y2={H}
                    stroke="var(--border)" strokeWidth={1}
                    vectorEffect="non-scaling-stroke"
                    pointerEvents="none"
                  />
                </React.Fragment>
              );
            })}
            {sparkSegments(activeSeries, timelineData.timestamps, allDownloadedTimestamps ?? null, W, H).map((seg, si) => (
              <polyline
                key={si}
                points={seg.points}
                stroke={seg.available ? TIMELINE_AVAILABLE : "var(--text-dim)"}
                strokeWidth={seg.available ? 1.8 : 1}
                vectorEffect="non-scaling-stroke"
                fill="none"
              />
            ))}
            {/* Draggable window */}
            <rect x={winX} y={0} width={winW} height={H} className="etl-window-rect" />
          </svg>
        </div>

        {/* Separator */}
        <div className="etl-sep" />

        {/* Right: detail sparkline */}
        <div className="etl-panel etl-panel--detail">
          <div className="etl-panel-header">
            <span className="etl-section-label">Detail — {detailRangeLabel}</span>
            {selectedFineIdx >= 0 && currentT && (
              <span className="etl-tick-label">{fmtLabel(currentT)}</span>
            )}
          </div>
          {fineTimestamps.length === 0 ? (
            <div className="etl-no-data">No data in range</div>
          ) : (
            <svg
              className="etl-sparkline etl-sparkline--detail"
              viewBox={`0 0 ${W} ${H}`}
              preserveAspectRatio="none"
              onClick={handleDetailClick}
              aria-label="Detail RMSE — click to select timestep"
            >
              {sparkSegments(fineSeries, fineTimestamps, effectiveTs, W, H).map((seg, si) => (
                <polyline
                  key={si}
                  points={seg.points}
                  stroke={seg.available ? TIMELINE_AVAILABLE : "var(--text-dim)"}
                  strokeWidth={seg.available ? 1.8 : 1}
                  vectorEffect="non-scaling-stroke"
                  fill="none"
                />
              ))}
              {/* Tick marks: available ticks are coloured and clickable; unavailable are greyed and inert */}
              {fineTimestamps.map((ts, fi) => {
                const x = fineTimestamps.length > 1 ? (fi / (fineTimestamps.length - 1)) * W : W / 2;
                const isSelected = fi === selectedFineIdx;
                const isAvailable = !effectiveTs || effectiveTs.has(ts);
                let className: string;
                if (isSelected) {
                  className = "etl-tick etl-tick--active";
                } else if (!isAvailable) {
                  className = "etl-tick etl-tick--unavailable";
                } else {
                  className = "etl-tick";
                }
                return (
                  <line
                    key={fi}
                    x1={x} y1={isSelected ? 0 : H - 6}
                    x2={x} y2={H}
                    className={className}
                  />
                );
              })}
              {/* Cursor line */}
              {cursorFrac >= 0 && (
                <line
                  x1={cursorFrac * W} y1={0}
                  x2={cursorFrac * W} y2={H}
                  className="etl-cursor"
                />
              )}
            </svg>
          )}
        </div>
      </div>

      {/* Month band label row — below both charts, so the coarse sparkline keeps
          the same height as the detail one. Spans the coarse panel width. */}
      {n > 1 && monthBands.length > 0 && (
        <div className="etl-bands-row">
          <div className="etl-band-row">
            {monthBands.map((band) => {
              const startIdx = timelineData.timestamps.indexOf(band.start);
              const endIdx = timelineData.timestamps.indexOf(band.end);
              if (startIdx < 0 || endIdx < 0) return null;
              const leftPct = (startIdx / (n - 1)) * 100;
              const rawWidthPct = ((endIdx - startIdx + 1) / (n - 1)) * 100;
              const widthPct = Math.min(rawWidthPct, 100 - leftPct);
              const isLoading = band.start === loadingBand;
              const isHovered = hoveredBand === band.start;
              const isDownloaded = downloadedBandStarts?.has(band.start) ?? false;
              const canClick = GCS_FETCH_ENABLED && !windowLoading && !isLoading;
              return (
                <div
                  key={band.start}
                  className={[
                    "etl-band-item",
                    isLoading ? "etl-band-item--loading" : "",
                    isDownloaded ? "etl-band-item--loaded" : "",
                    isHovered && canClick ? "etl-band-item--hovered" : "",
                  ].filter(Boolean).join(" ")}
                  style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
                  onMouseEnter={() => setHoveredBand(band.start)}
                  onMouseLeave={() => setHoveredBand(null)}
                  onClick={() => { if (canClick) onBandClick(band); }}
                  title={band.label}
                >
                  {isLoading ? (
                    <span className="etl-band-icon"><span className="etl-band-spinner" /></span>
                  ) : GCS_FETCH_ENABLED && isHovered && !windowLoading && !isDownloaded ? (
                    <span className="etl-band-icon"><DownloadIcon /></span>
                  ) : (
                    <span className="etl-band-text">{band.label}</span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── Controls row ── */}
      <div className="etl-controls">
        <button type="button" className="etl-step-btn" onClick={() => stepTo(-1)} title={`Back ${stepLabel}`}
          disabled={currentAvailIdx <= 0}>← {stepLabel}</button>
        <button type="button" className="etl-step-btn" onClick={() => stepTo(-bigStepCount)} title={`Back ${bigStepLabel}`}
          disabled={currentAvailIdx < bigStepCount}>← {bigStepLabel}</button>
        <input
          type="text"
          className="etl-date-input"
          value={displayedInputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onFocus={() => {
            setInputFocused(true);
            setInputValue(fmtInput(currentT));
          }}
          onBlur={() => { setInputFocused(false); commitInput(); }}
          onKeyDown={(e) => { if (e.key === "Enter") e.currentTarget.blur(); }}
          placeholder="YYYY-MM-DD HH:MM"
          aria-label="Selected timestep"
        />
        <button type="button" className="etl-step-btn" onClick={() => stepTo(bigStepCount)} title={`Forward ${bigStepLabel}`}
          disabled={currentAvailIdx >= availableList.length - bigStepCount}>{bigStepLabel} →</button>
        <button type="button" className="etl-step-btn" onClick={() => stepTo(1)} title={`Forward ${stepLabel}`}
          disabled={currentAvailIdx >= availableList.length - 1}>{stepLabel} →</button>
      </div>

    </div>
  );
}
