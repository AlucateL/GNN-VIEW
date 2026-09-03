"use client";

import React, { useState } from "react";
import { varLabel, stepLabel } from "../../lib/varLabels";

// ── Types ─────────────────────────────────────────────────────────────────────

export type SourceRankingEntry = {
  variable: string;
  level: number | null;
  score: number;
  score_prev_prev: number;
  score_prev: number;
  percentage?: number;
};

type Props = {
  sourceRanking: SourceRankingEntry[] | null;
  pAKey: string;
  pBKey: string;
  compareMode: "timestamps" | "variables";
  method: "ixg" | "ig";
  tsToggle?: "t-12h" | "t-6h";
  /** Hours between window timestamps, so the two input steps are labelled from the model's own
   *  cadence — t−6h/t−12h for ERA5, t−1d/t−2d for the ocean — rather than a hardcoded guess. */
  stepHours: number;
  loading?: boolean;
};

// ── Internal render row type ──────────────────────────────────────────────────

type RenderRow = {
  key: string;
  label: string;
  tooltip: string;
  score: number;
  barType: "total" | "t12" | "t6";
  t12Frac: number;
  t6Frac: number;
  variable: string;
  level: number | null;
};

// ── Short variable labels ─────────────────────────────────────────────────────

const VAR_SHORT: Record<string, string> = {
  geopotential:             "Geopot.",
  specific_humidity:        "Sp.Hum.",
  temperature:              "Temp.",
  u_component_of_wind:      "U-Wind",
  v_component_of_wind:      "V-Wind",
  vertical_velocity:        "Vert.Vel.",
  "2m_temperature":         "2m Temp",
  "10m_u_component_of_wind":"10m U",
  "10m_v_component_of_wind":"10m V",
  mean_sea_level_pressure:  "MSLP",
  total_precipitation_6hr:  "Precip.",
};

// ── Contextual labels (§6.9) ──────────────────────────────────────────────────

const IXG_LABEL =
  "Each bar shows a feature's signed Input×Gradient contribution, " +
  "normalized to percentages for comparison across features. Unlike " +
  "Integrated Gradients, IxG does not satisfy completeness, so the totals " +
  "do not decompose f(x)−f(x′) — only relative rankings are meaningful. " +
  "IxG can also miss important features when the gradient saturates " +
  "or when the input value is zero.";

const IG_LABEL =
  "Each bar shows a feature's contribution to the prediction shift f(x)−f(x′) " +
  " from the global-mean baseline x′ to the selected input x, so contributions " +
  "sum to 100% after normalization. Bars are stacked by the two input timesteps, " +
  "showing how much of each feature's total attribution originates from each step.";
// ── Helpers ───────────────────────────────────────────────────────────────────

const entryKey = (e: SourceRankingEntry): string =>
  e.level !== null ? `${e.variable}@${e.level}` : e.variable;

const baseShortLabel = (e: SourceRankingEntry): string => {
  // VAR_SHORT covers ERA5 only; an ocean channel fell through to the raw identifier, so this
  // panel showed "thetao 47m" while every other panel said "Temperature (47m)".
  const base = VAR_SHORT[e.variable] ?? varLabel(e.variable);
  return e.level !== null ? `${base}@${e.level}` : base;
};

const shortLabel = (e: SourceRankingEntry): string => {
  const s = baseShortLabel(e);
  return s.length > 18 ? s.slice(0, 15) + "…" : s;
};

const fullLabel = (e: SourceRankingEntry): string =>
  // "hPa" only ever applies to an ERA5 pressure level; ocean channels carry level === null
  // because their depth is already part of the name.
  e.level !== null ? `${e.variable}@${e.level}hPa` : varLabel(e.variable);

// ── Component ─────────────────────────────────────────────────────────────────

export default function ExplainRankingSidebar({
  sourceRanking,
  pAKey,
  pBKey,
  compareMode,
  method,
  tsToggle,
  stepHours,
  loading = false,
}: Props): React.ReactElement {
  const contextLabel = method === "ig" ? IG_LABEL : IXG_LABEL;

  // Aggregation toggles: defaults depend on compareMode
  const [timeAgg, setTimeAgg] = useState(() => compareMode === "variables");
  const [levelAgg, setLevelAgg] = useState(true);
  // Explanation text is collapsible to free up vertical space for the ranked
  // rows. Ephemeral (local state) — resets to expanded on every Explain mode
  // entry, matching the sidebar's own open/closed state (§8: not persisted).
  const [explanationCollapsed, setExplanationCollapsed] = useState(false);

  // ── Step 1: Level aggregation — group by variable, summing scores ──────────
  let baseEntries: SourceRankingEntry[] = sourceRanking ?? [];
  if (levelAgg && sourceRanking) {
    const grouped = new Map<string, SourceRankingEntry>();
    for (const e of sourceRanking) {
      const existing = grouped.get(e.variable);
      if (existing) {
        existing.score += e.score;
        existing.score_prev_prev += e.score_prev_prev;
        existing.score_prev += e.score_prev;
      } else {
        grouped.set(e.variable, { ...e, level: null });
      }
    }
    baseEntries = Array.from(grouped.values()).sort((a, b) => b.score - a.score);
  }

  // ── Step 2: Time aggregation ──────────────────────────────────────────────
  // ON  → one row per feature, bar shows proportional t-12h / t-6h split
  // OFF → each feature expands into two rows: one t-12h, one t-6h
  const renderRows: RenderRow[] = [];

  for (const e of baseEntries) {
    const sl = shortLabel(e);
    const fl = fullLabel(e);
    const key = entryKey(e);

    if (timeAgg) {
      renderRows.push({
        key: `${key}-total`,
        label: sl,
        tooltip: fl,
        score: e.score,
        barType: "total",
        t12Frac: e.score > 0 ? e.score_prev_prev / e.score : 0.5,
        t6Frac:  e.score > 0 ? e.score_prev       / e.score : 0.5,
        variable: e.variable,
        level: e.level,
      });
    } else {
      // Truncate base label to leave room for the timestep suffix
      const base = baseShortLabel(e);
      const truncBase = base.length > 11 ? base.slice(0, 9) + "…" : base;

      renderRows.push({
        key: `${key}-t12`,
        label: `${truncBase} ${stepLabel(stepHours, 2)}`,
        tooltip: `${fl} (${stepLabel(stepHours, 2)} input)`,
        score: e.score_prev_prev,
        barType: "t12",
        t12Frac: 1,
        t6Frac: 0,
        variable: e.variable,
        level: e.level,
      });
      renderRows.push({
        key: `${key}-t6`,
        label: `${truncBase} ${stepLabel(stepHours, 1)}`,
        tooltip: `${fl} (${stepLabel(stepHours, 1)} input)`,
        score: e.score_prev,
        barType: "t6",
        t12Frac: 0,
        t6Frac: 1,
        variable: e.variable,
        level: e.level,
      });
    }
  }

  // When time aggregation is OFF, each entry expands into two rows whose
  // displayed scores are per-timestep — re-sort by that displayed score so
  // bar lengths decrease monotonically top-to-bottom.
  if (!timeAgg) {
    renderRows.sort((a, b) => b.score - a.score);
  }

  // Percentage normalisation over all visible rows
  const totalScore = renderRows.reduce((s, r) => s + r.score, 0);

  return (
    <div className="ex-ranking-sidebar">

      {/* ── Contextual label (~60px, collapsible) ────────────────────── */}
      <div className={`ex-ranking-label${explanationCollapsed ? " ex-ranking-label--collapsed" : ""}`}>
        <button
          type="button"
          className="ex-ranking-label-toggle"
          title={explanationCollapsed ? "Show explanation" : "Hide explanation"}
          onClick={() => setExplanationCollapsed((v) => !v)}
        >
          <span aria-hidden="true">{explanationCollapsed ? "▸" : "▾"}</span>
          {explanationCollapsed && <span className="ex-ranking-label-toggle-text">Explanation</span>}
        </button>
        {!explanationCollapsed && (
          <div className="ex-ranking-label-text">
            {contextLabel}
            <div className="ex-ranking-score-note">Scores are Σ|attribution| per feature across all grid nodes.</div>
          </div>
        )}
      </div>

      {/* ── Aggregation toggles ──────────────────────────────────────── */}
      <div className="ex-ranking-agg-row">
        <span className="ex-ranking-agg-label">Aggregate:</span>
        <button
          type="button"
          title="Aggregate attributions across all input timepoints"
          className={`ex-ranking-agg-pill${timeAgg ? " active" : ""}`}
          onClick={() => setTimeAgg((v) => !v)}
        >
          time
        </button>
        <button
          type="button"
          title="Aggregate attributions across all pressure levels"
          className={`ex-ranking-agg-pill${levelAgg ? " active" : ""}`}
          onClick={() => setLevelAgg((v) => !v)}
        >
          level
        </button>
      </div>

      {/* ── Ranked rows (scrollable) ─────────────────────────────────── */}
      <div className="ex-ranking-rows">
        {!sourceRanking ? (
          loading ? (
            <div className="ex-ranking-empty" style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
              <div className="ex-spinner ex-spinner--sm" />
              <span>Computing {method === "ig" ? "Integrated Gradients" : "Input × Gradient"}…</span>
            </div>
          ) : (
            <div className="ex-ranking-empty">No attribution data</div>
          )
        ) : (
          renderRows.map((row) => {
            // Highlight matching: when level-agg is ON, match by variable prefix
            let matchA = levelAgg
              ? pAKey === row.variable || pAKey.startsWith(`${row.variable}@`)
              : (row.level !== null ? `${row.variable}@${row.level}` : row.variable) === pAKey;
            let matchB = levelAgg
              ? pBKey === row.variable || pBKey.startsWith(`${row.variable}@`)
              : (row.level !== null ? `${row.variable}@${row.level}` : row.variable) === pBKey;

            // When time aggregation is OFF, only highlight rows that match the current time toggle.
            // In timestamps mode both panels (t-12h + t-6h) are visible side-by-side, so each
            // timestep row keeps its highlight with a per-timestep colour below.
            if (!timeAgg && tsToggle && compareMode !== "timestamps") {
              const rowIsT12 = row.barType === "t12";
              const toggleIsT12 = tsToggle === "t-12h";
              if (rowIsT12 !== toggleIsT12) {
                matchA = false;
                matchB = false;
              }
            }

            let hlClass = "";
            if (compareMode === "timestamps") {
              if (matchA) {
                hlClass = row.barType === "t12"
                  ? " ex-ranking-row--hl-t12"
                  : " ex-ranking-row--hl-ts";
              }
            } else {
              if (matchA && matchB)    hlClass = " ex-ranking-row--hl-ab";
              else if (matchA)         hlClass = " ex-ranking-row--hl-a";
              else if (matchB)         hlClass = " ex-ranking-row--hl-b";
            }

            const pct = totalScore > 0 ? (row.score / totalScore) * 100 : 0;

            return (
              <div
                key={row.key}
                className={`ex-ranking-row${hlClass}`}
                title={row.tooltip}
              >
                <span className="ex-ranking-row-label">{row.label}</span>
                <div className="ex-ranking-bar-wrap">
                  <div className="ex-ranking-bar-filled" style={{ width: `${pct}%` }}>
                    {row.barType === "total" ? (
                      <>
                        <div className="ex-ranking-bar-t12" style={{ width: `${row.t12Frac * 100}%` }} />
                        <div className="ex-ranking-bar-t6"  style={{ width: `${row.t6Frac  * 100}%` }} />
                      </>
                    ) : row.barType === "t12" ? (
                      <div className="ex-ranking-bar-t12" style={{ width: "100%" }} />
                    ) : (
                      <div className="ex-ranking-bar-t6"  style={{ width: "100%" }} />
                    )}
                  </div>
                </div>
                <span className="ex-ranking-pct">{pct.toFixed(1)}%</span>
              </div>
            );
          })
        )}
      </div>


    </div>
  );
}
