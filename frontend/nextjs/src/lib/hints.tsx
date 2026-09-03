import React from "react";

// ── LeftPanel ────────────────────────────────────────────────────────────────

export const HINT_GRID_NODES = (
  <>Grid nodes are the model&apos;s lat/lon grid points — the ERA5 grid for the weather models, the ocean analysis grid for SeaCast. On the map, each node is drawn as a small square — one pixel per grid location. The G2M encoder reads features from every grid node and projects them onto the mesh; the M2G decoder projects processed mesh features back to each grid location to produce the final forecast.</>
);

// ── GNNExplorer (Inspect panel) ──────────────────────────────────────────────

export const HINT_SCALE_FADE = (
  <>When enabled, short edges gradually fade out as you zoom out, reducing visual clutter in dense regions. Only longer-range edges stay visible at low zoom levels, making the overall graph structure easier to read.</>
);

export const HINT_SRC_DEGREE = (
  <>Node degree is a discrete count (how many edges connect to a node). When there are few enough distinct values it is drawn as a <strong>lollipop plot</strong>, each stem one exact degree and its frequency. Some subgraphs span hundreds of distinct degrees — ocean M2G source degree runs from 6 to 4694 — and stems become unreadable there, so the plot falls back to a <strong>bar histogram</strong>. The slider below sets the valid range; nodes outside (or equal to) it are flagged as degree violations on the map.</>
);

export const HINT_TGT_DEGREE = (
  <>Target node degree counts how many incoming edges each target node receives. Like source degree it is discrete, so it uses a <strong>lollipop plot</strong> when the distinct values are few, and a <strong>bar histogram</strong> when they are not. For intra-level (M2M) subgraphs, source and target degree distributions are identical — one note is shown in that case.</>
);

export const HINT_EDGE_LENGTH = (
  <>The plot adapts to the data: when edge lengths fall into a few tight, well-separated levels — as they do for an algorithmically constructed processor (M2M) mesh — it draws a <strong>lollipop</strong> marking those exact levels. When the lengths form a continuous spread, as encoder (G2M) and decoder (M2G) edges do and as some irregular meshes do at every level, it draws a <strong>bar histogram</strong> instead. The slider sets the valid range; edges outside (or equal to) it are flagged as length violations on the map.</>
);

// ── ExplainDrawer ────────────────────────────────────────────────────────────

export const HINT_IXG_METHOD = (
  <>Multiplies each input feature by its gradient at the target node (Input × Gradient). Fast approximation good for quick exploration. Does not satisfy completeness — only relative rankings are meaningful.</>
);

export const HINT_IG_METHOD = (
  <>Integrates gradients along a straight path from the baseline (zero in normalised space = climatological mean per feature) to the actual input. Satisfies completeness: contributions sum to the total prediction shift f(x)−f(x′). Use for rigorous analysis.</>
);

// ── ExplainRightPanel ────────────────────────────────────────────────────────

export const HINT_COLORBAR = (
  <>Each coloured cell on the map corresponds to one grid node, sized to that model&apos;s grid spacing. Its colour encodes the normalised prediction error at that location — white means near-zero error, orange to red means progressively larger error relative to the current scale.</>
);

export const HINT_COLORBAR_PREDICTION = (
  <>Each coloured cell on the map corresponds to one grid node, sized to that model&apos;s grid spacing. Its colour encodes the prediction in the physical scale of the selected variable at that location.</>
);

export const HINT_SHARE_COLORBAR = (
  <>When enabled, the scale is fixed to the broadest range across every channel the model predicts, so the same colour always means the same absolute error — useful for fair comparison across panels. When disabled, each channel uses its own scale for maximum local contrast.</>
);

export const HINT_TOTAL_ERROR = (
  <>Average RMSE error across every channel the model predicts at each grid node, giving a single per-location score. Use it for geographic navigation. Select a specific variable below to unlock the attribution view.</>
);

// ── ExplainTimeline ──────────────────────────────────────────────────────────

export const HINT_TIMELINE_OVERVIEW = (
  <>
    Each point on this curve is the normalised Root Mean Squared Error (RMSE) for one prediction timestep — lower is better. The demo ships with the month of August available (highlighted in green).
  </>
);
