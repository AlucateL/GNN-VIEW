import currentPalette from './colors.json';
// eslint-disable-next-line @typescript-eslint/no-unused-vars
import prevPalette from './colors_prev.json';

// ── Palette switch — swap which line is commented out ────────────────────────
const palette = currentPalette;
// const palette = prevPalette;

// Map node colors
export const NODE_GRID          = palette.mapNodes.grid;
export const NODE_MESH          = palette.mapNodes.mesh;

// Map edge colors
export const EDGE_ENCODER       = palette.mapEdges.encoder;   // G2M
export const EDGE_DECODER       = palette.mapEdges.decoder;   // M2G
export const EDGE_MESH          = palette.mapEdges.mesh;      // M2M / inter-level

// Violation state colors
export const VIOL_ABOVE         = palette.violation.above;    // above upper threshold
export const VIOL_BELOW         = palette.violation.below;    // below lower threshold
export const VIOL_NEUTRAL       = palette.violation.neutral;  // within normal range

// Interactive UI
export const ACCENT             = palette.interactive.accent;
export const MINIMAP_ACCENT     = palette.interactive.minimapAccent;

// Base text
export const TEXT               = palette.ui.text;
export const TEXT_MUTED         = palette.ui.textMuted;
export const TEXT_DIM           = palette.ui.textDim;

// Explain mode
export const EXPLAIN_T12H       = palette.explain.t12h;       // T-12h horizon bar / chip
export const EXPLAIN_T6H        = palette.explain.t6h;        // T-6h horizon bar / chip
export const ATTR_COLD          = palette.explain.attrCold;
export const ATTR_COLD_MID      = palette.explain.attrColdMid;
export const ATTR_HOT_MID       = palette.explain.attrHotMid;
export const ATTR_HOT           = palette.explain.attrHot;
export const TIMELINE_AVAILABLE = palette.explain.timelineAvailable;

// Error heatmap colormap stops [position, r, g, b]
function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ];
}
export const HEATMAP_STOPS: [number, number, number, number][] = [
  [0.000, ...hexToRgb(palette.errorHeatmap.stop0)],
  [0.010, ...hexToRgb(palette.errorHeatmap.stop1)],
  [0.333, ...hexToRgb(palette.errorHeatmap.stop2)],
  [0.667, ...hexToRgb(palette.errorHeatmap.stop3)],
  [0.833, ...hexToRgb(palette.errorHeatmap.stop4)],
  [1.000, ...hexToRgb(palette.errorHeatmap.stop5)],
];
export const HEATMAP_NO_DATA    = hexToRgb(palette.errorHeatmap.noData);

// Prediction colormap — sequential cool→warm, distinguishes "showing pred values"
// mode from the warm error heatmap. Stops are evenly spaced so a value mapped to
// [0, 1] (via physical_scales[var@level] min/max) traverses the full palette.
export const PREDICTION_STOPS: [number, number, number, number][] = [
  [0.000, ...hexToRgb(palette.predictionHeatmap.stop0)],
  [0.200, ...hexToRgb(palette.predictionHeatmap.stop1)],
  [0.400, ...hexToRgb(palette.predictionHeatmap.stop2)],
  [0.600, ...hexToRgb(palette.predictionHeatmap.stop3)],
  [0.800, ...hexToRgb(palette.predictionHeatmap.stop4)],
  [1.000, ...hexToRgb(palette.predictionHeatmap.stop5)],
];
export const PREDICTION_NO_DATA = hexToRgb(palette.predictionHeatmap.noData);

// Per-variable prediction colormaps (perceptually uniform, colorblind-safe).
// All 5-stop; diverging maps have the neutral stop exactly at t=0.5.

// cmocean balance — diverging blue–white–red.
// Used for U/V wind, 10m U/V wind, vertical velocity: symmetric [-A, +A].
export const DIVERGING_WIND_STOPS: [number, number, number, number][] = [
  [0.00, ...hexToRgb(palette.divergingWind.stop0)],
  [0.25, ...hexToRgb(palette.divergingWind.stop1)],
  [0.50, ...hexToRgb(palette.divergingWind.stop2)],
  [0.75, ...hexToRgb(palette.divergingWind.stop3)],
  [1.00, ...hexToRgb(palette.divergingWind.stop4)],
];
export const DIVERGING_WIND_NO_DATA = hexToRgb(palette.divergingWind.noData);

// magma (perceptually-uniform, colorblind-safe): black→purple→rose→orange→pale-yellow.
// 12 stops evenly spaced across [0,1]. Used for temperature and 2m temperature.
export const THERMAL_FIELD_STOPS: [number, number, number, number][] = [
  [ 0/11, ...hexToRgb(palette.thermalField.stop0)],
  [ 1/11, ...hexToRgb(palette.thermalField.stop1)],
  [ 2/11, ...hexToRgb(palette.thermalField.stop2)],
  [ 3/11, ...hexToRgb(palette.thermalField.stop3)],
  [ 4/11, ...hexToRgb(palette.thermalField.stop4)],
  [ 5/11, ...hexToRgb(palette.thermalField.stop5)],
  [ 6/11, ...hexToRgb(palette.thermalField.stop6)],
  [ 7/11, ...hexToRgb(palette.thermalField.stop7)],
  [ 8/11, ...hexToRgb(palette.thermalField.stop8)],
  [ 9/11, ...hexToRgb(palette.thermalField.stop9)],
  [10/11, ...hexToRgb(palette.thermalField.stop10)],
  [11/11, ...hexToRgb(palette.thermalField.stop11)],
];
export const THERMAL_FIELD_NO_DATA = hexToRgb(palette.thermalField.noData);

// viridis — sequential purple→teal→green→yellow.
// Used for geopotential and MSLP.
export const DENSE_FIELD_STOPS: [number, number, number, number][] = [
  [0.00, ...hexToRgb(palette.denseField.stop0)],
  [0.25, ...hexToRgb(palette.denseField.stop1)],
  [0.50, ...hexToRgb(palette.denseField.stop2)],
  [0.75, ...hexToRgb(palette.denseField.stop3)],
  [1.00, ...hexToRgb(palette.denseField.stop4)],
];
export const DENSE_FIELD_NO_DATA = hexToRgb(palette.denseField.noData);

// cmocean haline — sequential purple→teal→green→yellow.
// Used for specific humidity (anchored so 0 g/kg = darkest end).
export const HALINE_FIELD_STOPS: [number, number, number, number][] = [
  [0.00, ...hexToRgb(palette.halineField.stop0)],
  [0.25, ...hexToRgb(palette.halineField.stop1)],
  [0.50, ...hexToRgb(palette.halineField.stop2)],
  [0.75, ...hexToRgb(palette.halineField.stop3)],
  [1.00, ...hexToRgb(palette.halineField.stop4)],
];
export const HALINE_FIELD_NO_DATA = hexToRgb(palette.halineField.noData);

// white→dark blue — sequential, sqrt-scaled.
// Used for precipitation: zero maps to transparent/white; log-heavy tail is compressed.
export const RAIN_FIELD_STOPS: [number, number, number, number][] = [
  [0.00, ...hexToRgb(palette.rainField.stop0)],
  [0.25, ...hexToRgb(palette.rainField.stop1)],
  [0.50, ...hexToRgb(palette.rainField.stop2)],
  [0.75, ...hexToRgb(palette.rainField.stop3)],
  [1.00, ...hexToRgb(palette.rainField.stop4)],
];
export const RAIN_FIELD_NO_DATA = hexToRgb(palette.rainField.noData);

// dark navy→white — sequential, matches the conventional sea-ice colormap
// (cmocean "ice"): open water (0% concentration / no thickness) is dark blue,
// full ice cover is white. Used for siconc and sithick.
export const ICE_FIELD_STOPS: [number, number, number, number][] = [
  [0.00, ...hexToRgb(palette.iceField.stop0)],
  [0.25, ...hexToRgb(palette.iceField.stop1)],
  [0.50, ...hexToRgb(palette.iceField.stop2)],
  [0.75, ...hexToRgb(palette.iceField.stop3)],
  [1.00, ...hexToRgb(palette.iceField.stop4)],
];
export const ICE_FIELD_NO_DATA = hexToRgb(palette.iceField.noData);

export { palette };

export function buildCssVars(): string {
  const { mapNodes, mapEdges, violation, interactive, explain, errorHeatmap, ui, dark } = palette;
  return (
    ':root{' +
    `--bg:${ui.background};--bg-panel:${ui.panelBackground};--bg-card:${ui.cardBackground};` +
    `--border:${ui.border};--border-light:${ui.borderSubtle};` +
    `--text:${ui.text};--text-muted:${ui.textMuted};--text-dim:${ui.textDim};` +
    `--grid-color:${mapNodes.grid};--mesh-color:${mapNodes.mesh};` +
    `--g2m-color:${mapEdges.encoder};--m2g-color:${mapEdges.decoder};` +
    `--viol-hi:${violation.above};--viol-lo:${violation.below};--viol-grey:${violation.neutral};` +
    `--blue-viol:${violation.below};--blue-viol-light:rgba(58,172,190,0.15);` +
    `--red-light:rgba(200,88,120,0.15);` +
    `--accent:${interactive.accent};--minimap-accent:${interactive.minimapAccent};` +
    `--rank-t12h:${explain.t12h};--rank-t6h:${explain.t6h};` +
    `--src-a:${explain.sourceA};--src-b:${explain.sourceB};` +
    `--timeline-available:${explain.timelineAvailable};` +
    // Peak-error markers in the Selected Variable list. Reuses the hot end of the
    // error heatmap so the markers read in the same language as the map:
    // rank 1 red -> rank 2 orange -> rank 3 amber. --err-peak is the level-profile
    // ring, which carries intensity rather than rank.
    `--err-peak:${errorHeatmap.stop4};` +
    `--err-rank1:${errorHeatmap.stop5};--err-rank2:${errorHeatmap.stop4};--err-rank3:${errorHeatmap.stop2};` +
    '}' +
    '#app.dark{' +
    `--bg:${dark.background};--bg-panel:${dark.panelBackground};--bg-card:${dark.cardBackground};` +
    `--border:${dark.border};--border-light:${dark.borderSubtle};` +
    `--text:${dark.text};--text-muted:${dark.textMuted};--text-dim:${dark.textDim};` +
    `--red-light:rgba(200,88,120,0.12);--blue-viol-light:rgba(58,172,190,0.12);` +
    `--rank-t12h:${dark.t12h};--rank-t6h:${dark.t6h};` +
    `--src-a:${dark.sourceA};--src-b:${dark.sourceB};` +
    `--timeline-available:${dark.timelineAvailable};` +
    `--err-peak:${errorHeatmap.stop4};` +
    `--err-rank1:${errorHeatmap.stop5};--err-rank2:${errorHeatmap.stop4};--err-rank3:${errorHeatmap.stop2};` +
    '}'
  );
}
