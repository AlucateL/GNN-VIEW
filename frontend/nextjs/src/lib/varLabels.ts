/**
 * Display names for model channels, shared by every Explain-mode panel.
 *
 * Extracted because the ocean map had been copied into two components and was about to be
 * copied into a third — and the one place that lacked it (the attribution ranking sidebar)
 * fell back to raw identifiers, showing "thetao 47m" where the rest of the UI said
 * "Temperature (47m)".
 */

/** ERA5 channels. */
export const VAR_LABELS: Record<string, string> = {
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

/** Ocean channels are named "<var>_<depth>m" (thetao_47m); zos/siconc/sithick carry no depth. */
export const OCEAN_VAR_LABELS: Record<string, string> = {
  zos: "Sea Surface Height",
  siconc: "Sea Ice Concentration",
  sithick: "Sea Ice Thickness",
  thetao: "Temperature",
  so: "Salinity",
  uo: "U Velocity",
  vo: "V Velocity",
};

const DEPTH_RE = /^([a-z]+)_(\d+)m$/;

/** Human-readable name for any channel of either domain; falls back to the raw name. */
export function varLabel(v: string): string {
  if (VAR_LABELS[v]) return VAR_LABELS[v];
  const m = DEPTH_RE.exec(v);
  if (m && OCEAN_VAR_LABELS[m[1]]) return `${OCEAN_VAR_LABELS[m[1]]} (${m[2]}m)`;
  if (OCEAN_VAR_LABELS[v]) return OCEAN_VAR_LABELS[v];
  return v.replace(/_/g, " ");
}

/** True for ocean channels — used where a pressure level would otherwise be assumed. */
export function isOceanChannel(v: string): boolean {
  const m = DEPTH_RE.exec(v);
  return (m ? !!OCEAN_VAR_LABELS[m[1]] : false) || !!OCEAN_VAR_LABELS[v];
}

/** Label for the n-th input step back, from the window's own cadence: "t−6h" for ERA5's
 *  6-hourly grid, "t−1d" for the ocean's daily one. Hardcoding either misreports the other —
 *  the ocean inputs were shown as t−6h/t−12h when they are actually a day and two days back. */
export function stepLabel(stepHours: number, stepsBack: number): string {
  const h = stepHours * stepsBack;
  return h % 24 === 0 ? `t−${h / 24}d` : `t−${h}h`;
}
