/**
 * Set to false to disable GCS dataset fetching (e.g. for ECMWF educational deployment).
 * When false: month-band download buttons are hidden and band clicks are no-ops.
 */
export const GCS_FETCH_ENABLED = false;

/**
 * Set to false to disable upload buttons globally (GNN upload in Explore mode,
 * model upload in Explain mode). Buttons remain visible but are grayed out.
 */
export const UPLOADS_ENABLED = false;
