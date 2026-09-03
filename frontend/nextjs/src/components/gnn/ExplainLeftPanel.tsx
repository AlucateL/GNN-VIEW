"use client";

import React from "react";
import { UPLOADS_ENABLED } from "../../lib/config";

export type ExplainModelOption = {
  model_id: string;
  name: string;
  description?: string | null;
  domain: string;
};

// Dataset a domain is backed by. Shown as a badge next to the picker — the only
// "which dataset am I looking at" indicator, and derived from the selected model rather
// than configured separately so an uploaded model labels itself correctly.
const DATASET_LABELS: Record<string, string> = {
  era5: "ERA5",
  ocean: "OceanBench",
};

export type ExplainLeftPanelProps = {
  modelSwitching: string | null;
  /** Full model catalogue from GET /models — replaces the old hardcoded preset list. */
  models: ExplainModelOption[];
  selectedModelId: string;
  /** Receives a DB model_id (not a preset key): the only path that is domain-aware. */
  onModelSelect: (modelId: string) => void;
  onModelUpload?: () => void;
};

export default function ExplainLeftPanel({
  modelSwitching,
  models,
  selectedModelId,
  onModelSelect,
  onModelUpload,
}: ExplainLeftPanelProps): React.ReactElement {
  const activeModel =
    models.find((m) => m.model_id === selectedModelId) ?? models[0] ?? null;

  // Blank when the stored description just repeats the name, which is what seed_db writes
  // for uploaded models -- showing the name twice adds nothing.
  const stored = (activeModel?.description ?? "").trim();
  const description =
    stored.toLowerCase() === (activeModel?.name ?? "").toLowerCase() ? "" : stored;

  const datasetLabel = activeModel
    ? DATASET_LABELS[activeModel.domain] ?? activeModel.domain
    : "";

  return (
    <div className="sb-sec" id="ex-model-sec">
      <div className="h-upload-row">
        {UPLOADS_ENABLED ? (
          <button
            className="upload-btn"
            type="button"
            onClick={onModelUpload}
            title="Upload a model file"
          >
            Upload Model
          </button>
        ) : (
          <button
            className="upload-btn"
            type="button"
            disabled
            title="Model uploads disabled"
            style={{ cursor: "not-allowed" }}
          >
            Upload Model
          </button>
        )}
        <select
          className="graph-select"
          value={activeModel?.model_id ?? ""}
          disabled={!!modelSwitching || models.length === 0}
          onChange={(e) => { onModelSelect(e.target.value); }}
          aria-label="Select model"
        >
          {models.length === 0 && <option value="">No models available</option>}
          {models.map((model) => (
            <option key={model.model_id} value={model.model_id}>
              {modelSwitching === model.model_id ? "Loading…" : model.name}
            </option>
          ))}
        </select>
        {modelSwitching && <span className={`loading-badge active`} style={{ marginLeft: 6 }} />}
        {/* Dataset is derived from the selected model, not an independent control —
            this badge is the only "which dataset am I looking at" indicator. */}
        {datasetLabel && (
          <span className="ex-dataset-badge" title={`Backed by the ${datasetLabel} dataset`}>
            {datasetLabel}
          </span>
        )}
      </div>
      <div className="graph-name-row ex-model-desc">
        {description}
      </div>
    </div>
  );
}
