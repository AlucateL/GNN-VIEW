import React from "react";
import { NODE_GRID, NODE_MESH, EDGE_ENCODER, EDGE_DECODER, EDGE_MESH } from "./colors";
import { UPLOADS_ENABLED } from "./config";

export type TourPlacement = "center" | "right" | "left" | "bottom" | "top";

export const ONBOARD_COOKIE  = "gnn_onboarded=1";
export const EXPLORE_COOKIE  = "gnn_explore_toured=1";
export const INSPECT_COOKIE  = "gnn_inspect_toured=1";
export const EXPLAIN_COOKIE  = "gnn_explain_toured=1";

const ICON_SHADOW = "drop-shadow(2px 3px 5px rgba(0,0,0,0.42))";

export const ONBOARD_STEPS: Array<{ title: string; icon: React.ReactNode; body: string; targetId: string | null; placement: TourPlacement; mapInteractive?: boolean }> = [
  { title: "Welcome to V.I.E.W.", icon: "🌐",
    body: "V.I.E.W. - Visual Inspection and Explanation of graph Weather models - is an interactive tool to help developers understand and explain GNN-based weather models. \n This short introduction walks you through the three main functionalities.",
    targetId: null, placement: "center" },

  { title: "Explore / Inspect / Explain", icon: "🗂️",
    body: "Explore — Browse the graph structure, and get a first understanding of it. \n\nInspect — Have a more detailed look at specific parts of the graph.\n\nExplain — Trace what drives the model's predictions and errors.",
    targetId: "mode-nav", placement: "bottom" },

  { title: "Guided Tours", icon: "❓",
    body: "Each mode has its own guided tour. Click the ? button at any time to open it.",
    targetId: "help-btn", placement: "bottom" },

  { title: "You're in Explore Mode", icon: "🗺️",
    body: "Want to start the Explore tour? It walks you through understanding each visual element and interactions with the tool.",
    targetId: null, placement: "center" },
];

type ExploreStep = { title: string; icon: React.ReactNode; body: string; targetId: string | null; placement: TourPlacement; mapInteractive?: boolean };

const UPLOAD_STEPS: ExploreStep[] = UPLOADS_ENABLED ? [
  { title: "Upload Your Graph", icon: "📂",
    body: "Click 'Upload GNN' to load your model's graph. Supports neural-lam format (folder with .pt files) and Anemoi HeteroGraph format (single graph.pt file). Once uploaded, the graph appears on the map instantly.",
    targetId: "upload-sec", placement: "right" },
  { title: "Preset Graphs", icon: "🗂️",
    body: "The dropdown lists all graphs already available on the server — select one to switch to it immediately without uploading.",
    targetId: "upload-sec", placement: "right" },
] : [
  { title: "Select a Graph", icon: "🗂️",
    body: "Use the dropdown to select from the preset models available on this server. Click a graph name to load it onto the map.",
    targetId: "upload-sec", placement: "right" },
];

export const EXPLORE_STEPS: ExploreStep[] = [
  { title: "Welcome to Explore Mode", icon: "🗺️",
    body: "In Explore mode you can interactively explore the full graph structure of your GNN weather model.",
    targetId: null, placement: "center" },

  ...UPLOAD_STEPS,

  { title: "Show & Hide Layers", icon: "👁",
    body: "The left panel lists every element type in your graph. Click any toggle to show or hide that layer on the map — useful for reducing clutter in dense graphs. Now let's look at each visual encoding.",
    targetId: "panel-legend-sec", placement: "right" },

  { title: "Grid Nodes", icon: (
    <svg width="30" height="30" viewBox="0 0 10 10" style={{ filter: ICON_SHADOW }}>
      <rect x="0.5" y="0.5" width="9" height="9" rx="1" fill={NODE_GRID} stroke="rgba(0,0,0,0.3)" strokeWidth="1"/>
    </svg>
  ), body: "Amber squares mark the grid nodes — the model's input and output locations. The grid is regular and covers the entire globe. Each node represents a specific latitude-longitude cell where the model predicts weather variables.",
    targetId: "toggle-grid", placement: "right" },

  { title: "Mesh Nodes", icon: (
    <svg width="30" height="30" viewBox="0 0 10 10" style={{ filter: ICON_SHADOW }}>
      <circle cx="5" cy="5" r="4.5" fill={NODE_MESH} stroke="rgba(0,0,0,0.3)" strokeWidth="1"/>
    </svg>
  ), body: "Purple circles mark the mesh nodes — the model's internal processing locations. The tool ships with four default graphs to explore different mesh configurations.",
    targetId: "toggle-mesh", placement: "right" },

  { title: "G2M Edges — Encoder", icon: (
    <svg width="105" height="25" viewBox="0 0 42 10" style={{ filter: ICON_SHADOW }}>
      <rect x="0.5" y="0.5" width="9" height="9" rx="1" fill={NODE_GRID} stroke="rgba(0,0,0,0.3)" strokeWidth="1"/>
      <line x1="11" y1="5" x2="22" y2="5" stroke={EDGE_ENCODER} strokeWidth="1.5" strokeLinecap="round"/>
      <polyline points="22,2 26,5 22,8" fill="none" stroke={EDGE_ENCODER} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round"/>
      <circle cx="31" cy="5" r="4.5" fill={NODE_MESH} stroke="rgba(0,0,0,0.3)" strokeWidth="1"/>
    </svg>
  ), body: "Rose arrows mark the Encoder edges - also known as Grid-to-Mesh edges (G2M). They map input data from the regular grid to the processing mesh. This is the first step in the model's computation, where raw input features are projected onto the mesh for further processing.",
    targetId: "toggle-g2m", placement: "right" },

  { title: "M2M Edges — Processor", icon: (
    <svg width="105" height="25" viewBox="0 0 42 10" style={{ filter: ICON_SHADOW }}>
      <circle cx="5" cy="5" r="4.5" fill={NODE_MESH} stroke="rgba(0,0,0,0.3)" strokeWidth="1"/>
      <line x1="11" y1="5" x2="31" y2="5" stroke={EDGE_MESH} strokeWidth="1.5" strokeLinecap="round"/>
      <circle cx="37" cy="5" r="4.5" fill={NODE_MESH} stroke="rgba(0,0,0,0.3)" strokeWidth="1"/>
    </svg>
  ), body: "Purple lines mark Processor edges - also known as Mesh-to-Mesh edges (M2M). They define the icosahedral mesh's internal connectivity. Message-passing along these edges is where most of the model's computation happens.",
    targetId: "toggle-m2m", placement: "right" },

  { title: "M2G Edges — Decoder", icon: (
    <svg width="105" height="25" viewBox="0 0 42 10" style={{ filter: ICON_SHADOW }}>
      <circle cx="5" cy="5" r="4.5" fill={NODE_MESH} stroke="rgba(0,0,0,0.3)" strokeWidth="1"/>
      <line x1="11" y1="5" x2="22" y2="5" stroke={EDGE_DECODER} strokeWidth="1.5" strokeLinecap="round"/>
      <polyline points="22,2 26,5 22,8" fill="none" stroke={EDGE_DECODER} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round"/>
      <rect x="28.5" y="0.5" width="9" height="9" rx="1" fill={NODE_GRID} stroke="rgba(0,0,0,0.3)" strokeWidth="1"/>
    </svg>
  ), body: "Teal arrows mark the Decoder edges - also known as Mesh-to-Grid edges (M2G). They project processed mesh features back down to every grid node. This decoder step produces the final weather forecast at each grid location.",
    targetId: "toggle-m2g", placement: "right" },

  { title: "Hover for a Quick Preview", icon: "🖱️",
    body: "Hover over any node on the map to see a floating tooltip with its position, type, and degree. Try it now.",
    targetId: "map-wrap", placement: "center", mapInteractive: true },

  { title: "Click to Explore Neighbours", icon: "🔗",
    body: "Click any node on the map to select it. Its direct neighbours will light up, so you can trace local connectivity at a glance. Click again or elsewhere to deselect. Try it now by clicking a few different nodes.",
    targetId: "map-wrap", placement: "center", mapInteractive: true },

  { title: "Right Panel — Viewport Details", icon: "📊",
    body: "The right panel shows live Viewport Statistics — how many nodes and edges of each type are currently visible, as well as the 'Scale fade' toggle. To know more about it, hover over the question mark next to it.",
    targetId: "panel-right", placement: "left" },
];

export const INSPECT_STEPS: Array<{ title: string; icon: React.ReactNode; body: string; targetId: string | null; placement: TourPlacement }> = [
  { title: "Inspect Mode", icon: "🔍",
    body: "In Inspect Mode you can check for structural anomalies in your graph — nodes with too many or too few connections, and edges that are unusually long or short. This helps you verify that the graph structure matches your expectations and identify potential issues.",
    targetId: null, placement: "center" },

  { title: "Subgraph Selector", icon: "🗂️",
    body: "Use the chips to switch between G2M (encoder), M2M (processor), and M2G (decoder). For hierarchical models, M2M expands into per-level chips.",
    targetId: "panel-legend-sec", placement: "right" },

  { title: "Node Degree", icon: "📐",
    body: "The lollipop plots show how many edges connect to each source and target node. Drag the range sliders to set valid degree bounds — nodes outside (or equal to) the range are flagged in rose (over-degree) or teal (under-degree) on the map.",
    targetId: "inspect-deg-wrap", placement: "left" },

  { title: "Edge Length", icon: "📏",
    body: "This shows the distribution of edge lengths in km. Set the min/max sliders to define a valid range — edges outside it are flagged on the map. The plot adapts to the data: lengths that sit at a few discrete levels get a lollipop, one stem per level, and continuous lengths get a bar histogram.",
    targetId: "edge-len-section", placement: "left" },

  { title: "An Important Note", icon: "⚠️",
    body: "Mercator projection distorts distances, especially near the poles. An edge that looks long on the map might actually be shorter in real-world distance. The edge length threshold is based on true geographic distance, not pixel length - so you don't have to worry about it. \n Yet, keep this in mind when setting thresholds and interpreting flagged edges.",
    targetId: null, placement: "center" },
];

export const EXPLAIN_STEPS: Array<{ title: string; icon: React.ReactNode; body: string; targetId: string | null; placement: TourPlacement; mapInteractive?: boolean }> = [
  { title: "Welcome to Explain Mode", icon: "🔬",
    body: "Explain mode lets you interrogate individual GNN predictions — for any location, time, and variable, you can trace which input features drove the model's output.\n\nA prediction RMSE heatmap is a navigation aid: it highlights where the model struggles, helping you pick predictions worth investigating. Yet, investigating raw model outputs is also possible.",
    targetId: null, placement: "center" },

  { title: "Model Selection", icon: "👾",
    body: "Start by selecting the GNN model you want to explain. Use the dropdown to choose from the available models loaded on this server. The selected model determines which predictions and attributions you will see throughout Explain mode.",
    targetId: "ex-model-sec", placement: "right", mapInteractive: true },

  { title: "Select a Prediction", icon: "🎯",
    body: "To explain a prediction you need to specify three things:\n\n1) Time — the prediction time, selected from the timeline\n2) Variable — which weather variable and pressure level to examine\n3) Location — a grid node on the map\n\nOnce all three are set, the attribution panel unlocks.",
    targetId: null, placement: "center" },

  { title: "Timestep Selection", icon: "⏱️",
    body: "The timeline shows the single-step (6-hour lead for weather, 1-day lead for ocean) prediction RMSE of the selected model over time. \nThe left panel is a coarse-grained overview of the full period — drag or click the green window to pan it. The right panel is a zoomed-in view of that window, where you scrub to select the exact prediction time. \n\nHighlighted in green are the timestamps for which data is available to explain. \nOnly August is available now, full test set coming soon.",
    targetId: "explain-timeline", placement: "top", mapInteractive: true },

  { title: "A Note on Timestep Selection", icon: "⚠️",
    body: "The left-side timeline is not obtained by autoregressive rollout, but by comparing single-step predictions against ERA5 reanalysis at each timestamp. It serves as a guide to pick interesting timestamps to investigate, but it doesn't reflect how errors evolve over time in a real forecast. ",
    targetId: "explain-timeline", placement: "top", mapInteractive: true },

  { title: "Prediction Target", icon: "📋",
    body: "Pick what you want to look at: variable (upper-air variables require a pressure level in hPa; surface variables have a single level), and a grid node on the map. Selecting a specific variable and a single level unlocks the attribution panel.\n\nThe 'Error / Prediction' toggle at the top of this panel switches the map between two views — see the next steps. Aggregate views like 'Total Error' and 'Σ levels' only exist in Error mode, since averaging predictions across variables or pressure levels has no physical meaning.",
    targetId: "ex-target-sec", placement: "left", mapInteractive: true },

  { title: "Map: RMSE Heatmap or Raw Prediction", icon: "🗺️",
    body: "In Error mode the map shows normalised RMSE — near-white cells have low error, orange to red cells are where the model struggles most. Click any cell to set that grid node as the prediction location to explain.\n\nFlip the Error/Prediction toggle and the map switches to the model's raw prediction for the selected variable and level, in physical units (e.g. Kelvin for temperature, Pa for pressure). The colour scale is fixed per variable across the loaded window, so colours mean the same thing as you scrub the timeline.",
    targetId: "map-wrap", placement: "center", mapInteractive: true },

  { title: "Colorbar & Distribution", icon: "📊",
    body: "The colorbar reflects the current map view. In Error mode it shows the RMSE scale for the chosen variable/level, with a histogram of error magnitudes across all grid nodes — toggle 'log x'/'log y' to reveal tail structure, or 'share colorbar' to fix the scale across variables for fair comparison.\n\nIn Prediction mode the colorbar shows the physical range of the variable (e.g. ~219 K → ~279 K for temperature @ 500 hPa) and the histogram shows the distribution of predicted values across the visible grid in physical units.",
    targetId: "ex-colorbar-sec", placement: "left" },

  { title: "Selected Node", icon: "📌",
    body: "The selected node's coordinates and its current value at the chosen variable/timestep. In Error mode you see the normalised RMSE there — high values flag locations the model struggles with. In Prediction mode the same readout shows the model's raw predicted value in physical units, useful for sanity-checking whether the prediction itself is plausible before opening attribution.",
    targetId: "ex-node-sec", placement: "left" },

  { title: "Raw Prediction View", icon: "📈",
    body: "The Prediction toggle lets you browse the model's raw outputs in physical units across any variable, level, and time — without the normalisation layer that error mode uses. It's complementary to the RMSE heatmap: errors tell you where the model is wrong, the prediction view tells you what it actually said.",
    targetId: null, placement: "center" },

  { title: "Open Attribution", icon: "🧠",
    body: "With time, variable, and location all set, click 'Open attribution' to compute attribution scores for the selected prediction. \n\nAttribution-based explainability methods - as the one used by this tool - explain a model's prediction by assigning a relevance score to each input feature. They address the question: how much did each input contribute to this specific output?",
    targetId: "ex-panel-footer", placement: "left" },

  { title: "Attribution Heatmap", icon: "🌡️",
    body: "The attribution drawer shows a spatial map of input-feature contributions. Each cell answers: did this input location pushed model prediction above (rose) or below (teal) a baseline. Intensity shows how strongly. The baseline prediction is the model output when its input is the climatological mean for each field. Zoom and pan to explore the spatial structure.",
    targetId: "explain-drawer-root", placement: "top" },

  { title: "Variable Ranking", icon: "📋",
    body: "The sidebar ranks all input variables by their total attribution magnitude. The top entries are the features that mattered most for this prediction. Bar segments show the two input timesteps separately, revealing which lag is more influential — six-hourly for the weather models, daily for the ocean.",
    targetId: "ex-ranking-wrap", placement: "left" },

  { title: "Compare Mode", icon: "⚖️",
    body: "Toggle 'Compare' to place two attribution sources side by side. In Variables mode, compare two different input variables. In Timestamps mode, compare the attributions of the two input timesteps. The model, indeed, takes two input timesteps (t-12h,t-6h) to predict the target time t.",
    targetId: "ex-drawer-compare", placement: "bottom" },

  { title: "Replay the Guide", icon: "❓",
    body: "Click the ? button at any time to reopen this guide.",
    targetId: "explain-help-btn", placement: "bottom" },
];
