# Frontend - GNN Graph Explorer

This frontend is a Next.js + React + TypeScript application that renders the GNN visualizer UI and communicates with the backend API.

## Tech Stack

- Next.js (App Router)
- React
- TypeScript
- Leaflet (base map + minimap)
- deck.gl (GPU rendering of nodes/edges)
- D3 (histograms)
- Yarn

## Prerequisites

- Node.js 20+
- Yarn 1.22+ (or compatible modern Yarn)
- Backend API running and reachable at `http://localhost:8080` for local development
- In Kubernetes, set `BACKEND_URL` on the frontend pod; the app reads that at runtime and falls back to localhost only when it is not provided

If your backend runs on a different URL outside Kubernetes, set `BACKEND_URL` before starting the app.

## Install

```bash
yarn install
```

## Development

```bash
yarn dev
```

Open `http://localhost:3000`.

## Build

```bash
yarn build
```

## Production Start

```bash
yarn start
```

## Lint

```bash
yarn lint
```

## Project Structure

```text
app/
	layout.tsx
	page.tsx
	globals.css
src/
	components/
		gnn/
			GNNExplorer.tsx
			GNNExplorer.css
public/
	GNN_Real_Data_Explorer.html   # legacy reference implementation
```

## Notes

- The main visualizer entry point is `src/components/gnn/GNNExplorer.tsx`.
- The UI/interaction model mirrors the original standalone HTML implementation, now organized as a React component.
