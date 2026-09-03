"use client";

import dynamic from "next/dynamic";

type GNNExplorerNoSSRProps = {
  apiBaseUrl: string;
  cartoApiKey?: string;
};

const GNNExplorer = dynamic(() => import("./GNNExplorer"), {
  ssr: false,
  loading: () => <div style={{ padding: "1rem" }}>Loading explorer...</div>,
});

export default function GNNExplorerNoSSR({ apiBaseUrl, cartoApiKey }: GNNExplorerNoSSRProps) {
  return <GNNExplorer apiBaseUrl={apiBaseUrl} cartoApiKey={cartoApiKey} />;
}
