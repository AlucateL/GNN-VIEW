import GNNExplorerNoSSR from "@/src/components/gnn/GNNExplorerNoSSR";

export const dynamic = "force-dynamic";

const DEFAULT_BACKEND_URL = "http://localhost:8080";

const normalizeBackendUrl = (rawValue?: string): string => {
  const value = rawValue?.trim();
  if (!value) return DEFAULT_BACKEND_URL;

  const hasScheme = /^[a-z][a-z0-9+.-]*:\/\//i.test(value);
  const isLocalHost = /^(localhost|127(?:\.\d{1,3}){3}|\[::1\])(?::\d+)?(?:\/|$)/i.test(value);
  const withScheme = hasScheme ? value : `${isLocalHost ? "http" : "https"}://${value}`;

  try {
    const normalized = new URL(withScheme).toString().replace(/\/+$/, "");
    return normalized || DEFAULT_BACKEND_URL;
  } catch {
    return DEFAULT_BACKEND_URL;
  }
};

const getBackendUrl = (): string => {
  return normalizeBackendUrl(process.env.BACKEND_URL ?? process.env.NEXT_PUBLIC_BACKEND_URL);
};

const getCartoApiKey = (): string => {
  return (process.env.CARTO_API_KEY ?? process.env.NEXT_PUBLIC_CARTO_API_KEY ?? "").trim();
};

export default function Home() {
  return <GNNExplorerNoSSR apiBaseUrl={getBackendUrl()} cartoApiKey={getCartoApiKey()} />;
}
