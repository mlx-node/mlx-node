import * as React from 'react';
import type { LoadingProgress } from '../components/loading/Loading';

export type { LoadingProgress } from '../components/loading/Loading';

export type ModelLoaderStatus = 'idle' | 'loading' | 'ready' | 'error';

export interface ModelLoaderContextValue {
  status: ModelLoaderStatus;
  loadingText: string;
  loadingProgress: LoadingProgress | null; // or null for indeterminate
  modelLine: string | null; // display name
  errorBanner: string | null;
  hostedModelAvailable: boolean | null; // null = probing
  // Imperative callbacks needed by route components:
  kickoffLoad: () => void;
  resetForModelLoad: (label?: string) => void;
}

const ModelLoaderContext = React.createContext<ModelLoaderContextValue | null>(null);

export function ModelLoaderProvider({
  value,
  children,
}: {
  value: ModelLoaderContextValue;
  children: React.ReactNode;
}) {
  return <ModelLoaderContext.Provider value={value}>{children}</ModelLoaderContext.Provider>;
}

export function useModelLoader(): ModelLoaderContextValue {
  const ctx = React.useContext(ModelLoaderContext);
  if (!ctx) throw new Error('useModelLoader must be used within <ModelLoaderProvider>');
  return ctx;
}
