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
  // True once the WASM runtime + WebGPU device are up, regardless of whether
  // the big model is loaded. DEVICE_ONLY chapters (e.g. Training) gate on this
  // instead of `status === 'ready'`. A full-model 'ready' implies this too.
  deviceReady: boolean;
  // Imperative callbacks needed by route components:
  kickoffLoad: () => void;
  // Bring the WebGPU device up WITHOUT downloading the model (for DEVICE_ONLY
  // chapters). Idempotent; a no-op when a full load is already in-flight/done.
  kickoffDeviceOnly: () => void;
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
