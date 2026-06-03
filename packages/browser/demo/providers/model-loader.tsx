import * as React from 'react';

import type { LoadingProgress } from '../components/loading/Loading';
import type { ModelLoaderStatus } from '../lib/model-loader-state';

export type { LoadingProgress } from '../components/loading/Loading';

export type { ModelLoaderStatus } from '../lib/model-loader-state';

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
  // Monotonic counter bumped each time a load (full OR device-only) STARTS.
  // Exposed for the consent layer's DEVICE mode: a device-only bring-up reads as
  // model-status `idle`, so a device demo falls back to this to know its device
  // init is in flight. See selectConsentLayerState.
  loadKickoff: number;
  // Imperative callbacks needed by route components:
  kickoffLoad: () => void;
  // Bring the WebGPU device up WITHOUT downloading the model (for DEVICE_ONLY
  // chapters). Idempotent; a no-op when a full load is already in-flight/done.
  kickoffDeviceOnly: () => void;
  // Clear a surfaced load error and return the loader to the neutral 'idle'
  // state. Guarded to the error state only (a no-op otherwise), so it never
  // aborts an in-flight load or drops a ready model. Used to drop a stale
  // GLOBAL errorBanner when navigating to a fresh surface the user hasn't acted
  // on (e.g. a newly opened chapter).
  clearLoadError: () => void;
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
