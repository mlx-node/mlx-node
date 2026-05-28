import * as React from 'react';
import type { ProfileLikeStats } from '../lib/display-helpers';

export type { ProfileLikeStats };

export interface TelemetryContextValue {
  stats: ProfileLikeStats | null;
  prefillTokensPerSecond: number | null;
  decodeTokensPerSecond: number | null;
  modelLine: string | null;
}

const TelemetryContext = React.createContext<TelemetryContextValue | null>(null);

export function TelemetryProvider({
  value,
  children,
}: {
  value: TelemetryContextValue;
  children: React.ReactNode;
}) {
  return <TelemetryContext.Provider value={value}>{children}</TelemetryContext.Provider>;
}

export function useTelemetry(): TelemetryContextValue {
  const ctx = React.useContext(TelemetryContext);
  if (!ctx) throw new Error('useTelemetry must be used within <TelemetryProvider>');
  return ctx;
}
