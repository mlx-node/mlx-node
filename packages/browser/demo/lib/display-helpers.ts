// screen-state.ts — surviving helpers after Phase 2.D cleanup.
// The screen-state machine (ScreenState, ScreenEvent, reduceScreen,
// parseScreenFromUrl, screenToUrlQuery) has been deleted — TanStack Router
// now owns all navigation state.  Only the pure display helpers that other
// components import remain here.

export type ProfileLikeStats = {
  numTokens?: number;
  gpuRpcCount?: number;
  poolHits?: number;
  poolMisses?: number;
};

export type TelemetryView = {
  decodeTokPerSec: string;
  prefillTokPerSec: string;
  gpuRpc: string;
  pool: string;
  modelLine: string;
};

function formatTokRate(label: string, value: number | null | undefined): string {
  const rate = value ?? 0;
  return rate > 0 ? `${label} ${Math.round(rate)} tok/s` : `${label} —`;
}

/**
 * Formats the telemetry footer strip values.
 *
 * `stats` comes from the demo's ProfileStats reducer (worker counters).
 * `prefillTokensPerSecond` and `decodeTokensPerSecond` come from
 * ChatResult.performance (chat-stream API).
 * The two sources are merged at the call site (see chat screen wiring).
 */
export function formatTelemetry(
  stats: ProfileLikeStats | null | undefined,
  prefillTokensPerSecond: number | null | undefined,
  decodeTokensPerSecond: number | null | undefined,
  modelLine: string,
): TelemetryView {
  const decodeTokPerSec = formatTokRate('decode', decodeTokensPerSecond);
  const prefillTokPerSec = formatTokRate('prefill', prefillTokensPerSecond);

  let gpuRpc = '—';
  let pool = '—';

  if (stats && stats.numTokens) {
    const rpcPerTok = stats.gpuRpcCount && stats.numTokens ? Math.round(stats.gpuRpcCount / stats.numTokens) : null;
    if (rpcPerTok != null) {
      gpuRpc = `${rpcPerTok.toLocaleString()} gpu-rpc/tok`;
    }

    const hits = stats.poolHits ?? 0;
    const misses = stats.poolMisses ?? 0;
    const total = hits + misses;
    if (total > 0) {
      pool = `pool ${Math.round((hits / total) * 100)}%`;
    }
  }

  return { decodeTokPerSec, prefillTokPerSec, gpuRpc, pool, modelLine };
}

export function formatLoadingText(status: string | null): string {
  return status && status.trim().length > 0 ? status : 'Initializing model…';
}

export function formatBytes(value: number): string {
  if (value >= 1024 * 1024 * 1024) {
    return `${(value / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  }
  if (value >= 1024 * 1024) {
    return `${(value / (1024 * 1024)).toFixed(1)} MB`;
  }
  if (value >= 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }
  return `${Math.round(value)} B`;
}

export function formatFileName(file: string | undefined): string | null {
  const trimmed = file?.trim();
  if (!trimmed) return null;
  return trimmed.split('/').filter(Boolean).pop() ?? trimmed;
}

export type ReasoningEffort = 'off' | 'low' | 'medium' | 'high';

const REASONING_CYCLE: ReasoningEffort[] = ['off', 'low', 'medium', 'high'];

export function cycleReasoningEffort(current: ReasoningEffort): ReasoningEffort {
  const i = REASONING_CYCLE.indexOf(current);
  return REASONING_CYCLE[(i + 1) % REASONING_CYCLE.length];
}
