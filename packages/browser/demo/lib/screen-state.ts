export type ScreenState = 'landing' | 'loading' | 'chat';

export type ScreenEvent =
  | { type: 'init' }
  | { type: 'load_kickoff' }
  | { type: 'model_ready' }
  | { type: 'model_error' }
  | { type: 'reset_chat' };

export function reduceScreen(state: ScreenState | undefined, event: ScreenEvent): ScreenState {
  if (state === undefined || event.type === 'init') return 'landing';
  switch (state) {
    case 'landing':
      if (event.type === 'load_kickoff') return 'loading';
      return 'landing';
    case 'loading':
      if (event.type === 'model_ready') return 'chat';
      if (event.type === 'model_error') return 'landing';
      return 'loading';
    case 'chat':
      return 'chat';
  }
}

export type ProfileLikeStats = {
  numTokens?: number;
  gpuRpcCount?: number;
  poolHits?: number;
  poolMisses?: number;
};

export type TelemetryView = {
  tokPerSec: string;
  gpuRpc: string;
  pool: string;
  modelLine: string;
};

/**
 * Formats the telemetry footer strip values.
 *
 * `stats` comes from the demo's ProfileStats reducer (worker counters).
 * `decodeTokensPerSecond` comes from ChatResult.performance (chat-stream API).
 * The two sources are merged at the call site (see chat screen wiring).
 */
export function formatTelemetry(
  stats: ProfileLikeStats | null | undefined,
  decodeTokensPerSecond: number | null | undefined,
  modelLine: string,
): TelemetryView {
  if (!stats || !stats.numTokens) {
    return { tokPerSec: '—', gpuRpc: '—', pool: '—', modelLine };
  }

  const tps = decodeTokensPerSecond ?? 0;
  const tokPerSec = tps > 0 ? `${Math.round(tps)} tok/s` : '—';

  const rpcPerTok = stats.gpuRpcCount && stats.numTokens ? Math.round(stats.gpuRpcCount / stats.numTokens) : null;
  const gpuRpc = rpcPerTok != null ? `${rpcPerTok.toLocaleString()} gpu-rpc/tok` : '—';

  const hits = stats.poolHits ?? 0;
  const misses = stats.poolMisses ?? 0;
  const total = hits + misses;
  const pool = total > 0 ? `pool ${Math.round((hits / total) * 100)}%` : '—';

  return { tokPerSec, gpuRpc, pool, modelLine };
}

export function formatLoadingText(status: string | null): string {
  return status && status.trim().length > 0 ? status : 'Initializing model…';
}

export type ReasoningEffort = 'off' | 'low' | 'medium' | 'high';

const REASONING_CYCLE: ReasoningEffort[] = ['off', 'low', 'medium', 'high'];

export function cycleReasoningEffort(current: ReasoningEffort): ReasoningEffort {
  const i = REASONING_CYCLE.indexOf(current);
  return REASONING_CYCLE[(i + 1) % REASONING_CYCLE.length];
}
