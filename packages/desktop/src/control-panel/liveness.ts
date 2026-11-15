/**
 * The liveness contract between MAIN's broker and the CONTROL PANEL process, on the utility process's parent port.
 *
 * MAIN pings every {@link PING_INTERVAL_MS}; the panel answers from one synchronous main-thread function. Any message
 * from the child counts as life. Electron-free: both ends and the tests import it.
 */

export const PING_INTERVAL_MS = 5_000;
/** Silent on-time ticks before a restart when the last pong listed no running turn. */
export const IDLE_SILENT_TICKS = 6;
/** Silent on-time ticks before a restart when a turn runs, or before the child's first message. */
export const TURN_SILENT_TICKS = 24;
/** A tick further than this from its schedule proves MAIN itself did not run on time: that tick is not counted. */
export const TICK_LATE_TOLERANCE_MS = 1_000;
/** Silent ticks after which MAIN records a slow loop (record only). */
export const SLOW_LOOP_TICKS = 2;
/** A page call open this long is named in the pong. */
export const SLOW_CALL_REPORT_MS = 5_000;
/** A page call open this long gets one `slow rpc` row. */
export const SLOW_CALL_RECORD_MS = 30_000;
/** At most this many conversation ids cross in one message. */
export const MAX_CONVS = 8;

export type WorkerRestartReason = 'error' | 'exit' | 'no-ack' | 'gave-up';

/** The oldest page call still open, by route pattern only. `call` is the host's sequence number, never stored. */
export interface SlowCall {
  call: number;
  route: string | null;
  thread: 'main' | 'worker' | null;
  ageMs: number;
  wq: number;
  wbusyMs: number;
  wroute: string | null;
}

export interface PingMessage {
  type: 'mlx-ping';
  n: number;
}

export type PanelMessage =
  /** `n` 0: a push at a turn start or end. */
  | { type: 'mlx-pong'; n: number; open: number; convs: string[]; slow: SlowCall | null }
  | {
      type: 'mlx-worker';
      reason: WorkerRestartReason;
      route: string | null;
      ageMs: number | null;
      attempt: number;
      open: number;
    }
  /** Posted by the panel's teardown before anything async: the turns it is cancelling as `app-closed`. */
  | { type: 'mlx-closing'; convs: string[] };

function count(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;
}

function text(value: unknown): string | null {
  return typeof value === 'string' ? value : null;
}

function ids(value: unknown): string[] | null {
  if (!Array.isArray(value)) return null;
  const out: string[] = [];
  for (const item of value.slice(0, MAX_CONVS)) if (typeof item === 'string') out.push(item);
  return out;
}

function slowOf(value: unknown): SlowCall | null | undefined {
  if (value === null) return null;
  if (typeof value !== 'object') return undefined;
  const v = value as Record<string, unknown>;
  const call = count(v.call);
  const ageMs = count(v.ageMs);
  const wq = count(v.wq);
  const wbusyMs = count(v.wbusyMs);
  if (call === null || ageMs === null || wq === null || wbusyMs === null) return undefined;
  const thread = v.thread === 'main' || v.thread === 'worker' ? v.thread : null;
  return { call, route: text(v.route), thread, ageMs, wq, wbusyMs, wroute: text(v.wroute) };
}

const WORKER_REASONS: readonly WorkerRestartReason[] = ['error', 'exit', 'no-ack', 'gave-up'];

/** A message from the child, or null for anything else. The ledger encoder still checks every field it writes. */
export function readPanelMessage(value: unknown): PanelMessage | null {
  if (typeof value !== 'object' || value === null) return null;
  const v = value as Record<string, unknown>;
  if (v.type === 'mlx-pong') {
    const n = count(v.n);
    const open = count(v.open);
    const convs = ids(v.convs);
    const slow = slowOf(v.slow);
    if (n === null || open === null || convs === null || slow === undefined) return null;
    return { type: 'mlx-pong', n, open, convs, slow };
  }
  if (v.type === 'mlx-worker') {
    const reason = WORKER_REASONS.find((r) => r === v.reason);
    const attempt = count(v.attempt);
    const open = count(v.open);
    if (reason === undefined || attempt === null || open === null) return null;
    return { type: 'mlx-worker', reason, route: text(v.route), ageMs: count(v.ageMs), attempt, open };
  }
  if (v.type === 'mlx-closing') {
    const convs = ids(v.convs);
    return convs === null ? null : { type: 'mlx-closing', convs };
  }
  return null;
}

/** MAIN's ping, as the child reads it. */
export function readPing(value: unknown): PingMessage | null {
  if (typeof value !== 'object' || value === null) return null;
  const v = value as Record<string, unknown>;
  const n = count(v.n);
  return v.type === 'mlx-ping' && n !== null ? { type: 'mlx-ping', n } : null;
}
