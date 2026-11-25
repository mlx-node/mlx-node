/**
 * What the CONTROL PANEL process tells MAIN, as pure functions of its runtime: the pong, the worker restart note and
 * the closing note. Electron-free so the tests drive them with a stub runtime.
 */

import type { OpenCalls, RuntimeLifecycleEvent } from '@mlx-node/dashboard';

import { MAX_CONVS, SLOW_CALL_REPORT_MS, type PanelMessage, type SlowCall } from './liveness.js';

export { readPing } from './liveness.js';
export { LEDGER_DIR_ENV, readAppEnds } from '../panel-ledger.js';
export { WINDOWS_STORE_ENV } from '../main/child-env.js';

/** The part of the runtime a pong reads: in-memory lookups, no worker call, no fs. */
export interface PongRuntime {
  busyConversations(): string[];
  closingConversations(): string[];
  workerQueue(): { wq: number; wbusyMs: number; wroute: string | null } | null;
}

/**
 * The answer to MAIN's ping `n` (0: a push at a turn start or end). Synchronous on purpose: it measures this
 * process's main loop and nothing else, so it must never wait on the worker, the disk or a route.
 */
export function pongOf(n: number, runtime: PongRuntime | null, calls: OpenCalls, now: number): PanelMessage {
  const busy = runtime?.busyConversations() ?? [];
  const oldest = calls.oldest();
  let slow: SlowCall | null = null;
  if (oldest !== null && now - oldest.startedAt >= SLOW_CALL_REPORT_MS) {
    const queue = runtime?.workerQueue() ?? null;
    slow = {
      call: oldest.seq,
      route: oldest.route,
      thread: oldest.thread,
      ageMs: now - oldest.startedAt,
      wq: queue?.wq ?? 0,
      wbusyMs: queue?.wbusyMs ?? 0,
      wroute: queue?.wroute ?? null,
    };
  }
  return { type: 'mlx-pong', n, open: busy.length, convs: busy.slice(0, MAX_CONVS), slow };
}

/**
 * The note the panel's teardown posts before anything async: the turns it is about to cancel. A turn the user already
 * stopped keeps that cause, so it is not listed.
 */
export function closingOf(runtime: PongRuntime | null): PanelMessage {
  return { type: 'mlx-closing', convs: (runtime?.closingConversations() ?? []).slice(0, MAX_CONVS) };
}

/**
 * {@link closingOf} for the first teardown only. A second signal finds the turns already cancelled, and its shorter
 * list would replace the one MAIN keeps.
 */
export function firstClosing(): (runtime: PongRuntime | null) => PanelMessage | null {
  let sent = false;
  return (runtime) => {
    if (sent) return null;
    sent = true;
    return closingOf(runtime);
  };
}

/**
 * A database worker event. A restarted or given-up worker is told to MAIN, which writes the row; the process, its
 * agents and their turns stay. Nothing here exits.
 */
export function onRuntimeLifecycle(
  event: RuntimeLifecycleEvent,
  post: (message: PanelMessage) => void,
  open: () => number,
  log: (line: string) => void,
): void {
  if (event.type === 'db-closed') return;
  log(`[mlx] dashboard db worker: ${event.type} (${event.reason}, attempt ${String(event.attempt)})`);
  post({
    type: 'mlx-worker',
    reason: event.type === 'worker-gave-up' ? 'gave-up' : event.reason,
    route: event.route,
    ageMs: event.ageMs,
    attempt: event.attempt,
    open: open(),
  });
}
