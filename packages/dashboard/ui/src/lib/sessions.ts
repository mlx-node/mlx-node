import { toast } from 'sonner';

import { DashboardApiError } from './api';
import type { TurnMetric } from './types';

/** Display title: the stored title, or the id when the session never got one. */
export function sessionTitle(session: { id: string; title: string }): string {
  const title = session.title.trim();
  return title === '' ? session.id : title;
}

/** ISO timestamp from `summary.json` → epoch ms; `NaN` for a blank or bad value. */
export function isoMs(iso: string): number {
  return iso === '' ? Number.NaN : Date.parse(iso);
}

/** A 409 from a write guard: the session is live, recently written, or changed underneath. */
export function isConflict(err: unknown): err is DashboardApiError {
  return err instanceof DashboardApiError && err.status === 409;
}

export function errMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export const LIVE_HINT = 'In use by a running agent. Rename and delete are off until it stops.';

export function copyResumeCommand(command: string): void {
  void navigator.clipboard.writeText(command).then(
    () => toast.success('Resume command copied', { description: command }),
    () => toast.error('Failed to copy to clipboard'),
  );
}

export interface TurnTokenBars {
  uncached: number;
  cached: number;
  output: number;
}

/**
 * Disjoint stack segments for one turn. grok's `inputTokens` already includes
 * the cache reads, so stacking `cachedTokens` on top would count them twice.
 */
export function turnTokenBars(t: Pick<TurnMetric, 'inputTokens' | 'cachedTokens' | 'outputTokens'>): TurnTokenBars {
  const cached = Math.min(t.cachedTokens, t.inputTokens);
  return { uncached: t.inputTokens - cached, cached, output: t.outputTokens };
}

/** Below this many new tokens a turn's prefill time is mostly fixed per-call cost, so its rate means nothing. */
export const PREFILL_MIN_TOKENS = 512;

export interface TurnPrefill {
  tokens: number;
  seconds: number;
  tps: number;
}

/**
 * A turn's prefill, or `null` when it is unreported, too small to judge, or of unknown size.
 *
 * The engine reports `prefillTps = work tokens / ttft`, and its work tokens can exceed
 * `prefillInputTokens` (LFM2 / NemotronH re-prefill the whole prompt on a partial hit),
 * so the size comes from `tps × ttft` whenever ttft is known.
 */
export function turnPrefill(t: Pick<TurnMetric, 'prefillTps' | 'prefillInputTokens' | 'ttftMs'>): TurnPrefill | null {
  const tps = t.prefillTps;
  if (tps === null || !Number.isFinite(tps) || tps <= 0) return null;
  let tokens: number;
  let seconds: number;
  if (t.ttftMs !== null && Number.isFinite(t.ttftMs) && t.ttftMs > 0) {
    seconds = t.ttftMs / 1000;
    tokens = tps * seconds;
  } else if (t.prefillInputTokens !== null && Number.isFinite(t.prefillInputTokens)) {
    tokens = t.prefillInputTokens;
    seconds = tokens / tps;
  } else {
    return null;
  }
  if (tokens < PREFILL_MIN_TOKENS) return null;
  return { tokens, seconds, tps };
}

/** Token-weighted prefill rate over the qualifying turns: total tokens / total seconds. */
export function prefillAverage(prefills: Array<TurnPrefill | null>): number | null {
  let tokens = 0;
  let seconds = 0;
  for (const p of prefills) {
    if (p === null) continue;
    tokens += p.tokens;
    seconds += p.seconds;
  }
  return seconds > 0 ? tokens / seconds : null;
}
