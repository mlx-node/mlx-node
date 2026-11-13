/** Prefill wording shared by the Metrics pages, the turn drawer and the session page. React-free. */

import { PREFILL_MIN_TOKENS, prefillRate } from '../../../src/prefill-rule.js';
import { formatCountNoun, formatDuration, formatNumber } from './format';

export { PREFILL_MIN_TOKENS, prefillRate };

/** `12,400 tokens in 6.74 s` */
export function prefillSizeText(tokens: number, ms: number): string {
  return `${formatNumber(tokens)} tokens in ${formatDuration(ms)}`;
}

export const PREFILL_UNDER_TEXT = `under ${String(PREFILL_MIN_TOKENS)} tokens`;

export const SETUP_HOLDS_TEXT = 'prefix lookup, cache restore, allocation, first decode step';

export const PREFILL_SMALL_HINT = 'Smaller prefills are mostly fixed cost, so they get no rate.';

export const PREFILL_NONE_TITLE = 'No measured prefill in this range';

export const PREFILL_NONE_HINT = 'Turns recorded before this version, and non-streamed turns, carry no prefill time.';

/** `Rated 312 of 793 completed turns · 420 not measured · 61 under 256 tokens`; null when every ok turn is rated. */
export function prefillCoverage(ok: number, measured: number, rated: number): string | null {
  if (ok === 0 || rated === ok) return null;
  const parts = [`Rated ${formatNumber(rated)} of ${formatCountNoun(ok, 'completed turn', 'completed turns')}`];
  if (ok - measured > 0) parts.push(`${formatNumber(ok - measured)} not measured`);
  if (measured - rated > 0) parts.push(`${formatNumber(measured - rated)} ${PREFILL_UNDER_TEXT}`);
  return parts.join(' · ');
}

/** Empty state of a prefill-speed card. */
export function prefillEmpty(measured: number): { title: string; hint: string } {
  if (measured === 0) return { title: PREFILL_NONE_TITLE, hint: PREFILL_NONE_HINT };
  return { title: `No prefill of ${String(PREFILL_MIN_TOKENS)}+ tokens in this range`, hint: PREFILL_SMALL_HINT };
}
