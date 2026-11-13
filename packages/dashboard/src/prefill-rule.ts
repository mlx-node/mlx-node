/** Smallest prefill that gets a rate: below it the fixed cost inside the window dominates. */
export const PREFILL_MIN_TOKENS = 256;

/** tok/s, or null when a value is missing or the prefill is under the floor. `calls` > 1 for a sum over calls. */
export function prefillRate(tokens: number | null, ms: number | null, calls = 1): number | null {
  if (tokens === null || ms === null) return null;
  if (!Number.isFinite(tokens) || !Number.isFinite(ms) || ms <= 0 || calls < 1) return null;
  if (tokens / calls < PREFILL_MIN_TOKENS) return null;
  return (tokens * 1000) / ms;
}
