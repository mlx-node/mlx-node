/**
 * Streaming text-recovery helpers shared between the `/v1/messages` and
 * `/v1/responses` endpoints.
 *
 * Both endpoints have a tool-call streaming recovery branch that has to
 * compute the unsent suffix of `finalText` given that some prefix of the
 * model's output may already have been streamed to the wire, but native-side
 * string normalization makes `finalText` diverge from the streamed-prefix
 * verbatim. Concrete divergences seen in practice:
 *
 *   * The native side trims leading whitespace after `</think>` via
 *     `split_at_think_end`, so the streamed text can end in `"\n\n"` while
 *     `finalText` starts at `"<tool_call>"` (no overlap — emit `finalText`
 *     whole).
 *   * The native side `.trim()`s tool-tag-bracketed content boundaries, so
 *     the streamed text can have a trailing space that `finalText` lacks
 *     (also no overlap — emit `finalText` whole).
 *
 * Internal-only — not exported from `packages/server/src/index.ts`.
 */

/**
 * Find the largest k such that `streamed.endsWith(final.slice(0, k))`.
 *
 * Returns 0 when there is no overlap (caller emits `final` whole).
 * Returns `final.length` when `final` is fully contained as a suffix of
 * `streamed` (caller emits nothing).
 *
 * Used by the `/v1/messages` and `/v1/responses` streaming tool-call
 * recovery branches to decide how much of `finalText` is already on the
 * wire when native-side normalization (e.g. `.trim()`, post-`</think>`
 * whitespace stripping) makes the streamed prefix diverge from the
 * `finalText` prefix verbatim.
 */
export function longestSuffixPrefixOverlap(streamed: string, final: string): number {
  const max = Math.min(streamed.length, final.length);
  for (let k = max; k > 0; k--) {
    if (streamed.endsWith(final.slice(0, k))) return k;
  }
  return 0;
}
