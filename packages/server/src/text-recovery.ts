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
 * The implementation lives in `@mlx-node/lm` next to `ToolCallTagBuffer`
 * (the suppression that creates the prefix invariant this relies on) so
 * non-server consumers — the pi agent `TurnEmitter` — share one source.
 *
 * Internal-only — not exported from `packages/server/src/index.ts`.
 */
export { longestSuffixPrefixOverlap } from '@mlx-node/lm';
