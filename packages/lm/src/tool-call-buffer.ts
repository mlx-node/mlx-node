/**
 * Buffers streaming text to detect and suppress model structural tags. Text
 * that cannot be part of a partial tag is released immediately; once a
 * full structural tag is seen, everything after it is suppressed until
 * the stream ends.
 *
 * LFM2's `<|tool_call_start|>` suppresses like every other tag — NOT only
 * until its `<|tool_call_end|>`. Post-call prose, the echoed call body,
 * and later sentinel blocks are all held back, because the final parse
 * decides whether the call block was valid: streamed output must stay a
 * verbatim prefix of the raw model output so the terminal recovery
 * (`finalText` minus the longest suffix/prefix overlap) lands the right
 * remainder — cleaned text on success, the verbatim raw block on failure.
 * Releasing post-call prose eagerly would break that prefix invariant on
 * a malformed call: the raw block could no longer be ordered before prose
 * the wire already saw.
 */
export class ToolCallTagBuffer {
  private static readonly TAGS = [
    '<tool_call>',
    '</tool_call>',
    '<|tool_call>',
    '<tool_call|>',
    '<|tool_response>',
    '<tool_response|>',
    '<|tool>',
    '<tool|>',
    '<|channel>',
    '<channel|>',
    '<|turn>',
    '<turn|>',
    '<|tool_call_start|>',
  ] as const;

  private pendingText = '';
  /** Permanent suppression: a structural tag was seen. */
  private _terminalSuppressed = false;

  get suppressed(): boolean {
    return this._terminalSuppressed;
  }

  /**
   * Feed text in. Returns `safeText` (emit as delta), `tagFound` (a full
   * structural tag was just seen), and `cleanPrefix` (text before the tag
   * when `tagFound` — may contain whitespace; use `.trim()` only for
   * emptiness checks, never for emission).
   */
  push(text: string): { safeText: string; tagFound: boolean; cleanPrefix: string } {
    if (this._terminalSuppressed) {
      return { safeText: '', tagFound: false, cleanPrefix: '' };
    }
    this.pendingText += text;

    let tagIdx = -1;
    for (const tag of ToolCallTagBuffer.TAGS) {
      const idx = this.pendingText.indexOf(tag);
      if (idx >= 0 && (tagIdx < 0 || idx < tagIdx)) {
        tagIdx = idx;
      }
    }
    if (tagIdx >= 0) {
      const cleanPrefix = this.pendingText.slice(0, tagIdx);
      this._terminalSuppressed = true;
      this.pendingText = '';
      return { safeText: '', tagFound: true, cleanPrefix };
    }

    // Hold back any suffix that could be the start of a tag.
    let safeLen = this.pendingText.length;
    const maxTagLength = Math.max(...ToolCallTagBuffer.TAGS.map((tag) => tag.length));
    for (let i = 1; i <= Math.min(this.pendingText.length, maxTagLength - 1); i++) {
      const suffix = this.pendingText.slice(-i);
      if (ToolCallTagBuffer.TAGS.some((tag) => tag.startsWith(suffix))) {
        safeLen = this.pendingText.length - i;
        break;
      }
    }

    const safeText = this.pendingText.slice(0, safeLen);
    this.pendingText = this.pendingText.slice(safeLen);
    return { safeText, tagFound: false, cleanPrefix: '' };
  }

  /** Release any held-back text at stream end. */
  flush(): string {
    if (this._terminalSuppressed) {
      this.pendingText = '';
      return '';
    }
    const out = this.pendingText;
    this.pendingText = '';
    return out;
  }
}

/**
 * Find the largest k such that `streamed.endsWith(final.slice(0, k))`.
 *
 * Returns 0 when there is no overlap (caller emits `final` whole).
 * Returns `final.length` when `final` is fully contained as a suffix of
 * `streamed` (caller emits nothing).
 *
 * Used by the terminal tool-call recovery branches: once the tag buffer
 * has suppressed, the streamed text is a verbatim prefix of the raw model
 * output, so `final.slice(overlap)` is exactly the part of the finalized
 * `finalText` that never reached the wire — cleaned post-call text on
 * success, the verbatim raw block when the call was rejected.
 */
export function longestSuffixPrefixOverlap(streamed: string, final: string): number {
  const max = Math.min(streamed.length, final.length);
  for (let k = max; k > 0; k--) {
    if (streamed.endsWith(final.slice(0, k))) return k;
  }
  return 0;
}
