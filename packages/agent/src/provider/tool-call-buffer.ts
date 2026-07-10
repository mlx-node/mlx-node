/**
 * Port of `packages/server/src/tool-call-buffer.ts` — the agent package
 * must not depend on `@mlx-node/server`, so the class is duplicated here
 * with identical semantics. Keep the two in sync.
 *
 * Buffers streaming text to detect and suppress model structural tags. Text
 * that cannot be part of a partial tag is released immediately; once a
 * full structural tag is seen, everything after it is suppressed until
 * the stream ends.
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
  ] as const;
  private pendingText = '';
  private _suppressed = false;

  get suppressed(): boolean {
    return this._suppressed;
  }

  /**
   * Feed text in. Returns `safeText` (emit as delta), `tagFound` (a full
   * structural tag was just seen), and `cleanPrefix` (text before the tag
   * when `tagFound` — may contain whitespace; use `.trim()` only for
   * emptiness checks, never for emission).
   */
  push(text: string): { safeText: string; tagFound: boolean; cleanPrefix: string } {
    if (this._suppressed) {
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
      this._suppressed = true;
      this.pendingText = '';
      return { safeText: '', tagFound: true, cleanPrefix };
    }

    // Hold back any suffix that could be the start of the tag.
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
    const text = this.pendingText;
    this.pendingText = '';
    return text;
  }
}
