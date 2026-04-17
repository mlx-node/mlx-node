/**
 * Buffers streaming text to detect and suppress `<tool_call>` tags. Text
 * that cannot be part of a partial tag is released immediately; once a
 * full tag is seen, everything after it is suppressed until the stream
 * ends.
 */
export class ToolCallTagBuffer {
  private static readonly TAG = '<tool_call>';
  private pendingText = '';
  private _suppressed = false;

  get suppressed(): boolean {
    return this._suppressed;
  }

  /**
   * Feed text in. Returns `safeText` (emit as delta), `tagFound` (a full
   * `<tool_call>` was just seen), and `cleanPrefix` (text before the tag
   * when `tagFound` — may contain whitespace; use `.trim()` only for
   * emptiness checks, never for emission).
   */
  push(text: string): { safeText: string; tagFound: boolean; cleanPrefix: string } {
    if (this._suppressed) {
      return { safeText: '', tagFound: false, cleanPrefix: '' };
    }

    this.pendingText += text;

    const tagIdx = this.pendingText.indexOf(ToolCallTagBuffer.TAG);
    if (tagIdx >= 0) {
      const cleanPrefix = this.pendingText.slice(0, tagIdx);
      this._suppressed = true;
      this.pendingText = '';
      return { safeText: '', tagFound: true, cleanPrefix };
    }

    // Hold back any suffix that could be the start of the tag.
    let safeLen = this.pendingText.length;
    for (let i = 1; i <= Math.min(this.pendingText.length, ToolCallTagBuffer.TAG.length - 1); i++) {
      const suffix = this.pendingText.slice(-i);
      if (ToolCallTagBuffer.TAG.startsWith(suffix)) {
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
