/**
 * Buffers streaming text to detect and suppress <tool_call> tags.
 *
 * The buffer accumulates incoming text tokens and detects the '<tool_call>'
 * tag. Text that is safe to emit (i.e. cannot be part of a partial tag) is
 * released immediately. Once a full tag is detected, subsequent text is
 * suppressed until the stream ends.
 */
export class ToolCallTagBuffer {
  private static readonly TAG = '<tool_call>';
  private pendingText = '';
  private _suppressed = false;

  /** Whether a tool-call tag has been detected and text is being suppressed. */
  get suppressed(): boolean {
    return this._suppressed;
  }

  /**
   * Feed new text into the buffer.
   *
   * Returns an object with:
   * - `safeText`: text that can be safely emitted as a delta (empty string if none)
   * - `tagFound`: true if a full `<tool_call>` tag was detected in this call
   * - `cleanPrefix`: when `tagFound` is true, the text before the tag (may contain
   *   whitespace -- use `.trim()` only for emptiness checks, not for emission)
   */
  push(text: string): { safeText: string; tagFound: boolean; cleanPrefix: string } {
    if (this._suppressed) {
      return { safeText: '', tagFound: false, cleanPrefix: '' };
    }

    this.pendingText += text;

    // Check for full tool-call tag
    const tagIdx = this.pendingText.indexOf(ToolCallTagBuffer.TAG);
    if (tagIdx >= 0) {
      const cleanPrefix = this.pendingText.slice(0, tagIdx);
      this._suppressed = true;
      this.pendingText = '';
      return { safeText: '', tagFound: true, cleanPrefix };
    }

    // Check if pendingText ends with a partial prefix of '<tool_call>'
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

  /**
   * Flush any remaining pending text. Call this when the stream ends
   * without a tool-call tag being found, or when suppression was not triggered.
   */
  flush(): string {
    const text = this.pendingText;
    this.pendingText = '';
    return text;
  }
}
