/**
 * Port of `packages/server/src/tool-call-buffer.ts` — the agent package
 * must not depend on `@mlx-node/server`, so the class is duplicated here
 * with identical semantics. Keep the two in sync.
 *
 * Buffers streaming text to detect and suppress model structural tags. Text
 * that cannot be part of a partial tag is released immediately; once a
 * full structural tag is seen, everything after it is suppressed until
 * the stream ends.
 *
 * LFM2's pythonic sentinels are the exception to "suppressed forever":
 * `<|tool_call_start|>…<|tool_call_end|>` is a paired block — only its
 * interior is suppressed, then post-call prose resumes. LFM2 also re-emits
 * the call body after the first end sentinel capped by a second end (the
 * "echo"); following vLLM's `Lfm2ToolParser` streaming logic, trailing
 * text that starts with `[` or `<` is held until a later end sentinel
 * resolves it, and everything through the LAST orphan end is dropped.
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
  private static readonly LFM2_START = '<|tool_call_start|>';
  private static readonly LFM2_END = '<|tool_call_end|>';

  private pendingText = '';
  /** Permanent suppression: a non-LFM2 structural tag was seen. */
  private _terminalSuppressed = false;
  /**
   * LFM2 paired-sentinel state:
   * - `normal`: scanning for tags.
   * - `interior`: inside `<|tool_call_start|>…` — suppress until the end.
   * - `post`: after `<|tool_call_end|>` — hold suspect text (echo watch)
   *   but release genuine prose. Persists to stream end: an echo can begin
   *   in any later delta, so the suspicion never expires (vLLM parity).
   */
  private mode: 'normal' | 'interior' | 'post' = 'normal';

  get suppressed(): boolean {
    return this._terminalSuppressed || this.mode !== 'normal';
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

    // Transitions can cascade within one delta (end → post → prose in the
    // same chunk), so evaluate states in a loop.
    for (;;) {
      if (this.mode === 'interior') {
        const endIdx = this.pendingText.indexOf(ToolCallTagBuffer.LFM2_END);
        if (endIdx < 0) {
          return { safeText: '', tagFound: false, cleanPrefix: '' };
        }
        this.pendingText = this.pendingText.slice(endIdx + ToolCallTagBuffer.LFM2_END.length);
        this.mode = 'post';
        continue;
      }

      if (this.mode === 'post') {
        // A completed echo (or a second call block) ends with another end
        // sentinel: drop everything through the LAST one, then re-evaluate.
        const lastEnd = this.pendingText.lastIndexOf(ToolCallTagBuffer.LFM2_END);
        if (lastEnd >= 0) {
          this.pendingText = this.pendingText.slice(lastEnd + ToolCallTagBuffer.LFM2_END.length);
        }
        // A fresh start sentinel begins another suppressed block. Text
        // before it is only released when it isn't itself suspect — held
        // echo fragments (`[f()`…) followed by a new call are markup, not
        // prose.
        const startIdx = this.pendingText.indexOf(ToolCallTagBuffer.LFM2_START);
        if (startIdx >= 0) {
          const before = this.pendingText.slice(0, startIdx);
          const l = before.trimStart();
          const cleanPrefix = l.startsWith('[') || l.startsWith('<') ? '' : before;
          this.pendingText = this.pendingText.slice(startIdx + ToolCallTagBuffer.LFM2_START.length);
          this.mode = 'interior';
          return { safeText: '', tagFound: true, cleanPrefix };
        }
        const lstripped = this.pendingText.trimStart();
        if (lstripped.startsWith('[') || lstripped.startsWith('<')) {
          // Suspect echo body or a leading partial sentinel: hold until
          // resolved. (A partial sentinel after released prose leaks —
          // vLLM parity; sentinels are single added tokens that cannot
          // split at token granularity in practice.)
          return { safeText: '', tagFound: false, cleanPrefix: '' };
        }
        // Genuine post-call prose (or pure whitespace): release it. Whitespace
        // is safe to release eagerly — a later `[` re-enters the hold.
        const safeText = this.pendingText;
        this.pendingText = '';
        return { safeText, tagFound: false, cleanPrefix: '' };
      }

      // --- normal mode ---
      let tagIdx = -1;
      let matchedTag: string | undefined;
      for (const tag of ToolCallTagBuffer.TAGS) {
        const idx = this.pendingText.indexOf(tag);
        if (idx >= 0 && (tagIdx < 0 || idx < tagIdx)) {
          tagIdx = idx;
          matchedTag = tag;
        }
      }
      if (tagIdx >= 0) {
        const cleanPrefix = this.pendingText.slice(0, tagIdx);
        if (matchedTag === ToolCallTagBuffer.LFM2_START) {
          // Paired sentinel: keep the remainder so a same-chunk end tag
          // (or prose after it) is still processed — by the next push,
          // or by flush() at stream end.
          this.pendingText = this.pendingText.slice(tagIdx + ToolCallTagBuffer.LFM2_START.length);
          this.mode = 'interior';
        } else {
          this._terminalSuppressed = true;
          this.pendingText = '';
        }
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
  }

  /** Release any held-back text at stream end. */
  flush(): string {
    if (this._terminalSuppressed) {
      this.pendingText = '';
      return '';
    }
    // Drive the LFM2 machine to quiescence before dropping anything: a
    // same-chunk `…<|tool_call_end|>prose` tail left in pendingText by the
    // tagFound early-return must still release its prose.
    let released = '';
    while (this.mode !== 'normal' && this.pendingText.length > 0) {
      const { safeText, tagFound, cleanPrefix } = this.push('');
      released += safeText + cleanPrefix;
      // A pass that releases nothing and finds no tag means the remainder
      // is held suspect/markup text — it must never reach the wire.
      if (!tagFound && safeText === '' && cleanPrefix === '') {
        break;
      }
    }
    const out = released + (this.mode === 'normal' ? this.pendingText : '');
    this.pendingText = '';
    return out;
  }
}
