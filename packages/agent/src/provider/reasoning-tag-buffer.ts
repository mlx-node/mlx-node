/**
 * Removes model-protocol thinking tags from raw native reasoning deltas.
 *
 * The native stream deliberately exposes raw ChatML text, so `<think>` /
 * `</think>` (and the LongCat variants) may arrive as complete tags or split
 * across multiple deltas. Pi's `ThinkingContent` is already structured and
 * must contain only the reasoning body.
 *
 * Text that cannot belong to a partial structural tag is released
 * immediately. An ambiguous suffix is held until another delta disambiguates
 * it or `flush()` recovers it at a terminal boundary.
 */
export class ReasoningTagBuffer {
  private static readonly TAGS = ['<think>', '</think>', '<longcat_think>', '</longcat_think>'] as const;
  private pendingText = '';

  /** Feed one raw reasoning delta and return protocol-tag-free text. */
  push(text: string): string {
    this.pendingText += text;
    let safeText = '';

    while (this.pendingText) {
      const match = this.findFirstTag();
      if (match) {
        safeText += this.pendingText.slice(0, match.index);
        this.pendingText = this.pendingText.slice(match.index + match.tag.length);
        continue;
      }

      const safeLen = this.safePrefixLength();
      safeText += this.pendingText.slice(0, safeLen);
      this.pendingText = this.pendingText.slice(safeLen);
      break;
    }

    return safeText;
  }

  /** Release an incomplete, therefore non-structural, tag prefix at stream end. */
  flush(): string {
    const text = this.pendingText;
    this.pendingText = '';
    return text;
  }

  private findFirstTag(): { index: number; tag: (typeof ReasoningTagBuffer.TAGS)[number] } | null {
    let first: { index: number; tag: (typeof ReasoningTagBuffer.TAGS)[number] } | null = null;
    for (const tag of ReasoningTagBuffer.TAGS) {
      const index = this.pendingText.indexOf(tag);
      if (index >= 0 && (first === null || index < first.index)) {
        first = { index, tag };
      }
    }
    return first;
  }

  private safePrefixLength(): number {
    const maxTagLength = Math.max(...ReasoningTagBuffer.TAGS.map((tag) => tag.length));
    for (let length = 1; length <= Math.min(this.pendingText.length, maxTagLength - 1); length++) {
      const suffix = this.pendingText.slice(-length);
      if (ReasoningTagBuffer.TAGS.some((tag) => tag.startsWith(suffix))) {
        return this.pendingText.length - length;
      }
    }
    return this.pendingText.length;
  }
}
