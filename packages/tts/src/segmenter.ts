/** Text policy is independent of model tokens, codec frames and transport chunks. */
export class TextSegmenter {
  #pending = '';
  readonly #graphemes = new Intl.Segmenter(undefined, { granularity: 'grapheme' });
  readonly #sentences = new Intl.Segmenter(undefined, { granularity: 'sentence' });
  readonly #words = new Intl.Segmenter(undefined, { granularity: 'word' });
  constructor(readonly maxGraphemes = 256) {
    if (!Number.isSafeInteger(maxGraphemes) || maxGraphemes < 1)
      throw new RangeError('maxSegmentGraphemes must be a positive integer');
  }
  *push(text: string, flush = false): Generator<string> {
    this.#pending += text;
    while (this.#pending) {
      const graphemeEnds: number[] = [];
      let limit = this.#pending.length;
      let overflow = false;
      for (const part of this.#graphemes.segment(this.#pending)) {
        if (graphemeEnds.length === this.maxGraphemes) {
          limit = part.index;
          overflow = true;
          break;
        }
        graphemeEnds.push(part.index + part.segment.length);
      }
      let boundary = 0;
      for (const part of this.#sentences.segment(this.#pending)) {
        const end = part.index + part.segment.length;
        if (end > limit) break;
        // A trailing period can still become a decimal or abbreviation in the
        // next input chunk. Unambiguous terminal punctuation can commit now.
        if (end < this.#pending.length || /[。！？!?；;\n][\s”’"')）\]]*$/u.test(part.segment) || flush) {
          boundary = end;
          break;
        }
      }
      if (!boundary && overflow) {
        const prefix = this.#pending.slice(0, limit);
        const clause = [...prefix.matchAll(/[,，、:：;；\s]/gu)].at(-1);
        boundary = clause ? clause.index + clause[0].length : 0;
        if (!boundary) {
          for (const word of this.#words.segment(prefix)) if (word.index > 0) boundary = word.index;
        }
        boundary ||= limit;
      }
      if (!boundary && flush) boundary = this.#pending.length;
      if (!boundary) break;
      // Clause punctuation or whitespace can carry combining marks. Keep the
      // entire grapheme even when the preferred boundary follows its base.
      boundary = graphemeEnds.find((end) => end >= boundary) ?? boundary;
      const text = this.#pending.slice(0, boundary);
      this.#pending = this.#pending.slice(boundary);
      yield text;
    }
  }
}
