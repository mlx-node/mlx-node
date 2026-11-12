/** The longest stop sequence a request may carry, in characters (Splash `MAX_STOP_CHARACTERS`). */
export const MAX_STOP_SEQUENCE_CHARACTERS = 16 * 1024;
/** The most stop sequences a request may carry. */
export const MAX_STOP_SEQUENCES = 4;

/**
 * Finds the first stop sequence the streamed text contains, as it arrives
 * (Splash `StopSearch`). Each stop keeps a Knuth-Morris-Pratt state, so a
 * character costs the same however long the stops are. Text that may begin a
 * stop is held back; once a stop matches, everything after it is suppressed.
 */
export class StopSequenceBuffer {
  private readonly stops: readonly string[];
  private readonly fallbacks: readonly Uint32Array[];
  private readonly states: Uint32Array;
  private pending_ = '';
  private matched_: string | null = null;

  constructor(stopSequences: readonly string[]) {
    for (const stop of stopSequences) {
      if (stop.length === 0) throw new Error('A stop sequence must not be empty');
    }
    this.stops = stopSequences.slice();
    this.fallbacks = this.stops.map(prefixFunction);
    this.states = new Uint32Array(this.stops.length);
  }

  /** The stop sequence that has matched so far, or `null` if none has. */
  get matched(): string | null {
    return this.matched_;
  }

  /** The text held back because it may begin a stop. */
  get pending(): string {
    return this.pending_;
  }

  /**
   * Feed text in. Returns `safeText` (emit as delta) and `matched` (the stop
   * sequence that has matched, or `null`). After a match every push returns
   * empty `safeText` and keeps reporting the matched sequence.
   */
  push(text: string): { safeText: string; matched: string | null } {
    if (this.matched_ !== null) {
      return { safeText: '', matched: this.matched_ };
    }
    if (this.stops.length === 0) {
      return { safeText: text, matched: null };
    }
    this.pending_ += text;
    const found = this.search(text);
    if (found !== null) {
      const safeText = this.pending_.slice(0, this.pending_.length - found.back);
      this.matched_ = found.stop;
      this.pending_ = '';
      return { safeText, matched: found.stop };
    }
    let held = 0;
    for (const state of this.states) held = Math.max(held, state);
    if (held === this.pending_.length) {
      return { safeText: '', matched: null };
    }
    const cut = this.pending_.length - held;
    const safeText = this.pending_.slice(0, cut);
    this.pending_ = this.pending_.slice(cut);
    return { safeText, matched: null };
  }

  /** Release the held text at stream end. */
  flush(): { safeText: string; matched: string | null } {
    if (this.matched_ !== null) {
      return { safeText: '', matched: this.matched_ };
    }
    const safeText = this.pending_;
    this.pending_ = '';
    this.states.fill(0);
    return { safeText, matched: null };
  }

  /**
   * The stop that the text, `text` appended, contains first, and how many
   * code units before the end of `text` it begins; `null` when none.
   */
  private search(text: string): { stop: string; back: number } | null {
    let first: { stop: string; back: number } | null = null;
    for (let index = 0; index < this.stops.length; index++) {
      const stop = this.stops[index]!;
      const fallback = this.fallbacks[index]!;
      let state = this.states[index]!;
      for (let position = 0; position < text.length; position++) {
        const unit = text.charCodeAt(position);
        while (state > 0 && stop.charCodeAt(state) !== unit) state = fallback[state - 1]!;
        if (stop.charCodeAt(state) === unit) state++;
        if (state === stop.length) {
          const back = text.length - position - 1 + stop.length;
          if (first === null || back > first.back) first = { stop, back };
          break;
        }
      }
      this.states[index] = state;
    }
    return first;
  }
}

/** For each prefix of `text`, the length of its longest proper end that is also a prefix of `text`. */
function prefixFunction(text: string): Uint32Array {
  const lengths = new Uint32Array(text.length);
  let length = 0;
  for (let index = 1; index < text.length; index++) {
    const unit = text.charCodeAt(index);
    while (length > 0 && unit !== text.charCodeAt(length)) length = lengths[length - 1]!;
    if (unit === text.charCodeAt(length)) length++;
    lengths[index] = length;
  }
  return lengths;
}
