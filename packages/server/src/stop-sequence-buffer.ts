/**
 * Buffers streaming text to detect configured stop sequences. Text that
 * cannot be part of a partial stop sequence is released immediately; a
 * trailing suffix that could be the start of a stop sequence is held back
 * until a later push resolves it or the stream is flushed. Once a full stop
 * sequence is seen, everything after it is suppressed.
 */
export class StopSequenceBuffer {
  private readonly stopSequences: string[];
  private readonly maxLength: number;
  private pending_ = '';
  private _matched: string | null = null;

  constructor(stopSequences: string[]) {
    // Drop empty AND whitespace-only entries: a whitespace-only stop would
    // truncate normal output at the first space/newline, and the real
    // Anthropic API rejects such stops outright. Mirrors the same trim filter
    // in the request mapper so a whitespace-only configuration is a no-op.
    this.stopSequences = stopSequences.filter((s) => s.trim().length > 0);
    this.maxLength = this.stopSequences.reduce((max, s) => Math.max(max, s.length), 0);
  }

  /**
   * Earliest index wins; on a tie at the same index the longest wins. Returns
   * `{ idx, seq }` for the winning stop, or `{ idx: -1, seq: null }` when none
   * is present in `pending`.
   */
  private findMatch(): { idx: number; seq: string | null } {
    let matchIdx = -1;
    let matchSeq: string | null = null;
    for (const seq of this.stopSequences) {
      const idx = this.pending_.indexOf(seq);
      if (idx < 0) {
        continue;
      }
      if (matchIdx < 0 || idx < matchIdx || (idx === matchIdx && seq.length > (matchSeq?.length ?? 0))) {
        matchIdx = idx;
        matchSeq = seq;
      }
    }
    return { idx: matchIdx, seq: matchSeq };
  }

  /** The stop sequence that has matched so far, or `null` if none has. */
  get matched(): string | null {
    return this._matched;
  }

  /**
   * The text currently held back (received but neither emitted nor matched).
   * The streaming done-path reads this so it can scan the terminal/recovered
   * text on the SAME buffer with the held partial still in place, and so it
   * can reconstruct the full received-but-unemitted prefix for overlap math.
   */
  get pending(): string {
    return this.pending_;
  }

  /**
   * The earliest start index `j` in `[0, limit]` such that `pending.slice(j)`
   * is a non-empty STRICT prefix of some configured stop — i.e. a partial that
   * a later push could still grow into that stop. Returns -1 when no pending
   * suffix at or before `limit` is viable. Scanning from the front yields the
   * earliest start index, which is the one whose completed stop would win the
   * earliest-index tiebreak. A suffix longer than `maxLength - 1` can never be
   * a strict prefix of any stop, so the search starts no earlier than that.
   */
  private earliestViablePrefixIndex(limit: number): number {
    if (this.pending_.length === 0) {
      return -1;
    }
    const lowerBound = Math.max(0, this.pending_.length - (this.maxLength - 1));
    const upper = Math.min(limit, this.pending_.length - 1);
    for (let j = lowerBound; j <= upper; j++) {
      const suffix = this.pending_.slice(j);
      if (this.stopSequences.some((seq) => suffix.length < seq.length && seq.startsWith(suffix))) {
        return j;
      }
    }
    return -1;
  }

  /**
   * Feed text in. Returns `safeText` (emit as delta) and `matched` (the stop
   * sequence that has been matched, or `null`). After a match every push
   * returns empty `safeText` and keeps reporting the matched sequence.
   */
  push(text: string): { safeText: string; matched: string | null } {
    if (this._matched !== null) {
      return { safeText: '', matched: this._matched };
    }

    // Transparent pass-through when there is nothing to detect.
    if (this.stopSequences.length === 0) {
      return { safeText: text, matched: null };
    }

    this.pending_ += text;

    const { idx: matchIdx, seq: matchSeq } = this.findMatch();

    // Earliest start index of a still-growable stop prefix. When a full match
    // exists we only look at or before it (`limit = matchIdx`): a viable prefix
    // beginning AFTER the match would complete at a later index and lose the
    // earliest-index tiebreak, and the bytes from the match onward are
    // suppressed anyway. A viable prefix at or before the match could still
    // complete into a stop that WINS (earlier index, or longer at the same
    // index), so the match must be held. With no full match we consider the
    // whole pending text.
    const limit = matchIdx >= 0 ? matchIdx : this.pending_.length - 1;
    const holdIdx = this.earliestViablePrefixIndex(limit);

    if (matchIdx >= 0 && matchSeq !== null) {
      if (holdIdx >= 0) {
        // A longer/earlier stop could still complete from `holdIdx`; emit only
        // the bytes before it and keep the rest pending for a later push or
        // `flush()` to resolve.
        const safeText = this.pending_.slice(0, holdIdx);
        this.pending_ = this.pending_.slice(holdIdx);
        return { safeText, matched: null };
      }
      const safeText = this.pending_.slice(0, matchIdx);
      this._matched = matchSeq;
      this.pending_ = '';
      return { safeText, matched: matchSeq };
    }

    // No full match: release everything before the earliest viable prefix and
    // hold that suffix back, since a later push could complete it.
    const safeLen = holdIdx >= 0 ? holdIdx : this.pending_.length;
    const safeText = this.pending_.slice(0, safeLen);
    this.pending_ = this.pending_.slice(safeLen);
    return { safeText, matched: null };
  }

  /**
   * Release any held-back text at stream end. If a stop sequence already
   * matched, nothing more is emitted; otherwise the residue could not
   * complete any sequence and is released.
   */
  flush(): { safeText: string; matched: string | null } {
    if (this._matched !== null) {
      return { safeText: '', matched: this._matched };
    }
    // The stream has ended, so any match `push()` held back for a possible
    // longer same-index stop can no longer be extended — resolve it now.
    // Re-scan `pending` for the earliest match (longest on tie) and commit it
    // if present; otherwise the residue could not complete any sequence and is
    // released verbatim.
    const { idx: matchIdx, seq: matchSeq } = this.findMatch();
    if (matchIdx >= 0 && matchSeq !== null) {
      const safeText = this.pending_.slice(0, matchIdx);
      this._matched = matchSeq;
      this.pending_ = '';
      return { safeText, matched: matchSeq };
    }
    const safeText = this.pending_;
    this.pending_ = '';
    return { safeText, matched: null };
  }
}
