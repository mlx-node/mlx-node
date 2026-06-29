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
  private pending = '';
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
      const idx = this.pending.indexOf(seq);
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

    this.pending += text;

    const { idx: matchIdx, seq: matchSeq } = this.findMatch();
    if (matchIdx >= 0 && matchSeq !== null) {
      // A full match sits at `matchIdx`. Commit it only when no LONGER stop
      // sharing the same start index could still complete from a later push —
      // otherwise longest-on-tie (C4) would pick that longer stop once it
      // arrives. `fromMatch` is the text from the match index to the tail; if
      // it is still a strict prefix of a longer stop, a future push could
      // finish that stop, so we hold the match (emit only the safe text before
      // `matchIdx`) and let a later push or `flush()` resolve the tie.
      const fromMatch = this.pending.slice(matchIdx);
      const longerStillViable = this.stopSequences.some(
        (seq) => seq.length > fromMatch.length && seq.startsWith(fromMatch),
      );
      const safeText = this.pending.slice(0, matchIdx);
      if (longerStillViable) {
        this.pending = fromMatch;
        return { safeText, matched: null };
      }
      this._matched = matchSeq;
      this.pending = '';
      return { safeText, matched: matchSeq };
    }

    // Hold back the longest suffix that is a proper prefix of any stop
    // sequence, since a later push could complete it.
    let holdLen = 0;
    const maxHold = Math.min(this.pending.length, this.maxLength - 1);
    for (let i = maxHold; i >= 1; i--) {
      const suffix = this.pending.slice(-i);
      if (this.stopSequences.some((seq) => suffix.length < seq.length && seq.startsWith(suffix))) {
        holdLen = i;
        break;
      }
    }

    const safeLen = this.pending.length - holdLen;
    const safeText = this.pending.slice(0, safeLen);
    this.pending = this.pending.slice(safeLen);
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
      const safeText = this.pending.slice(0, matchIdx);
      this._matched = matchSeq;
      this.pending = '';
      return { safeText, matched: matchSeq };
    }
    const safeText = this.pending;
    this.pending = '';
    return { safeText, matched: null };
  }
}
