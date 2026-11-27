/**
 * Lane time by phase, and the token/time pairs behind the live Output rate and Prefill speed.
 *
 * Phase time comes from the native per-turn clocks (`waitMs`, `slotMs`, `prepareMs`, `setupMs` + `prefillElapsedMs`,
 * `decodeElapsedMs`), credited per lane as their deltas, so a phase shorter than one sampler tick still counts in
 * full. Native stops the clocks when the turn ends, so a terminal chunk that waits for its reader adds no time.
 * Counters are cumulative and never decrease; readers difference two reads.
 *
 * A rate pair (tokens, ms) is credited only when its tokens move: a one-shot prefill, or a core that reports its
 * tokens only at the end, adds its time together with its tokens, never before. A lane's decode time waits for that
 * lane's own tokens; a lane that ends with no kept token after its first gives its decode time to no rate.
 *
 * The live rates cover the last {@link RATE_WINDOW_MS} of work: every credit keeps the interval its work ran in, and
 * the window cuts those intervals at its start.
 */

import type { TurnProgressSnapshot } from './native-progress.js';

/** Lane-ms per state. `loading` is wall ms with at least one model load in flight. */
export interface WorkMs {
  loading: number;
  /** Native `waitMs` + `slotMs`: dispatched, not started. */
  waiting: number;
  prepare: number;
  /** Native phase `prefill`: a whole turn's setup before its prefill window, and the window. */
  prefill: number;
  decode: number;
  /** Lanes without native progress (non-streamed turns, System One decisions). */
  running: number;
}

export interface WorkTotals {
  /** Lane-ms per state since process start. */
  ms: WorkMs;
  /** Kept tokens after each streamed lane's first, and the ms with at least one lane decoding that produced them. */
  outTokens: number;
  outMs: number;
  /** Computed prompt tokens of rated prefills (`prefillTotal >= RATED_PREFILL_MIN_TOKENS`) and their lane prefill ms. */
  prefillTokens: number;
  prefillMs: number;
}

/** What the ledger has already credited for one live turn. Times are on the clock the ledger is called with. */
export interface LaneMark {
  /** Native clock values already credited (ms). `prefill` is `setupMs + prefillElapsedMs`. */
  wait: number;
  slot: number;
  prepare: number;
  prefill: number;
  decode: number;
  /** A lane without native progress (a System One decision): every advance credits it to `running`. */
  plain: boolean;
  /** A lane without progress is credited to `running` from here. */
  runningFrom: number;
  /** `max(0, generated - 1)` already credited. */
  outTokens: number;
  /** Kept tokens credited since the last advance, not yet paired with decode time. */
  newTokens: number;
  /** `prefillDone` already credited (rated lanes). */
  prefillTokens: number;
  /** Prefill ms already paired with those tokens. */
  prefillRatedMs: number;
  /** Where the native prefill window started. */
  prefillStartAt: number | null;
  /** Where native decode started, and where it ended once the turn is done. */
  decodeStartAt: number | null;
  decodeEndAt: number | null;
  /** The lane's decode time up to here is paired with its tokens. */
  pairedTo: number | null;
  endAt: number | null;
}

/** One model load: wall clock, `endedAt` null while it runs. */
export interface LoadSpan {
  startedAt: number;
  endedAt: number | null;
}

export interface WorkRates {
  windowMs: number;
  /** Output tokens per ms with at least one lane decoding, ×1000; null without decode in the window. */
  outputTps: number | null;
  /** Rated prefill tokens per lane prefill ms, ×1000; null without a rated prefill credit in the window. */
  prefillTps: number | null;
}

/** Same value as the dashboard's `PREFILL_MIN_TOKENS`: below it the fixed cost inside the window dominates. */
export const RATED_PREFILL_MIN_TOKENS = 256;
/** The live rates cover the last this-many ms of work. */
export const RATE_WINDOW_MS = 10_000;
/** A rate needs at least this much phase time in its window. */
export const RATE_MIN_MS = 250;
/** Closed lanes kept for the next {@link WorkLedger.advance}; bounds memory when nothing advances the ledger. */
const CLOSED_LANES_MAX = 256;
/** Credits kept for the window; bounds memory when nothing advances the ledger. */
const CREDITS_MAX = 4096;

/** Tokens and the interval of the work that produced them. */
interface Credit {
  start: number;
  end: number;
  tokens: number;
}

type Span = [number, number];

function isMs(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0;
}

function zeroMs(): WorkMs {
  return { loading: 0, waiting: 0, prepare: 0, prefill: 0, decode: 0, running: 0 };
}

function copyTotals(totals: Readonly<WorkTotals>): WorkTotals {
  return {
    ms: { ...totals.ms },
    outTokens: totals.outTokens,
    outMs: totals.outMs,
    prefillTokens: totals.prefillTokens,
    prefillMs: totals.prefillMs,
  };
}

/** The time the native phase read `prefill`: a whole turn's setup plus its prefill window. */
function prefillPhaseMs(snap: TurnProgressSnapshot): number | undefined {
  const window = snap.prefillElapsedMs;
  if (!isMs(window)) return undefined;
  return isMs(snap.setupMs) ? snap.setupMs + window : window;
}

/** Sorted, disjoint union of `spans`. */
function union(spans: Span[]): Span[] {
  spans.sort((a, b) => a[0] - b[0]);
  const out: Span[] = [];
  for (const [s, e] of spans) {
    const last = out.at(-1);
    if (last !== undefined && s <= last[1]) {
      if (e > last[1]) last[1] = e;
    } else out.push([s, e]);
  }
  return out;
}

/** Length of `spans` inside `[from, to]`. */
function lengthIn(spans: ReadonlyArray<Span>, from: number, to: number): number {
  let total = 0;
  for (const [s, e] of spans) {
    const length = Math.min(e, to) - Math.max(s, from);
    if (length > 0) total += length;
  }
  return total;
}

/** The share of a credit's tokens whose work falls inside `[from, to]`, the tokens spread evenly over its interval. */
function tokensIn(credit: Credit, from: number, to: number): number {
  const length = credit.end - credit.start;
  if (length <= 0) return credit.end > from && credit.end <= to ? credit.tokens : 0;
  return (credit.tokens * lengthIn([[credit.start, credit.end]], from, to)) / length;
}

function pushCredit(credits: Credit[], credit: Credit): void {
  if (credits.length >= CREDITS_MAX) credits.shift();
  credits.push(credit);
}

function rateOf(tokens: number, ms: number): number | null {
  if (!(ms >= RATE_MIN_MS) || !(tokens > 0)) return null;
  const rate = (tokens * 1000) / ms;
  if (!Number.isFinite(rate)) return null;
  const rounded = Math.round(rate * 100) / 100;
  return rounded > 0 ? rounded : rate;
}

export class WorkLedger {
  private readonly sum: WorkTotals = {
    ms: zeroMs(),
    outTokens: 0,
    outMs: 0,
    prefillTokens: 0,
    prefillMs: 0,
  };
  private readonly live = new Set<LaneMark>();
  private readonly closed: LaneMark[] = [];
  private cursor: number | null = null;
  /** Union of the decode time paired with tokens; kept as far back as a live lane can still pair, and one window. */
  private paired: Span[] = [];
  private readonly outCredits: Credit[] = [];
  private readonly prefillCredits: Credit[] = [];

  /** A new lane, credited from `now`. `plain`: it has no native progress, and every advance credits it to running. */
  open(now: number, plain = false): LaneMark {
    this.cursor ??= now;
    const mark: LaneMark = {
      wait: 0,
      slot: 0,
      prepare: 0,
      prefill: 0,
      decode: 0,
      plain,
      runningFrom: now,
      outTokens: 0,
      newTokens: 0,
      prefillTokens: 0,
      prefillRatedMs: 0,
      prefillStartAt: null,
      decodeStartAt: null,
      decodeEndAt: null,
      pairedTo: null,
      endAt: null,
    };
    this.live.add(mark);
    return mark;
  }

  /** Credit what the lane did since its mark. `snap` null: the lane has no native progress. */
  observe(mark: LaneMark, snap: TurnProgressSnapshot | null, now: number): void {
    if (mark.endAt !== null) return;
    if (snap === null) {
      if (now > mark.runningFrom) this.sum.ms.running += now - mark.runningFrom;
      mark.runningFrom = now;
      return;
    }
    mark.runningFrom = now;
    // The instant the native clocks read: now, or the end of the turn once it is done.
    const at = isMs(snap.sinceDoneMs) ? now - snap.sinceDoneMs : now;
    const ms = this.sum.ms;
    ms.waiting += this.clock(mark, 'wait', snap.waitMs);
    ms.waiting += this.clock(mark, 'slot', snap.slotMs);
    ms.prepare += this.clock(mark, 'prepare', snap.prepareMs);
    ms.prefill += this.clock(mark, 'prefill', prefillPhaseMs(snap));
    const decode = snap.decodeElapsedMs;
    ms.decode += this.clock(mark, 'decode', decode);
    if (isMs(decode)) {
      mark.decodeStartAt ??= at - decode;
      if (snap.phase === 'done') mark.decodeEndAt ??= mark.decodeStartAt + decode;
    }
    const generated = snap.generatedTokens;
    if (typeof generated === 'number' && Number.isFinite(generated)) this.creditOutput(mark, generated);
    this.creditPrefill(mark, snap, at);
  }

  /**
   * The lane ended: a last {@link observe} with its final snapshot, then the final token count of a stream. The mark
   * stays until the next {@link advance}, which pairs its last decode time.
   */
  close(
    mark: LaneMark,
    snap: TurnProgressSnapshot | null,
    now: number,
    finalTokens: number | undefined,
    stream: boolean,
  ): void {
    if (mark.endAt !== null) return;
    this.observe(mark, snap, now);
    if (stream && mark.decodeStartAt !== null && typeof finalTokens === 'number' && Number.isFinite(finalTokens)) {
      this.creditOutput(mark, finalTokens);
    }
    mark.endAt = now;
    if (mark.decodeStartAt !== null) mark.decodeEndAt ??= Math.max(mark.decodeStartAt, now);
    if (!this.live.delete(mark)) return;
    this.closed.push(mark);
    if (this.closed.length > CLOSED_LANES_MAX) this.closed.shift();
  }

  /**
   * Fold up to `now`: the decode time each lane's new tokens pair with (counted once where lanes overlap), the
   * running time of lanes without progress, and the wall time with a load in flight.
   */
  advance(now: number, loads: ReadonlyArray<LoadSpan>): WorkTotals {
    const from = this.cursor ?? now;
    const fresh: Span[] = [];
    for (const lanes of [this.live, this.closed]) {
      for (const lane of lanes) {
        if (lane.plain && lane.endAt === null && now > lane.runningFrom) {
          this.sum.ms.running += now - lane.runningFrom;
          lane.runningFrom = now;
        }
        this.pair(lane, now, fresh);
      }
    }
    if (fresh.length > 0) {
      const before = lengthIn(this.paired, -Infinity, Infinity);
      this.paired = union([...this.paired, ...fresh]);
      this.sum.outMs += lengthIn(this.paired, -Infinity, Infinity) - before;
    }
    const loading: Span[] = [];
    for (const load of loads) {
      const s = Math.max(load.startedAt, from);
      const e = Math.min(load.endedAt ?? now, now);
      if (e > s) loading.push([s, e]);
    }
    this.sum.ms.loading += lengthIn(union(loading), -Infinity, Infinity);
    this.closed.length = 0;
    this.cursor = now;
    this.prune(now);
    return copyTotals(this.sum);
  }

  /** The rates over the last {@link RATE_WINDOW_MS} before `now`, from the credits folded so far. */
  rates(now: number): WorkRates {
    const from = now - RATE_WINDOW_MS;
    let outTokens = 0;
    for (const credit of this.outCredits) outTokens += tokensIn(credit, from, now);
    let prefillTokens = 0;
    let prefillMs = 0;
    for (const credit of this.prefillCredits) {
      prefillTokens += tokensIn(credit, from, now);
      prefillMs += lengthIn([[credit.start, credit.end]], from, now);
    }
    return {
      windowMs: RATE_WINDOW_MS,
      outputTps: rateOf(outTokens, lengthIn(this.paired, from, now)),
      prefillTps: rateOf(prefillTokens, prefillMs),
    };
  }

  /** The counters as they stand, without folding anything. */
  totals(): WorkTotals {
    return copyTotals(this.sum);
  }

  /**
   * Pair the lane's decode time since its last pairing with its new tokens. A closed lane that kept tokens also pairs
   * its last stretch without new ones; a lane without new tokens keeps its time for them.
   */
  private pair(lane: LaneMark, now: number, fresh: Span[]): void {
    const start = lane.decodeStartAt;
    if (start === null) return;
    const tokens = lane.newTokens;
    if (tokens === 0 && (lane.endAt === null || lane.outTokens === 0)) return;
    const from = lane.pairedTo ?? start;
    const end = Math.max(from, Math.min(lane.decodeEndAt ?? now, now));
    if (end > from) fresh.push([from, end]);
    lane.pairedTo = end;
    lane.newTokens = 0;
    if (tokens > 0) {
      this.sum.outTokens += tokens;
      pushCredit(this.outCredits, { start: from, end, tokens });
    }
  }

  /** Drop what neither a live lane nor the window can reach any more. */
  private prune(now: number): void {
    let keep = now - RATE_WINDOW_MS;
    for (const lane of this.live) {
      if (lane.decodeStartAt !== null) keep = Math.min(keep, lane.pairedTo ?? lane.decodeStartAt);
    }
    this.paired = this.paired.filter((span) => span[1] > keep);
    const cutoff = now - RATE_WINDOW_MS;
    for (const credits of [this.outCredits, this.prefillCredits]) {
      const kept = credits.filter((credit) => credit.end > cutoff);
      credits.length = 0;
      credits.push(...kept);
    }
  }

  /** The advance of one native clock since the mark, never negative. */
  private clock(mark: LaneMark, key: 'wait' | 'slot' | 'prepare' | 'prefill' | 'decode', value: unknown): number {
    if (!isMs(value) || value <= mark[key]) return 0;
    const delta = value - mark[key];
    mark[key] = value;
    return delta;
  }

  private creditOutput(mark: LaneMark, generated: number): void {
    const kept = Math.max(0, Math.floor(generated) - 1);
    if (kept <= mark.outTokens) return;
    mark.newTokens += kept - mark.outTokens;
    mark.outTokens = kept;
  }

  /** `at`: the instant the snapshot's clocks read. */
  private creditPrefill(mark: LaneMark, snap: TurnProgressSnapshot, at: number): void {
    const total = snap.prefillTotal;
    const elapsed = snap.prefillElapsedMs;
    if (typeof total !== 'number' || total < RATED_PREFILL_MIN_TOKENS || !isMs(elapsed)) return;
    const decode = snap.decodeElapsedMs;
    const done = isMs(decode) ? total : snap.prefillDone;
    if (typeof done !== 'number' || !Number.isFinite(done) || done <= mark.prefillTokens) return;
    mark.prefillStartAt ??= (isMs(decode) ? at - decode : at) - elapsed;
    const tokens = done - mark.prefillTokens;
    const ms = Math.max(0, elapsed - mark.prefillRatedMs);
    this.sum.prefillTokens += tokens;
    this.sum.prefillMs += ms;
    const start = mark.prefillStartAt + mark.prefillRatedMs;
    pushCredit(this.prefillCredits, { start, end: start + ms, tokens });
    mark.prefillTokens = done;
    mark.prefillRatedMs += ms;
  }
}

/** The process-wide ledger: every traced turn credits it, the runtime sampler advances it. */
export const WORK = new WorkLedger();
