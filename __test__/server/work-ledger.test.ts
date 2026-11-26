/**
 * The work ledger: lane time by phase from the native clocks, and the token/time pairs behind the live Output rate and
 * Prefill speed. The replay drives the ledger and the rate window over the Work loop fixture (traces 2420-2466) at the
 * sampler's tick rule.
 */

import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vite-plus/test';

import { PREFILL_MIN_TOKENS } from '../../packages/dashboard/src/prefill-rule.js';
import { DecisionRegistry, type SystemOneModel } from '../../packages/server/src/decision-registry.js';
import type { TurnProgressSnapshot } from '../../packages/server/src/native-progress.js';
import {
  RATE_MIN_MS,
  RATE_WINDOW_MS,
  RATED_PREFILL_MIN_TOKENS,
  WorkLedger,
  type LaneMark,
  type WorkRates,
  type WorkTotals,
} from '../../packages/server/src/work-ledger.js';

function snap(fields: Partial<TurnProgressSnapshot>): TurnProgressSnapshot {
  return { phase: 'decode', ...fields };
}

/** Every counter of `b` is at least the one of `a`. */
function expectNoDecrease(a: WorkTotals, b: WorkTotals): void {
  for (const key of ['loading', 'waiting', 'prepare', 'prefill', 'decode', 'running'] as const) {
    expect(b.ms[key]).toBeGreaterThanOrEqual(a.ms[key]);
  }
  for (const key of ['outTokens', 'outMs', 'prefillTokens', 'prefillMs'] as const) {
    expect(b[key]).toBeGreaterThanOrEqual(a[key]);
  }
}

describe('WorkLedger', () => {
  it('uses the same prefill floor as the dashboard', () => {
    expect(RATED_PREFILL_MIN_TOKENS).toBe(PREFILL_MIN_TOKENS);
  });

  // Rule 1. MUTATION: credit the snapshot's clock instead of its delta -> the second read of the same snapshot
  // doubles every state -> red.
  it('credits each native clock delta to its state, so a prefill between two ticks counts in full', () => {
    const ledger = new WorkLedger();
    const lane = ledger.open(0);
    ledger.observe(lane, snap({ phase: 'slot', waitMs: 40, slotMs: 60 }), 100);
    // The whole 300 ms prefill and the first 200 ms of decode happened between the two reads.
    const decoding = snap({
      phase: 'decode',
      waitMs: 40,
      slotMs: 60,
      prepareMs: 200,
      prefillElapsedMs: 300,
      decodeElapsedMs: 200,
    });
    ledger.observe(lane, decoding, 1100);
    ledger.observe(lane, decoding, 1100);
    expect(ledger.totals().ms).toEqual({
      loading: 0,
      waiting: 100,
      prepare: 200,
      prefill: 300,
      decode: 200,
      running: 0,
    });
    expect([lane.wait, lane.slot, lane.prepare, lane.prefill, lane.decode]).toEqual([40, 60, 200, 300, 200]);
  });

  // Rule 2. MUTATION: leave `runningFrom` where it was -> the second read credits the first second again -> red.
  it('credits a lane without progress to running, from its last read', () => {
    const ledger = new WorkLedger();
    const lane = ledger.open(1000);
    ledger.observe(lane, null, 2000);
    expect(ledger.totals().ms.running).toBe(1000);
    ledger.observe(lane, null, 2500);
    expect(ledger.totals().ms.running).toBe(1500);
    ledger.close(lane, null, 3000, 50, false);
    expect(ledger.totals().ms.running).toBe(2000);
    // No decode clock: neither its time nor its tokens reach the output pair.
    const totals = ledger.advance(3500, []);
    expect([totals.outTokens, totals.outMs]).toEqual([0, 0]);
  });

  // Rule 3. MUTATION: assign `decodeStartAt` on every read -> the read delay moves it to 751 -> red.
  it('fixes the decode start at the first snapshot that has a decode clock', () => {
    const ledger = new WorkLedger();
    const lane = ledger.open(0);
    ledger.observe(lane, snap({ phase: 'prefill', prefillElapsedMs: 500 }), 500);
    expect(lane.decodeStartAt).toBeNull();
    ledger.observe(lane, snap({ decodeElapsedMs: 250, generatedTokens: 3 }), 1000);
    expect(lane.decodeStartAt).toBe(750);
    ledger.observe(lane, snap({ decodeElapsedMs: 1249, generatedTokens: 60 }), 2000);
    expect(lane.decodeStartAt).toBe(750);
  });

  // Rule 4. MUTATION: add the union time to `outMs` every advance -> the rate sags to 0 tok/s while a core holds its
  // tokens back -> red.
  it('credits decode time only together with tokens, so a core that counts at the end adds both at once', () => {
    const ledger = new WorkLedger();
    const lane = ledger.open(0);
    for (const t of [1000, 2000]) {
      ledger.observe(lane, snap({ decodeElapsedMs: t, generatedTokens: 0 }), t);
      const totals = ledger.advance(t, []);
      expect([totals.outTokens, totals.outMs]).toEqual([0, 0]);
    }
    ledger.close(lane, snap({ phase: 'done', decodeElapsedMs: 2500, generatedTokens: 0 }), 2500, 101, true);
    const totals = ledger.advance(3000, []);
    expect([totals.outTokens, totals.outMs]).toEqual([100, 2500]);
  });

  // Rule 4. MUTATION "sum lane decode ms instead of the union": outMs 5000 -> red.
  it('pairs the output tokens of overlapping lanes with the union of their decode time', () => {
    const ledger = new WorkLedger();
    const a = ledger.open(0);
    const b = ledger.open(0);
    ledger.observe(a, snap({ decodeElapsedMs: 1000, generatedTokens: 51 }), 1000);
    ledger.observe(b, snap({ phase: 'prefill', prefillElapsedMs: 1000 }), 1000);
    ledger.advance(1000, []);
    ledger.observe(a, snap({ decodeElapsedMs: 3000, generatedTokens: 151 }), 3000);
    ledger.observe(b, snap({ decodeElapsedMs: 2000, generatedTokens: 101 }), 3000);
    const totals = ledger.advance(3000, []);
    expect(totals.outTokens).toBe(250);
    expect(totals.outMs).toBe(3000);
    // Lane-ms is a sum: two lanes decoding for one second are two lane-seconds.
    expect(totals.ms.decode).toBe(5000);
  });

  // Rule 4. MUTATION "count the first token": 1 / 11 / 21 instead of 0 / 10 / 20 -> red.
  it("leaves each lane's first token out of the output pair", () => {
    const ledger = new WorkLedger();
    const lane = ledger.open(0);
    const seen: number[] = [];
    ledger.observe(lane, snap({ decodeElapsedMs: 0, generatedTokens: 1 }), 1000);
    seen.push(ledger.advance(1000, []).outTokens);
    ledger.observe(lane, snap({ decodeElapsedMs: 1000, generatedTokens: 11 }), 2000);
    seen.push(ledger.advance(2000, []).outTokens);
    ledger.close(lane, snap({ phase: 'done', decodeElapsedMs: 2000, generatedTokens: 11 }), 3000, 21, true);
    seen.push(ledger.advance(3000, []).outTokens);
    expect(seen).toEqual([0, 10, 20]);
  });

  // Rule 5. MUTATION "credit prefill time without tokens": prefillMs 400 after the first read -> red.
  it('credits prefill tokens and their time together, and only for prefills of 256+ tokens', () => {
    const ledger = new WorkLedger();
    const oneShot = ledger.open(0);
    ledger.observe(oneShot, snap({ phase: 'prefill', prefillTotal: 1000, prefillDone: 0, prefillElapsedMs: 400 }), 500);
    expect([ledger.totals().prefillTokens, ledger.totals().prefillMs]).toEqual([0, 0]);
    ledger.observe(
      oneShot,
      snap({ prefillTotal: 1000, prefillDone: 1000, prefillElapsedMs: 900, decodeElapsedMs: 500, generatedTokens: 5 }),
      1500,
    );
    expect([ledger.totals().prefillTokens, ledger.totals().prefillMs]).toEqual([1000, 900]);

    const sliced = ledger.open(0);
    const prefill = (done: number, elapsed: number) =>
      snap({ phase: 'prefill', prefillTotal: 4096, prefillDone: done, prefillElapsedMs: elapsed });
    ledger.observe(sliced, prefill(2048, 1000), 1000);
    expect([ledger.totals().prefillTokens, ledger.totals().prefillMs]).toEqual([3048, 1900]);
    ledger.observe(sliced, prefill(2048, 1500), 1500);
    expect([ledger.totals().prefillTokens, ledger.totals().prefillMs]).toEqual([3048, 1900]);
    ledger.observe(
      sliced,
      snap({ prefillTotal: 4096, prefillDone: 4096, prefillElapsedMs: 2000, decodeElapsedMs: 10, generatedTokens: 1 }),
      2010,
    );
    expect([ledger.totals().prefillTokens, ledger.totals().prefillMs]).toEqual([5096, 2900]);

    const small = ledger.open(0);
    ledger.observe(
      small,
      snap({ prefillTotal: 255, prefillDone: 255, prefillElapsedMs: 100, decodeElapsedMs: 1, generatedTokens: 1 }),
      101,
    );
    expect([ledger.totals().prefillTokens, ledger.totals().prefillMs]).toEqual([5096, 2900]);
    // Its time still counts as prefill time.
    expect(ledger.totals().ms.prefill).toBe(900 + 2000 + 100);
  });

  // Rule 6. MUTATION: skip the final observe in `close` -> decode ms stays 1000 -> red.
  it('closes a lane with its final snapshot and final token count, and then ignores it', () => {
    const ledger = new WorkLedger();
    const lane = ledger.open(0);
    ledger.observe(lane, snap({ phase: 'prefill', prefillTotal: 300, prefillDone: 0, prefillElapsedMs: 500 }), 500);
    // Native reached its first token and finished between two reads.
    const final = snap({
      phase: 'done',
      prefillTotal: 300,
      prefillDone: 300,
      prefillElapsedMs: 800,
      decodeElapsedMs: 1000,
    });
    ledger.close(lane, final, 1800, 41, true);
    expect(lane.endAt).toBe(1800);
    expect(lane.decodeStartAt).toBe(800);
    ledger.observe(lane, snap({ decodeElapsedMs: 5000, generatedTokens: 500 }), 5800);
    ledger.close(lane, snap({ decodeElapsedMs: 5000, generatedTokens: 500 }), 5800, 500, true);
    const totals = ledger.advance(2000, []);
    expect(totals.ms).toMatchObject({ prefill: 800, decode: 1000 });
    expect([totals.outTokens, totals.outMs]).toEqual([40, 1000]);
    expect([totals.prefillTokens, totals.prefillMs]).toEqual([300, 800]);

    // A non-streamed close credits no output pair, whatever its token count.
    const plain = ledger.open(2000);
    ledger.close(plain, null, 2600, 99, false);
    expect(ledger.advance(3000, [])).toMatchObject({ outTokens: 40, outMs: 1000, ms: { running: 600 } });
  });

  // Rule 6. MUTATION "drop a lane closed between two advances": outMs stays 1000 -> red.
  it('folds the decode time of a lane closed between two advances', () => {
    const ledger = new WorkLedger();
    const lane = ledger.open(0);
    ledger.observe(lane, snap({ decodeElapsedMs: 1000, generatedTokens: 51 }), 1000);
    expect(ledger.advance(1000, []).outMs).toBe(1000);
    ledger.close(lane, snap({ phase: 'done', decodeElapsedMs: 1500, generatedTokens: 76 }), 1500, 76, true);
    const totals = ledger.advance(2000, []);
    expect([totals.outTokens, totals.outMs]).toEqual([75, 1500]);
    // Folded once: the next advance has nothing of it.
    expect(ledger.advance(3000, []).outMs).toBe(1500);
  });

  // Rule 7. MUTATION: credit `snap - mark` signed, or let the mark follow a clock that reads back -> the decode read of
  // 50 ms after 100 ms subtracts or credits again -> red.
  it('never decreases a counter, when a read goes back, tokens drop or the wall clock goes back', () => {
    const ledger = new WorkLedger();
    const lane = ledger.open(0);
    const prefill = { prefillTotal: 512, setupMs: 300 };
    const steps: Array<() => void> = [
      () => ledger.observe(lane, snap({ phase: 'prefill', prefillElapsedMs: 300 }), 300),
      () => ledger.observe(lane, snap({ phase: 'prefill', ...prefill, prefillDone: 0, prefillElapsedMs: 20 }), 320),
      () => ledger.observe(lane, snap({ phase: 'prefill', ...prefill, prefillDone: 0, prefillElapsedMs: 500 }), 800),
      () =>
        ledger.observe(
          lane,
          snap({ ...prefill, prefillDone: 512, prefillElapsedMs: 600, decodeElapsedMs: 100, generatedTokens: 9 }),
          1000,
        ),
      () => void ledger.advance(1000, []),
      () =>
        ledger.observe(
          lane,
          snap({ ...prefill, prefillElapsedMs: 600, decodeElapsedMs: 50, generatedTokens: 4 }),
          1100,
        ),
      () => void ledger.advance(900, [{ startedAt: 0, endedAt: null }]),
      () =>
        ledger.observe(
          lane,
          snap({ ...prefill, prefillElapsedMs: 600, decodeElapsedMs: 400, generatedTokens: 20 }),
          1300,
        ),
      () => void ledger.advance(1300, []),
    ];
    let before = ledger.totals();
    for (const step of steps) {
      step();
      const after = ledger.totals();
      expectNoDecrease(before, after);
      before = after;
    }
    expect(before.ms).toMatchObject({ prefill: 900, decode: 400 });
  });

  // Finding: the prefill phase total depended on whether a read landed before `begin_prefill` (600 or 870 ms).
  // MUTATION "credit only the prefill window clock" (drop `setupMs`): 600 with every schedule -> red.
  it('counts a whole turn as prefill from its start, the same whatever reads land around begin_prefill', () => {
    // `begin_turn` at 0, `begin_prefill` at 300, first token at 900.
    const at = (t: number): TurnProgressSnapshot => {
      if (t < 300) return snap({ phase: 'prefill', prefillElapsedMs: t });
      const prefill = { prefillTotal: 512, setupMs: 300 };
      if (t < 900) return snap({ phase: 'prefill', ...prefill, prefillDone: 0, prefillElapsedMs: t - 300 });
      return snap({
        ...prefill,
        prefillDone: 512,
        prefillElapsedMs: 600,
        decodeElapsedMs: t - 900,
        generatedTokens: 2,
      });
    };
    const seen = [[1000], [280, 1000], [280, 310, 1000], [100, 290, 305, 700, 1000]].map((reads) => {
      const ledger = new WorkLedger();
      const lane = ledger.open(0);
      for (const t of reads) ledger.observe(lane, at(t), t);
      const totals = ledger.totals();
      return [totals.ms.prefill, totals.prefillTokens, totals.prefillMs];
    });
    // Setup and window together on the stack; the rated pair keeps the window alone.
    expect(seen).toEqual(Array.from({ length: 4 }, () => [900, 512, 600]));
  });

  // Finding: native kept its clocks running after `finish`, and the union ran on to the JS settle, so a terminal event
  // held by a slow reader counted as decode time (8,000 ms for 2,000) and a later turn at 40 tok/s read 5.7.
  // MUTATION "end a lane's decode at its JS settle, not at the native end": outMs 8000 -> red. MUTATION "place a lane
  // first seen at its end at now - decodeElapsedMs" (ignore `sinceDoneMs`): a rate 40 at 18 800 -> red.
  it('stops a lane at the native end of its turn, also when its terminal event reaches JS late', () => {
    const ledger = new WorkLedger();
    const lane = ledger.open(0);
    ledger.observe(lane, snap({ decodeElapsedMs: 0, generatedTokens: 1 }), 0);
    ledger.advance(0, []);
    ledger.observe(lane, snap({ decodeElapsedMs: 1000, generatedTokens: 41 }), 1000);
    ledger.advance(1000, []);
    // Native finished at 2000; the reader takes the terminal chunk at 8000.
    const done = (t: number) =>
      snap({ phase: 'done', decodeElapsedMs: 2000, sinceDoneMs: t - 2000, generatedTokens: 81 });
    for (let t = 3000; t < 8000; t += 1000) {
      ledger.observe(lane, done(t), t);
      ledger.advance(t, []);
    }
    ledger.close(lane, done(8000), 8000, 81, true);
    const totals = ledger.advance(8000, []);
    expect(totals.ms.decode).toBe(2000);
    expect([totals.outTokens, totals.outMs]).toEqual([80, 2000]);

    // A short turn that ran 8500-9000 and is seen only at its settle, 3 s later.
    const short = ledger.open(8000);
    ledger.close(
      short,
      snap({ phase: 'done', decodeElapsedMs: 500, sinceDoneMs: 3000, generatedTokens: 21 }),
      12_000,
      21,
      true,
    );
    const after = ledger.advance(12_000, []);
    expect([after.outTokens - totals.outTokens, after.outMs - totals.outMs]).toEqual([20, 500]);
    expect(ledger.rates(18_600).outputTps).toBe(40);
    expect(ledger.rates(18_800).outputTps).toBeNull();
  });

  // Finding: held decode time was one process-wide pool, so a lane that ended after its first token lent its second
  // to the next unrelated turn: 20 tok/s for a 40 tok/s turn. MUTATION "pair the decode time of a closed lane without
  // kept tokens": outMs 2000 -> red.
  it('gives the decode time of a lane with no kept token to no rate', () => {
    const ledger = new WorkLedger();
    const lone = ledger.open(0);
    ledger.observe(lone, snap({ decodeElapsedMs: 0, generatedTokens: 1 }), 0);
    ledger.advance(0, []);
    ledger.close(
      lone,
      snap({ phase: 'done', decodeElapsedMs: 1000, sinceDoneMs: 0, generatedTokens: 1 }),
      1000,
      1,
      true,
    );
    for (let t = 1000; t <= 21_000; t += 5000) ledger.advance(t, []);
    const next = ledger.open(21_000);
    ledger.observe(next, snap({ decodeElapsedMs: 0, generatedTokens: 1 }), 21_000);
    ledger.advance(21_000, []);
    ledger.observe(next, snap({ decodeElapsedMs: 1000, generatedTokens: 41 }), 22_000);
    const totals = ledger.advance(22_000, []);
    expect([totals.outTokens, totals.outMs]).toEqual([40, 1000]);
    expect(ledger.rates(22_000).outputTps).toBe(40);
    // Its lane time is still decode time.
    expect(totals.ms.decode).toBe(2000);
  });

  // Finding: System One decisions were charged count x interval at the sampler tick: 5,150 ms for a 250 ms decision
  // after an idle stretch, 0 for one between two ticks. MUTATION "credit plain lanes only at close": running 0 at 5150
  // -> red.
  it('credits a lane without progress to running from its open to its close, at every advance', () => {
    const ledger = new WorkLedger();
    ledger.advance(0, []);
    const late = ledger.open(4900, true);
    expect(ledger.advance(5150, []).ms.running).toBe(250);
    ledger.close(late, null, 5300, undefined, false);
    const short = ledger.open(6000, true);
    ledger.close(short, null, 6100, undefined, false);
    expect(ledger.advance(6250, []).ms.running).toBe(500);
    const long = ledger.open(7000, true);
    expect(ledger.advance(8000, []).ms.running).toBe(1500);
    ledger.close(long, null, 9500, undefined, false);
    expect(ledger.advance(10_000, []).ms.running).toBe(3000);
  });

  it('counts load wall time once while loads overlap', () => {
    const ledger = new WorkLedger();
    ledger.advance(0, []);
    const totals = ledger.advance(10_000, [
      { startedAt: -5000, endedAt: 3000 },
      { startedAt: 2000, endedAt: 4000 },
      { startedAt: 8000, endedAt: null },
    ]);
    expect(totals.ms.loading).toBe(6000);
  });
});

describe('System One decisions', () => {
  // Finding: the sampler charged the decisions it saw active at a tick for the whole interval since the previous tick.
  // MUTATION "open no ledger lane in the decision registry" (the sampled count): running 0 -> red.
  it('run as ledger lanes from their start edge to their settle edge', async () => {
    let clock = 0;
    const ledger = new WorkLedger();
    const registry = new DecisionRegistry(undefined, { work: ledger, now: () => clock });
    registry.register('clef', {} as SystemOneModel);
    const flush = () => new Promise((resolve) => setImmediate(resolve));
    const decide = async (from: number, to: number, before?: () => void) => {
      clock = from;
      let finish!: () => void;
      const run = registry.acquire('clef')!.run(() => new Promise<void>((resolve) => (finish = resolve)));
      await flush();
      before?.();
      clock = to;
      finish();
      await run;
    };
    ledger.advance(0, []);
    // Starts after a long idle stretch, still running at the edge tick.
    await decide(4900, 5300, () => expect(ledger.advance(5150, []).ms.running).toBe(250));
    // Starts and settles between two ticks.
    await decide(6000, 6100);
    expect(ledger.advance(6250, []).ms.running).toBe(500);
  });
});

describe('WorkLedger.rates', () => {
  // Finding: the window anchored at a heartbeat, so credits folded after it counted although their work ran before the
  // cutoff (40 tok/s reported 10.05 s after the last decode). MUTATION "count a credit whole once it reaches into the
  // window": 33.33 at 18 500 -> red.
  it('covers the last 10 s of work, cutting each credit at the window start', () => {
    expect(RATE_WINDOW_MS).toBe(10_000);
    const ledger = new WorkLedger();
    const lane = ledger.open(8000);
    ledger.observe(lane, snap({ decodeElapsedMs: 0, generatedTokens: 1 }), 8000);
    ledger.advance(8000, []);
    // 40 tok/s for a second, then 10 tok/s for a second; the settle edge folds the last second at 10 250.
    ledger.observe(lane, snap({ decodeElapsedMs: 1000, generatedTokens: 41 }), 9000);
    ledger.advance(9000, []);
    ledger.close(
      lane,
      snap({ phase: 'done', decodeElapsedMs: 2000, sinceDoneMs: 0, generatedTokens: 51 }),
      10_000,
      51,
      true,
    );
    ledger.advance(10_250, []);
    ledger.advance(15_250, []);
    // 20 tok/s: 20 of the fast tokens and all 10 slow ones over 1.5 s.
    expect(ledger.rates(18_500).outputTps).toBe(20);
    expect(ledger.rates(19_700).outputTps).toBe(10);
    // Under RATE_MIN_MS of decode left in the window, then none at all.
    expect(ledger.rates(19_800).outputTps).toBeNull();
    const next = ledger.open(19_800);
    ledger.observe(
      next,
      snap({ phase: 'prefill', prefillTotal: 20_000, prefillDone: 0, prefillElapsedMs: 250 }),
      20_050,
    );
    ledger.advance(20_050, []);
    expect(ledger.rates(20_050).outputTps).toBeNull();
  });

  it('is null, never 0, without tokens or with under RATE_MIN_MS of phase time', () => {
    expect(new WorkLedger().rates(0)).toEqual({ windowMs: RATE_WINDOW_MS, outputTps: null, prefillTps: null });
    const ledger = new WorkLedger();
    const lane = ledger.open(0);
    ledger.observe(lane, snap({ decodeElapsedMs: 0, generatedTokens: 1 }), 0);
    ledger.advance(0, []);
    ledger.observe(lane, snap({ decodeElapsedMs: RATE_MIN_MS - 1, generatedTokens: 6 }), RATE_MIN_MS - 1);
    ledger.advance(RATE_MIN_MS - 1, []);
    expect(ledger.rates(RATE_MIN_MS - 1).outputTps).toBeNull();
    const prefill = ledger.open(1000);
    ledger.observe(
      prefill,
      snap({ phase: 'prefill', prefillTotal: 4096, prefillDone: 2048, prefillElapsedMs: 1000 }),
      2000,
    );
    ledger.advance(2000, []);
    expect(ledger.rates(2000)).toEqual({ windowMs: RATE_WINDOW_MS, outputTps: null, prefillTps: 2048 });
  });
});

interface FixtureRow {
  id: number;
  startMs: number;
  durationMs: number;
  ttftMs: number | null;
  prefillMs: number | null;
  prefillTokens: number | null;
  cachedTokens: number;
  outputTokens: number;
  decodeTps: number | null;
}

interface Request {
  start: number;
  end: number;
  /** Prefill start. */
  ps: number;
  /** First token. */
  ds: number;
  ptok: number;
  out: number;
  cached: number;
}

/** The sampler's tick rule: 1 s while a lane is live, 5 s idle, 250 ms after each turn begin and settle. */
const TICK_BUSY_MS = 1000;
const TICK_IDLE_MS = 5000;
const TICK_EDGE_MS = 250;
const PREFILL_SLICE = 2048;

function loadRequests(): Request[] {
  const fixture = JSON.parse(readFileSync(new URL('./fixtures/work-loop.json', import.meta.url), 'utf8')) as {
    rows: FixtureRow[];
  };
  // The cancelled turn has no trace timing: 700 ms to its first token, 650 ms of it prefill, 545 tokens.
  return fixture.rows
    .map((row) => {
      const ds = row.startMs + (row.ttftMs ?? 700);
      return {
        start: row.startMs,
        end: row.startMs + row.durationMs,
        ps: ds - (row.prefillMs ?? 650),
        ds,
        ptok: row.prefillTokens ?? 545,
        out: row.outputTokens,
        cached: row.cachedTokens,
      };
    })
    .sort((a, b) => a.start - b.start);
}

/** Tokens kept by `t`, linear over the decode window. */
function generatedAt(r: Request, t: number): number {
  if (t < r.ds) return 0;
  return Math.min(r.out, Math.trunc((r.out * (t - r.ds)) / Math.max(1, r.end - r.ds)));
}

/** `prefillDone` as native reports it: 2,048-token slices over 2,048 tokens, else 0 until the first token. */
function prefillDoneAt(r: Request, t: number): number {
  if (t >= r.ds) return r.ptok;
  if (t < r.ps || r.ptok <= PREFILL_SLICE) return 0;
  const fraction = (t - r.ps) / Math.max(1, r.ds - r.ps);
  return Math.min(r.ptok, Math.trunc((r.ptok * fraction) / PREFILL_SLICE) * PREFILL_SLICE);
}

function progressAt(r: Request, t: number): TurnProgressSnapshot {
  const base = { waitMs: 0, slotMs: 0, prepareMs: Math.min(t, r.ps) - r.start, cachedTokens: r.cached };
  if (t < r.ps) return { ...base, phase: 'prepare' };
  const prefill = {
    ...base,
    prefillTotal: r.ptok,
    prefillDone: prefillDoneAt(r, t),
    prefillElapsedMs: Math.min(t, r.ds) - r.ps,
  };
  if (t < r.ds) return { ...prefill, phase: 'prefill' };
  return {
    ...prefill,
    phase: t >= r.end ? 'done' : 'decode',
    decodeElapsedMs: t - r.ds,
    generatedTokens: generatedAt(r, t),
  };
}

interface Tick {
  ts: number;
  rates: WorkRates;
}

function replay(reqs: Request[]): { ticks: Tick[]; totals: WorkTotals } {
  const first = reqs[0]!.start - 3000;
  const last = Math.max(...reqs.map((r) => r.end)) + 16_000;
  const events = [
    ...reqs.map((r, i) => ({ t: r.start, kind: 0, i })),
    ...reqs.map((r, i) => ({ t: r.end, kind: 1, i })),
  ].sort((a, b) => a.t - b.t || a.kind - b.kind || a.i - b.i);
  const ledger = new WorkLedger();
  const live = new Map<number, LaneMark>();
  const ticks: Tick[] = [];
  let next = first;
  let ev = 0;
  for (;;) {
    const event = events[ev];
    if (event !== undefined && event.t < next) {
      ev += 1;
      const r = reqs[event.i]!;
      if (event.kind === 0) live.set(event.i, ledger.open(event.t));
      else {
        ledger.close(live.get(event.i)!, progressAt(r, r.end), r.end, r.out, true);
        live.delete(event.i);
      }
      next = Math.min(next, event.t + TICK_EDGE_MS);
      continue;
    }
    const t = next;
    if (t > last) break;
    for (const [i, mark] of live) ledger.observe(mark, progressAt(reqs[i]!, t), t);
    ledger.advance(t, []);
    ticks.push({ ts: t, rates: ledger.rates(t) });
    next = t + (live.size > 0 ? TICK_BUSY_MS : TICK_IDLE_MS);
  }
  return { ticks, totals: ledger.totals() };
}

/** Runs of consecutive non-null values: [runs of 2+, runs of 1]. */
function runs(values: ReadonlyArray<number | null>): [number, number] {
  let long = 0;
  let lone = 0;
  let current = 0;
  for (const value of [...values, null]) {
    if (value !== null) {
      current += 1;
      continue;
    }
    if (current === 1) lone += 1;
    else if (current >= 2) long += 1;
    current = 0;
  }
  return [long, lone];
}

function expectWithin(value: number, target: number, tolerance: number): void {
  expect(Math.abs(value - target)).toBeLessThanOrEqual(tolerance);
}

describe('Work loop replay (traces 2420-2466)', () => {
  const reqs = loadRequests();
  const { ticks, totals } = replay(reqs);

  it('has the fixture: 47 requests, two of them overlapping', () => {
    expect(reqs).toHaveLength(47);
    expect(reqs[1]!.start).toBeLessThan(reqs[0]!.end);
  });

  it('measures the phase time of the loop', () => {
    expectWithin(totals.ms.prefill, 78_800, 78_800 * 0.005);
    expectWithin(totals.ms.decode, 360_100, 360_100 * 0.005);
  });

  it('pairs every kept token after the first with its decode time', () => {
    expect(totals.outTokens).toBe(22_968);
    expectWithin((totals.outTokens * 1000) / totals.outMs, 63.8, 0.5);
  });

  it('pairs the rated prefill tokens with their prefill time', () => {
    expect(totals.prefillTokens).toBe(85_738);
    expectWithin((totals.prefillTokens * 1000) / totals.prefillMs, 1135, 15);
  });

  it('draws the Output rate as three unbroken pieces, no lone point', () => {
    const loopStart = reqs[0]!.start;
    const loopEnd = reqs.at(-1)!.end + 1000;
    const inLoop = ticks.filter((tick) => tick.ts >= loopStart && tick.ts <= loopEnd);
    expect(runs(inLoop.map((tick) => tick.rates.outputTps))).toEqual([3, 0]);
  });

  it('never writes a rate of 0', () => {
    expect(ticks.filter((tick) => tick.rates.outputTps === 0 || tick.rates.prefillTps === 0)).toEqual([]);
  });
});
