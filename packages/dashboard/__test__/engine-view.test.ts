import { describe, expect, expectTypeOf, it } from 'vite-plus/test';

import type * as Worker from '../src/api/handlers/runtime.js';
import {
  batchUse,
  historyTotals,
  memoryPoints,
  nativeErrorText,
  nowItems,
  occupancyPoints,
  pickProcess,
  poolSegments,
  procKeyOf,
  recentSpark,
  seriesStarts,
  sparkPoints,
  tickerChip,
  tickerDelta,
} from '../ui/src/lib/engine-view.js';
import type { LiveSample } from '../ui/src/lib/live-history.js';
import type * as Page from '../ui/src/lib/types.js';
import { GIB, live, process, runtimeSeries, sched } from './helpers/observability-fixtures.js';

function sample(t: number, outTps: number | null, pid: number | null = 812): LiveSample {
  return { t, pid, outTps, memActive: 10 * GIB, memCache: GIB, inFlight: 1, queued: 0 };
}

describe('pickProcess', () => {
  it('prefers the asked pid, then the first running, then the first unresponsive', () => {
    const a = process({ pid: 1, status: 'unresponsive' });
    const b = process({ pid: 2, status: 'running' });
    const c = process({ pid: 3, status: 'dead' });
    const snapshot = live({ processes: [a, b, c] });
    expect(pickProcess(snapshot, 1)?.pid).toBe(1);
    expect(pickProcess(snapshot, null)?.pid).toBe(2);
    expect(pickProcess(snapshot, 3)?.pid).toBe(2);
    expect(pickProcess(live({ processes: [a, c] }), null)?.pid).toBe(1);
    expect(pickProcess(live({ processes: [c] }), null)).toBeNull();
    expect(pickProcess(undefined, null)).toBeNull();
  });
});

describe('pool and batching', () => {
  it('counts in use as allocated minus reclaimable', () => {
    const pool = poolSegments(
      sched({ allocatedBlocks: 30, reclaimableBlocks: 5, reservedBlocks: 3, watermarkBlocks: 0 }),
    );
    expect(pool.segments.map((segment) => [segment.key, segment.value])).toEqual([
      ['used', 25],
      ['reusable', 5],
    ]);
    expect(pool.reserved).toBe(3);
    expect(pool.threshold).toBe(0);
  });

  it('takes the step-weighted mean batch', () => {
    const view = occupancyPoints({
      ...runtimeSeries().sched[0].delta,
      occupancy: [
        { occupancy: 4, steps: 100 },
        { occupancy: 1, steps: 100 },
        { occupancy: 2, steps: 300 },
      ],
    });
    expect(view.data.map((point) => point.label)).toEqual(['Batch of 1', 'Batch of 2', 'Batch of 4']);
    expect(view.steps).toBe(500);
    expect(view.mean).toBeCloseTo(2.2, 6);
    expect(occupancyPoints({ ...runtimeSeries().sched[0].delta, occupancy: [] }).mean).toBeNull();
  });
});

describe('historyTotals', () => {
  it('sums measured busy time and never reads more than the observed time', () => {
    const series = runtimeSeries();
    const base = series.points[0];
    const now = series.range.to;
    series.points = [
      { ...base, t: series.range.from, samples: 6, busyMs: 90_000 },
      { ...base, t: series.range.from + 60_000, samples: 0, busyMs: 0 },
      { ...base, t: series.range.from + 120_000, samples: 3, busyMs: 70_000 },
    ];
    const totals = historyTotals(series, now);
    expect(totals.busyMs).toBe(160_000);
    expect(totals.coveredMs).toBe(120_000);
    expect(Math.min(1, totals.busyMs / totals.coveredMs)).toBe(1);
    expect(totals.rejected).toBe(6);
    expect(totals.loads).toBe(1);
    expect(totals.failedLoads).toBe(0);
    series.points = [{ ...base, t: now - 15_000, samples: 2, busyMs: 9_000 }];
    expect(historyTotals(series, now).coveredMs).toBe(15_000);
  });

  it('counts a process with no exit that is not alive as unclean', () => {
    const series = runtimeSeries();
    const key = series.procs[0].procKey;
    expect(historyTotals(series, series.range.to).unclean).toBe(1);
    expect(historyTotals(series, series.range.to, new Set([key])).unclean).toBe(0);
  });
});

describe('memoryPoints', () => {
  it('leaves buckets with no sample as gaps and picks one byte unit', () => {
    const series = runtimeSeries();
    const starts = [series.points[0].t - 60_000, series.points[0].t];
    const view = memoryPoints(series, starts, 30 * GIB);
    expect(view.unit).toBe('GB');
    expect(view.data[0].values).toEqual({ active: null, cache: null, limit: null });
    expect(view.data[1].values).toEqual({ active: 18, cache: 2, limit: 30 });
    expect(view.data[1].samples).toBeUndefined();
  });
});

describe('nativeErrorText', () => {
  it('prints 20+ at the cap', () => {
    expect(nativeErrorText({ nativeErrorCount: 20, nativeErrorsCapped: true })).toBe('20+');
    expect(nativeErrorText({ nativeErrorCount: 3, nativeErrorsCapped: false })).toBe('3');
  });
});

describe('the page copies of the runtime series types', () => {
  it('mirror the worker types exactly', () => {
    // Checked by `tsc`: a field added to or changed on either side fails the typecheck.
    expectTypeOf<Page.SchedSnapshot>().toEqualTypeOf<Worker.SchedSnapshot>();
    // `branded`: the worker's delta is an intersection; compare the resolved shape.
    expectTypeOf<Page.RuntimeSeries['sched']>().branded.toEqualTypeOf<Worker.RuntimeSeries['sched']>();
  });
});

describe('batchUse', () => {
  const entry = (delta: { admitted?: number; globalSteps?: number; turns?: number }, turns: number | null = 9) => {
    const base = runtimeSeries().sched[0];
    return {
      ...base,
      latest: sched({ turns }),
      delta: { ...base.delta, admitted: 0, globalSteps: 0, turns: 0, ...delta },
    };
  };

  it('reads batched, alone or quiet from the growth in the range', () => {
    expect(batchUse(entry({ admitted: 2 }))).toBe('batched');
    expect(batchUse(entry({ globalSteps: 40 }))).toBe('batched');
    expect(batchUse(entry({ turns: 9 }))).toBe('alone');
    expect(batchUse(entry({}))).toBe('quiet');
    expect(batchUse(entry({ turns: 0 }, null))).toBe('quiet');
  });
});

describe('ticker', () => {
  const NOW = 1_000_000_000;

  it('builds six items in order', () => {
    const items = nowItems(process({ startedAt: NOW - 3_600_000 }), [], NOW);
    expect(items.map((item) => item.key)).toEqual(['out', 'flight', 'queued', 'mem', 'turns', 'uptime']);
    expect(items[0].value).toBe('42.7');
    expect(items[3].value).toBe('18.0');
    expect(items[3].unit).toBe('GB');
    expect(items[4].value).toBe('1,204');
    expect(items[5].value).toBe('1h 0m');
  });

  it('never has more than 60 spark points and keeps an empty bucket null', () => {
    const history: LiveSample[] = [];
    for (let i = 0; i < 600; i += 1) history.push(sample(NOW - 599_000 + i * 1000, i < 100 ? null : 10));
    const points = sparkPoints(history, (s) => s.outTps, NOW, 812);
    expect(points).toHaveLength(60);
    expect(points[0]).toBeNull();
    expect(points[59]).toBe(10);
    const sparse = sparkPoints([sample(NOW - 5_000, 4)], (s) => s.outTps, NOW, 812);
    expect(sparse.filter((value) => value !== null)).toEqual([4]);
  });

  it('gives no delta in the first minute or after a gap, else the change over 60 s', () => {
    const pick = (s: LiveSample) => s.outTps;
    expect(tickerDelta([sample(NOW - 30_000, 5), sample(NOW, 9)], pick, NOW, 812)).toBeNull();
    expect(tickerDelta([sample(NOW - 60_000, 5), sample(NOW, 9)], pick, NOW, 812)).toBe(4);
    expect(tickerDelta([sample(NOW - 90_000, 5), sample(NOW, 9)], pick, NOW, 812)).toBeNull();
    expect(tickerDelta([sample(NOW - 60_000, 5, 1), sample(NOW, 9, 2)], pick, NOW, 2)).toBeNull();
    expect(tickerDelta([sample(NOW - 60_000, 40), sample(NOW, 78.9)], pick, NOW, 812)).toBeCloseTo(38.9, 9);
    expect(tickerChip(4.24, 1, 'up')).toEqual({ text: '+4.2', direction: 'up', goodWhen: 'up' });
    expect(tickerChip(-1, 0, 'down')).toEqual({ text: '−1', direction: 'down', goodWhen: 'down' });
    expect(tickerChip(0.01, 1, 'neutral')).toEqual({ text: '0', direction: 'flat', goodWhen: 'neutral' });
  });

  it('gives no delta when either reading is 0 or missing: a change from or to nothing is not a trend', () => {
    const pick = (s: LiveSample) => s.outTps;
    expect(tickerDelta([sample(NOW - 60_000, 0), sample(NOW, 78.9)], pick, NOW, 812)).toBeNull();
    expect(tickerDelta([sample(NOW - 60_000, 78.9), sample(NOW, 0)], pick, NOW, 812)).toBeNull();
    expect(tickerDelta([sample(NOW - 60_000, null), sample(NOW, 78.9)], pick, NOW, 812)).toBeNull();
    expect(tickerDelta([sample(NOW - 54_000, 40), sample(NOW, 78.9)], pick, NOW, 812)).toBeNull();
    const items = nowItems(process({ decodeTps: 78.9 }), [sample(NOW - 60_000, 0), sample(NOW, 78.9)], NOW);
    expect(items[0].delta).toBeNull();
    expect('delta' in items[0]).toBe(true);
  });

  it('reads the delta and the spark of the picked pid only', () => {
    const history: LiveSample[] = [];
    for (let i = 0; i <= 70; i += 1) history.push(sample(NOW - 70_000 + i * 1000, i < 10 ? 0 : 60, 111));
    history.push(sample(NOW - 61_000, 3, 222), sample(NOW - 2_000, 5, 222));
    const other = nowItems(process({ pid: 333, startedAt: NOW - 3_600_000 }), history, NOW);
    expect(other[0].delta).toBeNull();
    expect(other[0].spark?.data.filter((value) => value !== null)).toEqual([]);
    const own = nowItems(process({ pid: 222, startedAt: NOW - 3_600_000 }), history, NOW);
    expect(own[0].delta).toEqual({ text: '+2.0', direction: 'up', goodWhen: 'up' });
  });

  it('keeps the recent spark at a fixed length, breaks it at a gap and reads one pid', () => {
    const pick = (s: LiveSample) => s.outTps;
    const history = [sample(1_000, 1), sample(2_000, 2, 900), sample(2_000, 3), sample(300_000, 4), sample(301_000, 5)];
    expect(recentSpark(history, pick, 812, 6, 20_000)).toEqual([null, 1, 3, null, 4, 5]);
    expect(recentSpark(history, pick, 812, 3, 20_000)).toEqual([null, 4, 5]);
    expect(recentSpark(history, pick, 900, 3, 20_000)).toEqual([null, null, 2]);
    expect(recentSpark([], pick, 812, 2, 20_000)).toEqual([null, null]);
  });

  it('keys a process by start and pid', () => {
    expect(procKeyOf({ startedAt: 5, pid: 7 })).toBe('5-7');
  });
});

describe('seriesStarts', () => {
  it('draws a held one-minute answer on its own minutes, not on the five-minute grid asked next', () => {
    const minute = runtimeSeries();
    const starts = seriesStarts(minute);
    expect(starts).toHaveLength(60);
    expect(starts[1] - starts[0]).toBe(60_000);
    const drawn = memoryPoints(minute, starts, null).data.filter((point) => point.values.active !== null);
    expect(drawn.length).toBe(minute.points.filter((point) => point.samples > 0).length);
  });
});
