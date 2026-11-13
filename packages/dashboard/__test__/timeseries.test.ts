import { describe, expect, it } from 'vite-plus/test';

import { formatCountNoun } from '../ui/src/lib/format.js';
import { alignedRange } from '../ui/src/lib/range.js';
import {
  moreInTableLabel,
  OTHER_KEY,
  otherLabel,
  pivotByModel,
  pivotMeasures,
  topModels,
} from '../ui/src/lib/timeseries.js';
import type { MetricsSummary, MetricsTimeseries } from '../ui/src/lib/types.js';

type TsPoint = MetricsTimeseries['points'][number];
type ByModel = MetricsSummary['byModel'][number];

const NOW = new Date(2026, 10, 11, 14, 37, 0).getTime();
const RANGE = alignedRange('24h', NOW);

function point(t: number, model: string, turns: number, p50: number | null): TsPoint {
  return {
    t,
    model,
    totals: { turns } as TsPoint['totals'],
    decodeTps: { p50 } as TsPoint['decodeTps'],
  } as TsPoint;
}

function byModel(model: string, turns: number): ByModel {
  return { model, totals: { turns } } as ByModel;
}

const turns = (p: TsPoint): number => p.totals.turns;
const speed = (p: TsPoint): number | null => p.decodeTps.p50;

describe('pivotByModel', () => {
  it('a missing bucket is 0 for sums and null for medians', () => {
    const points = [point(RANGE.starts[3], 'a', 4, 42)];
    const sums = pivotByModel(points, RANGE, turns, { empty: 0, models: ['a'], fold: 'drop' }).data;
    const medians = pivotByModel(points, RANGE, speed, { empty: null, models: ['a'], fold: 'drop' }).data;
    expect(sums).toHaveLength(24);
    expect(sums[0].values.a).toBe(0);
    expect(sums[3].values.a).toBe(4);
    expect(medians[0].values.a).toBeNull();
    expect(medians[3].values.a).toBe(42);
  });

  it("fold 'sum' adds an other key with the rest summed", () => {
    const t = RANGE.starts[5];
    const points = [point(t, 'a', 1, null), point(t, 'b', 2, null), point(t, 'c', 3, null)];
    const { data, dropped } = pivotByModel(points, RANGE, turns, { empty: 0, models: ['a'], fold: 'sum' });
    expect(data[5].values[OTHER_KEY]).toBe(5);
    expect(data[0].values[OTHER_KEY]).toBe(0);
    expect(dropped).toEqual(['b', 'c']);
  });

  it("fold 'drop' leaves the rest out and reports their names", () => {
    const t = RANGE.starts[5];
    const points = [point(t, 'a', 1, 10), point(t, 'z', 2, 20)];
    const { data, dropped } = pivotByModel(points, RANGE, speed, { empty: null, models: ['a'], fold: 'drop' });
    expect(Object.keys(data[5].values)).toEqual(['a']);
    expect(dropped).toEqual(['z']);
  });

  it('carries sample counts per key', () => {
    const points = [point(RANGE.starts[2], 'a', 3, 40)];
    const { data } = pivotByModel(points, RANGE, speed, { empty: null, models: ['a'], fold: 'drop', samples: turns });
    expect(data[2].samples?.a).toBe(3);
    expect(data[1].samples?.a).toBe(0);
  });

  it("the last bucket's label ends with ', so far'", () => {
    const { data } = pivotByModel([], RANGE, turns, { empty: 0, models: [], fold: 'drop' });
    expect(data[data.length - 1].label.endsWith(', so far')).toBe(true);
    expect(data[0].label.endsWith(', so far')).toBe(false);
    expect(data[0].key).toBe(String(RANGE.starts[0]));
  });
});

describe('pivotMeasures', () => {
  it('several measures of one row become series', () => {
    const data = pivotMeasures([point(RANGE.starts[1], '', 7, 30)], RANGE, { turns, speed }, null);
    expect(data[1].values).toEqual({ turns: 7, speed: 30 });
    expect(data[0].values).toEqual({ turns: null, speed: null });
  });
});

describe('topModels', () => {
  it('orders by turns, ties by name, and cuts at n', () => {
    const rows = [byModel('b', 5), byModel('a', 5), byModel('c', 9), byModel('d', 1), byModel('idle', 0)];
    expect(topModels(rows)).toEqual(['c', 'a', 'b', 'd']);
    expect(topModels(rows, 2)).toEqual(['c', 'a']);
  });
});

describe('fold labels', () => {
  it('the noun agrees with the count', () => {
    expect(otherLabel(1)).toBe('Other (1 model)');
    expect(otherLabel(3)).toBe('Other (3 models)');
    expect(moreInTableLabel(1)).toBe('+1 model in the table');
    expect(moreInTableLabel(12)).toBe('+12 models in the table');
    expect(formatCountNoun(1, 'turn', 'turns')).toBe('1 turn');
    expect(formatCountNoun(0, 'turn', 'turns')).toBe('0 turns');
    expect(formatCountNoun(1284, 'turn', 'turns')).toBe('1,284 turns');
  });
});
