import { describe, expect, it } from 'vite-plus/test';

import { dailyTurns, joinModelUsage, matchesModel, usageKeys } from '../ui/src/lib/model-usage.js';
import type { LocalModel, MetricsSummary, MetricsTimeseries, MetricsTotals, Quantiles } from '../ui/src/lib/types.js';

const GGUF: LocalModel = {
  name: 'qwen3.8-27b-gguf',
  path: '/models/qwen3.8-27b-gguf',
  modelType: 'qwen3_5',
  quant: 'Q4_K_XL',
  contextWindow: 131072,
  sizeBytes: 17e9,
  fileCount: 2,
  servedIds: ['Qwen3.8-27B-UD-Q4_K_XL', 'Qwen3.8-27B-Q8_0'],
};

const PLAIN: LocalModel = { ...GGUF, name: 'gemma-4-e4b', path: '/models/gemma-4-e4b', servedIds: [] };

function quantiles(p50: number): Quantiles {
  return { n: 10, p05: p50 / 2, p50, p95: p50 * 2 };
}

function totals(turns: number, error = 0, cancelled = 0): MetricsTotals {
  return {
    turns,
    ok: turns - error - cancelled,
    error,
    cancelled,
    prefillMeasured: 0,
    inputTokens: 0,
    cachedTokens: 0,
    outputTokens: 0,
    reasoningTokens: 0,
    queueWallMs: 0,
    loadWaitWallMs: 0,
    resolveWallMs: 0,
    waitWallMs: 0,
    slotWallMs: 0,
    prepareWallMs: 0,
    firstTokenWallMs: 0,
    setupWallMs: 0,
    prefillWallMs: 0,
    decodeWallMs: 0,
    otherWallMs: 0,
    totalWallMs: 0,
  };
}

function row(
  model: string,
  turns: number,
  decode: number,
  lastTs: number,
  extra: { error?: number; ctx?: number } = {},
) {
  return {
    model,
    modelType: null,
    ctxWindow: extra.ctx ?? null,
    totals: totals(turns, extra.error ?? 0),
    latency: {
      ttftMs: quantiles(300),
      decodeTps: quantiles(decode),
      prefillTps: quantiles(500),
      totalMs: quantiles(4000),
      queueMs: quantiles(0),
      ctxFillPct: quantiles(20),
    },
    mtpTurns: 1,
    lastTs,
  };
}

function summary(rows: Array<ReturnType<typeof row>>): MetricsSummary {
  return { byModel: rows } as unknown as MetricsSummary;
}

describe('matchesModel', () => {
  it('a GGUF folder matches its served id and not its own folder name', () => {
    expect(matchesModel(GGUF, 'Qwen3.8-27B-UD-Q4_K_XL')).toBe(true);
    expect(matchesModel(GGUF, 'qwen3.8-27b-gguf')).toBe(false);
  });

  it('a folder with no served id matches its name', () => {
    expect(matchesModel(PLAIN, 'gemma-4-e4b')).toBe(true);
    expect(matchesModel(PLAIN, 'gemma')).toBe(false);
    expect(usageKeys(PLAIN)).toEqual(['gemma-4-e4b']);
    expect(usageKeys(GGUF)).toEqual(['Qwen3.8-27B-UD-Q4_K_XL', 'Qwen3.8-27B-Q8_0']);
  });
});

describe('joinModelUsage', () => {
  it('a GGUF folder whose served id differs from its name gets its usage', () => {
    const map = joinModelUsage([GGUF], summary([row('Qwen3.8-27B-UD-Q4_K_XL', 12, 40, 500, { ctx: 65536 })]));
    expect(map.get('qwen3.8-27b-gguf')).toMatchObject({
      key: 'Qwen3.8-27B-UD-Q4_K_XL',
      keys: ['Qwen3.8-27B-UD-Q4_K_XL'],
      turns: 12,
      ctxWindow: 65536,
    });
  });

  it('two used variants of one folder sum their turns and take quantiles from the one with more turns', () => {
    const map = joinModelUsage(
      [GGUF],
      summary([
        row('Qwen3.8-27B-UD-Q4_K_XL', 5, 40, 900, { error: 1 }),
        row('Qwen3.8-27B-Q8_0', 9, 25, 700, { error: 2 }),
        row('qwen3.8-27b-gguf', 50, 99, 999),
      ]),
    );
    const usage = map.get('qwen3.8-27b-gguf');
    expect(usage).toMatchObject({ key: 'Qwen3.8-27B-Q8_0', turns: 14, failed: 3, mtpTurns: 2, lastTs: 900 });
    expect(usage?.keys).toEqual(['Qwen3.8-27B-UD-Q4_K_XL', 'Qwen3.8-27B-Q8_0']);
    expect(usage?.decodeTps.p50).toBe(25);
  });

  it('a model without a row is absent from the map', () => {
    const map = joinModelUsage([GGUF, PLAIN], summary([row('gemma-4-e4b', 3, 60, 100)]));
    expect(map.has('qwen3.8-27b-gguf')).toBe(false);
    expect(map.get('gemma-4-e4b')?.turns).toBe(3);
    expect(joinModelUsage([GGUF], undefined).size).toBe(0);
  });
});

describe('dailyTurns', () => {
  it('sums over the keys and fills missing days with 0', () => {
    const day = 86_400_000;
    const starts = [0, day, 2 * day];
    const point = (t: number, model: string, turns: number) =>
      ({ t, model, totals: totals(turns) }) as unknown as MetricsTimeseries['points'][number];
    const points = [
      point(0, 'Qwen3.8-27B-UD-Q4_K_XL', 2),
      point(0, 'Qwen3.8-27B-Q8_0', 3),
      point(0, 'other', 50),
      point(2 * day, 'Qwen3.8-27B-Q8_0', 4),
    ];
    expect(dailyTurns(points, usageKeys(GGUF), starts)).toEqual([5, 0, 4]);
  });
});
