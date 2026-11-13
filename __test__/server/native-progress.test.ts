import { describe, expect, it } from 'vite-plus/test';

import {
  loadProgressOf,
  pipelineStatsOf,
  readLoadProgress,
  readPipelineStats,
  readWarmupReport,
  warmupReportOf,
  type WarmupReport,
} from '../../packages/server/src/native-progress.js';

const REPORT: WarmupReport = {
  ms: 3_412.5,
  steps: 50,
  skipped: 2,
  reset: 0,
  pipelines: 212,
  compileMs: 74_900.25,
  peakBytes: 21_474_836_480,
};

describe('native load progress', () => {
  it('passes a whole report through and keeps steps done within the steps', () => {
    expect(loadProgressOf({ phase: 'kernels', steps: 50, stepsDone: 3 })).toEqual({
      phase: 'kernels',
      steps: 50,
      stepsDone: 3,
    });
    expect(loadProgressOf({ phase: 'weights', steps: 0, stepsDone: 0 })).toEqual({
      phase: 'weights',
      steps: 0,
      stepsDone: 0,
    });
    expect(loadProgressOf({ phase: 'kernels', steps: 50, stepsDone: 51 })).toEqual({
      phase: 'kernels',
      steps: 50,
      stepsDone: 50,
    });
  });

  it('reads a missing or broken field as not known, never as 0', () => {
    expect(loadProgressOf({ phase: 'kernels', steps: 50 })).toBeNull();
    expect(loadProgressOf({ phase: 'kernels', stepsDone: 3 })).toBeNull();
    expect(loadProgressOf({ phase: 'kernels', steps: 50, stepsDone: -1 })).toBeNull();
    expect(loadProgressOf({ phase: 'kernels', steps: 50, stepsDone: 1.5 })).toBeNull();
    expect(loadProgressOf({ phase: 'compile', steps: 50, stepsDone: 3 })).toBeNull();
    expect(loadProgressOf(null)).toBeNull();
    expect(loadProgressOf(undefined)).toBeNull();
  });

  it('a path no load reports, or an addon without the export, reads as null', () => {
    expect(readLoadProgress('/no/such/model/path')).toBeNull();
  });
});

describe('native pipeline counter', () => {
  it('passes a whole counter through; a missing field makes it not known', () => {
    const stats = { createdLoading: 212, createdServing: 3, loadingMs: 74_900.5, servingMs: 120 };
    expect(pipelineStatsOf(stats)).toEqual(stats);
    expect(pipelineStatsOf({ createdLoading: 212, createdServing: 3, loadingMs: 74_900.5 })).toBeNull();
    expect(pipelineStatsOf({ createdLoading: 212, loadingMs: 74_900.5, servingMs: 120 })).toBeNull();
    expect(pipelineStatsOf(null)).toBeNull();
  });

  it('reading the counter never throws, and an absent counter reads as null', () => {
    const stats = readPipelineStats();
    if (stats !== null) expect(pipelineStatsOf(stats)).toEqual(stats);
  });
});

describe('native warm-up report', () => {
  it('passes a whole report through', () => {
    expect(warmupReportOf({ ...REPORT })).toEqual(REPORT);
  });

  it.each(Object.keys(REPORT))('a report without %s is not known', (key) => {
    const partial: Record<string, unknown> = { ...REPORT };
    delete partial[key];
    expect(warmupReportOf(partial)).toBeNull();
  });

  it('reads the getter of a loaded model, as the addon class declares it', () => {
    class Loaded {
      get warmup(): WarmupReport | null {
        return { ...REPORT };
      }
    }
    class Flat {
      get warmup(): WarmupReport | null {
        return null;
      }
    }
    class Gone {
      get warmup(): WarmupReport | null {
        throw new Error('model was unloaded');
      }
    }
    expect(readWarmupReport(new Loaded())).toEqual(REPORT);
    expect(readWarmupReport(new Flat())).toBeNull();
    expect(readWarmupReport(new Gone())).toBeNull();
    expect(readWarmupReport({})).toBeNull();
    expect(readWarmupReport(null)).toBeNull();
  });
});
