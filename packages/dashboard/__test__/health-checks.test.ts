import { describe, expect, it } from 'vite-plus/test';

import { healthChecks, type HealthCheck } from '../ui/src/lib/health-checks.js';
import type { ModelsResponse } from '../ui/src/lib/types.js';
import { busyTotals, cacheResponse, GIB, live, process, totals } from './helpers/observability-fixtures.js';

function row(checks: HealthCheck[], key: HealthCheck['key']): Pick<HealthCheck, 'tone' | 'text'> {
  const found = checks.find((check) => check.key === key);
  if (found === undefined) throw new Error(`no ${key} row`);
  return { tone: found.tone, text: found.text };
}

const NONE = { live: undefined, totals: undefined, cache: undefined, models: undefined };

function models(count: number, warnings: string[] = []): ModelsResponse {
  return {
    models: Array.from({ length: count }, (_, i) => ({
      name: `m${String(i)}`,
      path: `/models/m${String(i)}`,
      modelType: 'qwen3',
      quant: null,
      contextWindow: null,
      sizeBytes: 1,
      fileCount: 1,
      servedIds: [],
    })),
    warnings,
    dir: '/models',
    host: { totalMemBytes: 64 * 1024 ** 3, diskFreeBytes: null },
  };
}

describe('healthChecks', () => {
  it('always gives five rows in a fixed order with their links', () => {
    const checks = healthChecks(NONE);
    expect(checks.map((check) => check.key)).toEqual(['engine', 'memory', 'turns', 'cache', 'models']);
    expect(checks.map((check) => check.to)).toEqual([
      '/engine',
      '/engine',
      '/metrics?tab=failures&range=24h',
      '/cache',
      '/models',
    ]);
    expect(checks.map((check) => check.text)).toEqual(['Unknown', '—', '—', '—', '—']);
  });

  it('reads the engine status first match wins', () => {
    const cases: Array<[Parameters<typeof live>[0], HealthCheck['tone'], string]> = [
      [{ status: 'untrusted' }, 'danger', 'Output not trustworthy'],
      [{ status: 'failed' }, 'danger', 'Failed'],
      [{ status: 'unresponsive' }, 'warning', 'Not responding'],
      [{ status: 'restarting' }, 'warning', 'Restarting'],
      [{ status: 'starting' }, 'warning', 'Starting'],
      [{ status: 'loading' }, 'neutral', 'Loading a model'],
      [{ status: 'running' }, 'success', 'Running'],
      [{ status: 'stopped' }, 'neutral', 'Stopped'],
    ];
    for (const [overrides, tone, text] of cases) {
      expect(row(healthChecks({ ...NONE, live: live(overrides) }), 'engine')).toEqual({ tone, text });
    }
  });

  it('grades memory against the wired limit of running processes', () => {
    const at = (active: number, wiredLimit: number) =>
      row(
        healthChecks({
          ...NONE,
          live: live({ processes: [process({ mem: { active, peak: active, cache: 0, wiredLimit, rss: 0 } })] }),
        }),
        'memory',
      );
    expect(at(10 * GIB, 100 * GIB)).toEqual({ tone: 'success', text: '10% of the wired limit' });
    expect(at(80 * GIB, 100 * GIB)).toEqual({ tone: 'warning', text: '80% of the wired limit' });
    expect(at(95 * GIB, 100 * GIB)).toEqual({ tone: 'danger', text: '95% of the wired limit' });
    expect(at(4 * GIB, 0)).toEqual({ tone: 'neutral', text: '4.0 GB in use, limit unknown' });
    expect(
      row(healthChecks({ ...NONE, live: live({ processes: [process({ status: 'unresponsive' })] }) }), 'memory'),
    ).toEqual({
      tone: 'neutral',
      text: '—',
    });
  });

  it('grades failures over the last 24 hours', () => {
    const turns = (overrides: Parameters<typeof totals>[0]) =>
      row(healthChecks({ ...NONE, totals: totals(overrides) }), 'turns');
    expect(turns({ turns: 0 })).toEqual({ tone: 'neutral', text: 'None in 24 hours' });
    expect(turns({ turns: 10, ok: 10 })).toEqual({ tone: 'success', text: 'No failures in 24 hours' });
    expect(turns({ turns: 100, error: 3 })).toEqual({ tone: 'warning', text: '3 of 100 failed' });
    expect(turns({ turns: 100, error: 5 })).toEqual({ tone: 'danger', text: '5 of 100 failed' });
    expect(row(healthChecks({ ...NONE, totals: busyTotals() }), 'turns').tone).toBe('warning');
  });

  it('reads the cold cache faults in order and never warns on a full quota', () => {
    const base = cacheResponse();
    const cache = (health: Partial<typeof base.health>, extra: Partial<typeof base> = {}) =>
      row(healthChecks({ ...NONE, cache: { ...base, ...extra, health: { ...base.health, ...health } } }), 'cache');
    expect(cache({ writeErrorsTotal: 1, corruptions: 1 })).toEqual({ tone: 'danger', text: 'Refusing writes' });
    expect(cache({ corruptionsTotal: 2 })).toEqual({ tone: 'danger', text: 'Corruption seen' });
    expect(cache({ queueDropsTotal: 4 })).toEqual({ tone: 'warning', text: 'Dropped writes' });
    expect(cache({}, { disk: { ...base.disk, entryCount: 0, sidecarCount: 0 }, trend: [] })).toEqual({
      tone: 'neutral',
      text: 'Empty',
    });
    expect(cache({}, { disk: { ...base.disk, totalBytes: base.disk.quotaBytes } })).toEqual({
      tone: 'success',
      text: 'Healthy',
    });
  });

  it('counts installed models and unreadable folders', () => {
    expect(row(healthChecks({ ...NONE, models: models(0) }), 'models')).toEqual({
      tone: 'neutral',
      text: 'None installed',
    });
    expect(row(healthChecks({ ...NONE, models: models(2, ['bad']) }), 'models')).toEqual({
      tone: 'warning',
      text: '1 folder could not be read',
    });
    expect(row(healthChecks({ ...NONE, models: models(6) }), 'models')).toEqual({
      tone: 'success',
      text: '6 installed',
    });
  });
});
