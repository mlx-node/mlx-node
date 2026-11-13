/** The five rows of the Overview health list: one rule each, first match wins. React-free. */

import { formatBytes, formatNumber, formatPercent } from './format';
import type { CacheResponse, MetricsTotals, ModelsResponse, RuntimeLive } from './types';

export type CheckTone = 'success' | 'warning' | 'danger' | 'neutral';

export interface HealthCheck {
  key: 'engine' | 'memory' | 'turns' | 'cache' | 'models';
  label: string;
  tone: CheckTone;
  text: string;
  to: string;
}

const NONE = '—';

function engineCheck(live: RuntimeLive | undefined): HealthCheck {
  const row = (tone: CheckTone, text: string): HealthCheck => ({
    key: 'engine',
    label: 'Engine',
    tone,
    text,
    to: '/engine',
  });
  switch (live?.status) {
    case 'untrusted':
      return row('danger', 'Output not trustworthy');
    case 'failed':
      return row('danger', 'Failed');
    case 'unresponsive':
      return row('warning', 'Not responding');
    case 'restarting':
      return row('warning', 'Restarting');
    case 'starting':
      return row('warning', 'Starting');
    case 'loading':
      return row('neutral', 'Loading a model');
    case 'running':
      return row('success', 'Running');
    case 'stopped':
      return row('neutral', 'Stopped');
    default:
      return row('neutral', 'Unknown');
  }
}

function memoryCheck(live: RuntimeLive | undefined): HealthCheck {
  const row = (tone: CheckTone, text: string): HealthCheck => ({
    key: 'memory',
    label: 'Memory',
    tone,
    text,
    to: '/engine',
  });
  const running = live?.processes.filter((process) => process.status === 'running') ?? [];
  if (running.length === 0) return row('neutral', NONE);
  let active = 0;
  let limit = 0;
  for (const process of running) {
    active += process.mem.active;
    limit = Math.max(limit, process.mem.wiredLimit);
  }
  if (limit <= 0) return row('neutral', `${formatBytes(active)} in use, limit unknown`);
  const share = active / limit;
  const tone: CheckTone = share >= 0.95 ? 'danger' : share >= 0.8 ? 'warning' : 'success';
  return row(tone, `${formatPercent(share)} of the wired limit`);
}

function turnsCheck(totals: MetricsTotals | undefined): HealthCheck {
  const row = (tone: CheckTone, text: string): HealthCheck => ({
    key: 'turns',
    label: 'Turns',
    tone,
    text,
    to: '/metrics?tab=failures&range=24h',
  });
  if (totals === undefined) return row('neutral', NONE);
  if (totals.turns === 0) return row('neutral', 'None in 24 hours');
  if (totals.error === 0) return row('success', 'No failures in 24 hours');
  const tone: CheckTone = totals.error / totals.turns >= 0.05 ? 'danger' : 'warning';
  return row(tone, `${formatNumber(totals.error)} of ${formatNumber(totals.turns)} failed`);
}

function cacheCheck(cache: CacheResponse | undefined): HealthCheck {
  const row = (tone: CheckTone, text: string): HealthCheck => ({
    key: 'cache',
    label: 'Cold cache',
    tone,
    text,
    to: '/cache',
  });
  if (cache === undefined) return row('neutral', NONE);
  const { health, disk, trend } = cache;
  if (health.writeErrors > 0 || health.writeErrorsTotal > 0) return row('danger', 'Refusing writes');
  if (health.corruptions > 0 || health.corruptionsTotal > 0) return row('danger', 'Corruption seen');
  if (health.queueDropsTotal > 0) return row('warning', 'Dropped writes');
  const lookups = trend.reduce((total, day) => total + day.hits + day.misses, 0);
  if (disk.entryCount + disk.sidecarCount === 0 && lookups === 0) return row('neutral', 'Empty');
  return row('success', 'Healthy');
}

function modelsCheck(models: ModelsResponse | undefined): HealthCheck {
  const row = (tone: CheckTone, text: string): HealthCheck => ({
    key: 'models',
    label: 'Models',
    tone,
    text,
    to: '/models',
  });
  if (models === undefined) return row('neutral', NONE);
  if (models.models.length === 0) return row('neutral', 'None installed');
  const warnings = models.warnings.length;
  if (warnings > 0)
    return row('warning', `${formatNumber(warnings)} ${warnings === 1 ? 'folder' : 'folders'} could not be read`);
  return row('success', `${formatNumber(models.models.length)} installed`);
}

/** Always 5 rows, in this order: engine, memory, turns, cold cache, models. A full cold-cache quota is not a warning. */
export function healthChecks(input: {
  live: RuntimeLive | undefined;
  totals: MetricsTotals | undefined;
  cache: CacheResponse | undefined;
  models: ModelsResponse | undefined;
}): HealthCheck[] {
  return [
    engineCheck(input.live),
    memoryCheck(input.live),
    turnsCheck(input.totals),
    cacheCheck(input.cache),
    modelsCheck(input.models),
  ];
}
