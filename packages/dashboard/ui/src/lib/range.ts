import { useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';

import type { Segment } from '../components/arc/segmented-control/segmented-control';
import type { MetricsBucket } from './types';

export type RangeKey = '1h' | '24h' | '7d' | '30d';

/** Typed as Arc's mutable `Segment[]` (arc/segmented-control) so they go to SegmentedControl / Select as they are. */
export const RANGE_OPTIONS: Segment[] = [
  { value: '1h', label: '1 hour' },
  { value: '24h', label: '24 hours' },
  { value: '7d', label: '7 days' },
  { value: '30d', label: '30 days' },
];

/** For a page narrower than 640 px. */
export const RANGE_OPTIONS_SHORT: Segment[] = [
  { value: '1h', label: '1h' },
  { value: '24h', label: '24h' },
  { value: '7d', label: '7d' },
  { value: '30d', label: '30d' },
];

/** Exact string compares. */
export function isRangeKey(value: string): value is RangeKey {
  return value === '1h' || value === '24h' || value === '7d' || value === '30d';
}

/** Reads and writes ?range= with replace navigation. Unknown value -> fallback. */
export function useRange(fallback: RangeKey = '7d'): [RangeKey, (next: RangeKey) => void] {
  const [params, setParams] = useSearchParams();
  const raw = params.get('range');
  const range = raw !== null && isRangeKey(raw) ? raw : fallback;
  const setRange = useCallback(
    (next: RangeKey) => {
      setParams(
        (current) => {
          const updated = new URLSearchParams(current);
          updated.set('range', next);
          return updated;
        },
        { replace: true },
      );
    },
    [setParams],
  );
  return [range, setRange];
}

export interface AlignedRange {
  from: number;
  /** End of the bucket that contains `now`; may be up to one bucket in the future. */
  to: number;
  bucket: MetricsBucket;
  /** Bucket starts, ascending. Length: 1h 12 · 24h 24 · 7d 7 · 30d 30. */
  starts: number[];
}

const FIVE_MINUTES_MS = 300_000;
const HOUR_MS = 3_600_000;

function evenStarts(from: number, step: number, count: number): number[] {
  const starts: number[] = [];
  for (let i = 0; i < count; i += 1) starts.push(from + i * step);
  return starts;
}

/** A window that ends at the end of the current bucket, so a request path is stable inside one bucket. */
export function alignedRange(key: RangeKey, now: number = Date.now()): AlignedRange {
  if (key === '1h') {
    const to = Math.floor(now / FIVE_MINUTES_MS) * FIVE_MINUTES_MS + FIVE_MINUTES_MS;
    const from = to - 12 * FIVE_MINUTES_MS;
    return { from, to, bucket: '5m', starts: evenStarts(from, FIVE_MINUTES_MS, 12) };
  }
  if (key === '24h') {
    const hour = new Date(now);
    hour.setMinutes(0, 0, 0);
    const to = hour.getTime() + HOUR_MS;
    const from = to - 24 * HOUR_MS;
    return { from, to, bucket: 'hour', starts: evenStarts(from, HOUR_MS, 24) };
  }
  const days = key === '7d' ? 7 : 30;
  const today = new Date(now);
  const y = today.getFullYear();
  const m = today.getMonth();
  const d = today.getDate();
  const starts: number[] = [];
  for (let i = 0; i < days; i += 1) starts.push(new Date(y, m, d + 1 - days + i).getTime());
  return { from: starts[0], to: new Date(y, m, d + 1).getTime(), bucket: 'day', starts };
}

export type EngineRangeKey = '1h' | '6h' | '24h' | '7d';

/** Arc's mutable `Segment[]`, so it goes to SegmentedControl as it is. */
export const ENGINE_RANGE_OPTIONS: Segment[] = [
  { value: '1h', label: '1 hour' },
  { value: '6h', label: '6 hours' },
  { value: '24h', label: '24 hours' },
  { value: '7d', label: '7 days' },
];

/** Exact string compares. */
export function isEngineRangeKey(value: string): value is EngineRangeKey {
  return value === '1h' || value === '6h' || value === '24h' || value === '7d';
}

const ENGINE_BUCKETS: Record<EngineRangeKey, { bucketMs: number; points: number }> = {
  '1h': { bucketMs: 60_000, points: 60 },
  '6h': { bucketMs: 300_000, points: 72 },
  '24h': { bucketMs: 900_000, points: 96 },
  '7d': { bucketMs: 3_600_000, points: 168 },
};

/** bucketMs: 1h 60000 · 6h 300000 · 24h 900000 · 7d 3600000  ->  60 / 72 / 96 / 168 points. Epoch aligned. */
export function engineRange(
  key: EngineRangeKey,
  now: number = Date.now(),
): { from: number; to: number; bucketMs: number; starts: number[] } {
  const { bucketMs, points } = ENGINE_BUCKETS[key];
  const to = Math.floor(now / bucketMs) * bucketMs + bucketMs;
  const from = to - points * bucketMs;
  return { from, to, bucketMs, starts: evenStarts(from, bucketMs, points) };
}
