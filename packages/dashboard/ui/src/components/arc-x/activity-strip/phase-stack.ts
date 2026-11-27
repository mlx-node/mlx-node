import { BANDS } from '../../../lib/metric-strings';
import type { StackStep } from '../chart/paths';

/** One band of a lanes-by-phase stack. */
export interface StackBand {
  key: string;
  label: string;
  /** CSS color expression. */
  color: string;
}

/** Mean lanes in each band over one interval of time (ms): lane-ms of the band ÷ (t1 − t0). */
export interface StackPoint {
  t0: number;
  t1: number;
  values: Readonly<Record<string, number>>;
}

/** The four engine bands, bottom to top. */
export const PHASE_BANDS: readonly StackBand[] = [
  { key: 'decode', label: BANDS.decode, color: 'var(--phase-decode)' },
  { key: 'prefill', label: BANDS.prefill, color: 'var(--phase-prefill)' },
  { key: 'other', label: BANDS.other, color: 'var(--phase-prepare)' },
  { key: 'waiting', label: BANDS.waiting, color: 'var(--phase-queue)' },
];

function lanes(point: StackPoint, key: string): number {
  const value = point.values[key];
  return value !== undefined && Number.isFinite(value) && value > 0 ? value : 0;
}

function overlapMs(point: StackPoint, from: number, to: number): number {
  return Math.max(0, Math.min(point.t1, to) - Math.max(point.t0, from));
}

/** Lane-ms per band inside [from, to], and the ms of that span the intervals cover. */
export function stackLaneMs(
  points: readonly StackPoint[],
  keys: readonly string[],
  from: number,
  to: number,
): { laneMs: Record<string, number>; coveredMs: number } {
  const laneMs: Record<string, number> = {};
  for (const key of keys) laneMs[key] = 0;
  let coveredMs = 0;
  for (const point of points) {
    const overlap = overlapMs(point, from, to);
    if (overlap <= 0) continue;
    coveredMs += overlap;
    for (const key of keys) laneMs[key] += lanes(point, key) * overlap;
  }
  return { laneMs, coveredMs };
}

/** The largest stacked total of an interval that reaches into [from, to]; 0 when none does. */
export function stackPeak(points: readonly StackPoint[], keys: readonly string[], from: number, to: number): number {
  let peak = 0;
  for (const point of points) {
    if (point.t1 <= from || point.t0 >= to) continue;
    let total = 0;
    for (const key of keys) total += lanes(point, key);
    peak = Math.max(peak, total);
  }
  return peak;
}

/** Whole lanes for the top of a stack axis: at least 1. */
export function stackAxisMax(peak: number): number {
  return Math.max(1, Math.ceil(peak - 1e-9));
}

/** Steps for `stepStackPaths`: the intervals that reach into [from, to], cut at its edges, x from `x`. */
export function stackSteps(
  points: readonly StackPoint[],
  keys: readonly string[],
  from: number,
  to: number,
  x: (t: number) => number,
): StackStep[] {
  const steps: StackStep[] = [];
  for (const point of points) {
    if (point.t1 <= from || point.t0 >= to || point.t1 <= point.t0) continue;
    steps.push({
      x0: x(Math.max(point.t0, from)),
      x1: x(Math.min(point.t1, to)),
      values: keys.map((key) => lanes(point, key)),
    });
  }
  return steps;
}

/** The interval that holds `t` (t0 < t ≤ t1), or null. */
export function stackPointAt(points: readonly StackPoint[], t: number): StackPoint | null {
  for (let i = points.length - 1; i >= 0; i -= 1) {
    const point = points[i];
    if (point.t0 < t && t <= point.t1) return point;
  }
  return null;
}

/** Mean lanes as text: one decimal. */
export function formatLanes(value: number): string {
  return value.toFixed(1);
}
