/** Pure geometry for the own charts: nice ticks, linear scales and bar bands. */

const NICE_STEPS = [1, 2, 5, 10] as const;

function niceStep(rough: number): number {
  if (!(rough > 0) || !Number.isFinite(rough)) return 1;
  const exponent = Math.floor(Math.log10(rough));
  const base = 10 ** exponent;
  const fraction = rough / base;
  for (const step of NICE_STEPS) {
    if (fraction <= step + 1e-9) return step * base;
  }
  return 10 * base;
}

/** Strips float noise such as 0.30000000000000004 from a tick. */
function clean(value: number): number {
  return Number(value.toPrecision(12));
}

/**
 * 3–5 ticks on 1 / 2 / 5 steps that cover [min, max]. A flat or empty domain gets a unit span above it, so a chart of
 * zeros still has an axis.
 */
export function niceTicks(min: number, max: number, target = 4): { ticks: number[]; min: number; max: number } {
  let low = Number.isFinite(min) ? min : 0;
  let high = Number.isFinite(max) ? max : 0;
  if (high < low) [low, high] = [high, low];
  if (high === low) high = low === 0 ? 1 : low + Math.abs(low);
  const count = Math.min(5, Math.max(3, Math.round(target)));
  let step = niceStep((high - low) / (count - 1));
  let start = Math.floor(low / step) * step;
  let end = Math.ceil(high / step) * step;
  while ((end - start) / step + 1 > 5) {
    step = niceStep(step * 1.5);
    start = Math.floor(low / step) * step;
    end = Math.ceil(high / step) * step;
  }
  const ticks: number[] = [];
  for (let value = start; value <= end + step / 2; value += step) ticks.push(clean(value));
  while (ticks.length < 3) ticks.push(clean(ticks[ticks.length - 1] + step));
  return { ticks, min: ticks[0], max: ticks[ticks.length - 1] };
}

/** Maps `domain` onto `range` linearly. A zero-width domain maps everything to the start of the range. */
export function linearScale(
  domain: readonly [number, number],
  range: readonly [number, number],
): (value: number) => number {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0;
  if (span === 0) return () => r0;
  return (value) => r0 + ((value - d0) / span) * (r1 - r0);
}

export interface BandLayout {
  /** Width of one column's slot. */
  step: number;
  /** Drawn bar thickness: at most `maxBand`, leaving `gap` between slots. */
  band: number;
  /** x of the first bar's left edge, so the bars sit centered in their slots. */
  offset: number;
}

/** Splits `width` into `count` equal slots and centers a bar of at most `maxBand` px in each. */
export function bandLayout(count: number, width: number, maxBand = 24, gap = 2): BandLayout {
  if (count <= 0 || width <= 0) return { step: 0, band: 0, offset: 0 };
  const step = width / count;
  const band = Math.max(1, Math.min(maxBand, step - gap));
  return { step, band, offset: (step - band) / 2 };
}

/** Index of the slot under `x` (clamped), for a band layout of `count` slots. */
export function bandIndex(layout: BandLayout, count: number, x: number): number {
  if (count <= 0 || layout.step <= 0) return 0;
  return Math.min(count - 1, Math.max(0, Math.floor(x / layout.step)));
}

/** The largest finite value, or null when there is none. */
export function maxOf(values: Iterable<number | null | undefined>): number | null {
  let max: number | null = null;
  for (const value of values) {
    if (typeof value === 'number' && Number.isFinite(value) && (max === null || value > max)) max = value;
  }
  return max;
}

/** The smallest finite value, or null when there is none. */
export function minOf(values: Iterable<number | null | undefined>): number | null {
  let min: number | null = null;
  for (const value of values) {
    if (typeof value === 'number' && Number.isFinite(value) && (min === null || value < min)) min = value;
  }
  return min;
}

/**
 * Which x labels to keep so none overlap: every `stride`-th label, the stride doubling until labels of `labelPx`
 * fit in `width`. Returns the indices kept.
 */
export function sparseLabels(count: number, width: number, labelPx: number): number[] {
  if (count <= 0) return [];
  let stride = 1;
  while (stride < count && (width / count) * stride < labelPx) stride *= 2;
  const kept: number[] = [];
  for (let i = 0; i < count; i += stride) kept.push(i);
  return kept;
}
