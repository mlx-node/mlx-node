/** SVG path strings for the own charts. A null point is a gap: lines break there and are never bridged. */

export interface XY {
  x: number;
  y: number;
}

function n(value: number): string {
  return String(Math.round(value * 100) / 100);
}

/** The runs of consecutive non-null points. */
export function segments(points: readonly (XY | null)[]): XY[][] {
  const runs: XY[][] = [];
  let run: XY[] = [];
  for (const point of points) {
    if (point === null) {
      if (run.length > 0) runs.push(run);
      run = [];
    } else run.push(point);
  }
  if (run.length > 0) runs.push(run);
  return runs;
}

function monotoneTail(run: readonly XY[]): string {
  if (run.length < 3)
    return run
      .slice(1)
      .map((p) => `L${n(p.x)} ${n(p.y)}`)
      .join('');
  const count = run.length;
  const slopes: number[] = [];
  for (let i = 0; i < count - 1; i += 1) {
    const dx = run[i + 1].x - run[i].x;
    slopes.push(dx === 0 ? 0 : (run[i + 1].y - run[i].y) / dx);
  }
  const tangents: number[] = [slopes[0]];
  for (let i = 1; i < count - 1; i += 1) {
    const a = slopes[i - 1];
    const b = slopes[i];
    tangents.push(a * b <= 0 ? 0 : (2 * a * b) / (a + b));
  }
  tangents.push(slopes[count - 2]);
  let d = '';
  for (let i = 0; i < count - 1; i += 1) {
    const p = run[i];
    const q = run[i + 1];
    const third = (q.x - p.x) / 3;
    d += `C${n(p.x + third)} ${n(p.y + tangents[i] * third)} ${n(q.x - third)} ${n(q.y - tangents[i + 1] * third)} ${n(q.x)} ${n(q.y)}`;
  }
  return d;
}

function monotoneRun(run: readonly XY[]): string {
  return `M${n(run[0].x)} ${n(run[0].y)}${monotoneTail(run)}`;
}

/** Line through every run of two or more points; lone points come back separately so the caller draws them as dots. */
export function linePath(
  points: readonly (XY | null)[],
  curve: 'linear' | 'monotone' = 'linear',
): { d: string; lone: XY[] } {
  let d = '';
  const lone: XY[] = [];
  for (const run of segments(points)) {
    if (run.length === 1) {
      lone.push(run[0]);
      continue;
    }
    d += curve === 'monotone' ? monotoneRun(run) : `M${run.map((p) => `${n(p.x)} ${n(p.y)}`).join('L')}`;
  }
  return { d, lone };
}

/** The area between each run and the baseline. */
export function areaPath(
  points: readonly (XY | null)[],
  baseline: number,
  curve: 'linear' | 'monotone' = 'linear',
): string {
  let d = '';
  for (const run of segments(points)) {
    if (run.length < 2) continue;
    d += `M${n(run[0].x)} ${n(baseline)}L${n(run[0].x)} ${n(run[0].y)}`;
    if (curve === 'monotone') d += monotoneTail(run);
    else for (let i = 1; i < run.length; i += 1) d += `L${n(run[i].x)} ${n(run[i].y)}`;
    d += `L${n(run[run.length - 1].x)} ${n(baseline)}Z`;
  }
  return d;
}

/** The band between two lines; a position where either is null is a gap. */
export function bandPath(
  xs: readonly number[],
  low: readonly (number | null)[],
  high: readonly (number | null)[],
): string {
  const pairs: ({ x: number; lo: number; hi: number } | null)[] = xs.map((x, i) => {
    const lo = low[i];
    const hi = high[i];
    return lo === null || lo === undefined || hi === null || hi === undefined ? null : { x, lo, hi };
  });
  let d = '';
  let run: { x: number; lo: number; hi: number }[] = [];
  const flush = (): void => {
    if (run.length >= 2) {
      d += `M${run.map((p) => `${n(p.x)} ${n(p.hi)}`).join('L')}`;
      for (let i = run.length - 1; i >= 0; i -= 1) d += `L${n(run[i].x)} ${n(run[i].lo)}`;
      d += 'Z';
    }
    run = [];
  };
  for (const pair of pairs) {
    if (pair === null) flush();
    else run.push(pair);
  }
  flush();
  return d;
}

/**
 * A vertical bar from `top` down to `bottom`, with `radius` on its top corners only (the data end) and a square
 * base.
 */
export function barPath(x: number, top: number, width: number, bottom: number, radius: number): string {
  const height = Math.max(0, bottom - top);
  const r = Math.max(0, Math.min(radius, width / 2, height));
  if (r === 0) return `M${n(x)} ${n(bottom)}V${n(top)}H${n(x + width)}V${n(bottom)}Z`;
  return (
    `M${n(x)} ${n(bottom)}V${n(top + r)}` +
    `Q${n(x)} ${n(top)} ${n(x + r)} ${n(top)}` +
    `H${n(x + width - r)}` +
    `Q${n(x + width)} ${n(top)} ${n(x + width)} ${n(top + r)}` +
    `V${n(bottom)}Z`
  );
}

/** One interval of a stacked step chart: flat from `x0` to `x1`, one value per band, bottom band first. */
export interface StackStep {
  x0: number;
  x1: number;
  values: readonly number[];
}

/** Steps whose ends are closer than this many px meet: the band runs on without a break. */
const STEP_JOIN_PX = 0.5;

function stepValue(step: StackStep, band: number): number {
  const value = step.values[band];
  return value !== undefined && Number.isFinite(value) && value > 0 ? value : 0;
}

/**
 * Stacked step areas, one closed path per band from the bottom band up. Each step is flat over its own interval (no
 * smoothing), so an area is exactly the value times the interval. A step where the band is 0, or two steps that do
 * not meet, end a run: the path holds one closed subpath per run and only `M`, `H`, `V` and `Z`.
 */
export function stepStackPaths(points: readonly StackStep[], y: (value: number) => number): string[] {
  let bands = 0;
  for (const step of points) bands = Math.max(bands, step.values.length);
  const paths: string[] = [];
  const base = points.map(() => 0);
  for (let band = 0; band < bands; band += 1) {
    let d = '';
    let run: Array<{ step: StackStep; low: number; high: number }> = [];
    const flush = (): void => {
      if (run.length > 0) {
        const first = run[0];
        const last = run[run.length - 1];
        let at = n(y(first.high));
        const to = (value: number): string => {
          const next = n(y(value));
          const move = next === at ? '' : `V${next}`;
          at = next;
          return move;
        };
        d += `M${n(first.step.x0)} ${at}H${n(first.step.x1)}`;
        for (let i = 1; i < run.length; i += 1) d += `${to(run[i].high)}H${n(run[i].step.x1)}`;
        d += `${to(last.low)}H${n(last.step.x0)}`;
        for (let i = run.length - 2; i >= 0; i -= 1) d += `${to(run[i].low)}H${n(run[i].step.x0)}`;
        d += 'Z';
      }
      run = [];
    };
    points.forEach((step, index) => {
      const value = stepValue(step, band);
      const low = base[index];
      base[index] = low + value;
      const previous = run.length > 0 ? run[run.length - 1].step : null;
      if (value === 0 || step.x1 <= step.x0) {
        flush();
        return;
      }
      if (previous !== null && Math.abs(step.x0 - previous.x1) > STEP_JOIN_PX) flush();
      run.push({ step, low, high: low + value });
    });
    flush();
    paths.push(d);
  }
  return paths;
}

/** Pixel x of each of `count` resampled positions spread from `x0` to `x1`. */
export function sampleXs(count: number, x0: number, x1: number): number[] {
  const xs: number[] = [];
  for (let i = 0; i < count; i += 1) xs.push(count === 1 ? x0 : x0 + ((x1 - x0) * i) / (count - 1));
  return xs;
}

/** Points from parallel x and y arrays (null y = gap). */
export function zipPoints(xs: readonly number[], ys: readonly (number | null)[]): (XY | null)[] {
  return xs.map((x, i) => {
    const y = ys[i];
    return y === null || y === undefined ? null : { x, y };
  });
}

/** A React id made safe for `url(#…)`: letters, digits, `-` and `_` only. */
export function svgId(raw: string): string {
  let out = 'c';
  for (const ch of raw) {
    const ok =
      (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch === '-' || ch === '_';
    if (ok) out += ch;
  }
  return out;
}
