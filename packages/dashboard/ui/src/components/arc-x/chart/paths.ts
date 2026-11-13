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

function monotoneRun(run: readonly XY[]): string {
  if (run.length < 3) return `M${run.map((p) => `${n(p.x)} ${n(p.y)}`).join('L')}`;
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
  let d = `M${n(run[0].x)} ${n(run[0].y)}`;
  for (let i = 0; i < count - 1; i += 1) {
    const p = run[i];
    const q = run[i + 1];
    const third = (q.x - p.x) / 3;
    d += `C${n(p.x + third)} ${n(p.y + tangents[i] * third)} ${n(q.x - third)} ${n(q.y - tangents[i + 1] * third)} ${n(q.x)} ${n(q.y)}`;
  }
  return d;
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
export function areaPath(points: readonly (XY | null)[], baseline: number): string {
  let d = '';
  for (const run of segments(points)) {
    if (run.length < 2) continue;
    d += `M${n(run[0].x)} ${n(baseline)}`;
    for (const p of run) d += `L${n(p.x)} ${n(p.y)}`;
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
