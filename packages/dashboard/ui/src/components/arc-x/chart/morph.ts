/** The `reshape` math of the chart kit: pure, node-testable. */

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * Samples a series at n evenly spaced x positions (0..1). A position that falls in a gap, or between a value and a
 * gap, is null. Linear between two neighbours that both have a value.
 */
export function resample(values: readonly (number | null)[], n: number): (number | null)[] {
  const out: (number | null)[] = [];
  if (n <= 0) return out;
  const length = values.length;
  for (let i = 0; i < n; i += 1) {
    if (length === 0) {
      out.push(null);
      continue;
    }
    if (length === 1) {
      out.push(values[0] ?? null);
      continue;
    }
    const position = n === 1 ? 0 : (i / (n - 1)) * (length - 1);
    const lo = Math.floor(position);
    const hi = Math.min(length - 1, Math.ceil(position));
    const a = values[lo] ?? null;
    const b = values[hi] ?? null;
    if (lo === hi) {
      out.push(a);
      continue;
    }
    if (a === null || b === null) {
      out.push(null);
      continue;
    }
    out.push(a + (b - a) * (position - lo));
  }
  return out;
}

/**
 * Per position: both numbers -> from + (to - from) * t; `to` null -> null (the gap is there from the first frame);
 * `from` null -> `to` (it appears where it belongs).
 */
export function mixSamples(
  from: readonly (number | null)[],
  to: readonly (number | null)[],
  t: number,
): (number | null)[] {
  const out: (number | null)[] = [];
  for (let i = 0; i < to.length; i += 1) {
    const next = to[i] ?? null;
    const previous = from[i] ?? null;
    if (next === null) out.push(null);
    else if (previous === null) out.push(next);
    else out.push(previous + (next - previous) * t);
  }
  return out;
}

/**
 * Old and new rectangle of one bar -> the transform that draws the new rect where the old one was:
 * { dx, dy, sx, sy } (translate, then scale about the new rect's top-left corner). null when either is missing.
 */
export function rectDelta(
  from: Rect | null,
  to: Rect | null,
): { dx: number; dy: number; sx: number; sy: number } | null {
  if (from === null || to === null) return null;
  const sx = to.width === 0 ? 1 : from.width / to.width;
  const sy = to.height === 0 ? 1 : from.height / to.height;
  return { dx: from.x - to.x, dy: from.y - to.y, sx, sy };
}

/** The delta at progress t (1 = identity). */
export function deltaAt(
  delta: { dx: number; dy: number; sx: number; sy: number },
  t: number,
): { dx: number; dy: number; sx: number; sy: number } {
  const rest = 1 - t;
  return { dx: delta.dx * rest, dy: delta.dy * rest, sx: 1 + (delta.sx - 1) * rest, sy: 1 + (delta.sy - 1) * rest };
}
