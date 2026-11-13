/** Every loader shape is four cubic segments in a 24 x 24 box, so one shape can turn into another stroke by stroke. */

export type LoaderStatus = 'pending' | 'loading' | 'success' | 'error';

export interface Segment {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  x3: number;
  y3: number;
}

export type LoaderShape = readonly [Segment, Segment, Segment, Segment];

const CENTER = 12;
const RADIUS = 8.5;

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}

/** A circular arc of the loader ring from `fromDeg` to `toDeg` (clockwise on screen), as one cubic. */
function arc(fromDeg: number, toDeg: number): Segment {
  const a0 = (fromDeg * Math.PI) / 180;
  const a1 = (toDeg * Math.PI) / 180;
  const k = (4 / 3) * Math.tan((a1 - a0) / 4) * RADIUS;
  const x0 = CENTER + RADIUS * Math.cos(a0);
  const y0 = CENTER + RADIUS * Math.sin(a0);
  const x3 = CENTER + RADIUS * Math.cos(a1);
  const y3 = CENTER + RADIUS * Math.sin(a1);
  return {
    x0: round(x0),
    y0: round(y0),
    x1: round(x0 - k * Math.sin(a0)),
    y1: round(y0 + k * Math.cos(a0)),
    x2: round(x3 + k * Math.sin(a1)),
    y2: round(y3 - k * Math.cos(a1)),
    x3: round(x3),
    y3: round(y3),
  };
}

/** A straight stroke as a cubic whose control points sit on the line. */
function line(x0: number, y0: number, x3: number, y3: number): Segment {
  return {
    x0,
    y0,
    x1: round(x0 + (x3 - x0) / 3),
    y1: round(y0 + (y3 - y0) / 3),
    x2: round(x0 + ((x3 - x0) * 2) / 3),
    y2: round(y0 + ((y3 - y0) * 2) / 3),
    x3,
    y3,
  };
}

function ringOf(halfSpanDeg: number): LoaderShape {
  const at = (center: number): Segment => arc(center - halfSpanDeg, center + halfSpanDeg);
  return [at(45), at(135), at(225), at(315)];
}

/** Segment i of every shape sits in the same quarter (45, 135, 225, 315 degrees), so strokes travel the least. */
export const LOADER_SHAPES: Record<LoaderStatus, LoaderShape> = {
  pending: ringOf(45),
  loading: ringOf(28),
  success: [line(10, 16, 13.75, 12), line(8.25, 14.25, 10, 16), line(6.5, 12.5, 8.25, 14.25), line(13.75, 12, 17.5, 8)],
  error: [line(12, 12, 16.5, 16.5), line(12, 12, 7.5, 16.5), line(7.5, 7.5, 12, 12), line(16.5, 7.5, 12, 12)],
};

const KEYS = ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'] as const;

export function mixSegment(from: Segment, to: Segment, t: number): Segment {
  const out = { ...from };
  for (const key of KEYS) out[key] = from[key] + (to[key] - from[key]) * t;
  return out;
}

export function mixShape(from: LoaderShape, to: LoaderShape, t: number): LoaderShape {
  return [
    mixSegment(from[0], to[0], t),
    mixSegment(from[1], to[1], t),
    mixSegment(from[2], to[2], t),
    mixSegment(from[3], to[3], t),
  ];
}

function turnSegment(segment: Segment, quarters: number): Segment {
  const theta = (-quarters * Math.PI) / 2;
  const cos = Math.round(Math.cos(theta));
  const sin = Math.round(Math.sin(theta));
  const point = (x: number, y: number): [number, number] => [
    round(CENTER + (x - CENTER) * cos - (y - CENTER) * sin),
    round(CENTER + (x - CENTER) * sin + (y - CENTER) * cos),
  ];
  const [x0, y0] = point(segment.x0, segment.y0);
  const [x1, y1] = point(segment.x1, segment.y1);
  const [x2, y2] = point(segment.x2, segment.y2);
  const [x3, y3] = point(segment.x3, segment.y3);
  return { x0, y0, x1, y1, x2, y2, x3, y3 };
}

/**
 * `shape` as drawn inside a group that stands turned by `quarters` x 90 degrees, so it reads upright on screen.
 * Stroke j takes the segment of the quarter it shows on screen, so no stroke crosses the box.
 */
export function turnShape(shape: LoaderShape, quarters: number): LoaderShape {
  const k = ((quarters % 4) + 4) % 4;
  if (k === 0) return shape;
  const pick = (j: number): Segment => turnSegment(shape[(j + k) % 4], k);
  return [pick(0), pick(1), pick(2), pick(3)];
}

function num(value: number): string {
  return String(round(value));
}

/** "M x0 y0 C x1 y1 x2 y2 x3 y3" */
export function segmentPath(segment: Segment): string {
  return `M ${num(segment.x0)} ${num(segment.y0)} C ${num(segment.x1)} ${num(segment.y1)} ${num(segment.x2)} ${num(segment.y2)} ${num(segment.x3)} ${num(segment.y3)}`;
}
