/**
 * Geometry of the waiting orb: pure math, no canvas, no DOM. Derived from thinking-orbs 0.3.2 (MIT, Copyright (c) 2026
 * Jakub Antalik): engine/core.ts, engine/ribbon.ts and frameWave of engine/lattice.ts, with the package's 20 px
 * presets resolved to numbers. `LICENSE` and `PROVENANCE.md` in this folder.
 */

export interface Dot {
  x: number;
  y: number;
  z: number;
  r: number;
  /** Ink value in [0, 1]: 0 is the strongest ink. */
  white: number;
  a: number;
}

export type OrbShape = 'ring' | 'band' | 'wave';

interface Sphere {
  speed: number;
  /** Lanes of the band, or rings of the sphere. */
  rows: number;
  /** Segments of the band, or dots of the sphere's widest ring. */
  cols: number;
  rBase: number;
  rDepth: number;
}

interface Ribbon extends Sphere {
  ghost: number;
  wob: number;
  band: number;
}

/** `resolvePreset(state, 20)` of the package: breathing, composing, listening. */
const PRESETS: { ring: Ribbon; band: Ribbon; wave: Sphere } = {
  ring: { speed: 3.78, rows: 2, cols: 15, ghost: 0, rBase: 1.7842, rDepth: 2.7574, wob: 0.565, band: 3.968 },
  band: { speed: 3.12, rows: 2, cols: 20, ghost: 8, rBase: 1.1803, rDepth: 1.8241, wob: 1, band: 4.94 },
  wave: { speed: 3.998, rows: 5, cols: 13, rBase: 0.96, rDepth: 2.72 },
};

/** CSS px of the orb. */
export const ORB_SIZE = 20;

export const TAU = Math.PI * 2;
const C = ORB_SIZE / 2;
const RS = (ORB_SIZE / 300) ** 0.6;

type Project = (x: number, y: number, z: number) => [number, number, number];

function projector(yaw: number, tilt: number): Project {
  const st = Math.sin(tilt);
  const ct = Math.cos(tilt);
  const sy = Math.sin(yaw);
  const cy = Math.cos(yaw);
  return (x, y, z) => {
    const x1 = x * cy + z * sy;
    const z1 = -x * sy + z * cy;
    return [C + x1, C - (y * ct - z1 * st), y * st + z1 * ct];
  };
}

/** `frameRibbon` with `spin: 0`: the band's plane is fixed, so its basis is written out. `faceOn` is the ring. */
function ribbon(t: number, o: Ribbon, faceOn: boolean, dots: Dot[]): void {
  const R = C * 0.78;
  const camTilt = 0.3;
  const pt = projector(0, camTilt);
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < o.ghost; i += 1) {
    const y = 1 - (2 * (i + 0.5)) / o.ghost;
    const rad = Math.sqrt(1 - y * y);
    const [px, py, z] = pt(rad * Math.cos(i * golden) * R, y * R, rad * Math.sin(i * golden) * R);
    dots.push({ x: px, y: py, z, r: 0.8 * RS, white: 0.78, a: 0.1 + 0.22 * ((z / R + 1) / 2) });
  }
  const ta = faceOn ? -camTilt : 0.55;
  const vy = Math.cos(ta);
  const vz = Math.sin(ta);
  const baseR = faceOn ? R / (1 + 0.85 * 0.23 * o.wob) : R;
  const lanes = Math.max(1, Math.round(o.rows * o.band));
  const mid = (lanes - 1) / 2;
  for (let w = 0; w < lanes; w += 1) {
    const laneOff = (w - mid) * 0.075;
    const edge = Math.abs(w - mid) / Math.max(1, mid);
    for (let k = 0; k < o.cols; k += 1) {
      const a = (k / o.cols) * TAU;
      const wob = (0.16 * Math.sin(a * 3 - t * 1.7 + w * 0.22) + 0.07 * Math.sin(a * 5 + t * 1.1)) * o.wob;
      const off = faceOn ? laneOff : laneOff + wob;
      const x = Math.cos(a);
      const y = vy * Math.sin(a) - vz * off;
      const z = vz * Math.sin(a) + vy * off;
      const rr = (baseR * (faceOn ? 1 + wob : 1)) / Math.sqrt(x * x + y * y + z * z);
      const [px, py, zr] = pt(x * rr, y * rr, z * rr);
      const depth = (zr / R + 1) / 2;
      dots.push({
        x: px,
        y: py,
        z: zr,
        r: (o.rBase + o.rDepth * depth) * (1 - 0.25 * edge) * RS,
        white: 0.52 - 0.44 * depth + 0.18 * edge,
        a: 0.4 + 0.6 * depth,
      });
    }
  }
}

function wave(t: number, o: Sphere, dots: Dot[]): void {
  const R = C * 0.874;
  const pt = projector(t * 0.18, 0.38);
  for (let li = 0; li <= o.rows; li += 1) {
    const lat = -Math.PI / 2 + (li / o.rows) * Math.PI;
    const cosLat = Math.cos(lat);
    const sinLat = Math.sin(lat);
    const w = 0.62 * Math.sin(t * 2.1 - li * 0.52) + 0.38 * Math.sin(t * 1.27 + li * 0.83);
    const rr = R * (0.88 + 0.105 * w);
    const crest = Math.max(0, w);
    const count = Math.max(1, Math.round(Math.abs(cosLat) * o.cols));
    for (let lj = 0; lj < count; lj += 1) {
      const lon = (lj / count) * TAU;
      const [px, py, z] = pt(cosLat * Math.cos(lon) * rr, sinLat * rr, cosLat * Math.sin(lon) * rr);
      const depth = (z / R + 1) / 2;
      dots.push({
        x: px,
        y: py,
        z,
        r: (o.rBase + o.rDepth * depth) * (1 + 0.4 * crest) * RS,
        white: 0.66 - 0.56 * depth - 0.1 * crest,
        a: 1,
      });
    }
  }
}

/** The dots of `shape` at `seconds`, far to near. */
export function orbFrame(shape: OrbShape, seconds: number): Dot[] {
  const t = seconds * PRESETS[shape].speed;
  const dots: Dot[] = [];
  if (shape === 'wave') wave(t, PRESETS.wave, dots);
  else ribbon(t, PRESETS[shape], shape === 'ring', dots);
  for (const dot of dots) dot.r = Math.max(0.3, dot.r);
  return dots.sort((a, b) => a.z - b.z);
}
