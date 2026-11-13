import { describe, expect, it } from 'vite-plus/test';

import { ORB_SIZE, orbFrame, type OrbShape } from '../ui/src/components/arc-x/wait-orb/engine.js';

/**
 * Dot count, then the sums of x, y, r and a · (1 − clamp(white)) at 0.6 s, from the engine of the npm package
 * thinking-orbs 0.3.2 (`MODE_FRAMES[mode](20, 0.6 · speed, resolvePreset(state, 20).opts)`).
 */
const GOLDEN: Record<OrbShape, readonly [number, number, number, number, number]> = {
  ring: [120, 1200.0, 1200.0, 64.0715, 50.3412],
  band: [208, 2079.9686, 2080.0843, 76.3428, 88.198],
  wave: [42, 420.0, 412.1518, 19.8377, 26.3267],
};
const SHAPES = Object.keys(GOLDEN) as OrbShape[];
const INSTANTS = [0, 0.6, 1.234, 7.5, 123.456];

describe('wait-orb engine', () => {
  it('equals the package at 0.6 s, to 4 decimals', () => {
    for (const shape of SHAPES) {
      const dots = orbFrame(shape, 0.6);
      let x = 0;
      let y = 0;
      let r = 0;
      let ink = 0;
      for (const dot of dots) {
        x += dot.x;
        y += dot.y;
        r += dot.r;
        ink += dot.a * (1 - Math.min(1, Math.max(0, dot.white)));
      }
      const [count, sx, sy, sr, sink] = GOLDEN[shape];
      expect(dots.length, shape).toBe(count);
      expect(x, shape).toBeCloseTo(sx, 4);
      expect(y, shape).toBeCloseTo(sy, 4);
      expect(r, shape).toBeCloseTo(sr, 4);
      expect(ink, shape).toBeCloseTo(sink, 4);
    }
  });

  it('every dot is inside the box, visible, at least 0.3 px, its ink in [0, 1], and drawn far to near', () => {
    for (const shape of SHAPES) {
      for (const at of INSTANTS) {
        const dots = orbFrame(shape, at);
        expect(dots.length, `${shape} ${at}`).toBe(GOLDEN[shape][0]);
        for (const [n, dot] of dots.entries()) {
          expect(dot.x).toBeGreaterThanOrEqual(0);
          expect(dot.x).toBeLessThanOrEqual(ORB_SIZE);
          expect(dot.y).toBeGreaterThanOrEqual(0);
          expect(dot.y).toBeLessThanOrEqual(ORB_SIZE);
          expect(dot.r).toBeGreaterThanOrEqual(0.3);
          expect(dot.a).toBeGreaterThanOrEqual(0.02);
          expect(dot.white).toBeGreaterThanOrEqual(0);
          expect(dot.white).toBeLessThanOrEqual(1);
          if (n > 0) expect(dot.z).toBeGreaterThanOrEqual(dots[n - 1]!.z);
        }
      }
    }
  });

  it('is pure: the same arguments give equal dots, and the shapes move', () => {
    for (const shape of SHAPES) {
      expect(orbFrame(shape, 7.5)).toEqual(orbFrame(shape, 7.5));
      expect(orbFrame(shape, 7.5)).not.toEqual(orbFrame(shape, 7.6));
    }
  });
});
