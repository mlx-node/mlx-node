import { describe, expect, it } from 'vite-plus/test';

import { deltaAt, mixSamples, rectDelta, resample } from '../ui/src/components/arc-x/chart/morph.js';

describe('resample', () => {
  it('keeps the ends and interpolates between two values', () => {
    expect(resample([0, 10], 3)).toEqual([0, 5, 10]);
  });

  it('keeps nulls null and never bridges a gap', () => {
    const out = resample([10, null, 30], 5);
    expect(out[0]).toBe(10);
    expect(out[1]).toBeNull();
    expect(out[2]).toBeNull();
    expect(out[3]).toBeNull();
    expect(out[4]).toBe(30);
  });

  it('an all-null series stays all null', () => {
    expect(resample([null, null, null], 4)).toEqual([null, null, null, null]);
  });
});

describe('mixSamples', () => {
  it('moves a value toward its target', () => {
    expect(mixSamples([0, 10], [10, 20], 0.5)).toEqual([5, 15]);
  });

  it('a new gap is a gap from the first frame', () => {
    expect(mixSamples([5, 5], [5, null], 0)).toEqual([5, null]);
  });

  it('a value after a gap appears where it belongs', () => {
    expect(mixSamples([null, 5], [8, 5], 0.1)).toEqual([8, 5]);
  });
});

describe('rectDelta', () => {
  it('equal rectangles give a zero delta', () => {
    const rect = { x: 4, y: 10, width: 20, height: 40 };
    expect(rectDelta(rect, { ...rect })).toEqual({ dx: 0, dy: 0, sx: 1, sy: 1 });
  });

  it('draws the new rect where the old one was, then reaches identity', () => {
    const delta = rectDelta({ x: 0, y: 50, width: 10, height: 50 }, { x: 0, y: 0, width: 10, height: 100 });
    expect(delta).toEqual({ dx: 0, dy: 50, sx: 1, sy: 0.5 });
    expect(delta === null ? null : deltaAt(delta, 1)).toEqual({ dx: 0, dy: 0, sx: 1, sy: 1 });
  });

  it('is null when either rectangle is missing', () => {
    expect(rectDelta(null, { x: 0, y: 0, width: 1, height: 1 })).toBeNull();
    expect(rectDelta({ x: 0, y: 0, width: 1, height: 1 }, null)).toBeNull();
  });
});
