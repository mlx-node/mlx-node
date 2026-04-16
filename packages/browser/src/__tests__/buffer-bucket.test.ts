import { describe, expect, it } from 'vitest';

import { roundUpBucket } from '../buffer-bucket';

describe('roundUpBucket', () => {
  it('returns 0 for non-positive input', () => {
    expect(roundUpBucket(0)).toBe(0);
    expect(roundUpBucket(-1)).toBe(0);
  });

  it('rounds small sizes up to the next 256B multiple', () => {
    expect(roundUpBucket(1)).toBe(256);
    expect(roundUpBucket(256)).toBe(256);
    expect(roundUpBucket(257)).toBe(512);
    expect(roundUpBucket(4096)).toBe(4096);
  });

  it('rounds large sizes up to the next power of two', () => {
    expect(roundUpBucket(4097)).toBe(8192);
    expect(roundUpBucket(8192)).toBe(8192);
    expect(roundUpBucket(8193)).toBe(16384);
  });

  it('handles sizes near and above 32-bit boundaries', () => {
    expect(roundUpBucket(2 ** 30)).toBe(2 ** 30);
    expect(roundUpBucket(2 ** 31)).toBe(2 ** 31);
    expect(roundUpBucket(2 ** 31 + 1)).toBe(2 ** 32);
    expect(roundUpBucket(2 ** 32)).toBe(2 ** 32);
    expect(roundUpBucket(5 * 2 ** 30)).toBe(2 ** 33); // > 4 GiB
  });

  it('preserves exact power-of-two inputs at the 2^53 boundary', () => {
    expect(roundUpBucket(2 ** 50)).toBe(2 ** 50);
    expect(roundUpBucket(2 ** 53)).toBe(2 ** 53);
  });

  it('always returns a value >= input and a power of two for input > 4096', () => {
    for (const v of [4097, 5000, 65537, 123456, 2 ** 20 + 7, 2 ** 31 + 17]) {
      const out = roundUpBucket(v);
      expect(out).toBeGreaterThanOrEqual(v);
      expect(Math.log2(out) % 1).toBe(0); // power of two
    }
  });
});
