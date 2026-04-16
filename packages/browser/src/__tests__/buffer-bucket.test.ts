import { describe, expect, it } from 'vitest';

import { roundUpBucket } from '../buffer-bucket';

// Task D redesign (2026-04-16): empirical data showed bucketing did NOT
// improve pool hit rate (63.6% vs 62.4% baseline) but regressed decode by
// ~11% due to over-allocation on tiny-buffer misses. roundUpBucket is now
// an identity function (exact-size keying). These tests pin that behaviour
// so anyone re-introducing bucketing does so intentionally.
describe('roundUpBucket', () => {
  it('returns 0 for non-positive input (callers skip pooling on 0)', () => {
    expect(roundUpBucket(0)).toBe(0);
    expect(roundUpBucket(-1)).toBe(0);
    expect(roundUpBucket(-1024)).toBe(0);
  });

  it('returns the input unchanged for positive sizes (exact-size keying)', () => {
    // Boundary-ish cases preserved from the bucketed-era test so regressions
    // are easy to spot: each of these would bucket differently under the
    // old 256B/PO2 scheme.
    expect(roundUpBucket(1)).toBe(1);
    expect(roundUpBucket(8)).toBe(8);
    expect(roundUpBucket(256)).toBe(256);
    expect(roundUpBucket(257)).toBe(257);
    expect(roundUpBucket(4096)).toBe(4096);
    expect(roundUpBucket(4097)).toBe(4097);
    expect(roundUpBucket(8192)).toBe(8192);
    expect(roundUpBucket(8193)).toBe(8193);
    expect(roundUpBucket(49152)).toBe(49152);
    expect(roundUpBucket(1048576)).toBe(1048576);
  });

  it('handles sizes near and above 32-bit boundaries without truncation', () => {
    // 64-bit-safety: ensure no accidental bitwise coercion in the identity
    // path. If someone re-adds `size & ~255`, these break immediately.
    expect(roundUpBucket(2 ** 30)).toBe(2 ** 30);
    expect(roundUpBucket(2 ** 31)).toBe(2 ** 31);
    expect(roundUpBucket(2 ** 31 + 1)).toBe(2 ** 31 + 1);
    expect(roundUpBucket(2 ** 32)).toBe(2 ** 32);
    expect(roundUpBucket(5 * 2 ** 30)).toBe(5 * 2 ** 30); // > 4 GiB
  });

  it('preserves exact values at the 2^53 boundary', () => {
    expect(roundUpBucket(2 ** 50)).toBe(2 ** 50);
    expect(roundUpBucket(2 ** 53)).toBe(2 ** 53);
  });

  it('always returns a value >= input for positive sizes', () => {
    for (const v of [4097, 5000, 65537, 123456, 2 ** 20 + 7, 2 ** 31 + 17]) {
      const out = roundUpBucket(v);
      expect(out).toBeGreaterThanOrEqual(v);
      expect(out).toBe(v); // identity
    }
  });

  it('is stable under repeated application (idempotent)', () => {
    // Pool correctness relies on the bucketer being idempotent: applying
    // it on both create and release sides MUST yield the same key, even if
    // either side feeds the bucketed value back in.
    for (const v of [0, -1, 1, 256, 4096, 8192, 49152, 1048576, 2 ** 31 + 1]) {
      const once = roundUpBucket(v);
      const twice = roundUpBucket(once);
      expect(twice).toBe(once);
    }
  });
});
