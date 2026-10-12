/**
 * `MLX_SSE_DRAIN_TIMEOUT_MS` strict-parse regression: the drain timeout used
 * to go through `Number()`, which silently coerces `0x10`, `1e5`, `Infinity`,
 * and whitespace-padded input into plausible-looking timeouts. Only a plain
 * positive decimal integer is accepted now; anything else falls back to the
 * default and warns once.
 */

import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import {
  DEFAULT_SSE_DRAIN_TIMEOUT_MS,
  resolveSSEDrainTimeoutMs,
} from '../../packages/server/src/streaming.js';

const ENV = 'MLX_SSE_DRAIN_TIMEOUT_MS';
const saved = process.env[ENV];

afterEach(() => {
  if (saved === undefined) delete process.env[ENV];
  else process.env[ENV] = saved;
  vi.restoreAllMocks();
});

describe('resolveSSEDrainTimeoutMs', () => {
  it('falls back to the default when unset or empty', () => {
    delete process.env[ENV];
    expect(resolveSSEDrainTimeoutMs()).toBe(DEFAULT_SSE_DRAIN_TIMEOUT_MS);
    process.env[ENV] = '';
    expect(resolveSSEDrainTimeoutMs()).toBe(DEFAULT_SSE_DRAIN_TIMEOUT_MS);
    process.env[ENV] = '   ';
    expect(resolveSSEDrainTimeoutMs()).toBe(DEFAULT_SSE_DRAIN_TIMEOUT_MS);
  });

  it('accepts a positive decimal integer (and trims outer whitespace)', () => {
    process.env[ENV] = '5000';
    expect(resolveSSEDrainTimeoutMs()).toBe(5000);
    process.env[ENV] = ' 1500 ';
    expect(resolveSSEDrainTimeoutMs()).toBe(1500);
  });

  it('rejects Number()-coercible non-decimal input', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    for (const bad of ['0x10', '1e5', 'Infinity', '1.5', '-1', '12ms', 'NaN', '0']) {
      process.env[ENV] = bad;
      expect(resolveSSEDrainTimeoutMs()).toBe(DEFAULT_SSE_DRAIN_TIMEOUT_MS);
    }
    // Log-once: the warning fires on the first invalid read only.
    expect(warn).toHaveBeenCalledTimes(1);
  });
});
