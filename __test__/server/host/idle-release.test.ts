import {
  DEFAULT_IDLE_RELEASE_SECONDS,
  IDLE_RELEASE_ERROR,
  idleReleaseText,
  parseIdleRelease,
} from '@mlx-node/server/host/idle-release';
import { describe, expect, it } from 'vite-plus/test';

// Splash `parse_idle_release` / `_duration_seconds` and its
// `test_serve_options.py` table: seconds, or s, m or h after a Python float;
// `off` keeps the weights.
describe('parseIdleRelease', () => {
  it.each([
    ['600', 600],
    ['90s', 90],
    ['30M', 1800],
    ['1.5h', 5400],
    ['.5h', 1800],
    ['5.m', 300],
    ['1e1', 10],
    ['1_0m', 600],
    ['30 m', 1800],
    [' 2H ', 7200],
    ['+1E+1s', 10],
  ])('reads %j as %d seconds', (value, seconds) => {
    expect(parseIdleRelease(value)).toBe(seconds);
  });

  it("reads 'off' as Infinity", () => {
    expect(parseIdleRelease('off')).toBe(Infinity);
    expect(parseIdleRelease(' OFF ')).toBe(Infinity);
  });

  // Mutation: accept Python's `inf` spelling as Infinity → red ('inf' parses).
  it.each([
    '0',
    '-1m',
    'inf',
    'nan',
    'infinity',
    '10x',
    'm',
    'never',
    '',
    '1__0',
    '_1',
    '1_',
    '1e',
    '1e400',
    '- 1',
    '0x10',
    '1.2.3',
  ])('refuses %j with the Splash message', (value) => {
    expect(() => parseIdleRelease(value)).toThrow(IDLE_RELEASE_ERROR);
  });

  it('spells the value as Splash idle_release_text does', () => {
    expect(idleReleaseText(Infinity)).toBe('off');
    expect(idleReleaseText(1800)).toBe('1800');
    expect(idleReleaseText(90.5)).toBe('90.5');
    expect(DEFAULT_IDLE_RELEASE_SECONDS).toBe(600);
    expect(IDLE_RELEASE_ERROR).toBe('--idle-release must be off or a positive duration such as 30m, 2h or 600');
  });
});
