import { MAX_MEMORY_ERROR, parseMaxMemory } from '@mlx-node/server/host/max-memory';
import { describe, expect, it } from 'vite-plus/test';

const GIB = 1024 ** 3;

// Splash `parse_max_memory`: K/M/G (also KB/KiB, …) are powers of 1024, a
// fraction needs a unit and digits on both sides of the dot, and the bytes
// round down.
describe('parseMaxMemory', () => {
  it('reads byte counts with and without a unit', () => {
    expect(parseMaxMemory('32G')).toBe(34_359_738_368);
    expect(parseMaxMemory('2.5G')).toBe(2_684_354_560);
    expect(parseMaxMemory('28gib')).toBe(30_064_771_072);
    expect(parseMaxMemory(' 28GB ')).toBe(28 * GIB);
    expect(parseMaxMemory('512m')).toBe(512 * 1024 ** 2);
    expect(parseMaxMemory('1.5KiB')).toBe(1536);
    expect(parseMaxMemory('1024')).toBe(1024);
  });

  // Mutation: compute the size in floating point (`Math.floor(Number(normalized)
  // * unit)`) → red: 8388607.9999999999999 rounds to 2^23, so 2^53 bytes
  // is refused where the exact product floors to 2^53 − 1.
  it('rounds a fractional size down to whole bytes', () => {
    expect(parseMaxMemory('0.1G')).toBe(107_374_182);
    expect(parseMaxMemory('1.1G')).toBe(1_181_116_006);
    expect(parseMaxMemory('8388607.9999999999999G')).toBe(Number.MAX_SAFE_INTEGER);
  });

  it("returns undefined for 'auto'", () => {
    expect(parseMaxMemory('auto')).toBeUndefined();
    expect(parseMaxMemory(' AUTO ')).toBeUndefined();
  });

  // Mutation: accept a fraction without a unit (drop `multiplier > 1n`) → red
  // ('2.5' parses as 2).
  it.each(['2.5', '0', '-1', 'G', '', '1.2.3G', '.5G', '2.G', '1024B', '1T', '0.0001', '0.0000000001K'])(
    'refuses %j with the Splash message',
    (value) => {
      expect(() => parseMaxMemory(value)).toThrow(MAX_MEMORY_ERROR);
    },
  );

  it('refuses a byte count a JS number cannot carry exactly', () => {
    expect(parseMaxMemory(String(Number.MAX_SAFE_INTEGER))).toBe(Number.MAX_SAFE_INTEGER);
    expect(() => parseMaxMemory(String(BigInt(Number.MAX_SAFE_INTEGER) + 1n))).toThrow(MAX_MEMORY_ERROR);
    expect(() => parseMaxMemory('8388608G')).toThrow(MAX_MEMORY_ERROR);
  });
});
