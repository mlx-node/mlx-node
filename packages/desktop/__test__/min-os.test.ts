/**
 * The floor the .app advertises decides whether macOS 12-15 REFUSES to launch it
 * or launches it into an unkillable sidecar restart loop, and the difference is a
 * number parsed out of `otool -l`. Nothing about that failure is observable
 * without an old Mac, so it has to be right by construction.
 *
 * The fixtures are verbatim `otool -l` output from this repo's own artifacts
 * (2026-07-27): the addon reports `minos 11.0`, Electron's binary `minos 12.0`.
 */

import { describe, expect, it } from 'vite-plus/test';

import { compareOsVersions, maxOsVersion, parseMinOs } from '../scripts/min-os.js';

/** `otool -l packages/core/mlx-core.darwin-arm64.node`, trimmed to the block. */
const ADDON = `      uuid 59248ADE-01E7-3548-965D-733A93D936BD
Load command 9
      cmd LC_BUILD_VERSION
  cmdsize 32
 platform 1
    minos 11.0
      sdk 26.5
   ntools 1
     tool 3
  version 1267.0
Load command 10
      cmd LC_MAIN
`;

/** The same, from a release build: MACOSX_DEPLOYMENT_TARGET=26.0 reaches minos. */
const RELEASE_ADDON = ADDON.replace('minos 11.0', 'minos 26.0');

/** A fat binary prints one LC_BUILD_VERSION block per slice, and they can differ. */
const UNIVERSAL = `Architectures in the fat file: x are: x86_64 arm64
x (architecture x86_64):
Load command 9
      cmd LC_BUILD_VERSION
  cmdsize 32
 platform 1
    minos 12.0
      sdk 26.5
x (architecture arm64):
Load command 9
      cmd LC_BUILD_VERSION
  cmdsize 32
 platform 1
    minos 26.0
      sdk 26.5
`;

describe('parseMinOs', () => {
  it('reads the floor out of real otool output', () => {
    expect(parseMinOs(ADDON)).toBe('11.0');
    expect(parseMinOs(RELEASE_ADDON)).toBe('26.0');
  });

  it('takes the HIGHEST slice of a universal binary, not the first', () => {
    // The x86_64 block is printed first and says 12.0. Trusting it would ship an
    // app that claims to run on 12 while its arm64 slice needs 26.
    expect(parseMinOs(UNIVERSAL)).toBe('26.0');
  });

  it('is null when nothing carries LC_BUILD_VERSION', () => {
    // Must not silently become "0.0": that would read as "runs everywhere" and
    // package.ts would stamp a floor it never measured.
    expect(parseMinOs('Load command 0\n      cmd LC_SEGMENT_64\n')).toBeNull();
  });

  it('will not take the number off a following line', () => {
    // otool does not emit this, and that is the point: the pattern must not be
    // ABLE to span a line, or the field it reads depends on output nobody pinned.
    // Built with `\s` (which spans newlines) this returns 26.0.
    expect(parseMinOs('    minos\n26.0\n')).toBeNull();
  });
});

describe('compareOsVersions', () => {
  it('orders numerically, not lexically', () => {
    // The whole point: "26.0" < "9.0" as strings, and 26.0 is the release floor.
    expect(compareOsVersions('26.0', '9.0')).toBeGreaterThan(0);
    expect(compareOsVersions('9.0', '12.0')).toBeLessThan(0);
  });

  it('treats a missing component as zero', () => {
    expect(compareOsVersions('12', '12.0')).toBe(0);
    expect(compareOsVersions('12.0.1', '12.0')).toBeGreaterThan(0);
    expect(compareOsVersions('12.0', '12.0.1')).toBeLessThan(0);
  });
});

describe('maxOsVersion', () => {
  it('is the maximum, not the last', () => {
    // Every binary has to load, so the bundle runs only where the most demanding
    // one runs. Order of the probe list must not matter.
    expect(maxOsVersion(['26.0', '12.0', '11.0'])).toBe('26.0');
    expect(maxOsVersion(['11.0', '12.0', '26.0'])).toBe('26.0');
  });

  it('refuses an empty list instead of inventing a floor', () => {
    expect(() => maxOsVersion([])).toThrow(/LC_BUILD_VERSION/);
  });
});
