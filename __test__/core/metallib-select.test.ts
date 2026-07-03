import { mkdirSync, mkdtempSync, rmSync, utimesSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, it, expect, beforeEach, afterEach } from 'vite-plus/test';

import {
  BASE_KERNEL_MARKERS,
  NAX_KERNEL_MARKERS,
  assertMetallibIntegrity,
  collectMetallibCandidates,
  compareVersions,
  hostAppleTriple,
  profileDirName,
  shouldExpectNaxKernels,
} from '../../packages/core/metallib-select';

const TRIPLE = 'aarch64-apple-darwin';

describe('compareVersions', () => {
  it('orders dotted versions numerically, not lexically', () => {
    expect(compareVersions('26.2', '26.10')).toBeLessThan(0);
    expect(compareVersions('26.2', '26.2')).toBe(0);
    expect(compareVersions('26.5.2', '26.2')).toBeGreaterThan(0);
    expect(compareVersions('26', '26.0')).toBe(0);
    expect(compareVersions('15.0', '26.2')).toBeLessThan(0);
  });
});

describe('shouldExpectNaxKernels', () => {
  it('mirrors the forced-NAX cmake gate: SDK >= 26.2 AND effective deployment target >= 26.0', () => {
    // deployment target defaults to the host version when the env is unset
    expect(shouldExpectNaxKernels('26.5', '26.5.2', undefined)).toBe(true);
    expect(shouldExpectNaxKernels('26.2', '26.2', undefined)).toBe(true);
    // MLX_METAL_FORCE_NAX drops upstream's >= 26.2 floor clause: any
    // macOS 26 target (MSL 4.0) builds the NAX kernels
    expect(shouldExpectNaxKernels('26.5', '26.1', undefined)).toBe(true);
    // SDK too old builds no NAX regardless of target
    expect(shouldExpectNaxKernels('15.5', '26.5', undefined)).toBe(false);
    // the published-artifact configuration: 26.0 floor still carries NAX
    expect(shouldExpectNaxKernels('26.5', '26.5', '26.0')).toBe(true);
    // a pre-26 floor drops MSL below 4.0, which fails the gate's
    // MLX_METAL_VERSION >= 400 clause
    expect(shouldExpectNaxKernels('26.5', '26.5', '15.0')).toBe(false);
    expect(shouldExpectNaxKernels('26.5', '26.1', '26.2')).toBe(true);
    // empty env behaves like unset
    expect(shouldExpectNaxKernels('26.5', '26.5', '')).toBe(true);
  });
});

describe('profileDirName / hostAppleTriple', () => {
  it('derives the cargo profile dir from napi build options', () => {
    expect(profileDirName({ release: true })).toBe('release');
    expect(profileDirName({})).toBe('debug');
    expect(profileDirName({ profile: 'bench', release: true })).toBe('bench');
  });
  it('maps node arch to an apple triple', () => {
    expect(hostAppleTriple('arm64')).toBe('aarch64-apple-darwin');
    expect(hostAppleTriple('x64')).toBe('x86_64-apple-darwin');
  });
});

describe('collectMetallibCandidates', () => {
  let root: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'metallib-select-'));
  });
  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  function addOutDir(rel: string, content: string, ageDays: number, withTimestamp = false): string {
    const scriptDir = join(root, rel);
    const libDir = join(scriptDir, 'out', 'lib');
    mkdirSync(libDir, { recursive: true });
    const metallib = join(libDir, 'mlx.metallib');
    writeFileSync(metallib, content);
    const when = new Date(Date.now() - ageDays * 86_400_000);
    utimesSync(metallib, when, when);
    if (withTimestamp) {
      const stamp = join(scriptDir, 'invoked.timestamp');
      writeFileSync(stamp, 'This file has an mtime of when this was started.');
      utimesSync(stamp, when, when);
    }
    return metallib;
  }

  it('picks the most recently built mlx-sys dir, not the readdir-first one', () => {
    // lexically-first dir is a week old (stale pin); lexically-last is fresh —
    // the old first-match scan shipped the stale one.
    addOutDir(`${TRIPLE}/release/build/mlx-sys-aaaa1111`, 'stale-pin', 7, true);
    const fresh = addOutDir(`${TRIPLE}/release/build/mlx-sys-ffff2222`, 'fresh-pin', 0, true);

    const candidates = collectMetallibCandidates(root, TRIPLE, 'release');
    expect(candidates).toHaveLength(2);
    expect(candidates[0]!.metallibPath).toBe(fresh);
  });

  it('ranks by cargo activity (invoked.timestamp) when the metallib itself was cache-reused', () => {
    // dir A: metallib built recently but not used since (no fresh timestamp).
    addOutDir(`${TRIPLE}/release/build/mlx-sys-aaaa1111`, 'other-toolchain', 2, true);
    // dir B: metallib file is older, but cargo just re-used this dir — its
    // invoked.timestamp is fresh.
    const reused = addOutDir(`${TRIPLE}/release/build/mlx-sys-bbbb2222`, 'current-build', 5, true);
    const stamp = join(root, `${TRIPLE}/release/build/mlx-sys-bbbb2222`, 'invoked.timestamp');
    const now = new Date();
    utimesSync(stamp, now, now);

    const candidates = collectMetallibCandidates(root, TRIPLE, 'release');
    expect(candidates[0]!.metallibPath).toBe(reused);
  });

  it('never mixes the plain-cargo layout into the triple tree, but falls back to it when the triple tree is empty', () => {
    const plain = addOutDir('release/build/mlx-sys-cccc3333', 'plain-layout', 0);
    expect(collectMetallibCandidates(root, TRIPLE, 'release')[0]!.metallibPath).toBe(plain);

    const triple = addOutDir(`${TRIPLE}/release/build/mlx-sys-dddd4444`, 'triple-layout', 3);
    const candidates = collectMetallibCandidates(root, TRIPLE, 'release');
    expect(candidates).toHaveLength(1);
    expect(candidates[0]!.metallibPath).toBe(triple);
  });

  it('scans the profile the build actually used and skips dirs without a metallib', () => {
    mkdirSync(join(root, `${TRIPLE}/debug/build/mlx-sys-eeee5555/out/lib`), { recursive: true });
    const debug = addOutDir(`${TRIPLE}/debug/build/mlx-sys-ffff6666`, 'debug-build', 0);

    expect(collectMetallibCandidates(root, TRIPLE, 'release')).toHaveLength(0);
    const candidates = collectMetallibCandidates(root, TRIPLE, 'debug');
    expect(candidates).toHaveLength(1);
    expect(candidates[0]!.metallibPath).toBe(debug);
  });
});

describe('assertMetallibIntegrity', () => {
  const healthy = Buffer.from(['MTLB', ...BASE_KERNEL_MARKERS, ...NAX_KERNEL_MARKERS].join('\0'));
  const stalePin = Buffer.from(['MTLB', ...BASE_KERNEL_MARKERS].join('\0'));

  it('rejects a truncated metallib via the minimum-size floor', () => {
    expect(() => assertMetallibIntegrity(healthy, { path: 'x', expectNax: false })).toThrow(/below the .*-byte floor/);
  });

  it('rejects a metallib without the base kernel inventory', () => {
    expect(() =>
      assertMetallibIntegrity(Buffer.from('MTLB junk'), { path: 'x', expectNax: false, minBytes: 1 }),
    ).toThrow(/missing expected kernel/);
  });

  it('rejects a previous-pin metallib when the host builds NAX kernels', () => {
    expect(() => assertMetallibIntegrity(stalePin, { path: 'x', expectNax: true, minBytes: 1 })).toThrow(
      /missing NAX kernel/,
    );
    // ...but accepts it when NAX is legitimately not built on this host.
    expect(() => assertMetallibIntegrity(stalePin, { path: 'x', expectNax: false, minBytes: 1 })).not.toThrow();
  });

  it('accepts a current-pin metallib with NAX kernels', () => {
    expect(() => assertMetallibIntegrity(healthy, { path: 'x', expectNax: true, minBytes: 1 })).not.toThrow();
  });
});
