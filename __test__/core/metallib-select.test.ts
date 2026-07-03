import { chmodSync, mkdirSync, mkdtempSync, rmSync, utimesSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, it, expect, beforeEach, afterEach } from 'vite-plus/test';

import {
  BASE_KERNEL_MARKERS,
  MIN_PAGED_METALLIB_BYTES,
  NAX_KERNEL_MARKERS,
  assertMetallibFloor,
  assertMetallibIntegrity,
  assertPagedMetallibIntegrity,
  collectMetallibCandidates,
  compareVersions,
  extractBakedMetallibBinding,
  hostAppleTriple,
  parseMetallibMinOs,
  profileDirName,
  resolveTargetRoot,
  selectMetallib,
  selectPagedMetallib,
  shouldExpectNaxKernels,
} from '../../packages/core/metallib-select';

const TRIPLE = 'aarch64-apple-darwin';

// chmod-based unreadability tests are meaningless as root (root bypasses
// permission bits); CI and dev both run unprivileged, so this only guards
// exotic environments.
const runningAsRoot = typeof process.getuid === 'function' && process.getuid() === 0;

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

  it.skipIf(runningAsRoot)(
    'THROWS when the newest candidate is untraversable (EACCES), instead of skipping it and picking an older one',
    () => {
      addOutDir(`${TRIPLE}/release/build/mlx-sys-aaaa1111`, 'older-readable', 7);
      addOutDir(`${TRIPLE}/release/build/mlx-sys-ffff2222`, 'newest-unreadable', 0);
      const newestLibDir = join(root, `${TRIPLE}/release/build/mlx-sys-ffff2222`, 'out', 'lib');
      chmodSync(newestLibDir, 0o000); // stat on its metallib now fails EACCES, not ENOENT
      try {
        expect(() => collectMetallibCandidates(root, TRIPLE, 'release')).toThrow(/cannot stat/);
      } finally {
        chmodSync(newestLibDir, 0o755);
      }
    },
  );

  it.skipIf(runningAsRoot)('THROWS when a candidate build root exists but cannot be listed (EACCES)', () => {
    addOutDir(`${TRIPLE}/release/build/mlx-sys-aaaa1111`, 'unreachable', 0);
    const buildRoot = join(root, TRIPLE, 'release', 'build');
    chmodSync(buildRoot, 0o000);
    try {
      expect(() => collectMetallibCandidates(root, TRIPLE, 'release')).toThrow(/cannot list/);
    } finally {
      chmodSync(buildRoot, 0o755);
    }
  });
});

describe('resolveTargetRoot', () => {
  it('mirrors cargo precedence: --target-dir flag > CARGO_TARGET_DIR > CARGO_BUILD_TARGET_DIR > default', () => {
    const env = { CARGO_TARGET_DIR: '/env/ct', CARGO_BUILD_TARGET_DIR: '/env/cbt' };
    expect(resolveTargetRoot({ targetDir: '/flag', env, defaultRoot: '/repo/target' })).toBe('/flag');
    expect(resolveTargetRoot({ targetDir: undefined, env, defaultRoot: '/repo/target' })).toBe('/env/ct');
    expect(
      resolveTargetRoot({
        targetDir: undefined,
        env: { CARGO_BUILD_TARGET_DIR: '/env/cbt' },
        defaultRoot: '/repo/target',
      }),
    ).toBe('/env/cbt');
    expect(resolveTargetRoot({ targetDir: undefined, env: {}, defaultRoot: '/repo/target' })).toBe('/repo/target');
  });

  it('treats empty env values as unset, like cargo', () => {
    expect(
      resolveTargetRoot({
        targetDir: undefined,
        env: { CARGO_TARGET_DIR: '', CARGO_BUILD_TARGET_DIR: '' },
        defaultRoot: '/repo/target',
      }),
    ).toBe('/repo/target');
  });
});

/** A synthetic addon binary: NUL-separated strings, like a real Mach-O string table. */
function fakeAddon(strings: string[]): Buffer {
  return Buffer.concat([Buffer.from([0x42, 0x00]), Buffer.from(strings.join('\0')), Buffer.from([0x00, 0x42])]);
}

const BAKED_DIR_A = `/repo/target/${TRIPLE}/release/build/mlx-sys-aaaa1111/out`;
const bakedPathFor = (outDir: string, doubleSlash = true) =>
  `${outDir}/build/mlx/backend/metal/kernels/${doubleSlash ? '/' : ''}mlx.metallib`;

describe('extractBakedMetallibBinding', () => {
  it('finds the baked METAL_PATH among decoy strings and derives the out/lib layout', () => {
    const addon = fakeAddon([
      'No Metal device found',
      'mlx_paged_attn.metallibMTLB',
      'Failed to load metallib: ',
      bakedPathFor(BAKED_DIR_A),
      '/some/unrelated/mlx.metallib',
      '.metallib',
    ]);
    const binding = extractBakedMetallibBinding(addon);
    expect(binding).toBeDefined();
    expect(binding!.outDir).toBe(BAKED_DIR_A);
    expect(binding!.libDir).toBe(`${BAKED_DIR_A}/lib`);
    expect(binding!.installCopyPath).toBe(`${BAKED_DIR_A}/lib/mlx.metallib`);
    // the double slash in the baked string is normalized away
    expect(binding!.bakedPath).toBe(`${BAKED_DIR_A}/build/mlx/backend/metal/kernels/mlx.metallib`);
  });

  it('accepts the single-slash kernels path form too', () => {
    const binding = extractBakedMetallibBinding(fakeAddon([bakedPathFor(BAKED_DIR_A, false)]));
    expect(binding?.outDir).toBe(BAKED_DIR_A);
  });

  it('returns undefined when no METAL_PATH is baked (e.g. Metal-less build)', () => {
    expect(extractBakedMetallibBinding(fakeAddon(['Failed to load metallib: ', '.metallib>']))).toBeUndefined();
  });

  it('dedupes repeated identical paths but rejects two distinct baked paths', () => {
    const twice = fakeAddon([bakedPathFor(BAKED_DIR_A), 'x', bakedPathFor(BAKED_DIR_A)]);
    expect(extractBakedMetallibBinding(twice)?.outDir).toBe(BAKED_DIR_A);

    const conflicting = fakeAddon([
      bakedPathFor(BAKED_DIR_A),
      bakedPathFor(`/repo/target/${TRIPLE}/release/build/mlx-sys-bbbb2222/out`),
    ]);
    expect(() => extractBakedMetallibBinding(conflicting)).toThrow(/distinct METAL_PATH/);
  });
});

describe('selectMetallib', () => {
  let root: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'metallib-select-'));
  });
  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  function addOutDir(rel: string, content: string, ageDays: number): string {
    const libDir = join(root, rel, 'out', 'lib');
    mkdirSync(libDir, { recursive: true });
    const metallib = join(libDir, 'mlx.metallib');
    writeFileSync(metallib, content);
    const when = new Date(Date.now() - ageDays * 86_400_000);
    utimesSync(metallib, when, when);
    return metallib;
  }

  /** The cmake kernels build-tree file a baked METAL_PATH points at. */
  function addBakedFile(rel: string, content: string): string {
    const kernelsDir = join(root, rel, 'out', 'build', 'mlx', 'backend', 'metal', 'kernels');
    mkdirSync(kernelsDir, { recursive: true });
    const baked = join(kernelsDir, 'mlx.metallib');
    writeFileSync(baked, content);
    return baked;
  }

  it('rejects a same-pin NEWER but unbound candidate: the baked METAL_PATH wins over mtime rank', () => {
    const boundRel = `${TRIPLE}/release/build/mlx-sys-aaaa1111`;
    const bound = addOutDir(boundRel, 'bound-build', 7);
    const newerUnbound = addOutDir(`${TRIPLE}/release/build/mlx-sys-ffff2222`, 'newer-unbound', 0);

    // The heuristic alone would ship the newer dir...
    expect(collectMetallibCandidates(root, TRIPLE, 'release')[0]!.metallibPath).toBe(newerUnbound);

    // ...but the addon linked the OLDER dir, and the binding overrides.
    const addon = fakeAddon([bakedPathFor(join(root, boundRel, 'out'))]);
    const picked = selectMetallib({
      addonBinary: addon,
      addonPath: 'fake.node',
      targetRoot: root,
      triple: TRIPLE,
      profile: 'release',
    });
    expect(picked.source).toBe('baked');
    expect(picked.metallibPath).toBe(bound);
  });

  it('ships the exact baked build-tree file when it matches its out/lib install copy', () => {
    const rel = `${TRIPLE}/release/build/mlx-sys-aaaa1111`;
    addOutDir(rel, 'same-bytes', 0);
    const baked = addBakedFile(rel, 'same-bytes');
    const picked = selectMetallib({
      addonBinary: fakeAddon([bakedPathFor(join(root, rel, 'out'))]),
      addonPath: 'fake.node',
      targetRoot: root,
      triple: TRIPLE,
      profile: 'release',
    });
    expect(picked.source).toBe('baked');
    expect(picked.metallibPath).toBe(baked);
  });

  it('REJECTS a diverged out/lib install copy: baked file and install copy must be byte-identical', () => {
    const rel = `${TRIPLE}/release/build/mlx-sys-aaaa1111`;
    addOutDir(rel, 'stale-install-copy', 0);
    addBakedFile(rel, 'fresh-origin-bytes');
    expect(() =>
      selectMetallib({
        addonBinary: fakeAddon([bakedPathFor(join(root, rel, 'out'))]),
        addonPath: 'fake.node',
        targetRoot: root,
        triple: TRIPLE,
        profile: 'release',
      }),
    ).toThrow(/not byte-identical/);
  });

  it('SUCCEEDS with the baked file when the out/lib install copy is missing entirely', () => {
    const rel = `${TRIPLE}/release/build/mlx-sys-aaaa1111`;
    const baked = addBakedFile(rel, 'origin-only');
    const warnings: string[] = [];
    const picked = selectMetallib({
      addonBinary: fakeAddon([bakedPathFor(join(root, rel, 'out'))]),
      addonPath: 'fake.node',
      targetRoot: root,
      triple: TRIPLE,
      profile: 'release',
      warn: (m) => warnings.push(m),
    });
    expect(picked.source).toBe('baked');
    expect(picked.metallibPath).toBe(baked);
    expect(warnings.some((w) => w.includes('is missing'))).toBe(true);
  });

  it('throws (never falls back to another dir) when the bound build has no metallib left on disk', () => {
    addOutDir(`${TRIPLE}/release/build/mlx-sys-ffff2222`, 'newer-unbound', 0);
    const addon = fakeAddon([bakedPathFor(join(root, `${TRIPLE}/release/build/mlx-sys-aaaa1111`, 'out'))]);
    expect(() =>
      selectMetallib({
        addonBinary: addon,
        addonPath: 'fake.node',
        targetRoot: root,
        triple: TRIPLE,
        profile: 'release',
      }),
    ).toThrow(/neither that file nor its install copy .* exists/);
  });

  it.skipIf(runningAsRoot)(
    'THROWS when the baked file is uninspectable (EACCES), instead of treating it as absent and shipping the install copy',
    () => {
      const rel = `${TRIPLE}/release/build/mlx-sys-aaaa1111`;
      addOutDir(rel, 'good-install-copy', 0);
      addBakedFile(rel, 'origin-bytes');
      const kernelsDir = join(root, rel, 'out', 'build', 'mlx', 'backend', 'metal', 'kernels');
      chmodSync(kernelsDir, 0o000); // stat on the baked file now fails with EACCES, not ENOENT
      try {
        expect(() =>
          selectMetallib({
            addonBinary: fakeAddon([bakedPathFor(join(root, rel, 'out'))]),
            addonPath: 'fake.node',
            targetRoot: root,
            triple: TRIPLE,
            profile: 'release',
          }),
        ).toThrow(/cannot stat/);
      } finally {
        chmodSync(kernelsDir, 0o755);
      }
    },
  );

  it('falls back to the mtime scan only when the addon bakes no METAL_PATH', () => {
    const newest = addOutDir(`${TRIPLE}/release/build/mlx-sys-ffff2222`, 'fresh', 0);
    addOutDir(`${TRIPLE}/release/build/mlx-sys-aaaa1111`, 'stale', 7);
    const warnings: string[] = [];
    const picked = selectMetallib({
      addonBinary: fakeAddon(['no baked path here']),
      addonPath: 'fake.node',
      targetRoot: root,
      triple: TRIPLE,
      profile: 'release',
      warn: (m) => warnings.push(m),
    });
    expect(picked.source).toBe('scan');
    expect(picked.metallibPath).toBe(newest);
    expect(warnings.some((w) => w.includes('No baked METAL_PATH'))).toBe(true);
  });

  it('keeps the loud not-found error when neither binding nor candidates exist', () => {
    expect(() =>
      selectMetallib({
        addonBinary: fakeAddon(['nothing']),
        addonPath: 'fake.node',
        targetRoot: root,
        triple: TRIPLE,
        profile: 'release',
      }),
    ).toThrow(/mlx\.metallib not found under/);
  });

  it('STRICT (publish): throws instead of scanning when the addon bakes no METAL_PATH', () => {
    // A fresh, newest metallib is on disk — lenient mode would ship it — but a
    // publish build must not pair an unbound scan result with the addon.
    addOutDir(`${TRIPLE}/release/build/mlx-sys-ffff2222`, 'fresh', 0);
    const warnings: string[] = [];
    expect(() =>
      selectMetallib({
        addonBinary: fakeAddon(['no baked path here']),
        addonPath: 'fake.node',
        targetRoot: root,
        triple: TRIPLE,
        profile: 'release',
        strict: true,
        warn: (m) => warnings.push(m),
      }),
    ).toThrow(/MLX_METALLIB_STRICT/);
    expect(warnings).toHaveLength(0); // fail closed — no scan warning, no scan
  });

  it('non-strict (local dev): still scans and warns when the addon bakes no METAL_PATH', () => {
    const newest = addOutDir(`${TRIPLE}/release/build/mlx-sys-ffff2222`, 'fresh', 0);
    const warnings: string[] = [];
    const picked = selectMetallib({
      addonBinary: fakeAddon(['no baked path here']),
      addonPath: 'fake.node',
      targetRoot: root,
      triple: TRIPLE,
      profile: 'release',
      strict: false,
      warn: (m) => warnings.push(m),
    });
    expect(picked.source).toBe('scan');
    expect(picked.metallibPath).toBe(newest);
    expect(warnings.some((w) => w.includes('No baked METAL_PATH'))).toBe(true);
  });
});

describe('selectPagedMetallib / assertPagedMetallibIntegrity', () => {
  let root: string;
  let outDir: string;
  let libDir: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'metallib-select-paged-'));
    outDir = join(root, 'out');
    libDir = join(outDir, 'lib');
    mkdirSync(libDir, { recursive: true });
  });
  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  function writeOrigin(content: string): string {
    const p = join(outDir, 'paged_attn.metallib');
    writeFileSync(p, content);
    return p;
  }
  function writeInstallCopy(content: string): string {
    const p = join(libDir, 'paged_attn.metallib');
    writeFileSync(p, content);
    return p;
  }

  it('ships the build.rs origin when both copies are byte-identical', () => {
    const origin = writeOrigin('MTLB same-bytes');
    writeInstallCopy('MTLB same-bytes');
    const picked = selectPagedMetallib({ outDir, libDir });
    expect(picked.path).toBe(origin);
    expect(picked.contents.toString()).toBe('MTLB same-bytes');
  });

  it('REJECTS a both-present mismatch instead of silently preferring either copy', () => {
    writeOrigin('MTLB origin-bytes');
    writeInstallCopy('MTLB different-install-copy');
    expect(() => selectPagedMetallib({ outDir, libDir })).toThrow(/not byte-identical/);
  });

  it('REJECTS loudly when the origin is truncated next to a good install copy (no silent fallback)', () => {
    const good = 'MTLB ' + 'k'.repeat(64);
    writeOrigin(good.slice(0, 7)); // interrupted write of the same build
    writeInstallCopy(good);
    expect(() => selectPagedMetallib({ outDir, libDir })).toThrow(/not byte-identical/);
  });

  it('ships the surviving copy (with a warning) when only one of the two exists', () => {
    const origin = writeOrigin('MTLB origin-only');
    let warnings: string[] = [];
    const fromOrigin = selectPagedMetallib({ outDir, libDir, warn: (m) => warnings.push(m) });
    expect(fromOrigin.path).toBe(origin);
    expect(warnings.some((w) => w.includes('is missing'))).toBe(true);

    rmSync(origin);
    const installCopy = writeInstallCopy('MTLB install-copy-only');
    warnings = [];
    const fromCopy = selectPagedMetallib({ outDir, libDir, warn: (m) => warnings.push(m) });
    expect(fromCopy.path).toBe(installCopy);
    expect(fromCopy.contents.toString()).toBe('MTLB install-copy-only');
    expect(warnings.some((w) => w.includes('is missing'))).toBe(true);
  });

  it('throws the loud not-found error when neither copy exists', () => {
    expect(() => selectPagedMetallib({ outDir, libDir })).toThrow(/paged_attn\.metallib not found at/);
  });

  it.skipIf(runningAsRoot)(
    'THROWS when the origin is present but unreadable (EACCES), instead of shipping the install copy uncompared',
    () => {
      const origin = writeOrigin('MTLB origin-bytes');
      writeInstallCopy('MTLB good-install-copy');
      chmodSync(origin, 0o000);
      try {
        const warnings: string[] = [];
        expect(() => selectPagedMetallib({ outDir, libDir, warn: (m) => warnings.push(m) })).toThrow(/cannot read/);
        expect(warnings).toHaveLength(0); // no warn-and-degrade
      } finally {
        chmodSync(origin, 0o644);
      }
    },
  );

  it('integrity gate: rejects truncation via the size floor and non-MTLB content via the magic', () => {
    // An identically-truncated PAIR passes selection (byte-equal) — the
    // integrity gate is what catches it.
    expect(() => assertPagedMetallibIntegrity(Buffer.from('MTLB tiny'), { path: 'x' })).toThrow(
      new RegExp(`below the ${MIN_PAGED_METALLIB_BYTES}-byte floor`),
    );
    expect(() => assertPagedMetallibIntegrity(Buffer.from('NOPE junk'), { path: 'x', minBytes: 1 })).toThrow(
      /MTLB container magic/,
    );
    expect(() => assertPagedMetallibIntegrity(Buffer.from('MTLB healthy'), { path: 'x', minBytes: 1 })).not.toThrow();
  });
});

/**
 * MTLB container header with the given min-OS stamp (u16 LE major @12,
 * minor @14) in the recognized layout: platform tag 0x8001 @4, container
 * version 2 @6, library type 0x00 @10, macOS tag 0x81 @11. `opts` corrupts
 * individual fields to model future/unknown container revisions.
 */
function mtlbHeader(
  major: number,
  minor: number,
  opts?: { magic?: string; platform?: number; containerVersion?: number; libraryType?: number; osTag?: number },
): Buffer {
  const header = Buffer.alloc(24);
  header.write(opts?.magic ?? 'MTLB', 0, 'latin1');
  header.writeUInt16LE(opts?.platform ?? 0x8001, 4);
  header.writeUInt16LE(opts?.containerVersion ?? 2, 6);
  header.writeUInt16LE(9, 8); // container minor version — varies by toolchain, not validated
  header[10] = opts?.libraryType ?? 0x00;
  header[11] = opts?.osTag ?? 0x81;
  header.writeUInt16LE(major, 12);
  header.writeUInt16LE(minor, 14);
  return header;
}

describe('parseMetallibMinOs / assertMetallibFloor', () => {
  it('parses the min-OS stamp out of the recognized MTLB header layout', () => {
    expect(parseMetallibMinOs(mtlbHeader(26, 0))).toBe('26.0');
    expect(parseMetallibMinOs(mtlbHeader(26, 2))).toBe('26.2');
    expect(parseMetallibMinOs(mtlbHeader(15, 0))).toBe('15.0');
    // The exact header bytes of the shipped 26.0-floor artifacts.
    const real = Buffer.from('4d544c420180020009000081' + '1a000000', 'hex');
    expect(parseMetallibMinOs(real)).toBe('26.0');
  });

  it('returns undefined on unknown layouts instead of guessing', () => {
    expect(parseMetallibMinOs(mtlbHeader(26, 0, { magic: 'NOPE' }))).toBeUndefined();
    expect(parseMetallibMinOs(Buffer.from('MTLB'))).toBeUndefined();
    expect(parseMetallibMinOs(mtlbHeader(0, 0))).toBeUndefined();
    expect(parseMetallibMinOs(mtlbHeader(4242, 0))).toBeUndefined();
  });

  it('a future container revision (version-like bytes at 12/14 but unrecognized layout) SKIPS with a warning, never hard-fails', () => {
    // Each fixture stamps a would-fail min-OS 26.2 under floor 26.0, but a
    // single unrecognized layout field must downgrade enforcement to a skip.
    const futureLayouts = [
      mtlbHeader(26, 2, { containerVersion: 3 }),
      mtlbHeader(26, 2, { platform: 0x8002 }),
      mtlbHeader(26, 2, { libraryType: 0x01 }),
      mtlbHeader(26, 2, { osTag: 0x82 }),
    ];
    for (const fixture of futureLayouts) {
      expect(parseMetallibMinOs(fixture)).toBeUndefined();
      const warnings: string[] = [];
      expect(() =>
        assertMetallibFloor(fixture, { path: 'x', deploymentFloor: '26.0', warn: (m) => warnings.push(m) }),
      ).not.toThrow();
      expect(warnings.some((w) => w.includes('unrecognized MTLB header layout'))).toBe(true);
    }
  });

  it('rejects a metallib stamped above the intended deployment floor', () => {
    expect(() => assertMetallibFloor(mtlbHeader(26, 2), { path: 'x', deploymentFloor: '26.0' })).toThrow(
      /stamps min-OS 26\.2, above the intended deployment floor 26\.0/,
    );
  });

  it('accepts stamps at or below the floor and skips unparseable headers', () => {
    expect(() => assertMetallibFloor(mtlbHeader(26, 0), { path: 'x', deploymentFloor: '26.0' })).not.toThrow();
    expect(() => assertMetallibFloor(mtlbHeader(15, 0), { path: 'x', deploymentFloor: '26.0' })).not.toThrow();
    expect(() => assertMetallibFloor(mtlbHeader(26, 0), { path: 'x', deploymentFloor: '26.5.2' })).not.toThrow();
    expect(() =>
      assertMetallibFloor(mtlbHeader(26, 2, { magic: 'NOPE' }), { path: 'x', deploymentFloor: '26.0' }),
    ).not.toThrow();
  });

  it('STRICT (publish): throws instead of warn-skip when the MTLB header layout is unrecognized', () => {
    // A publish build must not ship a metallib whose floor cannot be verified.
    const future = mtlbHeader(26, 2, { containerVersion: 3 }); // parses to undefined
    expect(parseMetallibMinOs(future)).toBeUndefined();
    const warnings: string[] = [];
    expect(() =>
      assertMetallibFloor(future, { path: 'x', deploymentFloor: '26.0', strict: true, warn: (m) => warnings.push(m) }),
    ).toThrow(/MLX_METALLIB_STRICT/);
    expect(warnings).toHaveLength(0); // fail closed — no warn-and-skip
  });

  it('non-strict (local dev): still warns and skips on an unrecognized MTLB header layout', () => {
    const future = mtlbHeader(26, 2, { platform: 0x8002 });
    const warnings: string[] = [];
    expect(() =>
      assertMetallibFloor(future, { path: 'x', deploymentFloor: '26.0', strict: false, warn: (m) => warnings.push(m) }),
    ).not.toThrow();
    expect(warnings.some((w) => w.includes('unrecognized MTLB header layout'))).toBe(true);
  });

  it('STRICT still accepts a recognized stamp at/below the floor and enforces above it', () => {
    // Strict mode only changes the unparseable branch; a recognized stamp
    // behaves exactly as before.
    expect(() =>
      assertMetallibFloor(mtlbHeader(26, 0), { path: 'x', deploymentFloor: '26.0', strict: true }),
    ).not.toThrow();
    expect(() => assertMetallibFloor(mtlbHeader(26, 2), { path: 'x', deploymentFloor: '26.0', strict: true })).toThrow(
      /above the intended deployment floor/,
    );
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
