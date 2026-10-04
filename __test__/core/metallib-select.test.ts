import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  utimesSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, it, expect, beforeEach, afterEach } from 'vite-plus/test';

import {
  BASE_KERNEL_MARKERS,
  BRIDGE_KERNEL_MARKERS,
  KQUANT_KERNEL_MARKERS,
  KQUANT_NAX_KERNEL_MARKERS,
  MIN_PAGED_METALLIB_BYTES,
  MLX_NAX_ONLY_MARKERS,
  NAX_KERNEL_MARKERS,
  assertMetallibFloor,
  assertMetallibComplete,
  assertMetallibIntegrity,
  assertPagedMetallibIntegrity,
  collectMetallibCandidates,
  compareVersions,
  extractBakedMetallibBinding,
  assertMlxMetallibCarriesNax,
  hostAppleTriple,
  parseMetallibDeclaredSize,
  parseMetallibMinOs,
  profileDirName,
  resolveTargetRoot,
  selectMetallib,
  selectPagedMetallib,
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

  const pagedLibrary = (markers: readonly string[]) => metallibWith(markers);
  const KQUANT = [...KQUANT_KERNEL_MARKERS, ...KQUANT_NAX_KERNEL_MARKERS];
  const PREBUILT = [...KQUANT, ...BRIDGE_KERNEL_MARKERS];

  it('integrity gate: rejects truncation via the size floor and non-MTLB content via the magic', () => {
    // An identically-truncated PAIR passes selection (byte-equal) — the
    // integrity gate is what catches it.
    expect(() => assertPagedMetallibIntegrity(Buffer.from('MTLB tiny'), { path: 'x' })).toThrow(
      new RegExp(`below the ${MIN_PAGED_METALLIB_BYTES}-byte floor`),
    );
    expect(() => assertPagedMetallibIntegrity(Buffer.from('NOPE junk'), { path: 'x', minBytes: 1 })).toThrow(
      /MTLB container magic/,
    );
    expect(() => assertPagedMetallibIntegrity(pagedLibrary(PREBUILT), { path: 'x', minBytes: 1 })).not.toThrow();
  });

  it('integrity gate: rejects a paged library without the prebuilt K-quant kernels', () => {
    // A pre-K-quant paged_attn.metallib: the paged kernels only.
    expect(() =>
      assertPagedMetallibIntegrity(pagedLibrary(['paged_attention_bfloat16_t']), { path: 'x', minBytes: 1 }),
    ).toThrow(new RegExp(`missing K-quant kernel\\(s\\) ${KQUANT.join(', ')}`));
    for (const dropped of KQUANT) {
      const library = pagedLibrary(PREBUILT.filter((name) => name !== dropped));
      expect(() => assertPagedMetallibIntegrity(library, { path: 'x', minBytes: 1 })).toThrow(dropped);
    }
  });

  it('integrity gate: rejects a paged library without the K-quant NAX kernels', () => {
    const withoutNax = pagedLibrary([...KQUANT_KERNEL_MARKERS, ...BRIDGE_KERNEL_MARKERS]);
    expect(() => assertPagedMetallibIntegrity(withoutNax, { path: 'x', minBytes: 1 })).toThrow(
      new RegExp(`missing K-quant kernel\\(s\\) ${KQUANT_NAX_KERNEL_MARKERS.join(', ')}`),
    );
  });

  it('integrity gate: rejects a paged library without the segmented SDPA / mixed-affine kernels', () => {
    // The K-quant-only library of the previous build.
    expect(() => assertPagedMetallibIntegrity(pagedLibrary(KQUANT), { path: 'x', minBytes: 1 })).toThrow(
      new RegExp(`missing segmented SDPA / mixed-affine kernel\\(s\\) ${BRIDGE_KERNEL_MARKERS.join(', ')}`),
    );
    for (const dropped of BRIDGE_KERNEL_MARKERS) {
      const library = pagedLibrary(PREBUILT.filter((name) => name !== dropped));
      expect(() => assertPagedMetallibIntegrity(library, { path: 'x', minBytes: 1 })).toThrow(dropped);
    }
  });
});

describe('assertMlxMetallibCarriesNax', () => {
  const lib = (names: readonly string[], magic = 'MTLB') => Buffer.from([magic, ...names].join('\0'));

  it('accepts a library with every NAX-only kernel family', () => {
    expect(() =>
      assertMlxMetallibCarriesNax(lib([...BASE_KERNEL_MARKERS, ...MLX_NAX_ONLY_MARKERS]), 'm'),
    ).not.toThrow();
  });

  it('rejects an MLX_METAL_NO_NAX library', () => {
    expect(() => assertMlxMetallibCarriesNax(lib(BASE_KERNEL_MARKERS), 'm')).toThrow(
      /m has no NAX kernels .*MLX_METAL_NO_NAX/,
    );
  });

  it('rejects a library with only some NAX kernels, no base kernels, or no MTLB magic', () => {
    for (const only of MLX_NAX_ONLY_MARKERS) {
      expect(() => assertMlxMetallibCarriesNax(lib([...BASE_KERNEL_MARKERS, only]), 'm')).toThrow(
        /m is missing NAX kernel\(s\)/,
      );
    }
    expect(() => assertMlxMetallibCarriesNax(lib(MLX_NAX_ONLY_MARKERS), 'm')).toThrow(/missing the base MLX kernel/);
    expect(() =>
      assertMlxMetallibCarriesNax(lib([...BASE_KERNEL_MARKERS, ...MLX_NAX_ONLY_MARKERS], 'NOPE'), 'm'),
    ).toThrow(/MTLB container magic/);
  });
});

/** A complete MTLB container naming `names`: recognized header, declared size = byte length. */
function metallibWith(names: readonly string[]): Buffer {
  const metallib = Buffer.concat([mtlbHeader(26, 0), Buffer.from(['', ...names, ''].join('\0'))]);
  metallib.writeBigUInt64LE(BigInt(metallib.byteLength), 16);
  return metallib;
}

/**
 * MTLB container header with the given min-OS stamp (u16 LE major @12,
 * minor @14) in the recognized layout: platform tag 0x8001 @4, container
 * version 2 @6, library type 0x00 @10, macOS tag 0x81 @11. `opts` corrupts
 * individual fields to model future/unknown container revisions.
 */
function mtlbHeader(
  major: number,
  minor: number,
  opts?: {
    magic?: string;
    platform?: number;
    containerVersion?: number;
    libraryType?: number;
    osTag?: number;
    update?: number;
  },
): Buffer {
  const header = Buffer.alloc(24);
  header.write(opts?.magic ?? 'MTLB', 0, 'latin1');
  header.writeUInt16LE(opts?.platform ?? 0x8001, 4);
  header.writeUInt16LE(opts?.containerVersion ?? 2, 6);
  header.writeUInt16LE(9, 8); // container minor version — varies by toolchain, not validated
  header[10] = opts?.libraryType ?? 0x00;
  header[11] = opts?.osTag ?? 0x81;
  header.writeUInt16LE(major, 12);
  // minor and update are two separate bytes, not a u16 minor.
  header[14] = minor;
  header[15] = opts?.update ?? 0;
  return header;
}

describe('parseMetallibMinOs / assertMetallibFloor', () => {
  it('parses the min-OS stamp out of the recognized MTLB header layout', () => {
    expect(parseMetallibMinOs(mtlbHeader(26, 0))).toBe('26.0');
    expect(parseMetallibMinOs(mtlbHeader(26, 2))).toBe('26.2');
    expect(parseMetallibMinOs(mtlbHeader(15, 0))).toBe('15.0');
    // The exact header bytes of the shipped 26.2-floor artifacts.
    const real = Buffer.from('4d544c420180020009000081' + '1a000200', 'hex');
    expect(parseMetallibMinOs(real)).toBe('26.2');
  });

  it('reads minor and update as separate bytes, not one u16 minor', () => {
    // A non-zero update byte is where the two readings diverge: a u16 read of
    // `0502` yields 517. Bytes below are a real local build on macOS 26.5.2,
    // confirmed with `xcrun air-vtool -show` (Major 26, Minor 5, Update 2).
    const real2652 = Buffer.from('4d544c420180020009000081' + '1a000502', 'hex');
    expect(parseMetallibMinOs(real2652)).toBe('26.5.2');
    expect(parseMetallibMinOs(mtlbHeader(26, 5, { update: 2 }))).toBe('26.5.2');
    // update === 0 keeps the two-component form the shipped artifacts use.
    expect(parseMetallibMinOs(mtlbHeader(26, 5))).toBe('26.5');
  });

  it('accepts a metallib built at the host floor when that floor has a patch component', () => {
    // Regression: building on macOS 26.5.2 stamps 26.5.2 and the deployment
    // floor defaults to the host version, so the gate must compare equal. The
    // u16 misread reported 26.517 and failed a correct build.
    const stamped = mtlbHeader(26, 5, { update: 2 });
    expect(() => assertMetallibFloor(stamped, { path: 'x', deploymentFloor: '26.5.2' })).not.toThrow();
    expect(() => assertMetallibFloor(stamped, { path: 'x', deploymentFloor: '26.5.1' })).toThrow(
      /stamps min-OS 26\.5\.2, above the intended deployment floor 26\.5\.1/,
    );
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
  const healthy = metallibWith([...BASE_KERNEL_MARKERS, ...NAX_KERNEL_MARKERS, ...MLX_NAX_ONLY_MARKERS]);

  it('rejects a truncated metallib via the minimum-size floor', () => {
    expect(() => assertMetallibIntegrity(healthy, { path: 'x' })).toThrow(/below the .*-byte floor/);
  });

  it('rejects a metallib without the base kernel inventory', () => {
    expect(() => assertMetallibIntegrity(metallibWith(['junk']), { path: 'x', minBytes: 1 })).toThrow(
      /missing expected kernel/,
    );
  });

  it('rejects the 053e43fe fork-pin inventory', () => {
    const forkPin = metallibWith(['steel_attention', 'sdpa_vector', 'sdpa_vector_segmented_verify_2pass_1', 'qmv_sg8']);
    expect(() => assertMetallibIntegrity(forkPin, { path: 'x', minBytes: 1 })).toThrow(
      /missing expected kernel\(s\) sdpa_blocked_scale_copy, seq_gated_delta/,
    );
  });

  it('rejects a previous-pin or NAX-less metallib', () => {
    expect(() => assertMetallibIntegrity(metallibWith(BASE_KERNEL_MARKERS), { path: 'x', minBytes: 1 })).toThrow(
      /missing NAX kernel/,
    );
    // The pin markers alone are not the NAX-only kernel families.
    const noNax = metallibWith([...BASE_KERNEL_MARKERS, ...NAX_KERNEL_MARKERS]);
    expect(() => assertMetallibIntegrity(noNax, { path: 'x', minBytes: 1 })).toThrow(/has no NAX kernels/);
  });

  it('accepts a current-pin metallib with NAX kernels', () => {
    expect(() => assertMetallibIntegrity(healthy, { path: 'x', minBytes: 1 })).not.toThrow();
  });
});

describe('assertMetallibComplete', () => {
  it('reads the declared size from the shipped header bytes', () => {
    // First 24 bytes of the 31,863,042-byte paged_attn.metallib.
    const real = Buffer.from('4d544c4201800200090000811a000200' + '0231e60100000000', 'hex');
    expect(parseMetallibDeclaredSize(real)).toBe(31_863_042);
    expect(parseMetallibDeclaredSize(mtlbHeader(26, 0, { platform: 0x9001 }))).toBeUndefined();
    expect(parseMetallibDeclaredSize(Buffer.from('MTLB'))).toBeUndefined();
  });

  it('rejects a file shorter or longer than its header declares, and an unknown layout', () => {
    const full = metallibWith(BASE_KERNEL_MARKERS);
    expect(() => assertMetallibComplete(full, 'x')).not.toThrow();
    expect(() => assertMetallibComplete(full.subarray(0, full.byteLength - 1), 'x')).toThrow(/header declares/);
    expect(() => assertMetallibComplete(Buffer.concat([full, Buffer.alloc(1)]), 'x')).toThrow(/header declares/);
    expect(() => assertMetallibComplete(Buffer.from(['MTLB', ...BASE_KERNEL_MARKERS].join('\0')), 'x')).toThrow(
      /unrecognized MTLB header layout/,
    );
  });

  // Metal rejects this prefix as a truncated module, yet it keeps the magic,
  // clears the 4 MiB floor and still holds every kernel name.
  const built = join(import.meta.dirname, '..', '..', 'packages', 'core', 'paged_attn.metallib');
  it.skipIf(!existsSync(built))('rejects the 4 MiB prefix of the built paged_attn.metallib', () => {
    const dir = mkdtempSync(join(tmpdir(), 'metallib-trunc-'));
    try {
      const full = readFileSync(built);
      const prefixPath = join(dir, 'paged_attn.metallib');
      writeFileSync(prefixPath, full.subarray(0, 4 * 1024 * 1024));
      const prefix = readFileSync(prefixPath);
      expect(BRIDGE_KERNEL_MARKERS.every((name) => prefix.includes(name))).toBe(true);
      expect(() => assertPagedMetallibIntegrity(prefix, { path: prefixPath })).toThrow(
        new RegExp(`is ${prefix.byteLength} bytes but its header declares ${full.byteLength}`),
      );
      expect(() => assertPagedMetallibIntegrity(full, { path: built })).not.toThrow();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
