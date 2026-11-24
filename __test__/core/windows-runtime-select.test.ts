import { chmodSync, mkdirSync, mkdtempSync, rmSync, utimesSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import {
  WINDOWS_MLX_INCLUDE_DIRS,
  WINDOWS_RUNTIME_ROOT_FILE,
  extractWindowsBuildUnitHash,
  nativeAddonTriple,
  normalizeLineEndings,
  selectWindowsMlxRoot,
  validateWindowsMlxRoot,
} from '../../packages/core/windows-runtime-select';

const TRIPLE = 'aarch64-pc-windows-msvc';
const runningAsRoot = typeof process.getuid === 'function' && process.getuid() === 0;

function makeAddonWithMarker(unitHash?: string): Buffer {
  if (unitHash === undefined) {
    return Buffer.from('MZ\0some-pe-bytes-without-marker\0', 'latin1');
  }
  return Buffer.from(`MZ\0header\0MLX_NODE_WINDOWS_BUILD_UNIT=unit_${unitHash}\0trailer\0`, 'latin1');
}

function populateMlxRoot(
  dir: string,
  opts: { omitDll?: boolean; emptyDll?: boolean; omitInclude?: string } = {},
): void {
  mkdirSync(dir, { recursive: true });
  if (!opts.omitDll) {
    writeFileSync(join(dir, 'openblas.dll'), opts.emptyDll ? '' : 'openblas-dll-bytes');
  }
  for (const include of WINDOWS_MLX_INCLUDE_DIRS) {
    if (include === opts.omitInclude) continue;
    const includeDir = join(dir, 'include', include);
    mkdirSync(includeDir, { recursive: true });
    writeFileSync(join(includeDir, 'header.hpp'), `// ${include}\n`);
  }
}

describe('normalizeLineEndings', () => {
  it('normalizes CRLF to LF while leaving LF untouched', () => {
    expect(normalizeLineEndings('a\r\nb\r\nc\n')).toBe('a\nb\nc\n');
    expect(normalizeLineEndings('a\nb\nc\n')).toBe('a\nb\nc\n');
  });
});

describe('nativeAddonTriple', () => {
  it('extracts triples and validates platform prefixes for darwin, linux, and win32', () => {
    expect(nativeAddonTriple('mlx-core.darwin-arm64.node', 'darwin')).toBe('darwin-arm64');
    expect(nativeAddonTriple('mlx-core.linux-arm64-gnu.node', 'linux')).toBe('linux-arm64-gnu');
    expect(nativeAddonTriple('mlx-core.win32-arm64-msvc.node', 'win32')).toBe('win32-arm64-msvc');
    expect(nativeAddonTriple('mlx-core.win32-x64-msvc.node', 'win32')).toBe('win32-x64-msvc');

    expect(() => nativeAddonTriple('mlx-core.darwin-arm64.node', 'win32')).toThrow(/expected a Windows addon/);
    expect(() => nativeAddonTriple('mlx-core.win32-arm64-msvc.node', 'darwin')).toThrow(/expected a Darwin addon/);
    expect(() => nativeAddonTriple('mlx-core.win32-arm64-msvc.node', 'linux')).toThrow(/expected a Linux addon/);
    expect(() => nativeAddonTriple('not-an-addon.node', 'win32')).toThrow(/unrecognized native addon output/);
  });
});

describe('extractWindowsBuildUnitHash', () => {
  it('returns undefined when no marker is baked', () => {
    expect(extractWindowsBuildUnitHash(makeAddonWithMarker())).toBeUndefined();
  });

  it('extracts a single valid lower-hex build unit hash and deduplicates identical copies', () => {
    const hash = '0123456789abcdef';
    expect(extractWindowsBuildUnitHash(makeAddonWithMarker(hash))).toBe(hash);
    const afterNonZero = Buffer.concat([
      Buffer.from([0xff, 0xfe, 0x41]),
      Buffer.from(`\0MLX_NODE_WINDOWS_BUILD_UNIT=unit_${hash}\0`, 'ascii'),
    ]);
    expect(extractWindowsBuildUnitHash(afterNonZero)).toBe(hash);
    const duplicated = Buffer.concat([makeAddonWithMarker(hash), makeAddonWithMarker(hash)]);
    expect(extractWindowsBuildUnitHash(duplicated)).toBe(hash);
  });

  it('ignores non-NUL-terminated or malformed markers', () => {
    const nonNul = Buffer.from('MLX_NODE_WINDOWS_BUILD_UNIT=unit_abcdef01', 'ascii');
    expect(extractWindowsBuildUnitHash(nonNul)).toBeUndefined();
    const upperHex = Buffer.from('\0MLX_NODE_WINDOWS_BUILD_UNIT=unit_ABCDEF01\0', 'ascii');
    expect(extractWindowsBuildUnitHash(upperHex)).toBeUndefined();
    const prefixed = Buffer.from('\0noiseMLX_NODE_WINDOWS_BUILD_UNIT=unit_abcdef01\0', 'ascii');
    expect(extractWindowsBuildUnitHash(prefixed)).toBeUndefined();
  });

  it('throws when multiple distinct build unit hashes are baked', () => {
    const mixed = Buffer.concat([makeAddonWithMarker('aaaa1111'), makeAddonWithMarker('bbbb2222')]);
    expect(() => extractWindowsBuildUnitHash(mixed)).toThrow(/2 distinct MLX_NODE_WINDOWS_BUILD_UNIT/);
  });
});

describe('validateWindowsMlxRoot', () => {
  let root: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'win-mlx-validate-'));
  });
  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  it('accepts a complete root with non-empty openblas.dll and include/{cccl,cute,cutlass}', () => {
    populateMlxRoot(root);
    expect(() => validateWindowsMlxRoot(root)).not.toThrow();
  });

  it('rejects missing or empty openblas.dll', () => {
    populateMlxRoot(root, { omitDll: true });
    expect(() => validateWindowsMlxRoot(root)).toThrow(/missing openblas\.dll/);

    populateMlxRoot(root, { emptyDll: true });
    expect(() => validateWindowsMlxRoot(root)).toThrow(/non-empty file/);
  });

  it('rejects missing or non-directory include/{cccl,cute,cutlass}', () => {
    for (const include of WINDOWS_MLX_INCLUDE_DIRS) {
      const dir = join(root, `missing-${include}`);
      populateMlxRoot(dir, { omitInclude: include });
      expect(() => validateWindowsMlxRoot(dir)).toThrow(new RegExp(`missing include/${include}`));
    }

    const fileInsteadOfDir = join(root, 'file-include');
    populateMlxRoot(fileInsteadOfDir, { omitInclude: 'cute' });
    writeFileSync(join(fileInsteadOfDir, 'include', 'cute'), 'not-a-dir');
    expect(() => validateWindowsMlxRoot(fileInsteadOfDir)).toThrow(/include\/cute.*directory/);
  });
});

describe('selectWindowsMlxRoot', () => {
  let root: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'win-mlx-select-'));
  });
  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  it('binds authoritatively via pre-Cargo-1.100 mlx-sys-<hash>/out even when a newer unrelated dir exists', () => {
    const targetRoot = join(root, 'target');
    const boundMlxRoot = join(root, 'prebuilt-bound');
    const newerMlxRoot = join(root, 'prebuilt-newer');
    populateMlxRoot(boundMlxRoot);
    populateMlxRoot(newerMlxRoot);

    const boundOut = join(targetRoot, TRIPLE, 'release', 'build', 'mlx-sys-1111aaaa', 'out');
    const newerOut = join(targetRoot, TRIPLE, 'release', 'build', 'mlx-sys-9999ffff', 'out');
    mkdirSync(boundOut, { recursive: true });
    mkdirSync(newerOut, { recursive: true });

    const boundFile = join(boundOut, WINDOWS_RUNTIME_ROOT_FILE);
    writeFileSync(boundFile, `${boundMlxRoot}\n`);
    const oldTime = new Date(Date.now() - 86_400_000);
    utimesSync(boundFile, oldTime, oldTime);

    writeFileSync(join(newerOut, WINDOWS_RUNTIME_ROOT_FILE), `${newerMlxRoot}\n`);

    const result = selectWindowsMlxRoot({
      addonBinary: makeAddonWithMarker('1111aaaa'),
      addonPath: 'mlx-core.win32-arm64-msvc.node',
      targetRoot,
      triple: TRIPLE,
      profile: 'release',
    });
    expect(result).toEqual({
      mlxRoot: boundMlxRoot,
      source: 'build-unit',
      unitHash: '1111aaaa',
    });
  });

  it('binds authoritatively via Cargo 1.100 mlx-sys/<hash>/out layout and plain profile fallback root', () => {
    const targetRoot = join(root, 'target');
    const boundMlxRoot = join(root, 'prebuilt-1100');
    populateMlxRoot(boundMlxRoot);

    const outDir = join(targetRoot, 'release', 'build', 'mlx-sys', 'abcdef012345', 'out');
    mkdirSync(outDir, { recursive: true });
    writeFileSync(join(outDir, WINDOWS_RUNTIME_ROOT_FILE), `${boundMlxRoot}\r\n`);

    const result = selectWindowsMlxRoot({
      addonBinary: makeAddonWithMarker('abcdef012345'),
      addonPath: 'mlx-core.win32-arm64-msvc.node',
      targetRoot,
      triple: TRIPLE,
      profile: 'release',
    });
    expect(result).toEqual({
      mlxRoot: boundMlxRoot,
      source: 'build-unit',
      unitHash: 'abcdef012345',
    });
  });

  it('fails closed when a baked build unit hash has no mlx-runtime-root.txt (even if another unit or legacy mlx-cmake exists)', () => {
    const targetRoot = join(root, 'target');
    const boundOut = join(targetRoot, TRIPLE, 'release', 'build', 'mlx-sys-deadbeef', 'out');
    populateMlxRoot(join(boundOut, 'mlx-cmake'));

    const otherOut = join(targetRoot, TRIPLE, 'release', 'build', 'mlx-sys-cafebabe', 'out');
    const otherRoot = join(root, 'other-prebuilt');
    populateMlxRoot(otherRoot);
    mkdirSync(otherOut, { recursive: true });
    writeFileSync(join(otherOut, WINDOWS_RUNTIME_ROOT_FILE), `${otherRoot}\n`);

    expect(() =>
      selectWindowsMlxRoot({
        addonBinary: makeAddonWithMarker('deadbeef'),
        addonPath: 'mlx-core.win32-arm64-msvc.node',
        targetRoot,
        triple: TRIPLE,
        profile: 'release',
      }),
    ).toThrow(/bound to mlx-sys build unit deadbeef/);
  });

  it('fails closed when the bound mlx-runtime-root.txt points to an incomplete or empty MLX root', () => {
    const targetRoot = join(root, 'target');
    const incompleteRoot = join(root, 'incomplete');
    populateMlxRoot(incompleteRoot, { omitInclude: 'cute' });

    const outDir = join(targetRoot, TRIPLE, 'release', 'build', 'mlx-sys-aabbccdd', 'out');
    mkdirSync(outDir, { recursive: true });
    writeFileSync(join(outDir, WINDOWS_RUNTIME_ROOT_FILE), `${incompleteRoot}\n`);

    expect(() =>
      selectWindowsMlxRoot({
        addonBinary: makeAddonWithMarker('aabbccdd'),
        addonPath: 'mlx-core.win32-arm64-msvc.node',
        targetRoot,
        triple: TRIPLE,
        profile: 'release',
      }),
    ).toThrow(/missing include\/cute/);

    writeFileSync(join(outDir, WINDOWS_RUNTIME_ROOT_FILE), '   \n');
    expect(() =>
      selectWindowsMlxRoot({
        addonBinary: makeAddonWithMarker('aabbccdd'),
        addonPath: 'mlx-core.win32-arm64-msvc.node',
        targetRoot,
        triple: TRIPLE,
        profile: 'release',
      }),
    ).toThrow(/empty/);
  });

  it('fallback scan uses ONLY mlx-runtime-root.txt when present and never falls back to stale out/mlx-cmake in the same outDir', () => {
    const targetRoot = join(root, 'target');
    const outDir = join(targetRoot, TRIPLE, 'release', 'build', 'mlx-sys-12345678', 'out');
    // Stale legacy source install inside the same outDir.
    populateMlxRoot(join(outDir, 'mlx-cmake'));
    // Authoritative file points to a missing/incomplete prebuilt root.
    const missingPrebuilt = join(root, 'missing-prebuilt');
    writeFileSync(join(outDir, WINDOWS_RUNTIME_ROOT_FILE), `${missingPrebuilt}\n`);

    const warnings: string[] = [];
    expect(() =>
      selectWindowsMlxRoot({
        addonBinary: makeAddonWithMarker(),
        addonPath: 'mlx-core.win32-arm64-msvc.node',
        targetRoot,
        triple: TRIPLE,
        profile: 'release',
        warn: (msg) => warnings.push(msg),
      }),
    ).toThrow(/missing openblas\.dll/);
    expect(warnings.some((w) => w.includes('falling back to scanning'))).toBe(true);
  });

  it('fallback scan ranks across mlx-sys-* and mlx-sys/* by mtime descending and accepts legacy out/mlx-cmake', () => {
    const targetRoot = join(root, 'target');

    // Older legacy source install under mlx-sys-*
    const oldOut = join(targetRoot, TRIPLE, 'release', 'build', 'mlx-sys-aaaa1111', 'out');
    const oldCmake = join(oldOut, 'mlx-cmake');
    populateMlxRoot(oldCmake);
    const oldTime = new Date(Date.now() - 5 * 86_400_000);
    utimesSync(join(oldCmake, 'openblas.dll'), oldTime, oldTime);

    // Newer prebuilt root under Cargo 1.100 mlx-sys/<hash>/out
    const newOut = join(targetRoot, TRIPLE, 'release', 'build', 'mlx-sys', 'bbbb2222', 'out');
    const newPrebuilt = join(root, 'prebuilt-fresh');
    populateMlxRoot(newPrebuilt);
    mkdirSync(newOut, { recursive: true });
    writeFileSync(join(newOut, WINDOWS_RUNTIME_ROOT_FILE), `${newPrebuilt}\n`);
    // Even if the unpacked cache DLL has an older mtime, the fresh mlx-runtime-root.txt wins.
    const ancientCache = new Date(Date.now() - 30 * 86_400_000);
    utimesSync(join(newPrebuilt, 'openblas.dll'), ancientCache, ancientCache);

    const warnings: string[] = [];
    const picked = selectWindowsMlxRoot({
      addonBinary: makeAddonWithMarker(),
      addonPath: 'mlx-core.win32-arm64-msvc.node',
      targetRoot,
      triple: TRIPLE,
      profile: 'release',
      warn: (msg) => warnings.push(msg),
    });
    expect(picked).toEqual({
      mlxRoot: newPrebuilt,
      source: 'fallback-scan',
    });
    expect(warnings.some((w) => w.includes('Multiple Windows mlx-sys runtime candidates'))).toBe(true);

    // Removing the newer dir leaves the legacy out/mlx-cmake candidate valid.
    rmSync(join(targetRoot, TRIPLE, 'release', 'build', 'mlx-sys'), { recursive: true, force: true });
    const legacyPicked = selectWindowsMlxRoot({
      addonBinary: makeAddonWithMarker(),
      addonPath: 'mlx-core.win32-arm64-msvc.node',
      targetRoot,
      triple: TRIPLE,
      profile: 'release',
    });
    expect(legacyPicked).toEqual({
      mlxRoot: oldCmake,
      source: 'fallback-scan',
    });
  });

  it('throws when no build directories exist in fallback scan', () => {
    expect(() =>
      selectWindowsMlxRoot({
        addonBinary: makeAddonWithMarker(),
        addonPath: 'mlx-core.win32-arm64-msvc.node',
        targetRoot: join(root, 'nonexistent-target'),
        triple: TRIPLE,
        profile: 'release',
      }),
    ).toThrow(/cannot find a Windows MLX runtime root/);
  });

  it.skipIf(runningAsRoot || process.platform === 'win32')(
    'surfaces non-ENOENT/ENOTDIR errors during candidate probing instead of swallowing them',
    () => {
      const targetRoot = join(root, 'target');
      const buildRoot = join(targetRoot, TRIPLE, 'release', 'build');
      mkdirSync(buildRoot, { recursive: true });
      chmodSync(buildRoot, 0o000);
      try {
        expect(() =>
          selectWindowsMlxRoot({
            addonBinary: makeAddonWithMarker(),
            addonPath: 'mlx-core.win32-arm64-msvc.node',
            targetRoot,
            triple: TRIPLE,
            profile: 'release',
          }),
        ).toThrow(/cannot list/);
      } finally {
        chmodSync(buildRoot, 0o755);
      }
    },
  );
});
