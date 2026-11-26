/**
 * `paged_attn.metallib` carries the K-quant, segmented SDPA and mixed-affine
 * kernels, and the addon has no JIT fallback for them. A library from before
 * either change is ~19.5 MB or ~31.8 MB, well over the 10 MiB size floor, so
 * the floor alone would sign it into the bundle and the first GGUF K-quant
 * or mixed-affine matmul would throw. The payload check runs
 * the same marker gate `yarn build:native` does: both libraries must carry
 * their NAX kernels, which every Metal build at the 26.2 floor compiles.
 */

import { existsSync, mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import {
  BASE_KERNEL_MARKERS,
  BRIDGE_KERNEL_MARKERS,
  KQUANT_KERNEL_MARKERS,
  KQUANT_NAX_KERNEL_MARKERS,
  MLX_NAX_ONLY_MARKERS,
  NAX_KERNEL_MARKERS,
} from '../../core/metallib-select.js';
import { copyPdfjs, PDFJS_WORKER, pdfjsBase } from '../../dashboard/ui/pdfjs-assets.js';
import {
  APP_PDFJS_RUNTIME_DIRS,
  assertDarwinAppRuntimeAssets,
  checkAddonMinOs,
  checkMetallibPair,
  checkSpa,
  DARWIN_APP_RUNTIME_FILES,
  findStagedNativeAddons,
  needsDylibIdRewrite,
  parseDylibIds,
  PayloadError,
} from '../scripts/payload.js';

const PAGED_KERNELS = [
  'paged_attention_bfloat16_t_cache_bfloat16_t_hs128_bs16_nt256_nsl32_ps0',
  'reshape_and_cache_kv_bfloat16_t_cache_bfloat16_t',
];
const MLX_BASE = [...BASE_KERNEL_MARKERS];
const MLX_NAX = [...BASE_KERNEL_MARKERS, ...MLX_NAX_ONLY_MARKERS, ...NAX_KERNEL_MARKERS];
/** The previous pin's library: NAX kernels, but not every name of the current pin. */
const MLX_PREVIOUS_PIN = [...BASE_KERNEL_MARKERS, ...MLX_NAX_ONLY_MARKERS, ...NAX_KERNEL_MARKERS.slice(0, -1)];
const PAGED_BASE = [...PAGED_KERNELS, ...KQUANT_KERNEL_MARKERS, ...BRIDGE_KERNEL_MARKERS];
const PAGED_NAX = [...PAGED_BASE, ...KQUANT_NAX_KERNEL_MARKERS];

/**
 * A complete MTLB container over the 10 MiB paged floor that names exactly
 * `kernels`: recognized header (min-OS 26.2), declared size = byte length.
 */
function library(kernels: readonly string[], magic = 'MTLB'): Buffer {
  const header = Buffer.alloc(24);
  header.write(magic, 0, 'latin1');
  header.writeUInt16LE(0x8001, 4);
  header.writeUInt16LE(2, 6);
  header.writeUInt16LE(9, 8);
  header[11] = 0x81;
  header.writeUInt16LE(26, 12);
  header[14] = 2;
  const names = Buffer.from(['', ...kernels].join('\0') + '\0');
  const metallib = Buffer.concat([header, names, Buffer.alloc(11 * 1024 * 1024)]);
  metallib.writeBigUInt64LE(BigInt(metallib.byteLength), 16);
  return metallib;
}

/** `metallib` cut to `bytes`, as an interrupted copy leaves it. */
const truncated = (metallib: Buffer, bytes = 10.5 * 1024 * 1024) => metallib.subarray(0, bytes);

describe('checkMetallibPair', () => {
  let dir: string;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'desktop-payload-'));
  });
  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  function pair(mlx: Buffer, paged: Buffer): [string, string] {
    const mlxPath = join(dir, 'mlx.metallib');
    const pagedPath = join(dir, 'paged_attn.metallib');
    writeFileSync(mlxPath, mlx);
    writeFileSync(pagedPath, paged);
    return [mlxPath, pagedPath];
  }

  it('rejects a paged-only library above the size floor', () => {
    const [m, p] = pair(library(MLX_NAX), library(PAGED_KERNELS));
    expect(() => checkMetallibPair(m, p)).toThrow(PayloadError);
    expect(() => checkMetallibPair(m, p)).toThrow(/missing K-quant kernel/);
  });

  it('rejects a library with the K-quant kernels but not the segmented SDPA / mixed-affine ones', () => {
    const paged = library([...PAGED_KERNELS, ...KQUANT_KERNEL_MARKERS, ...KQUANT_NAX_KERNEL_MARKERS]);
    expect(() => checkMetallibPair(...pair(library(MLX_NAX), paged))).toThrow(
      /missing segmented SDPA \/ mixed-affine kernel/,
    );
  });

  it('rejects either file when it is shorter than its header declares', () => {
    expect(() => checkMetallibPair(...pair(library(MLX_NAX), truncated(library(PAGED_NAX))))).toThrow(
      /header declares/,
    );
    expect(() => checkMetallibPair(...pair(truncated(library(MLX_NAX)), library(PAGED_NAX)))).toThrow(
      /header declares/,
    );
  });

  it('accepts a pair that both carry NAX', () => {
    expect(() => checkMetallibPair(...pair(library(MLX_NAX), library(PAGED_NAX)))).not.toThrow();
  });

  it('rejects a paged library without the K-quant NAX kernels', () => {
    expect(() => checkMetallibPair(...pair(library(MLX_NAX), library(PAGED_BASE)))).toThrow(
      KQUANT_NAX_KERNEL_MARKERS[0],
    );
  });

  it('rejects an MLX_METAL_NO_NAX mlx.metallib', () => {
    for (const paged of [PAGED_BASE, PAGED_NAX]) {
      expect(() => checkMetallibPair(...pair(library(MLX_BASE), library(paged)))).toThrow(/has no NAX kernels/);
    }
  });

  it("rejects the previous pin's mlx.metallib (NAX kernels, but not the current pin's names)", () => {
    expect(() => checkMetallibPair(...pair(library(MLX_PREVIOUS_PIN), library(PAGED_NAX)))).toThrow(
      NAX_KERNEL_MARKERS[NAX_KERNEL_MARKERS.length - 1],
    );
  });

  it('rejects an mlx.metallib with some NAX kernels, or no MTLB magic', () => {
    expect(() =>
      checkMetallibPair(...pair(library([...BASE_KERNEL_MARKERS, MLX_NAX_ONLY_MARKERS[0]]), library(PAGED_NAX))),
    ).toThrow(/is missing NAX kernel/);
    expect(() => checkMetallibPair(...pair(library(MLX_NAX_ONLY_MARKERS), library(PAGED_NAX)))).toThrow(
      /missing the base MLX kernel/,
    );
    expect(() => checkMetallibPair(...pair(library(MLX_NAX, 'NOPE'), library(PAGED_NAX)))).toThrow(
      /MTLB container magic/,
    );
  });

  const core = join(import.meta.dirname, '..', '..', 'core');
  const built: [string, string] = [join(core, 'mlx.metallib'), join(core, 'paged_attn.metallib')];
  it.skipIf(!built.every((path) => existsSync(path)))('accepts the pair from `yarn build:native`', () => {
    expect(() => checkMetallibPair(...built)).not.toThrow();
  });
});

describe('checkAddonMinOs', () => {
  let dir: string;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'desktop-addon-'));
  });
  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  /** A thin arm64 Mach-O header with one LC_BUILD_VERSION (macOS, `minos`). */
  function addon(major: number, minor: number): string {
    const header = Buffer.alloc(32 + 24);
    header.writeUInt32LE(0xfeedfacf, 0);
    header.writeUInt32LE(0x0100000c, 4);
    header.writeUInt32LE(8, 12);
    header.writeUInt32LE(1, 16);
    header.writeUInt32LE(24, 20);
    header.writeUInt32LE(0x32, 32);
    header.writeUInt32LE(24, 36);
    header.writeUInt32LE(1, 40);
    header.writeUInt32LE((major << 16) | (minor << 8), 44);
    const path = join(dir, 'mlx-core.darwin-arm64.node');
    writeFileSync(path, header);
    return path;
  }

  it('rejects an addon linked below the 26.2 floor, which would lower LSMinimumSystemVersion', () => {
    expect(() => checkAddonMinOs(addon(11, 0))).toThrow(PayloadError);
    expect(() => checkAddonMinOs(addon(11, 0))).toThrow(/minos 11\.0, below/);
    expect(() => checkAddonMinOs(addon(26, 2))).not.toThrow();
  });

  const built = join(import.meta.dirname, '..', '..', 'core', 'mlx-core.darwin-arm64.node');
  it.skipIf(!existsSync(built))('accepts the addon from `yarn build:native`', () => {
    expect(() => checkAddonMinOs(built)).not.toThrow();
  });
});

describe('findStagedNativeAddons', () => {
  let root: string;
  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'desktop-addons-'));
  });
  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  function touch(relativePath: string, contents = 'x'): string {
    const fullPath = join(root, relativePath);
    mkdirSync(dirname(fullPath), { recursive: true });
    writeFileSync(fullPath, contents);
    return fullPath;
  }

  it('finds all regular *.node files recursively in sorted order and skips symlinks', () => {
    const nm = join(root, 'node_modules');
    const keyring = touch('node_modules/@napi-rs/keyring-darwin-arm64/keyring.darwin-arm64.node');
    const canvas = touch('node_modules/@napi-rs/canvas-darwin-arm64/skia.darwin-arm64.node');
    const nested = touch('node_modules/pkg/node_modules/sub/native.node');
    touch('node_modules/pkg/index.js');
    touch('node_modules/pkg/native.node.map');

    const outsideFile = touch('outside/linked.node');
    const outsideDir = dirname(touch('outside-dir/inside.node'));
    symlinkSync(outsideFile, join(nm, 'linked.node'));
    symlinkSync(outsideDir, join(nm, 'linked-dir'));

    expect(findStagedNativeAddons(nm)).toEqual([canvas, keyring, nested].sort());
  });

  it('returns an empty list when the directory does not exist', () => {
    expect(findStagedNativeAddons(join(root, 'missing'))).toEqual([]);
  });
});

describe('parseDylibIds and needsDylibIdRewrite', () => {
  it('detects a thin Mach-O dylib leaking a build path', () => {
    const otool = [
      'skia.darwin-arm64.node:',
      'Load command 0',
      '      cmd LC_SEGMENT_64',
      'Load command 3',
      '          cmd LC_ID_DYLIB',
      '      cmdsize 120',
      '         name /Users/runner/work/canvas/canvas/target/aarch64-apple-darwin/release/deps/libcanvas.dylib (offset 24)',
      '   time stamp 1 Thu Jan  1 07:30:01 1970',
      'Load command 4',
      '          cmd LC_LOAD_DYLIB',
      '      cmdsize 56',
      '         name /usr/lib/libSystem.B.dylib (offset 24)',
    ].join('\n');

    const ids = parseDylibIds(otool);
    expect(ids).toEqual(['/Users/runner/work/canvas/canvas/target/aarch64-apple-darwin/release/deps/libcanvas.dylib']);
    expect(needsDylibIdRewrite(ids)).toBe(true);
  });

  it('accepts an already-rewritten @rpath install name and rejects @rpath containing /target/', () => {
    const clean = [
      'Load command 3',
      '          cmd LC_ID_DYLIB',
      '      cmdsize 56',
      '         name @rpath/skia.darwin-arm64.node (offset 24)',
    ].join('\n');
    expect(parseDylibIds(clean)).toEqual(['@rpath/skia.darwin-arm64.node']);
    expect(needsDylibIdRewrite(parseDylibIds(clean))).toBe(false);
    expect(needsDylibIdRewrite(['@rpath/build/target/release/libfoo.dylib'])).toBe(true);
  });

  it('handles universal Mach-O binaries with multiple slices', () => {
    const mixedUniversal = [
      'addon.node (architecture x86_64):',
      'Load command 3',
      '          cmd LC_ID_DYLIB',
      '         name @rpath/addon.node (offset 24)',
      'addon.node (architecture arm64):',
      'Load command 3',
      '          cmd LC_ID_DYLIB',
      '         name /Users/builder/project/target/aarch64-apple-darwin/release/libaddon.dylib (offset 24)',
    ].join('\n');
    const cleanUniversal = [
      'addon.node (architecture x86_64):',
      'Load command 3',
      '          cmd LC_ID_DYLIB',
      '         name @rpath/addon.node (offset 24)',
      'addon.node (architecture arm64):',
      'Load command 3',
      '          cmd LC_ID_DYLIB',
      '         name @rpath/addon.node (offset 24)',
    ].join('\n');

    expect(parseDylibIds(mixedUniversal)).toEqual([
      '@rpath/addon.node',
      '/Users/builder/project/target/aarch64-apple-darwin/release/libaddon.dylib',
    ]);
    expect(needsDylibIdRewrite(parseDylibIds(mixedUniversal))).toBe(true);

    expect(parseDylibIds(cleanUniversal)).toEqual(['@rpath/addon.node', '@rpath/addon.node']);
    expect(needsDylibIdRewrite(parseDylibIds(cleanUniversal))).toBe(false);
  });

  it('returns no rewrite for MH_BUNDLE (0 LC_ID_DYLIB) and non-Mach-O (ELF/PE32+) output', () => {
    const mhBundle = [
      'bundle.node:',
      'Load command 0',
      '      cmd LC_SEGMENT_64',
      'Load command 1',
      '      cmd LC_LOAD_DYLIB',
      '         name /usr/lib/libSystem.B.dylib (offset 24)',
      'Load command 2',
      '      cmd LC_RPATH',
      '         path @loader_path (offset 12)',
    ].join('\n');
    const elfOutput = 'addon.linux-x64-gnu.node: is not an object file\n';
    const peOutput = 'addon.win32-arm64-msvc.node: is not an object file\n';

    expect(parseDylibIds(mhBundle)).toEqual([]);
    expect(needsDylibIdRewrite(parseDylibIds(mhBundle))).toBe(false);

    expect(parseDylibIds(elfOutput)).toEqual([]);
    expect(needsDylibIdRewrite(parseDylibIds(elfOutput))).toBe(false);

    expect(parseDylibIds(peOutput)).toEqual([]);
    expect(needsDylibIdRewrite(parseDylibIds(peOutput))).toBe(false);
  });
});

describe('assertDarwinAppRuntimeAssets', () => {
  let appDir: string;

  beforeEach(() => {
    appDir = mkdtempSync(join(tmpdir(), 'desktop-darwin-assets-'));
    for (const relPath of DARWIN_APP_RUNTIME_FILES) {
      writeAsset(relPath, 'ok');
    }
    writeAsset('node_modules/pdfjs-dist/wasm/openjpeg.wasm', 'wasm');
    writeAsset('node_modules/pdfjs-dist/cmaps/Adobe-GB1-UCS2.bcmap', 'cmap');
    writeAsset('node_modules/pdfjs-dist/standard_fonts/LiberationSans-Regular.ttf', 'font');
  });

  afterEach(() => {
    rmSync(appDir, { recursive: true, force: true });
  });

  function writeAsset(relativePath: string, contents: string): void {
    const fullPath = join(appDir, relativePath);
    mkdirSync(dirname(fullPath), { recursive: true });
    writeFileSync(fullPath, contents);
  }

  it('accepts a staged macOS app containing all required addons and pdfjs-dist assets', () => {
    expect(() => assertDarwinAppRuntimeAssets(appDir)).not.toThrow();
  });

  it.each(DARWIN_APP_RUNTIME_FILES)('rejects when required runtime file %s is missing or empty', (relPath) => {
    const fullPath = join(appDir, relPath);
    writeFileSync(fullPath, '');
    expect(() => assertDarwinAppRuntimeAssets(appDir)).toThrow(PayloadError);
    expect(() => assertDarwinAppRuntimeAssets(appDir)).toThrow(relPath);

    rmSync(fullPath);
    expect(() => assertDarwinAppRuntimeAssets(appDir)).toThrow(PayloadError);
    expect(() => assertDarwinAppRuntimeAssets(appDir)).toThrow(relPath);
  });

  it.each(APP_PDFJS_RUNTIME_DIRS)(
    'rejects when pdfjs-dist asset directory %s is missing or has no non-empty files',
    (relDir) => {
      const fullDir = join(appDir, relDir);
      rmSync(fullDir, { recursive: true, force: true });
      expect(() => assertDarwinAppRuntimeAssets(appDir)).toThrow(PayloadError);
      expect(() => assertDarwinAppRuntimeAssets(appDir)).toThrow(relDir);

      mkdirSync(fullDir, { recursive: true });
      writeFileSync(join(fullDir, 'empty.bin'), '');
      expect(() => assertDarwinAppRuntimeAssets(appDir)).toThrow(PayloadError);
      expect(() => assertDarwinAppRuntimeAssets(appDir)).toThrow(relDir);
    },
  );
});

describe('checkSpa', () => {
  let www: string;
  beforeEach(() => {
    www = mkdtempSync(join(tmpdir(), 'desktop-www-'));
    writeFileSync(join(www, 'index.html'), '<!doctype html>');
  });
  afterEach(() => {
    rmSync(www, { recursive: true, force: true });
  });

  it("the build's pdf.js copy step puts the worker where the page starts it", () => {
    copyPdfjs(www);
    expect(existsSync(join(www, pdfjsBase(), PDFJS_WORKER))).toBe(true);
    expect(() => checkSpa(www)).not.toThrow();
  });

  it('a www with index.html but no pdf.js worker is refused with the build command', () => {
    let caught: unknown;
    try {
      checkSpa(www);
    } catch (err) {
      caught = err;
    }
    expect(caught).toBeInstanceOf(PayloadError);
    expect((caught as PayloadError).message).toContain(join(pdfjsBase(), PDFJS_WORKER));
    expect((caught as PayloadError).remedy).toBe('yarn workspace @mlx-node/dashboard build:ui');
  });

  it('a www with no index.html is refused', () => {
    copyPdfjs(www);
    rmSync(join(www, 'index.html'));
    expect(() => checkSpa(www)).toThrow('index.html is missing');
  });
});
