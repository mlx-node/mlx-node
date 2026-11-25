import { mkdirSync, mkdtempSync, rmSync, utimesSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import {
  LINUX_CUDA_HEADER_MARKERS,
  LINUX_CUDA_INCLUDE_MARKER_PREFIX,
  extractBakedLinuxCudaIncludeDir,
  hasLinuxCudaHeaderMarkers,
  selectLinuxCudaIncludeDir,
} from '../../packages/core/linux-cuda-include';

const TRIPLE = 'aarch64-unknown-linux-gnu';
const ADDON_PATH = '/repo/packages/core/mlx-core.linux-arm64-gnu.node';

/** Synthetic addon binary: NUL-separated strings, like a real ELF `.rodata`. */
function fakeAddon(strings: string[]): Buffer {
  return Buffer.concat([
    Buffer.from([0x7f, 0x45, 0x4c, 0x46, 0x00]),
    Buffer.from(strings.join('\0')),
    Buffer.from([0]),
  ]);
}

function populateHeaders(includeDir: string, skipMarker?: string): void {
  for (const marker of LINUX_CUDA_HEADER_MARKERS) {
    if (marker === skipMarker) continue;
    const filePath = join(includeDir, marker);
    mkdirSync(dirname(filePath), { recursive: true });
    writeFileSync(filePath, `// ${marker}\n`);
  }
}

function setAgeDays(path: string, ageDays: number): void {
  const when = new Date(Date.now() - ageDays * 86_400_000);
  utimesSync(path, when, when);
}

describe('extractBakedLinuxCudaIncludeDir', () => {
  it('extracts a unique baked path and ignores empty/template/compiler-flag decoys', () => {
    const baked =
      '/home/runner/work/mlx-node/target/aarch64-unknown-linux-gnu/release/build/mlx-sys-aaaa1111/out/cuda-include';
    const addon = fakeAddon([
      'unrelated string',
      `${LINUX_CUDA_INCLUDE_MARKER_PREFIX}`,
      `${LINUX_CUDA_INCLUDE_MARKER_PREFIX}...`,
      `${LINUX_CUDA_INCLUDE_MARKER_PREFIX}<out>/cuda-include`,
      `-D${LINUX_CUDA_INCLUDE_MARKER_PREFIX}"/compiler/switch/cuda-include"`,
      `${LINUX_CUDA_INCLUDE_MARKER_PREFIX}${baked}`,
      `${LINUX_CUDA_INCLUDE_MARKER_PREFIX}${baked}`,
    ]);
    expect(extractBakedLinuxCudaIncludeDir(addon)).toBe(baked);
  });

  it('returns undefined when no valid marker is baked', () => {
    const addon = fakeAddon([
      'no marker here',
      `${LINUX_CUDA_INCLUDE_MARKER_PREFIX}`,
      `${LINUX_CUDA_INCLUDE_MARKER_PREFIX}%s`,
    ]);
    expect(extractBakedLinuxCudaIncludeDir(addon)).toBeUndefined();
  });

  it('throws when multiple distinct baked paths are present', () => {
    const addon = fakeAddon([
      `${LINUX_CUDA_INCLUDE_MARKER_PREFIX}/build/one/out/cuda-include`,
      `${LINUX_CUDA_INCLUDE_MARKER_PREFIX}/build/two/out/cuda-include`,
    ]);
    expect(() => extractBakedLinuxCudaIncludeDir(addon)).toThrow(/2 distinct MLX_NODE_CUDA_INCLUDE_DIR= paths/);
  });
});

describe('selectLinuxCudaIncludeDir', () => {
  let root: string;
  let targetRoot: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'linux-cuda-include-'));
    targetRoot = join(root, 'target');
    mkdirSync(targetRoot, { recursive: true });
  });

  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  function addLegacyBuildDir(
    baseRoot: string,
    rel: string,
    ageDays: number,
    subdir: 'cuda-include' | 'mlx-cmake/include' = 'cuda-include',
  ): string {
    const scriptDir = join(baseRoot, rel);
    const includeDir = join(scriptDir, 'out', subdir);
    populateHeaders(includeDir);
    const stamp = join(scriptDir, 'invoked.timestamp');
    writeFileSync(stamp, 'stamp');
    setAgeDays(includeDir, ageDays);
    setAgeDays(stamp, ageDays);
    return includeDir;
  }

  function addCargo1100BuildDir(
    baseRoot: string,
    rel: string,
    ageDays: number,
    subdir: 'cuda-include' | 'mlx-cmake/include' = 'cuda-include',
  ): string {
    const unitDir = join(baseRoot, rel);
    const includeDir = join(unitDir, 'out', subdir);
    populateHeaders(includeDir);
    const runDir = join(unitDir, 'run');
    mkdirSync(runDir, { recursive: true });
    const stamp = join(runDir, 'invoked.timestamp');
    writeFileSync(stamp, 'stamp');
    setAgeDays(includeDir, ageDays);
    setAgeDays(stamp, ageDays);
    return includeDir;
  }

  it('authoritatively selects the baked MLX_NODE_CUDA_INCLUDE_DIR even when a newer stale mlx-sys-* tree exists under target/', async () => {
    const bakedDir = join(
      root,
      'external-build-dir',
      TRIPLE,
      'release',
      'build',
      'mlx-sys-1111aaaa',
      'out',
      'cuda-include',
    );
    populateHeaders(bakedDir);
    setAgeDays(bakedDir, 7);

    // Newer stale build inside targetRoot — must NOT win over the authoritative baked marker.
    addLegacyBuildDir(targetRoot, `${TRIPLE}/release/build/mlx-sys-9999ffff`, 0);

    const selected = await selectLinuxCudaIncludeDir({
      addonBinary: fakeAddon([`${LINUX_CUDA_INCLUDE_MARKER_PREFIX}${bakedDir}`]),
      addonPath: ADDON_PATH,
      targetRoot,
      triple: TRIPLE,
      profile: 'release',
    });

    expect(selected).toEqual({ includeDir: bakedDir, source: 'baked' });
    expect(await hasLinuxCudaHeaderMarkers(selected.includeDir)).toBe(true);
  });

  it('throws loudly naming the baked path when headers are missing there, instead of falling back to a stale tree', async () => {
    // Valid stale tree under targetRoot that fallback scan would otherwise pick.
    addLegacyBuildDir(targetRoot, `${TRIPLE}/release/build/mlx-sys-9999ffff`, 0);

    const missingBakedDir = join(root, 'deleted-build', 'out', 'cuda-include');
    await expect(
      selectLinuxCudaIncludeDir({
        addonBinary: fakeAddon([`${LINUX_CUDA_INCLUDE_MARKER_PREFIX}${missingBakedDir}`]),
        addonPath: ADDON_PATH,
        targetRoot,
        triple: TRIPLE,
        profile: 'release',
      }),
    ).rejects.toThrow(missingBakedDir);

    const incompleteBakedDir = join(root, 'incomplete-build', 'out', 'cuda-include');
    populateHeaders(incompleteBakedDir, 'cutlass/cutlass.h');
    await expect(
      selectLinuxCudaIncludeDir({
        addonBinary: fakeAddon([`${LINUX_CUDA_INCLUDE_MARKER_PREFIX}${incompleteBakedDir}`]),
        addonPath: ADDON_PATH,
        targetRoot,
        triple: TRIPLE,
        profile: 'release',
      }),
    ).rejects.toThrow(incompleteBakedDir);
  });

  it('fallback scan supports a split CARGO_BUILD_BUILD_DIR and Cargo 1.100 mlx-sys/<hash>/out/cuda-include layout', async () => {
    const splitBuildDir = join(root, 'cargo-build-dir');
    mkdirSync(splitBuildDir, { recursive: true });

    // Older legacy dir under targetRoot using mlx-cmake/include.
    addLegacyBuildDir(targetRoot, `${TRIPLE}/release/build/mlx-sys-aaaa1111`, 5, 'mlx-cmake/include');

    // Fresher Cargo 1.100 dir under split CARGO_BUILD_BUILD_DIR using out/cuda-include.
    const cargo1100Dir = addCargo1100BuildDir(
      splitBuildDir,
      `${TRIPLE}/release/build/mlx-sys/bbbb2222bbbb2222`,
      0,
      'cuda-include',
    );

    const selectedViaEnv = await selectLinuxCudaIncludeDir({
      addonBinary: fakeAddon(['no baked marker']),
      addonPath: ADDON_PATH,
      targetRoot,
      triple: TRIPLE,
      profile: 'release',
      env: { CARGO_BUILD_BUILD_DIR: splitBuildDir },
    });
    expect(selectedViaEnv).toEqual({ includeDir: cargo1100Dir, source: 'scan' });

    const selectedViaOpt = await selectLinuxCudaIncludeDir({
      addonBinary: fakeAddon(['no baked marker']),
      addonPath: ADDON_PATH,
      targetRoot,
      buildDir: splitBuildDir,
      triple: TRIPLE,
      profile: 'release',
    });
    expect(selectedViaOpt).toEqual({ includeDir: cargo1100Dir, source: 'scan' });
  });

  it('throws a descriptive error when headers are missing everywhere', async () => {
    const missingBuildDir = join(root, 'empty-build-dir');
    await expect(
      selectLinuxCudaIncludeDir({
        addonBinary: fakeAddon(['no baked marker']),
        addonPath: ADDON_PATH,
        targetRoot,
        buildDir: missingBuildDir,
        triple: TRIPLE,
        profile: 'release',
      }),
    ).rejects.toThrow(
      /Cannot find Linux CUDA NVRTC headers.*cccl\/cuda\/std\/array.*cute\/tensor\.hpp.*cutlass\/cutlass\.h/,
    );
  });
});
