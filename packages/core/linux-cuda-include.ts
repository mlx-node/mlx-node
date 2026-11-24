// Deterministic selection of the Linux CUDA NVRTC header directory
// (`cccl`, `cute`, `cutlass`) produced by the `mlx-sys` build that the
// just-built native addon actually linked against, mirroring
// `packages/core/metallib-select.ts`.
//
//   1. AUTHORITATIVE binding: `crates/mlx-sys/build.rs` bakes
//      `MLX_NODE_CUDA_INCLUDE_DIR=<OUT_DIR>/cuda-include` into source-built
//      `libmlx_ffi.a` (`mlx_stream_cuda.cpp`). Reading that marker straight out
//      of the freshly built `.node` addon binds header packaging to the exact
//      `mlx-sys` build, immune to split `CARGO_BUILD_BUILD_DIR` layouts and
//      newer stale `mlx-sys-*` directories under `target/`.
//      If the marker is present but the named directory is missing the required
//      headers, fail loudly — never silently fall back to scanning stale trees.
//   2. Heuristic fallback (only when no marker is baked, e.g. an older addon):
//      scan `buildDir` (if provided, or `env?.CARGO_BUILD_BUILD_DIR`) and
//      `targetRoot` across `<root>/<triple>/<profile>/build` and
//      `<root>/<profile>/build` (handling both legacy `mlx-sys-*` and Cargo
//      1.100 `mlx-sys/*`), picking the freshest valid `out/cuda-include` or
//      `out/mlx-cmake/include` by `mtimeMs`.
import { lstat, readdir, stat } from 'node:fs/promises';
import { join, resolve } from 'node:path';

export const LINUX_CUDA_INCLUDE_DIRS = ['cccl', 'cute', 'cutlass'] as const;
export const LINUX_CUDA_HEADER_MARKERS = ['cccl/cuda/std/array', 'cute/tensor.hpp', 'cutlass/cutlass.h'] as const;
export const LINUX_CUDA_INCLUDE_MARKER_PREFIX = 'MLX_NODE_CUDA_INCLUDE_DIR=';

/** Longest plausible baked path; bounds the NUL-terminated-string scan. */
const MAX_BAKED_PATH_BYTES = 4096;

function isAbsenceError(error: unknown): boolean {
  const code = (error as NodeJS.ErrnoException | null)?.code;
  return code === 'ENOENT' || code === 'ENOTDIR';
}

async function statIfExists(path: string, followSymlinks = true) {
  try {
    return await (followSymlinks ? stat(path) : lstat(path));
  } catch (error: unknown) {
    if (isAbsenceError(error)) return undefined;
    throw error;
  }
}

export async function hasLinuxCudaHeaderMarkers(includeDir: string): Promise<boolean> {
  const stats = await Promise.all(LINUX_CUDA_HEADER_MARKERS.map((marker) => statIfExists(join(includeDir, marker))));
  return stats.every((info) => info?.isFile() === true);
}

function isPrecedingIdentifierByte(byte: number): boolean {
  return (
    (byte >= 0x30 && byte <= 0x39) || // 0-9
    (byte >= 0x41 && byte <= 0x5a) || // A-Z
    (byte >= 0x61 && byte <= 0x7a) || // a-z
    byte === 0x5f // _
  );
}

function isPlausibleBakedPath(value: string): boolean {
  if (value.length === 0 || value.trim() !== value) return false;
  // Ignore quoted macro definitions (e.g. `-DMLX_NODE_CUDA_INCLUDE_DIR="..."`)
  // and template/placeholder strings (`...`, `<path>`, `{path}`, `%s`, `$VAR`).
  const first = value[0];
  if (first === '"' || first === "'" || first === '<' || first === '{' || first === '%' || first === '$') {
    return false;
  }
  if (!value.includes('/') && !value.includes('\\')) return false;
  for (let i = 0; i < value.length; i++) {
    const code = value.charCodeAt(i);
    if (code < 0x20 || code === 0x7f || code === 0xfffd) return false;
  }
  return true;
}

/**
 * Scan `addonBinary` for baked `MLX_NODE_CUDA_INCLUDE_DIR=<path>\0` markers.
 * Returns the unique non-empty UTF-8 path (deduplicating identical repeats),
 * `undefined` when no valid marker is present, or throws when multiple
 * distinct paths are baked into one addon.
 */
export function extractBakedLinuxCudaIncludeDir(addonBinary: Buffer): string | undefined {
  const prefixBytes = Buffer.from(LINUX_CUDA_INCLUDE_MARKER_PREFIX, 'utf-8');
  const found = new Set<string>();
  let at = addonBinary.indexOf(prefixBytes);
  while (at !== -1) {
    const nextSearch = at + prefixBytes.length;
    if (at === 0 || !isPrecedingIdentifierByte(addonBinary[at - 1] ?? 0)) {
      const valueStart = nextSearch;
      const endCap = Math.min(addonBinary.length, valueStart + MAX_BAKED_PATH_BYTES);
      let end = valueStart;
      while (end < endCap && addonBinary[end] !== 0) end++;
      const terminated = end === addonBinary.length || addonBinary[end] === 0;
      if (terminated) {
        const value = addonBinary.subarray(valueStart, end).toString('utf-8');
        if (isPlausibleBakedPath(value)) {
          found.add(value);
        }
      }
    }
    at = addonBinary.indexOf(prefixBytes, nextSearch);
  }
  if (found.size === 0) return undefined;
  const paths = [...found];
  if (paths.length > 1) {
    throw new Error(
      `[build.ts linux-cuda-include gate] the addon bakes ${paths.length} distinct ` +
        `${LINUX_CUDA_INCLUDE_MARKER_PREFIX} paths — cannot bind a single cuda-include directory to it:\n` +
        paths.map((p) => `  ${p}`).join('\n'),
    );
  }
  return paths[0];
}

export interface SelectLinuxCudaIncludeDirOptions {
  addonBinary: Buffer;
  addonPath: string;
  targetRoot: string;
  buildDir?: string | undefined;
  triple: string;
  profile: string;
  env?: Record<string, string | undefined>;
}

export interface SelectedLinuxCudaIncludeDir {
  includeDir: string;
  source: 'baked' | 'scan';
}

function nonEmpty(value: string | undefined): string | undefined {
  return value !== undefined && value !== '' ? value : undefined;
}

export async function selectLinuxCudaIncludeDir(
  opts: SelectLinuxCudaIncludeDirOptions,
): Promise<SelectedLinuxCudaIncludeDir> {
  const bakedDir = extractBakedLinuxCudaIncludeDir(opts.addonBinary);
  if (bakedDir !== undefined) {
    if (await hasLinuxCudaHeaderMarkers(bakedDir)) {
      return { includeDir: bakedDir, source: 'baked' };
    }
    throw new Error(
      `[build.ts linux-cuda-include gate] ${opts.addonPath} bakes ` +
        `${LINUX_CUDA_INCLUDE_MARKER_PREFIX}${bakedDir}, but required Linux CUDA NVRTC headers ` +
        `(${LINUX_CUDA_HEADER_MARKERS.join(', ')}) are missing there. Refusing to fall back to ` +
        `scanning target dirs (which could ship stale headers from a different mlx-sys build). ` +
        `Re-run the native build so the addon and its staged cuda-include directory come from ` +
        `the same mlx-sys build.`,
    );
  }

  const resolvedBuildDir = nonEmpty(opts.buildDir) ?? nonEmpty(opts.env?.['CARGO_BUILD_BUILD_DIR']);
  const roots: string[] = [];
  const seenRoots = new Set<string>();
  for (const root of [resolvedBuildDir, opts.targetRoot]) {
    const clean = nonEmpty(root);
    if (clean === undefined) continue;
    const key = resolve(clean);
    if (seenRoots.has(key)) continue;
    seenRoots.add(key);
    roots.push(clean);
  }

  const buildRoots: string[] = [];
  for (const root of roots) {
    buildRoots.push(join(root, opts.triple, opts.profile, 'build'), join(root, opts.profile, 'build'));
  }

  const candidates: Array<{ includeDir: string; mtime: number }> = [];
  for (const buildRoot of buildRoots) {
    let entries: string[];
    try {
      entries = await readdir(buildRoot);
    } catch (error: unknown) {
      if (isAbsenceError(error)) continue;
      throw error;
    }

    const scriptDirs: Array<{ dir: string; probes: string[] }> = [];
    for (const name of entries) {
      if (name.startsWith('mlx-sys-')) {
        const dir = join(buildRoot, name);
        scriptDirs.push({
          dir,
          probes: [join(dir, 'invoked.timestamp'), join(dir, 'output')],
        });
      } else if (name === 'mlx-sys') {
        const packageDir = join(buildRoot, name);
        let units: string[];
        try {
          units = await readdir(packageDir);
        } catch (error: unknown) {
          if (isAbsenceError(error)) continue;
          throw error;
        }
        for (const unit of units) {
          const dir = join(packageDir, unit);
          scriptDirs.push({
            dir,
            probes: [join(dir, 'run', 'invoked.timestamp'), join(dir, 'run', 'stdout')],
          });
        }
      }
    }

    for (const { dir, probes } of scriptDirs) {
      const outDir = join(dir, 'out');
      let includeDir: string | undefined;
      for (const candidate of [join(outDir, 'cuda-include'), join(outDir, 'mlx-cmake', 'include')]) {
        if (await hasLinuxCudaHeaderMarkers(candidate)) {
          includeDir = candidate;
          break;
        }
      }
      if (includeDir === undefined) continue;

      const freshnessStats = await Promise.all([
        statIfExists(join(outDir, 'libmlx_ffi.a'), true),
        statIfExists(includeDir, false),
        ...probes.map((probe) => statIfExists(probe, true)),
      ]);
      const mtime = freshnessStats.reduce((max, info) => Math.max(max, info?.mtimeMs ?? 0), 0);
      candidates.push({ includeDir, mtime });
    }
  }

  const picked = candidates.sort((a, b) => b.mtime - a.mtime)[0];
  if (picked === undefined) {
    throw new Error(
      `[build.ts linux-cuda-include gate] Cannot find Linux CUDA NVRTC headers ` +
        `(${LINUX_CUDA_HEADER_MARKERS.join(', ')}) under ${buildRoots.join(' or ')} ` +
        `(checked mlx-sys{-*,/*}/out/cuda-include and mlx-sys{-*,/*}/out/mlx-cmake/include).`,
    );
  }
  return { includeDir: picked.includeDir, source: 'scan' };
}
