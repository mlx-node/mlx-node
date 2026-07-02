// Deterministic selection of the mlx.metallib produced by the mlx-sys build
// that the just-built native addon actually linked against.
//
// Background: cargo keeps one `mlx-sys-<hash>/out` dir per (target, profile,
// compiler-metadata) fingerprint, and stale hash dirs from earlier toolchains
// or earlier MLX submodule pins survive next to the current one (locally and
// in CI-restored cargo caches). A naive "first directory that contains
// mlx.metallib" scan can therefore pair a stale metallib with a fresh .node —
// the kernels then mismatch the compiled C++ and inference produces garbage
// without any error. Selection here is bound to the build that just ran:
//
//   1. Only the tree the napi build used is scanned:
//      `<targetRoot>/<triple>/<profile>/build` (napi always passes
//      `--target`); the plain `<targetRoot>/<profile>/build` layout is a
//      fallback for napi-less flows, never mixed with the primary tree.
//   2. Among candidate `mlx-sys-*` dirs, the one cargo touched most recently
//      wins (`invoked.timestamp` / build-script `output` / metallib mtime).
//   3. The chosen metallib must pass a content gate (minimum size + expected
//      kernel names, including the current pin's NAX kernels on hosts where
//      MLX builds them) or the build fails loudly instead of shipping it.
//
// Stale dirs are deliberately NOT deleted: they are live cargo cache for
// other toolchains/branches, and deleting them forces a full MLX rebuild.
import { readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

export interface MetallibCandidate {
  /** `.../mlx-sys-<hash>/out/lib` — also holds paged_attn.metallib. */
  libDir: string;
  metallibPath: string;
  size: number;
  /** Newest cargo activity observed for this build-script dir (ms). */
  rankMtimeMs: number;
}

/** Smallest healthy mlx.metallib observed is ~154 MB; anything far below is truncated. */
export const MIN_METALLIB_BYTES = 100 * 1024 * 1024;

/** Kernel names present in every healthy mlx.metallib from the vendored MLX. */
export const BASE_KERNEL_MARKERS = ['steel_attention', 'sdpa_vector'] as const;

/**
 * Kernel names introduced by the current MLX pin (e9463bbf): the NAX gen-17
 * family. Absent from the previous pin (a8776b7b), so their absence on a
 * NAX-building host means a stale metallib was selected.
 */
export const NAX_KERNEL_MARKERS = ['affine_qmv_wide', 'steel_gemm_segmented_nax'] as const;

export function hostAppleTriple(arch: string = process.arch): string {
  return arch === 'arm64' ? 'aarch64-apple-darwin' : 'x86_64-apple-darwin';
}

export function profileDirName(opts: { profile?: string | undefined; release?: boolean | undefined }): string {
  return opts.profile ?? (opts.release ? 'release' : 'debug');
}

/** Numeric dotted-version compare: negative if a < b, 0 if equal, positive if a > b. */
export function compareVersions(a: string, b: string): number {
  const pa = a.split('.').map((p) => Number.parseInt(p, 10) || 0);
  const pb = b.split('.').map((p) => Number.parseInt(p, 10) || 0);
  const len = Math.max(pa.length, pb.length);
  for (let i = 0; i < len; i++) {
    const diff = (pa[i] ?? 0) - (pb[i] ?? 0);
    if (diff !== 0) return diff;
  }
  return 0;
}

/**
 * Mirror of the NAX condition in MLX's
 * `mlx/backend/metal/kernels/CMakeLists.txt`: NAX kernels are compiled iff
 * the macOS SDK is >= 26.2 AND the effective `CMAKE_OSX_DEPLOYMENT_TARGET`
 * is >= 26.2, where the deployment target defaults to the build host's
 * macOS version when `MACOSX_DEPLOYMENT_TARGET` is not set.
 */
export function shouldExpectNaxKernels(
  sdkVersion: string,
  hostVersion: string,
  deploymentTargetEnv: string | undefined,
): boolean {
  const effectiveTarget = deploymentTargetEnv && deploymentTargetEnv !== '' ? deploymentTargetEnv : hostVersion;
  return compareVersions(sdkVersion, '26.2') >= 0 && compareVersions(effectiveTarget, '26.2') >= 0;
}

/**
 * Enumerate mlx-sys metallib output dirs for the build that just ran,
 * newest cargo activity first. The `<triple>` tree is authoritative (napi
 * always builds with `--target`); the plain `<profile>` tree is only used
 * when the triple tree has no candidates at all.
 */
export function collectMetallibCandidates(targetRoot: string, triple: string, profile: string): MetallibCandidate[] {
  const roots = [join(targetRoot, triple, profile, 'build'), join(targetRoot, profile, 'build')];
  for (const buildRoot of roots) {
    let entries: string[];
    try {
      entries = readdirSync(buildRoot);
    } catch {
      continue;
    }
    const candidates: MetallibCandidate[] = [];
    for (const dir of entries) {
      if (!dir.startsWith('mlx-sys-')) continue;
      const scriptDir = join(buildRoot, dir);
      const libDir = join(scriptDir, 'out', 'lib');
      const metallibPath = join(libDir, 'mlx.metallib');
      let metallibStat;
      try {
        metallibStat = statSync(metallibPath);
      } catch {
        continue;
      }
      // `invoked.timestamp` is rewritten when cargo (re)runs the build
      // script; the metallib/`output` mtimes cover reused cached outputs.
      let rankMtimeMs = metallibStat.mtimeMs;
      for (const probe of ['invoked.timestamp', 'output']) {
        try {
          rankMtimeMs = Math.max(rankMtimeMs, statSync(join(scriptDir, probe)).mtimeMs);
        } catch {
          // probe file absent — fall back to the mtimes we have
        }
      }
      candidates.push({ libDir, metallibPath, size: metallibStat.size, rankMtimeMs });
    }
    if (candidates.length > 0) {
      return candidates.sort((a, b) => b.rankMtimeMs - a.rankMtimeMs);
    }
  }
  return [];
}

/**
 * Hard gate before the metallib is copied anywhere: a truncated file or a
 * stale-pin kernel inventory must fail the build loudly, not ship to npm.
 */
export function assertMetallibIntegrity(
  metallib: Buffer,
  opts: { path: string; expectNax: boolean; minBytes?: number },
): void {
  const minBytes = opts.minBytes ?? MIN_METALLIB_BYTES;
  if (metallib.byteLength < minBytes) {
    throw new Error(
      `[build.ts metallib gate] ${opts.path} is ${metallib.byteLength} bytes, below the ` +
        `${minBytes}-byte floor of a healthy mlx.metallib — the file is truncated or the ` +
        `build was interrupted. Re-run the native build; if it persists, remove the ` +
        `containing mlx-sys-*/out dir to force a clean MLX kernel build.`,
    );
  }
  const missing = (markers: readonly string[]) => markers.filter((name) => !metallib.includes(name));
  const missingBase = missing(BASE_KERNEL_MARKERS);
  if (missingBase.length > 0) {
    throw new Error(
      `[build.ts metallib gate] ${opts.path} is missing expected kernel(s) ${missingBase.join(', ')} — ` +
        `this is not a healthy MLX kernel library for the vendored pin.`,
    );
  }
  if (opts.expectNax) {
    const missingNax = missing(NAX_KERNEL_MARKERS);
    if (missingNax.length > 0) {
      throw new Error(
        `[build.ts metallib gate] ${opts.path} is missing NAX kernel(s) ${missingNax.join(', ')} ` +
          `although this host builds them (SDK and deployment target >= 26.2). The metallib is ` +
          `stale — most likely from an out-of-date mlx-sys-*/out dir of a previous MLX pin. ` +
          `Re-run the native build; if it persists, remove the stale mlx-sys-* dirs under ` +
          `target/*/release/build/.`,
      );
    }
  }
}
