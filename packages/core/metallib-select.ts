// Deterministic selection of the mlx.metallib produced by the mlx-sys build
// that the just-built native addon actually linked against.
//
// Background: cargo keeps one `mlx-sys-<hash>/out` dir per (target, profile,
// compiler-metadata) fingerprint, and stale hash dirs from earlier toolchains
// or earlier MLX submodule pins survive next to the current one (locally and
// in CI-restored cargo caches). A naive "first directory that contains
// mlx.metallib" scan can therefore pair a stale metallib with a fresh .node —
// the kernels then mismatch the compiled C++ and inference produces garbage
// without any error. Selection is bound to the build that just ran:
//
//   1. AUTHORITATIVE binding: MLX bakes the absolute path of its cmake
//      kernels output into the addon as the `METAL_PATH` compile definition
//      (`mlx/backend/metal/device.cpp` `default_mtllib_path`). That string
//      names the exact `mlx-sys-<hash>/out` dir the addon linked against, so
//      `extractBakedMetallibBinding` reads it out of the freshly built .node
//      and the copy step ships that exact file (with its `out/lib` install
//      copy required to be byte-identical when both exist) — no heuristics,
//      immune to newer same-pin dirs left by other cargo invocations
//      (different feature unification, clippy/test builds, rust-cache
//      restores) that a recency ranking could wrongly prefer.
//   2. Heuristic fallback (only when no METAL_PATH is baked, e.g. a
//      Metal-less addon): scan the tree the napi build used —
//      `<targetRoot>/<triple>/<profile>/build` (napi always passes
//      `--target`), with the plain `<targetRoot>/<profile>/build` layout as
//      a napi-less fallback — and pick the `mlx-sys-*` dir cargo touched
//      most recently (`invoked.timestamp` / build-script `output` /
//      metallib mtime).
//   3. Content gates run on the chosen metallib either way: minimum size +
//      expected kernel names (including the current pin's NAX kernels on
//      hosts where MLX builds them), plus a min-OS stamp check against the
//      intended deployment floor. Failures abort the build loudly instead
//      of shipping a broken pairing.
//
// Stale dirs are deliberately NOT deleted: they are live cargo cache for
// other toolchains/branches, and deleting them forces a full MLX rebuild.
import { readdirSync, readFileSync, statSync, type Stats } from 'node:fs';
import { dirname, join, normalize } from 'node:path';

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

/**
 * Every healthy paged_attn.metallib observed to date (8 samples across
 * debug/release profiles and old/new MLX pins) is exactly 19,490,342 bytes
 * (~19.5 MB) — the kernel set is ours (crates/mlx-paged-attn) and stable.
 * A 4 MiB floor keeps generous headroom for future kernel trimming while
 * still catching a truncated or interrupted write.
 */
export const MIN_PAGED_METALLIB_BYTES = 4 * 1024 * 1024;

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

/**
 * Cargo target-dir resolution for the fallback scan, mirroring cargo's own
 * precedence: `--target-dir` flag > `CARGO_TARGET_DIR` env >
 * `CARGO_BUILD_TARGET_DIR` env (the `build.target-dir` config form) > the
 * repo default. Empty env values behave like unset, matching cargo.
 */
export function resolveTargetRoot(opts: {
  /** `--target-dir` as parsed by the napi CLI (its flag is forwarded to cargo). */
  targetDir: string | undefined;
  env: Record<string, string | undefined>;
  defaultRoot: string;
}): string {
  const nonEmpty = (v: string | undefined): string | undefined => (v !== undefined && v !== '' ? v : undefined);
  return (
    nonEmpty(opts.targetDir) ??
    nonEmpty(opts.env['CARGO_TARGET_DIR']) ??
    nonEmpty(opts.env['CARGO_BUILD_TARGET_DIR']) ??
    opts.defaultRoot
  );
}

/** The exact mlx-sys build a native addon linked against, read from the addon itself. */
export interface MetallibBinding {
  /**
   * The METAL_PATH the addon references at runtime (cmake kernels build-tree
   * file, path normalized). This is the origin file; `installCopyPath` is a
   * cmake-install duplicate of it.
   */
  bakedPath: string;
  /** `.../mlx-sys-<hash>/out` — OUT_DIR of the bound mlx-sys build. */
  outDir: string;
  /** `.../out/lib` — install dir holding copies of both metallibs. */
  libDir: string;
  /** `.../out/lib/mlx.metallib` — cmake-install copy of `bakedPath`. */
  installCopyPath: string;
}

/**
 * MLX compiles `device.cpp` with `-DMETAL_PATH="<kernels build dir>/mlx.metallib"`
 * (see `mlx/backend/metal/CMakeLists.txt`), so every Metal-enabled addon carries
 * the absolute path of the kernels dir it linked against. The trailing
 * `kernels//` double slash comes from CMake joining a dir that already ends in
 * `/`; accept both forms.
 */
const BAKED_METAL_PATH_RE = /^(\/.+\/mlx-sys-[0-9a-f]+\/out)\/build\/mlx\/backend\/metal\/kernels\/{1,2}mlx\.metallib$/;

/** Longest plausible baked path; bounds the NUL-terminated-string expansion. */
const MAX_BAKED_PATH_BYTES = 4096;

/**
 * Extract the baked METAL_PATH from an addon binary. Returns undefined when
 * the binary carries no baked path (Metal not built in); throws when it
 * carries more than one distinct path — that would mean two different MLX
 * builds were linked and no single metallib can be authoritative.
 */
export function extractBakedMetallibBinding(addonBinary: Buffer): MetallibBinding | undefined {
  const needle = Buffer.from('mlx.metallib');
  const bindings = new Map<string, MetallibBinding>();
  let at = addonBinary.indexOf(needle);
  while (at !== -1) {
    // Expand the hit to its enclosing NUL-terminated C string (bounded: a
    // hit inside non-string data must not walk megabytes backwards).
    let start = at;
    const startFloor = Math.max(0, at - MAX_BAKED_PATH_BYTES);
    while (start > startFloor && addonBinary[start - 1] !== 0) start--;
    let end = at + needle.length;
    const endCap = Math.min(addonBinary.length, end + MAX_BAKED_PATH_BYTES);
    while (end < endCap && addonBinary[end] !== 0) end++;
    const candidate = addonBinary.subarray(start, end).toString('utf-8');
    const match = BAKED_METAL_PATH_RE.exec(candidate);
    if (match) {
      const outDir = match[1]!;
      const libDir = join(outDir, 'lib');
      bindings.set(candidate, {
        // normalize() collapses the `kernels//mlx.metallib` double slash
        bakedPath: normalize(candidate),
        outDir,
        libDir,
        installCopyPath: join(libDir, 'mlx.metallib'),
      });
    }
    at = addonBinary.indexOf(needle, at + needle.length);
  }
  if (bindings.size === 0) return undefined;
  if (bindings.size > 1) {
    throw new Error(
      `[build.ts metallib gate] the addon bakes ${bindings.size} distinct METAL_PATH strings — ` +
        `cannot bind a single metallib to it:\n` +
        [...bindings.keys()].map((p) => `  ${p}`).join('\n'),
    );
  }
  return bindings.values().next().value;
}

export interface SelectedMetallib {
  /** `.../mlx-sys-<hash>/out` of the selected build. */
  outDir: string;
  libDir: string;
  metallibPath: string;
  /** 'baked' = authoritative binding from the addon binary; 'scan' = heuristic fallback. */
  source: 'baked' | 'scan';
}

/**
 * Only ENOENT/ENOTDIR mean "the artifact is not there". Anything else
 * (EACCES, EPERM, EISDIR, EIO, ...) means the artifact may well exist but
 * cannot be inspected — treating that as absence would silently skip the
 * byte-identity comparison (or, in the fallback scan, skip the current
 * candidate and select an older one). Every fs probe in this module routes
 * through this policy: absence continues, everything else fails loudly.
 */
function isAbsenceError(err: unknown): boolean {
  const code = (err as NodeJS.ErrnoException | null)?.code;
  return code === 'ENOENT' || code === 'ENOTDIR';
}

function statIfExists(path: string): Stats | undefined {
  try {
    return statSync(path);
  } catch (err) {
    if (isAbsenceError(err)) return undefined;
    throw new Error(`[build.ts metallib gate] cannot stat ${path}: ${(err as Error).message}`);
  }
}

function statSize(path: string): number | undefined {
  return statIfExists(path)?.size;
}

/**
 * Pick the metallib to ship with a freshly built addon. The baked METAL_PATH
 * inside the addon is authoritative: ship exactly the file the addon
 * references. Its `out/lib` cmake-install copy is accepted only when the
 * baked file is gone (e.g. a pruned build tree) — and when BOTH exist they
 * must be byte-identical, otherwise the install copy is stale and the build
 * fails loudly. The mtime-ranked directory scan runs only when the addon
 * carries no baked path at all; a bound build with no metallib left throws
 * instead of falling back — shipping a different build's metallib is exactly
 * the pairing hazard this exists to prevent.
 */
export function selectMetallib(opts: {
  addonBinary: Buffer;
  addonPath: string;
  targetRoot: string;
  triple: string;
  profile: string;
  /**
   * Publish/release mode (set from `MLX_METALLIB_STRICT` in CI). A
   * Metal-enabled darwin addon always bakes its METAL_PATH, so a missing
   * binding on a publish build means Metal was disabled or the addon is
   * broken — fail closed instead of scanning possibly-stale mlx-sys-* dirs
   * for a metallib that may not be this addon's kernels. Local dev leaves it
   * off and keeps the lenient scan.
   */
  strict?: boolean;
  warn?: (msg: string) => void;
}): SelectedMetallib {
  const warn = opts.warn ?? (() => {});
  const binding = extractBakedMetallibBinding(opts.addonBinary);
  if (binding) {
    const bakedSize = statSize(binding.bakedPath);
    const installSize = statSize(binding.installCopyPath);
    if (bakedSize === undefined && installSize === undefined) {
      throw new Error(
        `[build.ts metallib gate] ${opts.addonPath} was linked against the mlx-sys build at ` +
          `${binding.outDir} (baked METAL_PATH: ${binding.bakedPath}), but neither that file nor ` +
          `its install copy ${binding.installCopyPath} exists. That build dir was deleted or ` +
          `moved after linking; re-run the native build so the addon and metallib come out of ` +
          `the same mlx-sys build.`,
      );
    }
    const selected = (metallibPath: string): SelectedMetallib => ({
      outDir: binding.outDir,
      libDir: binding.libDir,
      metallibPath,
      source: 'baked',
    });
    if (bakedSize !== undefined && installSize !== undefined) {
      const identical =
        bakedSize === installSize && readFileSync(binding.bakedPath).equals(readFileSync(binding.installCopyPath));
      if (!identical) {
        throw new Error(
          `[build.ts metallib gate] ${binding.bakedPath} (the file the addon references) and its ` +
            `install copy ${binding.installCopyPath} are not byte-identical — the install copy is ` +
            `stale or partially restored. Re-run the native build; if it persists, remove ` +
            `${binding.outDir} to force a clean MLX kernel build.`,
        );
      }
      return selected(binding.bakedPath);
    }
    if (bakedSize !== undefined) {
      warn(
        `${binding.installCopyPath} is missing; shipping the baked build-tree file ` + `${binding.bakedPath} directly.`,
      );
      return selected(binding.bakedPath);
    }
    warn(
      `${binding.bakedPath} is missing (build tree pruned?); shipping its cmake-install copy ` +
        `${binding.installCopyPath} from the same bound build dir.`,
    );
    return selected(binding.installCopyPath);
  }
  if (opts.strict) {
    throw new Error(
      `[build.ts metallib gate] no baked METAL_PATH in ${opts.addonPath} while MLX_METALLIB_STRICT ` +
        `is set (publish/release build). A Metal-enabled darwin addon always bakes the METAL_PATH ` +
        `of the mlx-sys build it linked against; its absence means Metal was disabled for this ` +
        `build (MLX_DISABLE_METAL) or the addon is broken. Refusing to scan stale mlx-sys-* dirs — ` +
        `a same-pin metallib from an unrelated build passes the size/marker gates yet may not be ` +
        `the kernels this addon links. Build with Metal enabled (never set MLX_DISABLE_METAL for a ` +
        `published artifact), or skip metallib packaging for a CPU-only build.`,
    );
  }
  warn(
    `No baked METAL_PATH found in ${opts.addonPath}; falling back to scanning ` +
      `${opts.targetRoot} for the most recent mlx-sys build.`,
  );
  const candidates = collectMetallibCandidates(opts.targetRoot, opts.triple, opts.profile);
  if (candidates.length === 0) {
    throw new Error(
      `mlx.metallib not found under ${join(opts.targetRoot, opts.triple, opts.profile, 'build')}/mlx-sys-*/out/lib/ ` +
        `(nor the plain ${join(opts.targetRoot, opts.profile, 'build')} layout).`,
    );
  }
  const picked = candidates[0]!;
  if (candidates.length > 1) {
    warn(
      `Multiple mlx-sys metallib dirs found; using the most recently built one:\n` +
        candidates
          .map(
            (c, i) =>
              `  ${i === 0 ? '->' : '  '} ${c.metallibPath} (${c.size} bytes, ${new Date(c.rankMtimeMs).toISOString()})`,
          )
          .join('\n'),
    );
  }
  return { outDir: dirname(picked.libDir), libDir: picked.libDir, metallibPath: picked.metallibPath, source: 'scan' };
}

export interface SelectedPagedMetallib {
  path: string;
  contents: Buffer;
}

function readIfExists(path: string): Buffer | undefined {
  try {
    return readFileSync(path);
  } catch (err) {
    if (isAbsenceError(err)) return undefined;
    throw new Error(`[build.ts metallib gate] cannot read ${path}: ${(err as Error).message}`);
  }
}

/**
 * Resolve paged_attn.metallib inside the selected mlx-sys build dir, under
 * the same contract as the mlx.metallib binding: build.rs writes the origin
 * to the OUT_DIR root and copies it into `out/lib`, so when BOTH exist they
 * must be byte-identical (a mismatch means one of them is stale or
 * truncated — fail loudly, never silently prefer either); a single
 * surviving copy ships with a warning; neither existing throws.
 */
export function selectPagedMetallib(opts: {
  outDir: string;
  libDir: string;
  warn?: (msg: string) => void;
}): SelectedPagedMetallib {
  const warn = opts.warn ?? (() => {});
  const originPath = join(opts.outDir, 'paged_attn.metallib');
  const installCopyPath = join(opts.libDir, 'paged_attn.metallib');
  const origin = readIfExists(originPath);
  const installCopy = readIfExists(installCopyPath);
  if (origin === undefined && installCopy === undefined) {
    throw new Error(
      `paged_attn.metallib not found at ${originPath} nor at ${installCopyPath}. The paged-attention ` +
        `compile path (mlx_paged_dispatch.cpp) loads this metallib via dladdr ` +
        `at runtime; without it, the addon throws on first paged-attention use. ` +
        `Check that mlx-sys/build.rs ran compile_paged_attn_metallib successfully.`,
    );
  }
  if (origin !== undefined && installCopy !== undefined) {
    if (!origin.equals(installCopy)) {
      throw new Error(
        `[build.ts metallib gate] ${originPath} (the file build.rs produced) and its install copy ` +
          `${installCopyPath} are not byte-identical — one of them is stale or truncated. Re-run ` +
          `the native build; if it persists, remove ${opts.outDir} to force a clean build.`,
      );
    }
    return { path: originPath, contents: origin };
  }
  if (origin !== undefined) {
    warn(`${installCopyPath} is missing; shipping the build.rs origin ${originPath} directly.`);
    return { path: originPath, contents: origin };
  }
  warn(`${originPath} is missing; shipping its install copy ${installCopyPath} from the same build dir.`);
  return { path: installCopyPath, contents: installCopy! };
}

/**
 * Hard gate before paged_attn.metallib is copied anywhere, mirroring
 * `assertMetallibIntegrity`: a truncated file or a non-metallib container
 * must fail the build loudly. There is no kernel-name inventory here — the
 * paged-attn kernel set is small and ours — so the gate is the size floor
 * plus the MTLB container magic.
 */
export function assertPagedMetallibIntegrity(metallib: Buffer, opts: { path: string; minBytes?: number }): void {
  const minBytes = opts.minBytes ?? MIN_PAGED_METALLIB_BYTES;
  if (metallib.byteLength < minBytes) {
    throw new Error(
      `[build.ts metallib gate] ${opts.path} is ${metallib.byteLength} bytes, below the ` +
        `${minBytes}-byte floor of a healthy paged_attn.metallib (every observed healthy ` +
        `build is ~19.5 MB) — the file is truncated or the build was interrupted. Re-run ` +
        `the native build; if it persists, remove the containing mlx-sys-*/out dir.`,
    );
  }
  if (metallib.toString('latin1', 0, 4) !== 'MTLB') {
    throw new Error(
      `[build.ts metallib gate] ${opts.path} does not start with the MTLB container magic — ` +
        `this is not a Metal library. Re-run the native build; if it persists, remove the ` +
        `containing mlx-sys-*/out dir.`,
    );
  }
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
 * Mirror of the NAX condition in the vendored MLX's
 * `mlx/backend/metal/kernels/CMakeLists.txt` as configured by
 * `crates/mlx-sys/build.rs`. mlx-sys always passes `-DMLX_METAL_FORCE_NAX=ON`
 * (fork branch `nax-macos-26-0-floor`), which drops upstream's
 * deployment-target >= 26.2 clause, so NAX kernels are compiled iff the macOS
 * SDK is >= 26.2 AND the effective `CMAKE_OSX_DEPLOYMENT_TARGET` is >= 26.0
 * (below 26.0 the Metal language version falls under MSL 4.0 and the gate's
 * `MLX_METAL_VERSION GREATER_EQUAL 400` clause fails). The deployment target
 * defaults to the build host's macOS version when `MACOSX_DEPLOYMENT_TARGET`
 * is not set. The force-built NAX kernels internally compile against the
 * macOS 26.2 tensor-ops ABI while the metallib links (and load-gates) at the
 * floor; runtime dispatch still requires macOS 26.2 — kernel presence and
 * dispatch are deliberately decoupled so one 26.0-floor artifact serves
 * every macOS 26 host.
 */
export function shouldExpectNaxKernels(
  sdkVersion: string,
  hostVersion: string,
  deploymentTargetEnv: string | undefined,
): boolean {
  const effectiveTarget = deploymentTargetEnv && deploymentTargetEnv !== '' ? deploymentTargetEnv : hostVersion;
  return compareVersions(sdkVersion, '26.2') >= 0 && compareVersions(effectiveTarget, '26.0') >= 0;
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
    } catch (err) {
      // A missing tree just means the build used the other layout; anything
      // else (EACCES, ...) would silently hide the current candidates and
      // let an older readable one win — fail loudly instead.
      if (isAbsenceError(err)) continue;
      throw new Error(`[build.ts metallib gate] cannot list ${buildRoot}: ${(err as Error).message}`);
    }
    const candidates: MetallibCandidate[] = [];
    for (const dir of entries) {
      if (!dir.startsWith('mlx-sys-')) continue;
      const scriptDir = join(buildRoot, dir);
      const libDir = join(scriptDir, 'out', 'lib');
      const metallibPath = join(libDir, 'mlx.metallib');
      // Dirs without a metallib are skipped; an uninspectable one throws
      // (via statIfExists) rather than being mistaken for absent.
      const metallibStat = statIfExists(metallibPath);
      if (metallibStat === undefined) continue;
      // `invoked.timestamp` is rewritten when cargo (re)runs the build
      // script; the metallib/`output` mtimes cover reused cached outputs.
      let rankMtimeMs = metallibStat.mtimeMs;
      for (const probe of ['invoked.timestamp', 'output']) {
        // Probe files are optional; when absent the mtimes we have rank the
        // dir. Unreadable probes throw — they would corrupt the ranking.
        const probeStat = statIfExists(join(scriptDir, probe));
        if (probeStat !== undefined) {
          rankMtimeMs = Math.max(rankMtimeMs, probeStat.mtimeMs);
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

/**
 * Read the min-OS stamp from a metallib's MTLB container header. The floor
 * gate only ENFORCES on a fully recognized layout; any deviation returns
 * undefined so a future container revision degrades to a warn+skip, never a
 * hard failure on a good build. Recognized layout — every field validated
 * against `xcrun metal -mmacosx-version-min=15.0/26.0/26.2` scratch builds
 * plus the shipped artifacts (min-OS cross-checked with
 * `xcrun air-vtool -show`):
 *
 *   offset  0: magic `MTLB`
 *   offset  4: u16 LE platform tag, 0x8001 on all macOS outputs
 *   offset  6: u16 LE container version major, 2 (minor at 8 varies with
 *              the tools — 8 for a 15.0 target, 9 for 26.x — not pinned)
 *   offset 10: u8 library type, 0x00 (executable)
 *   offset 11: u8 target-OS tag, 0x81 (macOS)
 *   offset 12: u16 LE min-OS major, offset 14: u16 LE min-OS minor
 *              (`0f00 0000` / `1a00 0000` / `1a00 0200` for 15.0/26.0/26.2)
 */
export function parseMetallibMinOs(metallib: Buffer): string | undefined {
  if (metallib.byteLength < 16) return undefined;
  if (metallib.toString('latin1', 0, 4) !== 'MTLB') return undefined;
  if (metallib.readUInt16LE(4) !== 0x8001) return undefined;
  if (metallib.readUInt16LE(6) !== 2) return undefined;
  if (metallib[10] !== 0x00 || metallib[11] !== 0x81) return undefined;
  const major = metallib.readUInt16LE(12);
  const minor = metallib.readUInt16LE(14);
  // macOS majors run 10 (Yosemite era) through the year-based 26+; anything
  // outside a generous bound means the offset no longer holds a version.
  if (major < 10 || major > 99) return undefined;
  return `${major}.${minor}`;
}

/**
 * Deployment-floor gate: a metallib whose container min-OS stamp is ABOVE
 * the intended floor refuses to load on floor machines, and the kernel-name
 * inventory cannot tell floors apart. A stamp below the floor is harmless
 * (it loads on the floor and older). When the header layout is not
 * recognized the gate is best-effort: local dev warns and skips, but a
 * publish/release build (`strict`, from `MLX_METALLIB_STRICT`) fails closed —
 * shipping a metallib whose floor cannot be verified could publish a package
 * that fails to load on the floor OS.
 */
export function assertMetallibFloor(
  metallib: Buffer,
  opts: { path: string; deploymentFloor: string; strict?: boolean; warn?: (msg: string) => void },
): void {
  const stamped = parseMetallibMinOs(metallib);
  if (stamped === undefined) {
    if (opts.strict) {
      throw new Error(
        `[build.ts metallib gate] ${opts.path}: unrecognized MTLB header layout while ` +
          `MLX_METALLIB_STRICT is set (publish/release build) — the min-OS floor cannot be ` +
          `verified against ${opts.deploymentFloor}, and shipping an unverifiable metallib risks ` +
          `an artifact that fails to load on macOS ${opts.deploymentFloor}. The container header ` +
          `format changed (new toolchain?); update parseMetallibMinOs to recognize the new layout.`,
      );
    }
    opts.warn?.(
      `[build.ts metallib gate] ${opts.path}: unrecognized MTLB header layout; ` +
        `skipping the deployment-floor check (min-OS stamp not verifiable).`,
    );
    return;
  }
  if (compareVersions(stamped, opts.deploymentFloor) > 0) {
    throw new Error(
      `[build.ts metallib gate] ${opts.path} stamps min-OS ${stamped}, above the intended ` +
        `deployment floor ${opts.deploymentFloor} — it would fail to load on macOS ` +
        `${opts.deploymentFloor} hosts. It was linked under a different ` +
        `MACOSX_DEPLOYMENT_TARGET; re-run the native build with the intended floor ` +
        `(if it persists, purge the containing mlx-sys-*/out dir).`,
    );
  }
}
