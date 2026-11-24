// Deterministic selection of the Windows MLX runtime root (`openblas.dll` and
// `include/{cccl,cute,cutlass}`) produced by the exact `mlx-sys` build unit
// that the freshly built native addon linked against.
//
// Background:
//   1. AUTHORITATIVE binding: `crates/mlx-sys/build.rs` writes the absolute
//      MLX root (prebuilt unpack dir or `out/mlx-cmake`) to
//      `<OUT_DIR>/mlx-runtime-root.txt` and compiles the bridge with
//      `-DMLX_NODE_WINDOWS_BUILD_UNIT=unit_<hex>`. `mlx_stream_cuda.cpp` keeps
//      the NUL-terminated marker `MLX_NODE_WINDOWS_BUILD_UNIT=unit_<hex>` live
//      in the `.node` binary. `selectWindowsMlxRoot` reads that unit hash,
//      locates that exact `mlx-sys` build unit's `mlx-runtime-root.txt`, and
//      validates its contents — failing closed if anything is missing.
//   2. Fallback scan (only when no baked marker exists, e.g. an older addon):
//      scans `<targetRoot>/<triple>/<profile>/build` then
//      `<targetRoot>/<profile>/build` for `mlx-sys-*` (pre-Cargo 1.100) and
//      `mlx-sys/*` (Cargo 1.100+). If an `outDir` carries
//      `mlx-runtime-root.txt`, ONLY its recorded path is used (never a stale
//      `out/mlx-cmake` in the same `outDir`); otherwise legacy
//      `out/mlx-cmake/openblas.dll` is accepted. Candidates are ranked by
//      mtime descending and the winner is validated before use.
import { readdirSync, readFileSync, statSync, type Stats } from 'node:fs';
import { join } from 'node:path';

export const WINDOWS_MLX_INCLUDE_DIRS = ['cccl', 'cute', 'cutlass'] as const;
export const WINDOWS_RUNTIME_ROOT_FILE = 'mlx-runtime-root.txt';

const WINDOWS_BUILD_UNIT_PREFIX = 'MLX_NODE_WINDOWS_BUILD_UNIT=';
const WINDOWS_BUILD_UNIT_RE = /^MLX_NODE_WINDOWS_BUILD_UNIT=unit_([0-9a-f]+)$/;
const MAX_MARKER_BYTES = 256;

export interface SelectedWindowsMlxRoot {
  mlxRoot: string;
  source: 'build-unit' | 'fallback-scan';
  unitHash?: string;
}

export interface SelectWindowsMlxRootOptions {
  addonBinary: Buffer;
  addonPath: string;
  targetRoot: string;
  triple: string;
  profile: string;
  warn?: (msg: string) => void;
}

/** Normalize CRLF line endings to LF for cross-platform declaration comparisons. */
export function normalizeLineEndings(text: string): string {
  return text.replaceAll('\r\n', '\n');
}

/**
 * Derive the matching `npm/<triple>/` directory from napi-rs's actual output.
 * Does not use `process.arch`: e.g. an x64 Node under emulation can drive an
 * aarch64 Rust build.
 */
export function nativeAddonTriple(actualName: string, platform: NodeJS.Platform = process.platform): string {
  const match = /^mlx-core\.(.+)\.node$/.exec(actualName);
  if (match == null) {
    throw new Error(`[build.ts] unrecognized native addon output: ${actualName}`);
  }
  const triple = match[1]!;
  if (platform === 'darwin' && !triple.startsWith('darwin-')) {
    throw new Error(`[build.ts] expected a Darwin addon, got ${actualName}`);
  }
  if (platform === 'linux' && !triple.startsWith('linux-')) {
    throw new Error(`[build.ts] expected a Linux addon, got ${actualName}`);
  }
  if (platform === 'win32' && !triple.startsWith('win32-')) {
    throw new Error(`[build.ts] expected a Windows addon, got ${actualName}`);
  }
  return triple;
}

/**
 * Extract the baked `MLX_NODE_WINDOWS_BUILD_UNIT=unit_<hex>` hash from a built
 * Windows `.node` addon via a bounded NUL-terminated string scan.
 *
 * Returns `undefined` when no marker is present; throws if more than one
 * distinct unit hash is found (indicating conflicting `mlx-sys` builds).
 */
export function extractWindowsBuildUnitHash(addonBinary: Buffer): string | undefined {
  const prefixBytes = Buffer.from(WINDOWS_BUILD_UNIT_PREFIX, 'ascii');
  const hashes = new Set<string>();
  let at = addonBinary.indexOf(prefixBytes);
  while (at !== -1) {
    let start = at;
    const startFloor = Math.max(0, at - MAX_MARKER_BYTES);
    while (start > startFloor && addonBinary[start - 1] !== 0) start--;

    let end = at + prefixBytes.length;
    const endCap = Math.min(addonBinary.length, end + MAX_MARKER_BYTES);
    while (end < endCap && addonBinary[end] !== 0) end++;

    const boundedStart = start === 0 || addonBinary[start - 1] === 0;
    const boundedEnd = end < addonBinary.length && addonBinary[end] === 0;
    if (boundedStart && boundedEnd) {
      const candidate = addonBinary.subarray(start, end).toString('utf-8');
      const match = WINDOWS_BUILD_UNIT_RE.exec(candidate);
      if (match !== null) {
        hashes.add(match[1]!);
      }
    }
    at = addonBinary.indexOf(prefixBytes, at + prefixBytes.length);
  }
  if (hashes.size === 0) return undefined;
  if (hashes.size > 1) {
    throw new Error(
      `[build.ts windows runtime] addon bakes ${hashes.size} distinct MLX_NODE_WINDOWS_BUILD_UNIT ` +
        `hashes (${[...hashes].sort().join(', ')}) — cannot bind a single mlx-sys build unit.`,
    );
  }
  return [...hashes][0];
}

function isAbsenceError(err: unknown): boolean {
  const code = (err as NodeJS.ErrnoException | null)?.code;
  return code === 'ENOENT' || code === 'ENOTDIR';
}

function statIfExists(path: string): Stats | undefined {
  try {
    return statSync(path);
  } catch (err) {
    if (isAbsenceError(err)) return undefined;
    throw new Error(`[build.ts windows runtime] cannot stat ${path}: ${(err as Error).message}`);
  }
}

function readTextIfExists(path: string): string | undefined {
  try {
    return readFileSync(path, 'utf-8');
  } catch (err) {
    if (isAbsenceError(err)) return undefined;
    throw new Error(`[build.ts windows runtime] cannot read ${path}: ${(err as Error).message}`);
  }
}

/**
 * Validate that a Windows MLX runtime root carries a non-empty `openblas.dll`
 * file and the NVRTC header directories `include/{cccl,cute,cutlass}`.
 */
export function validateWindowsMlxRoot(mlxRoot: string): void {
  const openblasPath = join(mlxRoot, 'openblas.dll');
  const openblasStat = statIfExists(openblasPath);
  if (openblasStat === undefined) {
    throw new Error(`[build.ts windows runtime] MLX root ${mlxRoot} is missing openblas.dll (${openblasPath}).`);
  }
  if (!openblasStat.isFile() || openblasStat.size === 0) {
    throw new Error(
      `[build.ts windows runtime] ${openblasPath} must be a non-empty file (got ${openblasStat.size} bytes).`,
    );
  }
  for (const includeDir of WINDOWS_MLX_INCLUDE_DIRS) {
    const dirPath = join(mlxRoot, 'include', includeDir);
    const dirStat = statIfExists(dirPath);
    if (dirStat === undefined) {
      throw new Error(`[build.ts windows runtime] MLX root ${mlxRoot} is missing include/${includeDir} (${dirPath}).`);
    }
    if (!dirStat.isDirectory()) {
      throw new Error(`[build.ts windows runtime] expected include/${includeDir} at ${dirPath} to be a directory.`);
    }
  }
}

interface FallbackCandidate {
  mlxRoot: string;
  outDir: string;
  rankMtimeMs: number;
  rootFile?: string;
}

/**
 * Resolve the Windows MLX runtime root for a freshly built `.node` addon.
 */
export function selectWindowsMlxRoot(opts: SelectWindowsMlxRootOptions): SelectedWindowsMlxRoot {
  const warn = opts.warn ?? (() => {});
  const buildRoots = [
    join(opts.targetRoot, opts.triple, opts.profile, 'build'),
    join(opts.targetRoot, opts.profile, 'build'),
  ];

  const unitHash = extractWindowsBuildUnitHash(opts.addonBinary);
  if (unitHash !== undefined) {
    const checkedPaths: string[] = [];
    for (const buildRoot of buildRoots) {
      const unitOutDirs = [join(buildRoot, `mlx-sys-${unitHash}`, 'out'), join(buildRoot, 'mlx-sys', unitHash, 'out')];
      const matches: { rootFile: string; mtimeMs: number; text: string }[] = [];
      for (const outDir of unitOutDirs) {
        const rootFile = join(outDir, WINDOWS_RUNTIME_ROOT_FILE);
        checkedPaths.push(rootFile);
        const rootStat = statIfExists(rootFile);
        if (rootStat === undefined) continue;
        const text = readTextIfExists(rootFile);
        if (text === undefined) continue;
        matches.push({ rootFile, mtimeMs: rootStat.mtimeMs, text });
      }
      if (matches.length > 0) {
        matches.sort((a, b) => b.mtimeMs - a.mtimeMs);
        const picked = matches[0]!;
        const mlxRoot = picked.text.trim();
        if (mlxRoot === '') {
          throw new Error(
            `[build.ts windows runtime] bound mlx-sys build unit ${unitHash} (${opts.addonPath}) ` +
              `wrote an empty ${picked.rootFile}; re-run the native build.`,
          );
        }
        validateWindowsMlxRoot(mlxRoot);
        return { mlxRoot, source: 'build-unit', unitHash };
      }
    }
    throw new Error(
      `[build.ts windows runtime] ${opts.addonPath} is bound to mlx-sys build unit ${unitHash}, ` +
        `but ${WINDOWS_RUNTIME_ROOT_FILE} was not found at any expected path:\n` +
        checkedPaths.map((p) => `  ${p}`).join('\n'),
    );
  }

  warn(
    `No baked MLX_NODE_WINDOWS_BUILD_UNIT marker found in ${opts.addonPath}; ` +
      `falling back to scanning ${opts.targetRoot} for the most recent mlx-sys build.`,
  );

  for (const buildRoot of buildRoots) {
    let entries: string[];
    try {
      entries = readdirSync(buildRoot);
    } catch (err) {
      if (isAbsenceError(err)) continue;
      throw new Error(`[build.ts windows runtime] cannot list ${buildRoot}: ${(err as Error).message}`);
    }

    const scriptDirs: { outDir: string; probes: string[] }[] = [];
    for (const entry of entries) {
      if (entry.startsWith('mlx-sys-')) {
        const scriptDir = join(buildRoot, entry);
        scriptDirs.push({
          outDir: join(scriptDir, 'out'),
          probes: [join(scriptDir, 'invoked.timestamp'), join(scriptDir, 'output')],
        });
      } else if (entry === 'mlx-sys') {
        const pkgDir = join(buildRoot, entry);
        let units: string[];
        try {
          units = readdirSync(pkgDir);
        } catch (err) {
          if (isAbsenceError(err)) continue;
          throw new Error(`[build.ts windows runtime] cannot list ${pkgDir}: ${(err as Error).message}`);
        }
        for (const unit of units) {
          const unitDir = join(pkgDir, unit);
          scriptDirs.push({
            outDir: join(unitDir, 'out'),
            probes: [join(unitDir, 'run', 'invoked.timestamp'), join(unitDir, 'run', 'stdout')],
          });
        }
      }
    }

    const candidates: FallbackCandidate[] = [];
    for (const { outDir, probes } of scriptDirs) {
      let scriptMtimeMs = 0;
      for (const probe of probes) {
        const probeStat = statIfExists(probe);
        if (probeStat !== undefined) {
          scriptMtimeMs = Math.max(scriptMtimeMs, probeStat.mtimeMs);
        }
      }

      const rootFile = join(outDir, WINDOWS_RUNTIME_ROOT_FILE);
      const rootStat = statIfExists(rootFile);
      if (rootStat !== undefined) {
        // When mlx-runtime-root.txt exists in this outDir, use ONLY its
        // recorded path — never fall back to a stale out/mlx-cmake beside it.
        const text = readTextIfExists(rootFile);
        if (text === undefined) continue;
        candidates.push({
          mlxRoot: text.trim(),
          outDir,
          rankMtimeMs: Math.max(scriptMtimeMs, rootStat.mtimeMs),
          rootFile,
        });
        continue;
      }

      const legacyRoot = join(outDir, 'mlx-cmake');
      const blasStat = statIfExists(join(legacyRoot, 'openblas.dll'));
      if (blasStat !== undefined && blasStat.isFile()) {
        candidates.push({
          mlxRoot: legacyRoot,
          outDir,
          rankMtimeMs: Math.max(scriptMtimeMs, blasStat.mtimeMs),
        });
      }
    }

    if (candidates.length > 0) {
      candidates.sort((a, b) => b.rankMtimeMs - a.rankMtimeMs);
      const winner = candidates[0]!;
      if (candidates.length > 1) {
        warn(
          `Multiple Windows mlx-sys runtime candidates found; using the most recent:\n` +
            candidates
              .map(
                (c, i) =>
                  `  ${i === 0 ? '->' : '  '} ${c.mlxRoot || '<empty>'} (from ${c.outDir}, ${new Date(c.rankMtimeMs).toISOString()})`,
              )
              .join('\n'),
        );
      }
      if (winner.mlxRoot === '') {
        throw new Error(`[build.ts windows runtime] ${winner.rootFile ?? winner.outDir} recorded an empty MLX root.`);
      }
      validateWindowsMlxRoot(winner.mlxRoot);
      return { mlxRoot: winner.mlxRoot, source: 'fallback-scan' };
    }
  }

  throw new Error(
    `[build.ts windows runtime] cannot find a Windows MLX runtime root under ` + `${buildRoots.join(' or ')}.`,
  );
}
