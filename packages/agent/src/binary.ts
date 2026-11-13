/**
 * Where the `mlx-agent` binary lives, without downloading anything. Addon-free and synchronous: the CLI launcher and
 * the Control Panel both resolve the binary through this module.
 */

import { existsSync, readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

/** Env override for a locally built or externally installed agent binary. */
export const MLX_AGENT_BIN_ENV = 'MLX_AGENT_BIN';
/** Env override to pin a specific release tag instead of resolving latest. */
export const MLX_AGENT_TAG_ENV = 'MLX_AGENT_TAG';
/**
 * Minimum release accepted: a cached binary stamped below this is treated as stale and re-downloaded, and it is the
 * last-resort tag when the `latest` pointer is unreachable.
 */
export const MLX_AGENT_MIN_RELEASE = 'mlx-agent-v0.3.3';

const TAG_PREFIX = 'mlx-agent-v';

/**
 * The binary's platform filename (CreateProcessW needs `.exe`). With a release `tag` this is the versioned managed
 * name (`<tag>` already carries the `mlx-agent-` prefix); without one, the unversioned base name.
 */
export function agentBinName(tag?: string): string {
  const base = tag ?? 'mlx-agent';
  return process.platform === 'win32' ? `${base}.exe` : base;
}

/** Base path of the managed cache, the only location `mlx agent update` writes. */
export function managedAgentBinPath(homeDir: string = homedir()): string {
  return join(homeDir, '.mlx-node', 'bin', agentBinName());
}

export interface AgentInstallProbe {
  /**
   * What owns the resolved path: a user pin the updater must not touch (`env`/`vendored`), the signed app's bundle
   * (`bundled`), the managed download cache, or nothing found yet (`none`).
   */
  source: 'env' | 'vendored' | 'bundled' | 'managed' | 'none';
  /** The binary `mlx agent` would run; undefined when `source === 'none'`. */
  path?: string;
  /** Release tag stamped beside the managed cache, the bundled tag, and the newest of the pair (bundled wins ties). */
  managedTag?: string;
  bundledTag?: string;
  effectiveTag?: string;
}

/** A `semver` version: three numbers and an optional pre-release list. */
interface TagVersion {
  core: [number, number, number];
  pre: string[];
}

function isDigit(code: number): boolean {
  return code >= 48 && code <= 57;
}

/** A semver numeric part: digits, no leading zero, a safe integer. */
function numericPart(text: string): number | null {
  if (text === '' || (text.length > 1 && text.charCodeAt(0) === 48)) return null;
  for (let i = 0; i < text.length; i++) if (!isDigit(text.charCodeAt(i))) return null;
  const value = Number(text);
  return Number.isSafeInteger(value) ? value : null;
}

function isIdentifierChar(code: number): boolean {
  return isDigit(code) || (code >= 65 && code <= 90) || (code >= 97 && code <= 122) || code === 45;
}

/** `X.Y.Z` or `X.Y.Z-pre.release` in canonical semver form (no build part, no `v`, no spaces), else null. */
function parseVersion(text: string): TagVersion | null {
  const dash = text.indexOf('-');
  const core = dash === -1 ? text : text.slice(0, dash);
  const parts = core.split('.');
  if (parts.length !== 3) return null;
  const nums: number[] = [];
  for (const part of parts) {
    const n = numericPart(part);
    if (n === null) return null;
    nums.push(n);
  }
  const pre: string[] = [];
  if (dash !== -1) {
    for (const id of text.slice(dash + 1).split('.')) {
      if (id === '') return null;
      for (let i = 0; i < id.length; i++) if (!isIdentifierChar(id.charCodeAt(i))) return null;
      let allDigits = true;
      for (let i = 0; i < id.length; i++) if (!isDigit(id.charCodeAt(i))) allDigits = false;
      if (allDigits && numericPart(id) === null) return null;
      pre.push(id);
    }
  }
  return { core: [nums[0]!, nums[1]!, nums[2]!], pre };
}

/** `mlx-agent-v0.2.0` → its version when it is a canonical semver tag, else null. */
function tagVersion(tag: string): TagVersion | null {
  return tag.startsWith(TAG_PREFIX) ? parseVersion(tag.slice(TAG_PREFIX.length)) : null;
}

/** Whether `tag` is a well-formed `mlx-agent-vX.Y.Z[-pre]` release tag. */
export function isReleaseTag(tag: string): boolean {
  return tagVersion(tag) !== null;
}

function compareIdentifiers(a: string, b: string): number {
  const na = numericPart(a);
  const nb = numericPart(b);
  if (na !== null && nb !== null) return na === nb ? 0 : na < nb ? -1 : 1;
  if (na !== null) return -1;
  if (nb !== null) return 1;
  return a === b ? 0 : a < b ? -1 : 1;
}

/** Semver 2.0 precedence. */
function compareVersions(a: TagVersion, b: TagVersion): number {
  for (let i = 0; i < 3; i++) if (a.core[i] !== b.core[i]) return a.core[i]! < b.core[i]! ? -1 : 1;
  if (a.pre.length === 0 || b.pre.length === 0) {
    if (a.pre.length === b.pre.length) return 0;
    return a.pre.length === 0 ? 1 : -1;
  }
  const n = Math.min(a.pre.length, b.pre.length);
  for (let i = 0; i < n; i++) {
    const c = compareIdentifiers(a.pre[i]!, b.pre[i]!);
    if (c !== 0) return c;
  }
  return a.pre.length === b.pre.length ? 0 : a.pre.length < b.pre.length ? -1 : 1;
}

/** Compare `mlx-agent-vX.Y.Z` tags semantically. A malformed tag sorts below any valid one; two malformed tags are equal. */
export function compareReleaseTags(a: string, b: string): number {
  const va = tagVersion(a);
  const vb = tagVersion(b);
  if (va && vb) return compareVersions(va, vb);
  if (va) return 1;
  if (vb) return -1;
  return 0;
}

/**
 * A valid `MLX_AGENT_TAG` pin selects exactly that release. An invalid pin throws wherever it is consulted, so a stale
 * cache can never mask a misspelled pin.
 */
export function releasePin(env: NodeJS.ProcessEnv = process.env): string | undefined {
  const pinned = env[MLX_AGENT_TAG_ENV];
  if (!pinned) return undefined;
  if (!isReleaseTag(pinned) || compareReleaseTags(pinned, MLX_AGENT_MIN_RELEASE) < 0) {
    throw new Error(
      `${MLX_AGENT_TAG_ENV}=${pinned} is not a supported release tag (expected mlx-agent-vX.Y.Z at or above ${MLX_AGENT_MIN_RELEASE}).`,
    );
  }
  return pinned;
}

/** Tag marker written beside a downloaded/bundled binary, e.g. `mlx-agent-v0.2.0`. */
export function readAgentTag(bin: string): string | undefined {
  try {
    const tag = readFileSync(`${bin}.tag`, 'utf-8').trim();
    return tag.length > 0 ? tag : undefined;
  } catch {
    return undefined;
  }
}

/**
 * The binary bundled in the packaged desktop app: walk up to the `Contents/Resources/app` anchor; the release binary
 * sits at `Contents/Resources/native/mlx-agent`.
 */
export function findBundledAppBinary(start: string = fileURLToPath(import.meta.url)): string | undefined {
  let dir = dirname(start);
  for (let i = 0; i < 12; i++) {
    if (dir.endsWith(join('Contents', 'Resources', 'app'))) {
      const candidate = join(dir, '..', 'native', agentBinName());
      return existsSync(candidate) ? candidate : undefined;
    }
    const parent = dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return undefined;
}

/** The repository root (the directory holding `grok-build/` after the bootstrap clone), or undefined outside a checkout. */
export function findRepoRoot(start: string = fileURLToPath(import.meta.url)): string | undefined {
  let dir = dirname(start);
  for (let i = 0; i < 8; i++) {
    if (existsSync(join(dir, 'grok-build', 'Cargo.toml'))) return dir;
    const parent = dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return undefined;
}

/**
 * Filesystem probe, in order: `$MLX_AGENT_BIN` → repo build → `$MLX_AGENT_TAG` pin → bundled → managed cache.
 * `leasePinned` runs on a pinned managed cache hit: the caller protects it from a concurrent update sweep.
 */
export function probeAgentInstall(
  repoRoot = findRepoRoot(),
  env: NodeJS.ProcessEnv = process.env,
  homeDir: string = homedir(),
  bundledBin?: string,
  leasePinned?: (bin: string) => void,
): AgentInstallProbe {
  const managed = managedAgentBinPath(homeDir);
  const stamped = readAgentTag(managed);
  const versioned = stamped !== undefined ? join(dirname(managed), agentBinName(stamped)) : undefined;
  // The unversioned base is a fallback only for installs that predate pointers: a pointer whose versioned binary is
  // gone is a broken cache, and stamping the leftover legacy binary would run the wrong release.
  const managedBin =
    versioned !== undefined && existsSync(versioned)
      ? versioned
      : stamped === undefined && existsSync(managed)
        ? managed
        : undefined;
  // An orphaned pointer left by a failed install must not suppress the next download.
  const managedTag = managedBin !== undefined ? stamped : undefined;

  const envBin = env[MLX_AGENT_BIN_ENV];
  if (envBin) {
    const path = resolve(envBin);
    if (existsSync(path)) return { source: 'env', path, managedTag };
    throw new Error(`${MLX_AGENT_BIN_ENV}=${envBin} does not exist`);
  }

  const vendored = repoRoot ? join(repoRoot, 'grok-build', 'target', 'release', agentBinName()) : undefined;
  if (vendored && existsSync(vendored)) return { source: 'vendored', path: vendored, managedTag };

  const bundled = bundledBin ?? findBundledAppBinary();
  const bundledTag = bundled ? readAgentTag(bundled) : undefined;

  const pin = releasePin(env);
  if (pin !== undefined) {
    const pinnedBin = join(dirname(managed), agentBinName(pin));
    if (existsSync(pinnedBin)) {
      leasePinned?.(pinnedBin);
      return { source: 'managed', path: pinnedBin, managedTag, bundledTag, effectiveTag: pin };
    }
    // A pin naming the bundled release is satisfied by the bundle: an offline packaged app must not need a download.
    if (bundled !== undefined && bundledTag === pin) {
      return { source: 'bundled', path: bundled, managedTag, bundledTag, effectiveTag: pin };
    }
    return { source: 'none', managedTag, bundledTag };
  }

  // The newer release tag wins between the bundle and the cache; an unmarked cache never beats a stamped bundle.
  if (bundled) {
    const cacheWins = managedTag !== undefined && compareReleaseTags(managedTag, bundledTag ?? '') > 0;
    return {
      source: cacheWins ? 'managed' : 'bundled',
      path: cacheWins ? managedBin : bundled,
      managedTag,
      bundledTag,
      effectiveTag: cacheWins ? managedTag : bundledTag,
    };
  }

  // An unmarked or below-minimum cached binary is stale.
  if (managedTag !== undefined && compareReleaseTags(managedTag, MLX_AGENT_MIN_RELEASE) >= 0) {
    return { source: 'managed', path: managedBin, managedTag, effectiveTag: managedTag };
  }
  return { source: 'none', managedTag };
}

/** The release tag a versioned install path encodes (`…/mlx-agent-v1.2.3[.exe]`). */
export function agentTagFromBinPath(path: string): string | undefined {
  const name = basename(path);
  const tag = name.endsWith('.exe') ? name.slice(0, -'.exe'.length) : name;
  return isReleaseTag(tag) ? tag : undefined;
}
