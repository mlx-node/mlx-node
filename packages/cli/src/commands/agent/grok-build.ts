/**
 * Grok Build launcher for `mlx agent` / `mlx delegate`.
 *
 * The forked Grok Build binary (`mlx-agent`, built from the `mlx` branch of
 * github.com/mlx-node/grok-build) is a standalone Rust TUI. It does not embed
 * MLX — it talks to a local inference host over HTTP, exactly like
 * `mlx serve`. Boot contract:
 *
 *   mlx agent ──spawn──> mlx serve --port 0 --auth-token <token>
 *              ──spawn──> mlx-agent  (env: GROK_MODELS_BASE_URL=<serve url>,
 *                                     XAI_API_KEY=<token>)
 *
 * The binary's own defaults already point at 127.0.0.1:8080/v1 — the env vars
 * just pin the ACTUAL ephemeral port of the server we spawned. The auth token
 * travels as the sampling bearer (`XAI_API_KEY` feeds BYOK credential
 * resolution and the /v1/models prefetch), and the server accepts
 * `authorization: Bearer <token>`.
 *
 * Both children are killed when `mlx agent` exits; neither inherits stdio
 * except the agent binary, which takes over the terminal.
 */

import { spawn, spawnSync, type ChildProcess, type SpawnOptions } from 'node:child_process';
import { createHash, randomBytes } from 'node:crypto';
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, utimesSync, writeFileSync, type Stats } from 'node:fs';
import { connect, createServer, type Server } from 'node:net';
import { homedir, tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';

import semver from 'semver';

/** Env override for a locally built or externally installed agent binary. */
export const MLX_AGENT_BIN_ENV = 'MLX_AGENT_BIN';
/** Env override to pin a specific release tag instead of resolving latest. */
export const MLX_AGENT_TAG_ENV = 'MLX_AGENT_TAG';
/** Release repo the prebuilt binary downloads from. */
export const MLX_AGENT_RELEASE_REPO = 'mlx-node/grok-build';
/**
 * Minimum release the CLI will accept: a cached binary stamped below this is
 * treated as stale and re-downloaded, and it is the last-resort tag when the
 * `latest` pointer is unreachable. Bump only when a release is REQUIRED —
 * resolving `latest` handles ordinary upgrades with no code change here.
 */
export const MLX_AGENT_MIN_RELEASE = 'mlx-agent-v0.2.0';

/** Options for {@link launchGrokAgent}. */
export interface GrokAgentLaunchOptions {
  /** Args forwarded to the agent binary verbatim (already default-injected). */
  argv: string[];
  /** Models directory for the spawned `mlx serve` (host resolves env/config itself when omitted). */
  modelsDir?: string;
  /**
   * Whether to spawn `mlx serve` before the agent. The CALLER decides from
   * its value-aware argv scan — the launcher must never rescan `argv`
   * itself: a `--help`-looking token inside another option's value slot
   * (`--system-prompt-override --help`) is data, not a one-shot.
   */
  needsServer?: boolean;
  /** Test seam: custom spawn for the agent binary. */
  spawnAgent?: (bin: string, argv: string[], opts: SpawnOptions) => ChildProcess;
  /** Test seam: custom spawn for `mlx serve`. */
  spawnServe?: (argv: string[], opts: SpawnOptions) => ChildProcess;
  /** Test seam: how long to wait for `GET /health` before failing (default 30s). */
  serveReadyTimeoutMs?: number;
  /** Test seam: serve URL already running — skip spawning `mlx serve`. */
  serveUrl?: string;
  /** Test seam: serve auth token to use with `serveUrl`. */
  serveToken?: string;
}

/**
 * The binary's platform filename — CreateProcessW needs the `.exe` suffix.
 * With a release `tag` this is the versioned managed name (`<tag>` already
 * carries the `mlx-agent-` prefix, so the filename IS the tag); without one,
 * the legacy unversioned/base name.
 */
function agentBinName(tag?: string): string {
  const base = tag ?? 'mlx-agent';
  return process.platform === 'win32' ? `${base}.exe` : base;
}

/**
 * Base path of the managed cache — the only location `mlx agent update`
 * writes. Its `.tag` sibling is the pointer naming the release tag whose
 * versioned binary (`bin/<tag>`) is live; legacy installs keep the binary at
 * the base name itself.
 */
export function managedAgentBinPath(homeDir: string = homedir()): string {
  return join(homeDir, '.mlx-node', 'bin', agentBinName());
}

export interface AgentInstallProbe {
  /**
   * What owns the resolved path: a user pin the updater must not touch
   * (`env`/`vendored`), the signed app's bundle (`bundled` — still shadowed
   * by a NEWER managed download, since the bundle is read-only), the managed
   * download cache, or nothing found yet (`none`).
   */
  source: 'env' | 'vendored' | 'bundled' | 'managed' | 'none';
  /** The binary `mlx agent` would run; undefined when `source === 'none'`. */
  path?: string;
  /**
   * Release tag stamped beside the managed cache and the newest tag among
   * the managed+bundled pair — the "current version" an update compares
   * against. Bundled wins ties, mirroring resolveAgentBinary.
   */
  managedTag?: string;
  bundledTag?: string;
  effectiveTag?: string;
}

/**
 * Filesystem probe behind {@link resolveAgentBinary}, minus the download.
 * Everything is sync by contract: this runs inside the CLI utilityProcess,
 * never on the Electron main thread.
 */
export function probeAgentInstall(
  repoRoot = findRepoRoot(),
  env = process.env,
  homeDir: string = homedir(),
  bundledBin?: string,
): AgentInstallProbe {
  const managed = managedAgentBinPath(homeDir);
  // The pointer names the release whose versioned binary is live; a
  // versioned miss or a legacy install falls back to the unversioned base.
  const stamped = readAgentTag(managed);
  const versioned = stamped !== undefined ? join(dirname(managed), agentBinName(stamped)) : undefined;
  // The unversioned base is a fallback ONLY for installs that predate
  // pointers: when a pointer names a release whose versioned binary is
  // gone, that is a broken cache — stamping the leftover legacy binary
  // with the orphaned tag would both run the wrong release and suppress
  // the update check that could repair it.
  const managedBin =
    versioned !== undefined && existsSync(versioned)
      ? versioned
      : stamped === undefined && existsSync(managed)
        ? managed
        : undefined;
  // The tag counts only with a binary on disk — an orphaned pointer left by
  // a failed install must not suppress the next download.
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

  // An MLX_AGENT_TAG pin selects exactly that managed release — newer or
  // older than the pointer or bundle, the pin wins (developer env/vendored
  // pins still shadow it). A pin neither the cache nor the bundle can
  // satisfy resolves as 'none' so the launcher downloads that exact
  // release instead of silently running whatever is cached.
  const pin = releasePin(env);
  if (pin !== undefined) {
    const pinnedBin = join(dirname(managed), agentBinName(pin));
    if (existsSync(pinnedBin)) {
      // A cached pin still needs a lease: the launcher only spawns the
      // returned path after serve startup and health checks, and an
      // unpinned sweep in ANOTHER process could delete it in that window
      // if the original installer's lease died with it. This write is
      // best-effort — it lands outside the install lock, so a sweep that
      // already snapshotted the directory can still miss it. The launch
      // path (resolveAgentBinary) re-leases under the lock and re-verifies
      // the binary survived; the marker here only covers direct probe
      // callers such as the evals harness.
      writeLeaseMarker(pinnedBin);
      return { source: 'managed', path: pinnedBin, managedTag, bundledTag, effectiveTag: pin };
    }
    // A pin naming the bundled release is already satisfied: the bundle
    // ships that exact binary, so a managed-cache miss must not force a
    // download — a packaged app launched offline would fail for nothing.
    if (bundled !== undefined && bundledTag === pin) {
      return { source: 'bundled', path: bundled, managedTag, bundledTag, effectiveTag: pin };
    }
    return { source: 'none', managedTag, bundledTag };
  }

  // Bundled-vs-cache: the newer release tag wins. A freshly `mlx agent
  // update`d download in ~/.mlx-node/bin must not be shadowed by the older
  // bundled binary, and an unmarked cache never beats a stamped bundle.
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

  // The cached binary is valid only while it satisfies this CLI's minimum
  // release: its marker records which tag produced it. An unmarked binary is
  // treated as stale — cheaper than trusting whatever was downloaded before
  // markers existed. No lease here: callers that RETAIN a managed path
  // (resolveAgentBinary, the evals harness) re-lease under the install
  // lock — the probe itself only reports.
  if (managedTag !== undefined && compareReleaseTags(managedTag, MLX_AGENT_MIN_RELEASE) >= 0) {
    return { source: 'managed', path: managedBin, managedTag, effectiveTag: managedTag };
  }
  return { source: 'none', managedTag };
}

/**
 * Locate the agent binary, in priority order:
 *   1. `$MLX_AGENT_BIN` — developer / vendored-build override.
 *   2. `<repo>/grok-build/target/release/mlx-agent` — `yarn build:agent` output.
 *   3. `$MLX_AGENT_TAG` pin — exactly that release, ahead of bundled/latest:
 *      the managed cache hit, else the bundled binary when its stamped tag
 *      matches the pin, else a download into the cache.
 *   4. `…/Contents/Resources/native/mlx-agent` — bundled inside the packaged app.
 *   5. `~/.mlx-node/bin/<tag>` — the downloaded prebuilt the `.tag` pointer
 *      names (legacy: unversioned `~/.mlx-node/bin/mlx-agent`).
 *   6. Download a prebuilt release tarball into (5), then use it.
 */
export async function resolveAgentBinary(
  repoRoot = findRepoRoot(),
  env = process.env,
  homeDir: string = homedir(),
): Promise<string> {
  const managed = managedAgentBinPath(homeDir);
  const probe = probeAgentInstall(repoRoot, env, homeDir);
  const probed = probe.path;
  // Pass the caller's pin through explicitly: resolveReleaseCandidates reads
  // process.env, which can diverge from the env this call probed under.
  const pin = releasePin(env);
  // A fresh download holds the lock until installAgentRelease returns,
  // then releases it — a concurrent updater's sweep can still delete the
  // new binary before the launcher finishes serve startup and spawns
  // (a same-tag retry keeps only its tag + the previous pointer). Same
  // rule as the probe: no managed path leaves this function unleased.
  const download = async (): Promise<string> => {
    let path = await downloadAgentBinary(managed, pin, env);
    if ((await leasePinnedManaged(managed, path)) === undefined) {
      // Swept before the lease landed — retry once, then trust the result:
      // a third party outpacing two consecutive downloads is pathological,
      // and spawn fails ENOENT just as it would without this fix.
      path = await downloadAgentBinary(managed, pin, env);
      await leasePinnedManaged(managed, path);
    }
    return path;
  };
  if (probed === undefined) return download();

  // Any managed path — pinned or pointer-resolved — is leased under the
  // install lock before it is returned: the launcher spawns it only after
  // serve startup, and an updater's sweep between probe and spawn can
  // delete it (a fallback install keeps only its tag + the previous
  // pointer). env/bundled/vendored paths are exempt: no sweep touches
  // files outside the managed cache.
  if (probe.source !== 'managed') return probed;
  // Swept between probe and lock → download. The lock is released inside
  // the helper first: downloadAgentBinary acquires it itself.
  return (await leasePinnedManaged(managed, probed)) ?? download();
}

/**
 * Serialize a cached managed binary's lease against the install sweep and
 * confirm it survived until the locked re-verify. Applies to pinned AND
 * unpinned resolves: the probe's unlocked lease can be missed by a sweep
 * that snapshotted the directory just before it — and can lose the binary
 * outright — so every caller that retains a managed path (the launcher
 * above, the evals harness) must run this before trusting it. Returns the
 * path, or undefined when a sweep deleted it first and the caller must
 * download or abort.
 */
export async function leasePinnedManaged(managed: string, probed: string): Promise<string | undefined> {
  await acquireInstallLock(`${managed}.lock`);
  let survived = false;
  try {
    survived = existsSync(probed);
    if (survived) writeLeaseMarker(probed);
  } finally {
    releaseInstallLock(`${managed}.lock`);
  }
  return survived ? probed : undefined;
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
 * `mlx agent` invoked from the packaged desktop app: the CLI sits at
 * `Contents/Resources/app/node_modules/@mlx-node/cli/dist/…` and package.ts
 * drops the release binary at `Contents/Resources/native/mlx-agent`. Walking
 * up for the `Contents/Resources/app` anchor is more robust than guessing how
 * many `node_modules` levels the packager will add.
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

/**
 * Walk up from this module (…/packages/cli/src|dist/commands/agent/) to the
 * repository root — the directory that contains `grok-build/` after
 * `scripts/agent-bootstrap.sh` clones the fork. Returns undefined outside a
 * checkout (e.g. a global npm install), which skips the vendored probe.
 */
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

/** Public R2 mirror of the release assets — Cloudflare edge, much faster than GitHub. */
const MLX_AGENT_R2_BASE = 'https://dl.portato.si';
/** The R2 key holding the newest release tag, written by the release pipeline. */
const MLX_AGENT_LATEST_KEY = 'latest';

const TAG_PREFIX = 'mlx-agent-v';

/** `mlx-agent-v0.2.0` → '0.2.0' when it is a valid semver tag, else null. */
function tagVersion(tag: string): semver.SemVer | null {
  if (!tag.startsWith(TAG_PREFIX)) return null;
  const suffix = tag.slice(TAG_PREFIX.length);
  const v = semver.parse(suffix);
  // semver.parse normalizes ('v0.3.0', ' 0.3.0' all parse), but the tag
  // string is used literally for the asset name, versioned filename, and
  // .tag marker — only the canonical form is a real release tag.
  return v !== null && v.version === suffix ? v : null;
}

/**
 * Compare `mlx-agent-vX.Y.Z` tags semantically. A malformed tag sorts below
 * any valid one; two malformed tags are equal. Returns <0, 0, >0.
 */
export function compareReleaseTags(a: string, b: string): number {
  const va = tagVersion(a);
  const vb = tagVersion(b);
  if (va && vb) return semver.compare(va, vb);
  if (va) return 1;
  if (vb) return -1;
  return 0;
}

/**
 * A valid `MLX_AGENT_TAG` pin selects exactly that release. Invalid pins
 * throw wherever the pin is consulted — probe included — so a stale or
 * mismatched cache can never mask a misspelled pin and run anyway.
 */
function releasePin(env: NodeJS.ProcessEnv = process.env): string | undefined {
  const pinned = env[MLX_AGENT_TAG_ENV];
  if (!pinned) return undefined;
  // A pin that is malformed or below the minimum can never satisfy
  // probeAgentInstall — its marker sorts below the floor, so every launch
  // would redownload it and then reject its own cache as stale. Refuse the
  // pin instead of downloading something we know we cannot keep.
  if (!tagVersion(pinned) || compareReleaseTags(pinned, MLX_AGENT_MIN_RELEASE) < 0) {
    throw new Error(
      `${MLX_AGENT_TAG_ENV}=${pinned} is not a supported release tag (expected mlx-agent-vX.Y.Z at or above ${MLX_AGENT_MIN_RELEASE}).`,
    );
  }
  return pinned;
}

/**
 * The platform's release asset for `tag`. Linux releases are musl-only: the
 * static binary runs on gnu and musl hosts alike.
 */
function agentAssetName(tag: string): string {
  const platform = process.platform === 'darwin' ? 'macos' : process.platform === 'win32' ? 'windows' : 'linux-musl';
  const arch = process.arch === 'arm64' ? 'aarch64' : 'x86_64';
  return `mlx-agent-${tag}-${platform}-${arch}.tar.gz`;
}

/**
 * Ordered download candidates, NEWEST verified first: the `MLX_AGENT_TAG`
 * override alone (a pin requests exactly one release), then the R2
 * `latest` pointer and GitHub's /releases/latest — each only when it
 * verifies installable — sorted by version so mirror order never outranks
 * a newer release, then the baked minimum as the last resort.
 * Fork releases can ship at any cadence and a CLI rebuild picks them up
 * with no pin bump.
 */
async function resolveReleaseCandidates(env: NodeJS.ProcessEnv = process.env): Promise<string[]> {
  const pin = releasePin(env);
  if (pin !== undefined) return [pin];
  const candidates: string[] = [];
  // A candidate must be a well-formed tag at/above the floor and not
  // already queued — the R2 pointer usually names the same tag GitHub
  // calls latest.
  const usable = (tag: string | undefined): tag is string =>
    tag !== undefined &&
    !candidates.includes(tag) &&
    tagVersion(tag) !== null &&
    compareReleaseTags(tag, MLX_AGENT_MIN_RELEASE) >= 0;

  try {
    const res = await fetch(`${MLX_AGENT_R2_BASE}/${MLX_AGENT_LATEST_KEY}`, {
      cache: 'no-store',
      signal: AbortSignal.timeout(10_000),
    });
    if (res.ok) {
      // The tag pointer is a handful of bytes — cap the buffered body so a
      // hostile mirror cannot stream an unbounded response before the
      // GitHub fallback gets a chance.
      const body = await readCappedBody(res, MAX_META_RESPONSE_BYTES);
      const tag = body?.toString('utf-8').trim();
      // Syntax alone is not enough: a stale or rolled-back pointer can name
      // a release below this CLI's floor, and a misconfigured mirror can
      // point at a tag whose checksum is empty/malformed or whose archive
      // never shipped — accepting it would send downloadAgentBinary to a
      // dead end. Only a tag whose download can verify end-to-end is a
      // candidate; anything else falls through to GitHub's own latest.
      if (usable(tag) && (await releaseInstallable(tag, agentAssetName(tag)))) candidates.push(tag);
    }
  } catch {
    // fall through to GitHub
  }

  try {
    const res = await fetch(`https://api.github.com/repos/${MLX_AGENT_RELEASE_REPO}/releases/latest`, {
      headers: { Accept: 'application/vnd.github+json' },
      signal: AbortSignal.timeout(10_000),
    });
    if (res.ok) {
      const body = await readCappedBody(res, MAX_META_RESPONSE_BYTES);
      const tag =
        body !== undefined ? (JSON.parse(body.toString('utf-8')) as { tag_name?: string }).tag_name?.trim() : undefined;
      // Same gate as the mirror: a mid-publish latest whose assets are
      // still uploading must not win over an older, complete release.
      if (usable(tag) && (await releaseInstallable(tag, agentAssetName(tag)))) candidates.push(tag);
    }
  } catch {
    // fall through to the baked minimum
  }

  // Mirror order is not version order: a stale-but-valid R2 pointer must
  // not outrank a newer GitHub latest — the NEWEST verified tag is what
  // `latest` means, with the rest kept as fallbacks.
  candidates.sort((a, b) => compareReleaseTags(b, a));
  if (!candidates.includes(MLX_AGENT_MIN_RELEASE)) candidates.push(MLX_AGENT_MIN_RELEASE);
  return candidates;
}

/**
 * Which release tag to download — the newest installable candidate. All
 * consumers (`mlx agent update` status, the desktop fetch step) get the
 * mirror-verification semantics for free.
 */
export async function resolveReleaseTag(): Promise<string> {
  const candidates = await resolveReleaseCandidates();
  return candidates[0] ?? MLX_AGENT_MIN_RELEASE;
}

/**
 * Byte cap for metadata fetches (the `latest` tag pointer, `.sha256`
 * checksums, the releases API reply): a valid payload is a few KB, so
 * anything bigger is a hostile or misconfigured mirror — `res.text()` /
 * `res.json()` would otherwise buffer an unbounded body before the
 * fallback gets a chance.
 */
const MAX_META_RESPONSE_BYTES = 64 * 1024;

/**
 * Read a response body with a hard byte ceiling. A truthful oversized
 * Content-Length short-circuits without reading; a lying or absent one is
 * caught mid-stream and the reader is cancelled. Returns undefined when
 * the body exceeds `cap` so the caller falls through to the next source.
 */
async function readCappedBody(res: Response, cap: number): Promise<Buffer | undefined> {
  if (res.body === null) return Buffer.alloc(0);
  const declared = Number(res.headers.get('content-length'));
  if (Number.isFinite(declared) && declared > cap) return undefined;
  const chunks: Buffer[] = [];
  let total = 0;
  let over = false;
  const reader = res.body.getReader();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > cap) {
        over = true;
        break;
      }
      chunks.push(Buffer.from(value));
    }
  } finally {
    if (over) {
      reader.cancel().catch(() => {});
    }
  }
  return over ? undefined : Buffer.concat(chunks);
}

async function fetchText(url: string): Promise<string | undefined> {
  try {
    const res = await fetch(url, { redirect: 'follow', signal: AbortSignal.timeout(15_000) });
    if (!res.ok) return undefined;
    const body = await readCappedBody(res, MAX_META_RESPONSE_BYTES);
    if (body === undefined) return undefined;
    return body.toString('utf-8').trim();
  } catch {
    // caller falls through
  }
  return undefined;
}

/** How long a single archive-body fetch may run before we try the next mirror. */
const ARCHIVE_FETCH_TIMEOUT_MS = 120_000;
/**
 * Hard cap on archive bytes buffered before verification — the digest is
 * only checked AFTER the body lands, so an unbounded `arrayBuffer()` lets a
 * misconfigured or hostile mirror exhaust the CLI's memory inside the fetch
 * timeout. Well above the ~100MB binary; also the ceiling `tar`'s own
 * `maxBuffer` assumes.
 */
const MAX_AGENT_ARCHIVE_BYTES = 256 * 1024 * 1024;

/**
 * The checksum is fetched ONLY from the GitHub release — the trust anchor —
 * while the archive itself may come from either mirror. Fetching
 * `${mirror}.sha256` would let a compromised R2 object pair a tampered
 * tarball with a tampered digest; GitHub is what the release pipeline
 * publishes first and what our credentials write. A mirror that cannot
 * produce bytes matching the GitHub checksum simply loses the race.
 *
 * Returns the normalized hex digest, or undefined: a 404, an empty sidecar,
 * or a body with no 64-hex token (error page, truncated write) is not a
 * checksum, and callers must treat the release as unverifiable rather than
 * carry junk into the digest comparison. Accepts both the bare digest the
 * pipeline writes and the `sha256sum` `<hex>[ *]<name>` form.
 */
/**
 * Parse a `.sha256` sidecar into its digest: the bare 64-hex token the
 * pipeline writes, or `sha256sum`'s `<hex>[ *]<name>` form. Anything else
 * — error page, truncated write, a hex run embedded in prose — is not a
 * checksum, so callers treat the release as unverifiable rather than
 * carry junk into the digest comparison.
 */
function parseChecksumBody(body: string): string | undefined {
  // The digest is always the FIRST token in every supported layout; take
  // it character-precisely and require what follows to be whitespace/end.
  if (body.length < 64) return undefined;
  const hex = '0123456789abcdefABCDEF';
  for (let i = 0; i < 64; i++) {
    if (!hex.includes(body[i]!)) return undefined;
  }
  const rest = body.slice(64);
  // Bare digest (possibly newline/space-terminated), or `sha256sum`'s
  // ` *name` / `  name` — either way the next char must be whitespace.
  if (rest.length > 0 && rest[0] !== ' ' && rest[0] !== '\t' && rest[0] !== '\n' && rest[0] !== '\r') {
    return undefined;
  }
  return body.slice(0, 64).toLowerCase();
}

async function releaseChecksum(tag: string, asset: string): Promise<string | undefined> {
  const body = await fetchText(`https://github.com/${MLX_AGENT_RELEASE_REPO}/releases/download/${tag}/${asset}.sha256`);
  return body !== undefined ? parseChecksumBody(body) : undefined;
}

/**
 * Can the archive be fetched from at least one mirror? A GET cancelled
 * after the headers lands costs a few KB at most (HEAD would do, but some
 * mirrors refuse it); presence is all this proves — the actual bytes are
 * verified against the checksum at download time.
 */
async function releaseArchiveReachable(tag: string, asset: string): Promise<boolean> {
  const urls = [
    `${MLX_AGENT_R2_BASE}/${tag}/${asset}`,
    `https://github.com/${MLX_AGENT_RELEASE_REPO}/releases/download/${tag}/${asset}`,
  ];
  for (const url of urls) {
    try {
      const res = await fetch(url, { redirect: 'follow', signal: AbortSignal.timeout(10_000) });
      res.body?.cancel().catch(() => {});
      if (res.ok) return true;
    } catch {
      // try the next mirror
    }
  }
  return false;
}

/**
 * A tag is a download candidate only when its archive exists on some
 * mirror AND a parseable checksum exists on the GitHub release — the two
 * things downloadAgentBinary will need. Missing either means this tag can
 * only fail, so resolution must not stop here. Presence is deliberately
 * all this checks: hashing a ~100MB body during resolution would tax every
 * `update --check`, and a candidate that lies about its bytes is caught by
 * the digest at download time, where the next candidate gets a turn.
 */
async function releaseInstallable(tag: string, asset: string): Promise<boolean> {
  return (await releaseChecksum(tag, asset)) !== undefined && (await releaseArchiveReachable(tag, asset));
}

/**
 * Download `mlx-agent-<tag>-<platform>-<arch>.tar.gz` and verify its SHA256
 * against the GitHub-published checksum before extracting — the desktop app
 * SIGNS these bytes, so a tampered mirror must never reach the bundle.
 * R2 mirror first, GitHub release as fallback.
 *
 * `dest` is the managed cache BASE path (`bin/mlx-agent`); the binary lands
 * at `bin/<tag>` and the `<dest>.tag` pointer flips to it last, so a running
 * previous version is never renamed over. A download servicing the
 * `MLX_AGENT_TAG` pin is a side-install: the pointer (the unpinned default
 * release) is left alone. Returns the versioned path.
 */
export async function downloadAgentBinary(
  dest: string,
  tag?: string,
  env: NodeJS.ProcessEnv = process.env,
  floor?: string,
): Promise<string> {
  if (tag !== undefined) return installAgentRelease(dest, tag, await fetchAgentRelease(dest, tag), env);
  // Unpinned: walk the resolved candidates. Resolution already gated on a
  // parseable checksum plus a reachable archive, but that probe raced the
  // real download — a tag whose fetch/verify still fails yields to the
  // next candidate instead of dead-ending the launch. `floor` is the
  // update path's downgrade guard: a candidate at or below the installed
  // release is skipped, so a failed latest keeps the working install
  // instead of flipping the pointer backward. Launch passes no floor —
  // repairing an empty/broken cache to ANY fetchable release beats none.
  const failures: string[] = [];
  let eligible = false;
  for (const candidate of await resolveReleaseCandidates(env)) {
    if (floor !== undefined && compareReleaseTags(candidate, floor) <= 0) continue;
    eligible = true;
    let binary: Buffer;
    try {
      binary = await fetchAgentRelease(dest, candidate);
    } catch (error) {
      failures.push(`${candidate}: ${error instanceof Error ? error.message : String(error)}`);
      continue;
    }
    // Install failures (lock contention, fs errors) are not tag-specific —
    // propagating beats retrying the same failure under another tag.
    return installAgentRelease(dest, candidate, binary, env);
  }
  if (!eligible) {
    throw new Error(`mlx agent: no release newer than ${floor} is fetchable — the installed version was kept.`);
  }
  throw new Error(
    `mlx agent: no local binary found and the verified download failed ` +
      `(${failures.join('; ')}). ` +
      `Set ${MLX_AGENT_BIN_ENV}, or run \`yarn build:agent\` in the mlx-node repo.`,
  );
}

/**
 * Fetch `tag`'s archive from the mirrors and unpack it. Returns the binary
 * bytes ONLY when they hash to the GitHub-published checksum — R2 first,
 * GitHub release as fallback, either mirror's mismatch falling through to
 * the next. Throws when no mirror produces verified bytes (or a usable
 * checksum does not exist), so the caller's next candidate gets a turn.
 */
async function fetchAgentRelease(dest: string, tag: string): Promise<Buffer> {
  const asset = agentAssetName(tag);
  const urls = [
    `${MLX_AGENT_R2_BASE}/${tag}/${asset}`,
    `https://github.com/${MLX_AGENT_RELEASE_REPO}/releases/download/${tag}/${asset}`,
  ];

  const expected = await releaseChecksum(tag, asset);
  if (!expected) {
    throw new Error(`cannot verify ${asset} — its .sha256 is missing or malformed on the GitHub release`);
  }

  mkdirSync(dirname(dest), { recursive: true, mode: 0o700 });
  let tar: Buffer | null = null;
  for (const url of urls) {
    try {
      // Bounded: a mirror that stalls mid-body must not hang the launch or
      // the desktop-release fetch step — the fallback mirror is one try away.
      const r = await fetch(url, {
        redirect: 'follow',
        signal: AbortSignal.timeout(ARCHIVE_FETCH_TIMEOUT_MS),
      });
      if (!r.ok || r.body === null) continue;
      // A truthful oversized Content-Length short-circuits the stream; a
      // lying or absent one is caught by the per-chunk cap below. Either
      // way this mirror loses the race and the next one is tried.
      const declared = Number(r.headers.get('content-length'));
      if (Number.isFinite(declared) && declared > MAX_AGENT_ARCHIVE_BYTES) continue;
      const chunks: Buffer[] = [];
      let total = 0;
      let over = false;
      const reader = r.body.getReader();
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          total += value.byteLength;
          if (total > MAX_AGENT_ARCHIVE_BYTES) {
            over = true;
            break;
          }
          chunks.push(Buffer.from(value));
        }
      } finally {
        if (over) {
          reader.cancel().catch(() => {});
        }
      }
      if (over) continue;
      const buf = Buffer.concat(chunks);
      const digest = createHash('sha256').update(buf).digest('hex');
      if (digest !== expected) continue;
      tar = buf;
      break;
    } catch {
      // fall through to the next mirror
    }
  }
  if (!tar) {
    throw new Error(`verified download failed (${asset} unavailable or checksum-mismatched on R2 and GitHub)`);
  }
  const tmp = join(dirname(dest), `${asset}.tmp-${process.pid}`);
  writeFileSync(tmp, tar, { mode: 0o600 });
  const out = spawnSync('tar', ['-xzf', tmp, '-O'], { encoding: 'buffer', maxBuffer: 256 * 1024 * 1024 });
  rmSync(tmp, { force: true });
  if (out.status !== 0 || out.stdout.length === 0) {
    throw new Error(`downloaded ${asset} but failed to unpack it`);
  }
  return out.stdout;
}

/**
 * Install verified `binary` bytes for `tag` into the managed cache under
 * the install lock. Failures here are never retried with another tag —
 * they are local (lock, fs), not mirror problems.
 */
async function installAgentRelease(dest: string, tag: string, binary: Buffer, env: NodeJS.ProcessEnv): Promise<string> {
  const versioned = join(dirname(dest), agentBinName(tag));
  const { chmodSync, renameSync, readdirSync } = await import('node:fs');
  // The staged write → pointer flip → sweep sequence runs under a lockfile:
  // two concurrent installs of different tags must not let one process'
  // sweep delete the version the other's pointer just selected. Entering
  // this section WITHOUT the lock is worse than not installing — a merely
  // suspended holder plus an unlocked waiter can interleave sweeps and
  // leave the pointer naming a deleted binary — so the lock is mandatory.
  await acquireInstallLock(`${dest}.lock`);
  try {
    // Under the lock, a same-tag install may already be complete: two first
    // launches can download the same release concurrently, and renaming
    // over a running `<tag>.exe` on Windows fails with a sharing/access
    // error. The versioned file lands atomically, so an existing path is a
    // finished install — adopt it and only repair the pointer.
    const alreadyInstalled = existsSync(versioned);
    // A download for the MLX_AGENT_TAG pin is a SIDE-INSTALL: pinned probes
    // resolve the versioned file directly, so repointing the pointer would
    // silently change the default release for every unpinned launch (and
    // let the sweep delete the previously active binary).
    const pin = releasePin(env);
    const sideInstall = pin !== undefined && pin === tag;
    // Snapshot the pointer BEFORE the flip — it is the release unpinned
    // launches currently run, so the sweep must keep it.
    const prevPointer = readAgentTag(dest);
    // Atomic install: a sibling temp file renamed over the versioned path.
    // Truncating the live path fails ETXTBSY on Linux while an agent runs and
    // can expose a half-written binary to a concurrent launch everywhere else
    // — and on Windows a running binary cannot be replaced at all, which is
    // why installs write a NEW filename per release instead of dest itself.
    const staged = join(dirname(dest), `.mlx-agent.tmp-${process.pid}`);
    try {
      if (!alreadyInstalled) {
        writeFileSync(staged, binary, { mode: 0o700 });
        renameSync(staged, versioned);
        chmodSync(versioned, 0o700);
      }
      // The pointer flips LAST and atomically (sibling tmp + rename): probes
      // either see the old tag with its binary still on disk, or the new tag
      // with this file already in place — never a tag naming a missing
      // binary. Side-installs skip the flip: the pointer belongs to the
      // unpinned default release.
      if (!sideInstall) {
        const stagedTag = `${dest}.tag.tmp-${process.pid}`;
        try {
          writeFileSync(stagedTag, tag, { mode: 0o600 });
          renameSync(stagedTag, `${dest}.tag`);
        } catch (error) {
          rmSync(stagedTag, { force: true });
          throw error;
        }
      }
    } catch (error) {
      rmSync(staged, { force: true });
      throw error;
    }
    // A pinned side-install leaves a `<file>.lease-<pid>` marker: the pin
    // is process-local, so nothing else records that this binary is about
    // to be spawned — without it, an unpinned sweep in ANOTHER process can
    // delete the side-install between our lock release and the pinner's
    // launch, and the pinned launch fails ENOENT.
    if (sideInstall) writeLeaseMarker(versioned);
    // Old versioned binaries accumulate (~100MB each). A still-running one
    // cannot be deleted on Windows — expected; leave it for the next sweep.
    // Keep: this install's tag, the pointer's target read before AND after
    // the flip (a third party may have moved it), the env pin, and every
    // binary a live lease names — the sweep must never remove a binary a
    // launch can still resolve.
    const live = readAgentTag(dest);
    const keep = new Set([agentBinName(tag)]);
    for (const t of [prevPointer, live, pin]) {
      if (t !== undefined) keep.add(agentBinName(t));
    }
    try {
      const entries = readdirSync(dirname(dest));
      const liveLeases: { entry: string; binary: string }[] = [];
      // Unix socket nodes outlive their owner; sweeping a lease removes the
      // marker AND the endpoint file (a Windows pipe name is not a file).
      // The marker unlink is identity-gated like scanMembers': a recycled
      // pid can revive this same marker (writeLeaseMarker re-advertises on
      // wx collision) between pidAlive and rmSync, and deleting it unprotects
      // a binary a live pinner is about to launch.
      const removeLease = (marker: string, baseline: Stats): boolean => {
        if (!removeIfUnchanged(marker, baseline)) return false;
        if (process.platform !== 'win32') rmSync(leaseProbePath(marker), { force: true });
        return true;
      };
      for (const entry of entries) {
        const lease = parseLeaseName(entry);
        if (lease === undefined) continue;
        const file = join(dirname(dest), entry);
        // Baseline BEFORE pidAlive so the recheck covers the whole
        // liveness→unlink window; a changed node is a live lease.
        let baseline: Stats;
        try {
          baseline = statSync(file);
        } catch {
          continue; // vanished mid-scan — nothing to sweep
        }
        if (pidAlive(lease.pid)) {
          liveLeases.push({ entry, binary: lease.binary });
        } else if (!removeLease(file, baseline)) {
          // pinner is gone — unless the marker was revived mid-scan;
          // then the lease is live and its binary is protected.
          keep.add(lease.binary);
        }
      }
      // A live pid is not ownership: the OS can have recycled it onto an
      // unrelated process, and honoring that lease pins a ~100MB binary
      // forever. Like lock members, the lease is honored only while its
      // probe endpoint ({@link leaseProbePath}) answers — the endpoint
      // dies with the real pinner. A refused marker fresher than
      // MEMBER_STALE_MS is a pinner whose endpoint is still binding, so
      // it keeps its binary until the next sweep.
      const leaseAnswers = await Promise.all(
        liveLeases.map((l) => endpointAlive(leaseProbePath(join(dirname(dest), l.entry)))),
      );
      for (const [i, { entry, binary }] of liveLeases.entries()) {
        const file = join(dirname(dest), entry);
        if (leaseAnswers[i]) {
          keep.add(binary);
          continue;
        }
        let baseline: Stats;
        try {
          baseline = statSync(file);
          if (Date.now() - baseline.mtimeMs <= MEMBER_STALE_MS) {
            keep.add(binary);
            continue;
          }
        } catch {
          // Vanished mid-probe — keep the binary; the next sweep re-decides.
          keep.add(binary);
          continue;
        }
        // An aged refusal unlinks only the probed node: a revived or
        // rebound marker is a live lease whose binary stays protected.
        if (!removeLease(file, baseline)) keep.add(binary);
      }
      for (const entry of entries) {
        if (!entry.startsWith('mlx-agent-v') || entry.includes('.lease-') || keep.has(entry)) continue;
        try {
          rmSync(join(dirname(dest), entry), { force: true });
        } catch {
          // locked by a running agent — next sweep gets it
        }
      }
    } catch {
      // cache dir unreadable — not fatal to a completed install
    }
  } finally {
    releaseInstallLock(`${dest}.lock`);
  }
  return versioned;
}

/**
 * Test seam for {@link acquireInstallLock}.
 */
export interface AcquireLockHooks {
  /** Retry deadline override (default 10s). */
  deadlineMs?: number;
}

/** True while a pid still exists (EPERM = alive, another user). */
function pidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === 'EPERM';
  }
}

/**
 * Locks this process currently holds: logical lock path → member record.
 */
const HELD_INSTALL_LOCKS = new Map<string, { server?: Server; marker: string }>();
/** Per-acquire sequence so one process's concurrent acquires get distinct tickets. */
let memberSeq = 0;
/**
 * Per-process token carried in every marker name. A pid the OS recycles can
 * collide on pid+seq with a dead owner — it cannot collide on a random
 * nonce it never knew, so a live process can never rebind a stale marker's
 * name and the sweep's unlink-by-name cannot remove a live replacement.
 */
const instanceNonce = randomBytes(5).toString('hex');

/**
 * The directory advertising lock members. Single-name locks cannot be
 * reclaimed safely — every scheme eventually has to unlink or replace a
 * name a live process may already hold — so the lock is a membership set
 * instead: each contender advertises `b-<ticket>-<pid>-<seq>-<nonce>`
 * (unix socket; marker file backed by a named pipe on Windows) and the
 * lowest ticket among live members holds, where tickets are bakery
 * numbers (`1 + max(seen)`), not pids — pids wrap, so pid order is not
 * arrival order. Nobody ever removes a name owned by a live process: the
 * nonce is drawn fresh per process, so a name a dead pid's recycled
 * successor could rebind does not exist, and deleting a stale marker by
 * name is unconditional and race-free. The directory sits under tmpdir
 * when `.lock.d` would exceed the ~104-byte socket path cap (member name
 * adds ~32 bytes).
 */
/** @internal Exported for tests; {@link acquireInstallLock} is the only production caller. */
export function lockDirPath(path: string): { dir: string; legacy: string | undefined } {
  const dir = `${path}.d`;
  if (process.platform === 'win32') return { dir, legacy: undefined };
  // Budget: `dir/b-<ticket>-<pid>-<seq>-<nonce>` must stay under the
  // ~104-byte sun_path cap (macOS reports over-long socket paths as
  // EINVAL), so an over-long `.lock.d` — a deep $HOME — is hashed into a
  // short tmpdir name. The name itself runs ~32 bytes worst case.
  if (Buffer.byteLength(dir) < 70) return { dir, legacy: path };
  const hash = createHash('sha256').update(resolve(dir)).digest('hex').slice(0, 16);
  return { dir: join(tmpdir(), `a-${hash}.d`), legacy: path };
}

/**
 * Where a marker's liveness endpoint listens. On unix the marker path IS
 * a bound socket; on Windows files cannot be bound, so each marker owns a
 * named pipe derived from its path — pipe names die with their owner,
 * which is what proves the recorded pid still holds the marker (a
 * recycled pid never created the pipe and answers refused). The hash
 * keeps names short and unique per lock+marker.
 */
/** @internal Exported for tests; {@link memberListening} probes it in production. */
export function markerProbePath(node: string): string {
  if (process.platform !== 'win32') return node;
  const hash = createHash('sha256').update(resolve(node)).digest('hex').slice(0, 32);
  return `\\\\.\\pipe\\mlx-agent-lock-${hash}`;
}

/**
 * Where a lease marker's liveness endpoint listens. The marker itself is a
 * regular file — it must exist synchronously for callers that lease inside
 * the install lock and release before the async bind finishes — so the
 * socket sits BESIDE it at `${marker}.sock`, hashed into tmpdir past the
 * ~104-byte sun_path cap (deep $HOME). Windows uses a derived named pipe
 * like {@link markerProbePath}. Either way the endpoint dies with the
 * pinner: a live pid whose endpoint refuses is a recycled pid fronting a
 * stale lease, never authoritative.
 * @internal Exported for tests; the sweep probes it in production.
 */
export function leaseProbePath(node: string): string {
  if (process.platform === 'win32') {
    const hash = createHash('sha256').update(resolve(node)).digest('hex').slice(0, 32);
    return `\\\\.\\pipe\\mlx-agent-lease-${hash}`;
  }
  const sock = `${node}.sock`;
  if (Buffer.byteLength(sock) < 100) return sock;
  const hash = createHash('sha256').update(resolve(node)).digest('hex').slice(0, 32);
  return join(tmpdir(), `mlx-agent-lease-${hash}`);
}

/** True while something accepts connections on `probe` — i.e. a live owner. */
function endpointAlive(probe: string): Promise<boolean> {
  return new Promise((res) => {
    const sock = connect(probe);
    const timer = setTimeout(() => done(true), 250); // wedged listener: assume live, never evict
    const done = (alive: boolean) => {
      clearTimeout(timer);
      sock.destroy();
      res(alive);
    };
    sock.once('connect', () => done(true));
    sock.once('error', () => done(false));
  });
}

/** True while something accepts connections on `node`'s probe endpoint — i.e. a live member. */
function memberListening(node: string): Promise<boolean> {
  return endpointAlive(markerProbePath(node));
}

/**
 * Lease endpoints this process owns: marker path → in-flight bind. Lease
 * servers are held for the process lifetime — a lease protects its binary
 * only while the pinner is launchable. `unref` keeps the listener from
 * pinning the event loop (`mlx agent update` must still exit).
 */
const HELD_LEASES = new Map<string, Promise<Server | undefined>>();

/**
 * Advertise that this process is about to launch `binary`: write the
 * `<binary>.lease-<pid>` marker and bind its probe endpoint
 * ({@link leaseProbePath}). The marker file exists synchronously so the
 * sweep sees it before the async bind completes — a fresh marker that
 * refuses is mid-bind and keeps its binary via the MEMBER_STALE_MS
 * grace, exactly like lock members. Best-effort: an unwritable dir or a
 * failed bind leaves a plain marker the next sweep ages out.
 * @internal Exported for tests; production callers are the three lease sites.
 */
export function writeLeaseMarker(binary: string): void {
  const marker = `${binary}.lease-${process.pid}`;
  try {
    writeFileSync(marker, '', { flag: 'wx' });
  } catch {
    // lease already exists or dir unwritable — best-effort. When the pid was
    // recycled the stale marker carries a stale mtime, which a sweep reads
    // as "refused endpoint, past the mid-bind grace" and deletes before our
    // async bind lands — re-advertise by refreshing mtime so the grace
    // covers PID-reuse recovery too.
    try {
      const now = new Date();
      utimesSync(marker, now, now);
    } catch {
      // not our marker after all — nothing to refresh
    }
  }
  if (HELD_LEASES.has(marker)) return;
  const bind = bindLeaseEndpoint(marker)
    .then((server) => {
      server.unref();
      return server;
    })
    .catch(() => {
      // No endpoint → the marker is honored only via the mid-bind grace
      // and then swept; retry on the next lease attempt.
      HELD_LEASES.delete(marker);
      return undefined;
    });
  HELD_LEASES.set(marker, bind);
}

/**
 * Bind `marker`'s probe endpoint. One EADDRINUSE retry clears a stale
 * socket node our dead predecessor left (the marker file itself stays —
 * removing it would open a window where a sweep sees NO lease at all).
 */
function bindLeaseEndpoint(marker: string): Promise<Server> {
  const probe = leaseProbePath(marker);
  const listen = (): Promise<Server> =>
    new Promise<Server>((res, rej) => {
      const server = createServer((sock) => sock.destroy());
      server.on('error', () => {}); // stray runtime errors must not crash the holder
      server.once('error', rej);
      server.listen(probe, () => {
        server.removeListener('error', rej);
        res(server);
      });
    });
  return listen().catch((error) => {
    if ((error as NodeJS.ErrnoException).code !== 'EADDRINUSE') throw error;
    if (process.platform !== 'win32') rmSync(probe, { force: true });
    return listen();
  });
}

/** @internal Exported for tests; {@link acquireInstallLock} is the only production consumer. */
export interface LockMember {
  /**
   * Bakery ticket: `1 + max(tickets seen at choose time)`, so a marker
   * that advertised AFTER us always sorts behind us. Ticket order — not
   * pid order — is what makes arrival order win: pids wrap and get reused,
   * so a later process can carry a lower pid than the current holder.
   */
  ticket: number;
  pid: number;
  seq: number;
  file: string;
}

/** @internal Exported for tests; {@link acquireInstallLock} is the only production consumer. */
export interface MemberScan {
  members: LockMember[];
  /** A contender is mid-choose, or the snapshot is stale and the caller must rescan. */
  choosing: boolean;
}

/**
 * A `b-`/`m-`/`c-` marker is swept on a refused endpoint probe only when
 * it is older than this: on unix the marker node exists between bind()
 * and listen() completing, so a fresh marker that refuses is a member
 * mid-bind, and unlinking it would leave that acquirer scanning forever
 * for a name that is gone. (On Windows the marker file is written only
 * after its pipe is already listening, so the window cannot occur — the
 * bound stays for symmetry.) Recycled-pid markers are by definition old.
 */
const MEMBER_STALE_MS = 2_000;

/** Decimal digits only — leading zeros are fine, signs are not. */
function isDigits(s: string): boolean {
  if (s.length === 0) return false;
  for (const c of s) {
    if (c < '0' || c > '9') return false;
  }
  return true;
}

/**
 * Parse one lock-directory entry. Marker layouts:
 *   `b-<ticket>-<pid>-<seq>-<nonce>` — bakery member, kind 'member'
 *   `b-<ticket>-<pid>-<seq>`        — member without an ownership nonce
 *   `m-<pid>-<seq>`                 — legacy member (this PR's first design), ticket 0
 *   `c-<pid>-<seq>-<nonce>`         — contender mid-choose
 *   `c-<pid>-<seq>`                 — contender without an ownership nonce
 * The nonce ({@link instanceNonce}) makes the name un-rebindable: unlinking
 * a nonced name can never remove a live process's marker, so sweeps are
 * free of the stat→unlink replacement race. The nonce-free forms are kept
 * parseable — a marker written before this change still counts/sweeps.
 * Returns undefined for anything else — foreign files in the directory are
 * not markers and are never swept.
 * @internal Exported for tests; {@link scanMembers} is the production consumer.
 */
export function parseLockMarker(entry: string): { member: LockMember; choosing: boolean } | undefined {
  const parts = entry.split('-');
  const kind = parts[0];
  if (kind === 'b' && (parts.length === 4 || parts.length === 5)) {
    const [ticket, pid, seq] = parts.slice(1);
    if (!isDigits(ticket!) || !isDigits(pid!) || !isDigits(seq!)) return undefined;
    return {
      member: { ticket: Number(ticket), pid: Number(pid), seq: Number(seq), file: '' },
      choosing: false,
    };
  }
  if ((kind === 'm' && parts.length === 3) || (kind === 'c' && (parts.length === 3 || parts.length === 4))) {
    const [pid, seq] = parts.slice(1);
    if (!isDigits(pid!) || !isDigits(seq!)) return undefined;
    // Legacy m- members predate bakery tickets (ticket 0 — ahead of all b-);
    // c- contenders are choosing (ticket -1 — never a member yet).
    return {
      member: { ticket: kind === 'm' ? 0 : -1, pid: Number(pid), seq: Number(seq), file: '' },
      choosing: kind === 'c',
    };
  }
  return undefined;
}

/**
 * Parse a `<versioned>.lease-<pid>` name. The versioned part is whatever
 * precedes the LAST `.lease-` separator, so binary names containing dots
 * (`mlx-agent-v1.2.3.exe`) parse correctly.
 */
function parseLeaseName(entry: string): { binary: string; pid: number } | undefined {
  const sep = entry.lastIndexOf('.lease-');
  if (sep < 0) return undefined;
  const binary = entry.slice(0, sep);
  const pid = entry.slice(sep + '.lease-'.length);
  if (binary.length === 0 || !isDigits(pid)) return undefined;
  return { binary, pid: Number(pid) };
}

/**
 * Live-and-present state of `dir`. Marker kinds (see {@link parseLockMarker}):
 *   `b-<ticket>-<pid>-<seq>-<nonce>` — a bakery member (socket on unix,
 *                              file on win); the nonce is the owner's
 *                              per-process token, so no live process can
 *                              rebind a stale marker's name.
 *   `m-<pid>-<seq>`          — a member from this lock's previous design;
 *                              honored as ticket 0 (its owner advertised
 *                              before any bakery member).
 *   `c-<pid>-<seq>-<nonce>`  — a contender between choosing a ticket and
 *                              advertising it (socket on unix, file on
 *                              win); live ones force a rescan.
 * Anything whose pid is dead is swept unconditionally — the owner cannot
 * exist. A marker ALSO has to answer on its probe endpoint (unix socket /
 * Windows named pipe): a live pid with a dead endpoint is a recycled pid
 * fronting a stale marker, and honoring it would wait out every acquire
 * behind a lock nobody owns. Age alone never sweeps a live marker — a
 * suspended contender is indistinguishable from a recycled pid by
 * liveness, and removing a LIVE `c-` lets a later contender choose the
 * same ticket and enter concurrently. Nonced marker names make unlinking
 * safe by name alone — a replacement owner draws its own nonce, so it
 * cannot rebind the dead marker's name and the stat→unlink window has no
 * replacement to hit. The identity recheck ({@link removeIfUnchanged})
 * stays anyway: legacy nonce-free markers can still be rebinded mid-scan.
 */

/**
 * Unlink `file` only when a fresh stat still names the SAME node observed
 * at `baseline` (device + inode + mtime + ctime). A nonced marker name
 * cannot be rebound by a live process ({@link instanceNonce}), but a
 * legacy nonce-free one can — the unlink must refuse any replacement.
 * Returns false when the node changed or vanished: the directory snapshot
 * is stale and the caller must rescan rather than act on a member list
 * that may already name a different owner.
 * @internal Exported for tests; {@link scanMembers} is the production consumer.
 */
export function removeIfUnchanged(file: string, baseline: Stats): boolean {
  try {
    const now = statSync(file);
    if (
      now.dev !== baseline.dev ||
      now.ino !== baseline.ino ||
      now.mtimeMs !== baseline.mtimeMs ||
      now.ctimeMs !== baseline.ctimeMs
    ) {
      return false;
    }
    rmSync(file, { force: true });
    return true;
  } catch {
    return false;
  }
}
/** @internal Exported for tests; {@link acquireInstallLock} is the only production caller. */
export async function scanMembers(dir: string): Promise<MemberScan> {
  let entries: string[];
  try {
    entries = readdirSync(dir);
  } catch {
    return { members: [], choosing: false };
  }
  const members: LockMember[] = [];
  const pending: { member: LockMember; choosing: boolean }[] = [];
  let choosing = false;
  for (const entry of entries) {
    const marker = parseLockMarker(entry);
    if (marker === undefined) continue;
    const member = { ...marker.member, file: join(dir, entry) };
    if (pidAlive(member.pid)) {
      pending.push({ member, choosing: marker.choosing });
      continue;
    }
    // Dead pid — unlink the node we observed. A nonced name can never be
    // rebound by a live process; a legacy name still can, so refuse any
    // replacement and force a rescan when the snapshot is stale.
    try {
      if (!removeIfUnchanged(member.file, statSync(member.file))) choosing = true;
    } catch {
      // Gone before stat — another sweeper beat us, or a recycled pid is
      // mid-rebind. The snapshot is stale either way: rescan.
      choosing = true;
    }
  }
  // Probe live-pid markers in parallel: the endpoint check is the cheap
  // ECONNREFUSED path for recycled pids, so the extra round-trip per
  // marker is only paid when markers exist. A refused marker is swept only
  // once it is too old to be an owner still finishing bind()+listen().
  const sockets = await Promise.all(pending.map((p) => memberListening(p.member.file)));
  pending.forEach(({ member, choosing: isChoosing }, i) => {
    let fresh: boolean | undefined;
    let baseline: Stats | undefined;
    if (!sockets[i]) {
      try {
        baseline = statSync(member.file);
        fresh = Date.now() - baseline.mtimeMs <= MEMBER_STALE_MS;
      } catch {
        // Vanished mid-probe: a c- marker's owner may have just swapped it
        // for a b- member our readdir snapshot predates, so the contender
        // is NOT absent — report "still choosing" and force the caller to
        // rescan rather than grant on a member list that missed the flip.
        choosing = true;
        return;
      }
    }
    if (!sockets[i] && !fresh) {
      // Same rule as the dead-pid sweep: the refusal was probed against
      // this node, so unlink only that node — a rebind (possible only for
      // legacy nonce-free names) is a live contender we missed.
      if (!removeIfUnchanged(member.file, baseline!)) choosing = true;
      return;
    }
    if (isChoosing) choosing = true;
    else members.push(member);
  });
  members.sort((a, b) => a.ticket - b.ticket || a.pid - b.pid || a.seq - b.seq);
  return { members, choosing };
}

/**
 * Bind a marker's probe endpoint — the unix socket at `marker` itself, or
 * its derived named pipe on Windows. One EADDRINUSE retry: the name can
 * outlive a dead owner's scan sweep (stale socket node; a same-name pipe
 * an unrelated process happens to hold is unbindable but unremovable, so
 * a second refusal fails the acquire outright rather than stealing it).
 */
async function bindMarkerListener(marker: string): Promise<Server> {
  const probe = markerProbePath(marker);
  const bind = (): Promise<Server> =>
    new Promise<Server>((res, rej) => {
      const server = createServer((sock) => sock.destroy());
      server.on('error', () => {}); // stray runtime errors must not crash the holder
      server.once('error', rej);
      server.listen(probe, () => {
        server.removeListener('error', rej);
        res(server);
      });
    });
  return bind().catch(async (error) => {
    if ((error as NodeJS.ErrnoException).code !== 'EADDRINUSE') throw error;
    // A dead-pid marker file can survive the scan that should have swept
    // it — removing the file frees a stale unix socket node. On Windows
    // the file is not the name; the retry just re-races the foreign pipe.
    rmSync(marker, { force: true });
    return bind();
  });
}

/**
 * Serialize the install critical section across processes. Bakery
 * protocol: advertise `c-<pid>-<seq>-<nonce>` while choosing, take ticket
 * `1 + max(seen)`, then bind `b-<ticket>-<pid>-<seq>-<nonce>` and wait
 * until ours is the lowest live ticket. Arrival order is preserved because any
 * member advertised after us necessarily saw our ticket when choosing;
 * a live `c-` marker forces waiters to rescan so a mid-choose contender
 * cannot be leapfrogged by a stale max. A dead member's marker is deleted
 * by the next scan unconditionally — no liveness probe is trusted to
 * unlink anything.
 *
 * On Unix each marker — choose and member — is a listening unix socket;
 * on Windows the marker is an empty file plus a named pipe derived from
 * its path ({@link markerProbePath}). Either way the endpoint dies with
 * its owner, so a live pid with a refused endpoint is a recycled pid
 * fronting a stale marker — never authoritative. The pipe is bound
 * BEFORE the marker file is written so a fresh marker's mtime always
 * post-dates a probeable endpoint.
 *
 * Once the deadline passes the caller must FAIL rather than proceed
 * unlocked: the pointer-flip/sweep section is unsafe to enter while a
 * holder may still be inside it, so this function throws instead of
 * returning.
 */
export async function acquireInstallLock(path: string, hooks: AcquireLockHooks = {}): Promise<void> {
  const deadline = Date.now() + (hooks.deadlineMs ?? 10_000);
  const { dir, legacy } = lockDirPath(path);
  mkdirSync(dir, { recursive: true });
  const seq = memberSeq++;
  // Bakery choose phase: advertise `c-` first so no scanner computes a
  // ticket while ours is unrecorded, then take `1 + max` of the tickets
  // seen — any member that advertised after us necessarily read a max that
  // includes ours, so arrival order is honored no matter how pids order.
  const choosing = join(dir, `c-${process.pid}-${seq}-${instanceNonce}`);
  let choosingServer: Server | undefined;
  if (process.platform === 'win32') {
    // The pipe is the probeable proof — it is bound BEFORE the marker
    // file so a scanner never sees an unanswerable marker as live.
    choosingServer = await bindMarkerListener(choosing);
    try {
      writeFileSync(choosing, '', { flag: 'wx' });
    } catch {
      // left over from a crashed choose — reuse it
    }
  } else {
    // A unix marker must answer on its socket path, or a live contender
    // suspended mid-choose gets swept like a recycled pid and a later
    // contender can choose the same ticket and enter concurrently.
    choosingServer = await bindMarkerListener(choosing);
  }
  const seen = await scanMembers(dir);
  const ticket = seen.members.reduce((max, m) => Math.max(max, m.ticket), 0) + 1;
  const marker = join(dir, `b-${ticket}-${process.pid}-${seq}-${instanceNonce}`);
  const server = await bindMarkerListener(marker);
  if (process.platform === 'win32') {
    // Pipe bound first — the marker file only advertises ticket/pid/seq;
    // the pipe is what proves the recorded pid owns it.
    try {
      writeFileSync(marker, '', { flag: 'wx' });
    } catch {
      // exists from a stale scan race — reuse is fine, ticket is per-acquire
    }
  }
  choosingServer?.close();
  rmSync(choosing, { force: true });
  try {
    while (true) {
      if (Date.now() > deadline) {
        throw new Error(
          `mlx agent: another install is in progress (${path}) — refusing to proceed without the install lock`,
        );
      }
      // Legacy single-file locks (this PR's earlier socket design, or a
      // pre-pointer pid file): an acceptor counts as a live holder — its
      // ticket predates all members, so we wait for it rather than steal.
      if (legacy !== undefined && existsSync(legacy)) {
        if (await memberListening(legacy)) {
          await new Promise((r) => setTimeout(r, 50));
          continue;
        }
        rmSync(legacy, { force: true });
      }
      const scan = await scanMembers(dir);
      if (scan.choosing) {
        // A contender is mid-choose — its ticket may undercut ours. Rescan
        // until it advertises (its c- marker dies with its pid if it never
        // finishes, so this cannot hang).
        await new Promise((r) => setTimeout(r, 50));
        continue;
      }
      // Lowest ticket wins; equal tickets break on pid then seq. Members
      // that advertised after us carry a strictly higher ticket by the
      // choose phase above. Our OWN marker must be in the scan: a member
      // suspended past MEMBER_STALE_MS with an unanswerable endpoint gets
      // swept as aged/refused, and resuming to an every() that contains
      // only higher tickets would grant the lock a second time while
      // another process holds it. Missing own marker → wait out the
      // deadline rather than enter.
      const us = (m: LockMember) => m.ticket === ticket && m.pid === process.pid && m.seq === seq;
      const behind = (m: LockMember) =>
        m.ticket > ticket || (m.ticket === ticket && (m.pid > process.pid || (m.pid === process.pid && m.seq > seq)));
      if (scan.members.some(us) && scan.members.every((m) => us(m) || behind(m))) {
        HELD_INSTALL_LOCKS.set(path, { server, marker });
        return;
      }
      await new Promise((r) => setTimeout(r, 50));
    }
  } catch (error) {
    server?.close();
    choosingServer?.close();
    rmSync(marker, { force: true });
    rmSync(choosing, { force: true });
    throw error;
  }
}

/**
 * Extract `http://host:port` from an `mlx serve` banner line like
 * `… listening on http://127.0.0.1:9821 …`. Anything before the marker is
 * skipped (timestamps, log prefixes); the URL ends at whitespace or EOL.
 */
export function serveBannerBaseUrl(line: string): string | undefined {
  const marker = 'listening on http://';
  const start = line.indexOf(marker);
  if (start < 0) return undefined;
  const urlStart = start + marker.length - 'http://'.length;
  let end = urlStart;
  while (end < line.length && line.charCodeAt(end) > 32) end++;
  return end > urlStart ? line.slice(urlStart, end) : undefined;
}

/** @internal Exported for tests; {@link downloadAgentBinary} is the only production caller. */
export function releaseInstallLock(path: string): void {
  try {
    const held = HELD_INSTALL_LOCKS.get(path);
    if (!held) return;
    HELD_INSTALL_LOCKS.delete(path);
    held.server?.close();
    // Our own marker — unconditionally safe to remove.
    rmSync(held.marker, { force: true });
  } catch {
    // lock cleanup is best-effort
  }
}

/** What `mlx agent update --check` reports and the desktop tray consumes. */
export interface AgentUpdateStatus {
  /** Release tag the active binary was built from; null when nothing is installed or stamped. */
  current: string | null;
  /** Newest published release tag. */
  latest: string;
  /** A newer managed download is available. */
  updateAvailable: boolean;
  /**
   * Base path of the managed cache `mlx agent update` writes — the binary
   * lands at the sibling `<tag>` file its `.tag` pointer names, not here.
   */
  managedPath: string;
  /**
   * How the NEXT `mlx agent` resolves its binary. `env`/`vendored` pins
   * shadow the managed cache, so an applied update has no effect until the
   * pin is removed — the CLI warns about this case.
   */
  pinned: AgentInstallProbe['source'];
  /**
   * Newest release tag on disk (the managed cache OR the effective install,
   * whichever is newer) — the floor a fallback update must beat, or the
   * pointer/install would move backward. Null when nothing is installed.
   */
  newestInstalled: string | null;
}

/** Injectable seams so tests never touch the network or $HOME. */
export interface AgentUpdateDeps {
  resolveTag?: () => Promise<string>;
  probe?: () => AgentInstallProbe;
  managedPath?: string;
  /**
   * `tag === undefined` asks the downloader to walk its candidate list;
   * `floor` skips candidates at or below the installed release so a
   * fallback can never downgrade it.
   */
  download?: (dest: string, tag?: string, floor?: string) => Promise<string>;
}

/**
 * Compare the published latest tag against the active install without
 * downloading anything. `MLX_AGENT_TAG` still pins `latest` — an explicit
 * pin reports whatever it points at.
 */
export async function agentUpdateStatus(deps: AgentUpdateDeps = {}): Promise<AgentUpdateStatus> {
  const probe = deps.probe ?? (() => probeAgentInstall());
  const resolveTag = deps.resolveTag ?? resolveReleaseTag;
  const managedPath = deps.managedPath ?? managedAgentBinPath();
  const [latest, install] = await Promise.all([resolveTag(), Promise.resolve(probe())]);
  const current = install.effectiveTag ?? null;
  // Update is needed when the newest artifact ON DISK trails latest. The
  // managed cache is counted even when a pin or an untagged bundle shadows
  // it: once it reaches latest, further downloads are pure waste.
  const newest = [current, install.managedTag].reduce<string | null>(
    (best, tag) =>
      tag !== undefined && tag !== null && (best === null || compareReleaseTags(tag, best) > 0) ? tag : best,
    null,
  );
  // An exact pin asks for that tag specifically: older cache, newer cache,
  // or bundle — anything else on disk still needs the pinned download.
  const pin = releasePin();
  const pinnedInstalled =
    pin !== undefined && (install.effectiveTag === pin || existsSync(join(dirname(managedPath), agentBinName(pin))));
  return {
    current,
    latest,
    updateAvailable: pin !== undefined ? !pinnedInstalled : newest === null || compareReleaseTags(latest, newest) > 0,
    managedPath,
    pinned: install.source,
    newestInstalled: newest,
  };
}

/**
 * `mlx agent update`: download the resolved release into the managed cache
 * when it is newer than the active install. Returns the status AFTER the
 * operation plus whether a download ran — a `null` current (missing or
 * unstamped binary) also counts as updatable.
 */
export async function applyAgentUpdate(
  deps: AgentUpdateDeps = {},
): Promise<{ status: AgentUpdateStatus; downloaded: boolean }> {
  const status = await agentUpdateStatus(deps);
  if (!status.updateAvailable) return { status, downloaded: false };
  const download = deps.download ?? ((dest, tag, floor) => downloadAgentBinary(dest, tag, process.env, floor));
  try {
    await download(status.managedPath, status.latest);
  } catch {
    // The latest tag verified at resolve time but cannot be fetched now
    // (corrupt mirror bytes, assets pulled mid-flight). Retry unpinned:
    // the downloader walks its verified-candidate list, so an older
    // complete release still lands rather than failing the update — but
    // only above the pre-update install: a candidate at or below it is a
    // downgrade, and keeping the working release beats flipping backward.
    await download(status.managedPath, undefined, status.newestInstalled ?? undefined);
  }
  // Report what actually installed — a fallback may have landed an older
  // candidate, so `latest` is not the truth. The pin wins when set (side
  // installs deliberately skip the pointer).
  const current = releasePin() ?? readAgentTag(status.managedPath) ?? status.latest;
  return {
    status: {
      ...status,
      current,
      // Availability is recomputed from what landed: a fallback that
      // installed below latest must still read as updatable, or callers
      // report "updated to latest" while running the older release.
      updateAvailable: compareReleaseTags(status.latest, current) > 0,
    },
    downloaded: true,
  };
}

/**
 * The release tag a versioned install path encodes (`…/mlx-agent-v1.2.3[.exe]`).
 * Lets packaging stamp the tag it actually downloaded rather than the tag
 * it asked for — a candidate fallback may have landed an older release.
 */
export function agentTagFromBinPath(path: string): string | undefined {
  const name = basename(path);
  const tag = name.endsWith('.exe') ? name.slice(0, -'.exe'.length) : name;
  return tagVersion(tag) !== null ? tag : undefined;
}

/**
 * Spawn `mlx serve` as a subprocess and wait for `/health`.
 * `--port 0` binds an ephemeral port; the token comes back to us through
 * `--auth-token`. The child CLI is THIS package's entry point — located via
 * `process.argv[1]` (the `mlx` bin) with `src/cli.ts` as a fallback for
 * test/dev invocations.
 */
async function spawnInferenceServer(
  opts: GrokAgentLaunchOptions,
): Promise<{ child: ChildProcess; url: string; token: string }> {
  const token = randomBytes(24).toString('base64url');
  const cliEntry = process.env.MLX_CLI_ENTRY ?? process.argv[1];
  if (!cliEntry) throw new Error('mlx agent: cannot locate the cli entry point for spawning `mlx serve`');

  const serveArgv = [
    cliEntry,
    'serve',
    '--port',
    '0',
    '--auth-token',
    token,
    ...(opts.modelsDir ? ['--models-dir', opts.modelsDir] : []),
  ];
  const child = (opts.spawnServe ?? ((argv, so) => spawn(process.execPath, argv, so)))(serveArgv, {
    stdio: ['ignore', 'pipe', 'inherit'],
  });

  // A child that never reports a port would otherwise leak: nothing else
  // retains the handle, so kill it before the launch path unwinds.
  try {
    const url = await waitForServe(child, opts.serveReadyTimeoutMs ?? 30_000);
    return { child, url, token };
  } catch (err) {
    child.kill('SIGTERM');
    throw err;
  }
}

/**
 * `mlx serve` prints `listening on http://HOST:PORT` — scrape it (the same
 * string `serve.ts` announces) and confirm `/health` answers.
 */
function waitForServe(child: ChildProcess, timeoutMs: number): Promise<string> {
  return new Promise((resolvePromise, reject) => {
    const rl = child.stdout ? createInterface({ input: child.stdout }) : undefined;
    const timer = setTimeout(() => fail(new Error('timed out waiting for `mlx serve` to report its port')), timeoutMs);

    const fail = (err: Error): void => {
      clearTimeout(timer);
      rl?.close();
      reject(err);
    };

    child.once('exit', (code) => fail(new Error(`mlx serve exited (${code}) before reporting a port`)));
    // A spawn failure emits 'error' without 'exit' — surface it instead of
    // hanging on the timeout.
    child.once('error', fail);
    rl?.on('line', (line) => {
      const base = serveBannerBaseUrl(line);
      if (base === undefined) return;
      const url = `${base}/v1`;
      // Health check: the banner prints before the listener is guaranteed
      // accept-ready under every platform, so confirm once.
      fetch(`${base}/health`)
        .then((res) => (res.ok ? url : Promise.reject(new Error(`health ${res.status}`))))
        .then((ready) => {
          clearTimeout(timer);
          rl?.close();
          resolvePromise(ready);
        })
        .catch(fail);
    });
  });
}

/**
 * Boot the inference host and exec the Grok Build binary over it.
 * Resolves with the agent's exit code; `run()` maps it to process.exitCode.
 */
export async function launchGrokAgent(opts: GrokAgentLaunchOptions): Promise<number> {
  const bin = await resolveAgentBinary();

  // Serving is the caller's routing decision — scanning `argv` again here
  // would reclassify help/version tokens sitting in a value slot as
  // one-shots and starve the turn of local inference.
  const needsServer = opts.needsServer !== false;

  let serve: { child: ChildProcess; url: string; token: string } | undefined;
  if (opts.serveUrl !== undefined) {
    serve = { child: undefined as unknown as ChildProcess, url: opts.serveUrl, token: opts.serveToken ?? '' };
  } else if (needsServer) {
    serve = await spawnInferenceServer(opts);
  }

  const env = { ...process.env };
  if (serve) {
    env.GROK_MODELS_BASE_URL = serve.url;
    env.GROK_CLI_CHAT_PROXY_BASE_URL = serve.url;
    env.GROK_XAI_API_BASE_URL = serve.url;
    env.XAI_API_KEY = serve.token || 'mlx-local';
    env.GROK_DISABLE_AUTOUPDATER = '1';
  }

  const spawnAgent = opts.spawnAgent ?? ((b, a, so) => spawn(b, a, so));
  const agent = spawnAgent(bin, opts.argv, { stdio: 'inherit', env });

  // Ctrl+C forwards to the TUI; the agent's exit then drives ours, so the
  // serve child is reaped in the finally below.
  const forward = (): void => {
    agent.kill('SIGINT');
  };
  // SIGTERM/SIGHUP end the process through the default disposition — which
  // never reaches children. Tear both down, then re-raise so the launcher
  // still dies with the same signal (process.once has already detached us).
  const teardown = (signal: NodeJS.Signals): void => {
    serve?.child?.kill(signal);
    agent.kill(signal);
    process.kill(process.pid, signal);
  };
  process.once('SIGINT', forward);
  process.once('SIGTERM', teardown);
  process.once('SIGHUP', teardown);

  let code: number;
  try {
    code = await new Promise<number>((resolvePromise, reject) => {
      agent.once('exit', (c, signal) => resolvePromise(c ?? (signal === 'SIGINT' ? 130 : 1)));
      // A spawn failure emits 'error' with no 'exit' — rejecting routes it
      // through the same teardown so the serve child is never orphaned.
      agent.once('error', reject);
    });
  } finally {
    process.removeListener('SIGINT', forward);
    process.removeListener('SIGTERM', teardown);
    process.removeListener('SIGHUP', teardown);
    serve?.child?.kill('SIGTERM');
  }
  return code;
}
