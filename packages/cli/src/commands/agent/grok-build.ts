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
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { connect, createServer, type Server } from 'node:net';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
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
      try {
        writeFileSync(`${pinnedBin}.lease-${process.pid}`, '', { flag: 'wx' });
      } catch {
        // lease already exists or dir unwritable — best-effort
      }
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
  // markers existed.
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
  // Pass the caller's pin through explicitly: resolveReleaseTag reads
  // process.env, which can diverge from the env this call probed under.
  const pin = releasePin(env);
  if (probed === undefined) return downloadAgentBinary(managed, pin, env);

  if (pin === undefined || probe.source !== 'managed' || probe.effectiveTag !== pin) {
    return probed;
  }
  // Swept between probe and lock → download the pin. The lock is released
  // inside the helper first: downloadAgentBinary acquires it itself.
  return (await leasePinnedManaged(managed, probed)) ?? downloadAgentBinary(managed, pin, env);
}

/**
 * Serialize a cached pinned binary's lease against the install sweep and
 * confirm it survived until the locked re-verify. The probe's unlocked
 * lease can be missed by a sweep that snapshotted the directory just
 * before it — and can lose the binary outright — so every caller that
 * retains a pinned managed path (the launcher above, the evals harness)
 * must run this before trusting it. Returns the path, or undefined when a
 * sweep deleted it first and the caller must download or abort.
 */
export async function leasePinnedManaged(managed: string, probed: string): Promise<string | undefined> {
  await acquireInstallLock(`${managed}.lock`);
  let survived = false;
  try {
    survived = existsSync(probed);
    if (survived) {
      try {
        writeFileSync(`${probed}.lease-${process.pid}`, '', { flag: 'wx' });
      } catch {
        // the probe already leased it, or the dir is unwritable — best-effort
      }
    }
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
 * Which release tag to download: `MLX_AGENT_TAG` override, then the R2
 * `latest` pointer (the release pipeline maintains it), then GitHub's
 * /releases/latest, then the baked minimum. Fork releases can ship at any
 * cadence and a CLI rebuild picks them up with no pin bump.
 */
export async function resolveReleaseTag(): Promise<string> {
  const pin = releasePin();
  if (pin !== undefined) return pin;

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
      // Syntax alone is not enough: a stale or rolled-back pointer can name a
      // release below this CLI's floor. Anything the pipeline publishes is
      // accepted once it clears the minimum.
      if (tag && tagVersion(tag) && compareReleaseTags(tag, MLX_AGENT_MIN_RELEASE) >= 0) return tag;
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
      if (tag && tagVersion(tag) && compareReleaseTags(tag, MLX_AGENT_MIN_RELEASE) >= 0) return tag;
    }
  } catch {
    // fall through to the baked minimum
  }

  return MLX_AGENT_MIN_RELEASE;
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
 */
async function releaseChecksum(tag: string, asset: string): Promise<string | undefined> {
  return fetchText(`https://github.com/${MLX_AGENT_RELEASE_REPO}/releases/download/${tag}/${asset}.sha256`);
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
): Promise<string> {
  tag ??= await resolveReleaseTag();
  const versioned = join(dirname(dest), agentBinName(tag));
  // Linux releases are musl-only: the static binary runs on gnu and musl hosts alike.
  const platform = process.platform === 'darwin' ? 'macos' : process.platform === 'win32' ? 'windows' : 'linux-musl';
  const arch = process.arch === 'arm64' ? 'aarch64' : 'x86_64';
  const asset = `mlx-agent-${tag}-${platform}-${arch}.tar.gz`;
  const urls = [
    `${MLX_AGENT_R2_BASE}/${tag}/${asset}`,
    `https://github.com/${MLX_AGENT_RELEASE_REPO}/releases/download/${tag}/${asset}`,
  ];

  const expected = await releaseChecksum(tag, asset);
  if (!expected) {
    throw new Error(
      `mlx agent: cannot verify ${asset} — its .sha256 is missing on the GitHub release ` +
        `(${MLX_AGENT_RELEASE_REPO} @ ${tag}). Set ${MLX_AGENT_BIN_ENV}, or run \`yarn build:agent\` ` +
        `in the mlx-node repo.`,
    );
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
    throw new Error(
      `mlx agent: no local binary found and the verified download failed ` +
        `(${asset} unavailable or checksum-mismatched on R2 and GitHub). ` +
        `Set ${MLX_AGENT_BIN_ENV}, or run \`yarn build:agent\` in the mlx-node repo.`,
    );
  }
  const tmp = join(dirname(dest), `${asset}.tmp-${process.pid}`);
  const { chmodSync, renameSync, readdirSync } = await import('node:fs');
  writeFileSync(tmp, Buffer.from(tar), { mode: 0o600 });
  const out = spawnSync('tar', ['-xzf', tmp, '-O'], { encoding: 'buffer', maxBuffer: 256 * 1024 * 1024 });
  rmSync(tmp, { force: true });
  if (out.status !== 0 || out.stdout.length === 0) {
    throw new Error(`mlx agent: downloaded ${asset} but failed to unpack it`);
  }
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
        writeFileSync(staged, out.stdout, { mode: 0o700 });
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
    if (sideInstall) {
      try {
        writeFileSync(`${versioned}.lease-${process.pid}`, '', { flag: 'wx' });
      } catch {
        // lease already exists or dir unwritable — best-effort
      }
    }
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
      for (const entry of entries) {
        const lease = /^(.+)\.lease-(\d+)$/.exec(entry);
        if (!lease) continue;
        if (pidAlive(Number(lease[2]))) {
          keep.add(lease[1]);
        } else {
          // pinner is gone — the lease is dead weight
          rmSync(join(dirname(dest), entry), { force: true });
        }
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
 * The directory advertising lock members. Single-name locks cannot be
 * reclaimed safely — every scheme eventually has to unlink or replace a
 * name a live process may already hold — so the lock is a membership set
 * instead: each contender advertises `b-<ticket>-<pid>-<seq>` (unix
 * socket; marker file on Windows) and the lowest ticket among live
 * members holds, where tickets are bakery numbers (`1 + max(seen)`), not
 * pids — pids wrap, so pid order is not arrival order. Nobody ever removes
 * a name owned by a live pid: a crashed member's marker refuses
 * connections and its pid is dead, so deleting it is unconditional and
 * race-free. The directory sits under tmpdir when `.lock.d` would exceed
 * the ~104-byte socket path cap (member name adds ~24 bytes).
 */
function lockDirPath(path: string): { dir: string; legacy: string | undefined } {
  const dir = `${path}.d`;
  if (process.platform === 'win32') return { dir, legacy: undefined };
  // Budget: `dir/m-<pid>-<seq>` must stay under the ~104-byte sun_path cap
  // (macOS reports over-long socket paths as EINVAL), so an over-long
  // `.lock.d` — a deep $HOME — is hashed into a short tmpdir name.
  if (Buffer.byteLength(dir) < 85) return { dir, legacy: path };
  const hash = createHash('sha256').update(resolve(dir)).digest('hex').slice(0, 16);
  return { dir: join(tmpdir(), `a-${hash}.d`), legacy: path };
}

/** True while something accepts connections on `node` — i.e. a live member. */
function memberListening(node: string): Promise<boolean> {
  return new Promise((res) => {
    const sock = connect(node);
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

interface LockMember {
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

interface MemberScan {
  members: LockMember[];
  /** A contender is mid-choose (its c- marker names a live pid). */
  choosing: boolean;
}

/**
 * A `b-`/`m-`/`c-` marker is swept on a refused socket probe only when it
 * is older than this: the marker file exists between bind() and listen()
 * completing, so a fresh marker that refuses is a member mid-bind, and
 * unlinking it would leave that acquirer scanning forever for a name that
 * is gone. Recycled-pid markers are by definition old.
 */
const MEMBER_STALE_MS = 2_000;

/**
 * Live-and-present state of `dir`. Marker kinds:
 *   `b-<ticket>-<pid>-<seq>` — a bakery member (socket on unix, file on win).
 *   `m-<pid>-<seq>`          — a member from this lock's previous design;
 *                              honored as ticket 0 (its owner advertised
 *                              before any bakery member).
 *   `c-<pid>-<seq>`          — a contender between choosing a ticket and
 *                              advertising it (socket on unix, file on
 *                              win); live ones force a rescan.
 * Anything whose pid is dead is swept unconditionally — the owner cannot
 * exist. On unix a marker ALSO has to answer on its socket path: a live
 * pid with a dead socket is a recycled pid fronting a stale marker, and
 * honoring it would wait out every acquire behind a lock nobody owns. Age
 * alone never sweeps a live marker — a suspended contender is
 * indistinguishable from a recycled pid by liveness, and removing a LIVE
 * `c-` lets a later contender choose the same ticket and enter
 * concurrently. On Windows the markers are plain files and pid liveness
 * stays authoritative (the recycled-pid stall is a known residual risk
 * there).
 */
async function scanMembers(dir: string): Promise<MemberScan> {
  let entries: string[];
  try {
    entries = readdirSync(dir);
  } catch {
    return { members: [], choosing: false };
  }
  const members: LockMember[] = [];
  const pending: { member: LockMember; choosing: boolean }[] = [];
  for (const entry of entries) {
    let m = /^b-(\d+)-(\d+)-(\d+)$/.exec(entry);
    if (m) {
      const member = { ticket: Number(m[1]), pid: Number(m[2]), seq: Number(m[3]), file: join(dir, entry) };
      if (pidAlive(member.pid)) pending.push({ member, choosing: false });
      else rmSync(member.file, { force: true });
      continue;
    }
    m = /^m-(\d+)-(\d+)$/.exec(entry);
    if (m) {
      const member = { ticket: 0, pid: Number(m[1]), seq: Number(m[2]), file: join(dir, entry) };
      if (pidAlive(member.pid)) pending.push({ member, choosing: false });
      else rmSync(member.file, { force: true });
      continue;
    }
    m = /^c-(\d+)-(\d+)$/.exec(entry);
    if (m) {
      const member = { ticket: -1, pid: Number(m[1]), seq: Number(m[2]), file: join(dir, entry) };
      if (pidAlive(member.pid)) pending.push({ member, choosing: true });
      else rmSync(join(dir, entry), { force: true });
    }
  }
  // Probe live-pid markers in parallel: the socket check is the cheap
  // ECONNREFUSED path for recycled pids, so the extra round-trip per
  // marker is only paid when markers exist. A refused marker is swept only
  // once it is too old to be a member still finishing bind()+listen().
  const sockets = await Promise.all(pending.map((p) => memberListening(p.member.file)));
  let choosing = false;
  pending.forEach(({ member, choosing: isChoosing }, i) => {
    let fresh = false;
    if (process.platform !== 'win32' && !sockets[i]) {
      try {
        fresh = Date.now() - statSync(member.file).mtimeMs <= MEMBER_STALE_MS;
      } catch {
        // vanished mid-probe — nothing to sweep or keep
        return;
      }
    }
    if (process.platform !== 'win32' && !sockets[i] && !fresh) {
      rmSync(member.file, { force: true });
      return;
    }
    if (isChoosing) choosing = true;
    else members.push(member);
  });
  members.sort((a, b) => a.ticket - b.ticket || a.pid - b.pid || a.seq - b.seq);
  return { members, choosing };
}

/**
 * Serialize the install critical section across processes. Bakery
 * protocol: advertise `c-<pid>-<seq>` while choosing, take ticket
 * `1 + max(seen)`, then bind `b-<ticket>-<pid>-<seq>` and wait until ours
 * is the lowest live ticket. Arrival order is preserved because any
 * member advertised after us necessarily saw our ticket when choosing;
 * a live `c-` marker forces waiters to rescan so a mid-choose contender
 * cannot be leapfrogged by a stale max. A dead member's marker is deleted
 * by the next scan unconditionally — no liveness probe is trusted to
 * unlink anything.
 *
 * On Unix each marker — choose and member — is a listening unix socket so
 * probes land harmlessly and a suspended contender stays live while a
 * recycled pid fails the probe; on Windows they are empty marker files
 * plus nothing to probe — pid liveness is authoritative since named-pipe
 * names die with the process.
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
  const choosing = join(dir, `c-${process.pid}-${seq}`);
  let choosingServer: Server | undefined;
  if (process.platform === 'win32') {
    try {
      writeFileSync(choosing, '', { flag: 'wx' });
    } catch {
      // left over from a crashed choose — reuse it
    }
  } else {
    // A unix marker must answer on its socket path, or a live contender
    // suspended mid-choose gets swept like a recycled pid and a later
    // contender can choose the same ticket and enter concurrently.
    choosingServer = createServer((sock) => sock.destroy());
    choosingServer.on('error', () => {});
    await new Promise<void>((res, rej) => {
      choosingServer!.once('error', rej);
      choosingServer!.listen(choosing, res);
    }).catch(async (error) => {
      choosingServer!.close();
      choosingServer = undefined;
      if ((error as NodeJS.ErrnoException).code === 'EADDRINUSE') {
        // Stale socket from a crashed choose under our recycled pid —
        // remove it and retry once, like the member bind below.
        rmSync(choosing, { force: true });
        choosingServer = createServer((sock) => sock.destroy());
        choosingServer.on('error', () => {});
        await new Promise<void>((res, rej) => {
          choosingServer!.once('error', rej);
          choosingServer!.listen(choosing, res);
        });
        return;
      }
      throw error;
    });
  }
  const seen = await scanMembers(dir);
  const ticket = seen.members.reduce((max, m) => Math.max(max, m.ticket), 0) + 1;
  const marker = join(dir, `b-${ticket}-${process.pid}-${seq}`);
  let server: Server | undefined;
  if (process.platform === 'win32') {
    // Marker file only: there is no shared name to steal, and the pipe is
    // unnecessary since pid liveness is authoritative on Windows.
    try {
      writeFileSync(marker, '', { flag: 'wx' });
    } catch {
      // exists from a stale scan race — reuse is fine, ticket is per-acquire
    }
  } else {
    server = createServer((sock) => sock.destroy());
    server.on('error', () => {}); // stray runtime errors must not crash the holder
    await new Promise<void>((res, rej) => {
      server!.once('error', rej);
      server!.listen(marker, res);
    }).catch(async (error) => {
      // EADDRINUSE means a dead-pid marker survived a scan — remove and
      // retry once; anything else fails the acquire outright.
      server!.close();
      if ((error as NodeJS.ErrnoException).code === 'EADDRINUSE') {
        rmSync(marker, { force: true });
        server = createServer((sock) => sock.destroy());
        server.on('error', () => {});
        await new Promise<void>((res, rej) => {
          server!.once('error', rej);
          server!.listen(marker, res);
        });
        return;
      }
      throw error;
    });
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
      // choose phase above. Our OWN marker must be in the scan: a unix
      // member suspended past MEMBER_STALE_MS between node creation and
      // listen gets swept as aged/refused, and resuming to an every()
      // that contains only higher tickets would grant the lock a second
      // time while another process holds it. Missing own marker → wait
      // out the deadline rather than enter.
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
}

/** Injectable seams so tests never touch the network or $HOME. */
export interface AgentUpdateDeps {
  resolveTag?: () => Promise<string>;
  probe?: () => AgentInstallProbe;
  managedPath?: string;
  download?: (dest: string, tag: string) => Promise<string>;
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
  const download = deps.download ?? downloadAgentBinary;
  await download(status.managedPath, status.latest);
  return { status: { ...status, current: status.latest, updateAvailable: false }, downloaded: true };
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

  const url = await waitForServe(child, opts.serveReadyTimeoutMs ?? 30_000);
  return { child, url, token };
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
    rl?.on('line', (line) => {
      const match = line.match(/listening on (http:\/\/\S+)/);
      if (!match) return;
      const url = `${match[1]!}/v1`;
      // Health check: the banner prints before the listener is guaranteed
      // accept-ready under every platform, so confirm once.
      fetch(`${match[1]}/health`)
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

  const code = await new Promise<number>((resolvePromise) => {
    agent.once('exit', (c, signal) => resolvePromise(c ?? (signal === 'SIGINT' ? 130 : 1)));
    // Parent Ctrl+C forwards to the TUI; serve dies with us.
    const forward = (): void => {
      agent.kill('SIGINT');
    };
    process.once('SIGINT', forward);
    agent.once('exit', () => process.removeListener('SIGINT', forward));
  });

  serve?.child?.kill('SIGTERM');
  return code;
}
