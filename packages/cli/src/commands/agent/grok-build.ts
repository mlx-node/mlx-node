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
import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import { homedir } from 'node:os';
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
  /** Headless one-shots (`--version`, `--help`) need no server. */
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

/** Where the prebuilt download lives — the only binary `mlx agent update` manages. */
export function managedAgentBinPath(homeDir: string = homedir()): string {
  return join(homeDir, '.mlx-node', 'bin', 'mlx-agent');
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
): AgentInstallProbe {
  const managed = managedAgentBinPath(homeDir);
  const managedTag = existsSync(managed) ? readAgentTag(managed) : undefined;

  const envBin = env[MLX_AGENT_BIN_ENV];
  if (envBin) {
    const path = resolve(envBin);
    if (existsSync(path)) return { source: 'env', path, managedTag };
    throw new Error(`${MLX_AGENT_BIN_ENV}=${envBin} does not exist`);
  }

  const vendored = repoRoot ? join(repoRoot, 'grok-build', 'target', 'release', 'mlx-agent') : undefined;
  if (vendored && existsSync(vendored)) return { source: 'vendored', path: vendored, managedTag };

  const bundled = findBundledAppBinary();
  const bundledTag = bundled ? readAgentTag(bundled) : undefined;

  // Bundled-vs-cache: the newer release tag wins. A freshly `mlx agent
  // update`d download in ~/.mlx-node/bin must not be shadowed by the older
  // bundled binary, and an unmarked cache never beats a stamped bundle.
  if (bundled) {
    const cacheWins = managedTag !== undefined && compareReleaseTags(managedTag, bundledTag ?? '') > 0;
    return {
      source: cacheWins ? 'managed' : 'bundled',
      path: cacheWins ? managed : bundled,
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
    return { source: 'managed', path: managed, managedTag, effectiveTag: managedTag };
  }
  return { source: 'none', managedTag };
}

/**
 * Locate the agent binary, in priority order:
 *   1. `$MLX_AGENT_BIN` — developer / vendored-build override.
 *   2. `<repo>/grok-build/target/release/mlx-agent` — `yarn build:agent` output.
 *   3. `…/Contents/Resources/native/mlx-agent` — bundled inside the packaged app.
 *   4. `~/.mlx-node/bin/mlx-agent` — previously downloaded prebuilt.
 *   5. Download a prebuilt release tarball into (4), then use it.
 */
export async function resolveAgentBinary(repoRoot = findRepoRoot()): Promise<string> {
  const probe = probeAgentInstall(repoRoot);
  if (probe.path !== undefined) return probe.path;
  return downloadAgentBinary(managedAgentBinPath());
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
      const candidate = join(dir, '..', 'native', 'mlx-agent');
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
  return semver.parse(tag.slice(TAG_PREFIX.length));
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
 * Which release tag to download: `MLX_AGENT_TAG` override, then the R2
 * `latest` pointer (the release pipeline maintains it), then GitHub's
 * /releases/latest, then the baked minimum. Fork releases can ship at any
 * cadence and a CLI rebuild picks them up with no pin bump.
 */
export async function resolveReleaseTag(): Promise<string> {
  const pinned = process.env[MLX_AGENT_TAG_ENV];
  if (pinned) return pinned;

  try {
    const res = await fetch(`${MLX_AGENT_R2_BASE}/${MLX_AGENT_LATEST_KEY}`, {
      cache: 'no-store',
      signal: AbortSignal.timeout(10_000),
    });
    if (res.ok) {
      const tag = (await res.text()).trim();
      // Syntax alone is not enough: a stale or rolled-back pointer can name a
      // release below this CLI's floor. Anything the pipeline publishes is
      // accepted once it clears the minimum.
      if (tagVersion(tag) && compareReleaseTags(tag, MLX_AGENT_MIN_RELEASE) >= 0) return tag;
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
      const tag = ((await res.json()) as { tag_name?: string }).tag_name?.trim();
      if (tag && tagVersion(tag) && compareReleaseTags(tag, MLX_AGENT_MIN_RELEASE) >= 0) return tag;
    }
  } catch {
    // fall through to the baked minimum
  }

  return MLX_AGENT_MIN_RELEASE;
}

async function fetchText(url: string): Promise<string | undefined> {
  try {
    const res = await fetch(url, { redirect: 'follow', signal: AbortSignal.timeout(15_000) });
    if (res.ok) return (await res.text()).trim();
  } catch {
    // caller falls through
  }
  return undefined;
}

/** How long a single archive-body fetch may run before we try the next mirror. */
const ARCHIVE_FETCH_TIMEOUT_MS = 120_000;

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
 */
export async function downloadAgentBinary(dest: string, tag?: string): Promise<string> {
  tag ??= await resolveReleaseTag();
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
  let tar: ArrayBuffer | null = null;
  for (const url of urls) {
    try {
      // Bounded: a mirror that stalls mid-body must not hang the launch or
      // the desktop-release fetch step — the fallback mirror is one try away.
      const r = await fetch(url, {
        redirect: 'follow',
        signal: AbortSignal.timeout(ARCHIVE_FETCH_TIMEOUT_MS),
      });
      if (!r.ok || r.body === null) continue;
      const buf = await r.arrayBuffer();
      const digest = createHash('sha256').update(Buffer.from(buf)).digest('hex');
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
  const { writeFileSync, rmSync, chmodSync, renameSync } = await import('node:fs');
  writeFileSync(tmp, Buffer.from(tar), { mode: 0o600 });
  const out = spawnSync('tar', ['-xzf', tmp, '-O'], { encoding: 'buffer', maxBuffer: 256 * 1024 * 1024 });
  rmSync(tmp, { force: true });
  if (out.status !== 0 || out.stdout.length === 0) {
    throw new Error(`mlx agent: downloaded ${asset} but failed to unpack it`);
  }
  // Atomic install: a sibling temp file renamed over dest. Truncating the
  // live path fails ETXTBSY on Linux while an agent runs and can expose a
  // half-written binary to a concurrent launch everywhere else.
  const staged = join(dirname(dest), `.mlx-agent.tmp-${process.pid}`);
  try {
    writeFileSync(staged, out.stdout, { mode: 0o700 });
    renameSync(staged, dest);
  } catch (error) {
    rmSync(staged, { force: true });
    throw error;
  }
  chmodSync(dest, 0o700);
  // Record which release produced this binary — the marker is what
  // resolveAgentBinary checks against the CLI's minimum on the next launch.
  writeFileSync(`${dest}.tag`, tag, { mode: 0o600 });
  return dest;
}

/** What `mlx agent update --check` reports and the desktop tray consumes. */
export interface AgentUpdateStatus {
  /** Release tag the active binary was built from; null when nothing is installed or stamped. */
  current: string | null;
  /** Newest published release tag. */
  latest: string;
  /** A newer managed download is available. */
  updateAvailable: boolean;
  /** Path `mlx agent update` writes; the managed download cache. */
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
  return {
    current,
    latest,
    updateAvailable: newest === null || compareReleaseTags(latest, newest) > 0,
    managedPath: deps.managedPath ?? managedAgentBinPath(),
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

  // One-shots that exit inside the binary (help/version) skip the server.
  const oneShot =
    opts.argv.includes('--version') ||
    opts.argv.includes('-v') ||
    opts.argv.includes('--help') ||
    opts.argv.includes('-h');
  const needsServer = opts.needsServer !== false && !oneShot;

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
