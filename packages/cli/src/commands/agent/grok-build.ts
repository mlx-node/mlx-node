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
 * except the agent binary, which takes over the terminal like pi did.
 */

import { spawn, spawnSync, type ChildProcess, type SpawnOptions } from 'node:child_process';
import { createInterface } from 'node:readline';
import { createHash, randomBytes } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

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

/**
 * Locate the agent binary, in priority order:
 *   1. `$MLX_AGENT_BIN` — developer / vendored-build override.
 *   2. `<repo>/grok-build/target/release/mlx-agent` — `yarn build:agent` output.
 *   3. `…/Contents/Resources/native/mlx-agent` — bundled inside the packaged app.
 *   4. `~/.mlx-node/bin/mlx-agent` — previously downloaded prebuilt.
 *   5. Download a prebuilt release tarball into (4), then use it.
 */
export async function resolveAgentBinary(repoRoot = findRepoRoot()): Promise<string> {
  const envBin = process.env[MLX_AGENT_BIN_ENV];
  if (envBin) {
    const path = resolve(envBin);
    if (existsSync(path)) return path;
    throw new Error(`${MLX_AGENT_BIN_ENV}=${envBin} does not exist`);
  }

  const vendored = repoRoot ? join(repoRoot, 'grok-build', 'target', 'release', 'mlx-agent') : undefined;
  if (vendored && existsSync(vendored)) return vendored;

  const bundled = findBundledAppBinary();
  if (bundled) return bundled;

  // The cached binary is valid only while it satisfies this CLI's minimum
  // release: its marker records which tag produced it. An unmarked binary is
  // treated as stale — cheaper than trusting whatever was downloaded before
  // markers existed.
  const installed = join(homedir(), '.mlx-node', 'bin', 'mlx-agent');
  if (existsSync(installed)) {
    try {
      const cachedTag = readFileSync(`${installed}.tag`, 'utf-8').trim();
      if (compareReleaseTags(cachedTag, MLX_AGENT_MIN_RELEASE) >= 0) return installed;
    } catch {
      // no marker — fall through to re-download
    }
  }

  return downloadAgentBinary(installed);
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

/**
 * Compare `mlx-agent-vX.Y.Z` tags by version triplet (non-triplet tails sort
 * lexically). Returns <0, 0, >0.
 */
export function compareReleaseTags(a: string, b: string): number {
  const parse = (t: string): number[] => {
    const m = t.match(/^mlx-agent-v(\d+)\.(\d+)\.(\d+)/);
    return m ? [Number(m[1]), Number(m[2]), Number(m[3])] : [-1];
  };
  const pa = parse(a);
  const pb = parse(b);
  for (let i = 0; i < 3; i++) {
    const d = (pa[i] ?? -1) - (pb[i] ?? -1);
    if (d !== 0) return d;
  }
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
      if (/^mlx-agent-v\d/.test(tag)) return tag;
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
      if (tag && /^mlx-agent-v\d/.test(tag)) return tag;
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

/**
 * Download `mlx-agent-<tag>-<platform>-<arch>.tar.gz` and verify its SHA256
 * against the release-published checksum before extracting — the desktop app
 * SIGNS these bytes, so a tampered mirror must never reach the bundle.
 * R2 mirror first, GitHub release as fallback.
 */
export async function downloadAgentBinary(dest: string, tag?: string): Promise<string> {
  tag ??= await resolveReleaseTag();
  // Linux releases are musl-only: the static binary runs on gnu and musl hosts alike.
  const platform =
    process.platform === 'darwin' ? 'macos' : process.platform === 'win32' ? 'windows' : 'linux-musl';
  const arch = process.arch === 'arm64' ? 'aarch64' : 'x86_64';
  const asset = `mlx-agent-${tag}-${platform}-${arch}.tar.gz`;
  const urls = [
    `${MLX_AGENT_R2_BASE}/${tag}/${asset}`,
    `https://github.com/${MLX_AGENT_RELEASE_REPO}/releases/download/${tag}/${asset}`,
  ];

  mkdirSync(dirname(dest), { recursive: true, mode: 0o700 });
  let tar: ArrayBuffer | null = null;
  let shaUrl = '';
  for (const url of urls) {
    // Pull the checksum from the same mirror — a per-release asset, so a
    // mirror that can serve a tampered tarball could also serve a tampered
    // checksum ONLY if the trusted side (the release pipeline's publish)
    // never wrote the real one. Both are produced by the same CI step.
    const expected = await fetchText(`${url}.sha256`);
    if (!expected) continue;
    try {
      const r = await fetch(url, { redirect: 'follow' });
      if (!r.ok || r.body === null) continue;
      const buf = await r.arrayBuffer();
      const digest = createHash('sha256').update(Buffer.from(buf)).digest('hex');
      if (digest !== expected) continue;
      tar = buf;
      shaUrl = url;
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
  const { writeFileSync, rmSync, chmodSync } = await import('node:fs');
  writeFileSync(tmp, Buffer.from(tar), { mode: 0o600 });
  const out = spawnSync('tar', ['-xzf', tmp, '-O'], { encoding: 'buffer', maxBuffer: 256 * 1024 * 1024 });
  rmSync(tmp, { force: true });
  if (out.status !== 0 || out.stdout.length === 0) {
    throw new Error(`mlx agent: downloaded ${asset} but failed to unpack it`);
  }
  writeFileSync(dest, out.stdout, { mode: 0o700 });
  chmodSync(dest, 0o700);
  // Record which release produced this binary — the marker is what
  // resolveAgentBinary checks against the CLI's minimum on the next launch.
  writeFileSync(`${dest}.tag`, tag, { mode: 0o600 });
  return dest;
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
  const oneShot = opts.argv.includes('--version') || opts.argv.includes('-v') || opts.argv.includes('--help') || opts.argv.includes('-h');
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
