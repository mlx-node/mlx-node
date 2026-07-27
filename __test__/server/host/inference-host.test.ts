import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdir, mkdtemp, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import type { LoadableModel } from '@mlx-node/lm';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { isServingStatus } from '../../../packages/desktop/src/main/supervisor/state.js';
import {
  createInferenceHost,
  InsecureBindError,
  LAUNCHER_ENGINE_POLICY,
  ModelNotFoundError,
  NoModelsDiscoveredError,
  type InferenceHost,
  type InferenceHostOptions,
} from '../../../packages/server/src/host/index.js';

/**
 * `createInferenceHost` is the piece three front-ends share — `mlx serve`,
 * `mlx launch claude`, and the desktop app's Electron `utilityProcess` — so
 * the invariants that matter here are the ones a supervisor cannot see and
 * cannot fix: temp roots reclaimed on every exit path, engine env written
 * exactly when asked and never otherwise, and out-of-band loads taking the
 * same allocator brackets the HTTP path takes.
 *
 * No native weights are ever materialized: `loadModel` is injected. The paged
 * override machinery IS real, because the thing being pinned is precisely
 * whether its temp clones survive.
 */

const CHUNK_ENV = 'MLX_PAGED_PREFILL_CHUNK_SIZE';

let hosts: InferenceHost[] = [];
let scratchDirs: string[] = [];
const originalChunkEnv = process.env[CHUNK_ENV];
const originalAnthropicModel = process.env.ANTHROPIC_MODEL;

afterEach(async () => {
  const openHosts = hosts;
  hosts = [];
  for (const host of openHosts) await host.close({ timeoutMs: 250 }).catch(() => undefined);

  const dirs = scratchDirs;
  scratchDirs = [];
  for (const dir of dirs) await rm(dir, { recursive: true, force: true }).catch(() => undefined);

  if (originalChunkEnv === undefined) delete process.env[CHUNK_ENV];
  else process.env[CHUNK_ENV] = originalChunkEnv;
  if (originalAnthropicModel === undefined) delete process.env.ANTHROPIC_MODEL;
  else process.env.ANTHROPIC_MODEL = originalAnthropicModel;
});

/** A models dir whose entries `discoverModels` accepts without reading weights. */
async function makeModelsDir(names: string[]): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), 'mlx-host-models-'));
  scratchDirs.push(root);
  for (const name of names) {
    const dir = join(root, name);
    await mkdir(dir, { recursive: true });
    await writeFile(join(dir, 'config.json'), JSON.stringify({ model_type: 'qwen3_5' }), 'utf-8');
    await writeFile(join(dir, 'model.safetensors'), 'not-real-weights', 'utf-8');
  }
  return root;
}

/** Stand-in for a materialized native model. Nothing here is ever invoked. */
function fakeModel(): LoadableModel {
  return {} as unknown as LoadableModel;
}

async function start(opts: InferenceHostOptions): Promise<InferenceHost> {
  const host = await createInferenceHost({
    port: 0,
    disableStore: true,
    // The startup sweep touches the real OS temp dir; the one test that wants
    // it opts back in explicitly.
    sweepOrphanTempRoots: false,
    loadModel: async () => fakeModel(),
    ...opts,
  });
  hosts.push(host);
  return host;
}

/** Temp roots this process' host machinery owns, by the pid-scoped name. */
async function ownTempRoots(): Promise<string[]> {
  const prefix = `mlx-inference-host-${process.pid}-`;
  return (await readdir(tmpdir())).filter((name) => name.startsWith(prefix)).sort();
}

/** Roots that appeared since `before` — isolates a test from its neighbours. */
async function newTempRoots(before: string[]): Promise<string[]> {
  const now = await ownTempRoots();
  return now.filter((name) => !before.includes(name));
}

describe('createInferenceHost — discovery and model binding', () => {
  it('binds the alphabetically-first model by default', async () => {
    const modelsDir = await makeModelsDir(['zeta', 'alpha', 'mid']);
    delete process.env.ANTHROPIC_MODEL;
    const host = await start({ modelsDir });
    expect(host.models.map((m) => m.name)).toEqual(['alpha', 'mid', 'zeta']);
    expect(host.boundModel).toBe('alpha');
    expect(host.modelsDir).toBe(modelsDir);
  });

  it('prefers ANTHROPIC_MODEL over the alphabetical default', async () => {
    const modelsDir = await makeModelsDir(['alpha', 'zeta']);
    process.env.ANTHROPIC_MODEL = 'zeta';
    const host = await start({ modelsDir });
    expect(host.boundModel).toBe('zeta');
  });

  it('prefers the explicit option over ANTHROPIC_MODEL', async () => {
    const modelsDir = await makeModelsDir(['alpha', 'zeta']);
    process.env.ANTHROPIC_MODEL = 'zeta';
    const host = await start({ modelsDir, model: 'alpha' });
    expect(host.boundModel).toBe('alpha');
  });

  it('throws NoModelsDiscoveredError for an empty models dir', async () => {
    const modelsDir = await mkdtemp(join(tmpdir(), 'mlx-host-empty-'));
    scratchDirs.push(modelsDir);
    await expect(start({ modelsDir })).rejects.toBeInstanceOf(NoModelsDiscoveredError);
  });

  it('throws ModelNotFoundError rather than silently falling back', async () => {
    const modelsDir = await makeModelsDir(['alpha']);
    // Silently binding `alpha` here would look like success and then serve the
    // wrong model for the whole session.
    const err = await start({ modelsDir, model: 'nope' }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ModelNotFoundError);
    expect((err as ModelNotFoundError).available).toEqual(['alpha']);
  });

  it('rejects a bogus ANTHROPIC_MODEL too, not just a bogus option', async () => {
    const modelsDir = await makeModelsDir(['alpha']);
    process.env.ANTHROPIC_MODEL = 'ghost';
    await expect(start({ modelsDir })).rejects.toBeInstanceOf(ModelNotFoundError);
  });
});

describe('createInferenceHost — binding and health', () => {
  it('reports the port the kernel actually assigned for port 0', async () => {
    const modelsDir = await makeModelsDir(['alpha']);
    const host = await start({ modelsDir, port: 0 });
    expect(host.port).toBeGreaterThan(0);
    expect(host.url).toBe(`http://127.0.0.1:${host.port}`);
  });

  it('answers GET /health on the advertised URL', async () => {
    const modelsDir = await makeModelsDir(['alpha']);
    const host = await start({ modelsDir });
    const res = await fetch(`${host.url}/health`);
    expect(res.status).toBe(200);
    const body = (await res.json()) as { status: string; pid: number };
    expect(body.status).toBe('ok');
    expect(body.pid).toBe(process.pid);
    expect(host.health().status).toBe('ok');
  });

  it('stops answering once closed', async () => {
    const modelsDir = await makeModelsDir(['alpha']);
    const host = await start({ modelsDir });
    const url = `${host.url}/health`;
    expect((await fetch(url)).status).toBe(200);
    await host.close({ timeoutMs: 0 });
    await expect(fetch(url)).rejects.toThrow();
  });
});

describe('createInferenceHost — a network-reachable bind must be gated', () => {
  /**
   * Every route but the `/health` carve-out runs inference, so an
   * unauthenticated bind on a routable interface hands the GPU, the RAM and
   * the on-disk model list to anyone who can route to this machine. There is
   * no serving posture that makes that acceptable, so it fails at startup.
   */

  const AUTH_ENV = 'MLX_SERVER_AUTH_TOKEN';

  afterEach(() => {
    delete process.env[AUTH_ENV];
  });

  it('refuses a wildcard bind with no token, and listens on nothing', async () => {
    const modelsDir = await makeModelsDir(['alpha']);
    for (const host of ['0.0.0.0', '::']) {
      await expect(start({ modelsDir, host, port: 0 }), host).rejects.toBeInstanceOf(InsecureBindError);
    }
    // `hostUrl` advertises a wildcard bind as loopback, so the refusal has to
    // come from the BIND address; a check on the advertised URL would pass.
    await expect(start({ modelsDir, host: '192.168.7.7', port: 0 })).rejects.toBeInstanceOf(InsecureBindError);
  });

  it('names the fix in the message — this is the only signal the operator gets', async () => {
    const modelsDir = await makeModelsDir(['alpha']);
    await expect(start({ modelsDir, host: '0.0.0.0' })).rejects.toThrow(/auth token|MLX_SERVER_AUTH_TOKEN/);
  });

  it('refuses BEFORE mutating anything outside the call', async () => {
    // The engine policy writes `process.env`, and the temp sweep and discovery
    // both touch the filesystem. A guard placed after them leaves the refused
    // start's side effects behind — and `MLX_PAGED_PREFILL_CHUNK_SIZE` latches
    // in the native layer through a `OnceLock`, so it cannot be taken back.
    delete process.env[CHUNK_ENV];
    await expect(
      start({ modelsDir: '/definitely/not/a/models/dir', host: '0.0.0.0', enginePolicy: LAUNCHER_ENGINE_POLICY }),
    ).rejects.toBeInstanceOf(InsecureBindError);
    expect(process.env[CHUNK_ENV]).toBeUndefined();
  });

  it('allows a non-loopback bind once a token is configured', async () => {
    const modelsDir = await makeModelsDir(['alpha']);
    const host = await start({ modelsDir, host: '0.0.0.0', port: 0, authToken: 'a-real-secret' });
    expect(host.port).toBeGreaterThan(0);
    expect((await fetch(`${host.url}/v1/models`, { headers: { 'x-api-key': 'a-real-secret' } })).status).toBe(200);
    expect((await fetch(`${host.url}/v1/models`)).status).toBe(401);
  });

  it('accepts the token from MLX_SERVER_AUTH_TOKEN, which the handler also honours', async () => {
    // Refusing here would be the guard disagreeing with the gate it is
    // guarding: `createServer` picks the env var up, so the server WOULD be
    // protected. Two copies of the "is there a token" rule is how that drifts.
    process.env[AUTH_ENV] = 'from-the-environment';
    const modelsDir = await makeModelsDir(['alpha']);
    const host = await start({ modelsDir, host: '0.0.0.0', port: 0 });
    expect((await fetch(`${host.url}/v1/models`)).status).toBe(401);
  });

  it('ignores an EMPTY MLX_SERVER_AUTH_TOKEN, which enables nothing', async () => {
    // `MLX_SERVER_AUTH_TOKEN=` in a launcher script does not configure auth —
    // treating it as a token here would open the bind it is meant to gate.
    process.env[AUTH_ENV] = '';
    const modelsDir = await makeModelsDir(['alpha']);
    await expect(start({ modelsDir, host: '0.0.0.0' })).rejects.toBeInstanceOf(InsecureBindError);
  });

  it('leaves an unauthenticated loopback bind exactly as it was', async () => {
    const modelsDir = await makeModelsDir(['alpha']);
    for (const bind of ['127.0.0.1', 'localhost', undefined]) {
      const host = await start({ modelsDir, host: bind, port: 0 });
      expect((await fetch(`${host.url}/v1/models`)).status, String(bind)).toBe(200);
    }
  });
});

describe('createInferenceHost — a tokenized host still answers the supervisor', () => {
  /**
   * The desktop sidecar now binds with a per-launch token. The supervisor
   * polls `/health` with NO credential and reads `response.ok` plus
   * `body.status`; if either changed, a healthy sidecar would never reach
   * `running` and the restart budget would crash-loop it. Asserted against the
   * supervisor's own predicate rather than a restated list of statuses.
   */

  it('serves an anonymous /health that the supervisor reads as serving', async () => {
    const modelsDir = await makeModelsDir(['alpha']);
    const host = await start({ modelsDir, authToken: 'per-launch-secret' });

    const res = await fetch(`${host.url}/health`);
    expect(res.ok).toBe(true);
    const body = (await res.json()) as { status: string };
    expect(isServingStatus(body.status)).toBe(true);
  });

  it('withholds the model names an anonymous poll must not see', async () => {
    const modelsDir = await makeModelsDir(['a-client-project-name']);
    const host = await start({ modelsDir, authToken: 'per-launch-secret' });
    await host.loadModel('a-client-project-name');

    expect(await (await fetch(`${host.url}/health`)).text()).not.toContain('a-client-project-name');
    const authed = await (await fetch(`${host.url}/health`, { headers: { 'x-api-key': 'per-launch-secret' } })).json();
    expect((authed as { models: { resident: string[] } }).models.resident).toContain('a-client-project-name');
  });

  it('gates the generative endpoints the supervisor never touches', async () => {
    const modelsDir = await makeModelsDir(['alpha']);
    const host = await start({ modelsDir, authToken: 'per-launch-secret' });

    const anonymous = await fetch(`${host.url}/v1/messages`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: 'alpha', messages: [{ role: 'user', content: 'hi' }], max_tokens: 8 }),
    });
    expect(anonymous.status).toBe(401);
    expect((await fetch(`${host.url}/v1/models`, { headers: { 'x-api-key': 'wrong-secret' } })).status).toBe(401);
    // …and no wildcard for a browser to spend a leaked token through.
    expect(anonymous.headers.get('access-control-allow-origin')).toBeNull();
  });
});

describe('createInferenceHost — out-of-band loadModel', () => {
  it('makes the model resident and records the load on /health', async () => {
    const modelsDir = await makeModelsDir(['alpha', 'beta']);
    const host = await start({ modelsDir });

    expect(host.health().models.resident).toEqual([]);
    expect(host.health().lastLoad).toBeNull();

    await host.loadModel('beta');

    expect(host.health().models.resident).toContain('beta');
    // `lastLoad` is written only by `ModelWorkCoordinator.withModelLoad`.
    // Asserting it is how we pin that the out-of-band path really took the
    // writer bracket instead of calling the swap controller bare — a bare
    // call would race the process-wide Metal allocator against live
    // inference and leave this null.
    expect(host.health().lastLoad).toMatchObject({ label: 'beta', ok: true, error: null });
  });

  it('swaps rather than accumulating residents', async () => {
    const modelsDir = await makeModelsDir(['alpha', 'beta']);
    const host = await start({ modelsDir });

    await host.loadModel('alpha');
    await host.loadModel('beta');

    // Routing through `ServerInstance.loadModel` (which only ever registers)
    // would leave BOTH here and, in production, both sets of weights in RAM.
    expect(host.health().models.resident).toEqual(['beta']);
  });

  it('rejects an unknown name instead of aliasing it onto the resident', async () => {
    const modelsDir = await makeModelsDir(['alpha']);
    const host = await start({ modelsDir });
    await expect(host.loadModel('ghost')).rejects.toBeInstanceOf(ModelNotFoundError);
    // The swap controller WOULD alias an unknown name (that is right for
    // Claude Code's hardcoded `claude-haiku-*`); a supervisor asking for a
    // specific model must get an error, not a silent substitution.
    expect(host.health().models.resident).toEqual([]);
  });

  it('surfaces a load failure and records it on /health', async () => {
    const modelsDir = await makeModelsDir(['alpha']);
    const host = await start({
      modelsDir,
      loadModel: async () => {
        throw new Error('weights are gibberish');
      },
    });

    await expect(host.loadModel('alpha')).rejects.toThrow('weights are gibberish');
    expect(host.health().lastLoad).toMatchObject({ label: 'alpha', ok: false });
    expect(host.health().status).toBe('error');
  });
});

describe('createInferenceHost — engine policy', () => {
  it('writes nothing to the environment when no policy is requested', async () => {
    const modelsDir = await makeModelsDir(['alpha']);
    delete process.env[CHUNK_ENV];
    await start({ modelsDir });
    expect(process.env[CHUNK_ENV]).toBeUndefined();
  });

  it('applies the launcher policy when one is requested', async () => {
    const modelsDir = await makeModelsDir(['alpha']);
    delete process.env[CHUNK_ENV];
    await start({ modelsDir, enginePolicy: LAUNCHER_ENGINE_POLICY });
    expect(process.env[CHUNK_ENV]).toBe('2048');
  });

  it('never overrides a value the operator already set', async () => {
    const modelsDir = await makeModelsDir(['alpha']);
    process.env[CHUNK_ENV] = '4096';
    await start({ modelsDir, enginePolicy: LAUNCHER_ENGINE_POLICY });
    expect(process.env[CHUNK_ENV]).toBe('4096');
  });
});

describe('createInferenceHost — paged-override temp roots', () => {
  it('names its temp root after this pid and removes it on close()', async () => {
    const modelsDir = await makeModelsDir(['alpha']);
    const before = await ownTempRoots();
    const host = await start({ modelsDir });

    await host.loadModel('alpha');

    // Both halves matter. A root under the pid-scoped prefix is what makes a
    // killed host's leftovers reclaimable at all; without it the sweep can
    // never match anything.
    expect(await newTempRoots(before)).toHaveLength(1);

    await host.close({ timeoutMs: 0 });
    expect(await newTempRoots(before)).toHaveLength(0);
  });

  it('still reclaims the temp root after a FAILED load', async () => {
    const modelsDir = await makeModelsDir(['alpha']);
    const before = await ownTempRoots();
    const host = await start({
      modelsDir,
      loadModel: async () => {
        throw new Error('boom');
      },
    });

    await expect(host.loadModel('alpha')).rejects.toThrow('boom');
    // The clone is created BEFORE the loader runs, so a failed load leaves a
    // root on disk with no model to show for it — the exact case where a
    // "clean up only what we successfully loaded" shortcut leaks forever.
    expect(await newTempRoots(before)).toHaveLength(1);

    await host.close({ timeoutMs: 0 });
    expect(await newTempRoots(before)).toHaveLength(0);
  });

  it('still reclaims the temp root when the logger fails to flush', async () => {
    const modelsDir = await makeModelsDir(['alpha']);
    const logDir = await mkdtemp(join(tmpdir(), 'mlx-host-log-'));
    scratchDirs.push(logDir);
    const before = await ownTempRoots();

    const host = await start({
      modelsDir,
      logDir,
      attachLogger: () => ({
        logDir,
        close: async () => {
          throw new Error('disk full');
        },
      }),
    });

    await host.loadModel('alpha');
    expect(host.logDir).toBe(logDir);

    // An unguarded `await logger.close()` would reject here and strand both
    // the server socket and the temp root.
    await expect(host.close({ timeoutMs: 0 })).resolves.toBeUndefined();
    expect(await newTempRoots(before)).toHaveLength(0);
  });

  it('is idempotent — a second close() neither throws nor re-runs disposal', async () => {
    const modelsDir = await makeModelsDir(['alpha']);
    const host = await start({ modelsDir });
    await host.loadModel('alpha');
    await host.close({ timeoutMs: 0 });
    await expect(host.close({ timeoutMs: 0 })).resolves.toBeUndefined();
  });

  it('sweeps a dead host root on startup and spares a live one', async () => {
    const modelsDir = await makeModelsDir(['alpha']);
    const deadPid = spawnSync(process.execPath, ['-e', '0'], { stdio: 'ignore' }).pid;
    if (typeof deadPid !== 'number') throw new Error('spawnSync did not report a pid');

    const orphan = join(tmpdir(), `mlx-inference-host-${deadPid}-orphan`);
    const ours = join(tmpdir(), `mlx-inference-host-${process.pid}-live`);
    await mkdir(orphan, { recursive: true });
    await mkdir(ours, { recursive: true });
    scratchDirs.push(orphan, ours);

    await start({ modelsDir, sweepOrphanTempRoots: true });

    expect(existsSync(orphan)).toBe(false);
    expect(existsSync(ours)).toBe(true);
  });

  it('does not sweep when the caller opts out', async () => {
    const modelsDir = await makeModelsDir(['alpha']);
    const deadPid = spawnSync(process.execPath, ['-e', '0'], { stdio: 'ignore' }).pid;
    if (typeof deadPid !== 'number') throw new Error('spawnSync did not report a pid');

    const orphan = join(tmpdir(), `mlx-inference-host-${deadPid}-orphan-optout`);
    await mkdir(orphan, { recursive: true });
    scratchDirs.push(orphan);

    await start({ modelsDir, sweepOrphanTempRoots: false });

    expect(existsSync(orphan)).toBe(true);
  });
});
