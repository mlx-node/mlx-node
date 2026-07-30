/**
 * The built INFERENCE entry, forked as a real process, talking to a real
 * `InferenceHost` over the real wire protocol.
 *
 * `sidecar.test.ts` drives `runSidecar` against a stub host, which proves the
 * rules but cannot prove the one thing the entry is for: that
 * `createInferenceHost({ port: 0 })` reports the port it bound and that the
 * addon loads at all in a forked child. Both are only observable in a process
 * that actually did it.
 *
 * Skipped when the addon or the build is absent — both are gitignored and built
 * out of band, so "missing" is the normal state of a fresh checkout.
 */

import { fork } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { afterEach, describe, expect, it } from 'vite-plus/test';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
const ENTRY = resolve(ROOT, 'packages/desktop/dist/inference/index.js');
const ADDON = resolve(ROOT, 'packages/core/mlx-core.darwin-arm64.node');
const RUNNABLE = existsSync(ENTRY) && existsSync(ADDON);

const dirs: string[] = [];
const kids: { kill(signal?: NodeJS.Signals): void }[] = [];

afterEach(() => {
  for (const child of kids.splice(0)) child.kill('SIGKILL');
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

/** A models root with one entry that `detectModelType` recognises but nothing loads. */
function fakeModelsDir(): { home: string; modelsDir: string } {
  const home = mkdtempSync(join(tmpdir(), 'mlx-sidecar-e2e-'));
  dirs.push(home);
  const modelsDir = join(home, 'models');
  mkdirSync(join(modelsDir, 'tiny-qwen3'), { recursive: true });
  writeFileSync(join(modelsDir, 'tiny-qwen3', 'config.json'), JSON.stringify({ model_type: 'qwen3' }));
  return { home, modelsDir };
}

interface Live {
  send(payload: unknown, id: number): Promise<{ ok: boolean; value?: unknown; error?: string }>;
  url: string;
  stderr: string[];
  exit: Promise<{ code: number | null; signal: string | null }>;
  kill(signal: NodeJS.Signals): void;
}

async function startSidecar(env: Record<string, string> = {}): Promise<Live> {
  const { home, modelsDir } = fakeModelsDir();
  const child = fork(ENTRY, [], {
    env: {
      ...process.env,
      // The child must not write a response store into the developer's real
      // ~/.mlx-node; `resolveMlxNodeHome()` is `$HOME/.mlx-node`.
      HOME: home,
      MLX_MODELS_DIR: modelsDir,
      // The whole point: the FILE, not its directory. See `child-env.ts`.
      NAPI_RS_NATIVE_LIBRARY_PATH: ADDON,
      ...env,
    },
    stdio: ['ignore', 'pipe', 'pipe', 'ipc'],
    serialization: 'advanced',
  });
  kids.push(child);

  const stderr: string[] = [];
  child.stderr?.on('data', (chunk: Buffer) => stderr.push(chunk.toString('utf8')));

  const exit = new Promise<{ code: number | null; signal: string | null }>((res) => {
    child.on('exit', (code, signal) => res({ code, signal }));
  });

  const url = await new Promise<string>((res, rej) => {
    const timer = setTimeout(() => rej(new Error(`no mlx:ready in 90s; stderr: ${stderr.join('')}`)), 90_000);
    child.on('message', (message: { kind?: string; url?: string }) => {
      if (message?.kind !== 'mlx:ready') return;
      clearTimeout(timer);
      res(message.url as string);
    });
    void exit.then(({ code }) => {
      clearTimeout(timer);
      rej(new Error(`sidecar exited ${code} before announcing; stderr: ${stderr.join('')}`));
    });
  });

  return {
    url,
    stderr,
    exit,
    kill: (signal) => child.kill(signal),
    send(payload, id) {
      return new Promise((res, rej) => {
        const timer = setTimeout(() => rej(new Error(`request ${id} never answered`)), 30_000);
        const onMessage = (message: { kind?: string; id?: number; ok?: boolean }): void => {
          if (message?.kind !== 'mlx:response' || message.id !== id) return;
          clearTimeout(timer);
          child.off('message', onMessage);
          res(message as { ok: boolean });
        };
        child.on('message', onMessage);
        child.send({ kind: 'mlx:request', id, payload });
      });
    },
  };
}

describe.skipIf(!RUNNABLE)('the built sidecar entry', () => {
  it('announces the port it actually bound', { timeout: 120_000 }, async () => {
    const live = await startSidecar();

    // The socket is the authority. Reaching /health on the announced URL is the
    // only evidence that the handshake is honest — the URL is a string until
    // something connects to it.
    const response = await fetch(new URL('/health', live.url));
    expect(response.ok).toBe(true);
    const health = (await response.json()) as { status: string };
    expect(typeof health.status).toBe('string');

    // …and it is THIS process answering, not something else that happens to be
    // on that port.
    const info = await live.send({ op: 'info' }, 1);
    expect(info.ok).toBe(true);
    expect((info.value as { url: string }).url).toBe(live.url);
    expect(new URL(live.url).port).toBe(String((info.value as { port: number }).port));
  });

  it('serves the models it discovered', { timeout: 120_000 }, async () => {
    const live = await startSidecar();
    const models = await live.send({ op: 'models' }, 1);
    expect(models.ok).toBe(true);
    expect(models.value).toEqual([expect.objectContaining({ name: 'tiny-qwen3', modelType: 'qwen3' }) as unknown]);
  });

  it('runs close() and exits 0 on SIGTERM', { timeout: 120_000 }, async () => {
    const live = await startSidecar();
    live.kill('SIGTERM');
    const { code, signal } = await live.exit;

    // Exit 0 with NO signal is the case the supervisor's intent tracking exists
    // for: `utilityProcess.kill()` reports exactly this for a stop it asked for
    // AND for a sidecar that fell off the end of its entry. A sidecar that let
    // the default SIGTERM disposition kill it would report `signal: 'SIGTERM'`
    // here, and its host's temp root would be left behind.
    expect({ code, signal }).toEqual({ code: 0, signal: null });

    // The socket is gone, not merely unreferenced.
    await expect(fetch(new URL('/health', live.url), { signal: AbortSignal.timeout(2_000) })).rejects.toThrow();
  });

  it('exits fast, with the reason on stderr, when there are no models', { timeout: 120_000 }, async () => {
    const empty = mkdtempSync(join(tmpdir(), 'mlx-sidecar-empty-'));
    dirs.push(empty);
    const child = fork(ENTRY, [], {
      env: { ...process.env, HOME: empty, MLX_MODELS_DIR: join(empty, 'models'), NAPI_RS_NATIVE_LIBRARY_PATH: ADDON },
      stdio: ['ignore', 'pipe', 'pipe', 'ipc'],
      serialization: 'advanced',
    });
    kids.push(child);
    const stderr: string[] = [];
    child.stderr?.on('data', (chunk: Buffer) => stderr.push(chunk.toString('utf8')));

    const code = await new Promise<number | null>((res) => child.on('exit', res));
    // 78, not 1: a crash report can tell "nothing to serve" from a throw.
    expect(code).toBe(78);
    expect(stderr.join('')).toContain('No models discovered');
  });
});
