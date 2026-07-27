import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import type { LoadableModel } from '@mlx-node/lm';
import type { InferenceHost } from '@mlx-node/server/host';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { parseServeArgs, run } from '../../packages/cli/src/commands/serve.js';

/**
 * `mlx serve` is the terminal-visible twin of the desktop app's supervised
 * sidecar. Its whole value is that it is the SAME `createInferenceHost` with
 * stdout attached, so what is pinned here is the wiring: flags in, a
 * listening host out, a signal in, a clean shutdown out.
 */

const CHUNK_ENV = 'MLX_PAGED_PREFILL_CHUNK_SIZE';
const originalChunkEnv = process.env[CHUNK_ENV];

let scratchDirs: string[] = [];
let installedSignalHandlers: { signal: NodeJS.Signals; handler: NodeJS.SignalsListener }[] = [];

afterEach(async () => {
  for (const { signal, handler } of installedSignalHandlers) process.removeListener(signal, handler);
  installedSignalHandlers = [];

  const dirs = scratchDirs;
  scratchDirs = [];
  for (const dir of dirs) await rm(dir, { recursive: true, force: true }).catch(() => undefined);

  if (originalChunkEnv === undefined) delete process.env[CHUNK_ENV];
  else process.env[CHUNK_ENV] = originalChunkEnv;

  vi.restoreAllMocks();
});

async function makeModelsDir(names: string[]): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), 'mlx-serve-models-'));
  scratchDirs.push(root);
  for (const name of names) {
    const dir = join(root, name);
    await mkdir(dir, { recursive: true });
    await writeFile(join(dir, 'config.json'), JSON.stringify({ model_type: 'qwen3_5' }), 'utf-8');
  }
  return root;
}

/** Snapshot the listeners for `signal` so we can find the one `run` adds. */
function listenersOf(signal: NodeJS.Signals): NodeJS.SignalsListener[] {
  return process.listeners(signal) as NodeJS.SignalsListener[];
}

describe('parseServeArgs', () => {
  it('defaults every optional flag to undefined so the host applies its own defaults', () => {
    const args = parseServeArgs([]);
    expect(args).toEqual({
      help: false,
      port: undefined,
      host: undefined,
      modelsDir: undefined,
      model: undefined,
      authToken: undefined,
      logDir: undefined,
    });
  });

  it('maps every documented flag', () => {
    const args = parseServeArgs([
      '--port',
      '8080',
      '--host',
      '0.0.0.0',
      '--models-dir',
      '/models',
      '--model',
      'qwen',
      '--auth-token',
      'secret',
    ]);
    expect(args).toMatchObject({
      port: 8080,
      host: '0.0.0.0',
      modelsDir: '/models',
      model: 'qwen',
      authToken: 'secret',
    });
  });

  it('accepts --port 0 (unlike `mlx launch claude`, which rejects it)', () => {
    // An operator asking for an ephemeral port gets one; the real port is
    // printed. A `<= 0` guard copied from the launcher would break this.
    expect(parseServeArgs(['--port', '0']).port).toBe(0);
  });

  it('rejects a non-integer or out-of-range --port', () => {
    expect(() => parseServeArgs(['--port', 'abc'])).toThrow(/Invalid --port/);
    expect(() => parseServeArgs(['--port', '1.5'])).toThrow(/Invalid --port/);
    // `--port -1` is rejected earlier, by node's own parseArgs ("argument is
    // ambiguous"); the `=` form is what reaches our range check.
    expect(() => parseServeArgs(['--port=-1'])).toThrow(/Invalid --port/);
    expect(() => parseServeArgs(['--port', '70000'])).toThrow(/Invalid --port/);
  });

  it('treats --log-dir as implying --verbose', () => {
    expect(parseServeArgs(['--log-dir', '/tmp/mlx-serve-logs']).logDir).toBe('/tmp/mlx-serve-logs');
  });

  it('leaves logDir unset without --verbose', () => {
    expect(parseServeArgs([]).logDir).toBeUndefined();
  });
});

describe('mlx serve — end to end', () => {
  it('starts on --port 0, answers /health, and shuts down cleanly on SIGINT', async () => {
    const modelsDir = await makeModelsDir(['alpha']);
    delete process.env[CHUNK_ENV];
    vi.spyOn(console, 'log').mockImplementation(() => undefined);

    const exitCodes: number[] = [];
    const sigintBefore = new Set(listenersOf('SIGINT'));
    const sigtermBefore = new Set(listenersOf('SIGTERM'));

    let healthUrl = '';
    let observedUrl = '';

    await run(['--port', '0', '--models-dir', modelsDir], {
      hostOverrides: {
        disableStore: true,
        sweepOrphanTempRoots: false,
        loadModel: async () => ({}) as unknown as LoadableModel,
      },
      exit: (code) => exitCodes.push(code),
      onReady: async (host: InferenceHost) => {
        observedUrl = host.url;
        healthUrl = `${host.url}/health`;

        // The kernel assigned the port; a URL still carrying `:0` means the
        // bound port was never read back off the socket.
        expect(host.port).toBeGreaterThan(0);
        expect(host.url).toBe(`http://127.0.0.1:${host.port}`);

        const res = await fetch(healthUrl);
        expect(res.status).toBe(200);
        const body = (await res.json()) as { status: string; pid: number };
        expect(body.status).toBe('ok');
        expect(body.pid).toBe(process.pid);

        // The launcher engine policy is applied by `serve`, not by the engine.
        expect(process.env[CHUNK_ENV]).toBe('2048');

        // Invoke the handler `run` actually registered rather than emitting a
        // real signal — emitting would also fire the test runner's own
        // handlers and tear the run down.
        const added = listenersOf('SIGINT').filter((l) => !sigintBefore.has(l));
        expect(added).toHaveLength(1);
        added[0]('SIGINT');

        for (let i = 0; i < 200 && exitCodes.length === 0; i++) {
          await new Promise((resolve) => setTimeout(resolve, 10));
        }
      },
    });

    expect(observedUrl).toMatch(/^http:\/\/127\.0\.0\.1:\d+$/);
    expect(exitCodes).toEqual([0]);
    await expect(fetch(healthUrl)).rejects.toThrow();

    // Handlers detached on shutdown: a leaked one would keep a closed host
    // alive in the listener list and swallow the next Ctrl+C.
    expect(listenersOf('SIGINT').filter((l) => !sigintBefore.has(l))).toEqual([]);
    expect(listenersOf('SIGTERM').filter((l) => !sigtermBefore.has(l))).toEqual([]);
  });

  it('reports a models dir with nothing in it instead of starting', async () => {
    const modelsDir = await mkdtemp(join(tmpdir(), 'mlx-serve-empty-'));
    scratchDirs.push(modelsDir);
    const errors: string[] = [];
    vi.spyOn(console, 'error').mockImplementation((line: unknown) => {
      errors.push(String(line));
    });
    const exitSpy = vi.spyOn(process, 'exit').mockImplementation(((): never => {
      throw new Error('__exit__');
    }) as never);

    await expect(run(['--models-dir', modelsDir])).rejects.toThrow('__exit__');
    expect(exitSpy).toHaveBeenCalledWith(1);
    expect(errors.join('\n')).toContain('No models discovered under');
    expect(errors.join('\n')).toContain('mlx download model');
  });

  it('lists the discovered models when --model names something absent', async () => {
    const modelsDir = await makeModelsDir(['alpha', 'beta']);
    const errors: string[] = [];
    vi.spyOn(console, 'error').mockImplementation((line: unknown) => {
      errors.push(String(line));
    });
    const exitSpy = vi.spyOn(process, 'exit').mockImplementation(((): never => {
      throw new Error('__exit__');
    }) as never);

    await expect(run(['--models-dir', modelsDir, '--model', 'ghost'])).rejects.toThrow('__exit__');
    expect(exitSpy).toHaveBeenCalledWith(1);
    expect(errors.join('\n')).toContain('Model "ghost" not found');
    expect(errors.join('\n')).toContain('  - alpha');
    expect(errors.join('\n')).toContain('  - beta');
  });
});
