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
const PERSIST_ENV = 'MLX_PERSIST_PAGED_CACHE';
const originalPersistEnv = process.env[PERSIST_ENV];
const BLOCKS_ENV = 'MLX_COLD_CAPTURE_BLOCKS_PER_TURN';
const BUDGET_ENV = 'MLX_COLD_CAPTURE_BUDGET_MS';
const originalBlocksEnv = process.env[BLOCKS_ENV];
const originalBudgetEnv = process.env[BUDGET_ENV];

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
  if (originalPersistEnv === undefined) delete process.env[PERSIST_ENV];
  else process.env[PERSIST_ENV] = originalPersistEnv;
  if (originalBlocksEnv === undefined) delete process.env[BLOCKS_ENV];
  else process.env[BLOCKS_ENV] = originalBlocksEnv;
  if (originalBudgetEnv === undefined) delete process.env[BUDGET_ENV];
  else process.env[BUDGET_ENV] = originalBudgetEnv;

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
  // Splash `parse_thinking_token_budget`: a nonnegative integer.
  it('parses --thinking-token-budget as a nonnegative integer', () => {
    expect(parseServeArgs(['--thinking-token-budget', '4096']).thinkingTokenBudget).toBe(4096);
    expect(parseServeArgs(['--thinking-token-budget', '0']).thinkingTokenBudget).toBe(0);
    expect(parseServeArgs([]).thinkingTokenBudget).toBeUndefined();
    for (const bad of ['-1', 'abc', '1.5', '', '2147483648']) {
      expect(() => parseServeArgs([`--thinking-token-budget=${bad}`])).toThrow('--thinking-token-budget');
    }
  });

  it('parses --thinking-cycle-guard and --thinking-end-text; none of them is set by default', () => {
    const args = parseServeArgs(['--thinking-cycle-guard', '--thinking-end-text=\n\nAnswer now.']);
    expect(args.thinkingCycleGuard).toBe(true);
    expect(args.thinkingEndText).toBe('\n\nAnswer now.');
    const none = parseServeArgs([]);
    expect('thinkingCycleGuard' in none).toBe(false);
    expect('thinkingEndText' in none).toBe(false);
    expect('thinkingTokenBudget' in none).toBe(false);
    for (const bad of ['', 'x'.repeat(513)]) {
      expect(() => parseServeArgs([`--thinking-end-text=${bad}`])).toThrow('--thinking-end-text');
    }
    expect(parseServeArgs([`--thinking-end-text=${'x'.repeat(512)}`]).thinkingEndText).toHaveLength(512);
  });

  it('validates the global residency policy', () => {
    expect(parseServeArgs(['--model-residency', 'single']).modelResidency).toBe('single');
    expect(parseServeArgs(['--model-residency', 'capability']).modelResidency).toBe('capability');
    expect(() => parseServeArgs(['--model-residency', 'invalid'])).toThrow(/model-residency/);
  });
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

  it('maps --preload and leaves it unset by default', () => {
    expect(parseServeArgs(['--preload', 'qwen']).preload).toBe('qwen');
    expect(parseServeArgs([]).preload).toBeUndefined();
  });

  it('maps --kv-format and rejects any other value', () => {
    expect(parseServeArgs(['--kv-format', 'bf16']).kvFormat).toBe('bf16');
    expect(parseServeArgs(['--kv-format=int8']).kvFormat).toBe('int8');
    expect(parseServeArgs([]).kvFormat).toBeUndefined();
    expect(() => parseServeArgs(['--kv-format', 'fp8'])).toThrow('--kv-format must be int8 or bf16');
  });

  it('maps --max-memory to bytes and rejects a bad size', () => {
    expect(parseServeArgs(['--max-memory', '28G']).maxMemoryBytes).toBe(30_064_771_072);
    expect(parseServeArgs(['--max-memory=2.5G']).maxMemoryBytes).toBe(2_684_354_560);
    expect(parseServeArgs(['--max-memory', 'auto']).maxMemoryBytes).toBeUndefined();
    expect('maxMemoryBytes' in parseServeArgs([])).toBe(false);
    expect(() => parseServeArgs(['--max-memory', '2.5'])).toThrow(
      "--max-memory must be 'auto' or a positive byte count such as 32G or 2.5G",
    );
  });

  it('maps --max-context to tokens and rejects a bad count', () => {
    expect(parseServeArgs(['--max-context', '100K']).maxContextTokens).toBe(102_400);
    expect(parseServeArgs(['--max-context=1M']).maxContextTokens).toBe(1_000_000);
    expect(parseServeArgs(['--max-context', 'auto']).maxContextTokens).toBeUndefined();
    expect('maxContextTokens' in parseServeArgs(['--max-context', 'auto'])).toBe(false);
    expect('maxContextTokens' in parseServeArgs([])).toBe(false);
    expect(() => parseServeArgs(['--max-context', '2M'])).toThrow(
      "--max-context must be 'auto' or a token count up to 1010000, such as 100K or 1M (K = 1024 tokens, M = 1000000 tokens)",
    );
  });

  // Splash `parse_decode_share`: a finite number of at least 0, unset 0.5.
  // Mutation D1: the parser accepts negatives → red (`-1` passes).
  it('decode_share_parses_as_splash', () => {
    expect(parseServeArgs(['--decode-share', '0']).decodeShare).toBe(0);
    expect(parseServeArgs(['--decode-share=0.5']).decodeShare).toBe(0.5);
    expect(parseServeArgs(['--decode-share', '2']).decodeShare).toBe(2);
    expect('decodeShare' in parseServeArgs([])).toBe(false);
    for (const refused of ['-1', '-0.5', 'nan', 'inf', 'x', '']) {
      expect(() => parseServeArgs([`--decode-share=${refused}`]), refused).toThrow(
        '--decode-share must be a nonnegative number such as 0.5',
      );
    }
  });

  it('maps --idle-release to seconds and rejects a bad duration', () => {
    expect(parseServeArgs(['--idle-release', '30m']).idleReleaseSeconds).toBe(1800);
    expect(parseServeArgs(['--idle-release=off']).idleReleaseSeconds).toBe(Infinity);
    expect('idleReleaseSeconds' in parseServeArgs([])).toBe(false);
    expect(() => parseServeArgs(['--idle-release', '0'])).toThrow(
      '--idle-release must be off or a positive duration such as 30m, 2h or 600',
    );
  });

  it('treats --log-dir as implying --verbose', () => {
    expect(parseServeArgs(['--log-dir', '/tmp/mlx-serve-logs']).logDir).toBe('/tmp/mlx-serve-logs');
  });

  it('leaves logDir unset without --verbose', () => {
    expect(parseServeArgs([]).logDir).toBeUndefined();
  });

  it('maps --persist-whole-prompt', () => {
    expect(parseServeArgs(['--persist-whole-prompt']).persistWholePrompt).toBe(true);
    expect('persistWholePrompt' in parseServeArgs([])).toBe(false);
  });

  it('maps --persist-cache and --no-persist-cache, and refuses both at once', () => {
    expect(parseServeArgs(['--persist-cache']).persistCache).toBe(true);
    expect(parseServeArgs(['--no-persist-cache']).persistCache).toBe(false);
    expect('persistCache' in parseServeArgs([])).toBe(false);
    expect(() => parseServeArgs(['--persist-cache', '--no-persist-cache'])).toThrow(
      '--persist-cache and --no-persist-cache cannot be used together',
    );
  });
});

interface ServeLoadSaw {
  env: string | undefined;
  host: boolean;
  line: string | undefined;
  capture: { blocks: string | undefined; ms: string | undefined; line: string | undefined };
}

/** Run `mlx serve` with `argv`, preload `alpha`, and return the disk-cache env its load saw and the startup lines. */
async function serveLoadSaw(argv: string[]): Promise<ServeLoadSaw> {
  const modelsDir = await makeModelsDir(['alpha']);
  const logged: string[] = [];
  vi.spyOn(console, 'log').mockImplementation((line: unknown) => {
    logged.push(String(line));
  });
  const seen: (string | undefined)[] = [];
  const captureSeen: { blocks: string | undefined; ms: string | undefined }[] = [];
  let hostSays = false;
  const exitCodes: number[] = [];
  const sigintBefore = new Set(listenersOf('SIGINT'));
  await run(['--port', '0', '--models-dir', modelsDir, '--preload', 'alpha', ...argv], {
    hostOverrides: {
      disableStore: true,
      sweepOrphanTempRoots: false,
      estimateContextWindow: async () => null,
      loadModel: async () => {
        seen.push(process.env[PERSIST_ENV]);
        captureSeen.push({ blocks: process.env[BLOCKS_ENV], ms: process.env[BUDGET_ENV] });
        return {} as unknown as LoadableModel;
      },
    },
    exit: (code) => exitCodes.push(code),
    onReady: async (host: InferenceHost) => {
      hostSays = host.persistPagedCache;
      for (let i = 0; i < 200 && seen.length === 0; i++) {
        await new Promise((resolve) => setTimeout(resolve, 10));
      }
      const added = listenersOf('SIGINT').filter((l) => !sigintBefore.has(l));
      added[0]!('SIGINT');
      for (let i = 0; i < 200 && exitCodes.length === 0; i++) {
        await new Promise((resolve) => setTimeout(resolve, 10));
      }
    },
  });
  expect(exitCodes).toEqual([0]);
  expect(seen).toHaveLength(1);
  delete process.env[BLOCKS_ENV];
  delete process.env[BUDGET_ENV];
  return {
    env: seen[0],
    host: hostSays,
    line: logged.find((l) => l.startsWith('[mlx] paged KV disk cache:')),
    capture: { ...captureSeen[0]!, line: logged.find((l) => l.startsWith('[mlx] paged KV disk capture')) },
  };
}

/** {@link serveLoadSaw} without the capture limits. */
async function persistEnvAtLoad(argv: string[]): Promise<Omit<ServeLoadSaw, 'capture'>> {
  const { capture: _capture, ...saw } = await serveLoadSaw(argv);
  return saw;
}

/** {@link serveLoadSaw}, the capture limits only. */
async function captureEnvAtLoad(argv: string[]): Promise<ServeLoadSaw['capture']> {
  return (await serveLoadSaw(argv)).capture;
}

describe('mlx serve — paged KV disk cache', () => {
  const ON_LINE = '[mlx] paged KV disk cache: on';
  const OFF_LINE = '[mlx] paged KV disk cache: off';

  it('leaves the disk cache off by default and lets MLX_PERSIST_PAGED_CACHE turn it on', async () => {
    delete process.env[PERSIST_ENV];
    expect(await persistEnvAtLoad([])).toEqual({ env: '0', host: false, line: OFF_LINE });
    process.env[PERSIST_ENV] = 'yes';
    expect(await persistEnvAtLoad([])).toEqual({ env: '1', host: true, line: ON_LINE });
  });

  it('--persist-cache turns the disk cache on for the load, and --no-persist-cache beats the env', async () => {
    delete process.env[PERSIST_ENV];
    expect(await persistEnvAtLoad(['--persist-cache'])).toEqual({ env: '1', host: true, line: ON_LINE });
    process.env[PERSIST_ENV] = '1';
    expect(await persistEnvAtLoad(['--no-persist-cache'])).toEqual({ env: '0', host: false, line: OFF_LINE });
  });
});

describe('mlx serve — per-turn disk capture', () => {
  const WHOLE = '[mlx] paged KV disk capture per turn: 63125 blocks, 10000 ms';
  const NATIVE = '[mlx] paged KV disk capture per turn: 128 blocks, 250 ms';

  it('keeps the native defaults without --persist-whole-prompt', async () => {
    delete process.env[BLOCKS_ENV];
    delete process.env[BUDGET_ENV];
    expect(await captureEnvAtLoad(['--persist-cache'])).toEqual({ blocks: undefined, ms: undefined, line: NATIVE });
  });

  it('--persist-whole-prompt writes the whole-prompt limits the env does not set, only with the cache on', async () => {
    delete process.env[BLOCKS_ENV];
    delete process.env[BUDGET_ENV];
    expect(await captureEnvAtLoad(['--persist-cache', '--persist-whole-prompt'])).toEqual({
      blocks: '63125',
      ms: '10000',
      line: WHOLE,
    });
    process.env[BUDGET_ENV] = '3000';
    expect(await captureEnvAtLoad(['--persist-cache', '--persist-whole-prompt'])).toEqual({
      blocks: '63125',
      ms: '3000',
      line: '[mlx] paged KV disk capture per turn: 63125 blocks, 3000 ms',
    });
    expect(await captureEnvAtLoad(['--no-persist-cache', '--persist-whole-prompt'])).toEqual({
      blocks: undefined,
      ms: undefined,
      line: undefined,
    });
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

  it('starts loading the --preload model at boot and keeps serving when that load fails', async () => {
    const modelsDir = await makeModelsDir(['alpha', 'beta']);
    vi.spyOn(console, 'log').mockImplementation(() => undefined);
    const errors: string[] = [];
    vi.spyOn(console, 'error').mockImplementation((line: unknown) => {
      errors.push(String(line));
    });
    const loads: string[] = [];
    const exitCodes: number[] = [];
    const sigintBefore = new Set(listenersOf('SIGINT'));

    await run(['--port', '0', '--models-dir', modelsDir, '--preload', 'beta'], {
      hostOverrides: {
        disableStore: true,
        sweepOrphanTempRoots: false,
        pagedModelTypes: [],
        loadModel: async (path: string) => {
          loads.push(path);
          throw new Error('weights unreadable');
        },
      },
      exit: (code) => exitCodes.push(code),
      onReady: async (host: InferenceHost) => {
        for (let i = 0; i < 200 && errors.length === 0; i++) {
          await new Promise((resolve) => setTimeout(resolve, 10));
        }
        expect(loads).toEqual([expect.stringMatching(/[/\\]beta$/)]);
        expect(errors).toEqual(['[mlx] preload of "beta" failed: weights unreadable']);
        expect((await fetch(`${host.url}/health`)).status).toBe(200);

        const added = listenersOf('SIGINT').filter((l) => !sigintBefore.has(l));
        added[0]!('SIGINT');
        for (let i = 0; i < 200 && exitCodes.length === 0; i++) {
          await new Promise((resolve) => setTimeout(resolve, 10));
        }
      },
    });
    expect(exitCodes).toEqual([0]);
  });

  // Mutation: drop `maxMemoryBytes` from the `createInferenceHost` call → red.
  it('hands --max-memory to the qwen model loader', async () => {
    const modelsDir = await makeModelsDir(['alpha']);
    vi.spyOn(console, 'log').mockImplementation(() => undefined);
    const loads: unknown[] = [];
    const exitCodes: number[] = [];
    const sigintBefore = new Set(listenersOf('SIGINT'));

    await run(['--port', '0', '--models-dir', modelsDir, '--preload', 'alpha', '--max-memory', '28G'], {
      hostOverrides: {
        disableStore: true,
        sweepOrphanTempRoots: false,
        pagedModelTypes: [],
        estimateContextWindow: async () => null,
        loadModel: async (_path: string, options?: unknown) => {
          loads.push(options);
          return {} as unknown as LoadableModel;
        },
      },
      exit: (code) => exitCodes.push(code),
      onReady: async () => {
        for (let i = 0; i < 200 && loads.length === 0; i++) {
          await new Promise((resolve) => setTimeout(resolve, 10));
        }
        expect(loads).toEqual([{ maxMemoryBytes: 30_064_771_072 }]);
        const added = listenersOf('SIGINT').filter((l) => !sigintBefore.has(l));
        added[0]!('SIGINT');
        for (let i = 0; i < 200 && exitCodes.length === 0; i++) {
          await new Promise((resolve) => setTimeout(resolve, 10));
        }
      },
    });
    expect(exitCodes).toEqual([0]);
  });

  // Mutation: drop `thinkingCycleGuard` from the `createInferenceHost` call → red.
  it('hands the three thinking flags to a Messages request, and without them sends none', async () => {
    const configsFor = async (argv: string[]): Promise<Array<Record<string, unknown>>> => {
      const modelsDir = await makeModelsDir(['alpha']);
      vi.spyOn(console, 'log').mockImplementation(() => undefined);
      const configs: Array<Record<string, unknown>> = [];
      const result = {
        text: 'ok',
        toolCalls: [],
        thinkingEnabled: true,
        numTokens: 1,
        promptTokens: 1,
        reasoningTokens: 0,
        finishReason: 'stop',
        rawText: 'ok',
        cachedTokens: 0,
      };
      const model = {
        chatSessionStart: async (_messages: unknown, config: Record<string, unknown>) => {
          configs.push({ ...config });
          return result;
        },
        resetCaches: async () => undefined,
      };
      const exitCodes: number[] = [];
      const sigintBefore = new Set(listenersOf('SIGINT'));
      await run(['--port', '0', '--models-dir', modelsDir, '--model', 'alpha', ...argv], {
        hostOverrides: {
          disableStore: true,
          sweepOrphanTempRoots: false,
          pagedModelTypes: [],
          estimateContextWindow: async () => null,
          loadModel: async () => model as unknown as LoadableModel,
        },
        exit: (code) => exitCodes.push(code),
        onReady: async (host) => {
          const res = await fetch(`${host.url}/v1/messages`, {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ model: 'alpha', messages: [{ role: 'user', content: 'hi' }], max_tokens: 8 }),
          });
          await res.text();
          const added = listenersOf('SIGINT').filter((l) => !sigintBefore.has(l));
          added[0]!('SIGINT');
          for (let i = 0; i < 200 && exitCodes.length === 0; i++) {
            await new Promise((resolve) => setTimeout(resolve, 10));
          }
        },
      });
      expect(exitCodes).toEqual([0]);
      return configs;
    };
    const flagged = await configsFor([
      '--thinking-token-budget',
      '12000',
      '--thinking-cycle-guard',
      '--thinking-end-text=\n\nAnswer now.',
    ]);
    expect(flagged).toHaveLength(1);
    expect(flagged[0]).toMatchObject({
      thinkingTokenBudget: 12000,
      thinkingCycleGuard: true,
      thinkingEndText: '\n\nAnswer now.',
    });
    const plain = await configsFor([]);
    expect(plain).toHaveLength(1);
    expect(plain[0]!.thinkingTokenBudget).toBeUndefined();
    expect(plain[0]!.thinkingCycleGuard).toBeUndefined();
    expect(plain[0]!.thinkingEndText).toBeUndefined();
  });

  // Mutation: drop `idleReleaseSeconds` from the `createInferenceHost` call → red.
  it('sets the residency keep-alive from --idle-release before any load', async () => {
    const modelsDir = await makeModelsDir(['alpha']);
    vi.spyOn(console, 'log').mockImplementation(() => undefined);
    const events: string[] = [];
    const exitCodes: number[] = [];
    const sigintBefore = new Set(listenersOf('SIGINT'));

    await run(['--port', '0', '--models-dir', modelsDir, '--preload', 'alpha', '--idle-release', '30m'], {
      hostOverrides: {
        disableStore: true,
        sweepOrphanTempRoots: false,
        pagedModelTypes: [],
        estimateContextWindow: async () => null,
        setResidencyKeepAlive: (seconds: number) => events.push(`keep-alive ${seconds}`),
        loadModel: async () => {
          events.push('load');
          return {} as unknown as LoadableModel;
        },
      },
      exit: (code) => exitCodes.push(code),
      onReady: async (host) => {
        for (let i = 0; i < 200 && !events.includes('load'); i++) {
          await new Promise((resolve) => setTimeout(resolve, 10));
        }
        expect(events).toEqual(['keep-alive 1800', 'load']);
        expect(host.health().weights?.idleReleaseSeconds).toBe(1800);
        const added = listenersOf('SIGINT').filter((l) => !sigintBefore.has(l));
        added[0]!('SIGINT');
        for (let i = 0; i < 200 && exitCodes.length === 0; i++) {
          await new Promise((resolve) => setTimeout(resolve, 10));
        }
      },
    });
    expect(exitCodes).toEqual([0]);
  });

  // Mutation: drop `maxContextTokens` from the `createInferenceHost` call → red.
  it('hands --max-context to the qwen model loader', async () => {
    const modelsDir = await makeModelsDir(['alpha']);
    vi.spyOn(console, 'log').mockImplementation(() => undefined);
    const loads: unknown[] = [];
    const exitCodes: number[] = [];
    const sigintBefore = new Set(listenersOf('SIGINT'));

    await run(['--port', '0', '--models-dir', modelsDir, '--preload', 'alpha', '--max-context', '100K'], {
      hostOverrides: {
        disableStore: true,
        sweepOrphanTempRoots: false,
        pagedModelTypes: [],
        estimateContextWindow: async () => null,
        loadModel: async (_path: string, options?: unknown) => {
          loads.push(options);
          return {} as unknown as LoadableModel;
        },
      },
      exit: (code) => exitCodes.push(code),
      onReady: async () => {
        for (let i = 0; i < 200 && loads.length === 0; i++) {
          await new Promise((resolve) => setTimeout(resolve, 10));
        }
        expect(loads).toEqual([{ maxContextTokens: 102_400 }]);
        const added = listenersOf('SIGINT').filter((l) => !sigintBefore.has(l));
        added[0]!('SIGINT');
        for (let i = 0; i < 200 && exitCodes.length === 0; i++) {
          await new Promise((resolve) => setTimeout(resolve, 10));
        }
      },
    });
    expect(exitCodes).toEqual([0]);
  });

  // Mutation: drop `decodeShare` from the `createInferenceHost` call → red.
  it('hands --decode-share to the qwen model loader', async () => {
    const modelsDir = await makeModelsDir(['alpha']);
    vi.spyOn(console, 'log').mockImplementation(() => undefined);
    const loads: unknown[] = [];
    const exitCodes: number[] = [];
    const sigintBefore = new Set(listenersOf('SIGINT'));

    await run(['--port', '0', '--models-dir', modelsDir, '--preload', 'alpha', '--decode-share', '0'], {
      hostOverrides: {
        disableStore: true,
        sweepOrphanTempRoots: false,
        pagedModelTypes: [],
        estimateContextWindow: async () => null,
        loadModel: async (_path: string, options?: unknown) => {
          loads.push(options);
          return {} as unknown as LoadableModel;
        },
      },
      exit: (code) => exitCodes.push(code),
      onReady: async () => {
        for (let i = 0; i < 200 && loads.length === 0; i++) {
          await new Promise((resolve) => setTimeout(resolve, 10));
        }
        expect(loads).toEqual([{ decodeShare: 0 }]);
        const added = listenersOf('SIGINT').filter((l) => !sigintBefore.has(l));
        added[0]!('SIGINT');
        for (let i = 0; i < 200 && exitCodes.length === 0; i++) {
          await new Promise((resolve) => setTimeout(resolve, 10));
        }
      },
    });
    expect(exitCodes).toEqual([0]);
  });

  // Mutation: drop `kvFormat` from the `createInferenceHost` call → red.
  it('hands --kv-format to the model loader', async () => {
    const modelsDir = await makeModelsDir(['alpha']);
    vi.spyOn(console, 'log').mockImplementation(() => undefined);
    const loads: unknown[] = [];
    const exitCodes: number[] = [];
    const sigintBefore = new Set(listenersOf('SIGINT'));

    await run(['--port', '0', '--models-dir', modelsDir, '--preload', 'alpha', '--kv-format', 'bf16'], {
      hostOverrides: {
        disableStore: true,
        sweepOrphanTempRoots: false,
        pagedModelTypes: [],
        loadModel: async (_path: string, options?: unknown) => {
          loads.push(options);
          return {} as unknown as LoadableModel;
        },
      },
      exit: (code) => exitCodes.push(code),
      onReady: async () => {
        for (let i = 0; i < 200 && loads.length === 0; i++) {
          await new Promise((resolve) => setTimeout(resolve, 10));
        }
        expect(loads).toEqual([{ kvFormat: 'bf16' }]);
        const added = listenersOf('SIGINT').filter((l) => !sigintBefore.has(l));
        added[0]!('SIGINT');
        for (let i = 0; i < 200 && exitCodes.length === 0; i++) {
          await new Promise((resolve) => setTimeout(resolve, 10));
        }
      },
    });
    expect(exitCodes).toEqual([0]);
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
