/**
 * `runAgent` boot-shell contract, via the `piImpl` seam: env seeding with
 * `??=` semantics, the exact extension set, registry-policy lifecycle, and
 * verbatim argv forwarding.
 */
import { mkdtemp, rm } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';

import { type InlineExtension, ModelRuntime } from '@earendil-works/pi-coding-agent';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import {
  agentGemmaDraftEnabled,
  runAgent,
  type AgentPagedConfigOverrides,
  type RunAgentMain,
  type RunAgentPi,
} from '../src/run-agent.js';

const FAKE_MODEL = {
  discovered: { name: 'local', path: '/models/local', modelType: 'qwen3' },
  piModel: {},
} as never;

const ENV_KEYS = [
  'PI_CODING_AGENT_DIR',
  'PI_SKIP_VERSION_CHECK',
  'MLX_PAGED_PREFILL_CHUNK_SIZE',
  'PI_OFFLINE',
] as const;

type EnvKey = (typeof ENV_KEYS)[number];

interface SeamCapture {
  argv: string[];
  extensionFactories: InlineExtension[];
  /** Env values observed INSIDE main — proves seeding happens before the call. */
  envAtCall: Record<EnvKey, string | undefined>;
}

function makeSeam(): { main: RunAgentMain; calls: SeamCapture[] } {
  const calls: SeamCapture[] = [];
  const main: RunAgentMain = async (argv, opts) => {
    calls.push({
      argv,
      extensionFactories: opts.extensionFactories,
      envAtCall: Object.fromEntries(ENV_KEYS.map((key) => [key, process.env[key]])) as Record<
        EnvKey,
        string | undefined
      >,
    });
  };
  return { main, calls };
}

function piImpl(main: RunAgentMain): RunAgentPi {
  return { main, ModelRuntime };
}

function pagedConfigOverrides(): AgentPagedConfigOverrides & { cleanup: ReturnType<typeof vi.fn> } {
  return {
    resolve: vi.fn(async (path: string) => `/paged${path}`),
    cleanup: vi.fn(async () => undefined),
  };
}

describe('runAgent', () => {
  let savedEnv: Record<EnvKey, string | undefined>;

  beforeEach(() => {
    savedEnv = Object.fromEntries(ENV_KEYS.map((key) => [key, process.env[key]])) as Record<EnvKey, string | undefined>;
    for (const key of ENV_KEYS) delete process.env[key];
  });

  afterEach(() => {
    for (const key of ENV_KEYS) {
      if (savedEnv[key] === undefined) delete process.env[key];
      else process.env[key] = savedEnv[key];
    }
  });

  it('seeds the three env vars before invoking main when they are absent', async () => {
    const { main, calls } = makeSeam();
    await runAgent({ modelsDir: '/models', models: [], argv: [], piImpl: piImpl(main) });

    expect(calls).toHaveLength(1);
    const env = calls[0]!.envAtCall;
    expect(env.PI_CODING_AGENT_DIR).toBe(join(homedir(), '.mlx-node', 'agent'));
    expect(env.PI_SKIP_VERSION_CHECK).toBe('1');
    expect(env.MLX_PAGED_PREFILL_CHUNK_SIZE).toBe('2048');
    expect(env.PI_OFFLINE).toBe('1');
  });

  it('never clobbers the ??= tunables, but always forces PI_OFFLINE=1', async () => {
    process.env.PI_CODING_AGENT_DIR = '/custom/agent-home';
    process.env.PI_SKIP_VERSION_CHECK = '0';
    process.env.MLX_PAGED_PREFILL_CHUNK_SIZE = '512';

    const { main, calls } = makeSeam();
    await runAgent({ modelsDir: '/models', models: [], argv: [], piImpl: piImpl(main) });

    const env = calls[0]!.envAtCall;
    expect(env.PI_CODING_AGENT_DIR).toBe('/custom/agent-home');
    expect(env.PI_SKIP_VERSION_CHECK).toBe('0');
    expect(env.MLX_PAGED_PREFILL_CHUNK_SIZE).toBe('512');
    // PI_OFFLINE was never user-set here; the offline invariant still forces it on.
    expect(env.PI_OFFLINE).toBe('1');
  });

  it('forces PI_OFFLINE=1 even when the user tried to disable it', async () => {
    // Hard offline invariant: no ambient PI_OFFLINE=0 may re-enable cloud network.
    process.env.PI_OFFLINE = '0';

    const { main, calls } = makeSeam();
    await runAgent({ modelsDir: '/models', models: [], argv: [], piImpl: piImpl(main) });

    expect(calls[0]!.envAtCall.PI_OFFLINE).toBe('1');
  });

  it('passes exactly the built-in mlx extensions, in order', async () => {
    const { main, calls } = makeSeam();
    await runAgent({ modelsDir: '/models', models: [], argv: [], piImpl: piImpl(main) });

    const factories = calls[0]!.extensionFactories;
    const names = factories.map((entry) => {
      // All mlx extensions use the named `{ name, factory }` form, never
      // the bare-function InlineExtension variant.
      expect(typeof entry).toBe('object');
      const named = entry as Extract<InlineExtension, { name: string; factory: unknown }>;
      expect(typeof named.factory).toBe('function');
      return named.name;
    });
    expect(names).toEqual(['mlx-provider', 'mlx-permission-gate', 'mlx-terminal-title']);
  });

  it('adds subagents only for a real parent model session', async () => {
    const { main, calls } = makeSeam();
    await runAgent({ modelsDir: '/models', models: [FAKE_MODEL], argv: [], piImpl: piImpl(main) });

    const names = calls[0]!.extensionFactories.map((entry) =>
      typeof entry === 'function' ? '<anonymous>' : entry.name,
    );
    expect(names).toEqual(['mlx-provider', 'mlx-permission-gate', 'mlx-subagent', 'mlx-terminal-title']);
  });

  it('adds a TUI trace notice only when the CLI supplies a log path', async () => {
    const { main, calls } = makeSeam();
    await runAgent({
      modelsDir: '/models',
      models: [FAKE_MODEL],
      argv: [],
      traceLogFile: '/logs/inference.log',
      piImpl: piImpl(main),
    });

    const names = calls[0]!.extensionFactories.map((entry) =>
      typeof entry === 'function' ? '<anonymous>' : entry.name,
    );
    expect(names).toEqual([
      'mlx-provider',
      'mlx-permission-gate',
      'mlx-subagent',
      'mlx-trace-notice',
      'mlx-terminal-title',
    ]);
  });

  it.each([['--no-extensions'], ['-ne']])('respects the extension opt-out %s', async (...argv) => {
    const { main, calls } = makeSeam();
    await runAgent({ modelsDir: '/models', models: [FAKE_MODEL], argv, piImpl: piImpl(main) });
    const names = calls[0]!.extensionFactories.map((entry) =>
      typeof entry === 'function' ? '<anonymous>' : entry.name,
    );
    expect(names).not.toContain('mlx-subagent');
  });

  it('forwards argv verbatim', async () => {
    const argv = ['--mode', 'json', '--no-session', '-p', 'Reply with exactly: hi'];
    const { main, calls } = makeSeam();
    await runAgent({ modelsDir: '/models', models: [], argv, piImpl: piImpl(main) });

    expect(calls[0]!.argv).toBe(argv);
    expect(calls[0]!.argv).toEqual(['--mode', 'json', '--no-session', '-p', 'Reply with exactly: hi']);
  });

  it('propagates a rejection from main', async () => {
    const boom = new Error('pi main failed');
    const paged = pagedConfigOverrides();
    await expect(
      runAgent({
        modelsDir: '/models',
        models: [],
        argv: [],
        pagedConfigOverrides: paged,
        piImpl: piImpl(async () => {
          throw boom;
        }),
      }),
    ).rejects.toBe(boom);
    expect(paged.cleanup).toHaveBeenCalledOnce();
  });

  it('cleans up paged config overlays when main returns', async () => {
    const paged = pagedConfigOverrides();
    await runAgent({
      modelsDir: '/models',
      models: [],
      argv: [],
      pagedConfigOverrides: paged,
      piImpl: piImpl(async () => undefined),
    });

    expect(paged.cleanup).toHaveBeenCalledOnce();
  });

  it('applies the runtime policy while main runs and restores it afterward', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mlx-run-agent-policy-'));
    try {
      // Isolated runtime: no models.json, an empty temp auth file. The full
      // builtin catalog still contains cloud providers regardless of auth.
      const runtime = await ModelRuntime.create({ authPath: join(dir, 'auth.json'), modelsPath: null });
      const groq = runtime.getModels().find((model) => model.provider === 'groq');
      expect(groq).toBeDefined();

      await runAgent({
        modelsDir: '/models',
        models: [FAKE_MODEL],
        argv: [],
        piImpl: {
          main: async () => {
            // The filter installs on the runtime prototype for `main`'s lifetime,
            // so this concrete instance's reads become mlx-only. No mlx/local is
            // registered on it, so the filtered catalog collapses to empty.
            expect(runtime.getModel('groq', groq!.id)).toBeUndefined();
            expect(runtime.getModels()).toEqual([]);
          },
          ModelRuntime,
        },
      });

      expect(runtime.getModel('groq', groq!.id)).toBeDefined();
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});

describe('agentGemmaDraftEnabled', () => {
  it('requires the explicit value 1', () => {
    expect(agentGemmaDraftEnabled({ MLX_AGENT_ENABLE_GEMMA_DRAFT: '1' })).toBe(true);
    expect(agentGemmaDraftEnabled({ MLX_AGENT_ENABLE_GEMMA_DRAFT: 'true' })).toBe(false);
    expect(agentGemmaDraftEnabled({ MLX_AGENT_ENABLE_GEMMA_DRAFT: '0' })).toBe(false);
    expect(agentGemmaDraftEnabled({})).toBe(false);
  });
});
