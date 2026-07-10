/**
 * `runAgent` boot-shell contract, via the `mainImpl` seam (pi itself is
 * never imported here): env seeding with `??=` semantics, the exact
 * extension pair, and verbatim argv forwarding.
 */
import { homedir } from 'node:os';
import { join } from 'node:path';

import type { InlineExtension } from '@earendil-works/pi-coding-agent';
import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { runAgent, type RunAgentMain } from '../src/run-agent.js';

const ENV_KEYS = ['PI_CODING_AGENT_DIR', 'PI_SKIP_VERSION_CHECK', 'MLX_PAGED_PREFILL_CHUNK_SIZE'] as const;

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
    await runAgent({ modelsDir: '/models', models: [], argv: [], mainImpl: main });

    expect(calls).toHaveLength(1);
    const env = calls[0]!.envAtCall;
    expect(env.PI_CODING_AGENT_DIR).toBe(join(homedir(), '.mlx-node', 'agent'));
    expect(env.PI_SKIP_VERSION_CHECK).toBe('1');
    expect(env.MLX_PAGED_PREFILL_CHUNK_SIZE).toBe('2048');
  });

  it('never clobbers user-set env values', async () => {
    process.env.PI_CODING_AGENT_DIR = '/custom/agent-home';
    process.env.PI_SKIP_VERSION_CHECK = '0';
    process.env.MLX_PAGED_PREFILL_CHUNK_SIZE = '512';

    const { main, calls } = makeSeam();
    await runAgent({ modelsDir: '/models', models: [], argv: [], mainImpl: main });

    const env = calls[0]!.envAtCall;
    expect(env.PI_CODING_AGENT_DIR).toBe('/custom/agent-home');
    expect(env.PI_SKIP_VERSION_CHECK).toBe('0');
    expect(env.MLX_PAGED_PREFILL_CHUNK_SIZE).toBe('512');
  });

  it('passes exactly the provider and permission-gate extensions, in order', async () => {
    const { main, calls } = makeSeam();
    await runAgent({ modelsDir: '/models', models: [], argv: [], mainImpl: main });

    const factories = calls[0]!.extensionFactories;
    const names = factories.map((entry) => {
      // Both mlx extensions use the named `{ name, factory }` form, never
      // the bare-function InlineExtension variant.
      expect(typeof entry).toBe('object');
      const named = entry as Extract<InlineExtension, { name: string; factory: unknown }>;
      expect(typeof named.factory).toBe('function');
      return named.name;
    });
    expect(names).toEqual(['mlx-provider', 'mlx-permission-gate']);
  });

  it('forwards argv verbatim', async () => {
    const argv = ['--mode', 'json', '--no-session', '-p', 'Reply with exactly: hi'];
    const { main, calls } = makeSeam();
    await runAgent({ modelsDir: '/models', models: [], argv, mainImpl: main });

    expect(calls[0]!.argv).toBe(argv);
    expect(calls[0]!.argv).toEqual(['--mode', 'json', '--no-session', '-p', 'Reply with exactly: hi']);
  });

  it('propagates a rejection from main', async () => {
    const boom = new Error('pi main failed');
    await expect(
      runAgent({
        modelsDir: '/models',
        models: [],
        argv: [],
        mainImpl: async () => {
          throw boom;
        },
      }),
    ).rejects.toBe(boom);
  });
});
