import type { MlxModelInfo } from '@mlx-node/agent';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { run as runAgent, type AgentRunDeps } from '../src/commands/agent/index.js';
import { DELEGATE_DEFAULT_ARGS, run as runDelegate } from '../src/commands/delegate.js';

function fixture() {
  const models = [{ discovered: { name: 'local' } }] as MlxModelInfo[];
  const run = vi.fn<NonNullable<AgentRunDeps['runAgent']>>().mockResolvedValue();
  const deps: AgentRunDeps = {
    resolveModelsDir: (explicit) => explicit ?? '/models',
    discoverMlxModels: vi.fn().mockResolvedValue(models),
    runAgent: run,
    readPersistedDefault: () => ({ provider: 'mlx', modelId: 'local' }),
    writePersistedDefault: vi.fn(),
    wizard: vi.fn().mockRejectedValue(new Error('Install a local model first')),
  };
  return { run, deps };
}

afterEach(() => vi.restoreAllMocks());

describe('delegate shares agent startup', () => {
  it.each([
    ['Summarize this project'],
    ['--thinking', 'high', '--mode', 'json', 'Task'],
    ['--session', 'existing-session', 'Continue'],
    ['--models-dir', '/other/models', '--model', 'local', '--no-persist-cache', 'Task'],
    ['--tools', 'read,bash', '--extension', './extension.ts', '--skill', './skill.md', 'Task'],
    ['--no-session', '--no-context-files', '--no-extensions', 'Task'],
    ['--', '--model', '--no-persist-cache'],
  ])('shares agent runtime configuration apart from the worker profile for %j', async (...args) => {
    const agent = fixture();
    const delegate = fixture();
    const priorAutoApprove = process.env.MLX_AGENT_AUTO_APPROVE;
    await runAgent([...DELEGATE_DEFAULT_ARGS, ...args], agent.deps);
    await runDelegate(args, delegate.deps);
    expect(delegate.run).toHaveBeenCalledTimes(1);
    expect(delegate.run.mock.calls[0]![0]).toEqual({
      ...agent.run.mock.calls[0]![0],
      mode: 'delegate',
      delegateCallerApproved: false,
    });
    expect(process.env.MLX_AGENT_AUTO_APPROVE).toBe(priorAutoApprove);
  });

  it('retains local-only model selection and cache/session defaults for legacy GitHub callers', async () => {
    const { run, deps } = fixture();
    await runDelegate(['github', '--repo', 'owner/repo', 'Inspect run 42'], deps);
    const options = run.mock.calls[0]![0];
    expect(options.argv.slice(0, 4)).toEqual(['--models', 'mlx/*', '--model', 'mlx/local']);
    expect(options.argv).toContain('--print');
    expect(options.argv).not.toContain('--no-session');
    expect(options.argv).not.toContain('--thinking');
    expect(options.persistPagedCache).toBe(true);
  });

  it('uses the same no-model failure and never starts inference', async () => {
    const { run, deps } = fixture();
    deps.discoverMlxModels = vi.fn().mockResolvedValue([]);
    const stderr = vi.spyOn(console, 'error').mockImplementation(() => {});
    const previous = process.exitCode;
    try {
      await runDelegate(['Task'], deps);
      expect(process.exitCode).toBe(1);
      expect(stderr).toHaveBeenCalledWith('Install a local model first');
      expect(run).not.toHaveBeenCalled();
    } finally {
      process.exitCode = previous;
    }
  });

  it('lets runtime failures propagate through the same CLI error handling', async () => {
    const { run, deps } = fixture();
    run.mockRejectedValue(new Error('Agent failed'));
    await expect(runDelegate(['Task'], deps)).rejects.toThrow('Agent failed');
  });
});
