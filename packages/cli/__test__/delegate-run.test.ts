import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { run as runAgentCli, type AgentRunDeps } from '../src/commands/agent/index.js';
import { delegateAgentArgs, run as runDelegate } from '../src/commands/delegate.js';

function fixture() {
  const launch = vi.fn<NonNullable<AgentRunDeps['launch']>>().mockResolvedValue(0);
  const deps: AgentRunDeps = {
    resolveModelsDir: (explicit) => explicit ?? '/models',
    discoverModels: vi.fn().mockResolvedValue([{ name: 'local' }]),
    launch,
    readPersistedDefault: () => ({ modelId: 'local' }),
    wizard: vi.fn().mockRejectedValue(new Error('Install a local model first')),
  };
  return { launch, deps };
}

afterEach(() => vi.restoreAllMocks());

describe('delegate shares agent startup', () => {
  it.each([
    ['Summarize this project'],
    ['--reasoning-effort', 'high', '--output-format', 'json', 'Task'],
    ['--resume', 'existing-session', 'Continue'],
    ['--models-dir', '/other/models', '--model', 'local', '--no-persist-cache', 'Task'],
    ['--tools', 'read_file,run_terminal_command', '--plugin-dir', './plugin', 'Task'],
    ['--', '--model', '--no-persist-cache'],
  ])('shares agent runtime configuration apart from the worker profile for %j', async (...args) => {
    const agent = fixture();
    const delegate = fixture();
    const previous = process.exitCode;
    try {
      await runAgentCli(delegateAgentArgs(args), agent.deps);
      await runDelegate(args, delegate.deps);
      expect(delegate.launch).toHaveBeenCalledTimes(1);
      expect(delegate.launch.mock.calls[0]![0]).toEqual(agent.launch.mock.calls[0]![0]);
    } finally {
      process.exitCode = previous;
    }
  });

  it('retains local-only model selection and cache/session defaults for legacy GitHub callers', async () => {
    const { launch, deps } = fixture();
    const previous = process.exitCode;
    try {
      await runDelegate(['github', '--repo', 'owner/repo', 'Inspect run 42'], deps);
      const options = launch.mock.calls[0]![0];
      // -m local is injected ahead of the worker profile; positionals fold into -p.
      expect(options.argv.slice(0, 2)).toEqual(['-m', 'local']);
      expect(options.argv[options.argv.indexOf('-p') + 1]).toBe('Inspect run 42');
      expect(options.argv).toContain('--append-system-prompt');
      expect(options.argv).not.toContain('--reasoning-effort');
      expect(options.modelsDir).toBe('/models');
    } finally {
      process.exitCode = previous;
    }
  });

  it('uses the same no-model failure and never starts inference', async () => {
    const { launch, deps } = fixture();
    deps.discoverModels = vi.fn().mockResolvedValue([]);
    const stderr = vi.spyOn(console, 'error').mockImplementation(() => {});
    const previous = process.exitCode;
    try {
      await runDelegate(['Task'], deps);
      expect(process.exitCode).toBe(1);
      expect(stderr).toHaveBeenCalledWith('Install a local model first');
      expect(launch).not.toHaveBeenCalled();
    } finally {
      process.exitCode = previous;
    }
  });

  it('lets runtime failures propagate through the same CLI error handling', async () => {
    const { launch, deps } = fixture();
    launch.mockRejectedValue(new Error('Agent failed'));
    await expect(runDelegate(['Task'], deps)).rejects.toThrow('Agent failed');
  });

  it('forwards headless one-shots to the binary without a model or server', async () => {
    const { launch, deps } = fixture();
    const previous = process.exitCode;
    try {
      await runDelegate(['--version'], deps);
      expect(launch).toHaveBeenCalledWith({ argv: ['--version'], needsServer: false });
      expect(deps.discoverModels).not.toHaveBeenCalled();
      expect(process.exitCode).toBe(0);
    } finally {
      process.exitCode = previous;
    }
  });
});
