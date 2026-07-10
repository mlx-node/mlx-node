import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import type { MlxModelInfo } from '@mlx-node/agent';
import { describe, expect, it, vi } from 'vite-plus/test';

import {
  type AgentRunDeps,
  chooseDefaultModel,
  readPersistedDefaultModel,
  run,
  scanAgentArgs,
  withDefaultModel,
} from '../../packages/cli/src/commands/agent/index.js';

describe('scanAgentArgs', () => {
  describe('--models-dir extraction', () => {
    it('extracts a leading --models-dir pair and removes it from passthrough', () => {
      const scan = scanAgentArgs(['--models-dir', '/models', '-p', 'hi']);
      expect(scan.modelsDir).toBe('/models');
      expect(scan.passthrough).toEqual(['-p', 'hi']);
      expect(scan.modelsDirMissingValue).toBe(false);
    });

    it('extracts a trailing --models-dir pair, preserving preceding args in order', () => {
      const scan = scanAgentArgs(['-p', 'hi', '--mode', 'json', '--models-dir', '/models']);
      expect(scan.modelsDir).toBe('/models');
      expect(scan.passthrough).toEqual(['-p', 'hi', '--mode', 'json']);
    });

    it('supports the --models-dir=<dir> form', () => {
      const scan = scanAgentArgs(['--models-dir=/models', '-c']);
      expect(scan.modelsDir).toBe('/models');
      expect(scan.passthrough).toEqual(['-c']);
    });

    it('flags a --models-dir without a value', () => {
      const scan = scanAgentArgs(['--models-dir']);
      expect(scan.modelsDirMissingValue).toBe(true);
      expect(scan.modelsDir).toBeUndefined();
      expect(scan.passthrough).toEqual([]);
    });

    it('flags an empty --models-dir= value', () => {
      const scan = scanAgentArgs(['--models-dir=']);
      expect(scan.modelsDirMissingValue).toBe(true);
      expect(scan.modelsDir).toBeUndefined();
    });

    it('flags an empty space-form value without eating later args', () => {
      const scan = scanAgentArgs(['--models-dir', '', '-p', 'hi']);
      expect(scan.modelsDirMissingValue).toBe(true);
      expect(scan.modelsDir).toBeUndefined();
      expect(scan.passthrough).toEqual(['-p', 'hi']);
    });

    it('never consumes an option-looking token as the space-form value', () => {
      for (const nextFlag of ['--local', '--help', '--no-session', '-c']) {
        const scan = scanAgentArgs(['--models-dir', nextFlag]);
        expect(scan.modelsDirMissingValue).toBe(true);
        expect(scan.modelsDir).toBeUndefined();
        // The flag stays in passthrough — it was never a value.
        expect(scan.passthrough).toEqual([nextFlag]);
      }
    });

    it('still accepts a dash-leading dir via the = form', () => {
      const scan = scanAgentArgs(['--models-dir=-odd-dir', '-p', 'hi']);
      expect(scan.modelsDir).toBe('-odd-dir');
      expect(scan.modelsDirMissingValue).toBe(false);
      expect(scan.passthrough).toEqual(['-p', 'hi']);
    });
  });

  describe('update intercept', () => {
    it('detects a leading update positional', () => {
      expect(scanAgentArgs(['update']).update).toBe(true);
      expect(scanAgentArgs(['update', '--all']).update).toBe(true);
    });

    it('does not trip on update in a non-leading position', () => {
      const scan = scanAgentArgs(['-p', 'update']);
      expect(scan.update).toBe(false);
      expect(scan.passthrough).toEqual(['-p', 'update']);
    });

    it('detects update behind a stripped --models-dir pair (pi would see it at args[0])', () => {
      const scan = scanAgentArgs(['--models-dir', '/x', 'update']);
      expect(scan.update).toBe(true);
      expect(scan.passthrough).toEqual(['update']);
    });
  });

  describe('help detection', () => {
    it('detects -h and --help', () => {
      expect(scanAgentArgs(['-h']).help).toBe(true);
      expect(scanAgentArgs(['--help']).help).toBe(true);
      expect(scanAgentArgs(['--mode', 'json', '--help']).help).toBe(true);
    });

    it('leaves per-command help to pi (install/remove/uninstall/list/config pass through)', () => {
      for (const command of ['install', 'remove', 'uninstall', 'list', 'config']) {
        const scan = scanAgentArgs([command, '--help']);
        expect(scan.help).toBe(false);
        expect(scan.passthrough).toEqual([command, '--help']);
      }
    });

    it('suppresses mlx help for a pass-through command behind --models-dir too', () => {
      const scan = scanAgentArgs(['--models-dir', '/x', 'install', '--help']);
      expect(scan.help).toBe(false);
      expect(scan.passthrough).toEqual(['install', '--help']);
    });

    it('does not detect help when absent', () => {
      expect(scanAgentArgs(['-p', 'hello']).help).toBe(false);
    });
  });

  describe('passthrough preservation', () => {
    it('passes install through untouched', () => {
      const scan = scanAgentArgs(['install', 'npm:some-extension']);
      expect(scan.update).toBe(false);
      expect(scan.help).toBe(false);
      expect(scan.passthrough).toEqual(['install', 'npm:some-extension']);
    });

    it('passes -c, --resume and unknown flags through untouched, in order', () => {
      const argv = ['-c', '--resume', 'abc123', '--totally-unknown-flag', 'value', '-p', 'prompt text'];
      const scan = scanAgentArgs(argv);
      expect(scan.passthrough).toEqual(argv);
      expect(scan.modelsDir).toBeUndefined();
      expect(scan.help).toBe(false);
      expect(scan.update).toBe(false);
    });

    it('returns empty passthrough for empty argv', () => {
      const scan = scanAgentArgs([]);
      expect(scan.passthrough).toEqual([]);
      expect(scan.help).toBe(false);
      expect(scan.update).toBe(false);
    });
  });
});

describe('withDefaultModel', () => {
  it('prepends --model mlx/<id> to a fresh run', () => {
    expect(withDefaultModel(['-p', 'hi'], 'qwen3.5-0.8b-mlx-bf16')).toEqual([
      '--model',
      'mlx/qwen3.5-0.8b-mlx-bf16',
      '-p',
      'hi',
    ]);
    expect(withDefaultModel([], 'some-model')).toEqual(['--model', 'mlx/some-model']);
  });

  it('respects an explicit --model, --models scope, or --provider', () => {
    const withModel = ['--model', 'mlx/other-model', '-p', 'hi'];
    expect(withDefaultModel(withModel, 'default-model')).toBe(withModel);
    const withScope = ['--models', 'mlx/a,mlx/b'];
    expect(withDefaultModel(withScope, 'default-model')).toBe(withScope);
    const withProvider = ['--provider', 'mlx', '--model', 'x'];
    expect(withDefaultModel(withProvider, 'default-model')).toBe(withProvider);
  });

  it('never overrides a session-carrying run (saved model must win)', () => {
    for (const args of [
      ['-c'],
      ['--continue'],
      ['-r'],
      ['--resume'],
      ['--session', 'abc'],
      ['--session-id', 'abc'],
      ['--fork', 'abc'],
    ]) {
      expect(withDefaultModel(args, 'default-model')).toBe(args);
    }
  });

  it('leaves a --fork run unchanged — the forked session restores its own model', () => {
    const argv = ['--fork', 'abc123', '-p', 'continue where we left off'];
    expect(withDefaultModel(argv, 'default-model')).toBe(argv);
  });

  it('does not treat prompt text as a flag', () => {
    const argv = ['-p', 'please run --continue for me'];
    expect(withDefaultModel(argv, 'm')).toEqual(['--model', 'mlx/m', '-p', 'please run --continue for me']);
  });
});

/**
 * End-to-end argv ROUTING through `run()`: pi's `parsePackageCommand`
 * and `handleConfigCommand` read ONLY args[0], so package commands and
 * `config` must reach `runAgent` verbatim — no `--model` injection
 * ahead of them and no first-run wizard.
 */
describe('run() argv routing', () => {
  function fakeModel(name: string): MlxModelInfo {
    return { discovered: { name } } as unknown as MlxModelInfo;
  }

  /**
   * Injected fakes for run(): `discoverBatches[i]` is the result of the
   * i-th discovery call (last batch repeats). Records every call.
   */
  function makeDeps(discoverBatches: MlxModelInfo[][] = [[fakeModel('fake-model')]]) {
    const calls = {
      discover: [] as string[],
      wizard: [] as string[],
      runAgent: [] as Array<{ modelsDir: string; models: MlxModelInfo[]; argv: string[] }>,
    };
    const deps: AgentRunDeps = {
      resolveModelsDir: (explicit) => explicit ?? '/fake/models',
      discoverMlxModels: (modelsDir) => {
        calls.discover.push(modelsDir);
        return Promise.resolve(discoverBatches[Math.min(calls.discover.length - 1, discoverBatches.length - 1)]!);
      },
      runAgent: (opts) => {
        calls.runAgent.push({ modelsDir: opts.modelsDir, models: opts.models, argv: opts.argv });
        return Promise.resolve();
      },
      wizard: (modelsDir) => {
        calls.wizard.push(modelsDir);
        return Promise.resolve();
      },
      // Hermetic default: never read the developer's real settings.json.
      readPersistedDefault: () => undefined,
    };
    return { deps, calls };
  }

  it('hands each package command to pi verbatim at argv[0] — no --model, no discovery, no wizard', async () => {
    for (const command of ['install', 'remove', 'uninstall', 'list']) {
      const { deps, calls } = makeDeps();
      await run([command, 'npm:some-extension'], deps);
      expect(calls.runAgent).toHaveLength(1);
      expect(calls.runAgent[0]!.argv).toEqual([command, 'npm:some-extension']);
      expect(calls.runAgent[0]!.argv).not.toContain('--model');
      expect(calls.runAgent[0]!.models).toEqual([]);
      expect(calls.discover).toHaveLength(0);
      expect(calls.wizard).toHaveLength(0);
    }
  });

  it('hands config to pi verbatim at argv[0] — no --model, no discovery, no wizard', async () => {
    const { deps, calls } = makeDeps();
    await run(['config', '--local'], deps);
    expect(calls.runAgent).toHaveLength(1);
    expect(calls.runAgent[0]!.argv).toEqual(['config', '--local']);
    expect(calls.runAgent[0]!.argv).not.toContain('--model');
    expect(calls.runAgent[0]!.models).toEqual([]);
    expect(calls.discover).toHaveLength(0);
    expect(calls.wizard).toHaveLength(0);
  });

  it('routes a pass-through command without any model present (empty dir, wizard stays out)', async () => {
    for (const argv of [['list'], ['config']]) {
      const { deps, calls } = makeDeps([[]]);
      await run(argv, deps);
      expect(calls.runAgent).toHaveLength(1);
      expect(calls.runAgent[0]!.argv).toEqual(argv);
      expect(calls.wizard).toHaveLength(0);
    }
  });

  it('routes a package command behind a stripped --models-dir pair', async () => {
    const { deps, calls } = makeDeps();
    await run(['--models-dir', '/x', 'install', 'npm:foo'], deps);
    expect(calls.runAgent).toHaveLength(1);
    expect(calls.runAgent[0]!.argv).toEqual(['install', 'npm:foo']);
    expect(calls.runAgent[0]!.modelsDir).toBe('/x');
    expect(calls.discover).toHaveLength(0);
  });

  it('exits 1 on a valueless --models-dir instead of consuming the next flag', async () => {
    for (const argv of [
      ['install', 'npm:x', '--models-dir', '--local'],
      ['--models-dir', '--help'],
      ['--models-dir', '--no-session', '-p', 'hi'],
      ['--models-dir'],
      ['--models-dir', ''],
    ]) {
      const { deps, calls } = makeDeps();
      const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      const prevExitCode = process.exitCode;
      try {
        await run(argv, deps);
        expect(process.exitCode).toBe(1);
        expect(errorSpy.mock.calls.flat().join('\n')).toContain('Missing value for --models-dir');
      } finally {
        process.exitCode = prevExitCode;
        errorSpy.mockRestore();
      }
      // Nothing ran: no pi handoff (help or otherwise), no discovery, no wizard.
      expect(calls.runAgent).toHaveLength(0);
      expect(calls.discover).toHaveLength(0);
      expect(calls.wizard).toHaveLength(0);
    }
  });

  it('still blocks update with exit code 1 before anything runs', async () => {
    const { deps, calls } = makeDeps();
    const prevExitCode = process.exitCode;
    try {
      await run(['update'], deps);
      expect(process.exitCode).toBe(1);
    } finally {
      process.exitCode = prevExitCode;
    }
    expect(calls.runAgent).toHaveLength(0);
    expect(calls.discover).toHaveLength(0);
    expect(calls.wizard).toHaveLength(0);
  });

  it('still injects --model mlx/<id> on a fresh agent run', async () => {
    const { deps, calls } = makeDeps();
    await run(['-p', 'hi'], deps);
    expect(calls.discover).toHaveLength(1);
    expect(calls.wizard).toHaveLength(0);
    expect(calls.runAgent).toHaveLength(1);
    expect(calls.runAgent[0]!.argv).toEqual(['--model', 'mlx/fake-model', '-p', 'hi']);
    expect(calls.runAgent[0]!.models.map((m) => m.discovered.name)).toEqual(['fake-model']);
  });

  it('still skips injection when the run already carries --model', async () => {
    const { deps, calls } = makeDeps();
    await run(['--model', 'mlx/other', '-p', 'hi'], deps);
    expect(calls.runAgent).toHaveLength(1);
    expect(calls.runAgent[0]!.argv).toEqual(['--model', 'mlx/other', '-p', 'hi']);
  });

  it('still runs the wizard for a fresh agent run with no models, then injects the downloaded one', async () => {
    const { deps, calls } = makeDeps([[], [fakeModel('downloaded-model')]]);
    await run(['-p', 'hi'], deps);
    expect(calls.wizard).toHaveLength(1);
    expect(calls.discover).toHaveLength(2);
    expect(calls.runAgent).toHaveLength(1);
    expect(calls.runAgent[0]!.argv).toEqual(['--model', 'mlx/downloaded-model', '-p', 'hi']);
  });

  it('forwards a --fork run unchanged so pi restores the forked session model', async () => {
    const { deps, calls } = makeDeps();
    await run(['--fork', 'abc123', '-p', 'hi'], deps);
    expect(calls.runAgent).toHaveLength(1);
    expect(calls.runAgent[0]!.argv).toEqual(['--fork', 'abc123', '-p', 'hi']);
    expect(calls.runAgent[0]!.argv).not.toContain('--model');
  });

  /**
   * Fresh-run injection vs pi's persisted `/model` default
   * (`<agentDir>/settings.json` → `defaultProvider` + `defaultModel`,
   * written by pi's SettingsManager): a still-discovered mlx default
   * must win over the lexicographic first; a non-mlx default is
   * overridden (local-first policy) with a stderr notice.
   */
  describe('persisted /model default', () => {
    async function withTempSettings(settings: unknown, fn: (agentDir: string) => Promise<void> | void): Promise<void> {
      const agentDir = await mkdtemp(join(tmpdir(), 'mlx-agent-settings-'));
      try {
        if (settings !== undefined) {
          await writeFile(
            join(agentDir, 'settings.json'),
            typeof settings === 'string' ? settings : JSON.stringify(settings),
          );
        }
        await fn(agentDir);
      } finally {
        await rm(agentDir, { recursive: true, force: true });
      }
    }

    const twoModels = () => [[fakeModel('model-a'), fakeModel('model-b')]];

    it('prepends a persisted mlx default that is still discovered', async () => {
      await withTempSettings({ defaultProvider: 'mlx', defaultModel: 'model-b' }, async (agentDir) => {
        const { deps, calls } = makeDeps(twoModels());
        deps.readPersistedDefault = () => readPersistedDefaultModel(agentDir);
        await run(['-p', 'hi'], deps);
        expect(calls.runAgent[0]!.argv).toEqual(['--model', 'mlx/model-b', '-p', 'hi']);
      });
    });

    it('falls back to the first discovered model when the persisted mlx default is gone', async () => {
      await withTempSettings({ defaultProvider: 'mlx', defaultModel: 'deleted-model' }, async (agentDir) => {
        const { deps, calls } = makeDeps(twoModels());
        deps.readPersistedDefault = () => readPersistedDefaultModel(agentDir);
        await run(['-p', 'hi'], deps);
        expect(calls.runAgent[0]!.argv).toEqual(['--model', 'mlx/model-a', '-p', 'hi']);
      });
    });

    it('overrides a persisted non-mlx default with the first local model AND a stderr notice', async () => {
      await withTempSettings({ defaultProvider: 'anthropic', defaultModel: 'claude-x' }, async (agentDir) => {
        const { deps, calls } = makeDeps(twoModels());
        deps.readPersistedDefault = () => readPersistedDefaultModel(agentDir);
        const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
        let notices = '';
        try {
          await run(['-p', 'hi'], deps);
          notices = errorSpy.mock.calls.flat().join('\n');
        } finally {
          errorSpy.mockRestore();
        }
        expect(calls.runAgent[0]!.argv).toEqual(['--model', 'mlx/model-a', '-p', 'hi']);
        expect(notices).toContain('anthropic/claude-x');
        expect(notices).toContain('mlx/model-a');
      });
    });

    it('stays silent about a non-mlx default when no injection happens (--model run)', async () => {
      await withTempSettings({ defaultProvider: 'anthropic', defaultModel: 'claude-x' }, async (agentDir) => {
        const { deps, calls } = makeDeps(twoModels());
        deps.readPersistedDefault = () => readPersistedDefaultModel(agentDir);
        const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
        let errorCallCount = -1;
        try {
          await run(['--model', 'mlx/model-b', '-p', 'hi'], deps);
          errorCallCount = errorSpy.mock.calls.length;
        } finally {
          errorSpy.mockRestore();
        }
        expect(calls.runAgent[0]!.argv).toEqual(['--model', 'mlx/model-b', '-p', 'hi']);
        expect(errorCallCount).toBe(0);
      });
    });

    it('treats a missing or malformed settings file as no persisted default', async () => {
      await withTempSettings(undefined, (agentDir) => {
        expect(readPersistedDefaultModel(agentDir)).toBeUndefined();
      });
      await withTempSettings('{not json', (agentDir) => {
        expect(readPersistedDefaultModel(agentDir)).toBeUndefined();
      });
      await withTempSettings({ defaultProvider: 'mlx' }, (agentDir) => {
        expect(readPersistedDefaultModel(agentDir)).toBeUndefined();
      });
    });

    it('resolves the agent dir from PI_CODING_AGENT_DIR when no dir is passed (runAgent parity)', async () => {
      await withTempSettings({ defaultProvider: 'mlx', defaultModel: 'model-b' }, (agentDir) => {
        const prev = process.env.PI_CODING_AGENT_DIR;
        process.env.PI_CODING_AGENT_DIR = agentDir;
        try {
          expect(readPersistedDefaultModel()).toEqual({ provider: 'mlx', modelId: 'model-b' });
        } finally {
          if (prev === undefined) {
            delete process.env.PI_CODING_AGENT_DIR;
          } else {
            process.env.PI_CODING_AGENT_DIR = prev;
          }
        }
      });
    });

    it('chooseDefaultModel is pure over the three policy branches', () => {
      const models = [fakeModel('model-a'), fakeModel('model-b')];
      expect(chooseDefaultModel(models, undefined)).toEqual({ modelId: 'model-a' });
      expect(chooseDefaultModel(models, { provider: 'mlx', modelId: 'model-b' })).toEqual({ modelId: 'model-b' });
      expect(chooseDefaultModel(models, { provider: 'mlx', modelId: 'gone' })).toEqual({ modelId: 'model-a' });
      const overridden = chooseDefaultModel(models, { provider: 'groq', modelId: 'llama' });
      expect(overridden.modelId).toBe('model-a');
      expect(overridden.notice).toContain('groq/llama');
    });
  });
});
