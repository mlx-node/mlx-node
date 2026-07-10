import type { MlxModelInfo } from '@mlx-node/agent';
import { describe, expect, it } from 'vite-plus/test';

import {
  type AgentRunDeps,
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

    it('leaves package-manager help to pi (install/remove/uninstall/list pass through)', () => {
      for (const command of ['install', 'remove', 'uninstall', 'list']) {
        const scan = scanAgentArgs([command, '--help']);
        expect(scan.help).toBe(false);
        expect(scan.passthrough).toEqual([command, '--help']);
      }
    });

    it('suppresses mlx help for a package command behind --models-dir too', () => {
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
    for (const args of [['-c'], ['--continue'], ['-r'], ['--resume'], ['--session', 'abc'], ['--session-id', 'abc']]) {
      expect(withDefaultModel(args, 'default-model')).toBe(args);
    }
  });

  it('does not treat prompt text as a flag', () => {
    const argv = ['-p', 'please run --continue for me'];
    expect(withDefaultModel(argv, 'm')).toEqual(['--model', 'mlx/m', '-p', 'please run --continue for me']);
  });
});

/**
 * End-to-end argv ROUTING through `run()`: pi's `parsePackageCommand`
 * reads ONLY args[0], so package commands must reach `runAgent` verbatim
 * — no `--model` injection ahead of them and no first-run wizard.
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

  it('routes a package command without any model present (empty dir, wizard stays out)', async () => {
    const { deps, calls } = makeDeps([[]]);
    await run(['list'], deps);
    expect(calls.runAgent).toHaveLength(1);
    expect(calls.runAgent[0]!.argv).toEqual(['list']);
    expect(calls.wizard).toHaveLength(0);
  });

  it('routes a package command behind a stripped --models-dir pair', async () => {
    const { deps, calls } = makeDeps();
    await run(['--models-dir', '/x', 'install', 'npm:foo'], deps);
    expect(calls.runAgent).toHaveLength(1);
    expect(calls.runAgent[0]!.argv).toEqual(['install', 'npm:foo']);
    expect(calls.runAgent[0]!.modelsDir).toBe('/x');
    expect(calls.discover).toHaveLength(0);
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
});
