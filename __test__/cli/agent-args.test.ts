import { describe, expect, it } from 'vite-plus/test';

import { scanAgentArgs, withDefaultModel } from '../../packages/cli/src/commands/agent/index.js';

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
