import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import { resolveModelsDir } from '../../packages/cli/src/config.js';

describe('resolveModelsDir', () => {
  let workRoot: string;
  let fakeHome: string;

  beforeEach(() => {
    workRoot = mkdtempSync(join(tmpdir(), 'mlx-config-test-'));
    fakeHome = join(workRoot, 'home');
    mkdirSync(fakeHome, { recursive: true });
    vi.stubEnv('HOME', fakeHome);
    delete process.env.MLX_MODELS_DIR;
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    rmSync(workRoot, { recursive: true, force: true });
  });

  it('uses the explicit argument when provided', () => {
    const explicit = join(workRoot, 'explicit');
    const got = resolveModelsDir(explicit);
    expect(got).toBe(resolve(explicit));
  });

  it('falls through to env when explicit is absent', () => {
    const envDir = join(workRoot, 'env-models');
    vi.stubEnv('MLX_MODELS_DIR', envDir);
    const got = resolveModelsDir();
    expect(got).toBe(resolve(envDir));
  });

  it('reads modelsDir from ~/.mlx-node/config.json when env is unset', () => {
    const configDir = join(fakeHome, '.mlx-node');
    mkdirSync(configDir, { recursive: true });
    const configuredDir = join(workRoot, 'configured');
    writeFileSync(join(configDir, 'config.json'), JSON.stringify({ modelsDir: configuredDir }));
    const got = resolveModelsDir();
    expect(got).toBe(resolve(configuredDir));
  });

  it('falls back to ~/.mlx-node/models when nothing else is set', () => {
    const got = resolveModelsDir();
    expect(got).toBe(join(fakeHome, '.mlx-node', 'models'));
  });

  it('tolerates malformed config.json with a warning and falls back to default', () => {
    const configDir = join(fakeHome, '.mlx-node');
    mkdirSync(configDir, { recursive: true });
    writeFileSync(join(configDir, 'config.json'), '{ not valid json');
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      const got = resolveModelsDir();
      expect(got).toBe(join(fakeHome, '.mlx-node', 'models'));
      expect(warn).toHaveBeenCalled();
    } finally {
      warn.mockRestore();
    }
  });

  it('prefers explicit arg over env and config.json', () => {
    vi.stubEnv('MLX_MODELS_DIR', join(workRoot, 'should-lose-env'));
    const configDir = join(fakeHome, '.mlx-node');
    mkdirSync(configDir, { recursive: true });
    writeFileSync(join(configDir, 'config.json'), JSON.stringify({ modelsDir: join(workRoot, 'should-lose-config') }));
    const explicit = join(workRoot, 'winner');
    const got = resolveModelsDir(explicit);
    expect(got).toBe(resolve(explicit));
  });
});
