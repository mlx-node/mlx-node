import { chmodSync, existsSync, lstatSync, mkdtempSync, mkdirSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import {
  ModelsDirUnavailableError,
  resolveModelsDir,
  resolveModelsDirAsync,
} from '../../../packages/server/src/host/paths.js';

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

describe('resolveModelsDirAsync', () => {
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

  it('uses the explicit argument when provided', async () => {
    const explicit = join(workRoot, 'explicit');
    await expect(resolveModelsDirAsync(explicit)).resolves.toBe(resolve(explicit));
  });

  it('falls through to env when explicit is absent', async () => {
    const envDir = join(workRoot, 'env-models');
    vi.stubEnv('MLX_MODELS_DIR', envDir);
    await expect(resolveModelsDirAsync()).resolves.toBe(resolve(envDir));
  });

  it('reads modelsDir from ~/.mlx-node/config.json when env is unset', async () => {
    const configDir = join(fakeHome, '.mlx-node');
    mkdirSync(configDir, { recursive: true });
    const configuredDir = join(workRoot, 'configured');
    writeFileSync(join(configDir, 'config.json'), JSON.stringify({ modelsDir: configuredDir }));
    await expect(resolveModelsDirAsync()).resolves.toBe(resolve(configuredDir));
  });

  it('falls back to ~/.mlx-node/models when nothing else is set', async () => {
    await expect(resolveModelsDirAsync()).resolves.toBe(join(fakeHome, '.mlx-node', 'models'));
  });

  it('tolerates malformed config.json with a warning and falls back to default', async () => {
    const configDir = join(fakeHome, '.mlx-node');
    mkdirSync(configDir, { recursive: true });
    writeFileSync(join(configDir, 'config.json'), '{ not valid json');
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      await expect(resolveModelsDirAsync()).resolves.toBe(join(fakeHome, '.mlx-node', 'models'));
      expect(warn).toHaveBeenCalled();
    } finally {
      warn.mockRestore();
    }
  });

  it('prefers explicit arg over env and config.json', async () => {
    vi.stubEnv('MLX_MODELS_DIR', join(workRoot, 'should-lose-env'));
    const configDir = join(fakeHome, '.mlx-node');
    mkdirSync(configDir, { recursive: true });
    writeFileSync(join(configDir, 'config.json'), JSON.stringify({ modelsDir: join(workRoot, 'should-lose-config') }));
    const explicit = join(workRoot, 'winner');
    await expect(resolveModelsDirAsync(explicit)).resolves.toBe(resolve(explicit));
  });
});

describe('a models folder that cannot be made', () => {
  let workRoot: string;
  let fakeHome: string;
  let link: string;
  let target: string;

  beforeEach(() => {
    workRoot = mkdtempSync(join(tmpdir(), 'mlx-config-test-'));
    fakeHome = join(workRoot, 'home');
    mkdirSync(join(fakeHome, '.mlx-node'), { recursive: true });
    vi.stubEnv('HOME', fakeHome);
    delete process.env.MLX_MODELS_DIR;
    link = join(fakeHome, '.mlx-node', 'models');
    target = join(workRoot, 'Volumes', 'Disk', 'models');
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    chmodSync(workRoot, 0o700);
    rmSync(workRoot, { recursive: true, force: true });
  });

  const sentence = (): string =>
    `The models folder ${link} is a link to ${target}, which is not available. Connect that disk, or change the link, then try again.`;

  it('names the link and its missing target, and leaves the link as it is', async () => {
    symlinkSync(target, link);
    expect(() => resolveModelsDir()).toThrow(ModelsDirUnavailableError);
    expect(() => resolveModelsDir()).toThrow(sentence());
    await expect(resolveModelsDirAsync()).rejects.toThrow(sentence());
    await expect(resolveModelsDirAsync()).rejects.toBeInstanceOf(ModelsDirUnavailableError);
    expect(lstatSync(link).isSymbolicLink()).toBe(true);
    expect(existsSync(join(workRoot, 'Volumes'))).toBe(false);
  });

  it('resolves a relative link target against the folder that holds the link', () => {
    symlinkSync(join('..', 'gone'), link);
    target = join(fakeHome, 'gone');
    expect(() => resolveModelsDir()).toThrow(sentence());
  });

  it('applies to an explicit or MLX_MODELS_DIR folder too', async () => {
    const elsewhere = join(workRoot, 'elsewhere');
    symlinkSync(target, elsewhere);
    link = elsewhere;
    expect(() => resolveModelsDir(elsewhere)).toThrow(sentence());
    vi.stubEnv('MLX_MODELS_DIR', elsewhere);
    await expect(resolveModelsDirAsync()).rejects.toThrow(sentence());
  });

  it('serves a live link as before', async () => {
    mkdirSync(target, { recursive: true });
    symlinkSync(target, link);
    expect(resolveModelsDir()).toBe(link);
    await expect(resolveModelsDirAsync()).resolves.toBe(link);
  });

  it('adds one sentence when a file is in the way', async () => {
    writeFileSync(link, '');
    const said = `The models folder ${link} cannot be made: it, or a folder above it, is a file.`;
    expect(() => resolveModelsDir()).toThrow(`EEXIST: file already exists, mkdir '${link}'. ${said}`);
    await expect(resolveModelsDirAsync()).rejects.toThrow(said);
    const below = join(link, 'below');
    expect(() => resolveModelsDir(below)).toThrow(
      `The models folder ${below} cannot be made: it, or a folder above it, is a file.`,
    );
  });

  it.skipIf(process.getuid?.() === 0)('keeps the permission error as it is', () => {
    chmodSync(join(fakeHome, '.mlx-node'), 0o500);
    try {
      let caught: unknown;
      try {
        resolveModelsDir();
      } catch (error) {
        caught = error;
      }
      expect((caught as NodeJS.ErrnoException).code).toBe('EACCES');
      expect((caught as Error).message).toBe(`EACCES: permission denied, mkdir '${link}'`);
    } finally {
      chmodSync(join(fakeHome, '.mlx-node'), 0o700);
    }
  });
});
