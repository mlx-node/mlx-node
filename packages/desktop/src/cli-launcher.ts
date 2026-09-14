/** The app-owned CLI runs in the caller's process tree, using Electron's bundled Node. */
import { execFile } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { chmod, lstat, mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { dirname, isAbsolute, join } from 'node:path';
import { promisify } from 'node:util';

import { CliVerificationCache } from './cli-verification-cache.js';

const execute = promisify(execFile);
const HEADER = '#!/bin/sh\n# Managed by mlx-node.\n';

export interface DesktopCliConfig {
  executable: string;
  entry: string;
  nativeAddon: string;
  modelsDir: string | null;
}

const quote = (value: string): string => `'${value.replaceAll("'", "'\\''")}'`;

export function launcherScript(config: DesktopCliConfig): string {
  for (const path of [config.executable, config.entry, config.nativeAddon]) {
    if (!isAbsolute(path) || /[\0\r\n]/.test(path)) throw new Error('The app command has an invalid runtime path.');
  }
  return `${HEADER}export ELECTRON_RUN_AS_NODE=1\nunset NAPI_RS_NATIVE_LIBRARY_PATH\nexport MLX_CORE_NATIVE_LIBRARY_PATH=${quote(config.nativeAddon)}\n${
    config.modelsDir ? `export MLX_MODELS_DIR=${quote(config.modelsDir)}\n` : ''
  }exec ${quote(config.executable)} ${quote(config.entry)} "$@"\n`;
}

/** Repaired on app launch; probes are cached until the launcher or bundled runtime changes. */
export function createCliLauncher(config: DesktopCliConfig, home = homedir()) {
  const path = join(home, '.mlx-node', 'bin', 'mlx');
  const script = launcherScript(config);
  const cache = new CliVerificationCache(join(home, '.mlx-node', 'cli-verification.json'));
  let pending: Promise<string> | undefined;
  const prepare = async (): Promise<string> => {
    let previous: string | undefined;
    try {
      const info = await lstat(path);
      if (!info.isFile()) throw new Error(`Cannot set up the app command: ${path} is not a regular file.`);
      previous = await readFile(path, 'utf8');
      if (!previous.startsWith(HEADER))
        throw new Error(`Cannot replace the existing command at ${path}. Move it aside and retry.`);
      if ((info.mode & 0o777) !== 0o755) await chmod(path, 0o755);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
    }
    if (previous !== script) {
      await mkdir(dirname(path), { recursive: true, mode: 0o700 });
      const temp = `${path}.${randomUUID()}.tmp`;
      try {
        await writeFile(temp, script, { mode: 0o755, flag: 'wx' });
        await rename(temp, path);
      } finally {
        await rm(temp, { force: true });
      }
    }
    await cache.verify([path, config.executable, config.entry, config.nativeAddon], async () => {
      // No user-installed Node/mlx, shell startup files, model load, or permission overrides.
      const { stdout } = await execute(path, ['delegate', '--help'], {
        env: { ...process.env, PATH: '/usr/bin:/bin:/usr/sbin:/sbin' },
        timeout: 10_000,
        maxBuffer: 128 * 1024,
      });
      if (!stdout.includes('Usage: mlx delegate') || !stdout.includes('mlx delegate github')) {
        throw new Error('The bundled command does not support delegation. Reinstall or update mlx-node.');
      }
    });
    return path;
  };
  return {
    path,
    prepare(): Promise<string> {
      pending ??= prepare().finally(() => {
        pending = undefined;
      });
      return pending;
    },
  };
}
