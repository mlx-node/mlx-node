/** The app-owned CLI runs in the caller's process tree, using Electron's bundled Node. */
import { execFile } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { chmod, lstat, mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { dirname, isAbsolute, join } from 'node:path';
import { promisify } from 'node:util';

import { CliVerificationCache } from './cli-verification-cache.js';

const execute = promisify(execFile);
const POSIX_HEADER = '#!/bin/sh\n# Managed by mlx-node.\n';
const WINDOWS_HEADER = '@echo off\r\nREM Managed by mlx-node.\r\n';

export interface DesktopCliConfig {
  executable: string;
  entry: string;
  nativeAddon: string;
  modelsDir: string | null;
}

const quote = (value: string): string => `'${value.replaceAll("'", "'\\''")}'`;
const batchValue = (value: string): string => value.replaceAll('%', '%%');

export function launcherScript(config: DesktopCliConfig, platform: NodeJS.Platform = process.platform): string {
  for (const path of [config.executable, config.entry, config.nativeAddon]) {
    if (!isAbsolute(path) || /[\0\r\n]/.test(path)) throw new Error('The app command has an invalid runtime path.');
  }
  if (platform === 'win32') {
    return `${WINDOWS_HEADER}setlocal DisableDelayedExpansion\r\nset "ELECTRON_RUN_AS_NODE=1"\r\nset "NAPI_RS_NATIVE_LIBRARY_PATH="\r\nset "MLX_CORE_NATIVE_LIBRARY_PATH=${batchValue(config.nativeAddon)}"\r\n${
      config.modelsDir ? `set "MLX_MODELS_DIR=${batchValue(config.modelsDir)}"\r\n` : ''
    }"${batchValue(config.executable)}" "${batchValue(config.entry)}" %*\r\nexit /b %ERRORLEVEL%\r\n`;
  }
  return `${POSIX_HEADER}export ELECTRON_RUN_AS_NODE=1\nunset NAPI_RS_NATIVE_LIBRARY_PATH\nexport MLX_CORE_NATIVE_LIBRARY_PATH=${quote(config.nativeAddon)}\n${
    config.modelsDir ? `export MLX_MODELS_DIR=${quote(config.modelsDir)}\n` : ''
  }exec ${quote(config.executable)} ${quote(config.entry)} "$@"\n`;
}

/** Repaired on app launch; probes are cached until the launcher or bundled runtime changes. */
export function createCliLauncher(
  config: DesktopCliConfig,
  home = homedir(),
  platform: NodeJS.Platform = process.platform,
) {
  const windows = platform === 'win32';
  const header = windows ? WINDOWS_HEADER : POSIX_HEADER;
  const path = join(home, '.mlx-node', 'bin', windows ? 'mlx.cmd' : 'mlx');
  const script = launcherScript(config, platform);
  const cache = new CliVerificationCache(join(home, '.mlx-node', 'cli-verification.json'));
  let pending: Promise<string> | undefined;
  const prepare = async (): Promise<string> => {
    let previous: string | undefined;
    try {
      const info = await lstat(path);
      if (!info.isFile()) throw new Error(`Cannot set up the app command: ${path} is not a regular file.`);
      previous = await readFile(path, 'utf8');
      if (!previous.startsWith(header))
        throw new Error(`Cannot replace the existing command at ${path}. Move it aside and retry.`);
      if (!windows && (info.mode & 0o777) !== 0o755) await chmod(path, 0o755);
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
      const options = {
        env: { ...process.env, PATH: windows ? '' : '/usr/bin:/bin:/usr/sbin:/sbin' },
        // `delegate --help` also starts the agent CLI; cold Windows startup exceeds the POSIX budget.
        timeout: windows ? 60_000 : 10_000,
        maxBuffer: 128 * 1024,
      };
      const { stdout } = windows
        ? await execute(
            process.env.ComSpec ?? 'cmd.exe',
            ['/d', '/s', '/c', 'call', path, 'delegate', '--help'],
            options,
          )
        : await execute(path, ['delegate', '--help'], options);
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
