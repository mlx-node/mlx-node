/** The app-owned CLI runs in the caller's process tree, using Electron's bundled Node. */
import { execFile } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { chmod, lstat, mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { dirname, isAbsolute, join } from 'node:path';
import { promisify } from 'node:util';

import { APP_NAME } from './app-name.js';
import { CliVerificationCache } from './cli-verification-cache.js';

const execute = promisify(execFile);
const POSIX_HEADER = '#!/bin/sh\n# Managed by mlx-node.\n';
const WINDOWS_HEADER = '@echo off\r\nREM Managed by mlx-node.\r\n';
const AGENT_UPDATE_MS = 600_000;

/**
 * Direct invocation of the bundled CLI through Electron's Node mode (`ELECTRON_RUN_AS_NODE=1`).
 * Spawning `config.executable` directly avoids routing through `cmd.exe`, so shell metacharacters
 * (e.g. `&`, `^`, `(`, `%`) in the user's home or install path are never reinterpreted.
 */
export function directCliInvocation(
  config: DesktopCliConfig,
  args: readonly string[],
  platform: NodeJS.Platform = process.platform,
  baseEnv: NodeJS.ProcessEnv = process.env,
): { file: string; args: string[]; env: NodeJS.ProcessEnv } {
  const env: NodeJS.ProcessEnv = {};
  let pathKey = 'PATH';
  for (const [key, value] of Object.entries(baseEnv)) {
    if (value === undefined) continue;
    const upper = key.toUpperCase();
    if (upper === 'NAPI_RS_NATIVE_LIBRARY_PATH' || upper === 'MLX_CORE_NATIVE_LIBRARY_PATH') continue;
    if (platform === 'win32' && upper === 'PATH') pathKey = key;
    env[key] = value;
  }
  env.ELECTRON_RUN_AS_NODE = '1';
  env.MLX_CORE_NATIVE_LIBRARY_PATH = config.nativeAddon;
  if (config.modelsDir) env.MLX_MODELS_DIR = config.modelsDir;
  if (platform === 'win32' && config.nativeRuntimeDir) {
    env.CUDA_HOME = config.nativeRuntimeDir;
    env.CUDA_PATH = config.nativeRuntimeDir;
    env[pathKey] = env[pathKey] ? `${config.nativeRuntimeDir};${env[pathKey]}` : config.nativeRuntimeDir;
    env.MLX_CUDA_USE_CUDNN_SDPA = '0';
  }
  return { file: config.executable, args: [config.entry, ...args], env };
}

/** `mlx agent update` through the bundled runtime directly (no `cmd.exe`). Its output is not ours to log. */
export async function runAgentUpdate(
  config: DesktopCliConfig,
  platform: NodeJS.Platform = process.platform,
): Promise<void> {
  const invocation = directCliInvocation(config, ['agent', 'update'], platform);
  try {
    await execute(invocation.file, invocation.args, {
      env: invocation.env,
      timeout: AGENT_UPDATE_MS,
      maxBuffer: 1024 * 1024,
    });
  } catch {
    throw new Error('mlx agent update failed');
  }
}

export interface DesktopCliConfig {
  executable: string;
  entry: string;
  nativeAddon: string;
  /** Set only for a packaged Windows app; development keeps its prepared CUDA_HOME. */
  nativeRuntimeDir?: string | null;
  modelsDir: string | null;
}

const quote = (value: string): string => `'${value.replaceAll("'", "'\\''")}'`;
const batchValue = (value: string): string => value.replaceAll('%', '%%');

export function launcherScript(config: DesktopCliConfig, platform: NodeJS.Platform = process.platform): string {
  for (const path of [
    config.executable,
    config.entry,
    config.nativeAddon,
    ...(config.nativeRuntimeDir == null ? [] : [config.nativeRuntimeDir]),
  ]) {
    if (!isAbsolute(path) || /[\0\r\n]/.test(path)) throw new Error('The app command has an invalid runtime path.');
  }
  if (platform === 'win32') {
    const runtimeDir = config.nativeRuntimeDir == null ? null : batchValue(config.nativeRuntimeDir);
    const runtimeEnv =
      runtimeDir === null
        ? ''
        : `set "CUDA_HOME=${runtimeDir}"\r\nset "CUDA_PATH=${runtimeDir}"\r\nset "PATH=${runtimeDir};%PATH%"\r\nset "MLX_CUDA_USE_CUDNN_SDPA=0"\r\n`;
    return `${WINDOWS_HEADER}setlocal DisableDelayedExpansion\r\nset "ELECTRON_RUN_AS_NODE=1"\r\nset "NAPI_RS_NATIVE_LIBRARY_PATH="\r\nset "MLX_CORE_NATIVE_LIBRARY_PATH=${batchValue(config.nativeAddon)}"\r\n${runtimeEnv}${
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
      // On Windows, invoke the bundled runtime directly instead of routing `path` through
      // `cmd.exe /c call`, so shell metacharacters in `home` cannot break verification.
      const baseEnv = { ...process.env, PATH: windows ? '' : '/usr/bin:/bin:/usr/sbin:/sbin' };
      const timeout = windows ? 60_000 : 10_000;
      const maxBuffer = 128 * 1024;
      const { stdout } = windows
        ? await (() => {
            const direct = directCliInvocation(config, ['delegate', '--help'], platform, baseEnv);
            return execute(direct.file, direct.args, { env: direct.env, timeout, maxBuffer });
          })()
        : await execute(path, ['delegate', '--help'], { env: baseEnv, timeout, maxBuffer });
      if (!stdout.includes('Usage: mlx delegate') || !stdout.includes('mlx delegate github')) {
        throw new Error(`The bundled command does not support delegation. Reinstall or update ${APP_NAME}.`);
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
    async runAgentUpdate(): Promise<void> {
      await this.prepare();
      await runAgentUpdate(config, platform);
    },
  };
}
