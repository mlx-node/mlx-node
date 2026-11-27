/**
 * Shared `$HOME/.mlx-node` layout helpers. Drives `mlx download model`
 * (output destination), every inference host (`mlx serve`, `mlx launch
 * claude`, the desktop sidecar) as the model discovery root, and the
 * default log root.
 *
 * Published as its own `@mlx-node/server/host/paths` subpath, and it must
 * stay a dependency-free leaf. `@mlx-node/server/host` value-imports
 * `@mlx-node/lm`, which dlopens the native addon at module scope; commands
 * that only need to know where models live — `mlx download model` above all,
 * the one you run BEFORE you have anything to run — must not pay for that,
 * and on a headless box the dlopen is a liability rather than a cost.
 */

import { lstatSync, mkdirSync, readFileSync, readlinkSync, statSync } from 'node:fs';
import { lstat, mkdir, readFile, readlink, stat } from 'node:fs/promises';
import { homedir } from 'node:os';
import { dirname, join, resolve } from 'node:path';

/**
 * The models folder is a link whose target does not exist, such as a folder on
 * a disk that is not connected. Nothing can be served or downloaded until the
 * target is back, so the desktop sidecar treats it as a permanent start failure.
 */
export class ModelsDirUnavailableError extends Error {
  constructor(
    readonly modelsDir: string,
    readonly linkTarget: string,
  ) {
    super(modelsDirUnavailableMessage(modelsDir, linkTarget));
    this.name = 'ModelsDirUnavailableError';
  }
}

/** The user-facing text of {@link ModelsDirUnavailableError}; the dashboard's `modelsDirUnavailable` says the same. */
export function modelsDirUnavailableMessage(modelsDir: string, linkTarget: string): string {
  return `The models folder ${modelsDir} is a link to ${linkTarget}, which is not available. Connect that disk, or change the link, then try again.`;
}

/** Absolute path to `$HOME/.mlx-node`. Used for `config.json` lookup and as the default parent of `models/`. */
export function resolveMlxNodeHome(): string {
  return join(homedir(), '.mlx-node');
}

/**
 * Resolve the log directory for a verbose launch.
 *
 * Order: explicit `--log-dir` > `MLX_LOG_DIR` env > a fresh timestamped
 * directory under `<mlxNodeHome>/logs/`. The timestamped default gives
 * each launch its own dir so concurrent / sequential runs don't
 * interleave into one file.
 */
export function resolveLogDir(explicit: string | undefined, mlxNodeHome: string): string {
  if (explicit) return explicit;
  const envDir = process.env.MLX_LOG_DIR;
  if (envDir) return envDir;
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  return join(mlxNodeHome, 'logs', stamp);
}

/**
 * Resolve the directory where downloaded models live.
 *
 * Resolution order:
 *   1. `explicit` arg (non-empty)
 *   2. `MLX_MODELS_DIR` env var
 *   3. `modelsDir` field in `$HOME/.mlx-node/config.json`
 *   4. `$HOME/.mlx-node/models`
 *
 * Creates the chosen directory (recursive) before returning. Throws
 * {@link ModelsDirUnavailableError} when it is a link whose target is missing.
 */
export function resolveModelsDir(explicit?: string): string {
  if (explicit && explicit.length > 0) {
    return ensureDir(resolve(explicit));
  }

  const envDir = process.env.MLX_MODELS_DIR;
  if (envDir && envDir.length > 0) {
    return ensureDir(resolve(envDir));
  }

  const configPath = join(resolveMlxNodeHome(), 'config.json');
  const fromConfig = readModelsDirFromConfig(configPath);
  if (fromConfig) {
    return ensureDir(resolve(fromConfig));
  }

  return ensureDir(join(resolveMlxNodeHome(), 'models'));
}

/**
 * Async twin of {@link resolveModelsDir} with the identical resolution order.
 * Use it where synchronous filesystem I/O is off limits — the desktop MAIN
 * process's event loop, where a `readFileSync`/`mkdirSync` on a stalled mount
 * would freeze window IPC, the tray, and quit handling until it returned.
 */
export async function resolveModelsDirAsync(explicit?: string): Promise<string> {
  if (explicit && explicit.length > 0) {
    return ensureDirAsync(resolve(explicit));
  }

  const envDir = process.env.MLX_MODELS_DIR;
  if (envDir && envDir.length > 0) {
    return ensureDirAsync(resolve(envDir));
  }

  const configPath = join(resolveMlxNodeHome(), 'config.json');
  const fromConfig = await readModelsDirFromConfigAsync(configPath);
  if (fromConfig) {
    return ensureDirAsync(resolve(fromConfig));
  }

  return ensureDirAsync(join(resolveMlxNodeHome(), 'models'));
}

function readModelsDirFromConfig(configPath: string): string | undefined {
  let raw: string;
  try {
    raw = readFileSync(configPath, 'utf-8');
  } catch {
    // Missing / unreadable file: fall through to default.
    return undefined;
  }
  return parseModelsDirConfig(raw, configPath);
}

async function readModelsDirFromConfigAsync(configPath: string): Promise<string | undefined> {
  let raw: string;
  try {
    raw = await readFile(configPath, 'utf-8');
  } catch {
    // Missing / unreadable file: fall through to default.
    return undefined;
  }
  return parseModelsDirConfig(raw, configPath);
}

function parseModelsDirConfig(raw: string, configPath: string): string | undefined {
  try {
    const parsed = JSON.parse(raw) as { modelsDir?: unknown };
    if (typeof parsed.modelsDir === 'string' && parsed.modelsDir.length > 0) {
      return parsed.modelsDir;
    }
    return undefined;
  } catch {
    console.warn(`[mlx] warning: malformed JSON in ${configPath}; falling back to default models dir`);
    return undefined;
  }
}

function ensureDir(path: string): string {
  try {
    mkdirSync(path, { recursive: true });
  } catch (error) {
    throw explainMkdirError(error, path, missingLinkTarget(path));
  }
  return path;
}

async function ensureDirAsync(path: string): Promise<string> {
  try {
    await mkdir(path, { recursive: true });
  } catch (error) {
    throw explainMkdirError(error, path, await missingLinkTargetAsync(path));
  }
  return path;
}

/** Where `path` points when it is a link whose target is missing (`lstat` a link, `stat` ENOENT); else null. */
function missingLinkTarget(path: string): string | null {
  try {
    if (!lstatSync(path).isSymbolicLink()) return null;
  } catch {
    return null;
  }
  try {
    statSync(path);
    return null;
  } catch (error) {
    if (errorCode(error) !== 'ENOENT') return null;
  }
  try {
    return resolve(dirname(path), readlinkSync(path));
  } catch {
    return null;
  }
}

async function missingLinkTargetAsync(path: string): Promise<string | null> {
  try {
    if (!(await lstat(path)).isSymbolicLink()) return null;
  } catch {
    return null;
  }
  try {
    await stat(path);
    return null;
  } catch (error) {
    if (errorCode(error) !== 'ENOENT') return null;
  }
  try {
    return resolve(dirname(path), await readlink(path));
  } catch {
    return null;
  }
}

/**
 * A missing link target gets its own error. A file in the way (EEXIST for the
 * folder itself, ENOTDIR for a folder above it) gets one sentence more, since
 * Node's text names only the syscall. Every other error is left as it is.
 */
function explainMkdirError(error: unknown, path: string, linkTarget: string | null): unknown {
  if (linkTarget !== null) return new ModelsDirUnavailableError(path, linkTarget);
  const code = errorCode(error);
  if (error instanceof Error && (code === 'EEXIST' || code === 'ENOTDIR')) {
    return new Error(
      `${error.message}. The models folder ${path} cannot be made: it, or a folder above it, is a file.`,
      { cause: error },
    );
  }
  return error;
}

function errorCode(error: unknown): unknown {
  return typeof error === 'object' && error !== null ? (error as { code?: unknown }).code : undefined;
}
