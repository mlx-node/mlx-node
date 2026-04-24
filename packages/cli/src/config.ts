/**
 * Shared `$HOME/.mlx-node` layout helpers. Drives both `mlx download model`
 * (output destination) and `mlx launch claude` (model discovery root).
 */

import { mkdirSync, readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join, resolve } from 'node:path';

/** Absolute path to `$HOME/.mlx-node`. Used for `config.json` lookup and as the default parent of `models/`. */
export function resolveMlxNodeHome(): string {
  return join(homedir(), '.mlx-node');
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
 * Creates the chosen directory (recursive) before returning.
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

function readModelsDirFromConfig(configPath: string): string | undefined {
  let raw: string;
  try {
    raw = readFileSync(configPath, 'utf-8');
  } catch {
    // Missing / unreadable file: fall through to default.
    return undefined;
  }
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
  mkdirSync(path, { recursive: true });
  return path;
}
