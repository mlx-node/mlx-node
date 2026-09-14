/**
 * `$HOME/.mlx-node` layout helpers owned by the agent.
 *
 * The agent must not import `@mlx-node/cli` (wrong dependency direction), so
 * the small home-directory layout it needs lives here. Mirrors
 * `resolveMlxNodeHome()` in `@mlx-node/server/host/paths`
 * (`packages/server/src/host/paths.ts`).
 */

import { homedir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * Expand an agent directory for both the CLI and desktop settings readers.
 * Accept home-relative paths and file URLs without trimming literal paths or
 * expanding another user's `~user` prefix. `home` is a test seam.
 */
export function expandPiAgentDir(dir: string, home: string = homedir()): string {
  if (dir === '~') return home;
  if (dir.startsWith('~/') || (process.platform === 'win32' && dir.startsWith('~\\'))) {
    return join(home, dir.slice(2));
  }
  if (dir.startsWith('file://')) return fileURLToPath(dir);
  return dir;
}

/** Absolute path to `$HOME/.mlx-node`. */
export function mlxNodeHome(): string {
  return join(homedir(), '.mlx-node');
}

/** Directory holding per-process `MetricsTrace` JSONL files. */
export function metricsTraceDir(): string {
  return join(mlxNodeHome(), 'metrics', 'traces');
}
