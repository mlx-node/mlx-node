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

/** Absolute path to `$HOME/.mlx-node`. */
export function mlxNodeHome(): string {
  return join(homedir(), '.mlx-node');
}

/** Directory holding per-process `MetricsTrace` JSONL files. */
export function metricsTraceDir(): string {
  return join(mlxNodeHome(), 'metrics', 'traces');
}
