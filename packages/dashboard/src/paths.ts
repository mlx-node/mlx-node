import { homedir } from 'node:os';
import { join } from 'node:path';

/** Absolute path to `$HOME/.mlx-node`. */
export function mlxNodeHome(): string {
  return join(homedir(), '.mlx-node');
}

/** Directory holding pi session JSONL files, one subdir per encoded cwd. */
export function agentSessionsRoot(): string {
  return join(mlxNodeHome(), 'agent', 'sessions');
}

/** Directory holding per-process `MetricsTrace` JSONL files. */
export function metricsTraceDir(): string {
  return join(mlxNodeHome(), 'metrics', 'traces');
}

/** The disposable SQLite index rebuilt from JSONL on demand. */
export function dashboardDbPath(): string {
  return join(mlxNodeHome(), 'dashboard.db');
}
