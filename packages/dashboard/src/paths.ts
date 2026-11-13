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

/** The disposable SQLite index rebuilt from JSONL on demand. */
export function dashboardDbPath(): string {
  return join(mlxNodeHome(), 'dashboard.db');
}

/** Directory holding the inference runtime series `<date>-<pid>.jsonl` and the `live-<pid>.json` heartbeats. */
export function metricsRuntimeDir(): string {
  return join(mlxNodeHome(), 'metrics', 'runtime');
}

/** The desktop supervisor state file written by the Electron main process. */
export function desktopSupervisorStatePath(): string {
  return join(mlxNodeHome(), 'desktop', 'supervisor.json');
}
