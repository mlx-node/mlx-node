import { homedir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

/** Absolute path to `$HOME/.mlx-node`. */
export function mlxNodeHome(): string {
  return join(homedir(), '.mlx-node');
}

/**
 * Expand a `PI_CODING_AGENT_DIR` value the same way pi 0.80.6 does
 * (`getAgentDir` → `expandTildePath`): a lone `~` or a leading `~/` (`~\` on
 * Windows) becomes `$HOME`, a `file://` URL becomes its path, everything else
 * (including `~user`) passes through verbatim. Byte-identical to the CLI's
 * `expandPiAgentDir` (packages/cli/src/commands/agent/index.ts); the dashboard
 * cannot import that helper because `@mlx-node/dashboard` may not depend on
 * `@mlx-node/cli` (wrong direction), so the rule is mirrored here and the two
 * MUST change together — otherwise the dashboard would read a different sessions
 * dir than pi actually writes.
 */
function expandPiAgentDir(dir: string): string {
  const home = homedir();
  if (dir === '~') return home;
  if (dir.startsWith('~/') || (process.platform === 'win32' && dir.startsWith('~\\'))) {
    return join(home, dir.slice(2));
  }
  if (dir.startsWith('file://')) return fileURLToPath(dir);
  return dir;
}

/**
 * Directory holding pi session JSONL files, one subdir per encoded cwd. Honors
 * `PI_CODING_AGENT_DIR` — pi's agent-home override, which the agent seeds and pi
 * resolves — so sessions written under a custom agent home stay discoverable;
 * otherwise the default `~/.mlx-node/agent`. An explicit `--session-dir` on
 * `mlx dashboard` takes precedence over this and is wired in the CLI.
 */
export function agentSessionsRoot(): string {
  const envDir = process.env.PI_CODING_AGENT_DIR;
  const agentDir = envDir ? expandPiAgentDir(envDir) : join(mlxNodeHome(), 'agent');
  return join(agentDir, 'sessions');
}

/** Directory holding per-process `MetricsTrace` JSONL files. */
export function metricsTraceDir(): string {
  return join(mlxNodeHome(), 'metrics', 'traces');
}

/** The disposable SQLite index rebuilt from JSONL on demand. */
export function dashboardDbPath(): string {
  return join(mlxNodeHome(), 'dashboard.db');
}
