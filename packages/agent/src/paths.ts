/**
 * `$HOME/.mlx-node` layout helpers owned by the agent.
 *
 * The agent must not import `@mlx-node/cli` (wrong dependency direction), so
 * the small home-directory layout it needs lives here. Mirrors
 * `resolveMlxNodeHome()` in `@mlx-node/server/host/paths`
 * (`packages/server/src/host/paths.ts`).
 */

import { readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * Expand an agent directory for both the CLI and desktop settings readers.
 * Accept home-relative paths and file URLs without trimming literal paths or
 * expanding another user's `~user` prefix. `home` is a test seam.
 */
export function expandAgentDir(dir: string, home: string = homedir()): string {
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

/**
 * The agent binary's config home: `GROK_HOME` when set, else `~/.mlx-agent`.
 * `GROK_HOME` gets the same `~`/`file://` expansion as a `--dir` flag.
 * `env`/`home` are test seams.
 */
export function agentConfigDir(env: NodeJS.ProcessEnv = process.env, home: string = homedir()): string {
  return env.GROK_HOME ? expandAgentDir(env.GROK_HOME, home) : join(home, '.mlx-agent');
}

/**
 * Read the persisted `[models] default = "…"` the agent binary wrote to
 * `config.toml` (its `/model` picker persists there). Minimal line scan —
 * TOML parsing is not worth a dependency for one key. `dir` overrides the
 * config home; `env`/`home` are test seams.
 */
export function readAgentDefaultModel(
  dir?: string,
  env: NodeJS.ProcessEnv = process.env,
  home: string = homedir(),
): string | undefined {
  try {
    const text = readFileSync(join(dir ?? agentConfigDir(env, home), 'config.toml'), 'utf8');
    return parseDefaultModelLine(text);
  } catch {
    return undefined;
  }
}

/**
 * Scan TOML lines for `default = "<value>"`, tolerating leading whitespace
 * and anything after the closing quote (a `#` comment). Returns the first
 * match — the picker writes its choice near the top of the file.
 */
function parseDefaultModelLine(text: string): string | undefined {
  for (const rawLine of text.split('\n')) {
    const line = rawLine.trim();
    if (!line.startsWith('default')) continue;
    let rest = line.slice('default'.length).trimStart();
    if (!rest.startsWith('=')) continue;
    rest = rest.slice(1).trimStart();
    if (!rest.startsWith('"')) continue;
    const end = rest.indexOf('"', 1);
    if (end < 1) continue;
    return rest.slice(1, end);
  }
  return undefined;
}
