/** Addon-free helpers for the desktop's global instruction setup. */
import { homedir } from 'node:os';
import { join, win32 } from 'node:path';

import { readAgentDefaultModel } from './paths.js';

export const DELEGATION_PROMPT =
  'Delegate GitHub investigation to `mlx delegate github --caller-approved --repo OWNER/REPO "TASK"`. Approve the bounded task and its tool execution before invoking. Include the PR, issue, or run number. Wait for the final handoff and read it once; do not poll by dumping worker transcripts. Use its findings and evidence for implementation; address specific gaps with a focused follow-up or relevant evidence excerpt. Add `--allow-write` only for GitHub changes already authorized by the user. If delegation fails or reports incomplete work, continue from its handoff.';

/** Absolute paths avoid dependence on each coding agent's shell startup/PATH. */
export function delegationCommand(path: string, platform: NodeJS.Platform = process.platform): string {
  if (platform === 'win32') {
    if (/[\0\r\n`"]/.test(path)) throw new Error('The delegation command path cannot be represented in instructions.');
    // PowerShell treats a bare quoted string as an expression and recognizes ASCII
    // `'` plus typographic single quotes (`\u2018`-`\u201B`) as delimiters; `& '...'`
    // invokes the path literally when every single-quote delimiter is doubled.
    return `& '${path.replace(/['\u2018\u2019\u201a\u201b]/g, '$&$&')}'`;
  }
  if (/[\0\r\n`]/.test(path)) throw new Error('The delegation command path cannot be represented in instructions.');
  return `'${path.replaceAll("'", "'\\''")}'`;
}

export function delegationPrompt(path: string, platform: NodeJS.Platform = process.platform): string {
  const command = delegationCommand(path, platform);
  // Use a callback so `$&` / `$$` / `` $` `` / `$'` in `command` are never interpreted by String#replace.
  return DELEGATION_PROMPT.replace('`mlx delegate', () => `\`${command} delegate`);
}

export function expandHome(path: string, home = homedir(), platform: NodeJS.Platform = process.platform): string {
  if (path === '~') return home;
  if (path.startsWith('~/') || (platform === 'win32' && path.startsWith('~\\'))) {
    const rest = path.slice(2);
    return platform === 'win32' && (/^[a-zA-Z]:[\\/]/.test(home) || home.startsWith('\\\\'))
      ? win32.join(home, rest)
      : join(home, rest);
  }
  return path;
}

/** Match the persisted default used by `mlx agent`. */
export async function preferredLocalModel(home = homedir(), env = process.env): Promise<string | undefined> {
  return readAgentDefaultModel(undefined, env, home);
}
