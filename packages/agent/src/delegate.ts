/** Addon-free helpers for the desktop's global instruction setup. */
import { homedir } from 'node:os';
import { join } from 'node:path';

import { readAgentDefaultModel } from './paths.js';

export const DELEGATION_PROMPT =
  'Delegate GitHub investigation to `mlx delegate github --caller-approved --repo OWNER/REPO "TASK"`. Approve the bounded task and its tool execution before invoking. Include the PR, issue, or run number. Wait for the final handoff and read it once; do not poll by dumping worker transcripts. Use its findings and evidence for implementation; address specific gaps with a focused follow-up or relevant evidence excerpt. Add `--allow-write` only for GitHub changes already authorized by the user. If delegation fails or reports incomplete work, continue from its handoff.';

/** Absolute paths avoid dependence on each coding agent's shell startup/PATH. */
export function delegationCommand(path: string): string {
  if (/[\0\r\n`]/.test(path)) throw new Error('The delegation command path cannot be represented in instructions.');
  return `'${path.replaceAll("'", "'\\''")}'`;
}

export function delegationPrompt(path: string): string {
  return DELEGATION_PROMPT.replace('`mlx delegate', `\`${delegationCommand(path)} delegate`);
}

export function expandHome(path: string, home = homedir()): string {
  return path === '~' ? home : path.startsWith('~/') ? join(home, path.slice(2)) : path;
}

/** Match the persisted default used by `mlx agent`. */
export async function preferredLocalModel(home = homedir(), env = process.env): Promise<string | undefined> {
  return readAgentDefaultModel(undefined, env, home);
}
