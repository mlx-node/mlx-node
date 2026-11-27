/**
 * The user's login-shell `PATH` for the Work agent. An app started from Finder or the Dock gets launchd's `PATH`, and
 * the agent runs plugin hooks (`sh -c`) and MCP servers with its own `PATH`, so a hook's `node` is not found. Its shell
 * commands are not affected: the agent asks the login shell for them itself (`work-path.ts`). Only `PATH` is read back;
 * an rc file may export provider keys, which Work strips on purpose.
 */

import { type ChildProcess, spawn } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { userInfo } from 'node:os';
import { basename } from 'node:path';

/** launchd's `PATH` for an app started from Finder or the Dock. */
export const LAUNCHD_PATH = '/usr/bin:/bin:/usr/sbin:/sbin';
export const LOGIN_PATH_TIMEOUT_MS = 5_000;
const OUT_CAP = 1024 * 1024;
const POSIX_SHELLS = new Set(['sh', 'bash', 'zsh', 'ksh', 'dash', 'mksh', 'yash']);

export interface Markers {
  begin: string;
  end: string;
}

/** `first` then `then`, empty entries dropped, the first occurrence kept. */
export function mergePath(first: string, then: string | undefined): string {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const entry of [...first.split(':'), ...(then ?? '').split(':')]) {
    if (entry === '' || seen.has(entry)) continue;
    seen.add(entry);
    out.push(entry);
  }
  return out.join(':');
}

/** `base` with the login entries first in its `PATH`: what the Work agent and the Firecrawl check start from. */
export function withLoginPath(base: NodeJS.ProcessEnv, loginPath: string | null | undefined): NodeJS.ProcessEnv {
  return loginPath == null ? base : { ...base, PATH: mergePath(loginPath, base.PATH) };
}

/** In the desktop app (`MLX_DESKTOP_CLI`), or under launchd's `PATH`. A terminal's `PATH` is the user's already. */
export function needsLoginPath(base: NodeJS.ProcessEnv, platform: NodeJS.Platform = process.platform): boolean {
  if (platform === 'win32') return false;
  return base.MLX_DESKTOP_CLI !== undefined || base.PATH === LAUNCHD_PATH;
}

/** The text between the first `begin` and the next `end`; null when either is missing or nothing is between. */
export function markedPath(output: string, markers: Markers): string | null {
  const start = output.indexOf(markers.begin);
  if (start === -1) return null;
  const from = start + markers.begin.length;
  const stop = output.indexOf(markers.end, from);
  if (stop === -1 || stop === from) return null;
  return output.slice(from, stop);
}

/** The `-c` script that prints `PATH` between the markers; null for a shell that is neither POSIX nor fish. */
export function loginPathScript(shell: string, markers: Markers): string | null {
  const name = basename(shell);
  // fish keeps PATH as a list; a bare `$PATH` would print it space-separated.
  if (name === 'fish') return `printf '%s%s%s' '${markers.begin}' (string join : $PATH) '${markers.end}'`;
  if (POSIX_SHELLS.has(name)) return `printf '%s%s%s' '${markers.begin}' "$PATH" '${markers.end}'`;
  return null;
}

function userShell(): string {
  try {
    return userInfo().shell || '/bin/zsh';
  } catch {
    return '/bin/zsh';
  }
}

export interface LoginPathInput {
  /** The Work base env: only `HOME`, `USER`, `LOGNAME`, `TMPDIR` and `PATH` reach the shell. */
  base: NodeJS.ProcessEnv;
  /** The shell's cwd, and its `HOME` when `base` has none. */
  home: string;
  /** Default: the shell of the user record, else `/bin/zsh`. */
  shell?: string;
  timeoutMs?: number;
  log?: (line: string) => void;
}

type Answer = { path: string } | { failure: string };

function askLoginShell(
  shell: string,
  script: string,
  markers: Markers,
  env: Record<string, string>,
  cwd: string,
  timeoutMs: number,
): Promise<Answer> {
  return new Promise((resolve) => {
    let child: ChildProcess;
    try {
      // Own process group, so a timeout kills what the rc files started too.
      child = spawn(shell, ['-ilc', script], { env, cwd, stdio: ['ignore', 'pipe', 'ignore'], detached: true });
    } catch (error) {
      resolve({ failure: error instanceof Error ? error.message : String(error) });
      return;
    }
    let settled = false;
    let exited = false;
    let out = '';
    const settle = (answer: Answer): void => {
      if (settled) return;
      settled = true;
      resolve(answer);
    };
    const timer = setTimeout(() => {
      if (!exited && child.pid !== undefined) {
        try {
          process.kill(-child.pid, 'SIGKILL');
        } catch {
          // The group is gone already.
        }
      }
      child.stdout?.destroy();
      settle({ failure: `no PATH in ${timeoutMs} ms` });
    }, timeoutMs);
    child.stdout?.setEncoding('utf8');
    child.stdout?.on('data', (chunk: string) => {
      if (settled) return;
      out += chunk;
      const path = markedPath(out, markers);
      if (path !== null) settle({ path });
      else if (out.length > OUT_CAP) settle({ failure: 'over 1 MiB of output before the PATH' });
    });
    child.on('exit', () => {
      exited = true;
    });
    child.on('error', (error) => {
      clearTimeout(timer);
      settle({ failure: error.message });
    });
    child.on('close', (code, signal) => {
      clearTimeout(timer);
      settle({ failure: `exited ${code ?? signal} with no PATH printed` });
    });
  });
}

/**
 * The PATH of `<shell> -ilc`, read once per app run; null when the shell is not POSIX or fish, fails, prints no PATH
 * or runs past the limit, with one log line. Never rejects.
 */
export async function resolveLoginPath(input: LoginPathInput): Promise<string | null> {
  const shell = input.shell ?? userShell();
  const log = input.log ?? ((line: string) => console.error(line));
  const nonce = randomBytes(8).toString('hex');
  const markers = { begin: `MLX_PATH_BEGIN_${nonce}`, end: `MLX_PATH_END_${nonce}` };
  const script = loginPathScript(shell, markers);
  if (script === null) {
    log(`[work] login shell ${shell} is neither POSIX nor fish; Work keeps the app PATH`);
    return null;
  }
  const env: Record<string, string> = { HOME: input.base.HOME ?? input.home, SHELL: shell, TERM: 'dumb' };
  for (const name of ['USER', 'LOGNAME', 'TMPDIR', 'PATH']) {
    const value = input.base[name];
    if (value !== undefined) env[name] = value;
  }
  const answer = await askLoginShell(shell, script, markers, env, input.home, input.timeoutMs ?? LOGIN_PATH_TIMEOUT_MS);
  if ('path' in answer) return answer.path;
  log(`[work] login shell ${shell} gave no PATH (${answer.failure}); Work keeps the app PATH`);
  return null;
}
