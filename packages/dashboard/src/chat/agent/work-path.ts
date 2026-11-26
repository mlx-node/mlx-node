/**
 * The `PATH` a Work command gets. It is not the app's own: `mlx-agent` asks the user's login shell for `$PATH` once per
 * session, puts its own entries after the answer, and runs each command with `<shell> -c`, which reads `~/.zshenv`
 * (zsh) or `$BASH_ENV` (bash) once more. This file asks the same two questions, following mlx-agent v0.3.3:
 * `capture_login_env`, `spawn_static_command` (`spawn_shell_command` with capture off) and `shell_env_overrides` in `xai-grok-tools/src/computer/local/`
 * (`terminal.rs`, `shell_state.rs`) and the shell choice in `xai-grok-config/src/shell.rs`. Only `$PATH` is printed and
 * read back; the agent's `env -0` part is left out.
 */

import { type ChildProcess, spawn } from 'node:child_process';
import { stat } from 'node:fs/promises';
import { basename } from 'node:path';

import { findOnPath, readPathEnv } from '../skills/cli-check.js';
import { workInherited } from './env.js';
import { mergePath } from './login-path.js';

type ShellKind = 'zsh' | 'bash';

const MARK = 0x01;
/** Limit on the marked text only; rc output before the first mark is read and dropped, of any size, as in the agent. */
const OUT_CAP = 1024 * 1024;
const PRINT_PATH = `printf '\\x01%s\\x01' "$PATH"`;
/** Set by the agent on every command it runs. */
const COMMAND_ENV: Readonly<Record<string, string>> = {
  TERM: 'dumb',
  NO_COLOR: '1',
  FORCE_COLOR: '0',
  GROK_AGENT: '1',
};
const SHELL_DIRS = ['/bin', '/usr/bin', '/usr/local/bin', '/opt/homebrew/bin'];

async function executable(path: string): Promise<boolean> {
  try {
    const info = await stat(path);
    return info.isFile() && (info.mode & 0o111) !== 0;
  } catch {
    return false;
  }
}

async function shellBinary(kind: ShellKind, env: Record<string, string>): Promise<string> {
  for (const named of [env.GROK_SHELL, env.SHELL]) {
    if (named !== undefined && basename(named) === kind && (await executable(named))) return named;
  }
  return (await findOnPath(kind, [env.PATH ?? '', ...SHELL_DIRS].join(':'))) ?? `/bin/${kind}`;
}

/**
 * One shell run: the text between the first two marks (or from the first to the end), trimmed, when it exits 0 inside
 * the limit; null when that text is empty, absent or over `OUT_CAP`. It resolves on `close`, as the agent's read waits
 * for end of file, so an rc job that keeps the pipe open runs into the limit; then the whole group is killed.
 */
function askShell(
  binary: string,
  args: string[],
  env: Record<string, string>,
  cwd: string,
  timeoutMs: number,
): Promise<string | null> {
  return new Promise((resolve) => {
    let child: ChildProcess;
    try {
      child = spawn(binary, args, { env, cwd, stdio: ['ignore', 'pipe', 'ignore'], detached: true });
    } catch {
      resolve(null);
      return;
    }
    let settled = false;
    const done = (answer: string | null): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(answer);
    };
    const timer = setTimeout(() => {
      if (child.pid !== undefined) {
        try {
          process.kill(-child.pid, 'SIGKILL');
        } catch {
          // The group is gone already.
        }
      }
      child.stdout?.destroy();
      done(null);
    }, timeoutMs);
    const kept: Buffer[] = [];
    let size = 0;
    let opened = false;
    let closed = false;
    child.stdout?.on('data', (chunk: Buffer) => {
      if (closed) return;
      let part = chunk;
      if (!opened) {
        const start = part.indexOf(MARK);
        if (start === -1) return;
        opened = true;
        part = part.subarray(start + 1);
      }
      const end = part.indexOf(MARK);
      if (end !== -1) {
        closed = true;
        part = part.subarray(0, end);
      }
      size += part.length;
      if (size > OUT_CAP) {
        closed = true;
        return;
      }
      kept.push(part);
    });
    const answer = (): string | null => {
      if (!opened || size > OUT_CAP) return null;
      const text = Buffer.concat(kept).toString('utf8').trim();
      return text === '' ? null : text;
    };
    child.on('error', () => done(null));
    child.on('close', (code) => done(code === 0 ? answer() : null));
  });
}

/**
 * The `PATH` a Work command gets, for the Work environment `base` (both shells run in `home`). A login shell that
 * fails or runs past the limit gives `base.PATH`, as in the agent; then the command shell is still asked. Never
 * rejects; writes nothing.
 */
export async function workCommandPath(
  base: NodeJS.ProcessEnv,
  home: string,
  timeoutMs = 5_000,
  platform: NodeJS.Platform = process.platform,
): Promise<string | undefined> {
  if (platform === 'win32') return readPathEnv(base);
  const env = workInherited(base, platform);
  const kind: ShellKind = (env.SHELL ?? '').includes('zsh') ? 'zsh' : 'bash';
  const binary = await shellBinary(kind, env);
  const login = await askShell(
    binary,
    ['-lc', `source "$HOME/.${kind}rc" 2>/dev/null; ${PRINT_PATH}`],
    env,
    home,
    timeoutMs,
  );
  const path = login === null ? env.PATH : mergePath(login, env.PATH);
  const commandEnv: Record<string, string> = { ...env, ...COMMAND_ENV };
  if (path === undefined) delete commandEnv.PATH;
  else commandEnv.PATH = path;
  const args = kind === 'zsh' ? ['-o', 'nonomatch', '-c', PRINT_PATH] : ['-c', PRINT_PATH];
  return (await askShell(binary, args, commandEnv, home, timeoutMs)) ?? path;
}
