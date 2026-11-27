/**
 * The branch of a Work folder, for the New screen's tray. Git runs with no shell, its own process group, a clean
 * environment and a deadline; reads take no optional lock. The only write is `git switch` to a local branch the read
 * just listed: never `--force`, `--discard-changes`, `-c` or `--detach`. A switch runs the folder's hooks and filters,
 * as `git switch` in a terminal does.
 */

import { spawn, type ChildProcess } from 'node:child_process';
import { constants } from 'node:fs';
import { access, lstat, realpath } from 'node:fs/promises';
import { dirname, join } from 'node:path';

import { mergePath } from './agent/login-path.js';
import { folderKey } from './agent/never-rules.js';
import { below } from './agent/roots.js';
import { visibleText } from './agent/visible-text.js';
import { findOnPath, readPathEnv } from './skills/cli-check.js';
import type { WorkHead } from './types.js';

export interface GitDeps {
  platform: NodeJS.Platform;
  /** macOS's Command Line Tools shim: run blind, it can raise the install prompt. */
  shim: string;
  /** The developer folder `xcode-select -p` names, or null. */
  developerDir: (env: Readonly<Record<string, string>>) => Promise<string | null>;
  /** One deadline for every command of a read. */
  readMs: number;
  /** A switch past this gets SIGTERM, then SIGKILL `killGraceMs` later. */
  switchMs: number;
  killGraceMs: number;
}

const XCODE_SELECT = '/usr/bin/xcode-select';
const XCODE_SELECT_MS = 2_000;

export const GIT_DEPS: GitDeps = {
  platform: process.platform,
  shim: '/usr/bin/git',
  developerDir: async (env) => {
    const run = await runBin(XCODE_SELECT, ['-p'], env, { timeoutMs: XCODE_SELECT_MS });
    if (run.ended !== 'exit' || run.code !== 0) return null;
    const dir = withoutNewline(run.stdout);
    return dir === '' ? null : dir;
  },
  readMs: 5_000,
  switchMs: 120_000,
  killGraceMs: 2_000,
};

/** `switch` and `%(worktreepath)` came in 2.23. */
const MIN_MAJOR = 2;
const MIN_MINOR = 23;
const STDOUT_CAP = 4 * 1024 * 1024;
const STDERR_CAP = 64 * 1024;
/** The branches a read shows, the current one always among them. */
export const BRANCHES_SHOWN = 1_000;
const REFUSAL_LINES = 12;
const REFUSAL_CHARS = 2_048;
const HEADS = 'refs/heads/';
/** Settings every git call gets: no file-system monitor daemon, plain text. */
const GIT_SETTINGS = ['-c', 'core.fsmonitor=false', '-c', 'color.ui=false', '-c', 'core.quotePath=false'];

export interface GitRun {
  ended: 'exit' | 'timeout' | 'error';
  code: number | null;
  stdout: string;
  stderr: string;
}

interface RunOptions {
  timeoutMs: number;
  /** SIGTERM first, SIGKILL this much later. Absent: SIGKILL at once. */
  termGraceMs?: number;
}

function killGroup(child: ChildProcess, signal: NodeJS.Signals, platform: NodeJS.Platform): void {
  if (child.pid === undefined) return;
  try {
    if (platform === 'win32') child.kill(signal);
    else process.kill(-child.pid, signal);
  } catch {
    // The group is gone already.
  }
}

/** Bytes kept up to `cap`; the rest is read and dropped, so the child never blocks on a full pipe. */
function collect(stream: NodeJS.ReadableStream | null, cap: number): () => string {
  const parts: Buffer[] = [];
  let size = 0;
  stream?.on('data', (chunk: Buffer) => {
    if (size >= cap) return;
    const part = size + chunk.length > cap ? chunk.subarray(0, cap - size) : chunk;
    parts.push(part);
    size += part.length;
  });
  return () => Buffer.concat(parts).toString('utf8');
}

/** One run of `bin` with no shell, stdin closed, its own process group, and a deadline. Never rejects. */
export function runBin(
  bin: string,
  args: readonly string[],
  env: Readonly<Record<string, string>>,
  options: RunOptions,
  platform: NodeJS.Platform = process.platform,
): Promise<GitRun> {
  return new Promise((resolve) => {
    let child: ChildProcess;
    try {
      child = spawn(bin, args, {
        env: { ...env },
        stdio: ['ignore', 'pipe', 'pipe'],
        detached: platform !== 'win32',
        windowsHide: true,
      });
    } catch {
      resolve({ ended: 'error', code: null, stdout: '', stderr: '' });
      return;
    }
    const out = collect(child.stdout, STDOUT_CAP);
    const err = collect(child.stderr, STDERR_CAP);
    let settled = false;
    let timedOut = false;
    let exited = false;
    let killed = false;
    const timers: Array<ReturnType<typeof setTimeout>> = [];
    const settle = (run: GitRun): void => {
      if (settled) return;
      settled = true;
      for (const timer of timers) clearTimeout(timer);
      resolve(run);
    };
    const endTimeout = (): void => {
      killed = true;
      // A grandchild that left the group may hold the pipes open; the reply does not wait for it.
      child.stdout?.destroy();
      child.stderr?.destroy();
      if (exited) settle({ ended: 'timeout', code: null, stdout: out(), stderr: err() });
    };
    timers.push(
      setTimeout(() => {
        timedOut = true;
        if (options.termGraceMs === undefined) {
          killGroup(child, 'SIGKILL', platform);
          endTimeout();
          return;
        }
        killGroup(child, 'SIGTERM', platform);
        timers.push(
          setTimeout(() => {
            killGroup(child, 'SIGKILL', platform);
            endTimeout();
          }, options.termGraceMs),
        );
      }, options.timeoutMs),
    );
    child.on('error', () => settle({ ended: 'error', code: null, stdout: '', stderr: '' }));
    child.on('exit', () => {
      exited = true;
      if (timedOut && killed) settle({ ended: 'timeout', code: null, stdout: out(), stderr: err() });
    });
    child.on('close', (code) => {
      if (timedOut) {
        if (killed) settle({ ended: 'timeout', code: null, stdout: out(), stderr: err() });
        return;
      }
      settle({ ended: 'exit', code, stdout: out(), stderr: err() });
    });
  });
}

function runGit(
  bin: string,
  dir: string,
  args: readonly string[],
  env: Readonly<Record<string, string>>,
  options: RunOptions,
  platform: NodeJS.Platform,
): Promise<GitRun> {
  return runBin(bin, ['-C', dir, ...GIT_SETTINGS, ...args], env, options, platform);
}

/**
 * The environment git gets, built from nothing: no key and no inherited `GIT_*` reach it. `PATH`: the login shell's
 * entries first, so the git a terminal finds is the one found here.
 */
export function gitEnv(
  base: NodeJS.ProcessEnv,
  home: string,
  loginPath: string | null,
  platform: NodeJS.Platform,
): Record<string, string> {
  const path = platform === 'win32' ? (readPathEnv(base) ?? '') : mergePath(loginPath ?? '', readPathEnv(base));
  const env: Record<string, string> = {
    HOME: base.HOME !== undefined && base.HOME !== '' ? base.HOME : home,
    PATH: path,
    LC_ALL: 'C',
    GIT_TERMINAL_PROMPT: '0',
  };
  for (const name of ['USER', 'LOGNAME', 'TMPDIR']) {
    const value = base[name];
    if (value !== undefined && value !== '') env[name] = value;
  }
  return env;
}

function readEnv(env: Readonly<Record<string, string>>): Record<string, string> {
  return { ...env, GIT_OPTIONAL_LOCKS: '0' };
}

function withoutNewline(text: string): string {
  return text.endsWith('\n') ? text.slice(0, -1) : text;
}

/** The digits at the start of `text` as a number, or null when it starts with none. */
function leadingNumber(text: string): number | null {
  let end = 0;
  while (end < text.length && text.charCodeAt(end) >= 0x30 && text.charCodeAt(end) <= 0x39) end += 1;
  return end === 0 ? null : Number(text.slice(0, end));
}

/** `git version 2.39.5 (Apple Git-154)`: 2.23 or newer. */
export function versionOk(output: string): boolean {
  const words = withoutNewline(output).split(' ');
  if (words[0] !== 'git' || words[1] !== 'version' || words[2] === undefined) return false;
  const parts = words[2].split('.');
  const major = leadingNumber(parts[0] ?? '');
  const minor = leadingNumber(parts[1] ?? '');
  if (major === null || minor === null) return false;
  return major > MIN_MAJOR || (major === MIN_MAJOR && minor >= MIN_MINOR);
}

/**
 * The git binary to run, or null. The first `git` on `env.PATH`; on macOS the shim at `/usr/bin/git` is replaced by
 * the git of the developer folder `xcode-select -p` names. Below 2.23 it is null too.
 */
export async function resolveGit(env: Readonly<Record<string, string>>, deps: GitDeps): Promise<string | null> {
  let found = await findOnPath('git', env.PATH, { platform: deps.platform });
  if (found === null) return null;
  if (deps.platform === 'darwin') {
    const real = await realpath(found).catch(() => found!);
    if (found === deps.shim || real === deps.shim) {
      const dir = await deps.developerDir(env);
      if (dir === null) return null;
      const tool = join(dir, 'usr', 'bin', 'git');
      try {
        await access(tool, constants.X_OK);
      } catch {
        return null;
      }
      found = tool;
    }
  }
  const version = await runBin(found, ['--version'], env, { timeoutMs: deps.readMs }, deps.platform);
  if (version.ended !== 'exit' || version.code !== 0 || !versionOk(version.stdout)) return null;
  return found;
}

export interface RefLine {
  name: string;
  /** Committer time, unix seconds; 0 when git gave none that reads. */
  at: number;
  /** The worktree that has it checked out, or ''. */
  worktree: string;
}

function unixTime(text: string): number {
  const value = leadingNumber(text);
  return value === null || String(value).length !== text.length ? 0 : value;
}

/**
 * `for-each-ref --format=%(refname)%00%(committerdate:unix)%00%(worktreepath)`: one branch per line, three fields.
 * A line of another shape is dropped, and so is a name that starts with `-` (git would read it as an option; `git
 * branch` refuses such names, `update-ref` does not).
 */
export function parseRefs(text: string): RefLine[] {
  const out: RefLine[] = [];
  for (const line of text.split('\n')) {
    const fields = line.split('\0');
    if (fields.length !== 3) continue;
    const [ref, time, worktree] = fields as [string, string, string];
    if (!ref.startsWith(HEADS)) continue;
    const name = ref.slice(HEADS.length);
    if (name === '' || name.startsWith('-')) continue;
    out.push({ name, at: unixTime(time), worktree });
  }
  return out;
}

/** The current branch first, then the newest commit, ties by name. */
export function orderBranches<T extends { name: string; at: number }>(refs: readonly T[], current: string | null): T[] {
  return [...refs].sort((a, b) => {
    if (a.name === current) return -1;
    if (b.name === current) return 1;
    if (a.at !== b.at) return b.at - a.at;
    return a.name < b.name ? -1 : a.name > b.name ? 1 : 0;
  });
}

/**
 * `status --porcelain=v2 -z`: one entry per NUL field (`1` changed, `u` unmerged, `?` untracked), but a rename or
 * copy (`2`) has a second field, its old path, which may itself start with `1 `. Ignored (`!`) and header (`#`)
 * entries are not changes.
 */
export function countChanges(text: string): number {
  const fields = text.split('\0');
  let n = 0;
  for (let i = 0; i < fields.length; i++) {
    const field = fields[i]!;
    if (field.length < 2 || field.charAt(1) !== ' ') continue;
    const kind = field.charAt(0);
    if (kind === '1' || kind === 'u' || kind === '?') n += 1;
    else if (kind === '2') {
      n += 1;
      i += 1;
    }
  }
  return n;
}

function cutPrefix(line: string): string {
  for (const prefix of ['error: ', 'fatal: ']) if (line.startsWith(prefix)) return line.slice(prefix.length);
  return line;
}

/** Git's refusal as the menu shows it: no `error: ` or `fatal: `, no closing "Aborting", at most 12 lines and 2 KB. */
export function refusalText(stderr: string): string {
  const lines = stderr.split('\n').map(cutPrefix);
  const trim = (): void => {
    while (lines.length > 0 && lines.at(-1)!.trim() === '') lines.pop();
  };
  trim();
  if (lines.at(-1) === 'Aborting') lines.pop();
  trim();
  let text = lines.slice(0, REFUSAL_LINES).join('\n');
  if (text.length > REFUSAL_CHARS) text = text.slice(0, REFUSAL_CHARS);
  return visibleText(text);
}

export type GitRead =
  | { kind: 'not-git' | 'slow' }
  | {
      kind: 'repo';
      top: string;
      head: WorkHead;
      refs: Array<RefLine & { elsewhere: string | null }>;
      changes: number | null;
    };

/** `true\n<top>\n` from `rev-parse --is-inside-work-tree --show-toplevel`; null for anything else. */
function topOf(stdout: string): string | null {
  const head = 'true\n';
  if (!stdout.startsWith(head)) return null;
  const top = withoutNewline(stdout.slice(head.length));
  return top === '' ? null : top;
}

/**
 * The folder's work tree, HEAD, local branches and uncommitted change count, every command under one deadline
 * (`slow` past it, the process group killed). Not a work tree (`.git`, bare, plain, dubious ownership): `not-git`.
 * The count is null when its command fails or runs out of time.
 */
export async function readRepo(
  bin: string,
  folder: string,
  env: Readonly<Record<string, string>>,
  deps: GitDeps,
): Promise<GitRead> {
  const deadline = Date.now() + deps.readMs;
  const reading = readEnv(env);
  const run = (dir: string, args: readonly string[]): Promise<GitRun> => {
    const left = deadline - Date.now();
    if (left <= 0) return Promise.resolve({ ended: 'timeout', code: null, stdout: '', stderr: '' });
    return runGit(bin, dir, args, reading, { timeoutMs: left }, deps.platform);
  };
  const where = await run(folder, ['rev-parse', '--is-inside-work-tree', '--show-toplevel']);
  if (where.ended === 'timeout') return { kind: 'slow' };
  const top = where.ended === 'exit' && where.code === 0 ? topOf(where.stdout) : null;
  if (top === null) return { kind: 'not-git' };

  const symbolic = await run(top, ['symbolic-ref', '--quiet', 'HEAD']);
  if (symbolic.ended === 'timeout') return { kind: 'slow' };
  if (symbolic.ended !== 'exit' || (symbolic.code !== 0 && symbolic.code !== 1)) return { kind: 'not-git' };
  const ref = symbolic.code === 0 ? withoutNewline(symbolic.stdout) : '';
  const headName = ref.startsWith(HEADS) ? ref.slice(HEADS.length) : null;
  let detached = '';
  if (headName === null) {
    const hash = await run(top, ['rev-parse', '--short', 'HEAD']);
    if (hash.ended === 'timeout') return { kind: 'slow' };
    if (hash.ended !== 'exit' || hash.code !== 0) return { kind: 'not-git' };
    detached = withoutNewline(hash.stdout);
    if (detached === '') return { kind: 'not-git' };
  }

  const listed = await run(top, [
    'for-each-ref',
    '--format=%(refname)%00%(committerdate:unix)%00%(worktreepath)',
    'refs/heads',
  ]);
  if (listed.ended === 'timeout') return { kind: 'slow' };
  if (listed.ended !== 'exit' || listed.code !== 0) return { kind: 'not-git' };
  const realTop = await realpath(top).catch(() => top);
  const refs = await Promise.all(
    parseRefs(listed.stdout).map(async (line) => {
      if (line.worktree === '') return { ...line, elsewhere: null };
      const there = await realpath(line.worktree).catch(() => line.worktree);
      return { ...line, elsewhere: there === realTop ? null : line.worktree };
    }),
  );

  const status = await run(top, ['status', '--porcelain=v2', '-z', '--untracked-files=normal']);
  const changes = status.ended === 'exit' && status.code === 0 ? countChanges(status.stdout) : null;

  const head: WorkHead =
    headName === null
      ? { detached }
      : refs.some((r) => r.name === headName)
        ? { branch: headName }
        : { unborn: headName };
  return { kind: 'repo', top, head, refs, changes };
}

export type SwitchRun = { outcome: 'switched' } | { outcome: 'refused' | 'timeout'; text: string };

/** `git switch --no-guess --no-progress <name>` in `top`, or git's refusal. */
export async function switchBranch(
  bin: string,
  top: string,
  name: string,
  env: Readonly<Record<string, string>>,
  deps: GitDeps,
): Promise<SwitchRun> {
  const run = await runGit(
    bin,
    top,
    ['switch', '--no-guess', '--no-progress', name],
    env,
    { timeoutMs: deps.switchMs, termGraceMs: deps.killGraceMs },
    deps.platform,
  );
  if (run.ended === 'timeout') return { outcome: 'timeout', text: refusalText(run.stderr) };
  if (run.ended === 'exit' && run.code === 0) return { outcome: 'switched' };
  return { outcome: 'refused', text: refusalText(run.stderr) };
}

/** How many of `folders` are `top`, hold it or lie in it, names folded as the disk compares them. */
export function overlapCount(top: string, folders: readonly string[]): number {
  const key = folderKey(top);
  return folders.filter((folder) => {
    const other = folderKey(folder);
    return below(other, key) || below(key, other);
  }).length;
}

/**
 * The nearest folder at or above `folder` with a `.git` entry, as git finds a work tree's top, or `folder` with none.
 * Found with no git run, so the lock of a repository is known before its first command.
 */
export async function repoKey(folder: string): Promise<string> {
  let current = folder;
  for (;;) {
    try {
      await lstat(join(current, '.git'));
      return current;
    } catch {
      // None here.
    }
    const parent = dirname(current);
    if (parent === current) return folder;
    current = parent;
  }
}
