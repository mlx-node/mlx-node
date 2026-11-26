/**
 * The Work branch tray's host (`src/chat/work-git.ts`, the service's `workGit` / `switchWorkBranch`, the two routes).
 * Real git runs only in throwaway repositories under a temp folder, with a temp `HOME`; the fake git (POSIX) logs its
 * argv and two env values per call.
 */

import { execFileSync } from 'node:child_process';
import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';

import { afterEach, beforeEach, describe, expect, expectTypeOf, it } from 'vite-plus/test';

import type { ApiResponse } from '../src/api/errors.js';
import { MemorySecretStore } from '../src/chat/providers/secrets.js';
import type * as Server from '../src/chat/types.js';
import {
  countChanges,
  GIT_DEPS,
  gitEnv,
  orderBranches,
  parseRefs,
  readRepo,
  resolveGit,
  switchBranch,
  type GitDeps,
  type GitRead,
} from '../src/chat/work-git.js';
import { createDashboardRuntime, type ChatRuntimeOptions, type DashboardRuntime } from '../src/runtime.js';
import type * as Page from '../ui/src/lib/chat-types.js';
import { endpointModel, FAKE_AGENT, INFERENCE_TOKEN, readScenario, until } from './helpers/chat-fake.js';
import { startFakeEngine, type FakeEngine } from './helpers/fake-engine.js';

const POSIX = process.platform !== 'win32';
const REAL_PATH = process.env.PATH ?? '/usr/bin:/bin';
const SHELL_PATH = '/usr/bin:/bin';

let base: string;
let home: string;
let runtime: DashboardRuntime | undefined;
let engine: FakeEngine | undefined;

beforeEach(() => {
  base = realpathSync(mkdtempSync(join(tmpdir(), 'dash-work-git-')));
  home = join(base, 'home');
  mkdirSync(join(home, 'code'), { recursive: true });
});

afterEach(async () => {
  await runtime?.close();
  runtime = undefined;
  await engine?.close();
  engine = undefined;
  rmSync(base, { recursive: true, force: true });
});

/** Real git for the test's own setup, in a throwaway repository only. */
function git(dir: string, args: string[], env: Record<string, string> = {}): string {
  return execFileSync('git', ['-c', 'user.name=t', '-c', 'user.email=t@example.com', ...args], {
    cwd: dir,
    env: { HOME: home, PATH: REAL_PATH, GIT_CONFIG_NOSYSTEM: '1', LC_ALL: 'C', ...env },
    encoding: 'utf8',
  });
}

/** A repository under the temp home with one commit of `f` on `main`. */
function repo(name = 'app'): string {
  const dir = join(home, 'code', name);
  mkdirSync(dir, { recursive: true });
  git(dir, ['init', '-q', '-b', 'main']);
  writeFileSync(join(dir, 'f'), 'one\n');
  git(dir, ['add', 'f']);
  git(dir, ['commit', '-q', '-m', 'one']);
  return dir;
}

function realEnv(): Record<string, string> {
  return gitEnv({ HOME: home, PATH: REAL_PATH }, home, null, process.platform);
}

async function realGit(): Promise<string> {
  const bin = await resolveGit(realEnv(), GIT_DEPS);
  if (bin === null) throw new Error('no git for the test');
  return bin;
}

function head(dir: string): string {
  return git(dir, ['symbolic-ref', '--quiet', '--short', 'HEAD']).trim();
}

interface FakeCall {
  locks: string;
  gitDir: string;
  argv: string[];
}

interface FakeGit {
  dir: string;
  bin: string;
  calls(): FakeCall[];
  /** Pids of the `sleep` children the fake started. */
  sleepers(): number[];
}

interface FakeOptions {
  version?: string;
  /** What `rev-parse --show-toplevel` prints. */
  top?: string;
  /** The exact `for-each-ref` output. */
  refs?: string;
  /** Subcommands that start a 30 s `sleep` child and wait for it. */
  sleep?: string[];
  dir?: string;
}

/** A POSIX shell script named `git` that logs each call and answers the commands of a read and a switch. */
function fakeGit(options: FakeOptions = {}): FakeGit {
  const dir = options.dir ?? mkdtempSync(join(base, 'fake-git-'));
  mkdirSync(dir, { recursive: true });
  const log = join(dir, 'calls.log');
  const pids = join(dir, 'pids.log');
  const refs = join(dir, 'refs.out');
  writeFileSync(refs, options.refs ?? 'refs/heads/main\x001700000000\x00\n');
  const script = [
    '#!/bin/sh',
    `{ printf '%s\\0' "\${GIT_OPTIONAL_LOCKS-}" "\${GIT_DIR-unset}"; for a in "$@"; do printf '%s\\0' "$a"; done; printf '\\n'; } >> '${log}'`,
    'while [ $# -gt 0 ]; do case "$1" in -C|-c) shift 2 ;; *) break ;; esac; done',
    `case " ${(options.sleep ?? []).join(' ')} " in *" $1 "*) sleep 30 & echo $! >> '${pids}'; wait; exit 1 ;; esac`,
    'case "$1" in',
    `  --version) printf 'git version ${options.version ?? '2.40.0'}\\n' ;;`,
    `  rev-parse) if [ "$2" = --short ]; then printf 'abc1234\\n'; else printf 'true\\n%s\\n' '${options.top ?? dir}'; fi ;;`,
    "  symbolic-ref) printf 'refs/heads/main\\n' ;;",
    `  for-each-ref) cat '${refs}' ;;`,
    '  status) : ;;',
    '  switch) exit 0 ;;',
    '  *) exit 2 ;;',
    'esac',
    '',
  ].join('\n');
  const bin = join(dir, 'git');
  writeFileSync(bin, script);
  chmodSync(bin, 0o755);
  return {
    dir,
    bin,
    calls: () =>
      existsSync(log)
        ? readFileSync(log, 'utf8')
            .split('\n')
            .filter((line) => line !== '')
            .map((line) => {
              const fields = line.split('\0').slice(0, -1);
              return { locks: fields[0]!, gitDir: fields[1]!, argv: fields.slice(2) };
            })
        : [],
    sleepers: () =>
      existsSync(pids)
        ? readFileSync(pids, 'utf8')
            .split('\n')
            .filter((line) => line !== '')
            .map(Number)
        : [],
  };
}

function subcommand(call: FakeCall): string | undefined {
  let i = 0;
  while (call.argv[i] === '-C' || call.argv[i] === '-c') i += 2;
  return call.argv[i];
}

function alive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function start(chat: Partial<ChatRuntimeOptions> = {}): void {
  runtime = createDashboardRuntime({
    dbPath: ':memory:',
    tracesDir: join(base, 'traces'),
    modelsDir: join(base, 'models'),
    runtimeDir: join(base, 'runtime'),
    supervisorPath: join(base, 'supervisor.json'),
    agentHome: join(base, 'agent-home'),
    chat: {
      root: join(base, 'chat'),
      homeDir: home,
      tempRoots: [],
      agentCommand: { command: process.execPath, args: [FAKE_AGENT, join(base, 'none.json')] },
      secrets: new MemorySecretStore(),
      readEndpoint: async () => null,
      baseEnv: { HOME: home, PATH: REAL_PATH },
      flushMs: 5,
      stateCoalesceMs: 5,
      ...chat,
    },
  });
}

const call = (method: string, path: string, body?: unknown): Promise<ApiResponse> =>
  runtime!.call({ method, path, ...(body === undefined ? {} : { body }) });

const gitPath = (cwd: string): string => `/api/chat/work/git?cwd=${encodeURIComponent(cwd)}`;

async function read(cwd: string): Promise<Server.WorkGit> {
  const reply = await call('GET', gitPath(cwd));
  if (!reply.ok) throw new Error(`${reply.code} ${reply.message}`);
  return reply.body as Server.WorkGit;
}

async function switchTo(cwd: string, branch: string): Promise<ApiResponse> {
  return call('POST', '/api/chat/work/git/switch', { cwd, branch });
}

function failure(reply: ApiResponse): { code: string; reason?: string } {
  if (reply.ok) throw new Error(`expected a failure, got ${JSON.stringify(reply.body)}`);
  return { code: reply.code, ...(reply.reason === undefined ? {} : { reason: reply.reason }) };
}

function repoOf(git: Server.WorkGit): Extract<Server.WorkGit, { kind: 'repo' }> {
  if (git.kind !== 'repo') throw new Error(`not a repo: ${git.kind}`);
  return git;
}

function readOf(read: GitRead): Extract<GitRead, { kind: 'repo' }> {
  if (read.kind !== 'repo') throw new Error(`not a repo: ${read.kind}`);
  return read;
}

describe('the wire types', () => {
  it('the page copy equals the server types', () => {
    expectTypeOf<Page.WorkGit>().toEqualTypeOf<Server.WorkGit>();
    expectTypeOf<Page.WorkBranch>().toEqualTypeOf<Server.WorkBranch>();
    expectTypeOf<Page.WorkSwitchReply>().toEqualTypeOf<Server.WorkSwitchReply>();
    expectTypeOf<Page.WorkGitErrorCode>().toEqualTypeOf<Server.WorkGitErrorCode>();
  });
});

describe('parsing and order', () => {
  // Mutations: split the fields on a space; swap `at` and `worktree`; keep a 2-field or a non-`refs/heads` line; keep a
  // `-f` name.
  it('reads NUL fields per line; drops lines of another shape', () => {
    const text = [
      'refs/heads/feat/deep/x\x001700000300\x00',
      'refs/heads/ünïcødé\x001700000200\x00/Users/me/my wt',
      'refs/heads/two-fields\x001700000100',
      'refs/tags/v1\x001700000400\x00',
      'refs/remotes/origin/main\x001700000400\x00',
      'refs/heads/bad-time\x00soon\x00',
      'refs/heads/-f\x001700000000\x00',
      '',
    ].join('\n');
    expect(parseRefs(text)).toEqual([
      { name: 'feat/deep/x', at: 1_700_000_300, worktree: '' },
      { name: 'ünïcødé', at: 1_700_000_200, worktree: '/Users/me/my wt' },
      { name: 'bad-time', at: 0, worktree: '' },
    ]);
  });

  // Mutations: the current branch not first; oldest first; ties not by name.
  it('the current branch first, then newest, ties by name', () => {
    const refs = [
      { name: 'zeta', at: 50 },
      { name: 'main', at: 1 },
      { name: 'alpha', at: 50 },
      { name: 'new', at: 90 },
    ];
    expect(orderBranches(refs, 'main').map((r) => r.name)).toEqual(['main', 'new', 'alpha', 'zeta']);
    expect(orderBranches(refs, null).map((r) => r.name)).toEqual(['new', 'alpha', 'zeta', 'main']);
  });

  // Mutations: count a rename's old path as an entry; count an ignored entry; skip untracked or unmerged.
  it('counts porcelain v2 entries: a rename takes two fields, ignored ones do not count', () => {
    const ordinary = '1 .M N... 100644 100644 100644 aaa aaa f';
    const rename = '2 R. N... 100644 100644 100644 bbb bbb R100 new name';
    const oldPath = '1 looks like an entry';
    const text = [ordinary, rename, oldPath, '? untracked', 'u UU N... 1 1 1 1 a b c conflicted', '! ignored', ''].join(
      '\0',
    );
    expect(countChanges(text)).toBe(4);
    expect(countChanges('')).toBe(0);
  });
});

describe.runIf(POSIX)('finding git', () => {
  // Mutation: a missing git answered as `not-git` (the route's `no-git` lost).
  it('no git on PATH: none, and the route answers no-git', async () => {
    const empty = join(base, 'empty');
    mkdirSync(empty);
    expect(await resolveGit(gitEnv({ HOME: home, PATH: empty }, home, null, process.platform), GIT_DEPS)).toBeNull();
    const dir = repo();
    start({ baseEnv: { HOME: home, PATH: empty } });
    expect(await read(dir)).toEqual({ kind: 'no-git' });
  });

  // Mutation: run the shim when it is the first git (it can raise the install prompt).
  it('the shim runs only through the developer folder xcode-select names', async () => {
    const shim = fakeGit();
    const env = gitEnv({ HOME: home, PATH: `${shim.dir}:${SHELL_PATH}` }, home, null, 'darwin');
    const deps: GitDeps = { ...GIT_DEPS, platform: 'darwin', shim: shim.bin, developerDir: async () => null };
    expect(await resolveGit(env, deps)).toBeNull();
    expect(shim.calls()).toEqual([]);
    const developer = join(base, 'developer');
    const tool = fakeGit({ dir: join(developer, 'usr', 'bin') });
    expect(await resolveGit(env, { ...deps, developerDir: async () => developer })).toBe(tool.bin);
    expect(shim.calls()).toEqual([]);
    expect(tool.calls().map((c) => c.argv)).toEqual([['--version']]);
  });

  // Mutation: compare versions as text ("2.100" < "2.23").
  it('below 2.23 is no git; versions compare as numbers', async () => {
    const check = async (version: string): Promise<boolean> => {
      const fake = fakeGit({ version, dir: join(base, `v-${version}`) });
      const env = gitEnv({ HOME: home, PATH: `${fake.dir}:${SHELL_PATH}` }, home, null, 'linux');
      return (await resolveGit(env, { ...GIT_DEPS, platform: 'linux' })) !== null;
    };
    expect(await check('2.20.0')).toBe(false);
    expect(await check('2.22.9')).toBe(false);
    expect(await check('2.23.0')).toBe(true);
    expect(await check('2.100.1')).toBe(true);
    expect(await check('2.39.5 (Apple Git-154)')).toBe(true);
  });
});

describe('reading a folder (real git)', () => {
  // Mutations: a detached HEAD read as a branch; an unborn branch read as detached or lost; `not-git` missed for a
  // `.git` folder, a bare repository or a plain folder; a subfolder not resolved to its top.
  it('branch, detached, unborn, subfolder; .git, bare and plain folders are not git', async () => {
    const bin = await realGit();
    const env = realEnv();
    const dir = repo();
    git(dir, ['branch', 'feat']);
    mkdirSync(join(dir, 'sub', 'deeper'), { recursive: true });
    const onMain = readOf(await readRepo(bin, join(dir, 'sub', 'deeper'), env, GIT_DEPS));
    expect([onMain.top, onMain.head]).toEqual([dir, { branch: 'main' }]);

    git(dir, ['switch', '-q', '--detach']);
    const hash = git(dir, ['rev-parse', '--short', 'HEAD']).trim();
    expect(readOf(await readRepo(bin, dir, env, GIT_DEPS)).head).toEqual({ detached: hash });

    const fresh = join(home, 'code', 'fresh');
    mkdirSync(fresh);
    git(fresh, ['init', '-q', '-b', 'trunk']);
    const unborn = readOf(await readRepo(bin, fresh, env, GIT_DEPS));
    expect([unborn.head, unborn.refs]).toEqual([{ unborn: 'trunk' }, []]);

    const bare = join(home, 'code', 'bare.git');
    mkdirSync(bare);
    git(bare, ['init', '-q', '--bare']);
    const plain = join(home, 'code', 'plain');
    mkdirSync(plain);
    for (const folder of [join(dir, '.git'), bare, plain]) {
      expect([folder, (await readRepo(bin, folder, env, GIT_DEPS)).kind]).toEqual([folder, 'not-git']);
    }
  });

  // Mutation: compare `worktreepath` to nothing → the branch of this worktree reads as checked out elsewhere.
  it("a linked worktree: its own branch is not elsewhere; the main worktree's branch is", async () => {
    const bin = await realGit();
    const env = realEnv();
    const dir = repo();
    const tree = join(home, 'code', 'app-feat');
    git(dir, ['worktree', 'add', '-q', '-b', 'feat', tree]);
    const linked = readOf(await readRepo(bin, tree, env, GIT_DEPS));
    expect(linked.head).toEqual({ branch: 'feat' });
    expect(Object.fromEntries(linked.refs.map((r) => [r.name, r.elsewhere]))).toEqual({ feat: null, main: dir });
    const main = readOf(await readRepo(bin, dir, env, GIT_DEPS));
    expect(Object.fromEntries(main.refs.map((r) => [r.name, r.elsewhere]))).toEqual({ feat: tree, main: null });
  });

  // Mutations: drop the cap; the cap drops the current branch; `total` counts the shown rows; the change count missing
  // from the reply or counting ignored files.
  it('the route: at most 1,000 branches with the current one first, the total, the busy count and the changes', async () => {
    const dir = repo();
    git(dir, ['commit', '-q', '--allow-empty', '-m', 'newer'], { GIT_COMMITTER_DATE: '2030-01-01T00:00:00Z' });
    const newer = git(dir, ['rev-parse', 'HEAD']).trim();
    const lines: string[] = [];
    for (let k = 0; k < 1_499; k++) lines.push(`create refs/heads/b${String(k).padStart(4, '0')} ${newer}`);
    execFileSync('git', ['update-ref', '--stdin'], {
      cwd: dir,
      input: `${lines.join('\n')}\n`,
      env: { HOME: home, PATH: REAL_PATH, GIT_CONFIG_NOSYSTEM: '1' },
    });
    git(dir, ['switch', '-q', '-c', 'zz-current', 'HEAD~1']);
    writeFileSync(join(dir, 'f'), 'changed\n');
    writeFileSync(join(dir, 'untracked'), 'u\n');
    writeFileSync(join(dir, '.gitignore'), 'ignored\n');
    writeFileSync(join(dir, 'ignored'), 'i\n');
    start();
    const shown = repoOf(await read(dir));
    expect([shown.top, shown.head, shown.total, shown.busy, shown.changes]).toEqual([
      dir,
      { branch: 'zz-current' },
      1_501,
      0,
      3,
    ]);
    expect(shown.branches.length).toBe(1_000);
    expect(shown.branches[0]).toEqual({
      name: 'zz-current',
      at: shown.branches[0]!.at,
      current: true,
      elsewhere: null,
    });
    expect(shown.branches.slice(1, 4).map((b) => b.name)).toEqual(['b0000', 'b0001', 'b0002']);
    expect(shown.branches.filter((b) => b.current).length).toBe(1);
  });
});

describe('switching (real git)', () => {
  // Mutations: a non-zero exit read as success; the lock removed first; git's text dropped or kept with its `fatal:`
  // and "Aborting" lines.
  it("a branch of another worktree and a held index.lock are git's refusals; nothing changes", async () => {
    const dir = repo();
    git(dir, ['branch', 'other']);
    git(dir, ['worktree', 'add', '-q', '-b', 'held', join(home, 'code', 'app-held')]);
    start();

    const elsewhere = await switchTo(dir, 'held');
    const refused = (elsewhere.ok ? elsewhere.body : null) as Server.WorkSwitchReply | null;
    expect(refused?.outcome).toBe('refused');
    const text = refused !== null && 'text' in refused ? refused.text : '';
    expect(text).toContain("'held'");
    expect(text.startsWith('fatal: ')).toBe(false);
    expect(refused !== null && repoOf(refused.git).head).toEqual({ branch: 'main' });
    expect(head(dir)).toBe('main');

    const lock = join(dir, '.git', 'index.lock');
    writeFileSync(lock, '');
    const locked = await switchTo(dir, 'other');
    expect(locked.ok && locked.body).toMatchObject({
      outcome: 'refused',
      text: expect.stringContaining('index.lock'),
    });
    expect([existsSync(lock), head(dir)]).toEqual([true, 'main']);
  });

  // Mutations: the dirty guard removed (git carries the edit or refuses for it); untracked files not counted.
  it('a modified tracked file or an untracked file alone is 409 dirty; HEAD and the files stay', async () => {
    const dir = repo();
    git(dir, ['branch', 'next']);
    writeFileSync(join(dir, 'f'), 'edited\n');
    start();
    expect(failure(await switchTo(dir, 'next'))).toEqual({ code: 'E_CONFLICT', reason: 'dirty' });
    expect([head(dir), readFileSync(join(dir, 'f'), 'utf8')]).toEqual(['main', 'edited\n']);
    expect(repoOf(await read(dir)).changes).toBe(1);

    git(dir, ['checkout', '--', 'f']);
    writeFileSync(join(dir, 'new-file'), 'n\n');
    expect(failure(await switchTo(dir, 'next'))).toEqual({ code: 'E_CONFLICT', reason: 'dirty' });
    expect([head(dir), existsSync(join(dir, 'new-file'))]).toEqual(['main', true]);
  });

  // Mutation: a clean tree refused (the guard reads null or a header line as a change).
  it('a clean tree switches; the reply holds the read after it', async () => {
    const dir = repo();
    git(dir, ['branch', 'next']);
    start();
    const reply = await switchTo(dir, 'next');
    expect(reply.ok && reply.body).toMatchObject({ outcome: 'switched' });
    const after = reply.ok ? repoOf((reply.body as Server.WorkSwitchReply).git) : null;
    expect([after?.head, after?.changes, head(dir)]).toEqual([{ branch: 'next' }, 0, 'next']);
  });
});

describe.runIf(POSIX)('switching (fake git)', () => {
  const ODD = 'wip/$HOME;x';

  function startFake(): { dir: string; fake: FakeGit } {
    const dir = join(home, 'code', 'app');
    mkdirSync(join(dir, '.git'), { recursive: true });
    const fake = fakeGit({
      top: dir,
      refs: `refs/heads/main\x001700000000\x00\nrefs/heads/${ODD}\x001700000100\x00\n`,
    });
    start({ baseEnv: { HOME: home, PATH: `${fake.dir}:${SHELL_PATH}`, GIT_DIR: '/elsewhere' } });
    return { dir, fake };
  }

  // Mutations: `--no-guess` dropped; the name not last; fsmonitor left on; the call made through a shell (`$HOME` and
  // `;` would be read); an inherited `GIT_*` passed on; a read that takes optional locks; a switch run for the current
  // branch.
  it('argv: settings first, `switch --no-guess --no-progress <name>` last, no shell; a clean env', async () => {
    const { dir, fake } = startFake();
    const current = await switchTo(dir, 'main');
    expect(current.ok && current.body).toMatchObject({ outcome: 'switched' });
    expect(fake.calls().some((c) => subcommand(c) === 'switch')).toBe(false);

    const reply = await switchTo(dir, ODD);
    expect(reply.ok && reply.body).toMatchObject({ outcome: 'switched' });
    const switches = fake.calls().filter((c) => subcommand(c) === 'switch');
    expect(switches.map((c) => c.argv)).toEqual([
      [
        '-C',
        dir,
        '-c',
        'core.fsmonitor=false',
        '-c',
        'color.ui=false',
        '-c',
        'core.quotePath=false',
        'switch',
        '--no-guess',
        '--no-progress',
        ODD,
      ],
    ]);
    expect(switches[0]!.locks).toBe('');
    const reads = fake.calls().filter((c) => subcommand(c) !== 'switch' && subcommand(c) !== '--version');
    expect(reads.length).toBeGreaterThan(0);
    expect(new Set(reads.map((c) => c.locks))).toEqual(new Set(['0']));
    expect(new Set(fake.calls().map((c) => c.gitDir))).toEqual(new Set(['unset']));
  });

  // Mutation: an unknown change count (status past the deadline) treated as clean → the switch runs.
  it('a change count git could not give in time is 409 status-unknown and runs no switch', async () => {
    const dir = join(home, 'code', 'app');
    mkdirSync(join(dir, '.git'), { recursive: true });
    const fake = fakeGit({
      top: dir,
      sleep: ['status'],
      refs: 'refs/heads/main\x001700000000\x00\nrefs/heads/next\x001700000100\x00\n',
    });
    start({ baseEnv: { HOME: home, PATH: `${fake.dir}:${SHELL_PATH}` }, gitDeps: { readMs: 1_000 } });
    expect(repoOf(await read(dir)).changes).toBeNull();
    expect(failure(await switchTo(dir, 'next'))).toEqual({ code: 'E_CONFLICT', reason: 'status-unknown' });
    expect(fake.calls().some((c) => subcommand(c) === 'switch')).toBe(false);
    await until(() => fake.sleepers().every((pid) => !alive(pid)), 3_000);
  });

  // Mutation: the name not checked against the list → `git switch -f` or a remote's name reaches git.
  it('a name that is not a listed local branch is 404 branch-unknown and runs no switch', async () => {
    const { dir, fake } = startFake();
    for (const branch of ['nosuch', 'origin/x', '-f', 'refs/heads/main', 'x'.repeat(1_024)]) {
      expect([branch, failure(await switchTo(dir, branch))]).toEqual([
        branch,
        { code: 'E_NOT_FOUND', reason: 'branch-unknown' },
      ]);
    }
    expect(fake.calls().some((c) => subcommand(c) === 'switch')).toBe(false);
  });

  // Mutations: the folder rule skipped; a loose body (extra field, 1,025 characters) taken.
  it('the folder rule first; the body is exactly {cwd, branch}', async () => {
    const { dir, fake } = startFake();
    const missing = join(home, 'code', 'gone');
    for (const cwd of [missing, home]) {
      expect(failure(await call('GET', gitPath(cwd)))).toEqual({ code: 'E_BAD_REQUEST', reason: 'cwd-invalid' });
      expect(failure(await switchTo(cwd, 'main'))).toEqual({ code: 'E_BAD_REQUEST', reason: 'cwd-invalid' });
    }
    expect(failure(await call('GET', '/api/chat/work/git'))).toEqual({ code: 'E_BAD_REQUEST', reason: 'cwd-invalid' });
    for (const body of [
      { cwd: dir, branch: ODD, force: true },
      { cwd: dir, branch: 'x'.repeat(1_025) },
      { cwd: dir },
    ]) {
      expect(failure(await call('POST', '/api/chat/work/git/switch', body)).code).toBe('E_BAD_REQUEST');
    }
    expect(fake.calls()).toEqual([]);
  });
});

describe.runIf(POSIX)('deadlines (fake git)', () => {
  // Mutations: no kill at the deadline (the read waits 30 s); only the child killed (its `sleep` lives on).
  it('a read past its deadline is slow and its process group is gone', async () => {
    const fake = fakeGit({ sleep: ['rev-parse'] });
    const env = gitEnv({ HOME: home, PATH: `${fake.dir}:${SHELL_PATH}` }, home, null, process.platform);
    const began = Date.now();
    expect(await readRepo(fake.bin, fake.dir, env, { ...GIT_DEPS, readMs: 200 })).toEqual({ kind: 'slow' });
    expect(Date.now() - began).toBeLessThan(5_000);
    expect(fake.sleepers().length).toBe(1);
    await until(() => !alive(fake.sleepers()[0]!), 3_000);
  });

  // Mutation: the change count's timeout fails the whole read.
  it('a status past the deadline leaves the count unknown, not the read', async () => {
    const fake = fakeGit({ sleep: ['status'] });
    const env = gitEnv({ HOME: home, PATH: `${fake.dir}:${SHELL_PATH}` }, home, null, process.platform);
    const got = readOf(await readRepo(fake.bin, fake.dir, env, { ...GIT_DEPS, readMs: 1_500 }));
    expect([got.head, got.changes]).toEqual([{ branch: 'main' }, null]);
    await until(() => fake.sleepers().every((pid) => !alive(pid)), 3_000);
  });

  // Mutations: no kill at the cap; only the child killed.
  it('a switch past its cap is a timeout and its process group is gone', async () => {
    const fake = fakeGit({ sleep: ['switch'] });
    const env = gitEnv({ HOME: home, PATH: `${fake.dir}:${SHELL_PATH}` }, home, null, process.platform);
    const deps = { ...GIT_DEPS, switchMs: 200, killGraceMs: 100 };
    const began = Date.now();
    expect(await switchBranch(fake.bin, fake.dir, 'main', env, deps)).toEqual({ outcome: 'timeout', text: '' });
    expect(Date.now() - began).toBeLessThan(5_000);
    expect(fake.sleepers().length).toBe(1);
    await until(() => !alive(fake.sleepers()[0]!), 3_000);
  });
});

describe.runIf(POSIX)('the busy guard (fake agent, fake git)', () => {
  async function startWork(): Promise<{ top: string; fake: FakeGit }> {
    engine = await startFakeEngine();
    const scenario = readScenario('work-plan');
    scenario.prompts = [
      { match: 'hold', steps: [{ sleepMs: 60_000 }, { end: { stopReason: 'end_turn' } }] },
      { match: '', steps: [{ text: 'ok' }, { end: { stopReason: 'end_turn' } }] },
    ];
    const scenarioPath = join(base, 'scenario.json');
    writeFileSync(scenarioPath, JSON.stringify(scenario));
    const top = join(home, 'code', 'app');
    mkdirSync(join(top, '.git'), { recursive: true });
    mkdirSync(join(top, 'sub'));
    mkdirSync(join(home, 'code', 'app-wt'));
    const fake = fakeGit({ top, refs: `refs/heads/main\x001700000000\x00\nrefs/heads/next\x001700000100\x00\n` });
    const url = engine.url;
    start({
      agentCommand: { command: process.execPath, args: [FAKE_AGENT, scenarioPath] },
      readEndpoint: async () => ({
        version: 1,
        pid: process.pid,
        url,
        token: INFERENCE_TOKEN,
        models: [endpointModel('qwen3.5-4b', '/models/qwen3.5-4b')],
      }),
      baseEnv: { HOME: home, PATH: `${fake.dir}:${SHELL_PATH}` },
      timings: { stopEofMs: 300, stopTermMs: 300 },
    });
    return { top, fake };
  }

  async function working(cwd: string, text: string): Promise<string> {
    const made = await call('POST', '/api/chat/conversations', { mode: 'work', text, cwd });
    if (!made.ok) throw new Error(made.message);
    return (made.body as { conversation: Server.ConversationInfo }).conversation.id;
  }

  async function busy(top: string): Promise<number> {
    return repoOf(await read(top)).busy;
  }

  /** Reads until the busy count is `want`, one read at a time. */
  async function busyIs(top: string, want: number): Promise<void> {
    const end = Date.now() + 10_000;
    while ((await busy(top)) !== want) {
      if (Date.now() > end) throw new Error(`busy never reached ${want}`);
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
  }

  // Mutations: only a folder inside the top counts (a session in the parent passes); an idle conversation counts; the
  // POST unguarded; a sibling worktree counted.
  it('a working session in the top, a subfolder or a parent: 409 folder-busy with no git; idle or a sibling: ok', async () => {
    const { top, fake } = await startWork();
    for (const cwd of [top, join(top, 'sub'), dirname(top)]) {
      const id = await working(cwd, 'hold');
      await busyIs(top, 1);
      const before = fake.calls().length;
      expect([cwd, failure(await switchTo(top, 'next'))]).toEqual([cwd, { code: 'E_CONFLICT', reason: 'folder-busy' }]);
      expect(fake.calls().length).toBe(before);
      await call('POST', `/api/chat/conversations/${id}/cancel`);
      await busyIs(top, 0);
    }
    const sibling = await working(join(home, 'code', 'app-wt'), 'hold');
    expect(await busy(top)).toBe(0);
    const reply = await switchTo(top, 'next');
    expect(reply.ok && reply.body).toMatchObject({ outcome: 'switched' });
    expect(fake.calls().filter((c) => subcommand(c) === 'switch').length).toBe(1);
    await call('POST', `/api/chat/conversations/${sibling}/cancel`);
  }, 60_000);
});
