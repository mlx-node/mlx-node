/**
 * A disk read that fails with anything but "not there" never allows: each call of `node:fs/promises` a decision makes
 * fails in turn with EIO, and the decision must ask every time. Only ENOENT and ENOTDIR read as absent. Only text goes
 * to the scan; no command of the table is run.
 */

import { execFileSync } from 'node:child_process';
import { linkSync, mkdirSync, mkdtempSync, realpathSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterAll, beforeAll, describe, expect, it, vi } from 'vite-plus/test';

import type { PermissionRequest } from '../src/chat/acp/messages.js';
import { autoFacts, decideAuto } from '../src/chat/agent/auto.js';
import { agentPath, type WorkPlace } from '../src/chat/agent/command-flags.js';
import { workInteraction } from '../src/chat/agent/permissions.js';
import { workGuardHookName } from '../src/chat/agent/profiles.js';
import { editLanding } from '../src/chat/work-folder.js';

const disk = vi.hoisted(() => ({ on: false, calls: 0, failAt: -1, failed: '' }));

vi.mock('node:fs/promises', async (importOriginal) => {
  const real = await importOriginal<typeof import('node:fs/promises')>();
  const wrap =
    <A extends unknown[], R>(name: string, fn: (...args: A) => Promise<R>) =>
    async (...args: A): Promise<R> => {
      if (disk.on) {
        const k = disk.calls++;
        if (k === disk.failAt) {
          disk.failed = `${name} ${String(args[0])}`;
          throw Object.assign(new Error(`EIO: i/o error, ${name}`), { code: 'EIO', errno: -5 });
        }
      }
      return fn(...args);
    };
  const mocked = {
    ...real,
    lstat: wrap('lstat', real.lstat as (...a: unknown[]) => Promise<unknown>),
    stat: wrap('stat', real.stat as (...a: unknown[]) => Promise<unknown>),
    realpath: wrap('realpath', real.realpath as (...a: unknown[]) => Promise<unknown>),
    readlink: wrap('readlink', real.readlink as (...a: unknown[]) => Promise<unknown>),
    readdir: wrap('readdir', real.readdir as (...a: unknown[]) => Promise<unknown>),
    readFile: wrap('readFile', real.readFile as (...a: unknown[]) => Promise<unknown>),
  };
  return { ...mocked, default: mocked };
});

const GUARD = workGuardHookName('mlx-work-guard-0123456789abcdef');
const OPTIONS = [
  { optionId: 'allow-once', name: 'once', kind: 'allow_once' },
  { optionId: 'reject-once', name: 'no', kind: 'reject_once' },
];

let root: string;
let place: WorkPlace;

beforeAll(() => {
  root = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-auto-eio-')));
  const home = join(root, 'home');
  const cwd = join(home, 'code', 'app');
  const out = join(root, 'out');
  for (const d of ['d', 'e', 'src', 'origin', '.git/hooks']) mkdirSync(join(cwd, d), { recursive: true });
  mkdirSync(out, { recursive: true });
  for (const f of ['h', 'g']) writeFileSync(join(out, f), 'ORIG\n');
  for (const f of ['a', 'a.txt', 'b.txt', 'main2', 'src/x.ts', 's.js']) writeFileSync(join(cwd, f), 'x\n');
  writeFileSync(
    join(cwd, '.git', 'config'),
    '[core]\n\tbare = false\n[remote "origin"]\n\turl = git@github.com:o/app.git\n[remote "up"]\n\turl = git@github.com:o/up.git\n',
  );
  linkSync(join(out, 'h'), join(cwd, 'hard'));
  linkSync(join(out, 'g'), join(cwd, 'd', 'a'));
  symlinkSync(out, join(cwd, 'k'));
  execFileSync('/usr/bin/mkfifo', [join(cwd, 'p')]);
  place = { cwd, home };
});

afterAll(() => {
  rmSync(root, { recursive: true, force: true });
});

function request(kind: 'bash' | 'edit', input: string): PermissionRequest {
  const bash = kind === 'bash';
  const tool = bash ? 'run_terminal_command' : 'write';
  return {
    sessionId: 's1',
    toolCallId: 'call-1',
    kind: bash ? 'execute' : 'edit',
    title: tool,
    tool: { name: tool, namespace: 'grok_build' },
    rawInput: bash
      ? { variant: 'Bash', command: input, description: 'run', is_background: false }
      : { file_path: input, content: 'x' },
    options: OPTIONS,
    hookAsk: { hookName: GUARD, reason: null },
  };
}

/** The decision as the host makes it: an edit's landing first (a failed one is unknown), then the facts. */
async function allows(kind: 'bash' | 'edit', input: string): Promise<boolean> {
  const req = request(kind, input);
  const landing = kind === 'edit' ? await editLanding(agentPath(input, place)).catch(() => null) : null;
  const facts = await autoFacts(req, place, landing, { cdpath: false });
  const decision = decideAuto({
    auto: true,
    watched: true,
    request: req,
    interaction: workInteraction('i1', req, [], place, landing),
    guardHook: GUARD,
    elsewhere: false,
    facts,
    allowed: 0,
  });
  return decision.allow;
}

/** Each row with what it gives with no error: the review's EIO rows ask, the plain ones run. */
const ROWS: ReadonlyArray<[kind: 'bash' | 'edit', input: string, plain: boolean]> = [
  ['bash', 'echo x > hard', false],
  ['bash', 'echo x > p', false],
  ['bash', 'git checkout main2', false],
  ['bash', 'git fetch origin', false],
  ['bash', 'cp a d', false],
  ['bash', 'diff d e', false],
  ['bash', 'echo x > k', false],
  ['edit', 'hard', false],
  ['edit', 'k/new.txt', false],
  ['bash', 'echo x > n.txt', true],
  ['bash', 'cat src/*.ts', true],
  ['bash', 'git status', true],
  ['bash', 'git fetch up', true],
  ['bash', 'git checkout main3', true],
  ['bash', 'cp a.txt b.txt', true],
  ['bash', 'cp a.txt e', true],
  ['bash', 'diff a.txt b.txt', true],
  ['bash', 'node s.js', true],
  ['bash', 'eslint src', true],
  ['bash', 'cd src && cat x.ts', true],
  ['bash', 'mv a.txt c.txt', true],
  ['edit', 'new.txt', true],
  ['edit', 'src/x.ts', true],
];

describe('a disk error asks', () => {
  it.each(ROWS)('%s %s: an EIO in any disk call of the decision never allows', async (kind, input, plain) => {
    disk.on = true;
    disk.calls = 0;
    disk.failAt = -1;
    try {
      expect(await allows(kind, input)).toBe(plain);
      const total = disk.calls;
      expect(total).toBeGreaterThan(0);
      const allowed: string[] = [];
      for (let k = 0; k < total; k++) {
        disk.calls = 0;
        disk.failAt = k;
        disk.failed = '';
        if (await allows(kind, input)) allowed.push(`${String(k)}: ${disk.failed}`);
      }
      expect(allowed).toEqual([]);
    } finally {
      disk.on = false;
      disk.failAt = -1;
    }
  });
});
