/**
 * Auto with a temp root beside the Work folder (`fixtures/auto/temp-root.json`): a throwaway folder injected as the
 * root with two spellings (its real path and a link to it, standing in for `/private/tmp` and `/tmp`), never the
 * machine's `/tmp`. Each row is request text the gate reads; no command of the table is run.
 */

import { execFileSync } from 'node:child_process';
import type { Stats } from 'node:fs';
import {
  chmodSync,
  existsSync,
  linkSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { afterAll, beforeAll, describe, expect, it, vi } from 'vite-plus/test';

import type { PermissionRequest } from '../src/chat/acp/messages.js';
import { autoScan, scanDirs } from '../src/chat/agent/auto-scan.js';
import { autoFacts, decideAuto, privateName } from '../src/chat/agent/auto.js';
import { agentPath, commandFlags, type WorkPlace } from '../src/chat/agent/command-flags.js';
import { workInteraction } from '../src/chat/agent/permissions.js';
import { workGuardHookName } from '../src/chat/agent/profiles.js';
import { rootOf, roots, rootText, type AllowedRoot } from '../src/chat/agent/roots.js';
import type { AutoAsk } from '../src/chat/types.js';
import { editLanding, realFolder, resolveTempRoot } from '../src/chat/work-folder.js';

/** Paths whose `lstat` reports another device: a volume mounted under the temp root. */
const disk = vi.hoisted(() => ({ otherDevice: [] as string[] }));

vi.mock('node:fs/promises', async (importOriginal) => {
  const real = await importOriginal<typeof import('node:fs/promises')>();
  const lstat = async (path: unknown, ...rest: unknown[]): Promise<Stats> => {
    const info = (await (real.lstat as (...a: unknown[]) => Promise<Stats>)(path, ...rest)) as Stats;
    const text = String(path);
    if (!disk.otherDevice.some((p) => text === p || text.startsWith(`${p}/`))) return info;
    return new Proxy(info, { get: (t, k) => (k === 'dev' ? t.dev + 1 : Reflect.get(t, k, t)) });
  };
  const mocked = { ...real, lstat };
  return { ...mocked, default: mocked };
});

const FIXTURES = join(dirname(fileURLToPath(import.meta.url)), 'fixtures', 'auto');
const GUARD = workGuardHookName('mlx-work-guard-0123456789abcdef');
const OPTIONS = [
  { optionId: 'allow-once', name: 'once', kind: 'allow_once' },
  { optionId: 'reject-once', name: 'no', kind: 'reject_once' },
];
const UID = process.getuid?.() ?? 0;
const LOGIN = 'owner';

type Suite = 'S1' | 'S2' | 'S3' | 'S4' | 'S5' | 'OFF';

interface Row {
  n: number;
  input: string;
  kind: 'bash' | 'edit';
  tool?: string;
  expect: 'allow' | AutoAsk;
  cause?: string;
  part?: string;
  suite?: Suite;
  setup?: 'config-0666' | 'hooks-0777';
  touched?: boolean;
  class?: string;
}

const ROWS = (JSON.parse(readFileSync(join(FIXTURES, 'temp-root.json'), 'utf8')) as { rows: Row[] }).rows;

let root: string;
let home: string;
let work: string;
/** The temp root: its real top and its other spelling (a link to it). */
let TR: string;
let T: string;

function plainRepo(folder: string, extra = ''): void {
  for (const d of ['.git/objects', '.git/refs/heads', '.git/hooks']) mkdirSync(join(folder, d), { recursive: true });
  writeFileSync(join(folder, '.git', 'HEAD'), 'ref: refs/heads/main\n');
  writeFileSync(join(folder, '.git', 'config'), `[core]\n\trepositoryformatversion = 0\n\tbare = false\n${extra}`);
  writeFileSync(join(folder, '.git', 'hooks', 'pre-commit.sample'), '#!/bin/sh\n');
}

beforeAll(() => {
  root = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-auto-temp-')));
  home = join(root, 'home');
  work = join(home, 'code', 'app');
  const other = join(home, 'code', 'other');
  TR = join(root, 'troot');
  T = join(root, 't');
  mkdirSync(join(work, 'sub'), { recursive: true });
  plainRepo(work);
  for (const f of ['a.txt', 'note.txt', 'sub/note.txt']) writeFileSync(join(work, f), 'x\n');
  mkdirSync(other, { recursive: true });
  writeFileSync(join(other, 'x'), 'x\n');
  symlinkSync(other, join(work, 'lnk2'));
  mkdirSync(TR, { mode: 0o755 });
  symlinkSync(TR, T);
  writeFileSync(join(TR, 'package.json'), JSON.stringify({ scripts: { test: 'true' } }));
  const scan = join(TR, 'scan');
  mkdirSync(join(scan, 'sub'), { recursive: true });
  mkdirSync(join(scan, 'lnk2'));
  for (const f of ['a.txt', 'b.txt', 'x', 'sub/note.txt', 'f.patch', 'lnk2/x']) writeFileSync(join(scan, f), 'x\n');
  symlinkSync(other, join(scan, 'lnk'));
  symlinkSync(join(work, 'sub'), join(scan, 'lnk-in'));
  symlinkSync(join(other, 'new'), join(scan, 'dang'));
  writeFileSync(join(other, 'hard'), 'x\n');
  linkSync(join(other, 'hard'), join(scan, 'hard'));
  execFileSync('/usr/bin/mkfifo', [join(scan, 'fifo')]);
  const keep = join(TR, 'keep');
  for (const d of ['.git', '.claude']) mkdirSync(join(keep, d), { recursive: true });
  writeFileSync(join(keep, '.git', 'config'), '[core]\n');
  writeFileSync(join(keep, '.claude', 'settings.json'), '{}\n');
  writeFileSync(join(keep, '.zshrc'), '\n');
  symlinkSync('.git', join(keep, 'cfg'));
  plainRepo(TR);
  const repo = join(TR, 'repo');
  plainRepo(repo);
  mkdirSync(join(repo, 'src'), { recursive: true });
  writeFileSync(join(repo, 'src', 'x.ts'), 'x\n');
  const bare = join(repo, 'barelike');
  for (const d of ['objects', 'refs']) mkdirSync(join(bare, d), { recursive: true });
  writeFileSync(join(bare, 'HEAD'), 'ref: refs/heads/main\n');
  plainRepo(join(repo, 'junkhead'));
  writeFileSync(join(repo, 'junkhead', '.git', 'HEAD'), 'not a ref\n');
  mkdirSync(join(repo, 'nohead', '.git'), { recursive: true });
  writeFileSync(join(repo, 'nohead', '.git', 'config'), '[core]\n\tbare = false\n');
  mkdirSync(join(TR, 'wt'));
  writeFileSync(join(TR, 'wt', '.git'), 'gitdir: /x\n');
  const web = join(TR, 'web');
  mkdirSync(web);
  writeFileSync(join(web, 'package.json'), JSON.stringify({ scripts: { test: 'true' } }));
  writeFileSync(join(web, 'a.js'), '\n');
  writeFileSync(join(web, 'tsconfig.json'), '{}\n');
  writeFileSync(join(TR, 'out.txt'), 'x\n');
  const tool = join(TR, `tool-${String(UID)}`);
  mkdirSync(tool);
  writeFileSync(join(tool, 'notes.txt'), 'x\n');
  writeFileSync(join(tool, 'tsconfig.json'), '{}\n');
  for (const d of [`pytest-of-${LOGIN}`, '.hidden', 'com.apple.launchd.x', 'chat-root', 'mnt']) mkdirSync(join(TR, d));
  writeFileSync(join(TR, 'mnt', 'a.txt'), 'x\n');
  const open = join(TR, 'open');
  mkdirSync(open);
  writeFileSync(join(open, 'a.txt'), 'x\n');
  symlinkSync(join(TR, 'scan', 'x'), join(open, 'x'));
  chmodSync(open, 0o777);
  symlinkSync(other, join(TR, 'lnk-out'));
  const locked = join(TR, 'locked');
  mkdirSync(locked);
  writeFileSync(join(locked, 'a.txt'), 'x\n');
  chmodSync(locked, 0o000);
  mkdirSync(join(TR, 'proj', 'sub'), { recursive: true });
  writeFileSync(join(TR, 'proj', 'a.txt'), 'x\n');
  plainRepo(join(TR, 'napi-rs-a6ih'));
  disk.otherDevice = [join(TR, 'mnt')];
});

afterAll(() => {
  chmodSync(join(TR, 'locked'), 0o755);
  rmSync(root, { recursive: true, force: true });
});

function tempRoot(): AllowedRoot {
  return { top: TR, spellings: [TR, T], kind: 'temp' };
}

/** The gate's place of a suite, and the card's (the Work folder only, as the host builds it). */
function places(suite: Suite): { gate: WorkPlace; card: WorkPlace } {
  const cwd = suite === 'S3' ? join(TR, 'proj') : suite === 'S5' ? TR : work;
  const card: WorkPlace = { cwd, home };
  const temp = suite === 'OFF' ? [] : [tempRoot()];
  const uid = suite === 'S2' ? UID + 1 : UID;
  return { card, gate: { ...card, temp, uid, login: LOGIN, refused: [join(TR, 'chat-root')] } };
}

function fill(text: string): string {
  const upper = join(dirname(T), basename(T).toUpperCase());
  return text
    .split('{TUP}')
    .join(upper)
    .split('{TR}')
    .join(TR)
    .split('{T}')
    .join(T)
    .split('{W}')
    .join(work)
    .split('{UID}')
    .join(String(UID));
}

function request(row: Row, input: string): PermissionRequest {
  const bash = row.kind === 'bash';
  const tool = bash ? 'run_terminal_command' : (row.tool ?? 'write');
  const rawInput = bash
    ? { variant: 'Bash', command: input, description: 'run', is_background: false }
    : tool === 'search_replace'
      ? { file_path: input, old_string: 'x', new_string: 'y' }
      : { file_path: input, content: 'x' };
  return {
    sessionId: 's1',
    toolCallId: 'call-1',
    kind: bash ? 'execute' : 'edit',
    title: tool,
    tool: { name: tool, namespace: 'grok_build' },
    rawInput,
    options: OPTIONS,
    hookAsk: { hookName: GUARD, reason: null },
  };
}

interface Seen {
  got: 'allow' | AutoAsk;
  cause: string | null;
  part: string | null;
  touched: boolean;
}

async function look(row: Row): Promise<Seen> {
  const { gate, card } = places(row.suite ?? 'S1');
  const input = fill(row.input);
  const req = request(row, input);
  const landing = row.kind === 'edit' ? await editLanding(agentPath(input, card)).catch(() => null) : null;
  const interaction = workInteraction('i1', req, [], card, landing);
  const facts = await autoFacts(req, gate, landing, { cdpath: false });
  const decision = decideAuto({
    auto: true,
    watched: true,
    request: req,
    interaction,
    guardHook: GUARD,
    elsewhere: false,
    facts,
    allowed: 0,
  });
  if (decision.allow) return { got: 'allow', cause: null, part: null, touched: facts.touched };
  return {
    got: decision.why ?? 'unread',
    cause: decision.detail?.cause ?? null,
    part: decision.detail?.part ?? null,
    touched: facts.touched,
  };
}

/** A row's setup on the layout; returns the undo. */
function setUp(row: Row): () => void {
  const git = join(TR, 'repo', '.git');
  const config = join(work, '.git', 'config');
  const kept = readFileSync(config, 'utf8');
  const undo: Array<() => void> = [];
  if (row.suite === 'S4') {
    writeFileSync(config, `${kept}\thooksPath = .husky/_\n`);
    undo.push(() => writeFileSync(config, kept));
  }
  if (row.setup === 'config-0666') {
    chmodSync(join(git, 'config'), 0o666);
    undo.push(() => chmodSync(join(git, 'config'), 0o644));
  }
  if (row.setup === 'hooks-0777') {
    chmodSync(join(git, 'hooks'), 0o777);
    undo.push(() => chmodSync(join(git, 'hooks'), 0o755));
  }
  return () => {
    for (const u of undo) u();
  };
}

/** What a row expects, in the shape of {@link look}: only the fields the row names. */
function wanted(row: Row): Partial<Seen> {
  return {
    got: row.expect,
    ...(row.cause === undefined ? {} : { cause: row.cause }),
    ...(row.part === undefined ? {} : { part: fill(row.part) }),
    ...(row.touched === undefined ? {} : { touched: row.touched }),
  };
}

async function check(row: Row): Promise<[string, Partial<Seen>]> {
  const undo = setUp(row);
  try {
    const seen = await look(row);
    const want = wanted(row);
    const shown: Partial<Seen> = {};
    for (const key of Object.keys(want) as Array<keyof Seen>) (shown as Record<string, unknown>)[key] = seen[key];
    return [`#${String(row.n)} ${row.suite ?? 'S1'} ${row.input}`, shown];
  } finally {
    undo();
  }
}

const root0 = UID === 0;

describe('the temp root fixture', () => {
  it.each(ROWS.filter((r) => !(root0 && (r.n === 84 || r.suite === 'S2'))).map((r) => [r.n, r.input, r] as const))(
    'row %i: %s',
    async (_n, _input, row) => {
      const [name, seen] = await check(row);
      expect([name, seen]).toEqual([name, wanted(row)]);
    },
  );

  it('no fixture row outside this file and no inline Auto row names a temp root', () => {
    const files = ['evidence-codex-r4', 'evidence-codex', 'evidence-disk-host', 'evidence-findings'];
    const named: string[] = [];
    for (const file of [...files, 'evidence-heads', 'evidence-review2', 'positive']) {
      const rows = (JSON.parse(readFileSync(join(FIXTURES, `${file}.json`), 'utf8')) as { rows: Row[] }).rows;
      for (const r of rows) if (namesTmp(r.input)) named.push(`${file}: ${r.input}`);
    }
    const tests = dirname(fileURLToPath(import.meta.url));
    for (const file of [
      'chat-auto-corpus',
      'chat-auto-decide',
      'chat-auto-disk-errors',
      'chat-auto-evidence',
      'chat-auto-grammar',
      'chat-auto-host-fail',
      'chat-auto-host-gen',
      'chat-auto-host',
      'chat-auto-known',
      'chat-auto-shell',
      'chat-mode-host',
    ]) {
      for (const literal of stringLiterals(readFileSync(join(tests, `${file}.test.ts`), 'utf8'))) {
        if (namesTmp(literal)) named.push(`${file}: ${literal}`);
      }
    }
    expect(named).toEqual([]);
  });
});

/** `tmp` as a whole path part right after a `/` that starts a path: `/tmp`, `//tmp`, `/private/tmp`, `../tmp`. */
function namesTmp(text: string): boolean {
  for (let at = text.indexOf('tmp'); at !== -1; at = text.indexOf('tmp', at + 1)) {
    if (text.charAt(at - 1) !== '/') continue;
    const after = text.charAt(at + 3);
    if (after !== '' && after !== '/' && after !== ' ' && after !== '"' && after !== "'" && after !== '`') continue;
    return true;
  }
  return false;
}

/** The text of each quoted string of a source file: single, double and backquoted. */
function stringLiterals(source: string): string[] {
  const out: string[] = [];
  for (let i = 0; i < source.length; i++) {
    const q = source.charAt(i);
    if (q === '/' && source.charAt(i + 1) === '/') {
      while (i < source.length && source.charAt(i) !== '\n') i++;
      continue;
    }
    if (q !== "'" && q !== '"' && q !== '`') continue;
    let j = i + 1;
    while (j < source.length && source.charAt(j) !== q) j += source.charAt(j) === '\\' ? 2 : 1;
    out.push(source.slice(i + 1, j));
    i = j;
  }
  return out;
}

describe("the owner's card", () => {
  const owner = (): string =>
    `ps aux | grep -E "cargo|rustc|lldb" | grep -v grep | head -10; echo "---"; ls ${T}/napi-rs-a6ih 2>/dev/null | head; echo "---"; cd ${T}/napi-rs-a6ih 2>/dev/null && git status --short | head -10`;

  async function verdictOf(input: string, suite: Suite = 'S1'): Promise<Seen> {
    const row: Row = { n: 0, input, kind: 'bash', expect: 'allow', suite };
    const undo = setUp(row);
    try {
      return await look(row);
    } finally {
      undo();
    }
  }

  it('each part runs with the temp root on, and so does the whole command', async () => {
    const parts: Array<[string, 'allow' | AutoAsk]> = [
      ['ps aux', 'allow'],
      ['echo x | grep -E "cargo|rustc|lldb"', 'allow'],
      ['echo x | grep -v grep', 'allow'],
      ['echo x | head -10', 'allow'],
      ['echo "---"', 'allow'],
      [`ls ${T}/napi-rs-a6ih 2>/dev/null`, 'allow'],
      ['echo x | head', 'allow'],
      [`cd ${T}/napi-rs-a6ih 2>/dev/null`, 'allow'],
      [`cd ${T}/napi-rs-a6ih 2>/dev/null && git status --short`, 'allow'],
    ];
    for (const [input, want] of parts) expect([input, (await verdictOf(input)).got]).toEqual([input, want]);
    expect(await verdictOf(owner())).toEqual({ got: 'allow', cause: null, part: null, touched: true });
    expect(await verdictOf(owner(), 'S4')).toEqual({ got: 'allow', cause: null, part: null, touched: true });
  });

  it('with no temp root it asks, and the reason names the cd', async () => {
    expect(await verdictOf(owner(), 'OFF')).toEqual({
      got: 'not-known',
      cause: null,
      part: `cd ${T}/napi-rs-a6ih`,
      touched: false,
    });
  });

  it('a repository there that is not yours, or that others can write, asks', async () => {
    const repo = join(TR, 'napi-rs-a6ih');
    chmodSync(repo, 0o775);
    try {
      expect(await verdictOf(owner())).toMatchObject({ got: 'temp', cause: 'shared' });
    } finally {
      chmodSync(repo, 0o755);
    }
    if (!root0) expect(await verdictOf(owner(), 'S2')).toMatchObject({ got: 'temp', cause: 'owner' });
  });
});

describe('roots as data', () => {
  it('the Work folder comes first; a temp root it holds or equals is dropped; under one it gets the alias spelling', () => {
    const t = tempRoot();
    expect(roots({ cwd: work, home, temp: [t] }).map((r) => r.kind)).toEqual(['work', 'temp']);
    expect(roots({ cwd: TR, home, temp: [t] })).toEqual([{ top: TR, spellings: [TR], kind: 'work' }]);
    expect(roots({ cwd: root, home, temp: [t] }).map((r) => r.kind)).toEqual(['work']);
    expect(roots({ cwd: join(TR, 'proj'), home, temp: [t] })[0]).toEqual({
      top: join(TR, 'proj'),
      spellings: [join(TR, 'proj'), join(T, 'proj')],
      kind: 'work',
    });
    const place: WorkPlace = { cwd: join(TR, 'proj'), home, temp: [t] };
    expect(rootText(join(T, 'proj', 'a.txt'), place)?.root.kind).toBe('work');
    expect(rootText(join(T, 'scan'), place)).toEqual({ root: t, path: join(TR, 'scan') });
    expect(rootText(`${T}x/scan`, place)).toBeNull();
    expect(rootOf('/dev/null', { cwd: '/dev', home })).toBeNull();
  });

  it('with no temp root every place rule is the Work folder alone', () => {
    const off: WorkPlace = { cwd: work, home, temp: [] };
    expect(roots(off)).toEqual([{ top: work, spellings: [work], kind: 'work' }]);
    expect(autoScan(`cat ${T}/scan/a.txt`, off, { cdpath: false }).checks).toBeNull();
    expect(commandFlags(`touch ${T}/scan/x`, off)).toEqual(['outside-folder']);
    expect(commandFlags(`touch ${T}/scan/x`, { ...off, temp: [tempRoot()] })).toEqual([]);
  });

  it('a private name: a dot name, com.apple., or a piece equal to the uid or the login', () => {
    for (const name of ['.X11-unix', 'com.apple.launchd.x', 'tool-501', 'pytest-of-Owner', '501']) {
      expect([name, privateName(name, 501, 'owner')]).toEqual([name, true]);
    }
    for (const name of ['napi-rs-a6ih', 'jiti', 'tool-5010', 'owners', 'scan']) {
      expect([name, privateName(name, 501, 'owner')]).toEqual([name, false]);
    }
    expect(privateName('COM.APPLE.launchd.x', 501, 'owner')).toBe(true);
  });
});

describe('a refused folder made after the host started', () => {
  it('is refused by its real path, also when it is named through a link', async () => {
    const refused = realFolder(join(T, 'late-root', 'chat'));
    expect(refused).toBe(join(TR, 'late-root', 'chat'));
    mkdirSync(join(TR, 'late-root', 'chat'), { recursive: true });
    writeFileSync(join(TR, 'late-root', 'chat', 'c.json'), '{}\n');
    try {
      const input = `cat ${T}/late-root/chat/c.json`;
      const req = request({ n: 0, input, kind: 'bash', expect: 'allow' }, input);
      const { gate, card } = places('S1');
      const facts = await autoFacts(req, { ...gate, refused: [refused] }, null, { cdpath: false });
      const decision = decideAuto({
        auto: true,
        watched: true,
        request: req,
        interaction: workInteraction('i1', req, [], card, null),
        guardHook: GUARD,
        elsewhere: false,
        facts,
        allowed: 0,
      });
      expect(decision.allow).toBe(false);
      if (!decision.allow) expect([decision.why, decision.detail?.cause]).toEqual(['temp', 'app']);
    } finally {
      rmSync(join(TR, 'late-root'), { recursive: true, force: true });
    }
  });
});

describe('another case of a refused or private folder', () => {
  async function verdictOf(input: string): Promise<Seen> {
    return look({ n: 0, input, kind: 'bash', expect: 'allow' });
  }

  it('asks through a cd and a pattern, as the lower-case spelling does', async () => {
    const blind = existsSync(join(TR, 'CHAT-ROOT'));
    const rows: Array<[string, string]> = [
      [`cd ${T}/CHAT-ROOT && ls -la`, 'app'],
      [`cat ${T}/CHAT-ROOT/*`, 'app'],
      [`cd ${T}/COM.APPLE.launchd.x && ls`, 'private'],
      [`wc -l ${T}/COM.APPLE.launchd.x/*`, 'private'],
    ];
    for (const [input, cause] of rows) {
      const seen = await verdictOf(input);
      const want = blind ? { got: 'temp', cause } : { got: 'unknown-target' };
      expect([input, seen]).toEqual([input, expect.objectContaining(want)]);
    }
    const mixed = await verdictOf(`cd ${T}/SCAN && ls`);
    expect(mixed.got).toBe('unknown-target');
  });
});

describe('where the shell is: && after an absolute cd into the temp root', () => {
  it('narrows only inside one and-or list with no ||', () => {
    const place: WorkPlace = { cwd: work, home, temp: [tempRoot()] };
    const repo = join(TR, 'repo');
    expect(scanDirs(`cd ${T}/repo && git status --short | head -10`, place)).toEqual([[work], [repo], [repo]]);
    expect(scanDirs(`cd ${T}/repo; git status --short`, place)).toEqual([[work], [work, repo]]);
    expect(scanDirs(`true || cd ${T}/repo && git status --short`, place)).toEqual([[work], [work], [work, repo]]);
    expect(scanDirs(`cd ${T}/repo && ls\nls`, place)).toEqual([[work], [repo], [work, repo]]);
    expect(scanDirs(`cd sub && ls`, place)).toEqual([[work], [work, join(work, 'sub')]]);
  });

  it("the cd's own redirect is placed where the shell was", () => {
    const place: WorkPlace = { cwd: work, home, temp: [tempRoot()] };
    const checks = autoScan(`cd ${T}/scan > o.txt && ls`, place, { cdpath: false }).checks ?? [];
    expect(checks.filter((c) => c.kind === 'write').map((c) => (c.kind === 'write' ? c.path : ''))).toEqual([
      join(work, 'o.txt'),
    ]);
  });
});

describe('the temp root resolver', () => {
  it('gives no root for a folder that is not root-owned and sticky, holds the home, or holds an app temp folder', () => {
    const plain = join(root, 'plain');
    mkdirSync(plain, { recursive: true });
    expect(resolveTempRoot(plain, home, [])).toEqual([]);
    expect(resolveTempRoot(join(root, 'missing'), home, [])).toEqual([]);
  });

  it("reads the machine's /tmp by the same rules", () => {
    const real = realpathSync('/tmp');
    const info = statSync(real);
    const ok = info.isDirectory() && info.uid === 0 && (info.mode & 0o1000) !== 0;
    const away = join(root, 'home-away');
    const want = ok ? [{ top: real, spellings: real === '/tmp' ? [real] : [real, '/tmp'], kind: 'temp' }] : [];
    expect(resolveTempRoot('/tmp', away, [tmpdir()])).toEqual(
      realpathSync(tmpdir()).startsWith(`${real}/`) ? [] : want,
    );
    expect(resolveTempRoot('/tmp', away, [join(real, 'x')])).toEqual([]);
    expect(resolveTempRoot('/tmp', join(real, 'home-in-tmp'), [])).toEqual([]);
  });
});
