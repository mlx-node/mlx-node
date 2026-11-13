/**
 * The Auto gate against every input of the fourth review (`fixtures/auto/evidence-codex-r4.json`) on a real folder
 * with links to outside, names stored decomposed, and a git folder whose config each row may change: every input of the
 * findings and of their class asks, and the plain uses listed run with no card. Only text goes to the scan; no command
 * of the table is run.
 */

import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  renameSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { afterAll, beforeAll, describe, expect, it } from 'vite-plus/test';

import type { PermissionRequest } from '../src/chat/acp/messages.js';
import { foldName } from '../src/chat/agent/auto-scan.js';
import {
  autoFacts,
  configAllowed,
  decideAuto,
  indexHasGitlink,
  mayMatch,
  networkRemote,
  parseGitConfig,
} from '../src/chat/agent/auto.js';
import type { WorkPlace } from '../src/chat/agent/command-flags.js';
import { workInteraction } from '../src/chat/agent/permissions.js';
import { workGuardHookName } from '../src/chat/agent/profiles.js';
import type { AutoAsk } from '../src/chat/types.js';

const FIXTURES = join(dirname(fileURLToPath(import.meta.url)), 'fixtures', 'auto');
const GUARD = workGuardHookName('mlx-work-guard-0123456789abcdef');
const OPTIONS = [
  { optionId: 'allow-once', name: 'once', kind: 'allow_once' },
  { optionId: 'reject-once', name: 'no', kind: 'reject_once' },
];

interface Row {
  input: string;
  kind: 'bash';
  expect: 'ask' | 'allow';
  class: string;
  setup?: string;
}

const ROWS = (JSON.parse(readFileSync(join(FIXTURES, 'evidence-codex-r4.json'), 'utf8')) as { rows: Row[] }).rows;

let root: string;
let out: string;
let place: WorkPlace;

interface Remote {
  url?: string;
  pushurl?: string;
  branchRemote?: string;
}

/** The folder's git config as `git clone` writes it, which every key of passes. */
function config(remote: Remote = {}): string {
  return [
    '[core]',
    '\trepositoryformatversion = 0',
    '\tfilemode = true',
    '\tbare = false',
    '\tlogallrefupdates = true',
    '\tignorecase = true',
    '\tprecomposeunicode = true',
    '[remote "origin"]',
    `\turl = ${remote.url ?? 'git@github.com:o/app.git'}`,
    ...(remote.pushurl === undefined ? [] : [`\tpushurl = ${remote.pushurl}`]),
    '\tfetch = +refs/heads/*:refs/remotes/origin/*',
    '[branch "main"]',
    `\tremote = ${remote.branchRemote ?? 'origin'}`,
    '\tmerge = refs/heads/main',
    '[user]',
    '\tname = x',
    '\temail = x@example.com',
    '',
  ].join('\n');
}

/** Config text added after the plain config, by setup name. */
const ADDED: Readonly<Record<string, () => string>> = {
  worktree: () => `[core]\n\tworktree = ${out}\n`,
  fsmonitor: () => '[core]\n\tfsmonitor = touch marker\n',
  'diff-external': () => '[diff]\n\texternal = x\n',
  'gpg-sign': () => '[commit]\n\tgpgsign = true\n[gpg]\n\tprogram = x\n',
  'show-signature': () => '[log]\n\tshowSignature = true\n',
  uploadpack: () => '[remote "origin"]\n\tuploadpack = touch marker\n',
  credential: () => '[credential]\n\thelper = x\n',
  'ssh-command': () => '[core]\n\tsshCommand = x\n',
  'worktree-config': () => '[extensions]\n\tworktreeConfig = true\n',
  textconv: () => '[diff "x"]\n\ttextconv = x\n',
  filter: () => '[filter "x"]\n\tclean = x\n',
  'hooks-path': () => '[core]\n\thooksPath = .husky/_\n',
  include: () => '[include]\n\tpath = x\n',
  'include-if': () => '[includeIf "gitdir:~/"]\n\tpath = x\n',
  quoted: () => '[user]\n\tname = "x"\n',
  continuation: () => '[user]\n\tname = a\\\n\tb\n',
  'comment-after': () => '[user]\n\tname = a # c\n',
  'dotted-section': () => '[remote.origin]\n\turl = git@github.com:o/app.git\n',
  bare: () => '[core]\n\tbare = true\n',
  'format-2': () => '[core]\n\trepositoryformatversion = 2\n',
  'partial-clone': () => '[extensions]\n\tpartialClone = origin\n',
  'url-insteadof': () => '[url "file:///x"]\n\tinsteadOf = https://\n',
  'core-editor': () => '[core]\n\teditor = vim\n',
  'remote-proxy': () => '[remote "origin"]\n\tproxy = x\n',
  'remote-vcs': () => '[remote "origin"]\n\tvcs = x\n',
  submodule: () => '[submodule "s"]\n\turl = x\n',
};

/** An absolute folder outside the Work folder: `<home>/code/other`. */
function outside(): string {
  return join(place.home, 'code', 'other');
}

/** Remote settings by setup name: they change only what fetch, pull and push reach. */
const REMOTES: Readonly<Record<string, () => Remote>> = {
  'local-url': () => ({ url: '../out/bare.git' }),
  'file-url': () => ({ url: `file://${outside()}/x.git` }),
  'ext-url': () => ({ url: 'ext::sh' }),
  'helper-url': () => ({ url: 'foo::https://example.com/x' }),
  'dash-host': () => ({ url: 'ssh://-oProxyCommand=x/y' }),
  'local-pushurl': () => ({ pushurl: `${outside()}/x.git` }),
  'abs-url': () => ({ url: '/srv/x.git' }),
  'scp-slash': () => ({ url: './x:y' }),
  'branch-local-remote': () => ({ branchRemote: '../out/bare.git' }),
  'branch-unknown-remote': () => ({ branchRemote: 'up' }),
};

/** A git index as git writes it (entries in order, checksum left zero), version 2 or 4. */
function indexBytes(entries: ReadonlyArray<[mode: number, name: string]>, version: 2 | 4): Buffer {
  const head = Buffer.alloc(12);
  head.write('DIRC', 0, 'latin1');
  head.writeUInt32BE(version, 4);
  head.writeUInt32BE(entries.length, 8);
  const parts = [head];
  let previous = 0;
  for (const [mode, name] of entries) {
    const text = Buffer.from(name);
    const fixed = Buffer.alloc(62);
    fixed.writeUInt32BE(mode, 24);
    fixed.writeUInt16BE(Math.min(text.length, 0xfff), 60);
    if (version === 4) parts.push(fixed, Buffer.from([previous]), text, Buffer.from([0]));
    else parts.push(fixed, text, Buffer.alloc(((62 + text.length + 8) & ~7) - 62 - text.length));
    previous = text.length;
  }
  parts.push(Buffer.alloc(20));
  return Buffer.concat(parts);
}

/** Changes the folder's git setup for one row; returns the undo. */
function setUp(name: string): () => void {
  const git = join(place.cwd, '.git');
  const file = join(git, 'config');
  const plain = config();
  const added = ADDED[name];
  if (added !== undefined) {
    writeFileSync(file, `${plain}${added()}`);
    return () => writeFileSync(file, plain);
  }
  const remote = REMOTES[name];
  if (remote !== undefined) {
    writeFileSync(file, config(remote()));
    return () => writeFileSync(file, plain);
  }
  const real = join(place.cwd, '.git-real');
  switch (name) {
    case 'git-file':
      renameSync(git, real);
      writeFileSync(git, 'gitdir: .git-real\n');
      return () => {
        rmSync(git);
        renameSync(real, git);
      };
    case 'git-link':
      renameSync(git, real);
      symlinkSync('.git-real', git);
      return () => {
        rmSync(git);
        renameSync(real, git);
      };
    case 'commondir':
      writeFileSync(join(git, 'commondir'), `${out}/common\n`);
      return () => rmSync(join(git, 'commondir'));
    case 'modules':
      mkdirSync(join(git, 'modules', 'sub'), { recursive: true });
      return () => rmSync(join(git, 'modules'), { recursive: true });
    case 'config-link':
      writeFileSync(join(out, 'config'), plain);
      rmSync(file);
      symlinkSync(join(out, 'config'), file);
      return () => {
        rmSync(file);
        writeFileSync(file, plain);
      };
    case 'config-missing':
      rmSync(file);
      return () => writeFileSync(file, plain);
    case 'nested-git':
      mkdirSync(join(place.cwd, 'e', '.git'), { recursive: true });
      writeFileSync(join(place.cwd, 'e', '.git', 'config'), '[core]\n\tfsmonitor = touch marker\n');
      return () => rmSync(join(place.cwd, 'e', '.git'), { recursive: true });
    case 'alternates':
      mkdirSync(join(git, 'objects', 'info'), { recursive: true });
      writeFileSync(join(git, 'objects', 'info', 'alternates'), `${out}/objects\n`);
      return () => rmSync(join(git, 'objects'), { recursive: true });
    case 'git-entry-link':
      symlinkSync(join(out, 'dir'), join(git, 'refs'));
      return () => rmSync(join(git, 'refs'));
    case 'gitlink-index':
    case 'gitlink-index-v4':
    case 'plain-index':
    case 'plain-index-v4': {
      const v4 = name.endsWith('-v4');
      const sub: Array<[number, string]> = name.startsWith('gitlink') ? [[0o160000, 'e']] : [];
      writeFileSync(join(git, 'index'), indexBytes([[0o100644, 'a.txt'], ...sub, [0o120000, 'k']], v4 ? 4 : 2));
      return () => rmSync(join(git, 'index'));
    }
    default:
      throw new Error(`no setup ${name}`);
  }
}

beforeAll(() => {
  root = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-auto-evidence-')));
  const home = join(root, 'home');
  const cwd = join(home, 'code', 'app');
  out = join(root, 'out');
  for (const d of ['d', 'e', 'src', 'nfd', 'tests', '.git/hooks', 'node_modules'])
    mkdirSync(join(cwd, d), { recursive: true });
  for (const d of ['pkg', 'dir', 'common']) mkdirSync(join(out, d), { recursive: true });
  for (const f of ['f', 'h', 't.py', 'test_case.py', 'pkg/package.json']) writeFileSync(join(out, f), 'ORIG\n');
  for (const f of ['a.txt', 'b.txt', 'file', 'src/a.ts', 'tests/test_a.py', 'tsconfig.json', 'nfd/é.ts']) {
    writeFileSync(join(cwd, f), 'x\n');
  }
  writeFileSync(join(cwd, '.git', 'config'), config());
  symlinkSync(out, join(cwd, 'd', 'k'));
  symlinkSync('e', join(cwd, 'lnkE'));
  symlinkSync(join(out, 'pkg'), join(cwd, 'node_modules', 'foo'));
  for (const name of ['éx', '한x', 'がx', 'Åx']) {
    symlinkSync(join(out, 'f'), join(cwd, 'src', name));
  }
  symlinkSync(join(out, 't.py'), join(cwd, 'test_link.py'));
  symlinkSync(out, join(cwd, 'k'));
  symlinkSync(join(out, 't.py'), join(cwd, 'a.test.ts'));
  symlinkSync(join(out, 'dir'), join(cwd, 'tests', 'far'));
  mkdirSync(join(cwd, 'sel'));
  for (const f of ['a.test.ts:3', 'k*', 'k.test_case.py', 'b.py::t']) writeFileSync(join(cwd, 'sel', f), 'x\n');
  symlinkSync(join(out, 't.py'), join(cwd, 'sel', 'a.test.ts'));
  symlinkSync(join(out, 't.py'), join(cwd, 'sel', 'b.py'));
  symlinkSync(out, join(cwd, 'sel', 'k'));
  place = { cwd, home };
});

afterAll(() => {
  rmSync(root, { recursive: true, force: true });
});

function request(command: string): PermissionRequest {
  return {
    sessionId: 's1',
    toolCallId: 'call-1',
    kind: 'execute',
    title: 'run_terminal_command',
    tool: { name: 'run_terminal_command', namespace: 'grok_build' },
    rawInput: { variant: 'Bash', command, description: 'run', is_background: false },
    options: OPTIONS,
    hookAsk: { hookName: GUARD, reason: null },
  };
}

/** `gate`: the place the gate judges (the card's is always the Work folder alone). */
async function verdict(command: string, gate: WorkPlace = place): Promise<'allow' | AutoAsk> {
  const req = request(command);
  const interaction = workInteraction('i1', req, [], place, null);
  const facts = await autoFacts(req, gate, null, { cdpath: false });
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
  return decision.allow ? 'allow' : (decision.why ?? 'unread');
}

async function wrong(list: readonly Row[]): Promise<string[]> {
  const seen: string[] = [];
  for (const row of list) {
    const undo = row.setup === undefined ? null : setUp(row.setup);
    try {
      const got = await verdict(row.input);
      if ((got === 'allow') !== (row.expect === 'allow')) seen.push(`${row.setup ?? '-'} | ${row.input} | ${got}`);
    } finally {
      undo?.();
    }
  }
  return seen;
}

const of = (prefix: string): Row[] => ROWS.filter((r) => r.class.startsWith(prefix));

interface Change {
  file: string;
  row: number;
  input: string;
  from: string;
  to: string;
  reason: string;
}

describe('the temp root is data: every row answers the same unless a reviewed line says why', () => {
  it('pass B (an empty temp root) changes no row; pass C (a temp root over the whole layout) changes only the listed rows', async () => {
    const changes = (
      JSON.parse(readFileSync(join(FIXTURES, 'temp-root-changes.json'), 'utf8')) as { root: Change[] }
    ).root.filter((x) => x.file === 'evidence-codex-r4.json');
    const empty = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-auto-empty-')));
    try {
      const b: WorkPlace = {
        ...place,
        temp: [{ top: empty, spellings: [empty], kind: 'temp' }],
        uid: process.getuid?.(),
      };
      const c: WorkPlace = {
        ...place,
        temp: [{ top: root, spellings: [root], kind: 'temp' }],
        uid: process.getuid?.(),
      };
      const offB: string[] = [];
      const changed: Array<Omit<Change, 'reason'>> = [];
      for (let k = 0; k < ROWS.length; k++) {
        const row = ROWS[k]!;
        const undo = row.setup === undefined ? null : setUp(row.setup);
        try {
          const a = await verdict(row.input);
          const bAnswer = await verdict(row.input, b);
          if (bAnswer !== a) offB.push(`#${String(k)} ${row.input} ${a} -> ${bAnswer}`);
          const cAnswer = await verdict(row.input, c);
          if (cAnswer !== a)
            changed.push({ file: 'evidence-codex-r4.json', row: k, input: row.input, from: a, to: cAnswer });
        } finally {
          undo?.();
        }
      }
      expect(offB).toEqual([]);
      expect(changed).toEqual(changes.map(({ file, row, input, from, to }) => ({ file, row, input, from, to })));
      for (const x of changes) expect([x.row, ROWS[x.row]?.input]).toEqual([x.row, x.input]);
    } finally {
      rmSync(empty, { recursive: true, force: true });
    }
  });
});

describe('the fourth review', () => {
  it('C1: printf passes only with a format of literal text, %s, %% and the escapes \\n \\t \\\\', async () => {
    expect(of('C1').length).toBeGreaterThan(50);
    expect(await wrong(of('C1'))).toEqual([]);
  });

  it('C2: mv with more than one source asks; one source, and cp, still run', async () => {
    expect(await wrong(of('C2'))).toEqual([]);
  });

  it('C3: a glob with a non-ASCII character asks, and a name the matcher cannot rule out after a fold is checked', async () => {
    expect(await wrong(of('C3'))).toEqual([]);
  });

  it('C4: a project tool operand must name an existing file or real folder with no link; selectors and filters ask', async () => {
    expect(await wrong(of('C4'))).toEqual([]);
  });

  it('C5: every git call asks unless .git is a real folder and every key of its config is on the allow list', async () => {
    const rows = of('C5');
    expect(rows.filter((r) => r.expect === 'ask').length).toBeGreaterThan(600);
    expect(await wrong(rows)).toEqual([]);
  });

  it('every row of the fixture is in one of the five classes', () => {
    expect(ROWS.filter((r) => !['C1', 'C2', 'C3', 'C4', 'C5'].some((c) => r.class.startsWith(c)))).toEqual([]);
  });
});

describe('git state the config does not show', () => {
  const rows: ReadonlyArray<[setup: string, input: string, expect: 'ask' | 'allow']> = [
    ['nested-git', 'cd e && git status', 'ask'],
    ['nested-git', 'cd e && git log', 'ask'],
    ['nested-git', 'cd src && git status', 'allow'],
    ['nested-git', 'git status', 'allow'],
    ['alternates', 'git log', 'ask'],
    ['alternates', 'git status', 'ask'],
    ['git-entry-link', 'git log', 'ask'],
    ['git-entry-link', 'git commit -m x', 'ask'],
    ['gitlink-index', 'git status', 'ask'],
    ['gitlink-index', 'git diff', 'ask'],
    ['gitlink-index', 'git log', 'ask'],
    ['gitlink-index-v4', 'git status', 'ask'],
    ['plain-index', 'git status', 'allow'],
    ['plain-index', 'git diff', 'allow'],
    ['plain-index-v4', 'git status', 'allow'],
  ];

  it('git asks with another repository on its way up, alternates, a link in .git, or a submodule in the index', async () => {
    const seen: string[] = [];
    for (const [setup, input, expected] of rows) {
      const undo = setUp(setup);
      try {
        const got = await verdict(input);
        if ((got === 'allow') !== (expected === 'allow')) seen.push(`${setup} | ${input} | ${got}`);
      } finally {
        undo();
      }
    }
    expect(seen).toEqual([]);
  });

  it('the index reader walks every entry to the checksum and reads only the extensions it knows', () => {
    const plain = indexBytes([[0o100644, 'a.txt']], 2);
    expect(indexHasGitlink(plain, 20)).toBe(false);
    expect(
      indexHasGitlink(
        indexBytes(
          [
            [0o100644, 'a.txt'],
            [0o100755, 'b'],
          ],
          4,
        ),
        20,
      ),
    ).toBe(false);
    expect(indexHasGitlink(indexBytes([[0o160000, 'sub']], 2), 20)).toBe(true);
    expect(indexHasGitlink(plain.subarray(0, plain.length - 1), 20)).toBe(true);
    expect(indexHasGitlink(plain, 32)).toBe(true);
    const bad = Buffer.from(plain);
    bad.writeUInt32BE(5, 4);
    expect(indexHasGitlink(bad, 20)).toBe(true);
    const withExtension = (name: string): Buffer => {
      const ext = Buffer.alloc(12);
      ext.write(name, 0, 'latin1');
      ext.writeUInt32BE(4, 4);
      return Buffer.concat([plain.subarray(0, plain.length - 20), ext, Buffer.alloc(20)]);
    };
    expect(indexHasGitlink(withExtension('TREE'), 20)).toBe(false);
    for (const name of ['link', 'sdir', 'FSMN', 'ZZZZ'])
      expect([name, indexHasGitlink(withExtension(name), 20)]).toEqual([name, true]);
    const extended = Buffer.from(plain);
    extended.writeUInt16BE(0x4000 | 5, 12 + 60);
    expect(indexHasGitlink(extended, 20)).toBe(true);
  });

  it('a tool selector, line filter or pattern asks even when a file of that very name exists', async () => {
    const seen: string[] = [];
    for (const input of [
      'vitest run sel/a.test.ts:3',
      'pytest sel/b.py::t',
      "prettier --check 'sel/k*'",
      'python3 -m unittest sel/k.test_case.py',
    ]) {
      const got = await verdict(input);
      if (got === 'allow') seen.push(input);
    }
    expect(seen).toEqual([]);
  });

  it('test and [ place the files they look at', async () => {
    const rows2: ReadonlyArray<[string, 'ask' | 'allow']> = [
      ['test -e ../out/f', 'ask'],
      ['test -d /etc', 'ask'],
      ['[ -f /etc/hosts ]', 'ask'],
      ['test /etc/hosts -nt a.txt', 'ask'],
      ['test a.txt -ot k/f', 'ask'],
      ['test -L d/k/f', 'ask'],
      ['test -f a.txt', 'allow'],
      ['[ -d src ]', 'allow'],
      ['test a.txt -nt b.txt', 'allow'],
      ['test -n /etc', 'allow'],
      ['test -z x', 'allow'],
    ];
    const seen: string[] = [];
    for (const [input, expected] of rows2) {
      const got = await verdict(input);
      if ((got === 'allow') !== (expected === 'allow')) seen.push(`${input} | ${got}`);
    }
    expect(seen).toEqual([]);
  });
});

describe('name rules', () => {
  it('a pattern may match more names than the shell would, never fewer: a non-ASCII name may take letters it can show as', () => {
    const yes: Array<[string, string]> = [
      ['e*', 'éx'],
      ['?x', 'éx'],
      ['*x', '한x'],
      ['k*', 'Kelvin'],
      ['s*', 'ſx'],
      ['a*', 'Åx'],
      ['*', 'がx'],
      ['e?.ts', 'é.ts'],
      ['*.ts', 'é.ts'],
      ['fi*', 'ﬁx'],
      ['é*', 'zzz'],
    ];
    const no: Array<[string, string]> = [
      ['*.ts', 'éx'],
      ['a*', '한x'],
      ['a*', 'がx'],
      ['x*', 'éx'],
    ];
    for (const [pattern, name] of yes) expect([pattern, name, mayMatch(pattern, name)]).toEqual([pattern, name, true]);
    for (const [pattern, name] of no) expect([pattern, name, mayMatch(pattern, name)]).toEqual([pattern, name, false]);
  });

  it('the name fold reaches a fixed point for every code point, and ẞ, ß and ss share one key', () => {
    const moving: string[] = [];
    for (let code = 0; code <= 0x10ffff; code++) {
      if (code >= 0xd800 && code <= 0xdfff) continue;
      const c = String.fromCodePoint(code);
      const once = foldName(`x${c}y`);
      if (foldName(once) !== once) moving.push(code.toString(16));
    }
    expect(moving).toEqual([]);
    expect(new Set(['ẞ', 'ß', 'ss', 'SS', 'sS'].map((n) => foldName(`n${n}`))).size).toBe(1);
  });
});

describe('git config rules', () => {
  it('the parser reads only the plain form git writes', () => {
    expect(parseGitConfig(config())).not.toBeNull();
    for (const text of [
      'name = x\n',
      '[core]\n\tname = "x"\n',
      '[core]\n\tname = a\\\n',
      '[core]\n\tname = a # c\n',
      '[core]\n\tname = a ; c\n',
      '[remote.origin]\n\turl = x\n',
      '[core\n',
      '[remote "a"b"]\n',
      '[core]\r\n\tbare = false\r\n',
      '[core]\n\t1x = y\n',
    ]) {
      expect([text, parseGitConfig(text)]).toEqual([text, null]);
    }
  });

  it('only keys on the list pass; a remote must be a network one for fetch, pull and push', () => {
    const allowed = (text: string, network: boolean): boolean => configAllowed(parseGitConfig(text) ?? [], network);
    expect(allowed(config(), true)).toBe(true);
    expect(allowed(`${config()}[core]\n\tworktree = /x\n`, false)).toBe(false);
    expect(allowed(`${config()}[CORE]\n\tFsMonitor = x\n`, false)).toBe(false);
    expect(allowed(`${config()}[toString]\n\tx = y\n`, false)).toBe(false);
    expect(allowed(config({ url: '../x.git' }), false)).toBe(true);
    expect(allowed(config({ url: '../x.git' }), true)).toBe(false);
    for (const url of [
      'https://github.com/o/r.git',
      'http://host:8080/r',
      'ssh://git@host/r.git',
      'git://host/r',
      'git@github.com:o/r.git',
      'host:r.git',
    ]) {
      expect([url, networkRemote(url)]).toEqual([url, true]);
    }
    for (const url of [
      '../x.git',
      '/srv/x.git',
      './x:y',
      'file:///x',
      'ext::sh -c x',
      'foo::https://x',
      'ssh://-oProxyCommand=x/y',
      'https://-x/y',
      'http://host:80x/r',
      ':x',
      'host:',
      'rsync://host/x',
    ]) {
      expect([url, networkRemote(url)]).toEqual([url, false]);
    }
  });
});
