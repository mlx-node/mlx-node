/**
 * The Auto gate against every input of the review evidence (fixtures under `fixtures/auto/`) on a real folder
 * with links and hard links inside and to outside: each input a review found allowed asks now, the plain uses of the
 * brief still run with no card, and each option that is not on a head's table asks. Only text goes to the scan; no
 * command of these tables is run.
 */

import { execFileSync } from 'node:child_process';
import {
  closeSync,
  linkSync,
  mkdirSync,
  openSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { afterAll, beforeAll, describe, expect, it } from 'vite-plus/test';

import type { PermissionRequest } from '../src/chat/acp/messages.js';
import { GIT_RULES, headTable, TEST_UNARY, TOOL_RULES, type OptionTable } from '../src/chat/agent/auto-scan.js';
import { autoFacts, decideAuto, mayMatch } from '../src/chat/agent/auto.js';
import { agentPath, type WorkPlace } from '../src/chat/agent/command-flags.js';
import { workInteraction } from '../src/chat/agent/permissions.js';
import { workGuardHookName } from '../src/chat/agent/profiles.js';
import type { AutoAsk } from '../src/chat/types.js';
import { editLanding } from '../src/chat/work-folder.js';

const FIXTURES = join(dirname(fileURLToPath(import.meta.url)), 'fixtures', 'auto');
const GUARD = workGuardHookName('mlx-work-guard-0123456789abcdef');
const OPTIONS = [
  { optionId: 'allow-once', name: 'once', kind: 'allow_once' },
  { optionId: 'reject-once', name: 'no', kind: 'reject_once' },
];

interface Row {
  input: string;
  kind: 'bash' | 'edit';
  tool?: string;
  expect: 'ask' | 'allow';
  class?: string;
}

function rows(name: string): Row[] {
  return (JSON.parse(readFileSync(join(FIXTURES, name), 'utf8')) as { rows: Row[] }).rows;
}

let root: string;
let place: WorkPlace;

/** The three review folders in one: links inside and out, hard links out, a protected folder by another name. */
beforeAll(() => {
  root = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-auto-corpus-')));
  const home = join(root, 'home');
  const cwd = join(home, 'code', 'app');
  const out = join(root, 'out');
  for (const d of [
    '.git/hooks',
    'src',
    'tests',
    'sub',
    'a/b',
    'q',
    'd',
    'e',
    'dest',
    'node_modules/.pnpm/react@18/node_modules/react',
  ]) {
    mkdirSync(join(cwd, d), { recursive: true });
  }
  mkdirSync(join(home, 'code', 'other'), { recursive: true });
  mkdirSync(join(out, 'sub'), { recursive: true });
  for (const n of ['f', 'g', 'hq', 'he', 'nb', 'file', 'h']) writeFileSync(join(out, n), 'ORIG\n');
  for (const f of [
    'a.txt',
    'b.txt',
    'a.js',
    'a.ts',
    'x.py',
    'Makefile',
    'package.json',
    'file',
    'script.js',
    'script.py',
  ]) {
    writeFileSync(join(cwd, f), 'x\n');
  }
  writeFileSync(join(cwd, 'package.json'), JSON.stringify({ scripts: { build: 'x', test: 'x' } }));
  writeFileSync(
    join(cwd, '.git', 'config'),
    '[core]\n\tbare = false\n[remote "origin"]\n\turl = git@example.com:o/app.git\n',
  );
  symlinkSync('a/b', join(cwd, 'L'));
  symlinkSync('a/b', join(cwd, 's'));
  symlinkSync('.pnpm/react@18/node_modules/react', join(cwd, 'node_modules', 'react'));
  linkSync(join(out, 'hq'), join(cwd, 'q', 'h'));
  symlinkSync(out, join(cwd, 'q', 'M'));
  symlinkSync(join(out, 'f'), join(cwd, 'd', 'a.txt'));
  linkSync(join(out, 'he'), join(cwd, 'e', 'h'));
  symlinkSync(out, join(cwd, 'k'));
  symlinkSync(out, join(cwd, 'src', 'lnk'));
  symlinkSync('.git/hooks', join(cwd, 'hooks'));
  symlinkSync(join(out, 'new-file'), join(cwd, 'dang'));
  symlinkSync('chain2', join(cwd, 'chain1'));
  symlinkSync(join(out, 'g'), join(cwd, 'chain2'));
  symlinkSync('loop', join(cwd, 'loop'));
  symlinkSync(join(out, 'nb'), join(cwd, 'n b.txt'));
  symlinkSync(join(out, 'file'), join(cwd, 'dest', 'a.txt'));
  linkSync(join(out, 'h'), join(cwd, 'h'));
  mkdirSync(join(home, '.ssh'));
  writeFileSync(join(home, '.ssh', 'id_rsa'), 'FAKE\n');
  writeFileSync(join(cwd, 'src', 'x.ts'), 'x\n');
  writeFileSync(join(out, 'evil.ts'), 'x\n');
  symlinkSync(join(home, '.ssh'), join(cwd, 'src', 'keys'));
  symlinkSync(join(home, '.ssh'), join(cwd, 'lk'));
  symlinkSync(join(home, '.ssh'), join(cwd, 'd', 's'));
  symlinkSync(out, join(cwd, 'd', 'l'));
  symlinkSync('/dev/stdout', join(cwd, 'so'));
  symlinkSync(join(out, 'evil.ts'), join(cwd, 'l.ts'));
  execFileSync('/usr/bin/mkfifo', [join(cwd, 'fifo')]);
  for (const decoy of [
    'https:/x/y.ts',
    'npm:pkg@1.0.0/x.js',
    'jsr:@std/x/mod.ts',
    'data:text/javascript,console.log%28%27x%27%29/x.js',
    'a/b.js',
  ]) {
    mkdirSync(dirname(join(cwd, decoy)), { recursive: true });
    writeFileSync(join(cwd, decoy), 'x\n');
  }
  place = { cwd, home };
});

afterAll(() => {
  rmSync(root, { recursive: true, force: true });
});

function request(row: Row): PermissionRequest {
  const bash = row.kind === 'bash';
  const tool = bash ? 'run_terminal_command' : (row.tool ?? 'write');
  const rawInput = bash
    ? { variant: 'Bash', command: row.input, description: 'run', is_background: false }
    : tool === 'search_replace'
      ? { file_path: row.input, old_string: 'x', new_string: 'y' }
      : { file_path: row.input, content: 'x' };
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

/** What Auto answers on the folder, as the host builds it: `allow`, or why it still asks. */
async function verdict(row: Row): Promise<'allow' | AutoAsk> {
  const req = request(row);
  const landing = row.kind === 'edit' ? await editLanding(agentPath(row.input, place)) : null;
  const interaction = workInteraction('i1', req, [], place, landing);
  const facts = await autoFacts(req, place, landing, { cdpath: false });
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

async function run(list: readonly Row[]): Promise<Array<[string, string]>> {
  const wrong: Array<[string, string]> = [];
  for (const row of list) {
    const seen = await verdict(row);
    if ((seen === 'allow') !== (row.expect === 'allow')) wrong.push([`${row.kind} ${row.input}`, seen]);
  }
  return wrong;
}

/**
 * Plain uses of the heads review that the allow-only grammar makes ask, each with its rule (design 13 §10). Every other
 * plain use of that review still runs with no card.
 */
const PLAIN_NOW_ASK: Readonly<Record<string, string>> = {
  'find -L . -name a.txt': 'find: -L is not on the table',
  'grep -f a.txt b.txt': 'grep: -f is refused (G3)',
  'cp -r src dst': 'a recursive writer asks (G6)',
  'mkdir -m 4777 e': 'mkdir: -m is not on the table',
  'touch -r a.txt b.txt': 'touch: -r is not on the table',
  'chmod u+s a.txt': 'chmod modes are [ugoa]*[+-=][rwxX]+ (G3)',
  'chmod -R 755 src': 'a recursive writer asks (G6)',
  'chmod --reference=a.txt b.txt': 'chmod: --reference is not on the table',
  'diff --to-file=b.txt a.txt': 'diff: --to-file is not on the table',
  'stat -c %s a.txt': 'stat: -c is not on the table (GNU -c, BSD has none)',
  'file -C -m a.txt': 'file: -C compiles a magic file (writes)',
  "echo $'\\x2essh'": '$ in any form is unread (G1)',
  "'ls' -la": 'a quoted head is not known (G2)',
  'l\\s -la': 'a backslash is unread (G1)',
  'env -i ls': 'env passes only as env NAME=value... head (G3)',
  'env -u PATH ls': 'env passes only as env NAME=value... head (G3)',
  'env -- ls': 'env passes only as env NAME=value... head (G3)',
  'time -p ls': 'time is a shell control word (G1)',
  'timeout -k 1 5 ls': 'timeout takes one duration then a head (G3)',
  'nice -5 ls': 'nice takes -n N then a head (G3)',
  'nice env time timeout 5 ls': 'env needs NAME=value; time is a control word',
};

describe('the review evidence', () => {
  it('every input of the 26 findings asks', async () => {
    expect(await run(rows('evidence-findings.json'))).toEqual([]);
  });

  it('every disk-host probe input asks (the one inside edit passes)', async () => {
    expect(await run(rows('evidence-disk-host.json'))).toEqual([]);
  });

  it('every row of the third reviewer asks', async () => {
    expect(await run(rows('evidence-codex.json'))).toEqual([]);
  });

  it('every row of the heads review asks; its plain uses run, except those the grammar makes ask', async () => {
    const list = rows('evidence-heads.json').map((r) =>
      r.expect === 'allow' && Object.hasOwn(PLAIN_NOW_ASK, r.input) ? { ...r, expect: 'ask' as const } : r,
    );
    expect(list.filter((r) => r.expect === 'allow').length).toBe(51 - Object.keys(PLAIN_NOW_ASK).length);
    expect(await run(list)).toEqual([]);
  });

  it('the plain uses of the brief run with no card', async () => {
    expect(await run(rows('positive.json'))).toEqual([]);
  });

  it('every input of the second review asks; the uses it lists still run', async () => {
    expect(await run(rows('evidence-review2.json'))).toEqual([]);
  });

  it('a /dev/fd name of a regular file the app holds open asks, as an edit and as a write target', async () => {
    const fd = openSync(join(root, 'out', 'held'), 'w');
    try {
      const name = `/dev/fd/${String(fd)}`;
      for (const row of [
        { input: name, kind: 'edit' as const, expect: 'ask' as const },
        { input: `echo x > ${name}`, kind: 'bash' as const, expect: 'ask' as const },
        { input: `cp a.txt ${name}`, kind: 'bash' as const, expect: 'ask' as const },
      ]) {
        expect([row.input, await verdict(row)]).not.toEqual([row.input, 'allow']);
      }
    } finally {
      closeSync(fd);
    }
  });

  it('git that can run a hook asks while a hook is installed or a git config moves the hooks folder or includes a file', async () => {
    const commit: Row = { input: "git commit -m 'x'", kind: 'bash', expect: 'allow' };
    const repo = join(place.cwd, '.git', 'config');
    const kept = readFileSync(repo, 'utf8');
    const configs: Array<[string, string]> = [
      [join(place.home, '.gitconfig'), '[core]\n\thooksPath = .husky\n'],
      [join(place.home, '.gitconfig'), '[includeIf "gitdir:~/code/"]\n\tpath = ~/.gitconfig-work\n'],
      [join(place.home, '.config', 'git', 'config'), '[core]\n\thookspath = hooks\n'],
      [repo, `${kept}[core]\n\thooksPath = .husky/_\n`],
    ];
    expect(await verdict(commit)).toBe('allow');
    for (const [file, text] of configs) {
      mkdirSync(dirname(file), { recursive: true });
      writeFileSync(file, text);
      try {
        for (const input of [
          commit.input,
          'git push origin main',
          'git checkout -b x2',
          'git stash',
          'git fetch',
          'git status',
          'git add a.txt',
          'git diff',
          'git mv a.txt n7',
        ]) {
          expect([file, input, await verdict({ input, kind: 'bash', expect: 'ask' })]).toEqual([
            file,
            input,
            'unknown-target',
          ]);
        }
        // git log runs no hook: a global hooks setting leaves it alone, a key of the folder's config off the list asks.
        expect([file, await verdict({ input: 'git log -n 5', kind: 'bash', expect: 'allow' })]).toEqual([
          file,
          file === repo ? 'unknown-target' : 'allow',
        ]);
      } finally {
        if (file === repo) writeFileSync(repo, kept);
        else rmSync(file);
      }
    }
    expect(await verdict(commit)).toBe('allow');
    const hook = join(place.cwd, '.git', 'hooks', 'pre-commit');
    writeFileSync(`${hook}.sample`, '#!/bin/sh\n');
    expect(await verdict(commit)).toBe('allow');
    writeFileSync(hook, '#!/bin/sh\n');
    try {
      expect(await verdict(commit)).toBe('unknown-target');
    } finally {
      rmSync(hook);
      rmSync(`${hook}.sample`);
    }
    expect(await verdict(commit)).toBe('allow');
  });

  it('a read that lands outside the folder asks, and diff of a folder asks (it opens the entries through their links)', async () => {
    const ask = [
      'grep -r x k',
      'grep -rn ORIG src/lnk',
      'rg ORIG k',
      'cat k/f',
      'cp k/f n9',
      'diff d e',
      'diff -N d e',
      'diff a.txt d',
      'diff -q d/ e',
      'diff a.t* b.txt',
    ];
    for (const input of ask) {
      expect([input, await verdict({ input, kind: 'bash', expect: 'ask' })]).not.toEqual([input, 'allow']);
    }
    for (const input of ['diff a.txt b.txt', 'grep -rn word src', 'cat node_modules/react/x', 'git status']) {
      expect([input, await verdict({ input, kind: 'bash', expect: 'allow' })]).toEqual([input, 'allow']);
    }
  });

  it('a pattern may match more names than the shell would, never fewer', () => {
    const yes: Array<[string, string]> = [
      ['*.ts', 'a.ts'],
      ['*.ts', 'A.TS'],
      ['e*', 'e\u0301x'],
      ['a?', 'a\u00df'],
      ['a?', 'a'],
      ['k*', '\u212aelvin'],
      ['s*', '\u017fx'],
      ['*', '.git'],
      ['?ey*', 'keys'],
      ['a*b*c', 'aXbYbZc'],
    ];
    const no: Array<[string, string]> = [
      ['*.ts', 'a.js'],
      ['a?', 'b'],
      ['x*', 'keys'],
    ];
    for (const [pattern, name] of yes) expect([pattern, name, mayMatch(pattern, name)]).toEqual([pattern, name, true]);
    for (const [pattern, name] of no) expect([pattern, name, mayMatch(pattern, name)]).toEqual([pattern, name, false]);
  });
});

/** A plain use of each head, with `OPT` where an option goes. */
const TEMPLATES: Readonly<Record<string, string>> = {
  ls: 'ls OPT src',
  cat: 'cat OPT a.txt',
  head: 'head OPT a.txt',
  tail: 'tail OPT a.txt',
  wc: 'wc OPT a.txt',
  sort: 'sort OPT a.txt',
  cut: 'cut OPT a.txt',
  diff: 'diff OPT a.txt b.txt',
  cmp: 'cmp OPT a.txt b.txt',
  stat: 'stat OPT a.txt',
  file: 'file OPT a.txt',
  du: 'du OPT src',
  df: 'df OPT .',
  realpath: 'realpath OPT a.txt',
  pwd: 'pwd OPT',
  true: 'true OPT',
  false: 'false OPT',
  echo: 'echo OPT x',
  which: 'which OPT node',
  basename: 'basename OPT a/b',
  dirname: 'dirname OPT a/b',
  tr: 'tr OPT a b',
  mkdir: 'mkdir OPT n1',
  touch: 'touch OPT n2',
  tee: 'tee OPT n3',
  grep: 'grep OPT x a.txt',
  rg: 'rg OPT x src',
  cp: 'cp OPT a.txt n4',
  mv: 'mv OPT a.txt n5',
  chmod: 'chmod OPT u+x a.txt',
  printf: 'printf OPT x',
  sleep: 'sleep OPT 1',
  date: 'date OPT',
  test: 'test OPT a.txt',
  node: 'node OPT script.js',
  python3: 'python3 OPT script.py',
  make: 'make OPT test',
  pytest: 'pytest OPT tests',
  jest: 'jest OPT',
  eslint: 'eslint OPT src',
  oxlint: 'oxlint OPT src',
  prettier: 'prettier --check OPT src',
  tsc: 'tsc --noEmit OPT',
  vitest: 'vitest run OPT',
  npm: 'npm test OPT',
  pnpm: 'pnpm test OPT',
  yarn: 'yarn test OPT',
  bun: 'bun test OPT',
  env: 'env OPT NODE_ENV=x ls',
  timeout: 'timeout OPT 5 ls',
  nice: 'nice OPT -n 5 ls',
};

/** The tables the generated cases subtract: what passes. */
function tableOf(head: string): OptionTable {
  return headTable(head) ?? {};
}

const LETTERS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';

/** Every single letter as a short option, minus the head's table; each must ask. */
function letterCases(head: string, template: string, table: OptionTable): string[] {
  const out: string[] = [];
  for (const c of LETTERS) {
    if (table.flags?.includes(c) === true || (table.values !== undefined && Object.hasOwn(table.values, c))) continue;
    if (table.digits === true && c >= '0' && c <= '9') continue;
    if (table.long?.includes(`-${c}`) === true || (head === 'test' && TEST_UNARY.has(`-${c}`))) continue;
    out.push(template.replace('OPT', `-${c}`));
  }
  return out;
}

const HELP = JSON.parse(readFileSync(join(FIXTURES, 'help-options.json'), 'utf8')) as {
  options: Record<string, string[]>;
};

/** Long option names of a head from its help text and from the evidence, minus its table. */
function longCases(key: string, template: string, table: OptionTable): string[] {
  const allowed = new Set([...(table.long ?? []), ...Object.keys(table.longValues ?? {})]);
  const names = new Set(HELP.options[key] ?? []);
  for (const file of ['evidence-heads.json', 'evidence-findings.json', 'evidence-codex.json']) {
    for (const r of rows(file)) {
      const words = r.input.split(' ');
      if (words[0] !== key.split(' ')[0]) continue;
      for (const w of words) if (w.startsWith('--') && w.length > 2) names.add(w.split('=')[0]!);
    }
  }
  const out: string[] = [];
  for (const name of names) {
    if (allowed.has(name)) continue;
    out.push(template.replace('OPT', name), template.replace('OPT', `${name}=x`));
  }
  return out;
}

describe('options not on a table ask', () => {
  it.each(Object.entries(TEMPLATES))(
    '%s: each letter and each long name outside its table asks',
    async (head, template) => {
      const table = tableOf(head);
      const cases = [...letterCases(head, template, table), ...longCases(head, template, table)];
      const allowed: string[] = [];
      for (const input of cases) {
        if ((await verdict({ input, kind: 'bash', expect: 'ask' })) === 'allow') allowed.push(input);
      }
      expect(cases.length).toBeGreaterThan(0);
      expect(allowed).toEqual([]);
    },
  );

  const GIT_TEMPLATES: Readonly<Record<string, string>> = {
    branch: 'git branch OPT',
    switch: 'git switch OPT main2',
    checkout: 'git checkout OPT main2',
    merge: 'git merge OPT main2',
    rebase: 'git rebase OPT main2',
    fetch: 'git fetch OPT origin',
    pull: 'git pull OPT origin',
    push: 'git push OPT origin main',
    stash: 'git stash push OPT',
    tag: 'git tag OPT',
    blame: 'git blame OPT a.txt',
    mv: 'git mv OPT a.txt n6',
    commit: "git commit -m 'x' OPT",
  };

  it.each(Object.keys(GIT_RULES))('git %s: each letter and each long name outside its table asks', async (sub) => {
    const template = GIT_TEMPLATES[sub] ?? `git ${sub} OPT`;
    const table = GIT_RULES[sub]!.table;
    const cases = [...letterCases(sub, template, table), ...longCases(`git ${sub}`, template, table)];
    const allowed: string[] = [];
    for (const input of cases) {
      if ((await verdict({ input, kind: 'bash', expect: 'ask' })) === 'allow') allowed.push(input);
    }
    expect(allowed).toEqual([]);
  });

  it.each(Object.entries(TOOL_RULES).flatMap(([tool, subs]) => Object.keys(subs).map((sub) => [tool, sub] as const)))(
    '%s %s: each letter and each long name outside its table asks',
    async (tool, sub) => {
      const template = `${tool} ${sub} OPT`;
      const table = TOOL_RULES[tool]![sub]!;
      const help = tool === 'go' ? `go ${sub === 'vet' ? 'build' : sub}` : `${tool} ${sub}`;
      const cases = [...letterCases(tool, template, table), ...longCases(help, template, table)];
      if (tool === 'go') {
        for (const name of HELP.options[help] ?? []) {
          if (!(table.long ?? []).includes(name) && !Object.hasOwn(table.longValues ?? {}, name)) {
            cases.push(template.replace('OPT', name), template.replace('OPT', `${name}=x`));
          }
        }
      }
      const allowed: string[] = [];
      for (const input of cases) {
        if ((await verdict({ input, kind: 'bash', expect: 'ask' })) === 'allow') allowed.push(input);
      }
      expect(allowed).toEqual([]);
    },
  );
});
