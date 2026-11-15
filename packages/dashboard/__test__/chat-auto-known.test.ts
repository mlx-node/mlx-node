/**
 * The known list of Auto on a real folder: each head passes in its plain use, each refused option of a head asks
 * with its reason, the hostile commands ask, and write targets are placed on disk (links and hard links followed).
 */

import { linkSync, mkdirSync, mkdtempSync, realpathSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterAll, beforeAll, describe, expect, it } from 'vite-plus/test';

import type { PermissionRequest } from '../src/chat/acp/messages.js';
import { AUTO_ASSIGNMENTS, GIT_RULES, KNOWN_HEADS } from '../src/chat/agent/auto-scan.js';
import { autoFacts, decideAuto, findUserHooks } from '../src/chat/agent/auto.js';
import type { WorkPlace } from '../src/chat/agent/command-flags.js';
import { workInteraction } from '../src/chat/agent/permissions.js';
import { workGuardHookName } from '../src/chat/agent/profiles.js';
import type { AutoAsk } from '../src/chat/types.js';

const GUARD = workGuardHookName('mlx-work-guard-0123456789abcdef');
const OPTIONS = [
  { optionId: 'allow-once', name: 'once', kind: 'allow_once' },
  { optionId: 'reject-once', name: 'no', kind: 'reject_once' },
];

let root: string;
let place: WorkPlace;

beforeAll(() => {
  root = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-auto-known-')));
  const home = join(root, 'home');
  const cwd = join(home, 'code', 'app');
  mkdirSync(join(cwd, '.git'), { recursive: true });
  writeFileSync(join(cwd, '.git', 'config'), '[remote "origin"]\n\turl = https://example.com/o/app.git\n');
  mkdirSync(join(cwd, 'src'), { recursive: true });
  mkdirSync(join(cwd, 'tests'), { recursive: true });
  for (const name of ['a.txt', 'b.txt', 'a.js', 'a.ts', 'x.py', 'Makefile']) writeFileSync(join(cwd, name), 'x\n');
  const outside = join(root, 'outside');
  mkdirSync(outside);
  writeFileSync(join(outside, 'f'), 'f\n');
  symlinkSync(outside, join(cwd, 'k'));
  linkSync(join(outside, 'f'), join(cwd, 'h'));
  place = { cwd, home };
});

afterAll(() => {
  rmSync(root, { recursive: true, force: true });
});

function bashRequest(command: string, extra: Record<string, unknown> = {}): PermissionRequest {
  return {
    sessionId: 's1',
    toolCallId: 'call-1',
    kind: 'execute',
    title: 'Execute',
    tool: { name: 'run_terminal_command', namespace: 'grok_build' },
    rawInput: { variant: 'Bash', command, description: 'run', is_background: false, ...extra },
    options: OPTIONS,
    hookAsk: { hookName: GUARD, reason: null },
  };
}

/** What Auto answers for one command on the real folder: `allow`, or why it still asks. */
/** `{OUT}` in a row: an absolute folder outside the Work folder. */
async function verdict(text: string, cdpath = false): Promise<'allow' | AutoAsk> {
  const command = text.split('{OUT}').join(join(place.home, 'code', 'other'));
  const req = bashRequest(command);
  const interaction = workInteraction('i1', req, [], place, null);
  const facts = await autoFacts(req, place, null, { cdpath });
  expect(facts.flags).toEqual(interaction.flags);
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

/** One plain use of every head on the list. A head without a row here fails the first test. */
const PLAIN: Readonly<Record<string, string>> = {
  ls: 'ls -la',
  cat: 'cat a.txt',
  head: 'head -n 5 a.txt',
  tail: 'tail -n 5 a.txt',
  wc: 'wc -l a.txt',
  pwd: 'pwd',
  echo: 'echo hi',
  ps: 'ps aux',
  sed: "sed -n '1,40p' a.txt",
  printf: "printf '%s\\n' hi",
  true: 'true',
  false: 'false',
  test: 'test -f a.txt',
  '[': '[ -f a.txt ]',
  grep: 'grep -n x a.txt',
  rg: 'rg -n x src',
  find: 'find . -name "*.ts"',
  sort: 'sort -u a.txt',
  cut: 'cut -d, -f1 a.txt',
  tr: 'tr a b',
  diff: 'diff a.txt b.txt',
  cmp: 'cmp a.txt b.txt',
  stat: 'stat a.txt',
  file: 'file a.txt',
  du: 'du -sh .',
  df: 'df -h .',
  date: 'date',
  which: 'which node',
  basename: 'basename src/a.ts',
  dirname: 'dirname src/a.ts',
  realpath: 'realpath a.txt',
  sleep: 'sleep 1',
  cd: 'cd src && ls',
  pushd: 'pushd src && ls',
  mkdir: 'mkdir -p out/lib',
  touch: 'touch b.txt',
  cp: 'cp a.txt c.txt',
  mv: 'mv b.txt d.txt',
  tee: 'echo x | tee e.txt',
  chmod: 'chmod +x a.js',
  git: 'git status',
  npm: 'npm test',
  pnpm: 'pnpm test',
  yarn: 'yarn test',
  bun: 'bun test',
  node: 'node a.js',
  deno: 'deno run a.ts',
  python: 'python x.py',
  python3: 'python3 -m pytest tests',
  make: 'make test',
  cargo: 'cargo test',
  go: 'go test ./...',
  uv: 'uv sync',
  pytest: 'pytest tests',
  vitest: 'vitest run',
  jest: 'jest',
  tsc: 'tsc --noEmit',
  vp: 'vp test',
  vite: 'vite build',
  eslint: 'eslint src',
  oxlint: 'oxlint',
  prettier: 'prettier --check src',
  timeout: 'timeout 5 ls',
  nice: 'nice -n 5 ls',
  env: 'env NODE_ENV=test node a.js',
};

/** Every refused option or subcommand of a head on the list: each asks. */
const REFUSED: Array<[string, AutoAsk | 'allow']> = [
  ['rg --pre=cat x', 'not-known'],
  ['rg --pre cat x', 'not-known'],
  ['find . -exec ls {} ;', 'unread'],
  ['find . -execdir ls {} +', 'unread'],
  ['find . -ok ls {} ;', 'unread'],
  ['find . -okdir ls {} ;', 'unread'],
  ['find . -name x -delete', 'risky'],
  ['find . -fprint out', 'not-known'],
  ['find . -fprintf out %p', 'not-known'],
  ['find . -fls out', 'not-known'],
  ['sort -o out a.txt', 'not-known'],
  ['sort -uo out a.txt', 'not-known'],
  ['sort --output=out a.txt', 'not-known'],
  ['sort --compress-program=gzip a.txt', 'not-known'],
  ['cp -t out a.txt', 'not-known'],
  ['cp --target-directory=out a.txt', 'not-known'],
  ['cp -l a.txt l.txt', 'not-known'],
  ['cp -s a.txt l.txt', 'not-known'],
  ['cp --link a.txt l.txt', 'not-known'],
  ['cp --symbolic-link a.txt l.txt', 'not-known'],
  ['mv -t out a.txt', 'not-known'],
  ['mv --target-directory=out a.txt', 'not-known'],
  ['git -c core.pager=x status', 'not-known'],
  ['git -C src status', 'not-known'],
  ['git --git-dir=x status', 'not-known'],
  ['git --work-tree=x status', 'not-known'],
  ['git', 'not-known'],
  ['git rm a.txt', 'not-known'],
  ['git config user.name x', 'not-known'],
  ['git clone https://example.com/r', 'not-known'],
  ['git diff --output=out', 'not-known'],
  ['git grep -Oless x', 'not-known'],
  ['git grep --open-files-in-pager=less x', 'not-known'],
  ['git diff --ext-diff', 'not-known'],
  ['git rebase --exec=ls main', 'not-known'],
  ['git rebase -x ls main', 'not-known'],
  ['git rebase -i main', 'not-known'],
  ['git rebase -ki main', 'not-known'],
  ['git add --interactive', 'not-known'],
  ['git fetch --upload-pack=x', 'not-known'],
  ['git push --receive-pack=x', 'not-known'],
  ['python3 -m pip install x', 'not-known'],
  ['python3 -m pipx run x', 'not-known'],
  ['python3 -m ensurepip', 'not-known'],
  ['deno a.ts', 'not-known'],
  ['deno eval 1', 'not-known'],
  ['make --eval=x', 'not-known'],
  ['make -E x', 'not-known'],
  ['make SHELL=/bin/sh', 'not-known'],
  ['make -C ../other', 'not-known'],
  ['make -f ~/Makefile', 'not-known'],
  ['cargo install x', 'not-known'],
  ['cargo publish', 'not-known'],
  ['cargo login', 'not-known'],
  ['cargo --config x=1 build', 'not-known'],
  ['go install x', 'not-known'],
  ['go run example.com/x@latest', 'not-known'],
  ['go env -w X=1', 'not-known'],
  ['go build -toolexec x', 'not-known'],
  ['go test -exec x', 'not-known'],
  ['uv tool run x', 'not-known'],
  ['uv run x', 'not-known'],
  ['uv python install', 'not-known'],
  ['uv self update', 'not-known'],
  ['uv publish', 'not-known'],
  ['time -v ls', 'unread'],
  ['nice --nope ls', 'not-known'],
  ['timeout soon ls', 'not-known'],
  ['env -S "ls -l"', 'risky'],
  ['env -C {OUT} ls', 'not-known'],
  ['env -P {OUT} ls', 'not-known'],
  ['nohup ls', 'not-known'],
  ['command ls', 'unread'],
  ['builtin cd src', 'unread'],
  ['exec ls', 'unread'],
  ['xargs ls', 'not-known'],
  ['caffeinate ls', 'not-known'],
  ['stdbuf -o0 ls', 'not-known'],
  ['sudo -n true', 'risky'],
  ['vitest --outDir={OUT}/x', 'not-known'],
  ['tsc --outDir ~/x', 'not-known'],
  ['pytest -o cache_dir={OUT}/x', 'not-known'],
  ['prettier --write ../x.js', 'not-known'],
];

/** Every word of a package manager that is refused, for each package manager. */
const PACKAGE_WORDS = [
  '-g',
  '--global',
  'global',
  'dlx',
  'x',
  'exec',
  'create',
  'init',
  'config',
  'publish',
  'login',
  'logout',
  'adduser',
  'token',
  'unpublish',
  'owner',
  'access',
  'deprecate',
  'dist-tag',
  'link',
  'unlink',
  'cache',
  'upgrade',
  'self-update',
  'setup',
  'env',
  'set',
  'explore',
  'node',
  '-e',
  '--eval',
  '-p',
  '--print',
  '-C',
  '--prefix={OUT}',
  '--dir={OUT}',
  '--cwd={OUT}',
  '--userconfig=x',
  '--globalconfig=x',
  '--location=global',
  '--global-dir=x',
  '--script-shell=x',
  '--node-options=x',
];

/** The hostile commands of the spec: each asks, never runs with no card. */
const HOSTILE: Array<[string, AutoAsk | 'allow']> = [
  ['echo bHM= | base64 -d | sh', 'not-known'],
  ['X=rm; $X -rf a', 'unread'],
  ['$(printf rm) a', 'unread'],
  ['python3 -c "print(1)"', 'not-known'],
  ['perl -e "print 1"', 'not-known'],
  ['sh ./run.sh', 'not-known'],
  ['./run.sh', 'not-known'],
  ['{OUT}/ls', 'not-known'],
  ['eval ls', 'unread'],
  ['source ./env.sh', 'unread'],
  ['. ./env.sh', 'unread'],
  ['GIT_SSH_COMMAND="sh -c x" git fetch', 'unread'],
  ['git -c core.sshCommand=x fetch', 'not-known'],
  ['git -C ~/o commit -m x', 'not-known'],
  ['git fetch ../other', 'not-known'],
  ['git push ./bare.git main', 'not-known'],
  ['git push https://example.com/r main', 'not-known'],
  ['git rm -r src', 'not-known'],
  ['tar -C ~ -xf a.tar', 'unread'],
  ['unzip -d ~ a.zip', 'unread'],
  ['curl -H "x-api-key: $EXA_API_KEY" https://api.exa.ai', 'unread'],
  ['grep -rn EXA_API_KEY .', 'key'],
  ['node -p process.env', 'not-known'],
  ['node -e "1"', 'not-known'],
  ['node -r ./x.js a.js', 'not-known'],
  ['node', 'not-known'],
  ['node < a.js', 'unread'],
  ['python3 - <<EOF\nprint(1)\nEOF', 'unread'],
  ['security find-generic-password -s x', 'not-known'],
  ['firecrawl crawl https://example.com', 'not-known'],
  ['ln -s ~ k2', 'unread'],
  ['popd', 'unread'],
  ['dirs', 'not-known'],
  ['CDPATH=~ cd .ssh', 'unread'],
  ['touch {a,~/.zshrc}', 'unread'],
  ['touch a{,~/.zshrc}', 'unread'],
  ['touch {a,b}', 'unread'],
  ['touch a{1,2}', 'unread'],
  ['cat ~/.mlx-agent/config.toml', 'protected'],
  ['cat .git/config', 'protected'],
  ['cat .GIT/config', 'protected'],
  ['cat ~/.ssh/id_rsa', 'protected'],
  ['nohup sleep 5', 'not-known'],
  ['sleep 5 &', 'unread'],
  ['coproc sleep 5', 'unread'],
  ['(sleep 5 &)', 'unread'],
  ['echo $(sleep 5 &)', 'unread'],
  ['f() { ls; }; f', 'unread'],
  ['function f { ls; }', 'unread'],
  ['for f in a b; do ls $f; done', 'unread'],
  ['bash -c "ls"', 'not-known'],
  ['echo $(rm -rf x)', 'unread'],
  ['cat `which ls`', 'unread'],
  ['`which ls`', 'unread'],
  ['ls $(echo src)', 'unread'],
  ['ls $(rm -rf src)', 'unread'],
  ['ls $(sh -c x)', 'unread'],
  ['echo x > ~/.zshrc', 'risky'],
  ['echo x > ../y', 'risky'],
  ['echo x > $OUT', 'unread'],
  ['echo x > `pwd`/y', 'unread'],
  ['echo x > a*', 'unread'],
  ['echo x > a~b', 'unread'],
  ['npx cowsay', 'risky'],
  ['PATH={OUT} ls', 'unread'],
  ['NODE_OPTIONS=--require=x node a.js', 'unread'],
  ['cd ~ && ls', 'unread'],
  ['cd {OUT} && ls', 'not-known'],
  ['cd - && touch a', 'not-known'],
  ['cd src || true; touch ../../x', 'risky'],
  ['(cd src; touch a)', 'unread'],
  ['echo $(cd src) && touch a', 'unread'],
  ['while true; do cd src; done', 'unread'],
  ['pushd', 'not-known'],
  ['pushd +1', 'not-known'],
  ['git status', 'allow'],
];

describe('ps and sed: closed forms', () => {
  it('ps: the forms of its table run; every other letter, an e form, another key or a quoted word asks', async () => {
    for (const command of [
      'ps',
      'ps aux',
      'ps ax',
      'ps auxww',
      'ps -A',
      'ps -ax',
      'ps -Af',
      'ps -p 1,22',
      'ps -o pid,ppid,command',
      'ps aux | grep x',
      'echo x | ps -p 1 -o pid,args',
    ]) {
      expect([command, await verdict(command)]).toEqual([command, 'allow']);
    }
    const asks: string[] = [];
    for (const c of 'abcdefghijklmnopqrstvyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') {
      if (!'auxw'.includes(c)) asks.push(`ps a${c}`);
      if (!'Aaxfw'.includes(c)) asks.push(`ps -a${c}`);
    }
    asks.push(
      'ps -o pid,environ',
      'ps -o pid,',
      'ps -p 1-2',
      'ps -p',
      "ps 'aux'",
      'ps auxwwxa',
      'ps aux x',
      'ps -- aux',
    );
    for (const command of asks) expect([command, await verdict(command)]).not.toEqual([command, 'allow']);
  });

  it('sed: only -n with one print script; files are reads', async () => {
    for (const command of ["sed -n '1,40p' a.txt", "sed -n '$p' a.txt", 'sed -n 3p', "sed -n '2,$p' a.txt b.txt"]) {
      expect([command, await verdict(command)]).toEqual([command, 'allow']);
    }
    for (const command of [
      'sed -n 1p k/f',
      'sed -n 1p ../x',
      'sed -n p a.txt',
      'sed -n 1d a.txt',
      'sed -n 1,p a.txt',
      'sed -n 1234567890p a.txt',
      'sed -n 1p -i a.txt',
      'sed -n 1p --in-place a.txt',
      'sed -E -n 1p a.txt',
      "sed -n '1p' -- a.txt",
      'sed -n',
      'sed',
    ]) {
      expect([command, await verdict(command)]).not.toEqual([command, 'allow']);
    }
  });
});

describe('the known list', () => {
  it('every head on the list has a plain use that passes', async () => {
    for (const head of KNOWN_HEADS) {
      const command = PLAIN[head];
      expect([head, command === undefined ? 'no row' : await verdict(command)]).toEqual([head, 'allow']);
    }
  });

  it('every git subcommand on the list passes', async () => {
    const uses: Readonly<Record<string, string>> = {
      add: 'git add a.txt',
      commit: 'git commit -m fix',
      switch: 'git switch main2',
      checkout: 'git checkout main2',
      merge: 'git merge main2',
      rebase: 'git rebase main2',
      fetch: 'git fetch origin',
      pull: 'git pull origin main',
      push: 'git push origin main',
      blame: 'git blame a.txt',
      'rev-parse': 'git rev-parse --show-toplevel',
      grep: 'git grep x',
      mv: 'git mv a.txt f.txt',
    };
    for (const sub of Object.keys(GIT_RULES)) {
      const command = uses[sub] ?? `git ${sub}`;
      expect([command, await verdict(command)]).toEqual([command, 'allow']);
    }
  });

  it('git checkout of a name that is a file asks: it would discard the file’s changes', async () => {
    expect(await verdict('git checkout a.txt')).toBe('unknown-target');
  });

  it('every assignment on the list passes; any other name is unread', async () => {
    for (const name of AUTO_ASSIGNMENTS) expect([name, await verdict(`${name}=1 ls`)]).toEqual([name, 'allow']);
    expect(await verdict('NODE_ENV_X=1 ls')).toBe('unread');
  });

  it.each(REFUSED)('refused: %s', async (command, why) => {
    expect(await verdict(command)).toBe(why);
  });

  it('every refused word of a package manager asks, for each package manager', async () => {
    for (const tool of ['npm', 'pnpm', 'yarn', 'bun', 'vp']) {
      for (const word of PACKAGE_WORDS) {
        const command = `${tool} ${word} x`;
        expect([command, await verdict(command)]).not.toEqual([command, 'allow']);
      }
    }
  });

  it.each(HOSTILE)('hostile: %s', async (command, why) => {
    expect(await verdict(command)).toBe(why);
  });

  it('a relative cd asks while the Work environment sets CDPATH; an absolute cd always asks', async () => {
    expect(await verdict('cd src && ls', true)).toBe('not-known');
    expect(await verdict('cd src && ls', false)).toBe('allow');
    expect(await verdict(`cd ${place.cwd}/src && ls`, false)).toBe('not-known');
  });
});

describe('write targets on disk', () => {
  it('a target through a link that leads outside asks', async () => {
    expect(await verdict('echo x >> k/a')).toBe('unknown-target');
    expect(await verdict('touch k/b')).toBe('unknown-target');
    expect(await verdict('cp a.txt k/')).toBe('unknown-target');
    expect(await verdict('cd k && ls')).toBe('unknown-target');
    expect(await verdict('mkdir -p k/../x')).toBe('unknown-target');
    expect(await verdict('node k/f.js')).toBe('unknown-target');
  });

  it('a file with a second hard link asks; a plain file inside passes', async () => {
    expect(await verdict('echo x > h')).toBe('unknown-target');
    expect(await verdict('cd nosuch; echo x > h')).toBe('unknown-target');
    expect(await verdict('echo x > a.txt')).toBe('allow');
    expect(await verdict('echo x > new/deep/file.txt')).toBe('allow');
  });

  it('a target whose landing has a protected segment asks', async () => {
    mkdirSync(join(place.cwd, 'hooks-link-target'), { recursive: true });
    symlinkSync(join(place.cwd, '.git'), join(place.cwd, 'g'));
    try {
      expect(await verdict('echo x > g/hooks/pre-commit')).toBe('protected');
    } finally {
      rmSync(join(place.cwd, 'g'));
    }
  });

  it('git asks when the folder holds no repository', async () => {
    const sub = { cwd: join(place.cwd, 'src'), home: place.home };
    const req = bashRequest('git status');
    const facts = await autoFacts(req, sub, null, { cdpath: false });
    expect(facts.repository).toBe(false);
  });
});

describe('the user’s hooks', () => {
  it('a tool hook in a settings file is found; no file, or one with no tool hook, is not', async () => {
    const home = join(root, 'hook-home');
    const cwd = join(root, 'hook-folder');
    const agentHome = join(root, 'hook-agent');
    mkdirSync(join(home, '.claude'), { recursive: true });
    mkdirSync(cwd, { recursive: true });
    mkdirSync(join(agentHome, 'hooks'), { recursive: true });
    const at = { home, cwd, agentHome };
    expect(await findUserHooks(at)).toBe(false);
    writeFileSync(join(home, '.claude', 'settings.json'), JSON.stringify({ hooks: { Stop: [{ hooks: [] }] } }));
    expect(await findUserHooks(at)).toBe(false);
    writeFileSync(
      join(home, '.claude', 'settings.json'),
      JSON.stringify({ hooks: { PreToolUse: [{ matcher: '*', hooks: [{ type: 'command', command: 'x' }] }] } }),
    );
    expect(await findUserHooks(at)).toBe(true);
    writeFileSync(join(home, '.claude', 'settings.json'), '{ not json');
    expect(await findUserHooks(at)).toBe(true);
    rmSync(join(home, '.claude', 'settings.json'));
    writeFileSync(join(agentHome, 'hooks', 'a.json'), '{}');
    expect(await findUserHooks(at)).toBe(true);
    rmSync(join(agentHome, 'hooks', 'a.json'));
    writeFileSync(join(agentHome, 'config.toml'), '[hooks]\nPreToolUse = []\n');
    expect(await findUserHooks(at)).toBe(true);
    rmSync(join(agentHome, 'config.toml'));
    mkdirSync(join(cwd, '.grok', 'hooks'), { recursive: true });
    writeFileSync(join(cwd, '.grok', 'hooks', 'h.json'), '{}');
    expect(await findUserHooks(at)).toBe(true);
  });
});
