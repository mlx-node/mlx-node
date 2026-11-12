/**
 * The allow-only grammar of the Auto scan, rule by rule (design 13 §10): the text subset (G1), heads (G2), names
 * folded as the disk compares them (G5), the disk rules (G6), cost and failure (G8), and the start folder (G9). Only
 * text goes to the scan; inside a construct a harmless stand-in is used, since the decision does not depend on it.
 */

import { linkSync, mkdirSync, mkdtempSync, realpathSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterAll, beforeAll, describe, expect, it } from 'vite-plus/test';

import type { PermissionRequest } from '../src/chat/acp/messages.js';
import { autoScan, foldName, namesProtected, SCAN_STEP_BUDGET } from '../src/chat/agent/auto-scan.js';
import { autoFacts, decideAuto } from '../src/chat/agent/auto.js';
import { commandCheck, type WorkPlace } from '../src/chat/agent/command-flags.js';
import { workInteraction } from '../src/chat/agent/permissions.js';
import { workGuardHookName } from '../src/chat/agent/profiles.js';
import type { AutoAsk } from '../src/chat/types.js';

const GUARD = workGuardHookName('mlx-work-guard-0123456789abcdef');
const OPTIONS = [
  { optionId: 'allow-once', name: 'once', kind: 'allow_once' },
  { optionId: 'reject-once', name: 'no', kind: 'reject_once' },
];
const TEXT_PLACE: WorkPlace = { cwd: '/w/app', home: '/w' };

let root: string;
let place: WorkPlace;

beforeAll(() => {
  root = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-auto-grammar-')));
  const home = join(root, 'home');
  const cwd = join(home, 'app');
  const out = join(root, 'out');
  for (const d of ['.git', 'sub/deep', 'd', 'src']) mkdirSync(join(cwd, d), { recursive: true });
  mkdirSync(out, { recursive: true });
  for (const f of ['a.txt', 'b.txt', 'a.js', 'Makefile', 'package.json']) writeFileSync(join(cwd, f), 'x\n');
  writeFileSync(join(out, 'f'), 'ORIG\n');
  symlinkSync(out, join(cwd, 'k'));
  symlinkSync('sub', join(cwd, 'lsub'));
  symlinkSync(join(out, 'f'), join(cwd, 'd', 'a.txt'));
  linkSync(join(out, 'f'), join(cwd, 'hard'));
  place = { cwd, home };
});

afterAll(() => {
  rmSync(root, { recursive: true, force: true });
});

function bashRequest(command: string): PermissionRequest {
  return {
    sessionId: 's1',
    toolCallId: 'call-1',
    kind: 'execute',
    title: 'Execute',
    tool: { name: 'run_terminal_command', namespace: 'grok_build' },
    rawInput: { variant: 'Bash', command, description: 'run', is_background: false },
    options: OPTIONS,
    hookAsk: { hookName: GUARD, reason: null },
  };
}

async function verdict(command: string): Promise<'allow' | AutoAsk> {
  const req = bashRequest(command);
  const interaction = workInteraction('i1', req, [], place, null);
  const facts = await autoFacts(req, place, null, { cdpath: false });
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

function unread(command: string): boolean {
  return autoScan(command, TEXT_PLACE, { cdpath: false }).unread;
}

/** Constructs outside the subset: each is unread, with a harmless stand-in inside. */
const UNREAD_ROWS: Array<[string, string]> = [
  ['$ variable', 'echo $x'],
  ['$ braces', 'echo ${x}'],
  ['$ substitution', 'echo $(true)'],
  ['$ arithmetic', 'echo $((1))'],
  ['$ ANSI-C quote', "echo $'x'"],
  ['$ locale quote', 'echo $"x"'],
  ['backquote', 'echo `true`'],
  ['backslash', 'echo x\\ y'],
  ['backslash newline', 'echo x\\\ny'],
  ['input redirect', 'cat < a.txt'],
  ['here-document', 'cat <<EOF\nx\nEOF'],
  ['here-string', 'cat <<< x'],
  ['process substitution in', 'cat <(true)'],
  ['process substitution out', 'echo x > >(true)'],
  ['subshell', '(true)'],
  ['group', '{ true; }'],
  ['brace expansion', 'echo a{b,c}'],
  ['bracket glob', 'ls [ab].txt'],
  ['test brackets', '[[ -f a ]]'],
  ['bang', '! true'],
  ['bang in a word', 'echo a!b'],
  ['comment', 'echo x # y'],
  ['hash in a word', 'echo a#b'],
  ['background', 'true &'],
  ['pipe stderr', 'true |& cat'],
  ['and-redirect', 'true &> x'],
  ['dup redirect to a file', 'true >& x'],
  ['clobber', 'true >| x'],
  ['read-write', 'true <> x'],
  ['fd before a redirect', 'true 3> x'],
  ['fd 1 before a redirect', 'true 1> x'],
  ['named fd', 'true {fd}> x'],
  ['digit glued to a redirect', 'echo a2> x'],
  ['2> to a file', 'true 2> x'],
  ['2>&2', 'true 2>&2'],
  ['double semicolon', 'true ;; true'],
  ['empty before ;', '; true'],
  ['dangling &&', 'true &&'],
  ['redirect only', '> x'],
  ['CR', 'true\rtrue'],
  ['VT', 'true\u000btrue'],
  ['FF', 'true\ftrue'],
  ['NUL', 'true\u0000'],
  ['ESC', 'echo \u001b[0m'],
  ['DEL', 'echo \u007f'],
  ['NEL', 'echo a\u0085b'],
  ['line separator', 'echo a b'],
  ['paragraph separator', 'echo a b'],
  ['zero-width space', 'echo a​b'],
  ['zero-width joiner in quotes', "echo 'a‍b'"],
  ['bidi override', "echo '‮x'"],
  ['no-break space', 'echo x'],
  ['ideographic space', 'echo　x'],
  ['zsh =name', 'echo =ls'],
  ['equals after equals', 'echo a==b'],
  ['equals after colon', 'echo a:=b'],
  ['tilde user', 'echo ~root'],
  ['tilde inside', 'echo a~b'],
  ['caret', 'echo a^b'],
  ['rc_quotes pair', "echo 'a''b'"],
  ['unterminated single quote', "echo 'a"],
  ['unterminated double quote', 'echo "a'],
  ['$ in double quotes', 'echo "a$b"'],
  ['backquote in double quotes', 'echo "a`true`"'],
  ['backslash in double quotes', 'echo "a\\b"'],
  ['bang in double quotes', 'echo "a!b"'],
  ['glob in a write target', 'echo x > a*'],
  ['glob in an echo word', 'echo a*'],
  ['glob in an option value', 'find . -name *.ts'],
];

const CONTROL_HEADS = [
  'if',
  'then',
  'else',
  'elif',
  'fi',
  'for',
  'while',
  'until',
  'do',
  'done',
  'case',
  'esac',
  'select',
  'function',
  'time',
  'coproc',
  'in',
  'exec',
  'eval',
  'source',
  '.',
  'command',
  'builtin',
  'trap',
  'set',
  'unset',
  'export',
  'readonly',
  'declare',
  'typeset',
  'local',
  'let',
  'alias',
  'unalias',
  'shopt',
  'setopt',
  'enable',
  'hash',
  'read',
  'popd',
];

describe('G1: the text subset', () => {
  it.each(UNREAD_ROWS)('%s is unread', (_name, command) => {
    expect(unread(command)).toBe(true);
  });

  it.each(CONTROL_HEADS)('the control word %s as a head is unread', (head) => {
    expect(unread(`${head} true`)).toBe(true);
  });

  it('the subset reads: operators, the exact redirect forms, quotes, a leading ~/, letters of other scripts', () => {
    for (const command of [
      'true && true || true; true',
      'true\ntrue',
      'true | cat',
      'true 2>&1',
      'true >/dev/null',
      'true > /dev/null',
      'true 2>/dev/null',
      "echo 'a $b `c` \\d !e (f) {g} [h] *i'",
      'echo "a b"',
      'echo ~/x',
      'echo a=b c,d e:f g@h i+j k%l',
      'echo café Ωμέγα привет 日本語 한국어',
      'ls src/*.ts',
    ]) {
      expect([command, unread(command)]).toEqual([command, false]);
    }
  });

  it('a pattern only in a reader operand that starts with a literal, never a dash or a dot part', () => {
    expect(autoScan('ls src/*.ts', TEXT_PLACE, { cdpath: false }).unknown).toBe(false);
    for (const command of ['ls *.ts', 'ls -*', 'ls src/.*', 'ls ../*', 'ls /etc/*', 'cat ~/a*']) {
      const scan = autoScan(command, TEXT_PLACE, { cdpath: false });
      expect([command, scan.unknown || scan.unread]).toEqual([command, true]);
    }
  });
});

describe('G2: heads', () => {
  it('a head is a bare name on the list: a path, a quote, another case, an unknown name ask', async () => {
    for (const command of ['/bin/ls', './ls', "'ls'", '"ls"', 'LS', 'Ls', 'nosuchtool', 'NODE_ENV=x']) {
      expect([command, await verdict(command)]).not.toEqual([command, 'allow']);
    }
    expect(await verdict('ls')).toBe('allow');
  });

  it('assignments: only the listed names with a plain value', async () => {
    expect(await verdict('NODE_ENV=test ls')).toBe('allow');
    for (const command of ['PATH=/tmp ls', "NODE_ENV='x' ls", 'NODE_ENV=a* ls', 'CDPATH=x ls', 'X=1 ls']) {
      expect([command, await verdict(command)]).toEqual([command, 'unread']);
    }
  });
});

describe('G5: names folded as the disk compares them', () => {
  it('folds case, long s, Kelvin, normal form and ignorable characters', () => {
    expect(foldName('.MCP.JSON')).toBe('.mcp.json');
    expect(foldName('.mcp.jſon')).toBe('.mcp.json');
    expect(foldName('.groK')).toBe('.grok');
    expect(foldName('.g​it')).toBe('.git');
    expect(foldName('.git')).toBe('.git');
  });

  it('a protected name in any word asks; a dot name with a non-ASCII letter asks', () => {
    for (const word of ['.git/config', 'x/.SSH/id', '.mcp.jſon', '--exclude-dir=.git', '.gіt', '.zshrc']) {
      expect([word, namesProtected(word)]).toEqual([word, true]);
    }
    for (const word of ['src/a.ts', '.gitignore', 'git', 'café/x', '.git.']) {
      expect([word, namesProtected(word)]).toEqual([word, false]);
    }
  });

  it('a reader operand that lands on a protected folder through a link asks', async () => {
    symlinkSync(join(place.cwd, '.git'), join(place.cwd, 'g'));
    try {
      expect(await verdict('cat g/config')).toBe('protected');
      expect(await verdict('cat a.txt')).toBe('allow');
    } finally {
      rmSync(join(place.cwd, 'g'));
    }
  });
});

describe('G6: the disk rules', () => {
  it('cd: one relative operand, every part a real folder inside', async () => {
    expect(await verdict('cd sub/deep && touch x')).toBe('allow');
    expect(await verdict('cd lsub && touch x')).toBe('unknown-target');
    expect(await verdict('cd k && touch x')).toBe('unknown-target');
    expect(await verdict('cd nosuch && touch x')).toBe('unknown-target');
    for (const command of [
      'cd',
      'cd ..',
      'cd -',
      'cd +1',
      'cd /tmp',
      'cd ~/app',
      'cd sub deep',
      'cd -P sub',
      'true | cd sub',
    ]) {
      expect([command, await verdict(command)]).toEqual([command, 'not-known']);
    }
  });

  it('cp and mv into an existing folder write <folder>/<source name>', async () => {
    expect(await verdict('cp a.txt d')).toBe('unknown-target');
    expect(await verdict('cp a.txt d/')).toBe('unknown-target');
    expect(await verdict('mv a.txt d')).toBe('unknown-target');
    expect(await verdict('cp b.txt d')).toBe('allow');
  });

  it('a later path under a path an earlier part copies or moves cannot be placed', async () => {
    expect(await verdict('mv k k2 && echo x > k2/f')).toBe('unknown-target');
    expect(await verdict('cp a.txt n && echo x > n')).toBe('unknown-target');
    expect(await verdict('mv src s2; cd s2')).toBe('unknown-target');
    expect(await verdict('cp a.txt n; echo x > m')).toBe('allow');
  });

  it('after a project tool or a git command that changes files, no later write is placed', async () => {
    expect(await verdict('npm test && echo x > out.txt')).toBe('unknown-target');
    expect(await verdict('git checkout main2 && touch x')).toBe('unknown-target');
    expect(await verdict('echo x > out.txt && npm test')).toBe('allow');
  });

  it('a recursive writer, a device and a second hard link ask', async () => {
    expect(await verdict('cp -R src s3')).toBe('not-known');
    expect(await verdict('chmod -R u+w src')).toBe('not-known');
    expect(await verdict('echo x >> /dev/null')).toBe('unknown-target');
    expect(await verdict('tee /dev/null')).toBe('unknown-target');
    expect(await verdict('echo x > hard')).toBe('unknown-target');
  });

  it('project tools: only first in a pipeline, with an explicit script; -, /dev/stdin and /dev/fd ask', async () => {
    expect(await verdict('node a.js | head -n 5')).toBe('allow');
    for (const command of [
      'echo x | node a.js',
      'node -',
      'node /dev/stdin',
      'node /dev/fd/0',
      'cat -',
      'make -f -',
      'python3 -',
      'node',
    ]) {
      expect([command, await verdict(command)]).not.toEqual([command, 'allow']);
    }
  });
});

describe('G8: cost and failure', () => {
  const SHAPES: Array<[string, (n: number) => string]> = [
    ['a list of writes', (n) => 'touch a.txt;'.repeat(n)],
    ['a list of cds', (n) => 'cd src && '.repeat(n) + 'ls'],
    ['one pipeline', (n) => 'cat a.txt|'.repeat(n) + 'wc'],
    ['long quoted words', (n) => `echo '${'x'.repeat(n)}' "${'y'.repeat(n)}"`],
    ['many options', (n) => `ls ${'-l '.repeat(n)}`],
    ['git paths', (n) => `git add ${'src/a.ts '.repeat(n)}`],
  ];

  it.each(SHAPES)('%s: four times the text takes about four times the steps', (_name, make) => {
    const small = autoScan(make(250), TEXT_PLACE, { cdpath: false });
    const large = autoScan(make(1_000), TEXT_PLACE, { cdpath: false });
    expect(small.steps).toBeGreaterThan(0);
    expect(large.steps).toBeLessThan(SCAN_STEP_BUDGET);
    expect(large.steps / small.steps).toBeLessThan(4.5);
  });

  it('wrappers: at most three in front of a head; forty thousand ask in steps that grow with the text', () => {
    const three = autoScan('timeout 1 nice -n 1 env CI=1 ls', TEXT_PLACE, { cdpath: false });
    expect([three.unread, three.unknown]).toEqual([false, false]);
    for (const wrapper of ['timeout 1 ', 'nice -n 1 ', 'env CI=1 ']) {
      const four = autoScan(`${wrapper.repeat(4)}ls`, TEXT_PLACE, { cdpath: false });
      expect([wrapper, four.unknown]).toEqual([wrapper, true]);
      const many = autoScan(`${wrapper.repeat(40_000)}ls`, TEXT_PLACE, { cdpath: false });
      expect([wrapper, many.unread, many.unknown]).toEqual([wrapper, false, true]);
      expect(many.steps).toBeLessThan(wrapper.length * 40_000 * 4);
    }
  });

  it('the card check counts the folder it tracks through cds; past its budget the command is unchecked', () => {
    const small = commandCheck(`${'cd sub && '.repeat(250)}ls`, TEXT_PLACE);
    const large = commandCheck(`${'cd sub && '.repeat(20_000)}ls`, TEXT_PLACE);
    expect(small.flags).not.toContain('unchecked');
    expect(large.flags).toContain('unchecked');
    expect(large.steps).toBeLessThan(2_000_000);
  });

  it('past the step budget the command is unread', () => {
    expect(autoScan('ls -la src', TEXT_PLACE, { cdpath: false }, 5).unread).toBe(true);
    expect(autoScan('ls -la src', TEXT_PLACE, { cdpath: false }).unread).toBe(false);
    const huge = `echo ${'x '.repeat(50_000)}`;
    const scan = autoScan(huge, TEXT_PLACE, { cdpath: false });
    expect(scan.steps).toBeLessThanOrEqual(SCAN_STEP_BUDGET);
  });

  it('a throw inside the scan is unread; a throw inside the facts asks', async () => {
    const broken = new Proxy(
      {},
      {
        get() {
          throw new Error('broken place');
        },
      },
    ) as WorkPlace;
    expect(autoScan('ls', broken, { cdpath: false }).unread).toBe(true);
    const req = bashRequest('ls');
    const facts = await autoFacts(req, broken, null, { cdpath: false });
    expect(facts.scan === null || facts.scan.unread).toBe(true);
    const decision = decideAuto({
      auto: true,
      watched: true,
      request: req,
      interaction: workInteraction('i1', req, [], place, null),
      guardHook: GUARD,
      elsewhere: false,
      facts,
      allowed: 0,
    });
    expect(decision).toEqual({ allow: false, why: 'unread' });
  });
});

describe('G9: each call starts in the Work folder', () => {
  it('a relative path is placed against the Work folder, whatever an earlier call did', async () => {
    expect(autoScan('cd sub', place, { cdpath: false }).checks).toEqual([
      { kind: 'cd', base: place.cwd, parts: ['sub'], part: 0 },
    ]);
    expect(autoScan('touch x', place, { cdpath: false }).checks).toEqual([
      { kind: 'write', path: `${place.cwd}/x`, part: 0 },
    ]);
  });

  it('with CDPATH set in the Work environment, a relative cd is not known', () => {
    for (const command of ['cd sub && touch x', 'pushd sub && ls', 'cd sub']) {
      expect([command, autoScan(command, place, { cdpath: true }).unknown]).toEqual([command, true]);
      expect([command, autoScan(command, place, { cdpath: false }).unknown]).toEqual([command, false]);
    }
  });
});
