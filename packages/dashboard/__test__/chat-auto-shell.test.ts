/**
 * The G1 differential test: the words the Auto scan reads are the words a real zsh and a real bash pass. A seeded
 * generator writes command texts from the subset alphabet plus the characters the subset leaves unread, every head
 * `echo`. Each text the scan allows runs in each shell (in a throwaway Work folder `A`, HOME there too, beside a
 * throwaway temp root `B` with a second spelling `b`) with each `echo` replaced by a `printf` that prints its arguments
 * NUL-separated; a text the scan reads as unread, or that writes outside `A` and `B`, is never run. A second
 * differential checks where the shell is after `cd` and `pushd`: every folder `pwd -P` prints is one the scan holds.
 */

import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readdirSync, realpathSync, rmSync, symlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterAll, beforeAll, describe, expect, it } from 'vite-plus/test';

import { autoScan, scanDirs, scanWords } from '../src/chat/agent/auto-scan.js';
import { autoFacts } from '../src/chat/agent/auto.js';
import type { WorkPlace } from '../src/chat/agent/command-flags.js';
import { rootText } from '../src/chat/agent/roots.js';

const SHELLS: ReadonlyArray<{ name: string; path: string; args: string[] }> = [
  { name: 'zsh', path: '/bin/zsh', args: ['-f', '-o', 'nonomatch', '-o', 'extendedglob', '-c'] },
  { name: 'bash', path: '/bin/bash', args: ['--norc', '--noprofile', '-O', 'extglob', '-c'] },
];
const SEED = 0x5eed_13b;
const TEXTS = 5_000;
const RUN_HEAD = "printf '%s\\0'";
const LOOP = 'for __t in "$@"; do builtin eval "$__t"; printf "\\001"; done';

/** mulberry32: a small seeded generator, so every run tests the same texts. */
function generator(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4_294_967_296;
  };
}

const PLAIN = 'abcxyzABZ019_.,:@+%/-';
/** A character above U+FFFF (two UTF-16 units): literal inside quotes. */
const WIDE = '\u{1F600}';
const QUOTED = 'abc xyz019_.,:@+%/-$`\\"!*?{}()[]#&|;<>~^=\t';
const DOUBLE = "abc xyz019_.,:@+%/-'*?{}()[]#&|;<>~^=\t";
const UNREAD_CHARS = [
  '$',
  '`',
  '\\',
  '(',
  ')',
  '{',
  '}',
  '[',
  ']',
  '!',
  '#',
  '&',
  '<',
  '>',
  '|',
  ';',
  '*',
  '?',
  '^',
  '~',
  '=',
  '\r',
  '\u000b',
  ' ',
  ' ',
  '​',
  'é',
  'ſ',
  '"',
  "'",
];

function pick<T>(rand: () => number, list: readonly T[] | string): T {
  return list[Math.floor(rand() * list.length)] as T;
}

function chars(rand: () => number, set: string, min: number, max: number): string {
  let out = '';
  const n = min + Math.floor(rand() * (max - min + 1));
  for (let k = 0; k < n; k++) out += pick<string>(rand, set);
  return out;
}

function piece(rand: () => number): string {
  const r = rand();
  if (r < 0.35) return chars(rand, PLAIN, 1, 6);
  if (r < 0.45) return `${chars(rand, PLAIN.replace('-', ''), 1, 4)}=${chars(rand, PLAIN, 0, 4)}`;
  if (r < 0.65) return `'${chars(rand, QUOTED, 0, 8)}${rand() < 0.15 ? WIDE : ''}'`;
  if (r < 0.8) return `"${chars(rand, DOUBLE, 0, 8)}${rand() < 0.15 ? WIDE : ''}"`;
  if (r < 0.85) return `~/${chars(rand, PLAIN, 1, 4)}`;
  if (r < 0.9) return `${chars(rand, 'abcéΩжあ', 1, 4)}`;
  if (r < 0.93) return tempPath(rand);
  const base = chars(rand, PLAIN, 1, 5);
  const at = Math.floor(rand() * (base.length + 1));
  return base.slice(0, at) + pick<string>(rand, UNREAD_CHARS) + base.slice(at);
}

/** A path under the temp root in one of its spellings: `{B}` is its real path, `{b}` a link to it. */
function tempPath(rand: () => number): string {
  const top = pick<string>(rand, ['{B}', '{b}', '/{B}']);
  const rest = chars(rand, 'abcxyz', 1, 3);
  return pick<string>(rand, [`${top}/${rest}`, `${top}//${rest}`, `${top}/${rest}/`, top]);
}

/** The text with the temp root's spellings filled in. */
function filled(text: string, b: string, alias: string): string {
  return text.split('{B}').join(b).split('{b}').join(alias);
}

function word(rand: () => number): string {
  return rand() < 0.2 ? piece(rand) + piece(rand) : piece(rand);
}

interface Text {
  scan: string;
  run: string;
  ops: string[];
}

function text(rand: () => number): Text {
  const parts = 1 + Math.floor(rand() * 3);
  const ops: string[] = [];
  let scan = '';
  let run = '';
  for (let p = 0; p < parts; p++) {
    if (p > 0) {
      const op = pick<string>(rand, [' ; ', ' && ', ' || ', '\n', ';', '&&']);
      ops.push(op.trim() === '' ? '\n' : op.trim());
      scan += op;
      run += op;
    }
    const words: string[] = [];
    const n = Math.floor(rand() * 4);
    for (let k = 0; k < n; k++) words.push(word(rand));
    const tail = rand() < 0.1 ? pick<string>(rand, [' 2>&1', ' 2>/dev/null']) : '';
    const sep = rand() < 0.1 ? '\t' : ' ';
    const args = words.length === 0 ? '' : sep + words.join(sep);
    scan += `echo${args}${tail}`;
    run += `${RUN_HEAD}${args}${tail}`;
  }
  return { scan, run, ops };
}

/**
 * The output each allowed text must give: printf runs for each part the operators reach (printf always succeeds); a
 * part that writes to a file prints nothing here.
 */
function expected(t: Text, home: string): string | null {
  const parts = scanWords(t.scan);
  if (parts === null || parts.length !== t.ops.length + 1) return null;
  let out = '';
  let skip = false;
  for (let p = 0; p < parts.length; p++) {
    const part = parts[p]!;
    if (!skip && part.writes === 0) {
      const args = part.words.slice(1).map((w) => (w.home ? home + w.value.slice(1) : w.value));
      out += args.length === 0 ? '\0' : args.map((a) => `${a}\0`).join('');
    }
    skip = t.ops[p] === '||';
  }
  return out;
}

let folder: string;
/** The Work folder, the temp root and its second spelling, all inside `folder`. */
let A: string;
let B: string;
let b: string;

beforeAll(() => {
  folder = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-auto-shell-')));
  A = join(folder, 'A');
  B = join(folder, 'B');
  b = join(folder, 'tlink');
  for (const d of [join(A, 'sub', 'deep'), join(B, 'x', 'y')]) mkdirSync(d, { recursive: true });
  symlinkSync(B, b);
  symlinkSync(join(B, 'x'), join(A, 'lnk'));
});

function twoRoots(): WorkPlace {
  return { cwd: A, home: A, temp: [{ top: B, spellings: [B, b], kind: 'temp' }], uid: process.getuid?.() };
}

/** Nothing but the two roots and the link was made beside them. */
function onlyRoots(): string[] {
  return readdirSync(folder).sort();
}

afterAll(() => {
  rmSync(folder, { recursive: true, force: true });
});

const rand = generator(SEED);
const TEXT_LIST: Text[] = [];
for (let k = 0; k < TEXTS; k++) TEXT_LIST.push(text(rand));

describe('G1 differential: the scan reads the words the shell passes', () => {
  it('the generator makes enough texts of both kinds', () => {
    const place = { cwd: '/w/app', home: '/w' };
    const allowed = TEXT_LIST.filter((t) => {
      const s = autoScan(filled(t.scan, '/w/B', '/w/b'), place, { cdpath: false });
      return !s.unread && !s.unknown && !s.protectedName && s.checks !== null;
    });
    expect(TEXT_LIST.length).toBeGreaterThanOrEqual(5_000);
    expect(allowed.length).toBeGreaterThan(1_000);
    expect(TEXT_LIST.length - allowed.length).toBeGreaterThan(1_000);
    expect(allowed.filter((t) => t.scan.includes(WIDE)).length).toBeGreaterThan(50);
  });

  for (const shell of SHELLS) {
    const present = existsSync(shell.path);
    it.skipIf(!present)(
      `${shell.name}: every allowed text passes exactly the words the scan read${present ? '' : ` (skipped: ${shell.path} is missing)`}`,
      () => {
        const place = twoRoots();
        const runs: Array<{ text: Text; want: string }> = [];
        let named = 0;
        for (const given of TEXT_LIST) {
          const t = { ...given, scan: filled(given.scan, B, b), run: filled(given.run, B, b) };
          const s = autoScan(t.scan, place, { cdpath: false });
          if (s.unread || s.unknown || s.protectedName || s.checks === null) continue;
          if (s.targets === null || s.targets.some((target) => rootText(target, place) === null)) continue;
          const want = expected(t, A);
          expect([t.scan, want === null]).toEqual([t.scan, false]);
          if (given.scan.includes('{B}') || given.scan.includes('{b}')) named += 1;
          runs.push({ text: t, want: want! });
        }
        expect(named).toBeGreaterThan(50);
        const wrong: Array<{ text: string; shell: string; want: string }> = [];
        for (let at = 0; at < runs.length; at += 200) {
          const batch = runs.slice(at, at + 200);
          const output = execFileSync(shell.path, [...shell.args, LOOP, shell.name, ...batch.map((r) => r.text.run)], {
            cwd: A,
            env: { PATH: '/usr/bin:/bin', HOME: A, LANG: 'en_US.UTF-8', LC_ALL: 'en_US.UTF-8' },
            encoding: 'utf8',
            maxBuffer: 64 * 1024 * 1024,
          });
          expect(onlyRoots()).toEqual(['A', 'B', 'tlink']);
          const seen = output.split('\u0001');
          expect(seen.length).toBe(batch.length + 1);
          batch.forEach((r, k) => {
            if (seen[k] !== r.want)
              wrong.push({ text: r.text.scan, shell: JSON.stringify(seen[k]), want: JSON.stringify(r.want) });
          });
        }
        expect(wrong).toEqual([]);
      },
    );
  }
});

/** `cd` and `pushd` operands: relative, absolute into each root by each spelling, `..`, `~/`, missing, a link. */
const CD_OPERANDS = [
  'sub',
  'sub/deep',
  'nosuch',
  'lnk',
  '..',
  'sub/..',
  '~/sub',
  '{A}/sub',
  '{B}/x',
  '{b}/x',
  '{B}/x/y',
  '{b}/x/y/',
  '{B}',
  '{B}/nosuch',
  '{b}/x/..',
];
const CD_OPS = [' && ', ' || ', ' ; ', '\n'];
const PWD_LOOP = 'for __t in "$@"; do builtin cd -P -- "$WORK"; builtin eval "$__t"; printf "\\001"; done';

/** 1 to 3 `cd` / `pushd` forms, each followed by `pwd -P`, joined by `;`, `&&`, `||` and newlines. */
function cdText(rand: () => number): string {
  let out = '';
  const n = 1 + Math.floor(rand() * 3);
  for (let k = 0; k < n; k++) {
    if (k > 0) out += pick<string>(rand, CD_OPS);
    const head = rand() < 0.3 ? 'pushd' : 'cd';
    out += `${head} ${pick<string>(rand, CD_OPERANDS)}${head === 'pushd' ? ' >/dev/null' : ''}`;
    out += `${pick<string>(rand, CD_OPS)}pwd -P`;
  }
  return out;
}

describe('where the shell is: every folder pwd -P prints is one the scan holds', () => {
  const cdRand = generator(SEED + 1);
  const CD_TEXTS: string[] = [];
  for (let k = 0; k < 600; k++) CD_TEXTS.push(cdText(cdRand));

  for (const shell of SHELLS) {
    const present = existsSync(shell.path);
    it.skipIf(!present)(`${shell.name}: after cd and pushd`, async () => {
      const place = twoRoots();
      const texts: Array<{ text: string; dirs: string[][]; pwds: number[] }> = [];
      for (const given of CD_TEXTS) {
        const text = filled(given, B, b).split('{A}').join(A);
        const s = autoScan(text, place, { cdpath: false });
        if (s.unread || s.unknown || s.checks === null) continue;
        const req = {
          sessionId: 's',
          toolCallId: 'c',
          kind: 'execute',
          title: 'x',
          tool: { name: 'run_terminal_command', namespace: 'grok_build' },
          rawInput: { command: text },
          options: [],
        };
        const facts = await autoFacts(req, place, null, { cdpath: false });
        if (!facts.placed || facts.temp !== null) continue;
        const dirs = scanDirs(text, place)!;
        const words = scanWords(text)!;
        const pwds = words.flatMap((p, k) => (p.words[0]?.value === 'pwd' ? [k] : []));
        texts.push({ text, dirs, pwds });
      }
      expect(texts.length).toBeGreaterThan(100);
      expect(texts.filter((t) => t.text.includes(B) || t.text.includes(b)).length).toBeGreaterThan(30);
      const wrong: string[] = [];
      for (let at = 0; at < texts.length; at += 100) {
        const batch = texts.slice(at, at + 100);
        const runs = batch.map((t) => {
          let k = 0;
          return t.text.split('pwd -P').reduce((acc, piece, i) => {
            if (i === 0) return piece;
            const index = t.pwds[k++]!;
            return `${acc}printf '%s:%s\\n' ${String(index)} "$(pwd -P)"${piece}`;
          }, '');
        });
        const output = execFileSync(shell.path, [...shell.args, PWD_LOOP, shell.name, ...runs], {
          cwd: A,
          env: { PATH: '/usr/bin:/bin', HOME: A, WORK: A, LANG: 'en_US.UTF-8', LC_ALL: 'en_US.UTF-8' },
          encoding: 'utf8',
          stdio: ['ignore', 'pipe', 'ignore'],
        });
        const seen = output.split('\u0001');
        expect(seen.length).toBe(batch.length + 1);
        batch.forEach((t, k) => {
          for (const line of seen[k]!.split('\n')) {
            if (line === '') continue;
            const colon = line.indexOf(':');
            const part = Number(line.slice(0, colon));
            const where = line.slice(colon + 1);
            if (!t.dirs[part]!.includes(where))
              wrong.push(`${shell.name}: ${t.text} | part ${String(part)} in ${where}`);
          }
        });
        expect(onlyRoots()).toEqual(['A', 'B', 'tlink']);
      }
      expect(wrong).toEqual([]);
    });
  }
});

describe('echo dash words in the shells', () => {
  /** What each shell prints; zsh prints no lone `-`, a difference of output only. */
  const CASES: ReadonlyArray<[string, string, string]> = [
    ['echo "---"', '---\n', '---\n'],
    ['echo --', '--\n', '--\n'],
    ['echo -x', '-x\n', '-x\n'],
    ['echo -', '\n', '-\n'],
    ['echo -n x', 'x', 'x'],
    ['echo -- -n x', '-- -n x\n', '-- -n x\n'],
  ];
  for (const shell of SHELLS) {
    const present = existsSync(shell.path);
    it.skipIf(!present)(`${shell.name}: a dash word that is not echo's option prints as text`, () => {
      const place = twoRoots();
      for (const [text, zsh, bash] of CASES) {
        const s = autoScan(text, place, { cdpath: false });
        expect([text, s.unknown, s.unread]).toEqual([text, false, false]);
        const out = execFileSync(shell.path, [...shell.args, text], {
          cwd: A,
          env: { PATH: '/usr/bin:/bin', HOME: A },
          encoding: 'utf8',
        });
        expect([text, out]).toEqual([text, shell.name === 'zsh' ? zsh : bash]);
      }
    });
  }
});
