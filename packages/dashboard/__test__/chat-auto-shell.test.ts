/**
 * The G1 differential test: the words the Auto scan reads are the words a real zsh and a real bash pass. A seeded
 * generator writes command texts from the subset alphabet plus the characters the subset leaves unread, every head
 * `echo`. Each text the scan allows runs in each shell (in an empty throwaway folder, HOME there too) with each `echo`
 * replaced by a `printf` that prints its arguments NUL-separated; a text the scan reads as unread, or that writes outside
 * the folder, is never run.
 */

import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, realpathSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterAll, beforeAll, describe, expect, it } from 'vite-plus/test';

import { autoScan, scanWords } from '../src/chat/agent/auto-scan.js';

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
  const base = chars(rand, PLAIN, 1, 5);
  const at = Math.floor(rand() * (base.length + 1));
  return base.slice(0, at) + pick<string>(rand, UNREAD_CHARS) + base.slice(at);
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

beforeAll(() => {
  folder = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-auto-shell-')));
});

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
      const s = autoScan(t.scan, place, { cdpath: false });
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
        const place = { cwd: folder, home: folder };
        const runs: Array<{ text: Text; want: string }> = [];
        for (const t of TEXT_LIST) {
          const s = autoScan(t.scan, place, { cdpath: false });
          if (s.unread || s.unknown || s.protectedName || s.checks === null) continue;
          if (s.targets === null || s.targets.some((t) => !t.startsWith(`${folder}/`))) continue;
          const want = expected(t, folder);
          expect([t.scan, want === null]).toEqual([t.scan, false]);
          runs.push({ text: t, want: want! });
        }
        const wrong: Array<{ text: string; shell: string; want: string }> = [];
        for (let at = 0; at < runs.length; at += 200) {
          const batch = runs.slice(at, at + 200);
          const output = execFileSync(shell.path, [...shell.args, LOOP, shell.name, ...batch.map((r) => r.text.run)], {
            cwd: folder,
            env: { PATH: '/usr/bin:/bin', HOME: folder, LANG: 'en_US.UTF-8', LC_ALL: 'en_US.UTF-8' },
            encoding: 'utf8',
            maxBuffer: 64 * 1024 * 1024,
          });
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
