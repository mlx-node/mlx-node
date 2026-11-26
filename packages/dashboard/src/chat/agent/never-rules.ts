/**
 * Work: the user's "never allow" rules, kept by the app per Work folder in `never-allow.json` (mode 0600):
 * `{version: 1, folders: {<folder>: [{id, words, at}]}}`. A shell command is denied by the app, in every mode and for
 * every agent of the folder's conversations, when one of its simple commands starts with a rule's words. The agent is
 * told with a fixed note and goes on. Not a sandbox: the app reads the command as written.
 */

import { randomUUID } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { posix, win32 } from 'node:path';

import { writeFileAtomic } from '../paths.js';
import { redactText } from '../redact.js';
import { commandParts, foldName, PACKAGE_SUBCOMMANDS, TOOL_RULES, type Word } from './auto-scan.js';
import { visibleText } from './visible-text.js';

export interface NeverRule {
  /** `n_` and 32 hex digits. */
  id: string;
  words: string[];
  at: number;
}

/** The two scopes a card offers for a command: the program (and its subcommand), and the whole first command. */
export interface NeverOffer {
  short: string[];
  exact: string[];
}

export const NEVER_RULES_MAX = 50;
export const NEVER_WORDS_MAX = 16;
export const NEVER_CHARS_MAX = 200;
/** Rule hits one answer may take with the note; the next is a plain deny, and the answer ends `never-loop`. */
export const NEVER_PER_TURN_MAX = 3;

export const NEVER_NOTE_HEAD = 'I never allow commands that start with `';
export const NEVER_NOTE_TAIL =
  '` in this folder. Do not run it, and do not get the same result another way. Go on with the rest of the task, or stop and say what is blocked.';

/** The note the agent gets for a call a rule denies: it reads it as the user's message and goes on. */
export function neverNote(words: string): string {
  return `${NEVER_NOTE_HEAD}${words}${NEVER_NOTE_TAIL}`;
}

/** The words of a rule as one line. */
export function ruleText(words: readonly string[]): string {
  return words.join(' ');
}

/** Commands that only set up the shell: the first real command comes after them, as the agent's own "never" picks. */
const SETUP = new Set(['cd', 'pushd', 'popd', 'export', 'unset', 'set', 'sleep']);
/** Programs whose second word names what they do. */
const SUBCOMMAND_TOOLS = new Set(['git', ...Object.keys(PACKAGE_SUBCOMMANDS), ...Object.keys(TOOL_RULES)]);
/** Wrappers that run the rest of the words as a command, when written with no option. */
const PLAIN_WRAPPERS = new Set(['command', 'builtin', 'exec', 'nohup', 'time', 'nice']);
const WRAPPERS_MAX = 8;

function isIdentChar(c: string, first: boolean): boolean {
  const code = c.charCodeAt(0);
  const letter = (code >= 0x41 && code <= 0x5a) || (code >= 0x61 && code <= 0x7a) || c === '_';
  return first ? letter : letter || (code >= 0x30 && code <= 0x39);
}

/** `NAME=value` with the name outside quotes. */
function isAssignment(word: Word): boolean {
  const eq = word.value.indexOf('=');
  if (eq <= 0 || (word.quoteAt !== -1 && word.quoteAt <= eq)) return false;
  for (let k = 0; k < eq; k++) if (!isIdentChar(word.value.charAt(k), k === 0)) return false;
  return true;
}

function isDuration(v: string): boolean {
  let k = 0;
  let digits = 0;
  while (k < v.length && v.charCodeAt(k) >= 0x30 && v.charCodeAt(k) <= 0x39) {
    k++;
    digits++;
  }
  if (k < v.length && v.charAt(k) === '.') {
    k++;
    while (k < v.length && v.charCodeAt(k) >= 0x30 && v.charCodeAt(k) <= 0x39) {
      k++;
      digits++;
    }
  }
  if (digits === 0) return false;
  if (k < v.length && 'smhd'.includes(v.charAt(k))) k++;
  return k === v.length;
}

/** The words after the leading assignments. */
function withoutAssignments(words: readonly Word[]): Word[] {
  let k = 0;
  while (k < words.length && isAssignment(words[k]!)) k++;
  return words.slice(k);
}

/** The command a wrapper written with no option runs: `env A=1`, `timeout 30`, `nohup`, `command` and the like. */
function peeled(words: readonly Word[]): Word[] {
  let rest = [...words];
  for (let round = 0; round < WRAPPERS_MAX && rest.length > 1; round++) {
    const head = rest[0]!.value;
    const next = rest[1]!.value;
    if (head === 'env') {
      const inner = withoutAssignments(rest.slice(1));
      if (inner.length === 0 || inner[0]!.value.startsWith('-')) break;
      rest = inner;
    } else if (head === 'timeout') {
      if (!isDuration(next) || rest.length < 3) break;
      rest = rest.slice(2);
    } else if (PLAIN_WRAPPERS.has(head)) {
      if (next.startsWith('-')) break;
      rest = rest.slice(1);
    } else break;
  }
  return rest;
}

/** The last part of a program written as a path. */
function programName(word: string): string {
  const cut = word.lastIndexOf('/');
  return cut === -1 ? word : word.slice(cut + 1);
}

/** A scope the card may show and the file may keep: visible text, nothing redacted, within the caps. */
function showable(words: readonly string[], needles: readonly string[]): boolean {
  if (words.length === 0 || words.length > NEVER_WORDS_MAX) return false;
  const text = ruleText(words);
  if (text.length > NEVER_CHARS_MAX) return false;
  for (const word of words) {
    if (word === '') return false;
    for (let k = 0; k < word.length; k++) if (word.charCodeAt(k) < 0x20) return false;
  }
  return visibleText(text) === text && redactText(text, needles) === text;
}

/**
 * The scopes a card offers for a shell command: `short` is the program, with its subcommand for a tool that has
 * them; `exact` is every word of the first real command (setup commands like `cd` skipped), cut to the caps. An
 * `exact` that cannot be shown is `short`. Null when the command is unread or no scope can be shown.
 */
export function offer(command: string, needles: readonly string[] = []): NeverOffer | null {
  const parts = commandParts(command);
  if (parts === null) return null;
  for (const part of parts) {
    const words = peeled(withoutAssignments(part)).map((w) => w.value);
    if (words.length === 0 || SETUP.has(words[0]!)) continue;
    const head = words[0]!;
    const sub = words[1];
    const short =
      SUBCOMMAND_TOOLS.has(head) && sub !== undefined && sub !== '' && !sub.startsWith('-') ? [head, sub] : [head];
    if (!showable(short, needles)) return null;
    let exact = words.slice(0, NEVER_WORDS_MAX);
    while (exact.length > short.length && !showable(exact, needles)) exact = exact.slice(0, -1);
    return { short, exact };
  }
  return null;
}

/**
 * How a wrapper takes its options: `flags` are letters that take no value, `values` letters that take one (the rest of
 * the word, else the next word); `longFlags` and `longValues` the same for `--name` (`--name=value` takes its own).
 * `assigns`: `NAME=value` words may follow the options. `stop`: letters after which the wrapper runs nothing.
 */
interface WrapperShape {
  flags: string;
  values: string;
  longFlags: readonly string[];
  longValues: readonly string[];
  assigns: boolean;
  stop: string;
  /** `nice -10`: a short option of digits only. */
  numeric: boolean;
}

function shape(parts: Partial<WrapperShape>): WrapperShape {
  return { flags: '', values: '', longFlags: [], longValues: [], assigns: false, stop: '', numeric: false, ...parts };
}

/** Programs that run the rest of their words as a command, and how each takes its options. */
const WRAPPERS: ReadonlyMap<string, WrapperShape> = new Map([
  [
    'sudo',
    shape({
      flags: 'AbBEeHiKkLlNnPSsVv',
      values: 'CDghpRrTtUu',
      longFlags: [
        'askpass',
        'background',
        'bell',
        'edit',
        'help',
        'list',
        'login',
        'non-interactive',
        'preserve-env',
        'preserve-groups',
        'remove-timestamp',
        'reset-timestamp',
        'set-home',
        'shell',
        'stdin',
        'validate',
        'version',
      ],
      longValues: [
        'chdir',
        'chroot',
        'close-from',
        'command-timeout',
        'group',
        'host',
        'other-user',
        'prompt',
        'role',
        'type',
        'user',
      ],
      assigns: true,
    }),
  ],
  [
    'env',
    shape({
      flags: 'i0v',
      values: 'uCP',
      longFlags: ['ignore-environment', 'null', 'debug'],
      longValues: ['unset', 'chdir'],
      assigns: true,
    }),
  ],
  ['command', shape({ flags: 'p', stop: 'vV' })],
  ['exec', shape({ flags: 'cl', values: 'a' })],
  ['nohup', shape({})],
  ['time', shape({ flags: 'ahlpqv', values: 'fo' })],
  ['nice', shape({ values: 'n', longValues: ['adjustment'], numeric: true })],
  ['builtin', shape({})],
]);

/** Shells whose `-c` runs the next word as a command line. */
const SHELLS = new Set(['sh', 'bash', 'zsh']);
/** Shell options that take the next word. */
const SHELL_LONG_VALUES = new Set(['--rcfile', '--init-file']);
/** Wrappers and `-c` shells read through, in all, for one simple command. */
const LAYERS_MAX = 8;

/** Options a matcher reads a command with. `foldCase`: the disk finds a program by its name in any case (macOS). */
export interface MatchOptions {
  foldCase?: boolean;
}

function isDigits(text: string): boolean {
  if (text === '') return false;
  for (let k = 0; k < text.length; k++) if (text.charCodeAt(k) < 0x30 || text.charCodeAt(k) > 0x39) return false;
  return true;
}

/** The name a program is found by: its last path part, folded when the disk ignores case. */
function programKey(word: string, fold: boolean): string {
  const name = programName(word);
  return fold ? foldName(name) : name;
}

/**
 * The words a wrapper runs, after its options (and its assignments): null when it runs nothing or is not a wrapper;
 * `unread` when it takes an option the app does not know (`env -S "git push"`), so what it runs cannot be read.
 */
function unwrapped(words: readonly Word[], fold: boolean): Word[] | null | 'unread' {
  const head = programKey(words[0]!.value, fold);
  if (head === 'timeout') return afterTimeout(words);
  const wrapper = WRAPPERS.get(head);
  if (wrapper === undefined) return null;
  let k = 1;
  while (k < words.length) {
    const v = words[k]!.value;
    if (v === '--') {
      k++;
      break;
    }
    if (v === '-' && head === 'env') {
      k++;
      continue;
    }
    if (!v.startsWith('-') || v === '-') break;
    if (v.startsWith('--')) {
      const eq = v.indexOf('=');
      const name = eq === -1 ? v.slice(2) : v.slice(2, eq);
      if (wrapper.longValues.includes(name)) k += eq === -1 ? 2 : 1;
      else if (wrapper.longFlags.includes(name)) k++;
      else return 'unread';
      continue;
    }
    if (wrapper.numeric && isDigits(v.slice(1))) {
      k++;
      continue;
    }
    let taken = 1;
    for (let c = 1; c < v.length; c++) {
      const letter = v.charAt(c);
      if (wrapper.stop.includes(letter)) return null;
      if (wrapper.values.includes(letter)) {
        if (c === v.length - 1) taken = 2;
        break;
      }
      if (!wrapper.flags.includes(letter)) return 'unread';
    }
    k += taken;
  }
  const rest = words.slice(k);
  const inner = wrapper.assigns ? withoutAssignments(rest) : rest;
  return inner.length === 0 ? null : inner;
}

/** `timeout [options] DURATION command…`: the command; `unread` for an option or a duration the app does not know. */
function afterTimeout(words: readonly Word[]): Word[] | null | 'unread' {
  let k = 1;
  while (k < words.length) {
    const v = words[k]!.value;
    if (v === '--') {
      k++;
      break;
    }
    if (!v.startsWith('-') || v === '-') break;
    if (v === '-s' || v === '-k' || v === '--signal' || v === '--kill-after') k += 2;
    else if (v === '--preserve-status' || v === '--foreground' || v === '--verbose' || v === '-v') k++;
    else if (v.startsWith('--signal=') || v.startsWith('--kill-after=') || v.startsWith('-s') || v.startsWith('-k'))
      k++;
    else return 'unread';
  }
  if (k >= words.length) return null;
  if (!isDuration(words[k]!.value)) return 'unread';
  const rest = words.slice(k + 1);
  return rest.length === 0 ? null : rest;
}

/**
 * The command line a `sh -c` / `bash -c` / `zsh -c` runs: the first word after its options when one of them holds
 * `c`; null for any other command (a shell that runs a script file is a program like any other).
 */
function shellLine(words: readonly Word[], fold: boolean): string | null {
  if (!SHELLS.has(programKey(words[0]!.value, fold))) return null;
  let line = false;
  let k = 1;
  while (k < words.length) {
    const v = words[k]!.value;
    if (v === '--' || v === '-') {
      k++;
      break;
    }
    if (v.startsWith('--')) {
      k += SHELL_LONG_VALUES.has(v) ? 2 : 1;
      continue;
    }
    if (!(v.startsWith('-') || v.startsWith('+')) || v.length < 2) break;
    let taken = 1;
    for (let c = 1; c < v.length; c++) {
      const letter = v.charAt(c);
      if (letter === 'c' && v.charAt(0) === '-') line = true;
      if (letter === 'o' || letter === 'O') taken = 2;
    }
    k += taken;
  }
  return line && k < words.length ? words[k]!.value : null;
}

/**
 * Every form of one simple command a rule may start with, into `out`: as written after its assignments, then after
 * each wrapper, and each simple command of a `-c` line read with the same scanner. False when a `-c` line or a
 * wrapper's options cannot be read (or the layers run out inside one).
 */
function formsOf(words: readonly Word[], fold: boolean, depth: number, out: string[][]): boolean {
  let rest = withoutAssignments(words);
  for (let layer = depth; layer < LAYERS_MAX && rest.length > 0; layer++) {
    out.push(rest.map((w) => w.value));
    const line = shellLine(rest, fold);
    if (line !== null) {
      const parts = commandParts(line);
      if (parts === null) return false;
      for (const part of parts) if (!formsOf(part, fold, layer + 1, out)) return false;
      return true;
    }
    const inner = unwrapped(rest, fold);
    if (inner === 'unread') return false;
    if (inner === null) return true;
    rest = inner;
  }
  return rest.length === 0;
}

function startsWith(words: readonly string[], rule: readonly string[], fold: boolean): boolean {
  if (rule.length === 0 || words.length < rule.length) return false;
  const head = rule[0]!;
  const first = words[0]!;
  const same = (a: string, b: string): boolean => (fold ? foldName(a) === foldName(b) : a === b);
  if (!same(first, head) && (head.includes('/') || !same(programName(first), head))) return false;
  for (let k = 1; k < rule.length; k++) if (words[k] !== rule[k]) return false;
  return true;
}

/**
 * The first rule a simple command of `command` starts with: as written after its assignments, after each wrapper it
 * starts with (`sudo`, `env`, `command`, `exec`, `nohup`, `time`, `timeout`, `nice`, `builtin`, with their options), or
 * inside the line of a `sh -c`, `bash -c` or `zsh -c`, read with the same scanner. A program written as a path matches
 * by its name; on a disk that ignores case (macOS) the program word is compared without case. `unread`: the app
 * cannot read the command, a `-c` line of it or a wrapper's options, and there are rules (a rule found elsewhere in
 * it wins). Null: no rule matches, or there are none.
 */
export function match(
  command: string,
  rules: readonly NeverRule[],
  options: MatchOptions = {},
): NeverRule | 'unread' | null {
  if (rules.length === 0) return null;
  const fold = options.foldCase ?? process.platform === 'darwin';
  const parts = commandParts(command);
  if (parts === null) return 'unread';
  let unread = false;
  for (const part of parts) {
    const forms: string[][] = [];
    if (!formsOf(part, fold, 0, forms)) unread = true;
    for (const rule of rules) if (forms.some((words) => startsWith(words, rule.words, fold))) return rule;
  }
  return unread ? 'unread' : null;
}

/** A Work folder's key: each part folded as the disk compares names. */
export function folderKey(cwd: string): string {
  return cwd
    .split('/')
    .map((p) => foldName(p))
    .join('/');
}

/** `n_` and 32 lowercase hex digits. */
export function isRuleId(value: unknown): value is string {
  if (typeof value !== 'string' || value.length !== 34 || !value.startsWith('n_')) return false;
  for (let k = 2; k < value.length; k++) if (!'0123456789abcdef'.includes(value.charAt(k))) return false;
  return true;
}

/** A rule of the file, field by field; anything else is null. */
function readRule(value: unknown): NeverRule | null {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return null;
  const r = value as Record<string, unknown>;
  if (!isRuleId(r.id) || !Array.isArray(r.words) || typeof r.at !== 'number' || !Number.isFinite(r.at)) return null;
  if (!r.words.every((w): w is string => typeof w === 'string')) return null;
  const words = r.words as string[];
  if (!showable(words, [])) return null;
  return { id: r.id, words: [...words], at: r.at };
}

interface Folder {
  path: string;
  rules: NeverRule[];
}

/** The rules of a file, by folder key; a file that is not the known shape reads as none. */
export function parseNeverFile(text: string): Map<string, Folder> {
  const out = new Map<string, Folder>();
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    return out;
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) return out;
  const record = raw as { version?: unknown; folders?: unknown };
  if (record.version !== 1 || typeof record.folders !== 'object' || record.folders === null) return out;
  if (Array.isArray(record.folders)) return out;
  for (const [path, list] of Object.entries(record.folders as Record<string, unknown>)) {
    if (!(posix.isAbsolute(path) || win32.isAbsolute(path)) || !Array.isArray(list)) continue;
    const key = folderKey(path);
    const folder = out.get(key) ?? { path, rules: [] };
    for (const item of list as unknown[]) {
      const rule = readRule(item);
      if (rule === null || folder.rules.length >= NEVER_RULES_MAX) continue;
      if (folder.rules.some((r) => r.id === rule.id || ruleText(r.words) === ruleText(rule.words))) continue;
      folder.rules.push(rule);
    }
    if (folder.rules.length > 0) out.set(key, folder);
  }
  return out;
}

export type RemoveResult = 'removed' | 'not-found' | 'failed';

/**
 * The rule file, read once; every write is the whole state, after the one before it. A rule is in memory from the
 * moment it is added, so a request that arrives while the file is written is matched; a write that fails takes it out
 * again. A broken file reads as no rules and is written again only when a rule is added.
 */
export class NeverRules {
  readonly path: string;
  private readonly now: () => number;
  private folders = new Map<string, Folder>();
  private loading: Promise<void> | null = null;
  private chain: Promise<void> = Promise.resolve();

  constructor(path: string, options: { now?: () => number } = {}) {
    this.path = path;
    this.now = options.now ?? Date.now;
  }

  /** Resolves once the file was read (a missing or broken file: no rules). */
  ready(): Promise<void> {
    this.loading ??= readFile(this.path, 'utf8').then(
      (text) => {
        this.folders = parseNeverFile(text);
      },
      () => {},
    );
    return this.loading;
  }

  /** The rules of a folder, oldest first; none before {@link ready}. */
  rulesOf(cwd: string): readonly NeverRule[] {
    return this.folders.get(folderKey(cwd))?.rules ?? [];
  }

  /** The words of a rule of the folder as one line, or null when it is gone. */
  wordsOf(cwd: string, id: string): string | null {
    const rule = this.rulesOf(cwd).find((r) => r.id === id);
    return rule === undefined ? null : ruleText(rule.words);
  }

  /**
   * Adds a rule (the same words again: the rule there is), in memory now. `saved` resolves false when the file was
   * not written (the rule is taken out again) or the folder holds {@link NEVER_RULES_MAX} rules (`rule` null).
   */
  add(cwd: string, words: readonly string[]): { rule: NeverRule | null; saved: Promise<boolean> } {
    const key = folderKey(cwd);
    const folder = this.folders.get(key) ?? { path: cwd, rules: [] };
    const same = folder.rules.find((r) => ruleText(r.words) === ruleText(words));
    if (same !== undefined) return { rule: same, saved: Promise.resolve(true) };
    if (folder.rules.length >= NEVER_RULES_MAX || !showable(words, [])) {
      return { rule: null, saved: Promise.resolve(false) };
    }
    const rule: NeverRule = { id: `n_${randomUUID().replaceAll('-', '')}`, words: [...words], at: this.now() };
    folder.rules.push(rule);
    this.folders.set(key, folder);
    const saved = this.write().then(
      () => true,
      () => {
        this.drop(key, rule.id);
        return false;
      },
    );
    return { rule, saved };
  }

  /** Removes a rule of the folder; a write that fails puts it back. */
  async remove(cwd: string, id: string): Promise<RemoveResult> {
    const key = folderKey(cwd);
    const folder = this.folders.get(key);
    const at = folder?.rules.findIndex((r) => r.id === id) ?? -1;
    if (folder === undefined || at === -1) return 'not-found';
    const [rule] = folder.rules.splice(at, 1);
    if (folder.rules.length === 0) this.folders.delete(key);
    try {
      await this.write();
      return 'removed';
    } catch {
      const back = this.folders.get(key) ?? folder;
      back.rules.splice(Math.min(at, back.rules.length), 0, rule!);
      this.folders.set(key, back);
      return 'failed';
    }
  }

  private drop(key: string, id: string): void {
    const folder = this.folders.get(key);
    if (folder === undefined) return;
    folder.rules = folder.rules.filter((r) => r.id !== id);
    if (folder.rules.length === 0) this.folders.delete(key);
  }

  private write(): Promise<void> {
    const run = (): Promise<void> => {
      const folders: Record<string, NeverRule[]> = {};
      for (const folder of this.folders.values()) folders[folder.path] = folder.rules.map((r) => ({ ...r }));
      return writeFileAtomic(this.path, `${JSON.stringify({ version: 1, folders })}\n`);
    };
    const next = this.chain.then(run, run);
    this.chain = next.catch(() => {});
    return next;
  }
}
