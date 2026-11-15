/**
 * The Auto scan: what a command may do with no card in Auto. It allows ONLY what it models in full: every
 * character and construct is in a small subset of shell (G1), every head is on the known list (G2), every option is
 * on its head's table (G3, G4, G7), and every path is placed as text so the disk check can follow it (G5, G6). Any
 * other text is `unread` or `unknown`, and the card shows. It never runs, expands or resolves anything on disk.
 *
 * Shell facts (design 13b): each Work call is a new `$SHELL -c` (zsh or bash) started in the Work folder; no state
 * carries from one call to the next, so relative paths are placed against the Work folder.
 */

import { KEY_VARIABLES, type WorkPlace } from './command-flags.js';
import { rootText } from './roots.js';

/** A path the disk check must follow, in the order the command uses it. `part` is the simple command's index. */
export type PathCheck =
  /**
   * Written: lands inside, not protected, one link, a file or folder (not a device). `mode`: the call changes its mode
   * (`chmod`), which in a temp root can open it to other users: there it asks.
   */
  | { kind: 'write'; path: string; part: number; mode?: true }
  /** `cp` / `mv` destination: an existing folder or a path that ends in `/` takes `<dest>/<name>` for each source. */
  | { kind: 'into'; path: string; names: string[]; slash: boolean; part: number }
  /** `cd`: every folder from `base` down through `parts` exists and is not a link. */
  | { kind: 'cd'; base: string; parts: string[]; part: number }
  /**
   * Read: it lands inside the folder (a reader that recurses follows a link it is given) and is not protected. `file`:
   * not a folder either (`diff` of a folder opens its entries through their links).
   */
  | { kind: 'read'; path: string; part: number; file?: true }
  /**
   * A reader's pattern `<base>/<parts>/<pattern>`: every folder of `parts` is real (not a link), and each entry the
   * pattern may match (any case, dot names too) is a file or a folder, not a link, not protected.
   */
  | { kind: 'glob'; base: string; parts: string[]; pattern: string; part: number }
  /**
   * A path a project tool opens: an existing regular file or folder inside, with no link on any part below the Work
   * folder, not protected. A name that does not exist may be a filter the tool matches against other paths.
   */
  | { kind: 'tool'; path: string; part: number }
  /** A `git fetch` / `pull` / `push` remote: a remote in the folder's `.git/config`, and no entry `base/name`. */
  | { kind: 'remote'; base: string; name: string; part: number }
  /** `run <name>` of a package manager: `name` is a script of `<base>/package.json`. */
  | { kind: 'script-name'; base: string; name: string; part: number }
  /**
   * Every git call: `.git` is a real folder with no `commondir`, no filled `modules`, no alternates and no link among
   * its entries; the index names no submodule; every key of `.git/config` is on the allow list; and no folder between
   * the Work folder and each of `dirs` (where git starts) holds another `.git`. `network` (fetch, pull, push): every
   * remote URL is a network one. `hooks` (the call can run a hook): `.git/hooks` holds only samples too, and no global
   * config moves the hooks folder or includes a file.
   */
  | { kind: 'git-config'; hooks: boolean; network: boolean; dirs: string[]; part: number }
  /** A script a project tool runs: a regular file that lands inside, not protected. */
  | { kind: 'script'; path: string; part: number }
  /** Must not exist (a `git checkout` operand that would name a file). */
  | { kind: 'absent'; path: string; part: number }
  /** Copied or moved: a later path equal to it or under it cannot be placed. */
  | { kind: 'moved'; path: string; part: number };

/** What the Auto check needs from a command, beside its flags. */
export interface AutoScan {
  /** Some of it is outside the subset the scan models (G1), or the scan ran past its step budget (G8). */
  unread: boolean;
  /** A head, subcommand or option that is not on its table. */
  unknown: boolean;
  /** A word names a protected file or folder (folded the way the disk compares names). */
  protectedName: boolean;
  /** A command keeps running in the background. `&` is unread, so the scan never sets it. */
  background: boolean;
  /** A word names a key variable. */
  keyName: boolean;
  /** It runs git: the Work folder itself must hold `.git`. */
  git: boolean;
  /** It may change the folder's links or names as it runs (`mv`, a git command that changes files, a project tool). */
  changes: boolean;
  /** Absolute paths, as text, that it writes to. Null when one cannot be placed. */
  targets: string[] | null;
  /** Every path the disk check follows, in order. Null when one cannot be placed. */
  checks: PathCheck[] | null;
  /** The folders a project tool or script may start in, as text: every folder the shell may be in at such a part. */
  starts: string[];
  /** The text of the first asking part of each simple command that asks (at most 3 kept). */
  asked: string[];
  /** How many simple commands ask. */
  askedCount: number;
  /** The text is unread because of `2>` to a file. */
  redirect: boolean;
  /** Work done, for the cost tests. */
  steps: number;
}

export interface AutoScanOptions {
  /** The Work environment sets `CDPATH`: a relative `cd` may land anywhere. */
  cdpath: boolean;
}

/** The work one command's scan may do; past it the command is unread. */
export const SCAN_STEP_BUDGET = 2_000_000;
/** Paths one command may name; past this many none is placed. */
const CHECKS_MAX = 256;
/** Folders a list of `cd`s may leave the shell in; past this many none is known. */
const DIRS_MAX = 8;

// ── names ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Files and folders that hold hooks, allow rules, MCP servers or shell start code. Compared folded ({@link foldName}). */
export const PROTECTED_NAMES: ReadonlySet<string> = new Set([
  '.git',
  '.grok',
  '.claude',
  '.claude.json',
  '.cursor',
  '.mcp.json',
  '.mlx-agent',
  '.mlx-node',
  '.ssh',
  '.config',
  '.npmrc',
  '.gitconfig',
  '.zshrc',
  '.zshenv',
  '.zprofile',
  '.zlogin',
  '.bashrc',
  '.bash_profile',
  '.profile',
  '.ugrep',
]);

/** Characters a file system ignores when it compares names, and that print as nothing. */
function ignorable(code: number): boolean {
  return (
    code === 0xad ||
    code === 0x34f ||
    code === 0x61c ||
    code === 0x115f ||
    code === 0x1160 ||
    code === 0x17b4 ||
    code === 0x17b5 ||
    (code >= 0x180b && code <= 0x180f) ||
    (code >= 0x200b && code <= 0x200f) ||
    (code >= 0x202a && code <= 0x202e) ||
    (code >= 0x2060 && code <= 0x206f) ||
    code === 0x3164 ||
    (code >= 0xfe00 && code <= 0xfe0f) ||
    code === 0xfeff ||
    code === 0xffa0 ||
    (code >= 0xfff0 && code <= 0xfff8) ||
    (code >= 0x1bca0 && code <= 0x1bca3) ||
    (code >= 0x1d173 && code <= 0x1d17a) ||
    (code >= 0xe0000 && code <= 0xe0fff)
  );
}

function withoutIgnorable(text: string): string {
  let out = '';
  for (const ch of text) if (!ignorable(ch.codePointAt(0)!)) out += ch;
  return out;
}

/**
 * A name as a case-insensitive disk compares it: normal form, full case folding, ignorable characters removed. The
 * fold repeats until it stops changing, so names the disk takes as one (ẞ, ß, ss) get one key.
 */
export function foldName(name: string): string {
  let out = withoutIgnorable(name.normalize('NFC'));
  for (let round = 0; round < 8; round++) {
    const next = withoutIgnorable(out.toUpperCase().toLowerCase().normalize('NFC'));
    if (next === out) break;
    out = next;
  }
  return out;
}

export function hasNonAscii(text: string): boolean {
  for (let k = 0; k < text.length; k++) if (text.charCodeAt(k) > 0x7f) return true;
  return false;
}

/** A path part that names a protected file, or a dot name with a non-ASCII character (it may fold to one). */
export function protectedPart(part: string): boolean {
  if (part.startsWith('.') && hasNonAscii(part)) return true;
  return PROTECTED_NAMES.has(foldName(part));
}

const NAME_SEPARATORS = '/=:, \t\n';

/** Whether a word names a protected file or folder as one of its parts. */
export function namesProtected(text: string): boolean {
  let start = 0;
  for (let k = 0; k <= text.length; k++) {
    if (k === text.length || NAME_SEPARATORS.includes(text.charAt(k))) {
      if (k > start && protectedPart(text.slice(start, k))) return true;
      start = k + 1;
    }
  }
  return false;
}

function isIdentChar(c: string | undefined): boolean {
  if (c === undefined || c.length !== 1) return false;
  const code = c.charCodeAt(0);
  return (
    (code >= 0x41 && code <= 0x5a) || (code >= 0x61 && code <= 0x7a) || (code >= 0x30 && code <= 0x39) || c === '_'
  );
}

function namesKey(text: string): boolean {
  for (const name of KEY_VARIABLES) {
    for (let at = text.indexOf(name); at !== -1; at = text.indexOf(name, at + 1)) {
      if (!isIdentChar(text[at - 1]) && !isIdentChar(text[at + name.length])) return true;
    }
  }
  return false;
}

// ── G1: the text subset ─────────────────────────────────────────────────────────────────────────────────────────────

class Unread extends Error {}
const UNREAD = new Unread('unread');

interface Meter {
  steps: number;
  budget: number;
  /** Set before the lexer throws on `2>` to a file. */
  note?: '2>';
}

function tick(meter: Meter, n: number): void {
  meter.steps += n;
  if (meter.steps > meter.budget) throw UNREAD;
}

/** Characters that are never read, inside quotes too: controls, invisible and bidi characters, non-ASCII spaces. */
function forbidden(code: number): boolean {
  if (code < 0x20) return code !== 0x09 && code !== 0x0a;
  if (code === 0x7f || (code >= 0x80 && code <= 0xa0)) return true;
  if (code < 0x2000) return code === 0xad || code === 0x34f || code === 0x61c || code === 0x1680 || ignorable(code);
  if (code <= 0x200f || (code >= 0x2028 && code <= 0x202f) || (code >= 0x205f && code <= 0x206f)) return true;
  if (code === 0x3000 || (code >= 0xd800 && code <= 0xdfff) || (code >= 0xe000 && code <= 0xf8ff)) return true;
  if (code >= 0xfff0 && code <= 0xffff) return true;
  return ignorable(code);
}

/** Non-ASCII letters a word may hold outside quotes: Latin, Greek, Cyrillic, kana, CJK, Hangul, combining accents. */
function wordLetter(code: number): boolean {
  return (
    (code >= 0xc0 && code <= 0x24f && code !== 0xd7 && code !== 0xf7) ||
    (code >= 0x300 && code <= 0x36f) ||
    (code >= 0x370 && code <= 0x3ff && code !== 0x37e && code !== 0x387) ||
    (code >= 0x400 && code <= 0x4ff) ||
    (code >= 0x3041 && code <= 0x30ff) ||
    (code >= 0x3400 && code <= 0x4dbf) ||
    (code >= 0x4e00 && code <= 0x9fff) ||
    (code >= 0xac00 && code <= 0xd7a3)
  );
}

function asciiWordChar(c: string): boolean {
  const code = c.charCodeAt(0);
  if ((code >= 0x30 && code <= 0x39) || (code >= 0x41 && code <= 0x5a) || (code >= 0x61 && code <= 0x7a)) return true;
  return (
    c === '_' || c === '-' || c === '.' || c === '/' || c === ',' || c === ':' || c === '@' || c === '+' || c === '%'
  );
}

function isDigit(c: string | undefined): boolean {
  return c !== undefined && c.length === 1 && c >= '0' && c <= '9';
}

function allDigits(text: string): boolean {
  if (text === '' || text.length > 9) return false;
  for (const c of text) if (!isDigit(c)) return false;
  return true;
}

export interface Word {
  /** Quotes removed. */
  value: string;
  /** Holds a quote. */
  quoted: boolean;
  /** Holds `*` or `?` outside quotes: a pattern the shell expands. */
  glob: boolean;
  /** Starts with `~/` outside quotes: placed in the home. */
  home: boolean;
  /** Value index where the first quote opened; -1 with no quote. */
  quoteAt: number;
  /** The word as written, quotes included. */
  raw: string;
}

/** The operator before a part: null for the first. */
type Join = '&&' | '||' | ';' | '\n' | '|' | null;

interface Part {
  words: Word[];
  /** `>` and `>>` targets. */
  writes: Word[];
  join: Join;
  pipeline: number;
  /** Place in its pipeline, 0 first. */
  pipeIndex: number;
  /** Parts in its pipeline. */
  pipeSize: number;
}

interface Build {
  value: string;
  quoted: boolean;
  glob: boolean;
  home: boolean;
  quoteAt: number;
  bracket: boolean;
  /** Text index just past the last closing single quote. */
  singleEnd: number;
  /** Text index of the word's first character. */
  start: number;
}

function isBoundary(c: string | undefined): boolean {
  return c === undefined || c === ' ' || c === '\t' || c === '\n' || c === ';' || c === '|' || c === '&';
}

/** A character that ends a word, or the end of the text. */
function wordEnd(c: string | undefined): boolean {
  return isBoundary(c) || c === '>';
}

/**
 * The index of the quote that closes the one at `open`, read by code point; throws {@link UNREAD} on a forbidden
 * character, on `$`, a backquote, a backslash or `!` inside double quotes, and when no quote closes it.
 */
function quoteEnd(text: string, open: number, quote: string, meter: Meter): number {
  let j = open + 1;
  while (j < text.length && text[j] !== quote) {
    tick(meter, 1);
    const code = text.codePointAt(j)!;
    if (forbidden(code)) throw UNREAD;
    const q = text[j];
    if (quote === '"' && (q === '$' || q === '`' || q === '\\' || q === '!')) throw UNREAD;
    j += code > 0xffff ? 2 : 1;
  }
  if (j >= text.length) throw UNREAD;
  return j;
}

/** Splits a command into simple commands, or throws {@link UNREAD} when any of it is outside the subset. */
function lex(text: string, meter: Meter): Part[] {
  const parts: Part[] = [];
  let pipeline = 0;
  let pipeIndex = 0;
  let part: Part = { words: [], writes: [], join: null, pipeline, pipeIndex, pipeSize: 1 };
  let build = null as Build | null;
  /** A redirect waits for its target: `write`, or `null2` for `2>` (only `/dev/null`). */
  let expect = null as { kind: 'write'; append: boolean } | { kind: 'null2' } | null;
  /** The last operator: a part must follow `&&`, `||` and `|`. */
  let pending: string | null = 'start';
  /** The part has a redirect, a dropped `/dev/null` or `2>&1` too: with no word, zsh runs its NULLCMD (`cat`). */
  let redirected = false;

  const startWord = (): Build => {
    build ??= {
      value: '',
      quoted: false,
      glob: false,
      home: false,
      quoteAt: -1,
      bracket: false,
      singleEnd: -1,
      start: i,
    };
    return build;
  };
  const endWord = (): void => {
    const b = build;
    if (b === null) return;
    build = null;
    if (b.bracket && (b.quoted || (b.value !== '[' && b.value !== ']'))) throw UNREAD;
    const word: Word = {
      value: b.value,
      quoted: b.quoted,
      glob: b.glob,
      home: b.home,
      quoteAt: b.quoteAt,
      raw: text.slice(b.start, i),
    };
    if (expect === null) {
      part.words.push(word);
      return;
    }
    const kind = expect;
    expect = null;
    if (word.glob || word.value === '' || b.bracket) throw UNREAD;
    const devNull = word.value === '/dev/null' && !word.quoted;
    if (kind.kind === 'null2') {
      if (!devNull) {
        meter.note = '2>';
        throw UNREAD;
      }
      return;
    }
    if (devNull && !kind.append) return;
    part.writes.push(word);
  };
  const operator = (op: Exclude<Join, null>): void => {
    if (expect !== null) throw UNREAD;
    if (part.words.length === 0) {
      if (part.writes.length > 0 || redirected) throw UNREAD;
      if (op !== '\n') throw UNREAD;
      return;
    }
    redirected = false;
    parts.push(part);
    if (op === '|') pipeIndex++;
    else {
      pipeline++;
      pipeIndex = 0;
    }
    part = { words: [], writes: [], join: op, pipeline, pipeIndex, pipeSize: 1 };
    pending = op;
  };

  const n = text.length;
  let i = 0;
  while (i < n) {
    tick(meter, 1);
    const code = text.codePointAt(i)!;
    const width = code > 0xffff ? 2 : 1;
    if (forbidden(code)) throw UNREAD;
    const c = text.slice(i, i + width);
    if (c === ' ' || c === '\t') {
      endWord();
      i++;
      continue;
    }
    if (c === '\n') {
      endWord();
      operator('\n');
      i++;
      continue;
    }
    if (c === ';') {
      endWord();
      if (text[i + 1] === ';') throw UNREAD;
      operator(';');
      i++;
      continue;
    }
    if (c === '&') {
      endWord();
      if (text[i + 1] !== '&') throw UNREAD;
      operator('&&');
      i += 2;
      continue;
    }
    if (c === '|') {
      endWord();
      if (text[i + 1] === '&') throw UNREAD;
      if (text[i + 1] === '|') {
        operator('||');
        i += 2;
      } else {
        operator('|');
        i++;
      }
      continue;
    }
    if (c === '>') {
      const b = build;
      if (b !== null) {
        if (b.value === '2' && !b.quoted && text[i - 1] === '2') {
          build = null;
          redirected = true;
          if (text.startsWith('>&1', i) && isBoundary(text[i + 3])) {
            i += 3;
            continue;
          }
          const next = text[i + 1];
          if (next === '>' || next === '&' || next === '|' || next === '(') throw UNREAD;
          if (expect !== null) throw UNREAD;
          expect = { kind: 'null2' };
          i++;
          continue;
        }
        if (isDigit(text[i - 1])) throw UNREAD;
        endWord();
      }
      if (expect !== null) throw UNREAD;
      let j = i + 1;
      let append = false;
      if (text[j] === '>') {
        append = true;
        j++;
      }
      const next = text[j];
      if (next === '&' || next === '|' || next === '(' || next === '>') throw UNREAD;
      expect = { kind: 'write', append };
      redirected = true;
      i = j;
      continue;
    }
    if (c === "'") {
      const b = startWord();
      if (b.singleEnd === i) throw UNREAD;
      const alone = b.value === '' && b.quoteAt === -1;
      if (b.quoteAt === -1) b.quoteAt = b.value.length;
      b.quoted = true;
      const j = quoteEnd(text, i, "'", meter);
      if (j === i + 1 && (!alone || !wordEnd(text[j + 1]))) throw UNREAD;
      b.value += text.slice(i + 1, j);
      b.singleEnd = j + 1;
      i = j + 1;
      continue;
    }
    if (c === '"') {
      const b = startWord();
      const alone = b.value === '' && b.quoteAt === -1;
      if (b.quoteAt === -1) b.quoteAt = b.value.length;
      b.quoted = true;
      const j = quoteEnd(text, i, '"', meter);
      if (j === i + 1 && (!alone || !wordEnd(text[j + 1]))) throw UNREAD;
      b.value += text.slice(i + 1, j);
      i = j + 1;
      continue;
    }
    if (c === '~') {
      if (build !== null || text[i + 1] !== '/') throw UNREAD;
      startWord().home = true;
      build!.value += '~';
      i++;
      continue;
    }
    if (c === '=') {
      const b = build;
      if (b === null || b.value === '' || b.value.endsWith('=') || b.value.endsWith(':') || b.singleEnd === i) {
        throw UNREAD;
      }
      if (text[i - 1] === '"') throw UNREAD;
      b.value += c;
      i++;
      continue;
    }
    if (c === '*' || c === '?') {
      const b = startWord();
      b.glob = true;
      b.value += c;
      i++;
      continue;
    }
    if (c === '[' || c === ']') {
      const b = startWord();
      b.bracket = true;
      b.value += c;
      i++;
      continue;
    }
    if ((code < 0x80 && asciiWordChar(c)) || (code >= 0x80 && wordLetter(code))) {
      startWord().value += c;
      i += width;
      continue;
    }
    throw UNREAD;
  }
  endWord();
  if (expect !== null) throw UNREAD;
  if (part.words.length > 0) parts.push(part);
  else if (part.writes.length > 0 || redirected || pending === '&&' || pending === '||' || pending === '|') {
    throw UNREAD;
  }
  const sizes = new Map<number, number>();
  for (const p of parts) sizes.set(p.pipeline, (sizes.get(p.pipeline) ?? 0) + 1);
  for (const p of parts) p.pipeSize = sizes.get(p.pipeline)!;
  return parts;
}

// ── G2: heads ───────────────────────────────────────────────────────────────────────────────────────────────────────

/** Shell words a command may not start with: control words and builtins that change the shell or run text. */
const CONTROL_WORDS: ReadonlySet<string> = new Set([
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
  '[[',
  ']]',
  '!',
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
  'unsetopt',
  'enable',
  'disable',
  'hash',
  'read',
  'popd',
  'noglob',
  'nocorrect',
  'repeat',
  'foreach',
  'emulate',
  'autoload',
  'zmodload',
  'functions',
  'unfunction',
  'integer',
  'float',
  'vared',
  'zle',
  'bindkey',
]);

/** Variables a command may set in front of its head, with a plain value. Any other name can hold a command. */
export const AUTO_ASSIGNMENTS: ReadonlySet<string> = new Set([
  'NODE_ENV',
  'CI',
  'DEBUG',
  'FORCE_COLOR',
  'NO_COLOR',
  'RUST_BACKTRACE',
  'TZ',
  'LANG',
  'LC_ALL',
  'PORT',
]);

// ── G3: option tables ───────────────────────────────────────────────────────────────────────────────────────────────

function own<T>(record: Readonly<Record<string, T>>, key: string): T | undefined {
  return Object.hasOwn(record, key) ? record[key] : undefined;
}

/** How an option value is read. */
type ValueKind =
  | 'count'
  | 'tail-count'
  | 'text'
  | 'name'
  | 'char'
  | 'list'
  | 'keydef'
  | 'read'
  | 'tool'
  | 'branch'
  | 'tb';

export interface OptionTable {
  /** Short letters that take no value. A cluster passes only when every letter is here. */
  flags?: string;
  /** Short letters that take a value (attached or the next word). */
  values?: Readonly<Record<string, ValueKind>>;
  /** Exact long options (with their dashes) that take no value. */
  long?: readonly string[];
  /** Exact long options that take a value: `--name=value` or `--name value`. */
  longValues?: Readonly<Record<string, ValueKind>>;
  /** Long options of `longValues` whose value the tool reads only in the `--name=value` form. */
  joined?: readonly string[];
  /** `--` ends the options. */
  dashdash?: boolean;
  /** A dash and digits is a count (`head -5`). */
  digits?: boolean;
  /** Options may come after operands (a reader: either reading passes). */
  permute?: boolean;
}

/** One head: its options and what its operands are. */
interface HeadRule {
  table: OptionTable;
  /** `read`: paths it reads (patterns allowed); `text`: literal words; `write`: paths it writes; `none`. */
  operands: 'read' | 'text' | 'write' | 'none';
  min?: number;
  max?: number;
}

const READER = (table: OptionTable, min = 0): HeadRule => ({
  table: { dashdash: true, permute: true, ...table },
  operands: 'read',
  min,
});

/**
 * Plain heads: their tables are checked against both the GNU and the BSD manual (design 13 §10). No table holds an
 * option that follows links while it recurses (`grep -R`, `rg -L`, `diff -r`): the folders it reaches are never placed.
 */
export const PLAIN_HEADS: Readonly<Record<string, HeadRule>> = {
  ls: READER({ flags: 'lahAR1trSdF' }),
  cat: READER({ flags: 'nbsvet' }),
  head: READER({ values: { n: 'count', c: 'count' }, digits: true }),
  tail: READER({ values: { n: 'tail-count', c: 'tail-count' }, digits: true }),
  wc: READER({ flags: 'lwcm' }),
  sort: READER({ flags: 'rnufbdhVc', values: { k: 'keydef', t: 'char' } }),
  cut: READER({ flags: 's', values: { d: 'char', f: 'list', c: 'list', b: 'list' } }),
  diff: READER({ flags: 'uqNiwbBa', values: { U: 'count' } }),
  cmp: READER({ flags: 'sl' }),
  stat: READER({ flags: 'L' }, 1),
  file: READER({ flags: 'bL' }, 1),
  du: READER({ flags: 'shack', values: { d: 'count' } }),
  df: READER({ flags: 'hkiP' }),
  realpath: READER({}, 1),
  pwd: { table: { flags: 'LP' }, operands: 'none' },
  true: { table: {}, operands: 'none' },
  false: { table: {}, operands: 'none' },
  echo: { table: { flags: 'n' }, operands: 'text' },
  which: { table: { flags: 'a' }, operands: 'text', min: 1 },
  basename: { table: {}, operands: 'text', min: 1, max: 2 },
  dirname: { table: {}, operands: 'text', min: 1 },
  tr: { table: { flags: 'dsc' }, operands: 'text', min: 1, max: 2 },
  mkdir: { table: { flags: 'pv', dashdash: true }, operands: 'write', min: 1 },
  touch: { table: { flags: 'cam', dashdash: true }, operands: 'write', min: 1 },
  tee: { table: { flags: 'a', dashdash: true }, operands: 'write' },
};

const GREP_TABLE: OptionTable = {
  flags: 'rnilcvwxHhosqEF',
  values: { e: 'text', A: 'count', B: 'count', C: 'count', m: 'count' },
  longValues: { '--include': 'text', '--exclude': 'text', '--exclude-dir': 'text' },
  dashdash: true,
  permute: true,
};

const RG_TABLE: OptionTable = {
  flags: 'nilcvwxHsSFou',
  values: { g: 'text', t: 'name', T: 'name', e: 'text', A: 'count', B: 'count', C: 'count', m: 'count' },
  long: [
    '--hidden',
    '--no-ignore',
    '--files',
    '--files-with-matches',
    '--count',
    '--line-number',
    '--ignore-case',
    '--smart-case',
    '--fixed-strings',
    '--word-regexp',
    '--no-heading',
    '--with-filename',
  ],
  longValues: { '--glob': 'text', '--type': 'name', '--max-count': 'count' },
  dashdash: true,
  permute: true,
};

const CP_TABLE: OptionTable = { flags: 'pvfn', dashdash: true };
const MV_TABLE: OptionTable = { flags: 'vfn', dashdash: true };
const CHMOD_TABLE: OptionTable = { flags: 'vf', dashdash: true };

/** `find` tests and actions that only read; each other word asks. */
const FIND_FLAGS: ReadonlySet<string> = new Set([
  '-print',
  '-print0',
  '-prune',
  '-empty',
  '-not',
  '-o',
  '-a',
  '-or',
  '-and',
]);
const FIND_VALUES: Readonly<Record<string, ValueKind | 'type' | 'size' | 'days'>> = {
  '-name': 'text',
  '-iname': 'text',
  '-path': 'text',
  '-ipath': 'text',
  '-type': 'type',
  '-maxdepth': 'count',
  '-mindepth': 'count',
  '-newer': 'read',
  '-mtime': 'days',
  '-size': 'size',
};

const DATE_TABLE: OptionTable = { flags: 'u' };
const MAKE_TABLE: OptionTable = { flags: 'ksn', values: { j: 'count' } };
const TSC_TABLE: OptionTable = { long: ['--noEmit'], longValues: { '--project': 'tool' }, values: { p: 'tool' } };

/** The option table of a head that has one, for the tests. */
export function headTable(head: string): OptionTable | undefined {
  const others: Readonly<Record<string, OptionTable>> = {
    grep: GREP_TABLE,
    rg: RG_TABLE,
    cp: CP_TABLE,
    mv: MV_TABLE,
    chmod: CHMOD_TABLE,
    date: DATE_TABLE,
    make: MAKE_TABLE,
    tsc: TSC_TABLE,
  };
  return own(PLAIN_HEADS, head)?.table ?? own(CHECKER_RULES, head) ?? own(others, head);
}

/** Unary and binary operators of `test` and `[`: file and string tests, and numbers compared as digits only. */
export const TEST_UNARY: ReadonlySet<string> = new Set(['-e', '-f', '-d', '-s', '-r', '-w', '-x', '-z', '-n', '-L']);
const TEST_NUMERIC: ReadonlySet<string> = new Set(['-eq', '-ne', '-lt', '-le', '-gt', '-ge']);
const TEST_FILES: ReadonlySet<string> = new Set(['-nt', '-ot']);

// ── G4: git ─────────────────────────────────────────────────────────────────────────────────────────────────────────

interface GitRule {
  table: OptionTable;
  /** `path`: revisions and paths, each placed inside; `remote`: a remote name then branch names; `branch`: one name. */
  operands: 'path' | 'remote' | 'branch' | 'none' | 'stash' | 'move';
  max?: number;
  /** It changes files of the folder: alone in its pipeline, and a later path of the command cannot be placed. */
  changes?: boolean;
  /** It can run a hook (a ref, the index or the work tree changes): only when no config moves the hooks folder. */
  hooks?: true;
}

export const GIT_RULES: Readonly<Record<string, GitRule>> = {
  status: {
    table: { flags: 'sbv', long: ['--short', '--branch', '--porcelain'], dashdash: true, permute: true },
    operands: 'path',
    hooks: true,
  },
  diff: {
    table: {
      long: [
        '--stat',
        '--cached',
        '--staged',
        '--name-only',
        '--name-status',
        '--shortstat',
        '--numstat',
        '--no-color',
      ],
      dashdash: true,
      permute: true,
    },
    operands: 'path',
    hooks: true,
  },
  log: {
    table: {
      flags: 'p',
      values: { n: 'count' },
      digits: true,
      long: [
        '--oneline',
        '--graph',
        '--stat',
        '--decorate',
        '--all',
        '--name-only',
        '--name-status',
        '--no-merges',
        '--reverse',
        '--follow',
        '--patch',
      ],
      longValues: {
        '--format': 'text',
        '--pretty': 'text',
        '--author': 'text',
        '--since': 'text',
        '--until': 'text',
        '--grep': 'text',
        '--max-count': 'count',
      },
      joined: ['--format', '--pretty'],
      dashdash: true,
      permute: true,
    },
    operands: 'path',
  },
  show: {
    table: {
      long: ['--stat', '--name-only', '--name-status', '--oneline', '--no-patch'],
      dashdash: true,
      permute: true,
    },
    operands: 'path',
  },
  add: {
    table: { flags: 'Auvn', long: ['--all', '--update', '--verbose', '--dry-run'], dashdash: true, permute: true },
    operands: 'path',
    hooks: true,
  },
  commit: {
    table: {
      flags: 'a',
      values: { m: 'text' },
      long: ['--all', '--amend', '--no-edit'],
      longValues: { '--message': 'text' },
      dashdash: true,
      permute: true,
    },
    operands: 'path',
    hooks: true,
  },
  branch: {
    table: { flags: 'avr', long: ['--list', '--show-current', '--all'] },
    operands: 'branch',
    max: 1,
    hooks: true,
  },
  switch: { table: { values: { c: 'branch' } }, operands: 'branch', max: 1, changes: true, hooks: true },
  checkout: { table: { values: { b: 'branch' } }, operands: 'branch', max: 1, changes: true, hooks: true },
  merge: {
    table: { long: ['--no-edit', '--ff-only', '--no-ff', '--squash', '--abort'] },
    operands: 'branch',
    max: 1,
    changes: true,
    hooks: true,
  },
  rebase: { table: { long: ['--abort', '--skip'] }, operands: 'branch', max: 1, changes: true, hooks: true },
  fetch: { table: { long: ['--all', '--tags'] }, operands: 'remote', hooks: true },
  pull: { table: { long: ['--ff-only', '--rebase', '--no-rebase'] }, operands: 'remote', changes: true, hooks: true },
  push: { table: { flags: 'u', long: ['--set-upstream', '--tags'] }, operands: 'remote', hooks: true },
  stash: { table: { flags: 'u', values: { m: 'text' } }, operands: 'stash', changes: true, hooks: true },
  tag: { table: { flags: 'l', long: ['--list'] }, operands: 'branch', max: 1, hooks: true },
  blame: { table: { values: { L: 'list' }, dashdash: true }, operands: 'path', max: 1 },
  'rev-parse': {
    table: {
      long: ['--show-toplevel', '--abbrev-ref', '--short', '--verify', '--is-inside-work-tree', '--git-dir'],
    },
    operands: 'path',
  },
  'ls-files': {
    table: {
      flags: 'mocd',
      long: ['--others', '--modified', '--cached', '--deleted', '--exclude-standard'],
      dashdash: true,
      permute: true,
    },
    operands: 'path',
  },
  grep: {
    table: { flags: 'nilwc', values: { e: 'text' }, long: ['--cached'], dashdash: true, permute: true },
    operands: 'path',
  },
  mv: { table: {}, operands: 'move', changes: true, hooks: true },
};

const STASH_SUBS: ReadonlySet<string> = new Set(['list', 'pop', 'apply', 'show', 'push']);

// ── G7: project tools ───────────────────────────────────────────────────────────────────────────────────────────────

/** Package managers: the subcommands that pass, exactly as written. `run` takes one script name. */
export const PACKAGE_SUBCOMMANDS: Readonly<Record<string, readonly string[]>> = {
  npm: ['test', 'run', 'install', 'ci'],
  pnpm: ['test', 'run', 'install'],
  yarn: ['test', 'run', 'install'],
  bun: ['test', 'run', 'install'],
};

/** Build tools: subcommand → option table. Operands are placed inside the folder. */
export const TOOL_RULES: Readonly<Record<string, Readonly<Record<string, OptionTable>>>> = {
  cargo: {
    build: cargoTable(),
    test: cargoTable(['--no-run']),
    check: cargoTable(),
  },
  go: {
    build: goTable([]),
    test: goTable(['-short', '-cover', '-failfast']),
    vet: goTable([]),
  },
  uv: {
    sync: { long: ['--frozen', '--locked'] },
    lock: { long: ['--locked'] },
  },
  vp: {
    test: { long: ['--run'] },
    check: {},
    lint: {},
    build: {},
  },
  vite: { build: {} },
};

/** Commands (not `run`) that the first operand of `vp test` names, not a filter; `watch` and `dev` keep running. */
const VITEST_COMMANDS: ReadonlySet<string> = new Set(['watch', 'dev', 'related', 'bench', 'list', 'init']);

function cargoTable(extra: readonly string[] = []): OptionTable {
  return {
    flags: 'qv',
    values: { p: 'name' },
    long: [
      '--release',
      '--workspace',
      '--all-targets',
      '--lib',
      '--bins',
      '--tests',
      '--locked',
      '--offline',
      '--quiet',
      '--verbose',
      ...extra,
    ],
    longValues: { '--package': 'name', '--bin': 'name', '--test': 'name', '--features': 'list' },
  };
}

function goTable(extra: readonly string[]): OptionTable {
  return {
    long: ['-v', '-race', ...extra],
    longValues: { '-run': 'text', '-count': 'count', '-timeout': 'text', '-tags': 'list' },
  };
}

/** Test, type and lint tools that take paths: their option tables. */
export const CHECKER_RULES: Readonly<Record<string, OptionTable>> = {
  pytest: {
    flags: 'qvxs',
    values: { k: 'text' },
    longValues: { '--maxfail': 'count', '--tb': 'tb' },
  },
  jest: { long: ['--ci', '--runInBand', '--silent'] },
  eslint: { long: ['--quiet'], longValues: { '--max-warnings': 'count' } },
  oxlint: { long: ['--quiet'] },
  prettier: { flags: 'c', long: ['--check'] },
};

/** Script names a runtime takes: a file name with one of these endings never names a subcommand of the tool. */
const NODE_ENDINGS: readonly string[] = ['.js', '.mjs', '.cjs', '.ts', '.mts', '.cts'];
const SCRIPT_ENDINGS: readonly string[] = [...NODE_ENDINGS, '.jsx', '.tsx'];

/** `python -m` modules that pass: test runners. */
const PYTHON_MODULES: ReadonlySet<string> = new Set(['pytest', 'unittest']);

/** Every head the scan knows, for the tests. */
export const KNOWN_HEADS: readonly string[] = [
  ...Object.keys(PLAIN_HEADS),
  'grep',
  'rg',
  'find',
  'test',
  '[',
  'printf',
  'sleep',
  'date',
  'cd',
  'pushd',
  'cp',
  'mv',
  'chmod',
  'git',
  'ps',
  'sed',
  'env',
  'timeout',
  'nice',
  'node',
  'bun',
  'deno',
  'python',
  'python3',
  'npm',
  'pnpm',
  'yarn',
  'make',
  ...Object.keys(TOOL_RULES),
  ...Object.keys(CHECKER_RULES),
  'vitest',
  'tsc',
];

// ── the run ─────────────────────────────────────────────────────────────────────────────────────────────────────────

interface Ctx {
  place: WorkPlace;
  meter: Meter;
  cdpath: boolean;
  scan: AutoScan;
  checks: PathCheck[];
  /** Folders the shell may be in. */
  dirs: string[];
  /** A path cannot be placed. */
  lost: boolean;
  /** An earlier part may have changed the folder's links: no later write, script or folder change is placed. */
  barrier: boolean;
  /** Set by a part that may change the folder's links; it becomes `barrier` at the next part. */
  barrierNext: boolean;
  part: number;
  current: Part;
  /** The head of the simple command being read, as written. */
  askHead: string | null;
  /** The text of this part that asks, once one rule asked. */
  askText: string | null;
  /** An absolute `cd` / `pushd` of this part into a temp root: the folder it enters. */
  cdTemp: string | null;
}

/** Thrown to stop reading one part: the scan already says why. */
class Stop extends Error {}
const STOP = new Stop('stop');

/** Asks for this part. `word`: the word that failed, shown after the head; none when the head or its form failed. */
function unknown(ctx: Ctx, word?: Word): never {
  ctx.scan.unknown = true;
  if (ctx.askText === null) {
    const head = ctx.askHead ?? ctx.current.words[0]?.raw ?? '';
    ctx.askText = word === undefined || word.raw === head ? head : `${head} ${word.raw}`;
  }
  throw STOP;
}

function lose(ctx: Ctx): void {
  ctx.lost = true;
}

function push(ctx: Ctx, check: PathCheck): void {
  tick(ctx.meter, 1);
  if (ctx.checks.length >= CHECKS_MAX) {
    ctx.lost = true;
    return;
  }
  ctx.checks.push(check);
}

/** A word as a path: absolute, `.` parts dropped. Null with a `..` part or no text. */
function placeText(word: Word, dir: string, place: WorkPlace, meter: Meter): string | null {
  tick(meter, word.value.length + 1);
  let base: string;
  let rest: string;
  if (word.home) {
    base = place.home;
    rest = word.value.slice(2);
  } else if (word.value.startsWith('/')) {
    base = '';
    rest = word.value;
  } else {
    base = dir;
    rest = word.value;
  }
  if (word.value === '') return null;
  const kept: string[] = [];
  for (const part of rest.split('/')) {
    if (part === '..') return null;
    if (part !== '' && part !== '.') kept.push(part);
  }
  if (kept.length === 0) return base === '' ? '/' : base;
  return `${base === '/' ? '' : base}/${kept.join('/')}`;
}

/** The path in the root its text names, the spelling mapped to the root's real top; null in no root. */
function inRoot(path: string | null, place: WorkPlace): string | null {
  return path === null ? null : (rootText(path, place)?.path ?? null);
}

/** A pattern a reader may take: relative, no `..` part, no dot part, a literal first character that is not `-`. */
function readerPattern(word: Word): boolean {
  const v = word.value;
  if (word.home || v.startsWith('/') || v.startsWith('-') || v.startsWith('*') || v.startsWith('?')) return false;
  for (const part of v.split('/')) if (part === '..' || part.startsWith('.')) return false;
  return true;
}

function readPath(ctx: Ctx, word: Word, file = false): void {
  if (ctx.barrier) lose(ctx);
  if (word.glob) {
    if (file) unknown(ctx);
    globPath(ctx, word);
    return;
  }
  for (const dir of ctx.dirs) {
    const path = inRoot(placeText(word, dir, ctx.place, ctx.meter), ctx.place);
    if (path === null) lose(ctx);
    else push(ctx, file ? { kind: 'read', path, part: ctx.part, file: true } : { kind: 'read', path, part: ctx.part });
  }
}

/**
 * A reader's pattern: ASCII only, `*` and `?` only in its last part, never `**`; the disk check reads the folder it
 * names. The shell matches names in a normal form and case the scan cannot rebuild from non-ASCII text.
 */
function globPath(ctx: Ctx, word: Word): void {
  if (!word.home && word.value.startsWith('/')) return tempGlob(ctx, word);
  if (!readerPattern(word) || hasNonAscii(word.value)) unknown(ctx, word);
  tick(ctx.meter, word.value.length + 1);
  const parts = word.value.split('/').filter((p) => p !== '');
  const pattern = parts.pop();
  if (pattern === undefined || pattern.includes('**')) unknown(ctx);
  for (const p of parts) if (p.includes('*') || p.includes('?')) unknown(ctx);
  for (const dir of ctx.dirs) push(ctx, { kind: 'glob', base: dir, parts, pattern, part: ctx.part });
}

/**
 * An absolute pattern: only into a temp root (an absolute pattern into the Work folder asks). ASCII, no `..`, no dot
 * part, the wildcard only in the last part, never `**`; the folder is read from the root's real top.
 */
function tempGlob(ctx: Ctx, word: Word): void {
  if (hasNonAscii(word.value)) unknown(ctx, word);
  tick(ctx.meter, word.value.length + 1);
  const path = placeText(word, '/', ctx.place, ctx.meter);
  const at = path === null ? null : rootText(path, ctx.place);
  if (at === null || at.root.kind !== 'temp') unknown(ctx, word);
  const parts = at.path
    .slice(at.root.top.length)
    .split('/')
    .filter((p) => p !== '');
  const pattern = parts.pop();
  if (pattern === undefined || pattern.includes('**') || pattern.startsWith('.')) unknown(ctx, word);
  for (const p of parts) if (p.includes('*') || p.includes('?') || p.startsWith('.')) unknown(ctx, word);
  push(ctx, { kind: 'glob', base: at.root.top, parts, pattern, part: ctx.part });
}

/** A path a project tool takes, as an operand or an option value: placed inside, even when it looks like no path. */
function toolPath(ctx: Ctx, word: Word): void {
  if (word.glob) throw UNREAD;
  if (ctx.barrier) lose(ctx);
  for (const dir of ctx.dirs) {
    const path = inRoot(placeText(word, dir, ctx.place, ctx.meter), ctx.place);
    if (path === null) lose(ctx);
    else push(ctx, { kind: 'tool', path, part: ctx.part });
  }
}

/** The part may change the folder's links or names: later paths of the command are not placed, and the host knows. */
function changing(ctx: Ctx): void {
  ctx.barrierNext = true;
  ctx.scan.changes = true;
}

function writePath(ctx: Ctx, word: Word, mode = false): void {
  if (word.glob) throw UNREAD;
  if (ctx.barrier || word.value.startsWith('/dev/')) lose(ctx);
  for (const dir of ctx.dirs) {
    const path = placeText(word, dir, ctx.place, ctx.meter);
    if (path === null) lose(ctx);
    else push(ctx, mode ? { kind: 'write', path, part: ctx.part, mode } : { kind: 'write', path, part: ctx.part });
  }
}

function scriptPath(ctx: Ctx, word: Word): void {
  if (word.glob) throw UNREAD;
  if (ctx.barrier) lose(ctx);
  for (const dir of ctx.dirs) {
    const path = inRoot(placeText(word, dir, ctx.place, ctx.meter), ctx.place);
    if (path === null) lose(ctx);
    else push(ctx, { kind: 'script', path, part: ctx.part });
  }
}

function movedPath(ctx: Ctx, word: Word): void {
  for (const dir of ctx.dirs) {
    const path = placeText(word, dir, ctx.place, ctx.meter);
    if (path === null) lose(ctx);
    else push(ctx, { kind: 'moved', path, part: ctx.part });
  }
}

function lastName(value: string): string | null {
  const parts = value.split('/').filter((p) => p !== '' && p !== '.');
  const last = parts[parts.length - 1];
  return last === undefined || last === '..' ? null : last;
}

/**
 * An operand of a project tool, placed as the file or folder the tool opens. `@file` reads options from a file (pytest,
 * tsc), and `true` / `false` can be the value of the boolean option before it (tsc, yargs): both ask. A `:` is a
 * selector or a line filter (`a.py::test`, `a.test.ts:3`), and a quoted `*`, `?`, `[` or `]` is a pattern the tool
 * expands itself: the tool opens another path than the text names, so these ask too.
 */
function toolOperand(ctx: Ctx, word: Word): void {
  if (word.glob) throw UNREAD;
  const v = word.value;
  if (v.startsWith('-') || v.startsWith('@') || v === 'true' || v === 'false') unknown(ctx);
  for (const c of ':*?[]') if (v.includes(c)) unknown(ctx);
  toolPath(ctx, word);
}

// ── option parsing ──────────────────────────────────────────────────────────────────────────────────────────────────

function nameValue(v: string): boolean {
  if (v === '' || v.length > 200 || v.startsWith('-') || v.startsWith('.')) return false;
  for (const c of v) {
    const ok = isIdentChar(c) || c === '-' || c === '.' || c === '/';
    if (!ok) return false;
  }
  return !v.includes('..') && !v.includes('//') && !v.endsWith('/') && !v.endsWith('.');
}

function valueOk(ctx: Ctx, kind: ValueKind | 'type' | 'size' | 'days', word: Word): boolean {
  tick(ctx.meter, word.value.length + 1);
  if (word.glob || word.home) throw UNREAD;
  const v = word.value;
  switch (kind) {
    case 'count':
      return allDigits(v);
    case 'tail-count':
      return allDigits(v.startsWith('+') ? v.slice(1) : v);
    case 'text':
      return true;
    case 'name':
      return nameValue(v);
    case 'branch':
      return nameValue(v);
    case 'char':
      return v.length === 1 && v !== '-';
    case 'list': {
      if (v === '' || v.startsWith('-')) return false;
      for (const c of v) if (!isIdentChar(c) && c !== ',' && c !== '-') return false;
      return true;
    }
    case 'keydef': {
      if (v === '') return false;
      for (const c of v) if (!isIdentChar(c) && c !== ',' && c !== '.') return false;
      return true;
    }
    case 'tb':
      return ['auto', 'short', 'long', 'no', 'line', 'native'].includes(v);
    case 'type':
      return v === 'f' || v === 'd' || v === 'l';
    case 'size': {
      const body = v.startsWith('+') || v.startsWith('-') ? v.slice(1) : v;
      const last = body.charAt(body.length - 1);
      return allDigits('ckMG'.includes(last) && body.length > 1 ? body.slice(0, -1) : body);
    }
    case 'days':
      return allDigits(v.startsWith('+') || v.startsWith('-') ? v.slice(1) : v);
    case 'read':
      readPath(ctx, word);
      return true;
    case 'tool':
      if (v.startsWith('@')) return false;
      toolPath(ctx, word);
      return true;
  }
}

/**
 * Reads `words` by `table`: every option must be on it (exact long names, so no abbreviation passes; a short cluster
 * only of letters that take no value, the last one may take a value). Returns the operands; asks on anything else.
 */
function parseOptions(ctx: Ctx, table: OptionTable, words: readonly Word[], seen?: Set<string>): Word[] {
  const operands: Word[] = [];
  let options = true;
  for (let k = 0; k < words.length; k++) {
    const w = words[k]!;
    const v = w.value;
    tick(ctx.meter, v.length + 1);
    if (!options && v.startsWith('-')) unknown(ctx, w);
    if (!options || !v.startsWith('-')) {
      operands.push(w);
      continue;
    }
    if (operands.length > 0 && table.permute !== true) unknown(ctx, w);
    if (w.glob || w.home) throw UNREAD;
    if (v === '--') {
      if (table.dashdash !== true) unknown(ctx, w);
      options = false;
      continue;
    }
    if (v === '-') unknown(ctx, w);
    const eq = v.indexOf('=');
    const name = eq === -1 ? v : v.slice(0, eq);
    const longValue = table.longValues === undefined ? undefined : own(table.longValues, name);
    if (table.long?.includes(name) === true || longValue !== undefined) {
      seen?.add(name);
      if (longValue === undefined) {
        if (eq !== -1) unknown(ctx, w);
        continue;
      }
      if (eq !== -1) {
        if (!valueOk(ctx, longValue, { ...w, value: v.slice(eq + 1) })) unknown(ctx, w);
        continue;
      }
      if (table.joined?.includes(name) === true) unknown(ctx, w);
      const next = words[k + 1];
      if (next === undefined || next.value.startsWith('-') || !valueOk(ctx, longValue, next)) unknown(ctx, w);
      k++;
      continue;
    }
    if (v.startsWith('--')) unknown(ctx, w);
    const letters = v.slice(1);
    if (table.digits === true && allDigits(letters)) continue;
    for (let j = 0; j < letters.length; j++) {
      const c = letters.charAt(j);
      seen?.add(c);
      if (table.flags?.includes(c) === true) continue;
      const kind = table.values === undefined ? undefined : own(table.values, c);
      if (kind === undefined) unknown(ctx, w);
      const attached = letters.slice(j + 1);
      if (attached !== '') {
        if (!valueOk(ctx, kind, { ...w, value: attached })) unknown(ctx, w);
      } else {
        const next = words[k + 1];
        if (next === undefined || next.value.startsWith('-') || !valueOk(ctx, kind, next)) unknown(ctx, w);
        k++;
      }
      break;
    }
  }
  return operands;
}

// ── heads ───────────────────────────────────────────────────────────────────────────────────────────────────────────

function isAssignment(word: Word): number {
  const v = word.value;
  const eq = v.indexOf('=');
  if (eq <= 0 || (word.quoteAt !== -1 && word.quoteAt <= eq)) return -1;
  const first = v.charAt(0);
  if (isDigit(first)) return -1;
  for (let k = 0; k < eq; k++) if (!isIdentChar(v.charAt(k))) return -1;
  return eq;
}

/** An assignment on the list with a plain value; any other assignment is unread. */
function assignmentOk(word: Word, eq: number): boolean {
  if (!AUTO_ASSIGNMENTS.has(word.value.slice(0, eq))) return false;
  return !word.quoted && !word.glob && !word.home;
}

function isDuration(v: string): boolean {
  if (v === '') return false;
  const last = v.charAt(v.length - 1);
  const body = 'smhd'.includes(last) ? v.slice(0, -1) : v;
  let dots = 0;
  if (body === '' || body.startsWith('.') || body.endsWith('.')) return false;
  for (const c of body) {
    if (c === '.') dots++;
    else if (!isDigit(c)) return false;
  }
  return dots <= 1 && body.length <= 9;
}

/**
 * A `printf` format the scan reads in full: literal characters, `%s`, `%%`, and the escapes `\n`, `\t`, `\\`. Any
 * other conversion, flag, width or `*` asks: zsh evaluates the argument of a numeric conversion, `*` and `%n` as math,
 * which runs a quoted subscript and can assign a variable.
 */
function plainFormat(format: string): boolean {
  for (let k = 0; k < format.length; k++) {
    const c = format.charAt(k);
    if (c !== '%' && c !== '\\') continue;
    const next = format.charAt(k + 1);
    if (c === '%' ? next !== 's' && next !== '%' : next !== 'n' && next !== 't' && next !== '\\') return false;
    k++;
  }
  return true;
}

/** Wrappers (`env`, `timeout`, `nice`) one command may stack in front of its head. */
const WRAPPERS_MAX = 3;

/** Reads one simple command from `at` (past assignments and `wrappers` wrappers). */
function readHead(ctx: Ctx, words: readonly Word[], at: number, wrappers: number): void {
  const w = words[at];
  if (w === undefined || wrappers > WRAPPERS_MAX) unknown(ctx);
  ctx.askHead = w.raw;
  if (w.quoted || w.glob || w.home || w.value.includes('/')) unknown(ctx);
  const head = w.value;
  if (CONTROL_WORDS.has(head)) throw UNREAD;
  const args = words.slice(at + 1);
  if (head === 'echo') return readEcho(ctx, args);
  const plain = own(PLAIN_HEADS, head);
  if (plain !== undefined) {
    const operands = parseOptions(ctx, plain.table, args);
    if (operands.length < (plain.min ?? 0) || operands.length > (plain.max ?? Number.MAX_SAFE_INTEGER)) unknown(ctx);
    for (const op of operands) {
      if (plain.operands === 'none') unknown(ctx);
      else if (plain.operands === 'read') readPath(ctx, op, head === 'diff');
      else if (plain.operands === 'write') writePath(ctx, op);
      else if (op.glob) throw UNREAD;
    }
    return;
  }
  switch (head) {
    case 'env': {
      let k = at + 1;
      let names = 0;
      for (; k < words.length; k++) {
        const eq = isAssignment(words[k]!);
        if (eq === -1) break;
        if (!assignmentOk(words[k]!, eq)) unknown(ctx);
        names++;
      }
      if (names === 0) unknown(ctx);
      return readHead(ctx, words, k, wrappers + 1);
    }
    case 'timeout': {
      const d = words[at + 1];
      if (d === undefined || d.quoted || d.glob || !isDuration(d.value)) unknown(ctx);
      return readHead(ctx, words, at + 2, wrappers + 1);
    }
    case 'nice': {
      const n = words[at + 2];
      if (words[at + 1]?.value !== '-n' || n === undefined || !allDigits(n.value)) unknown(ctx);
      return readHead(ctx, words, at + 3, wrappers + 1);
    }
    case 'grep':
    case 'rg':
      return readSearch(ctx, head, args);
    case 'find':
      return readFind(ctx, args);
    case 'test':
    case '[':
      return readTest(ctx, head, args);
    case 'printf': {
      const format = args[0];
      if (format === undefined || format.value.startsWith('-') || !plainFormat(format.value)) unknown(ctx, format);
      for (const a of args) if (a.glob) throw UNREAD;
      return;
    }
    case 'sleep': {
      const n = args[0];
      if (
        args.length !== 1 ||
        n === undefined ||
        n.glob ||
        !isDuration(n.value) ||
        'smhd'.includes(n.value.slice(-1))
      ) {
        unknown(ctx);
      }
      return;
    }
    case 'date': {
      const operands = parseOptions(ctx, DATE_TABLE, args);
      if (operands.length > 1 || operands.some((o) => o.glob || !o.value.startsWith('+'))) unknown(ctx);
      return;
    }
    case 'cd':
    case 'pushd':
      if (wrappers > 0) unknown(ctx);
      return readCd(ctx, args);
    case 'cp':
    case 'mv':
      return readCopy(ctx, head, args);
    case 'chmod':
      return readChmod(ctx, args);
    case 'git':
      return readGit(ctx, args);
    case 'ps':
      return readPs(ctx, args);
    case 'sed':
      return readSed(ctx, args);
    default:
      return readProject(ctx, head, args);
  }
}

/**
 * `echo`: zsh and bash read a leading word as options only when it is `-` and letters all of `n e E`. Such a word
 * must be `-n` alone (`-e` and `-E` change how the text prints); every other word is text.
 */
function readEcho(ctx: Ctx, args: readonly Word[]): void {
  let options = true;
  for (const w of args) {
    if (options && echoOptions(w.value)) {
      if (w.value.includes('e') || w.value.includes('E')) unknown(ctx, w);
      continue;
    }
    options = false;
    if (w.glob) throw UNREAD;
  }
}

function echoOptions(v: string): boolean {
  if (v.length < 2 || !v.startsWith('-')) return false;
  for (const c of v.slice(1)) if (c !== 'n' && c !== 'e' && c !== 'E') return false;
  return true;
}

/** `ps -o` keys: what each process shows, never its environment. */
const PS_KEYS: ReadonlySet<string> = new Set([
  'pid',
  'ppid',
  'pgid',
  'uid',
  'user',
  'comm',
  'command',
  'args',
  'etime',
  'time',
  'start',
  'lstart',
  'stat',
  'state',
  'tty',
  'pcpu',
  'pmem',
  'rss',
  'vsz',
]);

/** Comma-separated items, none empty, each passing `item`. */
function commaList(v: string, item: (text: string) => boolean): boolean {
  if (v === '' || v.length > 200) return false;
  for (const piece of v.split(',')) if (!item(piece)) return false;
  return true;
}

/**
 * `ps`: a first word of 1 to 6 letters of `a u x w`, dash clusters of `A a x f w`, `-p` with process ids, `-o` with
 * keys of {@link PS_KEYS}. Anything else asks: the forms with `e` print each process's environment.
 */
function readPs(ctx: Ctx, args: readonly Word[]): void {
  for (let k = 0; k < args.length; k++) {
    const w = args[k]!;
    const v = w.value;
    tick(ctx.meter, v.length + 1);
    if (w.quoted || w.glob || w.home) unknown(ctx);
    if (k === 0 && !v.startsWith('-')) {
      if (v === '' || v.length > 6) unknown(ctx);
      for (const c of v) if (!'auxw'.includes(c)) unknown(ctx);
      continue;
    }
    if (v === '-p' || v === '-o') {
      const list = args[k + 1];
      const ok =
        list !== undefined &&
        !list.quoted &&
        !list.glob &&
        !list.home &&
        commaList(list.value, v === '-p' ? allDigits : (key) => PS_KEYS.has(key));
      if (!ok) unknown(ctx);
      k++;
      continue;
    }
    if (v.length < 2 || !v.startsWith('-')) unknown(ctx);
    for (const c of v.slice(1)) if (!'Aaxfw'.includes(c)) unknown(ctx);
  }
}

/** A `sed` print script: `Np`, `N,Mp`, `$p` or `N,$p`. */
function printScript(v: string): boolean {
  if (!v.endsWith('p')) return false;
  const address = v.slice(0, -1);
  if (address === '$') return true;
  const comma = address.indexOf(',');
  if (comma === -1) return allDigits(address);
  const end = address.slice(comma + 1);
  return allDigits(address.slice(0, comma)) && (end === '$' || allDigits(end));
}

/** `sed -n SCRIPT [FILE...]`: one print script; each file is a read. */
function readSed(ctx: Ctx, args: readonly Word[]): void {
  const flag = args[0];
  if (flag === undefined || flag.value !== '-n' || flag.glob) unknown(ctx);
  const script = args[1];
  if (script === undefined || script.glob || script.home || !printScript(script.value)) unknown(ctx);
  for (const w of args.slice(2)) {
    if (w.value.startsWith('-')) unknown(ctx);
    readPath(ctx, w);
  }
}

function readSearch(ctx: Ctx, head: string, args: readonly Word[]): void {
  const table = head === 'grep' ? GREP_TABLE : RG_TABLE;
  const seen = new Set<string>();
  const operands = parseOptions(ctx, table, args, seen);
  const pattern = !seen.has('e') && !seen.has('--files');
  for (let k = 0; k < operands.length; k++) {
    const w = operands[k]!;
    if (k === 0 && pattern) {
      if (w.glob || w.home) throw UNREAD;
      continue;
    }
    readPath(ctx, w);
  }
}

function readFind(ctx: Ctx, args: readonly Word[]): void {
  let k = 0;
  for (; k < args.length && !args[k]!.value.startsWith('-'); k++) readPath(ctx, args[k]!);
  for (; k < args.length; k++) {
    const w = args[k]!;
    tick(ctx.meter, w.value.length + 1);
    if (w.glob) throw UNREAD;
    if (FIND_FLAGS.has(w.value)) continue;
    const kind = own(FIND_VALUES, w.value);
    const value = args[k + 1];
    if (kind === undefined || value === undefined || !valueOk(ctx, kind, value)) unknown(ctx);
    k++;
  }
}

function readTest(ctx: Ctx, head: string, words: readonly Word[]): void {
  let args = words;
  if (head === '[') {
    if (args[args.length - 1]?.value !== ']') unknown(ctx);
    args = args.slice(0, -1);
  }
  for (const a of args) if (a.glob) throw UNREAD;
  if (args.length === 1) return;
  if (args.length === 2) {
    const op = args[0]!.value;
    if (!TEST_UNARY.has(op) || args[0]!.quoted) unknown(ctx);
    if (op !== '-z' && op !== '-n') readPath(ctx, args[1]!);
    return;
  }
  if (args.length === 3) {
    const op = args[1]!.value;
    if (TEST_NUMERIC.has(op) && allDigits(args[0]!.value) && allDigits(args[2]!.value)) return;
    if (TEST_FILES.has(op) && !args[1]!.quoted) {
      readPath(ctx, args[0]!);
      readPath(ctx, args[2]!);
      return;
    }
  }
  unknown(ctx);
}

function readCd(ctx: Ctx, args: readonly Word[]): void {
  if (ctx.current.pipeSize > 1 || args.length !== 1) unknown(ctx);
  const w = args[0]!;
  const v = w.value;
  if (!w.glob && !w.home && v.startsWith('/')) return tempCd(ctx, w);
  if (w.glob || w.home || v === '' || v.startsWith('-') || v.startsWith('+')) unknown(ctx, w);
  if (ctx.cdpath) unknown(ctx, w);
  const parts: string[] = [];
  for (const p of v.split('/')) {
    if (p === '..') unknown(ctx, w);
    if (p !== '' && p !== '.') parts.push(p);
  }
  tick(ctx.meter, v.length + 1);
  if (ctx.barrier) lose(ctx);
  const next = [...ctx.dirs];
  for (const dir of ctx.dirs) {
    if (parts.length === 0) continue;
    push(ctx, { kind: 'cd', base: dir, parts, part: ctx.part });
    const to = `${dir}/${parts.join('/')}`;
    if (!next.includes(to)) next.push(to);
  }
  if (next.length > DIRS_MAX) lose(ctx);
  else ctx.dirs = next;
}

/**
 * An absolute `cd`: only into a temp root (into the Work folder, by any spelling, it asks). Every folder below the
 * root's top must be real; the scan's folders change after the part's own redirects ({@link enterTemp}).
 */
function tempCd(ctx: Ctx, w: Word): void {
  if (ctx.cdpath) unknown(ctx, w);
  const path = placeText(w, '/', ctx.place, ctx.meter);
  const at = path === null ? null : rootText(path, ctx.place);
  if (at === null || at.root.kind !== 'temp') unknown(ctx, w);
  if (ctx.barrier) lose(ctx);
  const parts = at.path
    .slice(at.root.top.length)
    .split('/')
    .filter((p) => p !== '');
  push(ctx, { kind: 'cd', base: at.root.top, parts, part: ctx.part });
  ctx.cdTemp = at.path;
}

function readCopy(ctx: Ctx, head: string, args: readonly Word[]): void {
  copyOperands(ctx, head, parseOptions(ctx, head === 'cp' ? CP_TABLE : MV_TABLE, args));
}

/**
 * `cp` / `mv` operands: the sources, then the destination, by the tool's rule for a folder destination. `mv` moves its
 * sources one by one, so an earlier move changes where a later source lands: more than one source asks. A copy only
 * makes regular files, which no later path can pass through.
 */
function copyOperands(ctx: Ctx, head: string, operands: readonly Word[]): void {
  if (operands.length < 2 || ctx.current.pipeSize > 1) unknown(ctx);
  if (head === 'mv' && operands.length > 2) unknown(ctx);
  for (const w of operands) if (w.glob) throw UNREAD;
  const dest = operands[operands.length - 1]!;
  const sources = operands.slice(0, -1);
  const names: string[] = [];
  for (const s of sources) {
    const name = lastName(s.value);
    if (name === null) unknown(ctx);
    names.push(name);
    if (head === 'cp') readPath(ctx, s);
    else {
      ctx.scan.changes = true;
      writePath(ctx, s);
    }
    movedPath(ctx, s);
  }
  if (ctx.barrier) lose(ctx);
  for (const dir of ctx.dirs) {
    const path = placeText(dest, dir, ctx.place, ctx.meter);
    if (path === null) lose(ctx);
    else push(ctx, { kind: 'into', path, names, slash: dest.value.endsWith('/'), part: ctx.part });
  }
  movedPath(ctx, dest);
}

function chmodMode(v: string): boolean {
  if (allDigits(v)) {
    if (v.length > 4 || (v.length === 4 && v.charAt(0) !== '0')) return false;
    for (const c of v) if (c > '7') return false;
    return true;
  }
  let k = 0;
  while (k < v.length && 'ugoa'.includes(v.charAt(k))) k++;
  if (k >= v.length || !'+-='.includes(v.charAt(k))) return false;
  k++;
  if (k >= v.length) return false;
  for (; k < v.length; k++) if (!'rwxX'.includes(v.charAt(k))) return false;
  return true;
}

function readChmod(ctx: Ctx, args: readonly Word[]): void {
  const operands = parseOptions(ctx, CHMOD_TABLE, args);
  const mode = operands[0];
  if (mode === undefined || operands.length < 2 || mode.glob || !chmodMode(mode.value)) unknown(ctx);
  for (const w of operands.slice(1)) writePath(ctx, w, true);
}

function readGit(ctx: Ctx, args: readonly Word[]): void {
  ctx.scan.git = true;
  let k = 0;
  if (args[0]?.value === '--no-pager' && !args[0].quoted) k = 1;
  const sub = args[k];
  if (sub === undefined || sub.quoted || sub.glob) unknown(ctx, sub);
  const rule = own(GIT_RULES, sub.value);
  if (rule === undefined) unknown(ctx, sub);
  if (rule.changes === true && ctx.current.pipeSize > 1) unknown(ctx);
  push(ctx, {
    kind: 'git-config',
    hooks: rule.hooks === true,
    network: rule.operands === 'remote',
    dirs: [...ctx.dirs],
    part: ctx.part,
  });
  const rest = args.slice(k + 1);
  if (sub.value === 'stash') return readStash(ctx, rule, rest);
  if (rule.changes === true) changing(ctx);
  const seen = new Set<string>();
  const operands = parseOptions(ctx, rule.table, rest, seen);
  if (operands.length > (rule.max ?? Number.MAX_SAFE_INTEGER)) unknown(ctx);
  for (const w of operands) if (w.glob) throw UNREAD;
  switch (rule.operands) {
    case 'path':
      for (const w of operands) {
        if (w.value.startsWith('-') || w.value.startsWith(':')) unknown(ctx);
        readPath(ctx, w);
      }
      break;
    case 'branch':
      for (const w of operands) {
        if (!nameValue(w.value)) unknown(ctx);
        if (sub.value === 'checkout') for (const dir of ctx.dirs) absent(ctx, w, dir);
      }
      break;
    case 'remote':
      for (let j = 0; j < operands.length; j++) {
        const v = operands[j]!.value;
        if (j === 0 ? !remoteName(v) : !nameValue(v)) unknown(ctx);
        if (j === 0) for (const dir of ctx.dirs) push(ctx, { kind: 'remote', base: dir, name: v, part: ctx.part });
      }
      break;
    case 'move':
      if (operands.length < 2) unknown(ctx);
      copyOperands(ctx, 'mv', operands);
      break;
    default:
      if (operands.length > 0) unknown(ctx);
  }
  if (sub.value === 'commit' && !seen.has('m') && !seen.has('--message') && !seen.has('--no-edit')) unknown(ctx);
}

function absent(ctx: Ctx, word: Word, dir: string): void {
  const path = placeText(word, dir, ctx.place, ctx.meter);
  if (path === null) lose(ctx);
  else push(ctx, { kind: 'absent', path, part: ctx.part });
}

function remoteName(v: string): boolean {
  if (v === '' || v.length > 100 || v.startsWith('-')) return false;
  for (const c of v) if (!isIdentChar(c) && c !== '-') return false;
  return true;
}

function readStash(ctx: Ctx, rule: GitRule, rest: readonly Word[]): void {
  const sub = rest[0];
  if (sub === undefined || (sub.value !== 'list' && sub.value !== 'show')) changing(ctx);
  if (sub === undefined) return;
  if (sub.quoted || !STASH_SUBS.has(sub.value)) unknown(ctx);
  const operands = parseOptions(ctx, sub.value === 'push' ? rule.table : {}, rest.slice(1));
  if (operands.length > 0) unknown(ctx);
}

/** A script a runtime runs: a file name with a script ending (never `-`, standard input or a descriptor), inside. */
function readScript(ctx: Ctx, word: Word | undefined, endings: readonly string[]): void {
  if (word === undefined || word.value.startsWith('-') || !endings.some((e) => word.value.endsWith(e))) unknown(ctx);
  if (word.value.includes(':') || word.value.includes('//')) unknown(ctx);
  scriptPath(ctx, word);
}

function readProject(ctx: Ctx, head: string, args: readonly Word[]): void {
  const known =
    head === 'node' ||
    head === 'bun' ||
    head === 'deno' ||
    head === 'python' ||
    head === 'python3' ||
    head === 'make' ||
    head === 'vitest' ||
    head === 'tsc' ||
    Object.hasOwn(PACKAGE_SUBCOMMANDS, head) ||
    Object.hasOwn(TOOL_RULES, head) ||
    Object.hasOwn(CHECKER_RULES, head);
  if (!known) unknown(ctx);
  if (ctx.current.pipeIndex !== 0) unknown(ctx);
  for (const dir of ctx.dirs) if (!ctx.scan.starts.includes(dir)) ctx.scan.starts.push(dir);
  for (const w of args) {
    if (w.glob) throw UNREAD;
    if (w.value.startsWith('@')) unknown(ctx);
  }
  switch (head) {
    case 'node':
      readScript(ctx, args[0], NODE_ENDINGS);
      for (const w of args.slice(1)) scriptArgument(ctx, w);
      break;
    case 'deno':
      if (args[0]?.value !== 'run' || args[0].quoted) unknown(ctx);
      readScript(ctx, args[1], SCRIPT_ENDINGS);
      for (const w of args.slice(2)) scriptArgument(ctx, w);
      break;
    case 'python':
    case 'python3':
      if (args[0]?.value === '-m') {
        const module = args[1];
        if (module === undefined || module.quoted || !PYTHON_MODULES.has(module.value)) unknown(ctx);
        const table = module.value === 'pytest' ? CHECKER_RULES.pytest! : {};
        const operands = parseOptions(ctx, table, args.slice(2));
        // unittest imports a dotted module name, turns a `.py` name into one, or reads `discover`: never the text.
        if (module.value === 'unittest' && operands.length > 0) unknown(ctx);
        for (const w of operands) toolOperand(ctx, w);
      } else {
        readScript(ctx, args[0], ['.py']);
        for (const w of args.slice(1)) scriptArgument(ctx, w);
      }
      break;
    case 'make':
      for (const w of parseOptions(ctx, MAKE_TABLE, args)) {
        if (!nameValue(w.value)) unknown(ctx);
      }
      break;
    case 'vitest':
      if (args[0]?.value !== 'run' || args[0].quoted) unknown(ctx);
      for (const w of parseOptions(ctx, {}, args.slice(1))) toolOperand(ctx, w);
      break;
    case 'tsc': {
      const seen = new Set<string>();
      const operands = parseOptions(ctx, TSC_TABLE, args, seen);
      if (!seen.has('--noEmit')) unknown(ctx);
      for (const w of operands) toolOperand(ctx, w);
      break;
    }
    default:
      readSubcommandTool(ctx, head, args);
  }
  changing(ctx);
}

/** An argument a script gets: a literal; when it looks like a path, placed inside. */
function scriptArgument(ctx: Ctx, word: Word): void {
  const v = word.value;
  if (word.home || v.startsWith('/') || v.startsWith('.') || v.includes('/')) readPath(ctx, word);
}

function readSubcommandTool(ctx: Ctx, head: string, args: readonly Word[]): void {
  const checker = own(CHECKER_RULES, head);
  if (checker !== undefined) {
    const seen = new Set<string>();
    for (const w of parseOptions(ctx, checker, args, seen)) toolOperand(ctx, w);
    if (head === 'prettier' && !seen.has('--check') && !seen.has('c')) unknown(ctx);
    return;
  }
  const sub = args[0];
  if (sub === undefined || sub.quoted) unknown(ctx);
  const packageSubs = own(PACKAGE_SUBCOMMANDS, head);
  if (packageSubs !== undefined) {
    if (head === 'bun' && sub.value !== 'test' && sub.value !== 'run' && sub.value !== 'install') {
      readScript(ctx, sub, SCRIPT_ENDINGS);
      for (const w of args.slice(1)) scriptArgument(ctx, w);
      return;
    }
    if (!packageSubs.includes(sub.value)) unknown(ctx);
    const rest = args.slice(1);
    if (sub.value === 'run') {
      const script = rest[0];
      if (rest.length !== 1 || script === undefined || script.quoted || !scriptName(script.value)) unknown(ctx);
      if (head === 'bun' && script.value.includes('.')) unknown(ctx);
      if (ctx.barrier) lose(ctx);
      for (const dir of ctx.dirs) push(ctx, { kind: 'script-name', base: dir, name: script.value, part: ctx.part });
    } else if (sub.value === 'test' && head === 'bun') {
      for (const w of parseOptions(ctx, {}, rest)) toolOperand(ctx, w);
    } else if (rest.length > 0) unknown(ctx);
    return;
  }
  const subs = own(TOOL_RULES, head);
  const table = subs === undefined ? undefined : own(subs, sub.value);
  if (table === undefined) unknown(ctx);
  const operands = parseOptions(ctx, table, args.slice(1));
  if (head === 'vp' && sub.value === 'test' && VITEST_COMMANDS.has(operands[0]?.value ?? '')) unknown(ctx);
  for (const w of operands) {
    if (head === 'go') {
      if (w.value !== '.' && !w.value.startsWith('./')) unknown(ctx);
      // `./x/...` names the packages under ./x; go skips links as it walks.
      const v = w.value === './...' ? '.' : w.value.endsWith('/...') ? w.value.slice(0, -4) : w.value;
      if (v.includes('...')) unknown(ctx);
      toolPath(ctx, { ...w, value: v });
    } else if (head === 'cargo') {
      if (sub.value !== 'test' || w.value.startsWith('-') || !nameValue(w.value)) unknown(ctx);
    } else toolOperand(ctx, w);
  }
}

function scriptName(v: string): boolean {
  if (v === '' || v.length > 100 || v.startsWith('-')) return false;
  for (const c of v) if (!isIdentChar(c) && c !== '-' && c !== ':' && c !== '.') return false;
  return true;
}

/**
 * The words of each simple command as the scan reads them, and how many `>` / `>>` files it writes, or null when the
 * text is unread. For the shell test.
 */
export function scanWords(
  command: string,
): Array<{ words: Array<{ value: string; home: boolean }>; writes: number }> | null {
  try {
    return lex(command, { steps: 0, budget: SCAN_STEP_BUDGET }).map((p) => ({
      words: p.words.map((w) => ({ value: w.value, home: w.home })),
      writes: p.writes.length,
    }));
  } catch {
    return null;
  }
}

/** For each simple command, the folders the scan holds the shell may be in when it runs; null when unread. For the shell test. */
export function scanDirs(
  command: string,
  place: WorkPlace,
  options: AutoScanOptions = { cdpath: false },
): string[][] | null {
  const run = scanRun(command, place, options, SCAN_STEP_BUDGET);
  return run.scan.unread ? null : run.dirs;
}

// ── entry ───────────────────────────────────────────────────────────────────────────────────────────────────────────

function emptyScan(): AutoScan {
  return {
    unread: false,
    unknown: false,
    protectedName: false,
    background: false,
    keyName: false,
    git: false,
    changes: false,
    targets: null,
    checks: null,
    starts: [],
    asked: [],
    askedCount: 0,
    redirect: false,
    steps: 0,
  };
}

/** Parts kept in the reason of a card. */
const ASKED_MAX = 3;

/**
 * The Auto scan of one command. Never throws: an error inside is `unread`. `budget` is for the cost tests.
 */
export function autoScan(
  command: string,
  place: WorkPlace,
  options: AutoScanOptions,
  budget: number = SCAN_STEP_BUDGET,
): AutoScan {
  return scanRun(command, place, options, budget).scan;
}

/** Folders of both lists, each once; null past {@link DIRS_MAX}. */
function joinDirs(a: readonly string[], b: readonly string[]): string[] | null {
  const out = [...a];
  for (const d of b) if (!out.includes(d)) out.push(d);
  return out.length > DIRS_MAX ? null : out;
}

/**
 * An and-or list is the parts between `;`, a newline, the start and the end. In a list with no `||`, the parts after
 * an absolute `cd` into a temp root run only when it worked, so in that folder alone; at the list's end the folders
 * before its first such `cd` count again.
 */
function scanRun(
  command: string,
  place: WorkPlace,
  options: AutoScanOptions,
  budget: number,
): { scan: AutoScan; dirs: string[][] } {
  const scan = emptyScan();
  const meter: Meter = { steps: 0, budget };
  const dirs: string[][] = [];
  try {
    const parts = lex(command, meter);
    const ctx: Ctx = {
      place,
      meter,
      cdpath: options.cdpath,
      scan,
      checks: [],
      dirs: [place.cwd],
      lost: false,
      barrier: false,
      barrierNext: false,
      part: 0,
      current: parts[0]!,
      askHead: null,
      askText: null,
      cdTemp: null,
    };
    if (parts.length === 0) scan.unknown = true;
    /** Per part: its and-or list holds `||`. */
    const orList: boolean[] = [];
    for (let start = 0; start < parts.length;) {
      let end = start + 1;
      while (end < parts.length && parts[end]!.join !== ';' && parts[end]!.join !== '\n') end++;
      let or = false;
      for (let k = start + 1; k < end; k++) if (parts[k]!.join === '||') or = true;
      for (let k = start; k < end; k++) orList.push(or);
      start = end;
    }
    /** The folders before the list's first narrowing `cd`; null when none narrowed. */
    let before: string[] | null = null;
    for (let p = 0; p < parts.length; p++) {
      const part = parts[p]!;
      if (p > 0 && (part.join === ';' || part.join === '\n') && before !== null) {
        const joined = joinDirs(before, ctx.dirs);
        if (joined === null) ctx.lost = true;
        else ctx.dirs = joined;
        before = null;
      }
      ctx.part = p;
      ctx.current = part;
      ctx.askHead = null;
      ctx.askText = null;
      ctx.cdTemp = null;
      dirs.push([...ctx.dirs]);
      if (ctx.barrierNext) ctx.barrier = true;
      for (const w of [...part.words, ...part.writes]) {
        tick(meter, w.value.length + 1);
        if (namesProtected(w.value)) scan.protectedName = true;
        if (namesKey(w.value)) scan.keyName = true;
      }
      try {
        readPart(ctx, part);
      } catch (error) {
        if (!(error instanceof Stop)) throw error;
      }
      if (ctx.askText !== null) {
        scan.askedCount += 1;
        if (scan.asked.length < ASKED_MAX) scan.asked.push(ctx.askText);
      }
      for (const w of part.writes) writePath(ctx, w);
      const to = ctx.cdTemp;
      if (to !== null) {
        if (!orList[p]!) {
          before ??= [...ctx.dirs];
          ctx.dirs = [to];
        } else {
          const joined = joinDirs(ctx.dirs, [to]);
          if (joined === null) ctx.lost = true;
          else ctx.dirs = joined;
        }
      }
    }
    if (!ctx.lost) {
      scan.checks = ctx.checks;
      scan.targets = ctx.checks.filter((c) => c.kind === 'write' || c.kind === 'into').map((c) => c.path);
    }
  } catch {
    scan.unread = true;
    scan.checks = null;
    scan.targets = null;
    if (meter.note === '2>') scan.redirect = true;
  }
  scan.steps = meter.steps;
  return { scan, dirs };
}

function readPart(ctx: Ctx, part: Part): void {
  const words = part.words;
  let k = 0;
  for (; k < words.length; k++) {
    const eq = isAssignment(words[k]!);
    if (eq === -1) break;
    if (!assignmentOk(words[k]!, eq)) throw UNREAD;
  }
  if (k === words.length) unknown(ctx);
  readHead(ctx, words, k, 0);
}
