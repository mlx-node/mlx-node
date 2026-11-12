/**
 * Warnings for a Work permission card, from a hand-written word scan of the command the agent asks to run. A warning
 * is a line on the card and blocks nothing, so the scan leans towards a false mark over a missed one. It never runs,
 * expands or resolves anything on disk: paths are judged as written. Text a shell will run as a command (`sh -c`,
 * `eval`, a heredoc fed to a shell, a command substitution) is scanned as a command too.
 */

import { posix } from 'node:path';

import type { CommandFlag } from '../types.js';
import { ALL_PROVIDER_KEY_ENV } from './env.js';

/** The order flags are listed in. */
export const COMMAND_FLAGS: readonly CommandFlag[] = [
  'elevated',
  'deletes',
  'outside-folder',
  'installs',
  'credentials',
  'prints-secrets',
  'provider-jobs',
  'unchecked',
];

/** Variables that hold a key in the Work agent's environment. */
export const KEY_VARIABLES: readonly string[] = [...ALL_PROVIDER_KEY_ENV, 'XAI_API_KEY'];

/** The folder of the Work session and the user's home: what "outside the work folder" is judged against. */
export interface WorkPlace {
  cwd: string;
  home: string;
}

interface Word {
  /** Quotes removed, escapes applied. */
  text: string;
  /** The characters that are not inside single quotes: where `$NAME` would expand. */
  live: string;
}

interface Part {
  words: Word[];
  /** Targets of `>`, `>>`, `&>`, `>|`. */
  writes: Word[];
  /** Heredoc bodies fed to this part. */
  heredocs: string[];
  /** Parts joined by `|` share a pipeline number. */
  pipeline: number;
  /** The head of the part whose word held this command substitution, or null at the top level. */
  parent: string | null;
  depth: number;
}

const MAX_DEPTH = 8;
/** The work one command's scan may do, nested text included; past it the rest is not read and the card says so. */
const STEP_BUDGET = 1_000_000;

interface Meter {
  steps: number;
  /** Some text a shell would run was left unread: past the step budget or the nesting depth. */
  unchecked: boolean;
}

function isSpace(c: string): boolean {
  return c === ' ' || c === '\t' || c === '\r' || c === '\f' || c === '\v';
}

function isIdent(c: string | undefined, first: boolean): boolean {
  if (c === undefined || c.length !== 1) return false;
  const code = c.charCodeAt(0);
  const letter = (code >= 0x41 && code <= 0x5a) || (code >= 0x61 && code <= 0x7a) || c === '_';
  return first ? letter : letter || (code >= 0x30 && code <= 0x39);
}

function isDigits(text: string): boolean {
  if (text === '') return false;
  for (const c of text) if (c < '0' || c > '9') return false;
  return true;
}

/** Index just after the `)` that closes the `(` at `open`, honouring quotes and nesting; the end when unclosed. */
function closeParen(s: string, open: number): number {
  let depth = 0;
  for (let i = open; i < s.length; i++) {
    const c = s[i];
    if (c === '\\') {
      i++;
    } else if (c === "'") {
      const end = s.indexOf("'", i + 1);
      i = end === -1 ? s.length : end;
    } else if (c === '"') {
      i++;
      while (i < s.length && s[i] !== '"') i += s[i] === '\\' ? 2 : 1;
    } else if (c === '(') {
      depth++;
    } else if (c === ')') {
      depth--;
      if (depth === 0) return i + 1;
    }
  }
  return s.length;
}

/** Index just after the back-tick that closes the one at `open`. */
function closeTick(s: string, open: number): number {
  for (let i = open + 1; i < s.length; i++) {
    if (s[i] === '\\') i++;
    else if (s[i] === '`') return i + 1;
  }
  return s.length;
}

/** The bodies of `$(…)`, `<(…)`, `>(…)` and back-tick substitutions in a word's live text. */
function substitutions(live: string): string[] {
  const out: string[] = [];
  for (let i = 0; i < live.length; i++) {
    const c = live[i];
    if ((c === '$' || c === '<' || c === '>') && live[i + 1] === '(') {
      const end = closeParen(live, i + 1);
      out.push(live.slice(i + 2, Math.max(i + 2, end - 1)));
      i = end - 1;
    } else if (c === '`') {
      const end = closeTick(live, i);
      out.push(live.slice(i + 1, Math.max(i + 1, end - 1)));
      i = end - 1;
    }
  }
  return out;
}

function isBlank(text: string): boolean {
  for (const c of text) if (!isSpace(c) && c !== '\n') return false;
  return true;
}

/**
 * Splits a command into simple parts at `;`, `&&`, `||`, `|`, `&`, new lines and parentheses, and appends them to
 * `out`, each followed by the parts of the text it runs as a command.
 */
function scan(
  command: string,
  parent: string | null,
  depth: number,
  pipelineStart: number,
  meter: Meter,
  out: Part[],
): void {
  meter.steps += command.length + 1;
  const parts: Part[] = [];
  let pipeline = pipelineStart;
  let part: Part = { words: [], writes: [], heredocs: [], pipeline, parent, depth };
  let text = '';
  let live = '';
  let inWord = false;
  /** 'write': the next word is a redirect target; 'skip': it is read from (`<`, `<<<`) or a heredoc delimiter. */
  let next: 'word' | 'write' | 'skip' | 'heredoc' = 'word';
  const heredocs: Array<{ delimiter: string; tabs: boolean; part: Part; body: string[] }> = [];
  let docAt = 0;
  let heredocTabs = false;

  const endWord = (): void => {
    if (!inWord) return;
    const word: Word = { text, live };
    if (next === 'write') part.writes.push(word);
    else if (next === 'heredoc') heredocs.push({ delimiter: text, tabs: heredocTabs, part, body: [] });
    else if (next !== 'skip') part.words.push(word);
    next = 'word';
    text = '';
    live = '';
    inWord = false;
  };
  const endPart = (piped: boolean): void => {
    endWord();
    next = 'word';
    if (part.words.length > 0 || part.writes.length > 0) parts.push(part);
    if (!piped) pipeline++;
    part = { words: [], writes: [], heredocs: [], pipeline, parent, depth };
  };
  const take = (raw: string): void => {
    text += raw;
    live += raw;
    inWord = true;
  };

  const n = command.length;
  let i = 0;
  while (i < n) {
    const c = command[i]!;
    const d = command[i + 1];
    if (c === '\\') {
      if (d === '\n') {
        i += 2;
        continue;
      }
      if (d !== undefined) take(d);
      i += 2;
      continue;
    }
    if (c === "'") {
      const end = command.indexOf("'", i + 1);
      const stop = end === -1 ? n : end;
      text += command.slice(i + 1, stop);
      inWord = true;
      i = stop + 1;
      continue;
    }
    if (c === '"') {
      inWord = true;
      i++;
      while (i < n && command[i] !== '"') {
        const q = command[i]!;
        if (q === '\\' && i + 1 < n) {
          take(command[i + 1]!);
          i += 2;
        } else if (q === '$' && command[i + 1] === '(') {
          const end = closeParen(command, i + 1);
          take(command.slice(i, end));
          i = end;
        } else if (q === '`') {
          const end = closeTick(command, i);
          take(command.slice(i, end));
          i = end;
        } else {
          take(q);
          i++;
        }
      }
      i++;
      continue;
    }
    if ((c === '$' || c === '<' || c === '>') && d === '(') {
      const end = closeParen(command, i + 1);
      take(command.slice(i, end));
      i = end;
      continue;
    }
    if (c === '`') {
      const end = closeTick(command, i);
      take(command.slice(i, end));
      i = end;
      continue;
    }
    if (c === '#' && !inWord) {
      while (i < n && command[i] !== '\n') i++;
      continue;
    }
    if (isSpace(c)) {
      endWord();
      i++;
      continue;
    }
    if (c === '\n') {
      endPart(false);
      i++;
      // A heredoc body belongs to the part it is fed to: read as a command only when that part is a shell.
      while (docAt < heredocs.length && i < n) {
        const doc = heredocs[docAt]!;
        const eol = command.indexOf('\n', i);
        const lineEnd = eol === -1 ? n : eol;
        let line = command.slice(i, lineEnd);
        if (doc.tabs) while (line.startsWith('\t')) line = line.slice(1);
        i = lineEnd + 1;
        if (line === doc.delimiter) {
          doc.part.heredocs.push(doc.body.join('\n'));
          docAt++;
        } else {
          doc.body.push(line);
        }
      }
      continue;
    }
    if (c === ';') {
      endPart(false);
      i += d === ';' ? 2 : 1;
      continue;
    }
    if (c === '&') {
      if (d === '&') {
        endPart(false);
        i += 2;
      } else if (d === '>') {
        endWord();
        next = 'write';
        i += command[i + 2] === '>' ? 3 : 2;
      } else {
        endPart(false);
        i++;
      }
      continue;
    }
    if (c === '|') {
      if (d === '|') {
        endPart(false);
        i += 2;
      } else {
        endPart(true);
        i += d === '&' ? 2 : 1;
      }
      continue;
    }
    if (c === '(' || c === ')' || c === '{' || c === '}') {
      if (inWord && (c === '{' || c === '}')) {
        take(c);
        i++;
        continue;
      }
      endPart(false);
      i++;
      continue;
    }
    if (c === '>') {
      // `2>` and `1>`: the digits name a descriptor, not a word.
      if (inWord && isDigits(text) && text === live) {
        text = '';
        live = '';
        inWord = false;
      } else {
        endWord();
      }
      let j = i + 1;
      if (command[j] === '>' || command[j] === '|') j++;
      if (command[j] === '&') {
        // `>&2`: a descriptor copy, not a file.
        j++;
        while (j < n && (isDigits(command[j]!) || command[j] === '-')) j++;
        i = j;
        continue;
      }
      next = 'write';
      i = j;
      continue;
    }
    if (c === '<') {
      if (inWord && isDigits(text) && text === live) {
        text = '';
        live = '';
        inWord = false;
      } else {
        endWord();
      }
      if (d === '<' && command[i + 2] === '<') {
        next = 'skip';
        i += 3;
      } else if (d === '<') {
        heredocTabs = command[i + 2] === '-';
        next = 'heredoc';
        i += heredocTabs ? 3 : 2;
      } else {
        next = 'skip';
        i += d === '&' ? 2 : 1;
      }
      continue;
    }
    take(c);
    i++;
  }
  endPart(false);
  for (let k = docAt; k < heredocs.length; k++) heredocs[k]!.part.heredocs.push(heredocs[k]!.body.join('\n'));

  meter.steps += parts.length;
  for (const p of parts) expand(p, meter, out, true);
}

/** Appends a part, then the parts of every text it runs as a command. */
function expand(p: Part, meter: Meter, out: Part[], withSubstitutions: boolean): void {
  out.push(p);
  const { head, args } = headOf(p);
  if (withSubstitutions) {
    for (const words of [p.words, p.writes]) {
      for (const w of words) {
        meter.steps += w.live.length;
        for (const inner of substitutions(w.live)) nested(inner, head, p, meter, out);
      }
    }
  }
  if (head === null) return;
  // Run as commands by the shell itself, so their output is not fed to it: no parent.
  for (const inner of shellText(head, args, p.heredocs)) nested(inner, null, p, meter, out);
  if (head !== 'find') return;
  for (const words of findCommands(args)) {
    if (p.depth >= MAX_DEPTH) {
      meter.unchecked = true;
      continue;
    }
    const pipeline = p.pipeline * 1_000 + 1;
    expand({ words, writes: [], heredocs: [], pipeline, parent: null, depth: p.depth + 1 }, meter, out, false);
  }
}

function nested(text: string, parent: string | null, p: Part, meter: Meter, out: Part[]): void {
  if (isBlank(text)) return;
  if (p.depth >= MAX_DEPTH || meter.steps + text.length > STEP_BUDGET) {
    meter.unchecked = true;
    return;
  }
  scan(text, parent, p.depth + 1, p.pipeline * 1_000 + 1, meter, out);
}

/** The commands `find` runs: the words after `-exec`, `-execdir`, `-ok` or `-okdir`, up to `;` or `{} +`. */
function findCommands(args: readonly Word[]): Word[][] {
  const out: Word[][] = [];
  for (let k = 0; k < args.length; k++) {
    const option = args[k]!.text;
    if (option !== '-exec' && option !== '-execdir' && option !== '-ok' && option !== '-okdir') continue;
    const words: Word[] = [];
    for (k++; k < args.length; k++) {
      const w = args[k]!;
      if (w.text === ';' || (w.text === '+' && words[words.length - 1]?.text === '{}')) break;
      words.push(w);
    }
    if (words.length > 0) out.push(words);
  }
  return out;
}

/** `-c`, or a cluster of short options that holds `c` (`-lc`, `-ec`). */
function isCommandOption(word: string): boolean {
  return word === '-c' || (isShortCluster(word) && word.includes('c'));
}

/** Text the part's command will run as shell commands: `sh -c TEXT`, `eval TEXT…`, a heredoc fed to a shell. */
function shellText(head: string, args: readonly Word[], heredocs: readonly string[]): string[] {
  if (head === 'eval') return args.length === 0 ? [] : [args.map((w) => w.text).join(' ')];
  if (!SHELLS.has(head)) return [];
  const out: string[] = [];
  const at = args.findIndex((w) => isCommandOption(w.text));
  if (at !== -1) {
    const text = args.slice(at + 1).find((w) => !w.text.startsWith('-') || w.text === '-');
    if (text !== undefined) out.push(text.text);
  } else {
    for (const body of heredocs) out.push(body);
  }
  return out;
}

function basename(word: string): string {
  const slash = word.lastIndexOf('/');
  return slash === -1 ? word : word.slice(slash + 1);
}

function isAssignment(word: string): boolean {
  const eq = word.indexOf('=');
  if (eq <= 0) return false;
  for (let i = 0; i < eq; i++) if (!isIdent(word[i], i === 0)) return false;
  return true;
}

const ELEVATING = new Set(['sudo', 'doas', 'pkexec', 'su']);
const PLAIN_WRAPPERS = new Set(['nohup', 'time', 'command', 'builtin', 'exec', 'caffeinate', 'stdbuf']);
/** Options of a wrapper that take the next word as their value. */
const WRAPPER_ARG_OPTIONS: Readonly<Record<string, readonly string[]>> = {
  sudo: ['-u', '-g', '-C', '-h', '-p', '-r', '-t', '-U', '-D', '-R', '-T'],
  doas: ['-u', '-C'],
  env: ['-u', '-C', '-S', '-P'],
  nice: ['-n'],
  exec: ['-a'],
  xargs: ['-I', '-n', '-P', '-L', '-s', '-d', '-E', '-a'],
  timeout: ['-s', '-k', '--signal', '--kill-after'],
};

/** Reserved words in front of a command: the word after them is the command. */
const LEADING_WORDS = new Set(['if', 'then', 'elif', 'else', 'do', 'while', 'until', '!', 'coproc']);
/** Reserved words followed by names or patterns, not a command. */
const NO_COMMAND_WORDS = new Set(['for', 'select', 'case', 'function']);

interface Head {
  head: string | null;
  args: Word[];
  elevated: boolean;
  /** `env` with no command after it: it prints the environment. */
  bareEnv: boolean;
}

/** The command a part runs, past assignments and wrappers (`sudo`, `env`, `nohup`, `xargs`, …). */
function headOf(part: Part): Head {
  const words = part.words;
  let elevated = false;
  let bareEnv = false;
  let reserved = true;
  let i = 0;
  const skipOptions = (wrapper: string): void => {
    const withArg = WRAPPER_ARG_OPTIONS[wrapper] ?? [];
    while (i < words.length && words[i]!.text.startsWith('-') && words[i]!.text.length > 1) {
      const option = words[i]!.text;
      i += withArg.includes(option) ? 2 : 1;
      if (option === '--') break;
    }
  };
  while (i < words.length) {
    const text = words[i]!.text;
    if (reserved && LEADING_WORDS.has(text)) {
      i++;
      continue;
    }
    if (reserved && NO_COMMAND_WORDS.has(text)) return { head: null, args: [], elevated, bareEnv };
    reserved = false;
    if (isAssignment(text)) {
      i++;
      continue;
    }
    const base = basename(text);
    if (ELEVATING.has(base)) {
      elevated = true;
      i++;
      if (base === 'su') return { head: null, args: words.slice(i), elevated, bareEnv };
      skipOptions(base);
      continue;
    }
    if (base === 'env') {
      i++;
      skipOptions('env');
      while (i < words.length && isAssignment(words[i]!.text)) i++;
      if (i >= words.length) bareEnv = true;
      continue;
    }
    if (PLAIN_WRAPPERS.has(base) || base === 'nice' || base === 'xargs') {
      i++;
      skipOptions(base);
      continue;
    }
    if (base === 'timeout') {
      i++;
      skipOptions('timeout');
      i++;
      continue;
    }
    return { head: base, args: words.slice(i + 1), elevated, bareEnv };
  }
  return { head: null, args: [], elevated, bareEnv };
}

/** `-abc` (a cluster of short options), not `--long` and not a lone `-`. */
function isShortCluster(word: string): boolean {
  return word.length > 1 && word[0] === '-' && word[1] !== '-';
}

function hasVariable(live: string, name: string): boolean {
  for (let at = live.indexOf('$'); at !== -1; at = live.indexOf('$', at + 1)) {
    const braced = live[at + 1] === '{';
    const start = at + (braced ? 2 : 1);
    if (live.startsWith(name, start) && !isIdent(live[start + name.length], false)) return true;
  }
  return false;
}

function namesKeyVariable(word: Word, viaDollar: boolean): boolean {
  return KEY_VARIABLES.some((name) => {
    if (viaDollar) return hasVariable(word.live, name);
    for (let at = word.text.indexOf(name); at !== -1; at = word.text.indexOf(name, at + 1)) {
      if (!isIdent(word.text[at - 1], false) && !isIdent(word.text[at + name.length], false)) return true;
    }
    return false;
  });
}

const SHELLS = new Set(['sh', 'bash', 'zsh', 'dash', 'ksh', 'fish', 'csh', 'tcsh']);
const INTERPRETERS = new Set(['python', 'python3', 'node', 'perl', 'ruby', 'deno', 'bun', 'php', 'osascript']);
const PACKAGE_MANAGERS = new Set(['npm', 'pnpm', 'yarn', 'bun']);
const DOWNLOADERS = new Set(['curl', 'wget']);

function words(args: readonly Word[]): string[] {
  return args.map((w) => w.text);
}

/** git's own options that take the next word as their value: `git -C <dir> push`. */
const GIT_ARG_OPTIONS = new Set([
  '-C',
  '-c',
  '--git-dir',
  '--work-tree',
  '--namespace',
  '--super-prefix',
  '--config-env',
  '--exec-path',
]);

/** The index of the first word that is not an option or an option's value: a subcommand such as `git push`. */
function subcommandAt(args: readonly Word[]): number {
  for (let k = 0; k < args.length; k++) {
    const w = args[k]!.text;
    if (GIT_ARG_OPTIONS.has(w)) k++;
    else if (!w.startsWith('-')) return k;
  }
  return -1;
}

function installs(head: string, args: readonly Word[]): boolean {
  const list = words(args);
  if (head === 'npx' || head === 'bunx' || head === 'pnpx') return true;
  if (!PACKAGE_MANAGERS.has(head)) return false;
  if (list.includes('dlx') || (head === 'bun' && list[0] === 'x')) return true;
  const global = list.includes('-g') || list.includes('--global') || list.includes('global');
  return global && (list.includes('i') || list.includes('install') || list.includes('add'));
}

/** An interpreter that reads its program from standard input: no script, no `-c` / `-e` / `-m`. */
function readsStdin(head: string, args: readonly Word[]): boolean {
  if (SHELLS.has(head)) return true;
  if (!INTERPRETERS.has(head)) return false;
  return args.every((w) => w.text === '-' || (w.text.startsWith('-') && !['-c', '-e', '-m', '-p'].includes(w.text)));
}

function deletes(head: string, args: readonly Word[]): boolean {
  const list = words(args);
  if (['rm', 'unlink', 'shred', 'srm', 'truncate'].includes(head)) return true;
  if (head.startsWith('mkfs')) return true;
  if (head === 'dd') return list.some((w) => w.startsWith('of='));
  if (head === 'find') {
    if (list.includes('-delete')) return true;
    return list.some(
      (w, k) => (w === '-exec' || w === '-execdir') && ['rm', 'shred'].includes(basename(list[k + 1] ?? '')),
    );
  }
  if (head === 'rsync') return list.some((w) => w.startsWith('--delete') || w.startsWith('--remove-source'));
  if (head === 'diskutil') return list.some((w) => w.startsWith('erase') || w === 'zeroDisk' || w === 'secureErase');
  if (head !== 'git') return false;
  const at = subcommandAt(args);
  const sub = at === -1 ? null : list[at]!;
  const rest = list.slice(at + 1);
  switch (sub) {
    case 'clean':
    case 'restore':
      return true;
    case 'reset':
      return rest.includes('--hard') || rest.includes('--merge') || rest.includes('--keep');
    case 'checkout':
      return rest.includes('--') || rest.includes('.') || rest.includes('-f') || rest.includes('--force');
    case 'push':
      return rest.some(
        (w) =>
          w === '-f' ||
          w === '-d' ||
          w.startsWith('--force') ||
          w === '--delete' ||
          w === '--mirror' ||
          w === '--prune' ||
          w.startsWith('+') ||
          w.startsWith(':'),
      );
    case 'branch':
      return rest.includes('-D') || ((rest.includes('-d') || rest.includes('--delete')) && rest.includes('--force'));
    case 'stash':
      return rest[0] === 'drop' || rest[0] === 'clear';
    default:
      return false;
  }
}

function credentials(head: string | null, all: readonly Word[]): boolean {
  const list = words(all);
  if (list.some((w) => w === '--api-key' || w.startsWith('--api-key=') || w === '-k')) return true;
  const firecrawl =
    head === 'firecrawl' || list.includes('firecrawl-cli') || list.some((w) => w.startsWith('firecrawl-cli@'));
  return firecrawl && list.some((w) => w === 'login' || w === 'init' || w === 'setup');
}

function printsSecrets(head: string, args: readonly Word[]): boolean {
  const list = words(args);
  if (head === 'printenv') return true;
  if (head === 'set') return list.length === 0;
  if (head === 'export' || head === 'declare' || head === 'typeset') {
    return list.every((w) => w === '-p' || w === '-x' || w === '-px' || w === '-xp');
  }
  if (head === 'curl') {
    return list.some(
      (w) =>
        w === '--verbose' ||
        w === '--trace' ||
        w === '--trace-ascii' ||
        w.startsWith('--trace=') ||
        w.startsWith('--trace-ascii=') ||
        (isShortCluster(w) && w.includes('v')),
    );
  }
  if (head === 'wget') return list.some((w) => w === '--debug' || w === '-d');
  if (head === 'ps') {
    return list.some(
      (w, k) => (isShortCluster(w) && w.includes('E')) || (k === 0 && !w.startsWith('-') && w.includes('e')),
    );
  }
  if (head === 'echo' || head === 'printf' || head === 'print') return args.some((w) => namesKeyVariable(w, true));
  if (INTERPRETERS.has(head)) return args.some((w) => namesKeyVariable(w, false));
  return false;
}

/** An interpreter fed a heredoc program that names a key variable. */
function heredocNamesKey(head: string, heredocs: readonly string[]): boolean {
  if (!INTERPRETERS.has(head)) return false;
  return heredocs.some((body) => namesKeyVariable({ text: body, live: body }, false));
}

function providerJobs(head: string | null, all: readonly Word[]): boolean {
  const list = words(all);
  const firecrawl =
    head === 'firecrawl' || list.includes('firecrawl-cli') || list.some((w) => w.startsWith('firecrawl-cli@'));
  if (
    firecrawl &&
    list.some((w) => w === 'monitor' || w === 'monitors' || w === 'feedback' || w.endsWith('-feedback'))
  ) {
    return true;
  }
  return list.some((w) => w.includes('firecrawl.dev/') && (w.includes('/feedback') || w.includes('/monitor')));
}

// ── outside the work folder ────────────────────────────────────────────────────────────────────────────────────────

/** Heads whose every path argument is written (or removed). */
const WRITES_EVERY_ARG = new Set([
  'rm',
  'rmdir',
  'mkdir',
  'touch',
  'unlink',
  'shred',
  'srm',
  'truncate',
  'chmod',
  'chown',
  'chgrp',
  'tee',
  'mv',
]);
/** Heads whose last path argument is the one written. */
const WRITES_LAST_ARG = new Set(['cp', 'rsync', 'install', 'scp', 'ln', 'ditto']);
const SAFE_TARGETS = new Set(['/dev/null', '/dev/stdout', '/dev/stderr', '/dev/tty']);

/** `/tmp`, `/var` and `/etc` are links into `/private` on macOS, and the work folder is a resolved path. */
function aliases(path: string): string[] {
  for (const top of ['/tmp', '/var', '/etc']) {
    if (path === top || path.startsWith(`${top}/`)) return [path, `/private${path}`];
  }
  return [path];
}

function inside(path: string, folder: string): boolean {
  return aliases(path).some((p) => p === folder || p.startsWith(folder.endsWith('/') ? folder : `${folder}/`));
}

const OTHER_HOME = Symbol('another user home');

/** A path as written, made absolute; {@link OTHER_HOME} for `~user`; null when it cannot be known. */
function resolvePath(word: Word, dir: string | null, home: string): string | typeof OTHER_HOME | null {
  let text = word.text;
  if (text === '' || text.includes(':')) return null;
  if (word.live.includes('$')) {
    const prefix = ['${HOME}', '$HOME'].find((p) => word.live.startsWith(p));
    if (prefix === undefined) return null;
    const rest = text.slice(prefix.length);
    if (rest.includes('$') || word.live.slice(prefix.length).includes('$')) return null;
    return posix.normalize(posix.join(home, rest));
  }
  if (text.startsWith('~')) {
    if (text !== '~' && !text.startsWith('~/')) return OTHER_HOME;
    text = posix.join(home, text.slice(1));
  }
  if (posix.isAbsolute(text)) return posix.normalize(text);
  if (dir === null) return null;
  return posix.normalize(posix.join(dir, text));
}

function writtenOutside(word: Word, dir: string | null, place: WorkPlace): boolean {
  const path = resolvePath(word, dir, place.home);
  if (path === null) return false;
  if (path === OTHER_HOME) return true;
  if (SAFE_TARGETS.has(path) || path.startsWith('/dev/fd/')) return false;
  return !inside(path, place.cwd);
}

function pathArgs(args: readonly Word[]): Word[] {
  const out: Word[] = [];
  let options = true;
  for (const w of args) {
    if (options && w.text === '--') {
      options = false;
      continue;
    }
    if (options && w.text.startsWith('-') && w.text.length > 1) continue;
    out.push(w);
  }
  return out;
}

/** The words a part writes to: redirect targets and the path arguments of the writing commands. */
function writeTargets(head: string | null, args: readonly Word[], part: Part): Word[] {
  const targets = [...part.writes];
  if (head === null) return targets;
  const paths = pathArgs(args);
  if (WRITES_EVERY_ARG.has(head)) for (const w of paths) targets.push(w);
  else if (WRITES_LAST_ARG.has(head) && paths.length > 1) targets.push(paths[paths.length - 1]!);
  else if (
    (head === 'sed' || head === 'gsed' || head === 'perl') &&
    args.some((w) => w.text.startsWith('-i') || w.text === '--in-place')
  ) {
    for (let k = head === 'perl' ? 0 : 1; k < paths.length; k++) targets.push(paths[k]!);
  } else if (head === 'dd') {
    for (const w of args) if (w.text.startsWith('of=')) targets.push({ text: w.text.slice(3), live: w.live.slice(3) });
  } else if (head === 'curl' || head === 'wget') {
    const names = head === 'curl' ? ['-o', '--output'] : ['-O', '--output-document', '-P', '--directory-prefix'];
    args.forEach((w, k) => {
      if (names.includes(w.text) && args[k + 1] !== undefined) targets.push(args[k + 1]!);
      for (const name of names) {
        if (name.startsWith('--') && w.text.startsWith(`${name}=`)) {
          targets.push({ text: w.text.slice(name.length + 1), live: w.live.slice(name.length + 1) });
        }
      }
    });
  }
  return targets;
}

// ── entry points ───────────────────────────────────────────────────────────────────────────────────────────────────

/** The warnings for a shell command, in {@link COMMAND_FLAGS} order. `place` enables the "outside the work folder" check. */
export function commandFlags(command: string, place?: WorkPlace): CommandFlag[] {
  return commandCheck(command, place).flags;
}

/** {@link commandFlags} with the number of steps the check took. */
export function commandCheck(command: string, place?: WorkPlace): { flags: CommandFlag[]; steps: number } {
  const found = new Set<CommandFlag>();
  const meter: Meter = { steps: 0, unchecked: false };
  const parts: Part[] = [];
  scan(command, null, 0, 0, meter, parts);
  if (meter.unchecked) found.add('unchecked');
  /** Pipelines that ran a downloader in an earlier part, and whether one ran at the top level. */
  const fetchedPipelines = new Set<number>();
  let fetchedTop = false;
  let dir: string | null = place?.cwd ?? null;
  /** A `cd` takes effect after its own redirects and substitutions, at the next top-level part. */
  let nextDir: string | null | undefined;
  for (const part of parts) {
    if (part.depth === 0 && nextDir !== undefined) {
      dir = nextDir;
      nextDir = undefined;
    }
    const { head, args, elevated, bareEnv } = headOf(part);
    const all = part.words;
    meter.steps += 1 + all.length;
    const piped = fetchedPipelines.has(part.pipeline);
    const fetched = fetchedTop;
    if (head !== null && DOWNLOADERS.has(head)) {
      fetchedPipelines.add(part.pipeline);
      if (part.depth === 0) fetchedTop = true;
    }
    if (elevated) found.add('elevated');
    if (bareEnv) found.add('prints-secrets');
    if (credentials(head, all)) found.add('credentials');
    if (providerJobs(head, all)) found.add('provider-jobs');
    if (head !== null) {
      if (installs(head, args)) found.add('installs');
      if (deletes(head, args)) found.add('deletes');
      if (printsSecrets(head, args) || heredocNamesKey(head, part.heredocs)) found.add('prints-secrets');
      if (piped && readsStdin(head, args)) found.add('installs');
      if (DOWNLOADERS.has(head) && part.parent !== null && (SHELLS.has(part.parent) || INTERPRETERS.has(part.parent))) {
        found.add('installs');
      }
      if (fetched && SHELLS.has(head) && part.depth === 0) found.add('installs');
    }
    if (place === undefined) continue;
    if (writeTargets(head, args, part).some((w) => writtenOutside(w, dir, place))) found.add('outside-folder');
    if (part.depth === 0 && (head === 'cd' || head === 'pushd')) {
      const target = pathArgs(args)[0];
      meter.steps += (dir?.length ?? 0) + (target?.text.length ?? 0);
      if (meter.steps > STEP_BUDGET) {
        meter.unchecked = true;
        nextDir = null;
      } else if (target === undefined) nextDir = place.home;
      else if (target.text === '-') nextDir = null;
      else {
        const resolved = resolvePath(target, dir, place.home);
        nextDir = resolved === OTHER_HOME ? null : resolved;
      }
    }
  }
  if (meter.unchecked) found.add('unchecked');
  return { flags: COMMAND_FLAGS.filter((flag) => found.has(flag)), steps: meter.steps };
}

/** Rust's `char::is_whitespace`, which the agent trims a path with. */
function isAgentSpace(code: number): boolean {
  return (
    (code >= 0x09 && code <= 0x0d) ||
    code === 0x20 ||
    code === 0x85 ||
    code === 0xa0 ||
    code === 0x1680 ||
    (code >= 0x2000 && code <= 0x200a) ||
    code === 0x2028 ||
    code === 0x2029 ||
    code === 0x202f ||
    code === 0x205f ||
    code === 0x3000
  );
}

function trimAgent(text: string, start: boolean, end: boolean): string {
  let from = 0;
  let to = text.length;
  while (start && from < to && isAgentSpace(text.charCodeAt(from))) from++;
  while (end && to > from && isAgentSpace(text.charCodeAt(to - 1))) to--;
  return text.slice(from, to);
}

function isQuote(c: string | undefined): boolean {
  return c === '"' || c === "'";
}

/** Path segments as a Rust `Path` compares them: empty and `.` segments dropped. */
function segments(path: string): string[] {
  return path.split('/').filter((s) => s !== '' && s !== '.');
}

/**
 * The path the agent writes for an edit, as text: trimmed, surrounding quotes removed, `~` and `~/` expanded, a
 * relative path that repeats the work folder without its leading slash taken as that folder, anything else relative
 * joined to the work folder. `..` is left for the disk to follow.
 */
export function agentPath(path: string, place: WorkPlace): string {
  const trimmed = trimAgent(path, true, true);
  const wrapped = trimmed.length >= 2 && isQuote(trimmed[0]) && isQuote(trimmed[trimmed.length - 1]);
  let from = 0;
  let to = trimmed.length;
  while (from < to && isQuote(trimmed[from])) from++;
  while (to > from && isQuote(trimmed[to - 1])) to--;
  let text = trimAgent(trimmed.slice(from, to), true, true);
  if (wrapped) {
    for (;;) {
      const tail = text.slice(-2);
      if (tail !== '\\n' && tail !== '\\r' && tail !== '\\t') break;
      text = trimAgent(text.slice(0, -2), false, true);
    }
  }
  if (text === '~') text = place.home;
  else if (text.startsWith('~/')) text = `${place.home}${text.slice(1)}`;
  if (text.startsWith('/')) return text;
  if (text !== '') {
    const base = segments(place.cwd);
    const asAbsolute = segments(text);
    if (asAbsolute.length >= base.length && base.every((s, k) => asAbsolute[k] === s)) {
      const suffix = asAbsolute.slice(base.length);
      return suffix.length === 0 ? place.cwd : `${place.cwd}/${suffix.join('/')}`;
    }
  }
  return text === '' ? place.cwd : `${place.cwd}/${text}`;
}

/** Whether an absolute path lies inside the work folder; `/dev` is never inside. */
export function landsInside(path: string, place: WorkPlace): boolean {
  if (path === '/dev' || path.startsWith('/dev/')) return false;
  return inside(path, place.cwd);
}

/** Whether an absolute path lies outside the work folder, for the card's flag: the streams in `/dev` do not count. */
export function landsOutside(path: string, place: WorkPlace): boolean {
  if (SAFE_TARGETS.has(path) || path.startsWith('/dev/fd/')) return false;
  return !inside(path, place.cwd);
}

/** The warnings for a file the agent asks to edit or write, judged by its path text: whether it lies outside the work folder. */
export function pathFlags(path: string, place: WorkPlace): CommandFlag[] {
  return landsOutside(posix.normalize(agentPath(path, place)), place) ? ['outside-folder'] : [];
}
