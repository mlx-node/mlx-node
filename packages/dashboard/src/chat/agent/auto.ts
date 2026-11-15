/**
 * Auto in Work: whether the app answers a permission request `allow-once` itself, or the card opens and
 * says why it still asks. The agent is told nothing; it asks as always, and only the app's answer changes. A call
 * passes only when the scan models all of it ({@link autoScan}), and every path it names is placed on disk.
 */

import type { Stats } from 'node:fs';
import { lstat, readdir, readFile, readlink, stat } from 'node:fs/promises';
import { basename, dirname, join, sep } from 'node:path';

import type { PermissionRequest } from '../acp/messages.js';
import type { AutoAsk, AutoAskCause, CommandFlag, PermissionInteraction, TempAsk } from '../types.js';
import { landingPath } from '../work-folder.js';
import { autoScan, foldName, hasNonAscii, protectedPart, type AutoScan, type PathCheck } from './auto-scan.js';
import { agentPath, commandFlags, landsOutside, type WorkPlace } from './command-flags.js';
import { requestTarget } from './permissions.js';
import { ALLOW_ONCE_OPTION_ID } from './profiles.js';
import { below, rootOf, rootText, tempRoots, typedPath } from './roots.js';

/** Calls Auto may allow in one answer; then each call asks. An allow by the user starts the count again. */
export const AUTO_PER_TURN_MAX = 100;

/** What the disk says about the paths of one request, read before the decision. */
export interface AutoFacts {
  /** `bash`: the scan of the command; null when the request has no command. */
  scan: AutoScan | null;
  /** The command's flags: the same as the card's. */
  flags: CommandFlag[];
  /** `bash`: every path was placed on disk. */
  placed: boolean;
  /** `bash`: a target lands outside the folder. */
  outside: boolean;
  /** A path or the edit's landing has a protected part below the folder. */
  protectedPath: boolean;
  /** A target or the edited file is a file with more than one hard link. */
  links: boolean;
  /** `bash` that runs git: the folder holds `.git`. */
  repository: boolean;
  /** `edit`: where the file lands on disk; null when unknown. */
  landing: string | null;
  /** `edit`: the landing is outside the folder and every root. */
  landingOutside: boolean;
  /** The first place in a temp root that a unit rule refuses (or an edit path with `..` there), as typed. */
  temp: { path: string; cause: AutoAskCause } | null;
  /** The first disk check that did not place, as typed: the Git repository, the global Git config, or the place. */
  asked: { path: string; cause: AutoAskCause } | null;
  /** A real place of the call is in a temp root. */
  touched: boolean;
  /** The tops of the temp roots the call can reach: wide on purpose (a project tool, or a call not read in full, reaches all). */
  reach: string[];
}

export interface AutoInput {
  /** The conversation is in Auto, read after every wait of the check. */
  auto: boolean;
  /** A window holds the host stream. */
  watched: boolean;
  request: PermissionRequest;
  /** The card as it would open. */
  interaction: PermissionInteraction;
  /** This start's guard hook name; null in Chat. */
  guardHook: string | null;
  /** A helper agent may work in another folder in this session. */
  elsewhere: boolean;
  facts: AutoFacts;
  /** Calls Auto allowed in this turn since the user's last allow. */
  allowed: number;
}

/** The part of a request that asks; `part` is raw text the host redacts before the card shows it. */
export type AutoAskDetail = NonNullable<PermissionInteraction['autoAskDetail']>;

export type AutoDecision = { allow: true } | { allow: false; why: AutoAsk | null; detail?: AutoAskDetail };

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

/** A helper start that can work in another folder: `spawn_subagent` with `cwd` or `isolation`, or any `workflow`. */
export function helperStart(req: PermissionRequest): boolean {
  const name = req.tool?.name ?? null;
  if (name === 'workflow') return true;
  if (name !== 'spawn_subagent') return false;
  const raw = req.rawInput;
  return !isPlainObject(raw) || Object.hasOwn(raw, 'cwd') || Object.hasOwn(raw, 'isolation');
}

function isDigitText(text: string): boolean {
  if (text === '') return false;
  for (const c of text) if (c < '0' || c > '9') return false;
  return true;
}

/** `is_background` other than false, or `block_until_ms` that is not a positive number: the command starts in the background. */
export function backgroundInput(raw: unknown): boolean {
  if (!isPlainObject(raw)) return false;
  if (Object.hasOwn(raw, 'is_background') && raw.is_background !== false && raw.is_background !== 'false') return true;
  if (!Object.hasOwn(raw, 'block_until_ms') || raw.block_until_ms === undefined || raw.block_until_ms === null) {
    return false;
  }
  const wait = raw.block_until_ms;
  if (typeof wait === 'number') return !(wait > 0);
  if (typeof wait === 'string' && isDigitText(wait)) return Number(wait) <= 0;
  return true;
}

/**
 * A call that can go on after it reports its end: a monitor, a helper, a workflow, a schedule, or a command started
 * in the background (by its input, or by a lone `&` in its text). Text only; it never changes an answer.
 */
export function staysBehind(req: PermissionRequest): boolean {
  const name = req.tool?.name ?? null;
  if (name === 'monitor' || name === 'spawn_subagent' || name === 'workflow' || name === 'scheduler_create') {
    return true;
  }
  if (name !== 'run_terminal_command') return false;
  if (backgroundInput(req.rawInput)) return true;
  const raw = req.rawInput;
  const command = isPlainObject(raw) && typeof raw.command === 'string' ? raw.command : '';
  return loneAmpersand(command);
}

/** An `&` with no `&` and no `>` right before or after it. Quotes are not read: a quoted one counts too. */
export function loneAmpersand(text: string): boolean {
  for (let i = 0; i < text.length; i++) {
    if (text[i] !== '&') continue;
    const before = i > 0 ? text[i - 1] : '';
    const after = i + 1 < text.length ? text[i + 1] : '';
    if (before !== '&' && before !== '>' && after !== '&' && after !== '>') return true;
  }
  return false;
}

/**
 * The decision for one request. First matching row wins; outside Auto every request is a card with no reason.
 * Any error inside asks: it can never answer allow after a throw.
 */
export function decideAuto(input: AutoInput): AutoDecision {
  try {
    return decide(input);
  } catch {
    return { allow: false, why: input.auto === true ? 'unread' : null };
  }
}

function decide(input: AutoInput): AutoDecision {
  if (!input.auto) return { allow: false, why: null };
  const ask = (why: AutoAsk, detail?: AutoAskDetail): AutoDecision =>
    detail === undefined ? { allow: false, why } : { allow: false, why, detail };
  const at = (found: { path: string; cause: AutoAskCause } | null): AutoAskDetail | undefined =>
    found === null ? undefined : { part: found.path, more: 0, cause: found.cause };
  const { request: req, interaction, facts } = input;
  if (!input.watched) return ask('no-window');
  if (!req.options.some((o) => o.optionId === ALLOW_ONCE_OPTION_ID && o.kind === 'allow_once')) {
    return ask('not-covered');
  }
  const access = interaction.access;
  const hook = req.hookAsk ?? null;
  const guarded =
    hook !== null && input.guardHook !== null && hook.hookName === input.guardHook && hook.reason === null;
  if ((access === 'bash' && !guarded) || (hook !== null && !guarded)) return ask('hook');
  if (helperStart(req)) return ask('helper-start');
  if (access !== 'bash' && access !== 'edit') return ask('not-covered');
  if (input.elsewhere) return ask('helper');
  if (access === 'bash') {
    const scan = facts.scan;
    if (interaction.command === null || scan === null || facts.flags.includes('unchecked') || scan.unread) {
      return scan?.redirect === true ? ask('unread', { part: '2>', more: 0, cause: 'redirect' }) : ask('unread');
    }
    if (facts.flags.length > 0) return ask('risky');
    if (scan.keyName) return ask('key');
    if (scan.protectedName || facts.protectedPath) return ask('protected');
    if (scan.unknown) {
      const part = scan.asked[0];
      return ask('not-known', part === undefined ? undefined : { part, more: scan.askedCount - 1, cause: null });
    }
    if (scan.git && !facts.repository) return ask('not-known', at(facts.asked));
    if (scan.background || backgroundInput(req.rawInput)) return ask('background');
    if (facts.temp !== null) return ask('temp', at(facts.temp));
    if (scan.targets === null || scan.checks === null || !facts.placed || facts.outside || facts.links) {
      return ask('unknown-target', at(facts.asked));
    }
  } else {
    if (facts.protectedPath) return ask('protected');
    if (facts.temp !== null) return ask('temp', at(facts.temp));
    if (facts.landing === null || facts.links) return ask('unknown-landing');
    if (facts.landingOutside || facts.flags.length > 0) return ask('risky');
  }
  if (input.allowed >= AUTO_PER_TURN_MAX) return ask('many');
  return { allow: true };
}

/** Paths a call only reads: every other check (a write, a copy or move, a project tool, a script) can change the folder. */
const READ_CHECKS: ReadonlySet<PathCheck['kind']> = new Set(['read', 'glob', 'cd', 'absent', 'remote', 'git-config']);

/**
 * Whether a call may change the folder's files, links or names while it runs: every edit, every command the scan did
 * not read in full, and every command with a path check that is not a read.
 */
export function changesFolder(facts: AutoFacts, access: PermissionInteraction['access']): boolean {
  if (access !== 'bash') return true;
  const scan = facts.scan;
  if (scan === null || scan.unread || scan.changes || scan.checks === null) return true;
  return scan.checks.some((c) => !READ_CHECKS.has(c.kind));
}

/** The parts of `path` below the top of its root, or all of its parts when it is in no root. */
function partsBelow(path: string, place: WorkPlace): string[] {
  const root = rootOf(path, place);
  if (root !== null) return path === root.top ? [] : path.slice(root.top.length + 1).split('/');
  const base = place.cwd.endsWith('/') ? place.cwd : `${place.cwd}/`;
  if (path.startsWith(`/private${base}`)) return path.slice(base.length + 8).split('/');
  return path.split('/');
}

/** Whether a path has a protected part (folded as the disk compares names): below its root's top, or anywhere outside. */
export function protectedLanding(path: string, place: WorkPlace): boolean {
  for (const part of partsBelow(path, place)) if (part !== '' && protectedPart(part)) return true;
  return false;
}

function missing(error: unknown): boolean {
  const code = (error as NodeJS.ErrnoException | null)?.code;
  return code === 'ENOENT' || code === 'ENOTDIR';
}

/**
 * The disk reads of a decision: null only when nothing is there (ENOENT, ENOTDIR); every other error throws, and the
 * caller's catch asks. An error read as "absent" or "not a link" would allow what the disk never confirmed.
 */
async function lstatOrNull(path: string): Promise<Stats | null> {
  try {
    return await lstat(path);
  } catch (error) {
    if (missing(error)) return null;
    throw error;
  }
}

async function statOrNull(path: string): Promise<Stats | null> {
  try {
    return await stat(path);
  } catch (error) {
    if (missing(error)) return null;
    throw error;
  }
}

/** A file with more than one hard link: a write through it changes the other names too. */
async function linked(path: string): Promise<boolean> {
  const info = await lstatOrNull(path);
  return info !== null && info.isFile() && info.nlink > 1;
}

async function exists(path: string): Promise<boolean> {
  return (await lstatOrNull(path)) !== null;
}

/** A path that exists and is neither a file nor a folder (a device, a FIFO, a socket). */
async function special(path: string): Promise<boolean> {
  const info = await lstatOrNull(path);
  return info !== null && !info.isFile() && !info.isDirectory();
}

async function isFolder(path: string): Promise<boolean> {
  return (await statOrNull(path))?.isDirectory() === true;
}

/** Every folder from `base` down through `parts` exists and is not a link. */
async function realFolders(base: string, parts: readonly string[]): Promise<boolean> {
  let at = base;
  for (const part of parts) {
    at = `${at}/${part}`;
    const info = await lstatOrNull(at);
    if (info === null || info.isSymbolicLink() || !info.isDirectory()) return false;
  }
  return true;
}

/**
 * A path a project tool opens is placed: in a root by text, every part below the root's top an existing real folder
 * (no link), the last one a regular file or a real folder. The top of a temp root never is.
 */
async function plainEntry(path: string, place: WorkPlace): Promise<boolean> {
  const root = rootOf(path, place);
  if (root === null) return false;
  if (path === root.top) return root.kind === 'work';
  const parts = path.slice(root.top.length + 1).split('/');
  if (parts.some((p) => p === '' || p === '.' || p === '..')) return false;
  if (!(await realFolders(root.top, parts.slice(0, -1)))) return false;
  const last = await lstatOrNull(path);
  return last !== null && (last.isFile() || last.isDirectory());
}

function under(path: string, root: string): boolean {
  return path === root || path.startsWith(root.endsWith('/') ? root : `${root}/`);
}

const NO_FACTS: AutoFacts = {
  scan: null,
  flags: [],
  placed: false,
  outside: false,
  protectedPath: false,
  links: false,
  repository: false,
  landing: null,
  landingOutside: false,
  temp: null,
  asked: null,
  touched: false,
  reach: [],
};

interface Moved {
  part: number;
  /** Folded ({@link foldPath}). */
  text: string;
  landing: string | null;
}

/** A path with each part folded as the disk compares names: a later spelling in another case or form still matches. */
function foldPath(path: string): string {
  return path
    .split('/')
    .map((p) => foldName(p))
    .join('/');
}

/** `path` (or where it lands) is equal to or under a path an earlier part copies or moves. */
function underMoved(path: string, lands: string | null, part: number, moved: readonly Moved[]): boolean {
  const text = foldPath(path);
  const landing = lands === null ? null : foldPath(lands);
  for (const m of moved) {
    if (m.part >= part) continue;
    if (under(text, m.text)) return true;
    if (m.landing !== null && landing !== null && under(landing, m.landing)) return true;
  }
  return false;
}

/**
 * Places one written path: in a root, not protected, one link, not a device; not under an earlier copy or move. `mode`:
 * the call changes its mode, which asks when its text or landing is in a temp root, after the unit rules.
 */
async function placeWrite(
  path: string,
  part: number,
  moved: readonly Moved[],
  place: WorkPlace,
  facts: AutoFacts,
  mode = false,
) {
  const lands = await landingPath(path);
  if (lands === null) {
    notPlaced(facts, path, place);
    return;
  }
  if (underMoved(path, lands, part, moved)) facts.placed = false;
  if (rootOf(lands, place) === null) {
    facts.outside = true;
    note(facts, path, 'place', place);
  }
  if (protectedLanding(lands, place)) facts.protectedPath = true;
  if (await linked(lands)) facts.links = true;
  if (await special(lands)) facts.placed = false;
  const held = await unitRules(path, lands, place, facts);
  if (held && mode && (inTempText(path, place) || inTempReal(lands, place))) refuse(facts, path, 'mode', place);
}

// ── the temp roots ─────────────────────────────────────────────────────────────────────────────────────────────────

/** Links one walk may follow. */
const WALK_LINKS_MAX = 8;
/** Write bits for group and other. */
const OTHERS_WRITE = 0o022;

function inTempText(path: string, place: WorkPlace): boolean {
  return rootText(path, place)?.root.kind === 'temp';
}

function inTempReal(path: string, place: WorkPlace): boolean {
  return rootOf(path, place)?.kind === 'temp';
}

/** The first cause of a refused place, as typed. */
function refuse(facts: AutoFacts, path: string, cause: AutoAskCause, place: WorkPlace): void {
  facts.temp ??= { path: typedPath(path, place), cause };
}

/** The first disk check that did not place, as typed. */
function note(facts: AutoFacts, path: string, cause: AutoAskCause, place: WorkPlace): void {
  facts.asked ??= { path: typedPath(path, place), cause };
}

function notPlaced(facts: AutoFacts, path: string, place: WorkPlace): void {
  facts.placed = false;
  note(facts, path, 'place', place);
}

/**
 * A first-level name another program keeps for itself: a dot name, `com.apple.`, or a piece equal to the uid or login.
 * Compared folded: a case-blind disk opens the same folder by any case.
 */
export function privateName(given: string, uid: number, login: string | undefined): boolean {
  const name = foldName(given);
  if (name.startsWith('.') || name.startsWith('com.apple.')) return true;
  const lower = login === undefined || login === '' ? null : login.toLowerCase();
  let piece = '';
  for (let k = 0; k <= name.length; k++) {
    const c = k < name.length ? name.charAt(k) : '';
    const code = c === '' ? -1 : c.charCodeAt(0);
    const alnum = (code >= 0x30 && code <= 0x39) || (code >= 0x41 && code <= 0x5a) || (code >= 0x61 && code <= 0x7a);
    if (alnum) {
      piece += c;
      continue;
    }
    if (piece !== '' && (piece === String(uid) || (lower !== null && piece.toLowerCase() === lower))) return true;
    piece = '';
  }
  return false;
}

function uidOf(place: WorkPlace): number {
  return place.uid ?? process.getuid?.() ?? -1;
}

/** `..` drops one part, `.` and empty parts none; null when it climbs above `/`. */
function collapse(path: string): string | null {
  const kept: string[] = [];
  for (const part of path.split('/')) {
    if (part === '' || part === '.') continue;
    if (part === '..') {
      if (kept.length === 0) return null;
      kept.pop();
    } else kept.push(part);
  }
  return `/${kept.join('/')}`;
}

/**
 * Walks an absolute path as written (no `..` part), from the top of the root its text names, one part at a time with
 * `lstat`: every link is read and followed (its target placed in a root again), and every part under a temp root (a
 * link and the folder it sits in too) must be yours (U3), in a folder no one else can write (U4), a real folder, a link
 * or the last part a regular file (U5), on the root's disk (U6); a first-level name of a temp root must not look
 * private (U2). The end must not be the top of a temp root (U1) nor hold or lie in app data (U7). `place`: the walk
 * cannot place the path in a root. A missing part ends the walk: the call makes the rest.
 */
async function tempWalk(text: string, place: WorkPlace): Promise<{ cause: TempAsk | 'place' | null; end: string }> {
  const start = rootText(text, place);
  if (start === null) return { cause: 'place', end: text };
  const uid = uidOf(place);
  const devices = new Map<string, number>();
  const device = async (top: string): Promise<number> => {
    let dev = devices.get(top);
    if (dev === undefined) {
      dev = (await lstat(top)).dev;
      devices.set(top, dev);
    }
    return dev;
  };
  let folder = start.root.top;
  let rest = start.path
    .slice(folder.length)
    .split('/')
    .filter((p) => p !== '');
  if (rest.includes('..')) return { cause: 'place', end: text };
  let links = 0;
  let end = folder;
  for (;;) {
    const name = rest.shift();
    if (name === undefined) {
      end = folder;
      break;
    }
    const next = `${folder === '/' ? '' : folder}/${name}`;
    let info: Stats | null;
    try {
      info = await lstat(next);
    } catch (error) {
      if (!missing(error)) return { cause: 'unreadable', end: next };
      info = null;
    }
    const root = rootOf(next, place);
    const temp = root !== null && root.kind === 'temp';
    if (temp && dirname(next) === root.top && privateName(name, uid, place.login)) {
      return { cause: 'private', end: next };
    }
    if (info === null) {
      end = [next, ...rest].join('/');
      break;
    }
    if (temp) {
      if (info.uid !== uid) return { cause: 'owner', end: next };
      if (info.dev !== (await device(root.top))) return { cause: 'volume', end: next };
    }
    if (info.isSymbolicLink()) {
      if (++links > WALK_LINKS_MAX) return { cause: 'place', end: next };
      const target = await readlink(next);
      const joined = collapse(target.startsWith('/') ? target : `${folder}/${target}`);
      if (joined === null) return { cause: 'place', end: next };
      if (!target.startsWith('/') && root !== null && !below(joined, root.top)) return { cause: 'place', end: next };
      const again = rootText(joined, place);
      if (again === null) return { cause: 'place', end: next };
      folder = again.root.top;
      rest = [
        ...again.path
          .slice(folder.length)
          .split('/')
          .filter((p) => p !== ''),
        ...rest,
      ];
      continue;
    }
    if (info.isDirectory()) {
      if (temp && (info.mode & OTHERS_WRITE) !== 0) return { cause: 'shared', end: next };
      folder = next;
      continue;
    }
    if (info.isFile() && rest.length === 0) {
      end = next;
      break;
    }
    return { cause: temp ? 'special' : 'place', end: next };
  }
  const root = rootOf(end, place);
  if (root !== null && root.kind === 'temp' && end === root.top) return { cause: 'top', end };
  const folded = foldPath(end);
  for (const refused of place.refused ?? []) {
    const other = foldPath(refused);
    if (below(folded, other) || below(other, folded)) return { cause: 'app', end };
  }
  return { cause: null, end };
}

/**
 * The unit rules for a path whose text or landing is in a temp root: walked as written ({@link tempWalk}), and the
 * walk must end where the disk lands it. A refused place sets `facts.temp`; a path the walk cannot place asks as a
 * place. True when every rule holds (or none applies).
 */
async function unitRules(text: string, lands: string | null, place: WorkPlace, facts: AutoFacts): Promise<boolean> {
  const real = lands ?? text;
  if (inTempReal(real, place)) facts.touched = true;
  if (!inTempText(text, place) && !inTempReal(real, place)) return true;
  const walk = await tempWalk(text, place);
  if (walk.cause === 'place' || (walk.cause === null && lands !== null && walk.end !== lands)) {
    notPlaced(facts, text, place);
    return false;
  }
  if (walk.cause !== null) {
    refuse(facts, text, walk.cause, place);
    return false;
  }
  return true;
}

/** A project tool or a script that starts in a temp root, or is given a path there, asks (U8), after U1-U7. */
async function toolRules(text: string, lands: string | null, place: WorkPlace, facts: AutoFacts): Promise<void> {
  const real = lands ?? text;
  if (!inTempText(text, place) && !inTempReal(real, place)) return;
  if (await unitRules(text, lands, place, facts)) refuse(facts, text, 'tool', place);
}

/** Entries of one folder a reader's pattern may read: more than this many is not placed. */
const GLOB_ENTRIES_MAX = 4096;
/** Matcher cells one pattern may spend on names with a non-ASCII character; past it each such name counts as a match. */
const GLOB_CELLS_MAX = 4_000_000;

/** Two characters a shell may take as the same with `nocaseglob`: equal, or equal in lower or upper case. */
function sameLetter(a: string, b: string): boolean {
  return a === b || a.toLowerCase() === b.toLowerCase() || a.toUpperCase() === b.toUpperCase();
}

/** Cells of the matcher for one name with a non-ASCII character; past this the name counts as a match. */
const MATCH_CELLS_MAX = 1 << 16;

/**
 * Whether `name` may match `pattern` in a shell, whatever its options and locale: a superset of what the shell matches.
 * `*` and `?` both match any run (a `?` is one byte in the C locale), letters in either case, a leading dot too. A
 * pattern with a non-ASCII character always may. In a name, each non-ASCII code point also matches any run of the
 * pattern: the shell may see the name in another normal form (a decomposed `é` is `e` plus a mark) or case.
 */
export function mayMatch(pattern: string, name: string): boolean {
  if (hasNonAscii(pattern)) return true;
  if (hasNonAscii(name)) return mayMatchWide(pattern, name);
  const p = Array.from(pattern);
  const s = Array.from(name);
  let i = 0;
  let j = 0;
  let star = -1;
  let mark = 0;
  while (j < s.length) {
    const wild = i < p.length && (p[i] === '*' || p[i] === '?');
    if (i < p.length && !wild && sameLetter(p[i]!, s[j]!)) {
      i++;
      j++;
    } else if (wild) {
      star = i++;
      mark = j;
    } else if (star !== -1) {
      i = star + 1;
      j = ++mark;
    } else return false;
  }
  while (i < p.length && (p[i] === '*' || p[i] === '?')) i++;
  return i === p.length;
}

/**
 * The ASCII letters and signs, lowered, a non-ASCII code point may show as to a shell: those of its decomposed and
 * compatibility forms and of its case forms (`é` has `e`, the Kelvin sign has `k`, `ﬁ` has `f` and `i`).
 */
function asciiShapes(c: string): Set<string> {
  const out = new Set<string>();
  for (const form of [c, c.toLowerCase(), c.toUpperCase()]) {
    for (const text of [form.normalize('NFD'), form.normalize('NFKD')]) {
      for (const ch of text) if (ch.charCodeAt(0) <= 0x7f) out.add(ch.toLowerCase());
    }
  }
  return out;
}

/**
 * {@link mayMatch} for a name with a non-ASCII code point: `can[i][j]` says whether `p[i..]` may match `s[j..]`. Such a
 * code point may match nothing (a mark the shell joins to the letter before it), or take pattern characters: a
 * wildcard, or a letter it may show as.
 */
function mayMatchWide(pattern: string, name: string): boolean {
  const p = Array.from(pattern);
  const s = Array.from(name);
  const width = s.length + 1;
  if ((p.length + 1) * width > MATCH_CELLS_MAX) return true;
  const can = new Uint8Array((p.length + 1) * width);
  const shapes = s.map((c) => (c.charCodeAt(0) > 0x7f ? asciiShapes(c) : null));
  const pw = (i: number): boolean => i < p.length && (p[i] === '*' || p[i] === '?');
  const sw = (j: number): boolean => j < s.length && shapes[j] !== null;
  const takes = (i: number, j: number): boolean => pw(i) || shapes[j]!.has(p[i]!.toLowerCase());
  for (let i = p.length; i >= 0; i--) {
    for (let j = s.length; j >= 0; j--) {
      let ok = i === p.length && j === s.length;
      if (!ok && pw(i)) ok = can[(i + 1) * width + j] === 1 || (j < s.length && can[i * width + j + 1] === 1);
      if (!ok && sw(j)) {
        ok = can[i * width + j + 1] === 1 || (i < p.length && takes(i, j) && can[(i + 1) * width + j] === 1);
      }
      if (!ok && i < p.length && j < s.length && !pw(i) && !sw(j) && sameLetter(p[i]!, s[j]!)) {
        ok = can[(i + 1) * width + j + 1] === 1;
      }
      if (ok) can[i * width + j] = 1;
    }
  }
  return can[0] === 1;
}

/** Places a reader's pattern: its folder chain is real, and every entry it may match is a plain file or folder. */
async function placeGlob(
  check: Extract<PathCheck, { kind: 'glob' }>,
  moved: readonly Moved[],
  place: WorkPlace,
  facts: AutoFacts,
) {
  const folder = check.parts.length === 0 ? check.base : `${check.base}/${check.parts.join('/')}`;
  if (!(await realFolders(check.base, check.parts)) || underMoved(folder, folder, check.part, moved)) {
    notPlaced(facts, folder, place);
    return;
  }
  if (!(await unitRules(folder, await landingPath(folder), place, facts))) return;
  const temp = inTempReal(folder, place);
  const uid = uidOf(place);
  let names: string[];
  try {
    names = await readdir(folder);
  } catch {
    facts.placed = false;
    return;
  }
  if (names.length > GLOB_ENTRIES_MAX) {
    facts.placed = false;
    return;
  }
  let cells = 0;
  for (const name of names) {
    if (hasNonAscii(name)) cells += (check.pattern.length + 1) * (name.length + 1);
    if (cells <= GLOB_CELLS_MAX && !mayMatch(check.pattern, name)) continue;
    if (protectedPart(name)) facts.protectedPath = true;
    try {
      const info = await lstat(`${folder}/${name}`);
      if (!info.isFile() && !info.isDirectory()) notPlaced(facts, `${folder}/${name}`, place);
      else if (temp && info.uid !== uid) refuse(facts, `${folder}/${name}`, 'owner', place);
    } catch {
      notPlaced(facts, `${folder}/${name}`, place);
    }
  }
  if (protectedLanding(folder, place)) facts.protectedPath = true;
}

/** One setting of a git config text: section and key lowered, the subsection as written; `null` value: a bare key. */
export interface ConfigEntry {
  section: string;
  sub: string | null;
  key: string;
  value: string | null;
}

function configNameChar(c: string, first: boolean): boolean {
  const code = c.charCodeAt(0);
  const letter = (code >= 0x41 && code <= 0x5a) || (code >= 0x61 && code <= 0x7a);
  return letter || (!first && ((code >= 0x30 && code <= 0x39) || c === '-'));
}

function configName(text: string, firstLetter: boolean): boolean {
  if (text === '') return false;
  for (let k = 0; k < text.length; k++) if (!configNameChar(text.charAt(k), firstLetter && k === 0)) return false;
  return true;
}

function trimBlank(text: string): string {
  let start = 0;
  let end = text.length;
  while (start < end && (text.charAt(start) === ' ' || text.charAt(start) === '\t')) start++;
  while (end > start && (text.charAt(end - 1) === ' ' || text.charAt(end - 1) === '\t')) end--;
  return text.slice(start, end);
}

/**
 * A git config text in the plain form git writes, or null for anything else (quotes, escapes, line continuations,
 * comments after a value, a dotted section head, a key outside a section): those ask.
 */
export function parseGitConfig(text: string): ConfigEntry[] | null {
  const entries: ConfigEntry[] = [];
  let section: string | null = null;
  let sub: string | null = null;
  for (const raw of text.split('\n')) {
    if (raw.includes('\r') || raw.includes('\0')) return null;
    const line = trimBlank(raw);
    if (line === '' || line.startsWith('#') || line.startsWith(';')) continue;
    if (line.startsWith('[')) {
      if (!line.endsWith(']')) return null;
      const inner = line.slice(1, -1);
      const space = inner.indexOf(' ');
      const name = space === -1 ? inner : inner.slice(0, space);
      if (!configName(name, false)) return null;
      section = name.toLowerCase();
      sub = null;
      if (space !== -1) {
        const quoted = inner.slice(space + 1);
        if (quoted.length < 2 || !quoted.startsWith('"') || !quoted.endsWith('"')) return null;
        sub = quoted.slice(1, -1);
        if (sub.includes('"') || sub.includes('\\')) return null;
      }
      continue;
    }
    if (section === null) return null;
    const eq = line.indexOf('=');
    const key = eq === -1 ? line : trimBlank(line.slice(0, eq));
    if (!configName(key, true)) return null;
    let value: string | null = null;
    if (eq !== -1) {
      value = trimBlank(line.slice(eq + 1));
      for (const c of '"\\#;') if (value.includes(c)) return null;
    }
    entries.push({ section, sub, key: key.toLowerCase(), value });
  }
  return entries;
}

/** Keys of sections with no subsection that change nothing git runs, opens or writes outside the work tree. */
const CONFIG_KEYS: Readonly<Record<string, readonly string[]>> = {
  core: [
    'repositoryformatversion',
    'filemode',
    'bare',
    'logallrefupdates',
    'ignorecase',
    'precomposeunicode',
    'symlinks',
    'autocrlf',
    'eol',
    'safecrlf',
    'untrackedcache',
    'checkstat',
    'trustctime',
    'quotepath',
    'abbrev',
  ],
  extensions: ['objectformat', 'refstorage'],
  user: ['name', 'email'],
  pull: ['rebase', 'ff'],
  push: ['default', 'autosetupremote'],
  fetch: ['prune'],
  init: ['defaultbranch'],
  merge: ['ff'],
  rebase: ['autostash'],
  lfs: ['repositoryformatversion'],
};

/** Keys of `[remote "NAME"]` and `[branch "NAME"]`. */
const CONFIG_SUB_KEYS: Readonly<Record<string, readonly string[]>> = {
  remote: ['url', 'pushurl', 'fetch', 'push', 'tagopt', 'prune'],
  branch: ['remote', 'merge', 'rebase', 'pushremote', 'description', 'vscode-merge-base', 'github-pr-owner-number'],
};

function hostChar(c: string): boolean {
  return isIdentText(c) || c === '.' || c === '-';
}

function isIdentText(c: string): boolean {
  const code = c.charCodeAt(0);
  return (
    (code >= 0x30 && code <= 0x39) || (code >= 0x41 && code <= 0x5a) || (code >= 0x61 && code <= 0x7a) || c === '_'
  );
}

/** `[user@]host` with a host that starts with a letter or digit (a leading `-` would be an ssh option). */
function remoteHost(text: string): boolean {
  const at = text.indexOf('@');
  const user = at === -1 ? '' : text.slice(0, at);
  const host = at === -1 ? text : text.slice(at + 1);
  for (const c of user) if (!hostChar(c)) return false;
  if (host === '' || !isIdentText(host.charAt(0)) || host.charAt(0) === '_') return false;
  for (const c of host) if (!hostChar(c)) return false;
  return true;
}

/**
 * A remote URL git reaches over the network: `https`, `http`, `ssh` or `git` with a plain host, or the scp form
 * `[user@]host:path`. A local path, `file://`, `ext::` or any `<helper>::` asks: git would read or write it on this disk
 * or run a program.
 */
export function networkRemote(url: string): boolean {
  if (url === '' || url.includes(' ') || url.includes('\t') || url.includes('::')) return false;
  for (const scheme of ['https://', 'http://', 'ssh://', 'git://']) {
    if (!url.startsWith(scheme)) continue;
    const rest = url.slice(scheme.length);
    const slash = rest.indexOf('/');
    const authority = slash === -1 ? rest : rest.slice(0, slash);
    const colon = authority.lastIndexOf(':');
    const host = colon === -1 ? authority : authority.slice(0, colon);
    if (colon !== -1) for (const c of authority.slice(colon + 1)) if (c < '0' || c > '9') return false;
    return remoteHost(host);
  }
  if (url.includes('://')) return false;
  const colon = url.indexOf(':');
  if (colon <= 0) return false;
  const host = url.slice(0, colon);
  if (host.includes('/')) return false;
  return remoteHost(host) && url.length > colon + 1;
}

/** Remote names of a parsed git config: `[remote "name"]` sections. */
export function configRemotes(entries: readonly ConfigEntry[]): string[] {
  const names: string[] = [];
  for (const e of entries) if (e.section === 'remote' && e.sub !== null && !names.includes(e.sub)) names.push(e.sub);
  return names;
}

/**
 * Every key of the folder's git config is on the allow list. With `network` (the call reaches a remote: fetch, pull,
 * push), each remote URL must be a network one, and a branch's remote one the config names (or `.`): a name git does
 * not know is read as a URL or a path.
 */
export function configAllowed(entries: readonly ConfigEntry[], network: boolean): boolean {
  const remotes = configRemotes(entries);
  for (const e of entries) {
    const table = e.sub === null ? CONFIG_KEYS : CONFIG_SUB_KEYS;
    if (!Object.hasOwn(table, e.section) || !table[e.section]!.includes(e.key)) return false;
    const where = `${e.section}.${e.key}`;
    if (where === 'core.bare' && e.value !== 'false') return false;
    if (where === 'core.repositoryformatversion' && e.value !== '0' && e.value !== '1') return false;
    if (!network) continue;
    if (e.section === 'remote' && (e.key === 'url' || e.key === 'pushurl')) {
      if (e.value === null || !networkRemote(e.value)) return false;
    }
    if (e.section === 'branch' && (e.key === 'remote' || e.key === 'pushremote')) {
      if (e.value === null || (e.value !== '.' && !remotes.includes(e.value))) return false;
    }
  }
  return true;
}

/** A global git config text that may set `core.hooksPath` or include another file: any line that names either asks. */
export function configMovesHooks(text: string): boolean {
  for (const raw of text.split('\n')) {
    const line = raw.toLowerCase();
    if (line.includes('hookspath') || line.includes('[include')) return true;
  }
  return false;
}

const CONFIG_MAX = 1024 * 1024;

/** A small text file; null when it is missing; `false` when it is there but cannot be read as one. */
async function smallText(path: string): Promise<string | null | false> {
  try {
    const info = await stat(path);
    if (!info.isFile() || info.size > CONFIG_MAX) return false;
    return await readFile(path, 'utf8');
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    return code === 'ENOENT' || code === 'ENOTDIR' ? null : false;
  }
}

/** The user's two global git config files, read for a moved hooks folder or an include. */
function globalConfigFiles(place: WorkPlace): string[] {
  return [join(place.home, '.gitconfig'), join(place.home, '.config', 'git', 'config')];
}

/**
 * A repository's own git config, parsed: `.git/config` must be a regular file (no link) the parser reads in full. Null
 * when it is missing, is not one, or holds a form the parser does not read.
 */
async function repoConfig(repo: string): Promise<ConfigEntry[] | null> {
  const path = join(repo, '.git', 'config');
  const info = await lstatOrNull(path);
  if (info === null || !info.isFile() || info.size > CONFIG_MAX) return null;
  return parseGitConfig(await readFile(path, 'utf8'));
}

/** Index extensions that name no file and run nothing; any other (a split index, fsmonitor data, sparse folders) asks. */
const INDEX_EXTENSIONS: ReadonlySet<string> = new Set(['TREE', 'REUC', 'UNTR', 'EOIE', 'IEOT']);

const INDEX_MAX = 256 * 1024 * 1024;

function u32(data: Uint8Array, at: number): number {
  return ((data[at]! << 24) >>> 0) + (data[at + 1]! << 16) + (data[at + 2]! << 8) + data[at + 3]!;
}

/**
 * Whether a git index (version 2, 3 or 4) holds a submodule entry (a gitlink): git status, diff and others then run
 * git inside that folder with its own config and hooks. An index the reader cannot walk to its checksum counts too.
 */
export function indexHasGitlink(data: Uint8Array, hashSize: number): boolean {
  if (data.length < 12 + hashSize) return true;
  if (data[0] !== 0x44 || data[1] !== 0x49 || data[2] !== 0x52 || data[3] !== 0x43) return true;
  const version = u32(data, 4);
  if (version < 2 || version > 4) return true;
  const count = u32(data, 8);
  const end = data.length - hashSize;
  const fixed = 40 + hashSize + 2;
  let at = 12;
  for (let n = 0; n < count; n++) {
    const start = at;
    if (at + fixed > end) return true;
    if (u32(data, at + 24) >>> 12 === 0xe) return true;
    const flags = (data[at + 40 + hashSize]! << 8) | data[at + 41 + hashSize]!;
    at += fixed;
    if ((flags & 0x4000) !== 0) {
      if (version < 3) return true;
      at += 2;
    }
    if (version === 4) {
      let steps = 0;
      for (;;) {
        if (at >= end || ++steps > 8) return true;
        if ((data[at++]! & 0x80) === 0) break;
      }
    }
    let nul = at;
    while (nul < end && data[nul] !== 0) nul++;
    if (nul >= end) return true;
    at = version === 4 ? nul + 1 : start + ((nul - start + 8) & ~7);
    if (at > end) return true;
  }
  while (at < end) {
    if (at + 8 > end) return true;
    const name = String.fromCharCode(data[at]!, data[at + 1]!, data[at + 2]!, data[at + 3]!);
    if (!INDEX_EXTENSIONS.has(name)) return true;
    at += 8 + u32(data, at + 4);
  }
  return at !== end;
}

/** The object hash size the config names: 20 (sha1, the default) or 32 (sha256); null for anything else. */
function hashSize(entries: readonly ConfigEntry[]): number | null {
  let size = 20;
  for (const e of entries) {
    if (e.section !== 'extensions' || e.sub !== null || e.key !== 'objectformat') continue;
    if (e.value === 'sha256') size = 32;
    else if (e.value !== 'sha1') return null;
  }
  return size;
}

/** The index names no submodule: missing, or a regular file the reader walks in full with no gitlink. */
async function indexPlain(git: string, entries: readonly ConfigEntry[]): Promise<boolean> {
  const size = hashSize(entries);
  if (size === null) return false;
  const path = join(git, 'index');
  const info = await lstatOrNull(path);
  if (info === null) return true;
  if (!info.isFile() || info.size > INDEX_MAX) return false;
  return !indexHasGitlink(await readFile(path), size);
}

/**
 * git works in a repository only as the scan models it: `.git` is a real folder that holds no `commondir` (config and
 * hooks of another folder), no filled `modules` and no gitlink in its index (submodules with their own config and
 * hooks), no alternates (objects read from another folder) and no link among its entries (a write that lands
 * elsewhere), and every key of its config is on the allow list. With `hooks`, git runs no hook the folder can change
 * either: `.git/hooks` holds only `.sample` files, and no global config moves the hooks folder or includes a file
 * (`git-global`). Null when it passes, else what failed.
 */
async function gitPlain(
  repo: string,
  place: WorkPlace,
  check: Extract<PathCheck, { kind: 'git-config' }>,
): Promise<'git' | 'git-global' | null> {
  const git = join(repo, '.git');
  const info = await lstatOrNull(git);
  if (info === null || info.isSymbolicLink() || !info.isDirectory()) return 'git';
  for (const name of await readdir(git)) {
    const entry = await lstatOrNull(join(git, name));
    if (entry === null || entry.isSymbolicLink()) return 'git';
  }
  for (const name of ['commondir', 'objects/info/alternates', 'objects/info/http-alternates']) {
    if ((await lstatOrNull(join(git, name))) !== null) return 'git';
  }
  try {
    if ((await readdir(join(git, 'modules'))).length > 0) return 'git';
  } catch (error) {
    if (!missing(error)) throw error;
  }
  const entries = await repoConfig(repo);
  if (entries === null || !configAllowed(entries, check.network)) return 'git';
  if (!(await indexPlain(git, entries))) return 'git';
  if (!check.hooks) return null;
  try {
    for (const name of await readdir(join(git, 'hooks'))) if (!name.endsWith('.sample')) return 'git';
  } catch (error) {
    if (!missing(error)) throw error;
  }
  for (const file of globalConfigFiles(place)) {
    const text = await smallText(file);
    if (text === false || (text !== null && configMovesHooks(text))) return 'git-global';
  }
  return null;
}

/** Each start folder below the Work folder finds no other `.git` on its way up to it. */
async function workDirsPlain(dirs: readonly string[], place: WorkPlace): Promise<boolean> {
  const base = place.cwd.endsWith('/') ? place.cwd : `${place.cwd}/`;
  for (const dir of dirs) {
    if (dir === place.cwd) continue;
    if (!dir.startsWith(base)) return false;
    let at = place.cwd;
    for (const part of dir.slice(base.length).split('/')) {
      at = `${at}/${part}`;
      if ((await lstatOrNull(join(at, '.git'))) !== null) return false;
    }
  }
  return true;
}

/** The names of a folder, folded as the disk compares them. */
async function foldedNames(folder: string): Promise<Set<string>> {
  return new Set((await readdir(folder)).map((n) => foldName(n)));
}

/**
 * The repository git uses when started in `dir` under a temp root's `top`: walking up, the first folder below the top
 * that holds `.git`. `bare`: a folder on the way holds `HEAD` (git may take it as a bare repository); `none`: no folder
 * below the top holds `.git` (the top's own is never used).
 */
async function tempRepo(dir: string, top: string): Promise<Repo> {
  for (let at = dir; at !== top && below(at, top); at = dirname(at)) {
    const names = await foldedNames(at);
    if (names.has('.git')) return { kind: 'repo', folder: at };
    if (names.has('head')) return { kind: 'bare' };
  }
  return { kind: 'none' };
}

/** Where git finds its repository: a folder, a folder that may be taken as a bare repository, or none. */
type Repo = { kind: 'repo'; folder: string } | { kind: 'bare' } | { kind: 'none' };

/** 40 or 64 hex digits. */
function objectName(text: string): boolean {
  if (text.length !== 40 && text.length !== 64) return false;
  for (const c of text) if (!'0123456789abcdef'.includes(c)) return false;
  return true;
}

/**
 * A repository under a temp root, beside {@link gitPlain}: `.git` is one git accepts at this level (`HEAD` a regular
 * file naming a ref or an object, `objects` and `refs` real folders), and `.git` and its first-level entries are yours,
 * with no write bit for group or other on a folder among them or on `config`, `HEAD`, `index`, `packed-refs`.
 */
async function tempRepoPlain(repo: string, place: WorkPlace): Promise<boolean> {
  const git = join(repo, '.git');
  const uid = uidOf(place);
  const info = await lstatOrNull(git);
  if (info === null || !info.isDirectory() || info.uid !== uid || (info.mode & OTHERS_WRITE) !== 0) return false;
  const head = await lstatOrNull(join(git, 'HEAD'));
  if (head === null || !head.isFile() || head.size > 4096) return false;
  let text = await readFile(join(git, 'HEAD'), 'utf8');
  if (text.endsWith('\n')) text = text.slice(0, -1);
  if (!text.startsWith('ref: refs/') && !objectName(text)) return false;
  for (const name of ['objects', 'refs']) {
    const folder = await lstatOrNull(join(git, name));
    if (folder === null || folder.isSymbolicLink() || !folder.isDirectory()) return false;
  }
  for (const name of await readdir(git)) {
    const entry = await lstatOrNull(join(git, name));
    if (entry === null || entry.uid !== uid) return false;
    const guarded = entry.isDirectory() || ['config', 'HEAD', 'index', 'packed-refs'].includes(name);
    if (guarded && (entry.mode & OTHERS_WRITE) !== 0) return false;
  }
  return true;
}

/** The repository folder of a start folder: the Work folder, or one found under a temp root. */
async function repoOf(dir: string, place: WorkPlace): Promise<Repo> {
  const at = rootText(dir, place);
  if (at === null || at.root.kind !== 'temp') return { kind: 'repo', folder: place.cwd };
  return tempRepo(at.path, at.root.top);
}

/** How a repository folder shows on a card: the Work folder by its name, a temp folder by its path as typed. */
function repoShown(repo: string, place: WorkPlace): string {
  return repo === place.cwd ? basename(repo) : typedPath(repo, place);
}

/** Every start folder of a git call has a repository git and the gate agree on, and it passes. */
async function placeGit(
  check: Extract<PathCheck, { kind: 'git-config' }>,
  place: WorkPlace,
  facts: AutoFacts,
): Promise<void> {
  const work = check.dirs.filter((d) => !inTempText(d, place));
  if (work.length > 0) {
    const cause = (await workDirsPlain(work, place)) ? await gitPlain(place.cwd, place, check) : 'git';
    if (cause !== null) {
      facts.placed = false;
      note(facts, repoShown(place.cwd, place), cause, place);
    }
  }
  for (const dir of check.dirs) {
    if (!inTempText(dir, place)) continue;
    const real = rootText(dir, place)!.path;
    if (!(await unitRules(real, real, place, facts))) continue;
    const found = await repoOf(real, place);
    if (found.kind !== 'repo') {
      facts.placed = false;
      note(facts, real, 'git', place);
      continue;
    }
    const repo = found.folder;
    let cause = (await tempRepoPlain(repo, place)) ? await gitPlain(repo, place, check) : 'git';
    if (cause === null && !(await unitRules(repo, repo, place, facts))) cause = 'git';
    if (cause !== null) {
      facts.placed = false;
      note(facts, repoShown(repo, place), cause, place);
    }
  }
}

/** The text of a check as the command names it: the place a disk error is reported for. */
function checkText(check: PathCheck, place: WorkPlace): string | null {
  switch (check.kind) {
    case 'cd':
    case 'glob':
      return check.parts.length === 0 ? check.base : `${check.base}/${check.parts.join('/')}`;
    case 'remote':
    case 'script-name':
      return check.base;
    case 'git-config':
      return check.dirs.find((d) => inTempText(d, place)) ?? null;
    default:
      return check.path;
  }
}

/**
 * Follows the scan's paths on disk, in order. A disk error on a check whose text is in a temp root asks there
 * (`unreadable`); any other error throws, and the caller asks.
 */
async function placeChecks(checks: readonly PathCheck[], place: WorkPlace, facts: AutoFacts): Promise<void> {
  const moved: Moved[] = [];
  for (const check of checks) {
    try {
      await placeCheck(check, moved, place, facts);
    } catch (error) {
      const text = checkText(check, place);
      if (text === null || !inTempText(text, place)) throw error;
      facts.placed = false;
      refuse(facts, text, 'unreadable', place);
    }
  }
}

async function placeCheck(check: PathCheck, moved: Moved[], place: WorkPlace, facts: AutoFacts): Promise<void> {
  switch (check.kind) {
    case 'moved': {
      const landing = await landingPath(check.path);
      moved.push({
        part: check.part,
        text: foldPath(check.path),
        landing: landing === null ? null : foldPath(landing),
      });
      break;
    }
    case 'write':
      await placeWrite(check.path, check.part, moved, place, facts, check.mode === true);
      break;
    case 'into': {
      const lands = await landingPath(check.path);
      if (check.slash || (lands !== null && (await isFolder(lands)))) {
        for (const name of check.names) await placeWrite(`${check.path}/${name}`, check.part, moved, place, facts);
      } else {
        await placeWrite(check.path, check.part, moved, place, facts);
      }
      break;
    }
    case 'cd': {
      const to = check.parts.length === 0 ? check.base : `${check.base}/${check.parts.join('/')}`;
      if (!(await realFolders(check.base, check.parts)) || underMoved(to, to, check.part, moved)) {
        notPlaced(facts, to, place);
      } else await unitRules(to, await landingPath(to), place, facts);
      break;
    }
    case 'read': {
      const lands = await landingPath(check.path);
      if (lands === null || underMoved(check.path, lands, check.part, moved)) notPlaced(facts, check.path, place);
      else {
        if (rootOf(lands, place) === null) {
          facts.outside = true;
          note(facts, check.path, 'place', place);
        }
        if (protectedLanding(lands, place)) facts.protectedPath = true;
        if (check.file === true && (await isFolder(lands))) facts.placed = false;
        await unitRules(check.path, lands, place, facts);
      }
      break;
    }
    case 'glob':
      await placeGlob(check, moved, place, facts);
      break;
    case 'tool':
      if (underMoved(check.path, check.path, check.part, moved) || !(await plainEntry(check.path, place))) {
        notPlaced(facts, check.path, place);
      }
      if (protectedLanding(check.path, place)) facts.protectedPath = true;
      await toolRules(check.path, null, place, facts);
      break;
    case 'script': {
      const lands = await landingPath(check.path);
      const file = lands !== null && (await statOrNull(lands))?.isFile() === true;
      if (lands === null || !file || underMoved(check.path, lands, check.part, moved))
        notPlaced(facts, check.path, place);
      else {
        if (rootOf(lands, place) === null) {
          facts.outside = true;
          note(facts, check.path, 'place', place);
        }
        if (protectedLanding(lands, place)) facts.protectedPath = true;
      }
      await toolRules(check.path, lands, place, facts);
      break;
    }
    case 'absent':
      if (await exists(check.path)) facts.placed = false;
      await unitRules(check.path, null, place, facts);
      break;
    case 'remote': {
      const repo = await repoOf(check.base, place);
      const entries = repo.kind === 'repo' ? await repoConfig(repo.folder) : null;
      if (entries === null || !configRemotes(entries).includes(check.name)) facts.placed = false;
      if (await exists(join(check.base, check.name))) facts.placed = false;
      await unitRules(check.base, check.base, place, facts);
      break;
    }
    case 'script-name': {
      if (!(await scriptOf(join(check.base, 'package.json'), check.name))) facts.placed = false;
      await toolRules(check.base, check.base, place, facts);
      break;
    }
    case 'git-config':
      await placeGit(check, place, facts);
      break;
    default:
      facts.placed = false;
  }
}

/** `name` is a script of the package file: `scripts` holds it as a string. */
async function scriptOf(path: string, name: string): Promise<boolean> {
  const text = await smallText(path);
  if (typeof text !== 'string') return false;
  try {
    const parsed: unknown = JSON.parse(text);
    if (!isPlainObject(parsed) || !isPlainObject(parsed.scripts)) return false;
    return Object.hasOwn(parsed.scripts, name) && typeof parsed.scripts[name] === 'string';
  } catch {
    return false;
  }
}

/** The tops of the temp roots a path names by text or lands in. */
function reachOf(path: string, place: WorkPlace, out: Set<string>): void {
  const byText = rootText(path, place)?.root;
  if (byText?.kind === 'temp') out.add(byText.top);
  const byReal = rootOf(path, place);
  if (byReal?.kind === 'temp') out.add(byReal.top);
}

/** Every git call has a repository: the Work folder holds `.git`, and each start folder in a temp root finds one. */
async function repository(scan: AutoScan, place: WorkPlace, facts: AutoFacts): Promise<boolean> {
  const checks = scan.checks ?? [];
  const dirs = checks.flatMap((c) => (c.kind === 'git-config' ? c.dirs : []));
  const work = !checks.some((c) => c.kind === 'git-config') || dirs.some((d) => !inTempText(d, place));
  if (work && !(await exists(join(place.cwd, '.git')))) {
    note(facts, repoShown(place.cwd, place), 'git', place);
    return false;
  }
  for (const dir of dirs) {
    if (!inTempText(dir, place)) continue;
    let repo: Repo;
    try {
      repo = await repoOf(dir, place);
    } catch {
      facts.placed = false;
      refuse(facts, dir, 'unreadable', place);
      continue;
    }
    if (repo.kind === 'none') {
      note(facts, dir, 'git', place);
      return false;
    }
  }
  return true;
}

async function commandFacts(command: string, place: WorkPlace, cdpath: boolean): Promise<AutoFacts> {
  const flags = commandFlags(command, place);
  const scan = autoScan(command, place, { cdpath });
  const facts: AutoFacts = { ...NO_FACTS, scan, flags, placed: true };
  const temps = tempRoots(place).map((r) => r.top);
  const reach = new Set<string>();
  if (scan.unread || scan.checks === null || scan.starts.length > 0) for (const top of temps) reach.add(top);
  for (const check of scan.checks ?? []) {
    const text = checkText(check, place);
    if (text !== null) reachOf(text, place, reach);
    if (check.kind === 'git-config') for (const dir of check.dirs) reachOf(dir, place, reach);
  }
  facts.reach = [...reach];
  if (scan.git) facts.repository = await repository(scan, place, facts);
  if (scan.unread || scan.checks === null) return facts;
  await placeChecks(scan.checks, place, facts);
  for (const start of scan.starts) {
    try {
      await toolRules(start, start, place, facts);
    } catch {
      facts.placed = false;
      refuse(facts, start, 'unreadable', place);
    }
  }
  if (facts.touched) for (const top of temps) reach.add(top);
  facts.reach = [...reach];
  return facts;
}

/** The facts of an edit: its landing, the gate's flags, and the unit rules when its text or landing is in a temp root. */
async function editFacts(path: string, place: WorkPlace, landing: string | null): Promise<AutoFacts> {
  const written = agentPath(path, place);
  const known = landing !== null && !(await special(landing)) && !(await isFolder(landing));
  const facts: AutoFacts = {
    ...NO_FACTS,
    flags: landing === null || landsOutside(landing, place) ? ['outside-folder'] : [],
    landing: known ? landing : null,
    landingOutside: landing !== null && rootOf(landing, place) === null,
    protectedPath: protectedLanding(written, place) || (landing !== null && protectedLanding(landing, place)),
    links: landing !== null && (await linked(landing)),
  };
  const temps = tempRoots(place).map((r) => r.top);
  const reach = new Set<string>();
  reachOf(written, place, reach);
  if (landing === null) for (const top of temps) reach.add(top);
  else reachOf(landing, place, reach);
  facts.reach = [...reach];
  const textTemp = inTempText(written, place);
  const landTemp = landing !== null && inTempReal(landing, place);
  if (landTemp) facts.touched = true;
  if (!textTemp && !landTemp) return facts;
  if (written.split('/').includes('..')) {
    refuse(facts, written, 'place', place);
    return facts;
  }
  try {
    const walk = await tempWalk(written, place);
    if (walk.cause !== null && walk.cause !== 'place') {
      refuse(facts, written, walk.cause, place);
      return facts;
    }
    if (landing !== null && landTemp) {
      const landed = await tempWalk(landing, place);
      if (landed.cause !== null && landed.cause !== 'place') {
        refuse(facts, landing, landed.cause, place);
        return facts;
      }
      if (landed.cause === 'place') facts.landing = null;
    }
    if (!facts.landingOutside && (walk.cause === 'place' || (landing !== null && walk.end !== landing))) {
      facts.landing = null;
    }
  } catch {
    refuse(facts, written, 'unreadable', place);
  }
  return facts;
}

/**
 * The facts of one request: the command's Auto scan with each path placed on disk, or the edit's landing.
 * `landing` is the edit's landing the card was built with. Never throws: an error inside gives facts that ask.
 */
export async function autoFacts(
  req: PermissionRequest,
  place: WorkPlace,
  landing: string | null,
  options: { cdpath: boolean },
): Promise<AutoFacts> {
  try {
    const target = requestTarget(req);
    if (target.command !== null) return await commandFacts(target.command, place, options.cdpath);
    if (target.path !== null) return await editFacts(target.path, place, landing);
    return { ...NO_FACTS };
  } catch {
    return { ...NO_FACTS, landing: null, reach: everyTemp(place) };
  }
}

/** Every temp root top of the place; none when the place cannot be read. */
function everyTemp(place: WorkPlace): string[] {
  try {
    return tempRoots(place).map((r) => r.top);
  } catch {
    return [];
  }
}

// ── the user's hooks ───────────────────────────────────────────────────────────────────────────────────────────────

const HOOK_FILE_MAX = 4 * 1024 * 1024;

/** A file that does not exist: `false`; one that cannot be read: `true` (it counts as a hook). */
async function readSource(path: string): Promise<string | boolean> {
  try {
    const info = await lstat(path);
    if (info.size > HOOK_FILE_MAX) return true;
    return await readFile(path, 'utf8');
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    return code !== 'ENOENT' && code !== 'ENOTDIR';
  }
}

/** A hook event of a settings file that can ask before a tool runs. */
function toolEvent(name: string): boolean {
  const lower = name.toLowerCase();
  return lower.includes('tool') || lower.includes('permission') || lower.startsWith('before');
}

/** A JSON settings file (Claude, Cursor) whose `hooks` hold a tool hook; a file that cannot be parsed counts. */
async function settingsHook(path: string): Promise<boolean> {
  const text = await readSource(path);
  if (typeof text === 'boolean') return text;
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return true;
  }
  if (!isPlainObject(parsed) || !Object.hasOwn(parsed, 'hooks')) return false;
  const hooks = parsed.hooks;
  if (!isPlainObject(hooks)) return hooks !== undefined && hooks !== null;
  for (const [event, list] of Object.entries(hooks)) {
    if (!toolEvent(event)) continue;
    if (!Array.isArray(list) || list.length > 0) return true;
  }
  return false;
}

/** A TOML config with a `[hooks]` table or a `hooks` key. */
async function tomlHook(path: string): Promise<boolean> {
  const text = await readSource(path);
  if (typeof text === 'boolean') return text;
  for (const raw of text.split('\n')) {
    let line = raw;
    while (line.startsWith(' ') || line.startsWith('\t')) line = line.slice(1);
    if (line.startsWith('[hooks') || line.startsWith('[[hooks')) return true;
    if (line.startsWith('hooks') && (line.charAt(5) === ' ' || line.charAt(5) === '=' || line.charAt(5) === '.')) {
      return true;
    }
  }
  return false;
}

/** A hooks folder with any entry, or a hooks file. */
async function hookFolder(path: string): Promise<boolean> {
  try {
    return (await readdir(path)).length > 0;
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code === 'ENOENT') return false;
    if (code === 'ENOTDIR') return true;
    return true;
  }
}

/** `hooks-paths`: any line that is not blank and not a comment names a hook source. */
async function hookPaths(path: string): Promise<boolean> {
  const text = await readSource(path);
  if (typeof text === 'boolean') return text;
  for (const raw of text.split('\n')) {
    const line = raw.trim();
    if (line !== '' && !line.startsWith('#')) return true;
  }
  return false;
}

/** The nearest folder at or above `cwd` that holds `.git`, else `cwd`. */
export async function gitRoot(cwd: string): Promise<string> {
  let dir = cwd;
  for (let step = 0; step < 64; step++) {
    if (await exists(join(dir, '.git')).catch(() => true)) return dir;
    const up = dirname(dir);
    if (up === dir) break;
    dir = up;
  }
  return cwd;
}

export interface HookPlace {
  home: string;
  /** The agent home of Work (`GROK_HOME`). */
  agentHome: string;
  cwd: string;
}

/**
 * Whether a hook source of the agent other than the app's guard holds a tool hook (a file check of the sources the
 * agent reads; a source that cannot be read counts). A fact for the texts only: it never makes a call ask. Only the
 * answer leaves this function, never a file's content.
 */
export async function findUserHooks(place: HookPlace): Promise<boolean> {
  const root = await gitRoot(place.cwd);
  const folders = root === place.cwd ? [place.cwd] : [place.cwd, root];
  const checks: Array<Promise<boolean>> = [
    settingsHook(join(place.home, '.claude', 'settings.json')),
    settingsHook(join(place.home, '.claude', 'settings.local.json')),
    settingsHook(join(place.home, '.cursor', 'hooks.json')),
    hookFolder(join(place.agentHome, 'hooks')),
    hookPaths(join(place.agentHome, 'hooks-paths')),
    tomlHook(join(place.agentHome, 'config.toml')),
    tomlHook(join(place.agentHome, 'managed_config.toml')),
    tomlHook(`${sep}etc${sep}grok${sep}managed_config.toml`),
  ];
  for (const folder of folders) {
    checks.push(
      settingsHook(join(folder, '.claude', 'settings.json')),
      settingsHook(join(folder, '.claude', 'settings.local.json')),
      settingsHook(join(folder, '.cursor', 'hooks.json')),
      hookFolder(join(folder, '.grok', 'hooks')),
      tomlHook(join(folder, '.grok', 'config.toml')),
    );
  }
  const found = await Promise.all(checks);
  return found.some((f) => f);
}
