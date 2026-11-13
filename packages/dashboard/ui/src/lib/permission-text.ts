/** Texts of Work permission cards and of the permission marks in the thread. */

import { CHAT, fill } from './chat-strings';
import type { AutoAsk, CommandFlag, PermissionInteraction, PermissionMark, PermissionMode } from './chat-types';
import type { TranscriptEntry } from './types';
import { oneLine } from './web-tools';

/** Warning lines in the Control Panel's order. */
const FLAG_ORDER: readonly CommandFlag[] = [
  'elevated',
  'deletes',
  'outside-folder',
  'installs',
  'credentials',
  'prints-secrets',
  'provider-jobs',
  'unchecked',
];

/** The warning lines of a card, in the fixed order. */
export function flagLines(flags: readonly CommandFlag[]): string[] {
  return FLAG_ORDER.filter((flag) => flags.includes(flag)).map((flag) => CHAT.permFlags[flag]);
}

/** The warning lines of a permission card: an edit whose file the app cannot tell says so instead of "outside". */
export function cardWarnings(interaction: Pick<PermissionInteraction, 'flags' | 'landingUnknown'>): string[] {
  const lines = flagLines(interaction.flags);
  if (interaction.landingUnknown !== true) return lines;
  const outside = CHAT.permFlags['outside-folder'];
  return [CHAT.permLandingUnknown, ...lines.filter((line) => line !== outside)];
}

/** The last segment of a path: the folder's own name. */
export function folderName(path: string): string {
  let end = path.length;
  while (end > 1 && path[end - 1] === '/') end -= 1;
  const start = path.lastIndexOf('/', end - 1) + 1;
  const name = path.slice(start, end);
  return name === '' ? path : name;
}

function folderWords(folder: string | null): string {
  return folder === null ? CHAT.permission.theFolder : folderName(folder);
}

/**
 * The temp folder Auto covers for a Work folder, as people type it: none when the Work folder holds or equals the
 * temp root's real top (the server then covers no temp root for that conversation).
 */
export function tempFor(
  host: { tempRoot: string | null; tempTop: string | null } | null | undefined,
  folder: string | null,
): string | null {
  const temp = host?.tempRoot ?? null;
  const top = host?.tempTop ?? null;
  if (temp === null || top === null || folder === null) return temp;
  const holds = top === folder || top.startsWith(folder.endsWith('/') ? folder : `${folder}/`);
  return holds ? null : temp;
}

/** Where Auto lets writes land: the Work folder, or "the folder or /tmp" when Auto covers a temp folder too. */
export function placeWords(folder: string | null, temp: string | null): string {
  const words = folderWords(folder);
  return temp === null ? words : fill(CHAT.permission.orTemp, { folder: words, temp });
}

export type AutoAskDetail = NonNullable<PermissionInteraction['autoAskDetail']>;

/** A part as the reason line draws it: in double quotes; the Work folder by its name. */
function partWords(part: string, folder: string | null): string {
  return `"${folder !== null && part === folder ? folderName(folder) : part}"`;
}

/**
 * The line of a card that still asks while Auto is on. With a detail it names the part that asks; `top` and
 * `git-global` name none.
 */
export function autoAskText(
  why: AutoAsk,
  folder: string | null,
  temp: string | null = null,
  detail?: AutoAskDetail,
): string {
  const p = CHAT.permission;
  const values = { folder: folderWords(folder), place: placeWords(folder, temp), temp: temp ?? p.theTemp };
  if (detail === undefined) return p.asksStart + fill(p.asks[why], values);
  const part = partWords(detail.part, folder);
  if (detail.cause === null) {
    const key = detail.more <= 0 ? 'not-known' : detail.more === 1 ? 'not-known-one' : 'not-known-more';
    return p.asksStart + fill(p.asksPart[key], { ...values, part, n: detail.more });
  }
  return p.asksStart + fill(p.asksPart[detail.cause], { ...values, part });
}

export type AllowedMark = Extract<PermissionMark, { kind: 'allowed' }>;
export type LineMark = Exclude<PermissionMark, { kind: 'allowed' }>;

const ORDER: Record<PermissionMode, number> = { ask: 0, auto: 1, bypass: 2 };

/** Ask < Auto < Bypass. */
export function modeRank(mode: PermissionMode): number {
  return ORDER[mode];
}

export interface MarkLine {
  label: string;
  /** One short sentence on the mark's line. */
  fact: string;
  /** Sentences of the second line, which the owner may see nowhere else. */
  detail: string[];
  tone: 'neutral' | 'warning' | 'danger';
}

/** The word of a call a mode allowed: "Allowed by auto · /tmp" when a real place of the call is in the temp folder. */
export function allowedWord(mark: AllowedMark, temp: string | null): string {
  const p = CHAT.permission;
  if (mark.mode === 'auto' && mark.temp === true && temp !== null) return fill(p.allowed.autoTemp, { temp });
  return p.allowed[mark.mode];
}

/**
 * A mode or settings mark: its line, and the sentences under it. A settings mark's files are {@link changedPaths}.
 * `temp`: the temp folder as people type it, named on an Auto mark whose confirm named it.
 */
export function markLine(mark: LineMark, temp: string | null = null): MarkLine {
  const p = CHAT.permission;
  if (mark.kind === 'settings') return { label: p.settingsLabel, fact: p.settingsLine, detail: [], tone: 'warning' };
  const label = p.line[mark.mode];
  if (mark.mode === 'bypass') {
    return { label, fact: p.bypass.line, detail: mark.userHooks === true ? [p.bypass.lineHooks] : [], tone: 'danger' };
  }
  if (mark.mode === 'auto') {
    const detail: string[] = [];
    if (modeRank(mark.from) > modeRank('auto')) detail.push(p.lineAutoDown);
    if (mark.editsUnasked === true) detail.push(p.lineEditsUnasked);
    if (mark.userHooks === true) detail.push(p.lineHooks);
    const fact = mark.temp === true && temp !== null ? fill(p.lineAutoTemp, { temp }) : p.lineAuto;
    return { label, fact, detail, tone: 'neutral' };
  }
  const ended = { mode: p.modes[mark.from] };
  const fact =
    mark.by === 'restart'
      ? fill(p.askRestart, ended)
      : mark.by === 'record-failed'
        ? fill(p.askRecord, ended)
        : p.askUser;
  return { label, fact, detail: [], tone: 'neutral' };
}

/** Paths a settings mark names before it counts the rest. */
export const CHANGED_SHOWN = 2;

/** The changed files a settings mark names: the first {@link CHANGED_SHOWN}, and how many more there are. */
export function changedPaths(changed: readonly string[]): { paths: string[]; more: number } {
  return { paths: changed.slice(0, CHANGED_SHOWN), more: Math.max(0, changed.length - CHANGED_SHOWN) };
}

/** A path longer than this is shown by its first and last segments. */
export const PATH_SHOWN_MAX = 56;

/**
 * A path short enough for one line of a mark or a toast: over {@link PATH_SHOWN_MAX} characters it keeps its first
 * named segment (with a leading `~` or `/`) and its last two, with `…` for the middle.
 */
export function shortPath(path: string): string {
  if (path.length <= PATH_SHOWN_MAX) return path;
  const parts = path.split('/');
  let head = 0;
  while (head < parts.length && (parts[head] === '' || parts[head] === '~')) head += 1;
  const tailFrom = parts.length - 2;
  if (head + 1 >= tailFrom) return path;
  const short = `${parts.slice(0, head + 1).join('/')}/…/${parts.slice(tailFrom).join('/')}`;
  return short.length < path.length ? short : path;
}

/** "Changed: a.", "Changed: a and b.", "Changed: a, b and 3 more." with each path shortened; empty when none is named. */
export function changedText(changed: readonly string[]): string {
  const p = CHAT.permission;
  const { paths, more } = changedPaths(changed);
  if (paths.length === 0) return '';
  let list = shortPath(paths[0]!);
  if (more > 0) list += `, ${shortPath(paths[1]!)} ${fill(p.settingsMore, { n: more })}`;
  else if (paths.length === 2) list += ` ${p.settingsAnd} ${shortPath(paths[1]!)}`;
  return `${p.settingsChanged} ${list}.`;
}

/** The toast of a settings change: what it means, then the files. */
export function settingsToast(changed: readonly string[]): string {
  const files = changedText(changed);
  return files === '' ? CHAT.permission.settingsText : `${CHAT.permission.settingsText} ${files}`;
}

const SPACES = [' ', ' ', '　'];
const SKIP = [...SPACES, '—', '–', '-', ':', ','];
/** The only words a command or file card's title can be when it says nothing more: the agent's verbs and tool names. */
const BARE_TITLES = ['Execute', 'Edit', 'Write', 'run_terminal_command', 'search_replace', 'write'];
/** A cut title goes back to its last space only when the word it drops is at most this long. */
const WORD_BACK = 24;

function firstSpace(text: string): number {
  for (let at = 0; at < text.length; at += 1) if (SPACES.includes(text[at]!)) return at;
  return -1;
}

/**
 * What a card shows of the agent's own title: null when it only names the command or file that the well shows (whole,
 * or cut short by the server); only the words after it when more follow; else the title. The target counts as named
 * only where it ends at a word edge, so a longer word that starts with it is the agent's own text. A title the server
 * cut ends with "…", at a whole word when the last word is short.
 */
export function cardDescription(
  i: Pick<PermissionInteraction, 'access' | 'title' | 'command' | 'path' | 'titleCut'>,
): string | null {
  const text = oneLine(i.title).trim();
  if (text === '') return null;
  const target = i.access === 'bash' ? i.command : i.access === 'edit' ? i.path : null;
  const want = target === null ? '' : oneLine(target).trim();
  const cut = i.titleCut === true;
  let shown = text;
  if (want !== '') {
    const space = firstSpace(text);
    if (space === -1 && BARE_TITLES.includes(text)) return null;
    const starts = space === -1 ? [text] : [text.slice(space + 1), text];
    for (const start of starts) {
      const ticked = start.startsWith('`');
      const named = ticked ? start.slice(1) : start;
      const whole = named.endsWith('`') ? named.slice(0, -1) : named;
      if (whole === want || (cut && whole !== '' && want.startsWith(whole))) return null;
      if (!named.startsWith(want)) continue;
      let at = want.length;
      const edge = named[at];
      if (ticked ? edge !== '`' : edge !== undefined && !SPACES.includes(edge)) continue;
      if (edge === '`') at += 1;
      while (at < named.length && SKIP.includes(named[at]!)) at += 1;
      if (at === named.length) return null;
      shown = named.slice(at);
      break;
    }
  }
  if (cut) {
    let end = shown.length;
    let back = end;
    while (back > 0 && !SPACES.includes(shown[back - 1]!)) back -= 1;
    if (back > 0 && end - back <= WORD_BACK) end = back;
    while (end > 0 && SKIP.includes(shown[end - 1]!)) end -= 1;
    shown = `${shown.slice(0, end)}…`;
  }
  return shown;
}

/** What an allowed call did, in one line: the first line of its command or path, else what kind of call it was. */
export function allowedSummary(mark: AllowedMark): string {
  const p = CHAT.permission;
  const fallback = mark.access === 'bash' ? p.allowedCommand : mark.access === 'edit' ? p.allowedEdit : p.allowedTool;
  const text = mark.card === null ? null : (mark.card.command ?? mark.card.path);
  if (text === null || text === undefined) return fallback;
  const cut = text.indexOf('\n');
  const first = cut === -1 ? text : text.slice(0, cut);
  return first === '' ? fallback : first;
}

export interface PlacedMarks {
  /** Lines drawn before the entry at that index. */
  before: Map<number, PermissionMark[]>;
  /** Lines drawn after the last entry. */
  end: PermissionMark[];
  /** The allowed mark of a tool row, by its tool call id. */
  allowed: Map<string, AllowedMark>;
}

/**
 * Where each mark goes. An allowed call whose tool row is in the thread marks that row; every other mark is a line
 * before the first entry that is later than it, else after the last entry.
 */
export function placeMarks(entries: readonly TranscriptEntry[], marks: readonly PermissionMark[]): PlacedMarks {
  const rows = new Set<string>();
  // Running maximum of the entry times: the first entry later than a time is where this maximum first passes it.
  const top: number[] = [];
  let max = Number.NEGATIVE_INFINITY;
  for (const entry of entries) {
    if (entry.kind === 'tool') rows.add(entry.id);
    const ts = entry.ts ?? 0;
    if (ts > max) max = ts;
    top.push(max);
  }
  const placed: PlacedMarks = { before: new Map(), end: [], allowed: new Map() };
  for (const mark of marks) {
    if (mark.kind === 'allowed' && rows.has(mark.toolCallId)) {
      placed.allowed.set(mark.toolCallId, mark);
      continue;
    }
    let lo = 0;
    let hi = top.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (top[mid]! > mark.at) hi = mid;
      else lo = mid + 1;
    }
    if (lo === top.length) {
      placed.end.push(mark);
      continue;
    }
    const list = placed.before.get(lo);
    if (list === undefined) placed.before.set(lo, [mark]);
    else list.push(mark);
  }
  return placed;
}
