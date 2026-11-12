/** Texts of Work permission cards and of the permission marks in the thread. */

import { CHAT, fill } from './chat-strings';
import type { AutoAsk, CommandFlag, PermissionInteraction, PermissionMark, PermissionMode } from './chat-types';
import type { TranscriptEntry } from './types';

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

/** The line of a card that still asks while Auto is on. */
export function autoAskText(why: AutoAsk, folder: string | null): string {
  return CHAT.permission.asksStart + fill(CHAT.permission.asks[why], { folder: folderWords(folder) });
}

export type AllowedMark = Extract<PermissionMark, { kind: 'allowed' }>;
export type LineMark = Exclude<PermissionMark, { kind: 'allowed' }>;

const ORDER: Record<PermissionMode, number> = { ask: 0, auto: 1, bypass: 2 };

/** Ask < Auto < Bypass. */
export function modeRank(mode: PermissionMode): number {
  return ORDER[mode];
}

/** A mode or settings mark as one line of the thread. */
export function markLine(mark: LineMark): { label: string; text: string; tone: 'neutral' | 'warning' } {
  const p = CHAT.permission;
  if (mark.kind === 'settings') return { label: p.settingsLabel, text: p.settingsText, tone: 'warning' };
  const label = p.line[mark.mode];
  if (mark.mode === 'bypass') {
    const text = mark.userHooks === true ? `${p.bypass.line} ${p.bypass.lineHooks}` : p.bypass.line;
    return { label, text, tone: 'neutral' };
  }
  if (mark.mode === 'auto') {
    let text = modeRank(mark.from) > modeRank('auto') ? p.lineAutoDown : p.lineAuto;
    if (mark.editsUnasked === true) text += ` ${p.lineEditsUnasked}`;
    if (mark.userHooks === true) text += ` ${p.lineHooks}`;
    return { label, text, tone: 'neutral' };
  }
  const ended = { mode: p.modes[mark.from] };
  const text =
    mark.by === 'restart'
      ? fill(p.askRestart, ended)
      : mark.by === 'record-failed'
        ? fill(p.askRecord, ended)
        : p.askUser;
  return { label, text, tone: 'neutral' };
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
