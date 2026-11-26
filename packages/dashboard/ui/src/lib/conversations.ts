/** Conversation list helpers: Chat day groups and Work folder groups (local time), the cap of a long group, the search match. */

import { CHAT } from './chat-strings';
import { folderName } from './permission-text';

const DAY_MS = 86_400_000;

function startOfDay(ms: number): number {
  const date = new Date(ms);
  date.setHours(0, 0, 0, 0);
  return date.getTime();
}

const MONTH = new Intl.DateTimeFormat('en-US', { month: 'long', year: 'numeric' });

/** The label of the group a time belongs to, seen from `now`. */
export function dayGroupLabel(ms: number, now: number): string {
  const today = startOfDay(now);
  if (ms >= today) return CHAT.groups[0];
  const yesterday = startOfDay(today - DAY_MS / 2);
  if (ms >= yesterday) return CHAT.groups[1];
  if (ms >= today - 7 * DAY_MS) return CHAT.groups[2];
  if (ms >= today - 30 * DAY_MS) return CHAT.groups[3];
  return MONTH.format(new Date(ms));
}

/** Newest first, in groups Today · Yesterday · Previous 7 days · Previous 30 days · one per month. */
export function groupByDay<T extends { updatedAt: number }>(
  rows: readonly T[],
  now: number,
): Array<{ key: string; label: string; items: T[] }> {
  const sorted = [...rows].sort((a, b) => b.updatedAt - a.updatedAt);
  const groups: Array<{ key: string; label: string; items: T[] }> = [];
  for (const row of sorted) {
    const label = dayGroupLabel(row.updatedAt, now);
    const last = groups[groups.length - 1];
    if (last !== undefined && last.label === label) last.items.push(row);
    else groups.push({ key: label, label, items: [row] });
  }
  return groups;
}

/** The key of the group of conversations started without a folder. No real path is empty. */
export const NO_FOLDER_KEY = '';

export interface FolderGroup<T> {
  /** The folder's path as stored, or NO_FOLDER_KEY. */
  key: string;
  /** The folder's own name; null for the No folder group, whose words the page puts. */
  name: string | null;
  /** Parent folders that tell this group from another of the same name. */
  hint: string | null;
  /** Newest first. */
  items: T[];
}

type Placed = { updatedAt: number; folder?: string | null; noFolder?: boolean };

/** A Work row's group: its folder as stored (no trim, no case fold), else NO_FOLDER_KEY. */
export function placeKey(row: Placed): string {
  if (row.noFolder === true || row.folder === undefined || row.folder === null) return NO_FOLDER_KEY;
  return row.folder;
}

const NAMES = new Intl.Collator('en', { numeric: true, sensitivity: 'base' });

function byPath(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

/**
 * One group per folder. Groups order by the local day of their newest row, newest day first; within a day the No
 * folder group is last and the others go by name, then by path. Nothing here reads the clock.
 */
export function groupByFolder<T extends Placed>(rows: readonly T[]): FolderGroup<T>[] {
  const byKey = new Map<string, T[]>();
  for (const row of [...rows].sort((a, b) => b.updatedAt - a.updatedAt)) {
    const key = placeKey(row);
    const items = byKey.get(key);
    if (items === undefined) byKey.set(key, [row]);
    else items.push(row);
  }
  const hints = folderHints([...byKey.keys()].filter((key) => key !== NO_FOLDER_KEY));
  const groups = [...byKey].map(([key, items]) => ({
    day: startOfDay(items[0]!.updatedAt),
    group: {
      key,
      name: key === NO_FOLDER_KEY ? null : folderName(key),
      hint: hints.get(key) ?? null,
      items,
    },
  }));
  groups.sort((a, b) => {
    if (a.day !== b.day) return b.day - a.day;
    if (a.group.key === NO_FOLDER_KEY) return 1;
    if (b.group.key === NO_FOLDER_KEY) return -1;
    return NAMES.compare(a.group.name ?? '', b.group.name ?? '') || byPath(a.group.key, b.group.key);
  });
  return groups.map((entry) => entry.group);
}

/** A folder group's name as a screen reader says it: with its hint when another folder shares the name. */
export function spokenFolder(group: { name: string; hint: string | null }): string {
  return group.hint === null ? group.name : `${group.name}, ${group.hint}`;
}

function parentParts(path: string): string[] {
  const parts = path.split('/').filter((part) => part !== '');
  return parts.slice(0, -1);
}

/**
 * For folders that share a name (compared exactly): the fewest parent folders, read `a/b`, that tell all of that name
 * apart. A folder whose name no other has gets no entry.
 */
export function folderHints(paths: readonly string[]): Map<string, string> {
  const byName = new Map<string, string[]>();
  for (const path of paths) {
    const name = folderName(path);
    const same = byName.get(name);
    if (same === undefined) byName.set(name, [path]);
    else if (!same.includes(path)) same.push(path);
  }
  const hints = new Map<string, string>();
  for (const same of byName.values()) {
    if (same.length < 2) continue;
    const parents = same.map(parentParts);
    const deepest = Math.max(...parents.map((parts) => parts.length));
    let words: string[] = [];
    for (let depth = 1; depth <= Math.max(1, deepest); depth += 1) {
      words = parents.map((parts) => parts.slice(Math.max(0, parts.length - depth)).join('/') || '/');
      if (new Set(words).size === same.length) break;
    }
    same.forEach((path, index) => hints.set(path, words[index]!));
  }
  return hints;
}

/** Rows an open group shows before its "Show n more" row. */
export const GROUP_CAP = 5;

/**
 * The rows a group shows: all of a short group, else the newest `cap` plus every row that runs, waits or is open, in
 * their time order. A cut that would hide one row hides none.
 */
export function shownRows<T extends { id: string; state: 'idle' | 'running' | 'waiting' }>(
  items: readonly T[],
  all: boolean,
  currentId: string | null,
  cap = GROUP_CAP,
): { shown: T[]; hidden: number } {
  if (all) return { shown: [...items], hidden: 0 };
  const shown = items.filter((item, index) => index < cap || item.state !== 'idle' || item.id === currentId);
  const hidden = items.length - shown.length;
  return hidden < 2 ? { shown: [...items], hidden: 0 } : { shown, hidden };
}

/** A folded group's dot: a row that waits for the person wins over a row that runs. */
export function groupLive(items: readonly { state: string }[]): 'waiting' | 'running' | null {
  if (items.some((item) => item.state === 'waiting')) return 'waiting';
  return items.some((item) => item.state === 'running') ? 'running' : null;
}

/** What a search matches besides the title: a Work row's folder name, or "No folder". */
export function placeName(row: Placed & { mode: 'chat' | 'work' }): string | null {
  if (row.mode !== 'work') return null;
  const key = placeKey(row);
  return key === NO_FOLDER_KEY ? CHAT.folderNone : folderName(key);
}

/** The query (lower-cased) is part of the title, or of the folder's own name: never of the folders above it. */
export function matchesQuery(row: { title: string; folderName?: string | null }, query: string): boolean {
  const needle = query.trim().toLowerCase();
  if (needle === '') return true;
  return row.title.toLowerCase().includes(needle) || (row.folderName ?? '').toLowerCase().includes(needle);
}
