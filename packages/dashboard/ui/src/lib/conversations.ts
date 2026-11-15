/** Conversation list helpers: day groups (local time) and the search match. */

import { CHAT } from './chat-strings';

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
): Array<{ label: string; items: T[] }> {
  const sorted = [...rows].sort((a, b) => b.updatedAt - a.updatedAt);
  const groups: Array<{ label: string; items: T[] }> = [];
  for (const row of sorted) {
    const label = dayGroupLabel(row.updatedAt, now);
    const last = groups[groups.length - 1];
    if (last !== undefined && last.label === label) last.items.push(row);
    else groups.push({ label, items: [row] });
  }
  return groups;
}

/** The query (lower-cased) is part of the title, or of the folder of a Work row. */
export function matchesQuery(row: { title: string; folder?: string | null }, query: string): boolean {
  const needle = query.trim().toLowerCase();
  if (needle === '') return true;
  return row.title.toLowerCase().includes(needle) || (row.folder ?? '').toLowerCase().includes(needle);
}
