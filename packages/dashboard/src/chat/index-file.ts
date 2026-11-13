/**
 * `ROOT/conversations.json`: which sessions in the user's agent home the app created in Work mode. Ids and creation
 * times only: no title, no path. The main thread is its only writer; the database worker reads it.
 */

import { readFileSync } from 'node:fs';

import { writeFileAtomic } from './paths.js';

export interface WorkIndexEntry {
  id: string;
  createdAt: number;
}

export interface WorkIndex {
  version: 1;
  work: WorkIndexEntry[];
}

const EMPTY: WorkIndex = { version: 1, work: [] };

/** The file's content when it is a valid index; anything else (missing, broken, wrong shape) reads as empty. */
export function parseWorkIndex(text: string): WorkIndex {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    return { ...EMPTY, work: [] };
  }
  if (typeof raw !== 'object' || raw === null) return { ...EMPTY, work: [] };
  const record = raw as { version?: unknown; work?: unknown };
  if (record.version !== 1 || !Array.isArray(record.work)) return { ...EMPTY, work: [] };
  const work: WorkIndexEntry[] = [];
  const seen = new Set<string>();
  for (const item of record.work as unknown[]) {
    if (typeof item !== 'object' || item === null) continue;
    const { id, createdAt } = item as { id?: unknown; createdAt?: unknown };
    if (typeof id !== 'string' || id === '' || seen.has(id)) continue;
    if (typeof createdAt !== 'number' || !Number.isFinite(createdAt)) continue;
    seen.add(id);
    work.push({ id, createdAt });
  }
  return { version: 1, work };
}

/** Synchronous read for the database worker. */
export function readWorkIndexSync(path: string): WorkIndex {
  try {
    return parseWorkIndex(readFileSync(path, 'utf8'));
  } catch {
    return { ...EMPTY, work: [] };
  }
}

/** The ids of the index, for tagging Work rows. */
export function workIds(index: WorkIndex): Set<string> {
  return new Set(index.work.map((entry) => entry.id));
}

/** Temp file and rename, mode 0600. */
export async function writeWorkIndex(path: string, index: WorkIndex): Promise<void> {
  await writeFileAtomic(path, `${JSON.stringify({ version: 1, work: index.work })}\n`);
}
