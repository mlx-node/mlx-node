/**
 * Tail-incremental reading of an append-only JSONL file.
 *
 * A pass reads only the bytes appended since the previous pass, as long as the
 * file grew and the bytes just before the previous end are unchanged. Anything
 * else (a new file, a shrunk or same-size file, a file rewritten in place) is
 * read whole. Only complete lines
 * are returned: a torn last line stays on disk for the next pass.
 */

import { closeSync, openSync, readSync, statSync, type Stats } from 'node:fs';
import { join } from 'node:path';

/** How many bytes before `consumed` prove that a file only grew. */
const TAIL_BYTES = 64;
const NEWLINE = 10;

export interface JsonlMark {
  /** Floored `mtimeMs`. */
  mtime: number;
  size: number;
  /** Bytes ingested; always ends on a `\n` (or is 0). */
  consumed: number;
  /** Base64 of the ≤ 64 bytes before `consumed`. */
  tail: string;
}

export type JsonlIncrement = { mode: 'skip' } | { mode: 'append' | 'replace'; lines: string[]; mark: JsonlMark };

function readRange(fd: number, start: number, length: number): Buffer {
  const buf = Buffer.alloc(Math.max(length, 0));
  let filled = 0;
  while (filled < buf.length) {
    const n = readSync(fd, buf, filled, buf.length - filled, start + filled);
    if (n === 0) break;
    filled += n;
  }
  return filled === buf.length ? buf : buf.subarray(0, filled);
}

/** Non-blank lines of `chunk` up to its last `\n`, and how many bytes that was. */
function wholeLines(chunk: Buffer): { lines: string[]; bytes: number } {
  const end = chunk.lastIndexOf(NEWLINE);
  if (end < 0) return { lines: [], bytes: 0 };
  const lines: string[] = [];
  for (const line of chunk.toString('utf8', 0, end).split('\n')) {
    const trimmed = line.trim();
    if (trimmed !== '') lines.push(trimmed);
  }
  return { lines, bytes: end + 1 };
}

function tailBefore(fd: number, consumed: number): string {
  const len = Math.min(TAIL_BYTES, consumed);
  return readRange(fd, consumed - len, len).toString('base64');
}

/**
 * Decide how much of `filePath` to read given its `stat` and the mark of the
 * previous pass, and read it. Throws only when the file cannot be opened or read.
 */
export function readJsonlIncrement(filePath: string, stat: Stats, previous: JsonlMark | undefined): JsonlIncrement {
  const mtime = Math.floor(stat.mtimeMs);
  const size = stat.size;
  if (previous !== undefined && previous.mtime === mtime && previous.size === size) return { mode: 'skip' };

  const fd = openSync(filePath, 'r');
  try {
    if (previous !== undefined && previous.consumed > 0 && size > previous.size) {
      if (tailBefore(fd, previous.consumed) === previous.tail) {
        const chunk = readRange(fd, previous.consumed, size - previous.consumed);
        const { lines, bytes } = wholeLines(chunk);
        const consumed = previous.consumed + bytes;
        return {
          mode: 'append',
          lines,
          mark: { mtime, size, consumed, tail: bytes > 0 ? tailBefore(fd, consumed) : previous.tail },
        };
      }
    }
    const { lines, bytes } = wholeLines(readRange(fd, 0, size));
    return { mode: 'replace', lines, mark: { mtime, size, consumed: bytes, tail: tailBefore(fd, bytes) } };
  } finally {
    closeSync(fd);
  }
}

/** A JSON object line, or null for a blank, malformed, scalar or array line. */
export function parseJsonObjectLine(line: string): Record<string, unknown> | null {
  let rec: unknown;
  try {
    rec = JSON.parse(line);
  } catch {
    return null;
  }
  return typeof rec === 'object' && rec !== null && !Array.isArray(rec) ? (rec as Record<string, unknown>) : null;
}

/**
 * Stat every `*.jsonl` entry ONCE so liveness is decided from what an entry IS, not
 * from what it is named. Only a regular file is a trace source: a directory or a FIFO
 * can carry the suffix too, and neither is readable as one. A directory fails
 * `readFileSync` with EISDIR (and `unlinkSync` with EPERM), so judged by name it stays
 * "live" forever and keeps its rows and watermark in the index; a FIFO is worse —
 * `readFileSync` BLOCKS on it until a writer closes, and since the server funnels every
 * pass through one serialized ingest chain that wedges the whole dashboard. Neither is
 * reachable from product code (the writer only ever appends to `<date>-<pid>.jsonl`),
 * but the scan cannot judge entries by name alone.
 *
 * A stat FAILURE is NOT "gone": a root that is readable but not searchable (mode 0600)
 * lists its entries and then gives EACCES per child. Those names map to `null` and stay
 * live — rows kept, file skipped this pass — the same stance the root guard below takes,
 * so a transient fault never reads as a deletion.
 *
 * `statSync` FOLLOWS symlinks: a symlinked trace entry is a legitimate source and
 * ingests today; `unlinkSync` does not follow, so the 30-day prune can only ever remove
 * the link itself, never an external target.
 */
export function statJsonlEntries(dir: string, entries: string[]): Map<string, Stats | null> {
  const kinds = new Map<string, Stats | null>();
  for (const name of entries) {
    if (!name.endsWith('.jsonl')) continue;
    try {
      kinds.set(name, statSync(join(dir, name)));
    } catch {
      kinds.set(name, null);
    }
  }
  return kinds;
}

/** Names reconciliation must treat as live: every regular file, plus every unstattable entry. */
export function liveSourceNames(kinds: Map<string, Stats | null>): Set<string> {
  const live = new Set<string>();
  for (const [name, stat] of kinds) {
    if (stat === null || stat.isFile()) live.add(name);
  }
  return live;
}
