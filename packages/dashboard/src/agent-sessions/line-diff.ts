/**
 * Line diffs of the agent's edits for the page: hunks with 3 lines of context, capped in rows and size, so a write of a
 * whole file never reaches the page as text. Pure data → data.
 */

import type { TranscriptEntry } from './types.js';

export type ToolEdit = NonNullable<Extract<TranscriptEntry, { kind: 'tool' }>['edits']>[number];
export type EditHunk = ToolEdit['hunks'][number];

/** Unchanged lines kept on each side of a change; changes this many times two apart share one hunk. */
export const DIFF_CONTEXT = 3;
/** Past this edit distance every old line is removed and every new line added. */
export const MAX_DISTANCE = 2_000;
/** Rows of all hunks of one tool entry. */
export const MAX_ROWS = 400;
/** Characters of all hunk bodies of one tool entry. */
export const MAX_BODY_CHARS = 24 * 1024;
/** Characters of one row's text; a longer row ends in `…`. */
export const MAX_ROW_CHARS = 400;

const SAME = 0;
const REMOVED = 1;
const ADDED = 2;

/** One replaced match of a result: the agent's strings and its place in the new file (`_meta.details[]`). */
export interface RawDetail {
  oldString: string;
  newString: string;
  /** 1-based line of the new file where the match starts. */
  line: number;
  /** Up to 3 whole lines of the new file before and after the match. */
  before: string;
  after: string;
  /** The start of the match's first line, when the match starts mid-line. */
  prefix: string;
}

/** One `diff` content block, read off the wire. */
export interface RawEdit {
  path: string;
  /** `''` when the agent sent none. */
  oldText: string;
  newText: string;
  /** A result's replaced matches; null for the asked-for change and for a block with none that fit. */
  details: RawDetail[] | null;
  /** The asked-for change's `_meta.old_line` and `new_line`; null when the agent named none. */
  line: [number, number] | null;
  /** A result with no old text: the file is new. */
  create: boolean;
}

/** `text` as lines: a last `\n` ends the last line, a `\r` before a `\n` or at the end is dropped. */
export function splitLines(text: string): string[] {
  const lines: string[] = [];
  let from = 0;
  while (from < text.length) {
    const end = text.indexOf('\n', from);
    const stop = end === -1 ? text.length : end;
    const cut = stop > from && text.charCodeAt(stop - 1) === 13 ? stop - 1 : stop;
    lines.push(text.slice(from, cut));
    if (end === -1) break;
    from = end + 1;
  }
  return lines;
}

function newlines(text: string): number {
  let n = 0;
  for (let i = text.indexOf('\n'); i !== -1; i = text.indexOf('\n', i + 1)) n += 1;
  return n;
}

/**
 * The line diff of `a` against `b`, one code per row in order: 0 unchanged, 1 removed (a line of `a`), 2 added (a
 * line of `b`). Myers O(ND) between the common start and end; past `maxDistance` the middle is all removed, then all
 * added.
 */
export function lineDiff(a: readonly string[], b: readonly string[], maxDistance: number = MAX_DISTANCE): Uint8Array {
  let head = 0;
  while (head < a.length && head < b.length && a[head] === b[head]) head += 1;
  let tail = 0;
  while (tail < a.length - head && tail < b.length - head && a[a.length - 1 - tail] === b[b.length - 1 - tail])
    tail += 1;
  const n = a.length - head - tail;
  const m = b.length - head - tail;
  const middle = n === 0 || m === 0 || Math.abs(n - m) > maxDistance ? null : myers(a, b, head, n, m, maxDistance);
  const out = new Uint8Array(head + (middle === null ? n + m : middle.length) + tail);
  if (middle !== null) out.set(middle, head);
  else {
    out.fill(REMOVED, head, head + n);
    out.fill(ADDED, head + n, head + n + m);
  }
  return out;
}

/** The rows of `a[head, head + n)` against `b[head, head + m)`, or null past `maxDistance`. */
function myers(
  a: readonly string[],
  b: readonly string[],
  head: number,
  n: number,
  m: number,
  maxDistance: number,
): Uint8Array | null {
  const ids = new Map<string, number>();
  const id = (line: string): number => {
    let value = ids.get(line);
    if (value === undefined) {
      value = ids.size;
      ids.set(line, value);
    }
    return value;
  };
  const x0 = new Int32Array(n);
  for (let i = 0; i < n; i += 1) x0[i] = id(a[head + i]!);
  const y0 = new Int32Array(m);
  for (let j = 0; j < m; j += 1) y0[j] = id(b[head + j]!);
  const limit = Math.min(maxDistance, n + m);
  const offset = limit + 1;
  const v = new Int32Array(2 * limit + 3);
  // trace[d]: v of k in [-d, d] before round d.
  const trace: Int32Array[] = [];
  for (let d = 0; d <= limit; d += 1) {
    trace.push(v.slice(offset - d, offset + d + 1));
    for (let k = -d; k <= d; k += 2) {
      let x =
        k === -d || (k !== d && v[offset + k - 1]! < v[offset + k + 1]!) ? v[offset + k + 1]! : v[offset + k - 1]! + 1;
      let y = x - k;
      while (x < n && y < m && x0[x] === y0[y]) {
        x += 1;
        y += 1;
      }
      v[offset + k] = x;
      if (x >= n && y >= m) return path(trace, d, n, m);
    }
  }
  return null;
}

function path(trace: readonly Int32Array[], distance: number, n: number, m: number): Uint8Array {
  // Each step of the path removes or adds one row; the rest of n + m pairs up unchanged.
  const rows = new Uint8Array((n + m + distance) / 2);
  let at = rows.length;
  let x = n;
  let y = m;
  for (let d = distance; d > 0; d -= 1) {
    const before = trace[d]!;
    const k = x - y;
    const down = k === -d || (k !== d && before[k - 1 + d]! < before[k + 1 + d]!);
    const pk = down ? k + 1 : k - 1;
    const px = before[pk + d]!;
    const py = px - pk;
    const snake = down ? x - px : y - py;
    at -= snake;
    x -= snake;
    y -= snake;
    at -= 1;
    rows[at] = down ? ADDED : REMOVED;
    if (down) y -= 1;
    else x -= 1;
  }
  return rows;
}

/** A row's text cut to {@link MAX_ROW_CHARS}, never inside a surrogate pair. */
function rowText(line: string): string {
  if (line.length <= MAX_ROW_CHARS) return line;
  let cut = MAX_ROW_CHARS;
  const code = line.charCodeAt(cut - 1);
  if (code >= 0xd800 && code <= 0xdbff) cut -= 1;
  return `${line.slice(0, cut)}…`;
}

/** What is left of the entry's row and size caps. */
export interface Budget {
  rows: number;
  chars: number;
}

export interface Segment {
  old: string[];
  new: string[];
  /** Old and new number of the segment's first line; null when the agent named no place. */
  base: [number, number] | null;
}

/**
 * The hunks of one segment, rows taken from `budget`: runs of unchanged lines longer than twice {@link DIFF_CONTEXT}
 * fold away (a segment with no place stays one hunk). `cut` counts the changed rows the budget left out.
 */
export function hunksOf(
  segment: Segment,
  budget: Budget,
): { hunks: EditHunk[]; added: number; removed: number; cut: number } {
  const rows = lineDiff(segment.old, segment.new);
  let added = 0;
  let removed = 0;
  for (const row of rows) {
    if (row === ADDED) added += 1;
    else if (row === REMOVED) removed += 1;
  }
  if (added + removed === 0) return { hunks: [], added, removed, cut: 0 };
  const spans: Array<[number, number]> = [];
  if (segment.base === null) spans.push([0, rows.length]);
  else {
    let i = 0;
    while (i < rows.length) {
      if (rows[i] === SAME) {
        i += 1;
        continue;
      }
      let end = i;
      while (end < rows.length && rows[end] !== SAME) end += 1;
      const from = Math.max(0, i - DIFF_CONTEXT);
      const to = Math.min(rows.length, end + DIFF_CONTEXT);
      const last = spans[spans.length - 1];
      if (last !== undefined && from <= last[1]) last[1] = to;
      else spans.push([from, to]);
      i = end;
    }
  }
  const hunks: EditHunk[] = [];
  let cut = 0;
  let index = 0;
  let oi = 0;
  let ni = 0;
  const oldBase = segment.base === null ? 0 : segment.old.length === 0 ? 0 : segment.base[0];
  for (const [from, to] of spans) {
    while (index < from) {
      if (rows[index] !== ADDED) oi += 1;
      if (rows[index] !== REMOVED) ni += 1;
      index += 1;
    }
    const start: [number, number] | null = segment.base === null ? null : [oldBase + oi, segment.base[1] + ni];
    const body: string[] = [];
    for (; index < to; index += 1) {
      const row = rows[index]!;
      const text = row === ADDED ? segment.new[ni]! : segment.old[oi]!;
      if (row !== ADDED) oi += 1;
      if (row !== REMOVED) ni += 1;
      if (budget.rows === 0) {
        if (row !== SAME) cut += 1;
        continue;
      }
      const line = `${row === SAME ? ' ' : row === REMOVED ? '-' : '+'}${rowText(text)}`;
      if (line.length + 1 > budget.chars) {
        budget.rows = 0;
        if (row !== SAME) cut += 1;
        continue;
      }
      budget.rows -= 1;
      budget.chars -= line.length + 1;
      body.push(line);
    }
    if (body.length > 0) hunks.push({ start, body: body.join('\n') });
  }
  return { hunks, added, removed, cut };
}

/** A segment still open for the next match: the after context of its last match is not in it yet. */
interface OpenSegment {
  segment: Segment;
  after: string[];
  /** New-file line where `after` starts; null when the line there is unknown. */
  end: number | null;
}

function closed(open: OpenSegment): Segment {
  open.segment.old.push(...open.after);
  open.segment.new.push(...open.after);
  return open.segment;
}

/**
 * The segments of a result's replaced matches, in their order. Matches whose context lines meet share one segment,
 * so no line shows twice.
 */
function matchSegments(edit: RawEdit, details: readonly RawDetail[]): Segment[] {
  // Each match before this one moved it by the change in line count; the agent names new-file lines only.
  const shift = newlines(edit.newText) - newlines(edit.oldText);
  const segments: Segment[] = [];
  let open: OpenSegment | null = null;
  for (let i = 0; i < details.length; i += 1) {
    const detail = details[i]!;
    const before = splitLines(detail.before);
    const oldRows = splitLines(detail.prefix + detail.oldString);
    const newRows = splitLines(detail.prefix + detail.newString);
    // A deleted block leaves no row, yet the agent counts the line after it as the match's: its after context starts one line later.
    const end = newRows.length === 0 ? null : detail.line + newRows.length;
    const after = end === null ? [] : splitLines(detail.after);
    const gap = open === null || open.end === null ? -1 : detail.line - open.end;
    if (open !== null && gap >= 0 && gap <= open.after.length + before.length) {
      const kept = Math.min(gap, open.after.length);
      const between = [...open.after.slice(0, kept), ...before.slice(before.length - (gap - kept))];
      open.segment.old.push(...between, ...oldRows);
      open.segment.new.push(...between, ...newRows);
      open.after = after;
      open.end = end;
      continue;
    }
    if (open !== null) segments.push(closed(open));
    const line = detail.line - before.length;
    open = {
      segment: { old: [...before, ...oldRows], new: [...before, ...newRows], base: [line - i * shift, line] },
      after,
      end,
    };
  }
  if (open !== null) segments.push(closed(open));
  return segments;
}

/** The segments of one diff block: its replaced matches when the result names them, else its old and new text. */
function segmentsOf(edit: RawEdit): Segment[] {
  const details = edit.details;
  if (details !== null && details.every((d) => d.oldString === edit.oldText && d.newString === edit.newText)) {
    return matchSegments(edit, details);
  }
  const base: [number, number] | null = edit.line ?? (details !== null || edit.oldText === '' ? [1, 1] : null);
  return [{ old: splitLines(edit.oldText), new: splitLines(edit.newText), base }];
}

/**
 * The page's form of a tool's diff blocks, in their order. Rows and size are capped for the whole entry
 * ({@link MAX_ROWS}, {@link MAX_BODY_CHARS}); `cut` counts each file's changed rows the caps left out.
 */
export function diffEdits(edits: readonly RawEdit[]): ToolEdit[] {
  const budget: Budget = { rows: MAX_ROWS, chars: MAX_BODY_CHARS };
  return edits.map((edit) => {
    const hunks: EditHunk[] = [];
    let added = 0;
    let removed = 0;
    let cut = 0;
    for (const segment of segmentsOf(edit)) {
      const part = hunksOf(segment, budget);
      hunks.push(...part.hunks);
      added += part.added;
      removed += part.removed;
      cut += part.cut;
    }
    return {
      path: edit.path,
      ...(edit.create ? { create: true as const } : {}),
      added,
      removed,
      hunks,
      ...(cut > 0 ? { cut } : {}),
      ...(added + removed === 0 && edit.oldText !== edit.newText ? { endsOnly: true as const } : {}),
    };
  });
}
