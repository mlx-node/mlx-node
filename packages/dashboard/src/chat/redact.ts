/**
 * Secret redaction for everything sent to the page: each occurrence of a needle (a current or retired provider key,
 * the inference token) becomes `[redacted]`. Plain substring scans, longest needle first at each position.
 */

import type { TranscriptEntry } from '../agent-sessions/types.js';

export const REDACTED = '[redacted]';

/** Distinct non-empty needles, longest first. */
export function needleList(needles: Iterable<string>): string[] {
  const set = new Set<string>();
  for (const n of needles) if (n.length > 0) set.add(n);
  return [...set].sort((a, b) => b.length - a.length);
}

function matchAt(text: string, at: number, needles: readonly string[]): string | null {
  for (const n of needles) if (text.startsWith(n, at)) return n;
  return null;
}

/** Redacts `text[0, stop)`; a match may run past `stop`. Returns the output and where scanning stopped. */
function scan(text: string, needles: readonly string[], stop: number): { out: string; next: number } {
  let out = '';
  let from = 0;
  let i = 0;
  while (i < stop) {
    const hit = matchAt(text, i, needles);
    if (hit === null) {
      i++;
      continue;
    }
    out += text.slice(from, i) + REDACTED;
    i += hit.length;
    from = i;
  }
  return { out: out + text.slice(from, Math.max(from, i)), next: Math.max(from, i) };
}

export function redactText(text: string, needles: readonly string[]): string {
  if (needles.length === 0 || text.length === 0) return text;
  return scan(text, needles, text.length).out;
}

/** The length of the longest end of `text` that is the start of a needle but not the whole needle. */
function needleStartAtEnd(text: string, needles: readonly string[]): number {
  let longest = 0;
  for (const n of needles) {
    for (let len = Math.min(n.length - 1, text.length); len > longest; len--) {
      if (text.endsWith(n.slice(0, len))) {
        longest = len;
        break;
      }
    }
  }
  return longest;
}

/**
 * Redacts text that was cut to length before it could be redacted: a key cut in two by the cut no longer matches, so
 * an end that could be the start of a key is dropped too.
 */
export function redactCutText(text: string, needles: readonly string[]): string {
  const out = redactText(text, needles);
  const drop = needleStartAtEnd(out, needles);
  return drop === 0 ? out : out.slice(0, out.length - drop);
}

/**
 * A title that may be a prompt cut to length with `…` (`promptTitle`): a key cut in two by the cut no longer matches,
 * so an end of the cut text that could be the start of a key is dropped too.
 */
export function redactTitle(title: string, needles: readonly string[]): string {
  if (needles.length === 0 || !title.endsWith('…')) return redactText(title, needles);
  return `${redactCutText(title.slice(0, -1), needles).trimEnd()}…`;
}

/**
 * Every string inside a JSON-like value, keys included; other values are copied as they are. A string under a `title`
 * key is redacted as a title that may be cut ({@link redactTitle}).
 */
export function redactDeep<T>(value: T, needles: readonly string[]): T {
  if (needles.length === 0) return value;
  const walk = (v: unknown): unknown => {
    if (typeof v === 'string') return redactText(v, needles);
    if (Array.isArray(v)) return v.map(walk);
    if (typeof v === 'object' && v !== null) {
      const out: Record<string, unknown> = {};
      for (const [k, item] of Object.entries(v)) {
        out[redactText(k, needles)] =
          k === 'title' && typeof item === 'string' ? redactTitle(item, needles) : walk(item);
      }
      return out;
    }
    return v;
  };
  return walk(value) as T;
}

/** A hunk body whose rows the reader may have cut (a cut row ends in `…`): a cut row's end that could start a key is dropped. */
function redactCutRows(body: string, needles: readonly string[]): string {
  return body
    .split('\n')
    .map((row) => (row.endsWith('…') ? `${redactCutText(row.slice(0, -1), needles)}…` : row))
    .join('\n');
}

/** A transcript entry for the page. A field the reader cut to length may end in part of a key, so that end is dropped. */
export function redactEntry(entry: TranscriptEntry, needles: readonly string[]): TranscriptEntry {
  const out = redactDeep({ ...entry }, needles);
  if (needles.length === 0) return out;
  if (out.kind === 'tool' && out.edits !== undefined) {
    out.edits = out.edits.map((edit) => ({
      ...edit,
      hunks: edit.hunks.map((hunk) => ({ ...hunk, body: redactCutRows(hunk.body, needles) })),
    }));
  }
  if (out.kind === 'tool' && out.truncated === true) {
    if (out.input !== undefined) out.input = redactCutText(out.input, needles);
    if (out.output !== undefined) out.output = redactCutText(out.output, needles);
  } else if (out.kind === 'wake' && out.truncated === true) {
    out.text = redactCutText(out.text, needles);
  } else if (out.kind === 'workflow' && out.report !== undefined) {
    out.report = redactReport(out.report, out.truncated === true, needles);
  } else if (out.kind === 'notice' && out.code === 'retry' && out.detail?.endsWith('…') === true) {
    out.detail = `${redactCutText(out.detail.slice(0, -1), needles)}…`;
  } else if (out.kind === 'notice' && out.code === 'picture') {
    out.notes = out.notes.map((note) => (note.endsWith('…') ? `${redactCutText(note.slice(0, -1), needles)}…` : note));
  } else if (out.kind === 'notice' && out.code === 'hook' && out.detail.endsWith('…')) {
    out.detail = `${redactCutText(out.detail.slice(0, -1), needles)}…`;
  } else if (out.kind === 'notice' && out.code === 'hook-note' && out.text.endsWith('…')) {
    out.text = `${redactCutText(out.text.slice(0, -1), needles)}…`;
  } else if (out.kind === 'recap' && out.text.endsWith('…')) {
    out.text = `${redactCutText(out.text.slice(0, -1), needles)}…`;
  } else if (out.kind === 'goal' && out.pauseMessage?.endsWith('…') === true) {
    out.pauseMessage = `${redactCutText(out.pauseMessage.slice(0, -1), needles)}…`;
  }
  if (out.kind === 'tool' && out.blocked?.endsWith('…') === true) {
    out.blocked = `${redactCutText(out.blocked.slice(0, -1), needles)}…`;
  }
  const snapshot = out.kind === 'wake' ? out.workflow?.snapshot : null;
  if (snapshot != null && snapshot.report !== undefined) {
    snapshot.report = redactReport(snapshot.report, snapshot.truncated === true, needles);
  }
  return out;
}

/**
 * A run's report, already redacted whole: the app's own cut (`truncated`) or the agent's, which ends in `…`, can split
 * a key, so an end that could start one is dropped.
 */
function redactReport(report: string, truncated: boolean, needles: readonly string[]): string {
  if (truncated) return redactCutText(report, needles);
  if (report.endsWith('…')) return `${redactCutText(report.slice(0, -1), needles)}…`;
  return report;
}

/**
 * Streaming redaction for one open text entry: an end that could be the start of a needle is held back until more
 * text arrives or the entry ends, so a secret split over two chunks is still caught; the rest goes out at once. The
 * whole text is kept, because a key that becomes known mid-entry may have started in text already sent.
 */
export class StreamRedactor {
  private pending = '';
  private raw: string;
  private readonly needles: readonly string[];

  /** `before`: the entry's text already sent before this redactor took it over. */
  constructor(needles: readonly string[], before = '') {
    this.needles = needles;
    this.raw = before;
  }

  /** Text that is safe to send now. */
  push(chunk: string): string {
    this.raw += chunk;
    if (this.needles.length === 0) return chunk;
    this.pending += chunk;
    let stop = this.pending.length - needleStartAtEnd(this.pending, this.needles);
    if (stop <= 0) return '';
    const code = this.pending.charCodeAt(stop - 1);
    if (code >= 0xd800 && code <= 0xdbff) stop -= 1;
    const { out, next } = scan(this.pending, this.needles, stop);
    this.pending = this.pending.slice(next);
    return out;
  }

  /**
   * The whole entry rescanned under a changed needle list (a key that became known mid-stream): `text` replaces
   * everything sent so far. Needles this redactor already had stay, so text it hid is never shown again.
   */
  rekey(needles: readonly string[]): { redactor: StreamRedactor; text: string } {
    const redactor = new StreamRedactor(needleList([...this.needles, ...needles]));
    return { redactor, text: redactor.push(this.raw) };
  }

  /** The held-back rest, redacted. */
  flush(): string {
    const rest = redactText(this.pending, this.needles);
    this.pending = '';
    return rest;
  }
}
