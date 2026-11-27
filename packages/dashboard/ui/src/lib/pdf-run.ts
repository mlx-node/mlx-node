/**
 * A run of the PDF tool (the saved workflow `read-pdf`) as its block, its end row and the Sessions page read it: the
 * PDF and the pages it reads, from its launch (the model's `workflow` call, or the person's Read it now); its reader
 * rows by their labels; the pages read, and the rate and the time left only once they are measured; the report's own
 * count after the end; and the line that offers Read it now when a turn sent a PDF to the tool and no run read it.
 * One function per rule, so the block, its end and the Sessions page say the same.
 */

import {
  parsePages,
  PDF_READER_PAGES,
  PDF_TOOL_MAX_PAGES,
  pdfIdPages,
  type PdfStartBy,
} from '../../../src/chat/attachment-rule.js';
import type { AgentNow, AgentRow } from './agents';
import { CHAT, fill } from './chat-strings';
import type { EngineRequest } from './chat-turns';
import { formatCount } from './format';
import type { TranscriptEntry } from './types';

type ToolEntry = Extract<TranscriptEntry, { kind: 'tool' }>;
type RunEntry = Extract<TranscriptEntry, { kind: 'workflow' }>;
type UserEntry = Extract<TranscriptEntry, { kind: 'user' }>;

export const READ_PDF = 'read-pdf';
/** The prompt Read it now and the app's own start send: the agent starts the run itself, with no model call. */
const HOST_START = `/workflow ${READ_PDF} `;
/** The script's labels: `pages 33-36` for a reader, `pages 33-36 again` for its second try. */
const READER_LABEL = 'pages ';
const AGAIN = ' again';
/** The report's first line: `PDF read: 208 of 212 pages.`, then ` Not read: 9-12.` when pages were not read. */
const HEAD = 'PDF read: ';
const HEAD_OF = ' of ';
const HEAD_PAGES = ' pages.';
const HEAD_NOT_READ = ' Not read: ';
/** The rate reads the pages of this many of the newest readers that ended well. */
const RATE_READERS = 5;
/** The rate shows once this many readers ended well. */
const RATE_FROM = 2;

/** A run of the PDF tool. */
export interface PdfRun {
  /** The PDF tool's id of the PDF: its folder's letters, `-`, its page count. */
  id: string;
  /** The PDF's name, from the note of the message that sent it; null when no message here names the id. */
  name: string | null;
  /** The pages the run reads, in order: its `pages` arg, else every page. */
  pages: number[];
  /** The model's `workflow` call started it, the person's Read it now, or the app after a turn that skipped it. */
  by: 'model' | PdfStartBy;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function json(text: string | undefined): unknown {
  if (text === undefined) return null;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return null;
  }
}

function digits(text: string): number | null {
  if (text === '' || text.length > 6) return null;
  let n = 0;
  for (let i = 0; i < text.length; i += 1) {
    const code = text.charCodeAt(i);
    if (code < 0x30 || code > 0x39) return null;
    n = n * 10 + (code - 0x30);
  }
  return n;
}

/** The script's own words for a list of pages: runs as `a-b`, single pages alone, joined by `, `. */
export function rangeText(pages: readonly number[]): string {
  const parts: string[] = [];
  let i = 0;
  while (i < pages.length) {
    const a = pages[i]!;
    let b = a;
    while (i + 1 < pages.length && pages[i + 1] === b + 1) {
      i += 1;
      b = pages[i]!;
    }
    parts.push(a === b ? String(a) : `${a}-${b}`);
    i += 1;
  }
  return parts.join(', ');
}

/** The args of a launch the PDF tool takes: the PDF's id and the `pages` text it was given. */
interface ReadArgs {
  pdf: string;
  pages: string | null;
}

function readArgs(value: unknown): ReadArgs | null {
  if (!isRecord(value)) return null;
  const pdf = value.pdf;
  if (typeof pdf !== 'string' || pdfIdPages(pdf) === null) return null;
  return { pdf, pages: typeof value.pages === 'string' ? value.pages : null };
}

/** A `/workflow read-pdf` prompt's args, and who sent it: the app when its `by` says so, else the person. */
interface StartArgs extends ReadArgs {
  by: PdfStartBy;
}

/** The args of the model's `workflow` call that starts `read-pdf` by name; null for any other call. */
export function launchArgs(launch: ToolEntry): ReadArgs | null {
  if (launch.name !== 'workflow') return null;
  const input = json(launch.input);
  if (!isRecord(input)) return null;
  const source = input.source;
  if (!isRecord(source) || source.type !== 'name' || source.name !== READ_PDF) return null;
  return readArgs(input.args);
}

/** The args of a Read it now or the app's own start, from its prompt text (`/workflow read-pdf {json}`); null for any other text. */
export function hostStartOf(text: string): StartArgs | null {
  if (!text.startsWith(HOST_START)) return null;
  let value: unknown;
  try {
    value = JSON.parse(text.slice(HOST_START.length)) as unknown;
  } catch {
    return null;
  }
  const args = readArgs(value);
  if (args === null) return null;
  return { ...args, by: isRecord(value) && value.by === 'app' ? 'app' : 'you' };
}

/** The name the note of `id` gives: a PDF a message of the conversation sent to the tool; null when none did. */
export function pdfNameOf(entries: readonly TranscriptEntry[], id: string): string | null {
  return pdfNames(entries).get(id) ?? null;
}

/** The prompt of the turn that holds entry `at`: the newest message before it, none past a turn's end or a wake. */
function promptOf(entries: readonly TranscriptEntry[], at: number): UserEntry | null {
  for (let i = at - 1; i >= 0; i -= 1) {
    const entry = entries[i]!;
    if (entry.kind === 'turn-end' || entry.kind === 'wake') return null;
    if (entry.kind === 'user' && entry.note !== true && entry.interjection !== true) return entry;
  }
  return null;
}

/**
 * A run is a run of the PDF tool when its launch names `read-pdf` and an id: the model's call that started it, else
 * the person's Read it now that opens the turn of its first update. Its pages come from the launch's `pages` (as the
 * script reads it), else every page; its name from the note with that id. Null for any other run.
 */
export function pdfRunOf(
  entries: readonly TranscriptEntry[],
  run: { entry: RunEntry | null; launch: ToolEntry | null; place: { at: number; kind: 'launch' | 'entry' | 'bare' } },
): PdfRun | null {
  if (run.entry !== null && run.entry.name !== READ_PDF) return null;
  let args: ReadArgs | null = null;
  let by: PdfRun['by'] = 'model';
  if (run.launch !== null) args = launchArgs(run.launch);
  else if (run.place.kind === 'entry') {
    const prompt = promptOf(entries, run.place.at);
    const start = prompt === null ? null : hostStartOf(prompt.text);
    args = start;
    if (start !== null) by = start.by;
  }
  if (args === null) return null;
  const total = pdfIdPages(args.pdf)!;
  const pages = args.pages === null ? parsePages(`1-${total}`, total) : parsePages(args.pages, total);
  return { id: args.pdf, name: pdfNameOf(entries, args.pdf), pages, by };
}

/** The block's name: `Reading {file}`, or `Reading a PDF` when no note names its id. */
export function pdfRunName(run: Pick<PdfRun, 'name'>): string {
  return run.name === null ? CHAT.pdf.runNameBare : fill(CHAT.pdf.runName, { name: run.name });
}

/** A reader's label as the script writes it: its pages, and whether it is a group's second try. */
export interface ReaderLabel {
  pages: number[];
  again: boolean;
}

/** `pages 33-36` or `pages 33-36 again`, its pages within `total`; null for the run's other agents (cuts, combine). */
export function readerOf(label: string, total: number): ReaderLabel | null {
  if (!label.startsWith(READER_LABEL)) return null;
  let rest = label.slice(READER_LABEL.length);
  const again = rest.endsWith(AGAIN);
  if (again) rest = rest.slice(0, rest.length - AGAIN.length);
  const pages = parsePages(rest, total);
  return pages.length > 0 && rangeText(pages) === rest ? { pages, again } : null;
}

/** A label of one of the run's agents as a reader of its PDF; null for its other agents. */
export function runReader(run: PdfRun, label: string): ReaderLabel | null {
  return readerOf(label, pdfIdPages(run.id) ?? 0);
}

/** A reader row's name: `Pages 33-36`, `Page 5`. */
export function readerName(reader: ReaderLabel): string {
  const range = rangeText(reader.pages);
  return reader.pages.length === 1 ? fill(CHAT.pdf.readerOne, { range }) : fill(CHAT.pdf.reader, { range });
}

/** The run's report head, read line by line: pages read of the pages it read, and the pages it says were not read. */
export interface ReportHead {
  read: number;
  total: number;
  unread: number[];
}

/** The first line of a report the script writes; null for any other text. */
export function readPdfReportHead(report: string): ReportHead | null {
  const end = report.indexOf('\n');
  const line = end === -1 ? report : report.slice(0, end);
  if (!line.startsWith(HEAD)) return null;
  const rest = line.slice(HEAD.length);
  const of = rest.indexOf(HEAD_OF);
  if (of === -1) return null;
  const read = digits(rest.slice(0, of));
  const after = rest.slice(of + HEAD_OF.length);
  const at = after.indexOf(HEAD_PAGES);
  if (read === null || at === -1) return null;
  const total = digits(after.slice(0, at));
  if (total === null || read > total) return null;
  const tail = after.slice(at + HEAD_PAGES.length);
  if (tail === '') return { read, total, unread: [] };
  if (!tail.startsWith(HEAD_NOT_READ) || !tail.endsWith('.')) return null;
  const list = tail.slice(HEAD_NOT_READ.length, tail.length - 1);
  const unread = parsePages(list, PDF_TOOL_MAX_PAGES);
  return unread.length > 0 && rangeText(unread) === list ? { read, total, unread } : null;
}

/** A row's start and end on the host's clock: its newest attempt's spawn and finish lines. */
function spanOf(row: AgentRow): { start: number; end: number | null } | null {
  const entry = row.entry;
  if (entry === undefined || entry.noSpawn === true) return null;
  const end = entry.endTs ?? (row.durationMs > 0 ? entry.ts + row.durationMs : null);
  return { start: entry.ts, end };
}

export interface PdfProgressInput {
  run: PdfRun;
  agents: readonly AgentNow[];
  /** The run's newest update; null for a run only spawn lines name. */
  entry: RunEntry | null;
  /** Its state now is running (not asking, paused or stopping), as the block's state says. */
  running: boolean;
  /** The block's clock. */
  now: number;
}

/** The numbers of the PDF line; each one only once it is measured. */
export interface PdfProgress {
  /** Pages the run reads. */
  total: number;
  /** Pages of the groups a reader ended well for; after the end, the report's own count. */
  read: number;
  /** Pages per second; null until {@link RATE_FROM} readers ended well. */
  rate: number | null;
  /** Time left in ms: while the run reads its pages (step 1) and runs, with a rate; else null. */
  leftMs: number | null;
  /** Pages no reader started yet, while the run is active; else null. */
  notStarted: number | null;
  /** The pages the report says were not read, after the end; null before or with no report. */
  unread: number[] | null;
}

/**
 * The run's PDF numbers. Pages read count a group's pages once, when one of its readers ended well (a first reader
 * whose group has a second try did not); after the end the report's first line is the word. The rate: live, the pages
 * of the {@link RATE_READERS} newest readers that ended well over the time from the first of them starting to the last
 * ending; ended, the pages read over the time of the read (the first reader's start to the last reader's end). Time
 * left: the pages left at that rate, less the time since a reader last ended, never under 0, only in step 1 while the
 * run runs.
 */
export function pdfProgress(input: PdfProgressInput): PdfProgress {
  const { run, agents, entry } = input;
  const total = run.pages.length;
  const groups: number[][] = [];
  for (let i = 0; i < run.pages.length; i += PDF_READER_PAGES) groups.push(run.pages.slice(i, i + PDF_READER_PAGES));
  const started = new Set<string>();
  const readGroups = new Set<string>();
  const ended: Array<{ start: number; end: number; pages: number }> = [];
  let first: number | null = null;
  let last: number | null = null;
  let read = 0;
  const retried = new Set<string>();
  for (const agent of agents) {
    const reader = runReader(run, agent.row.label);
    if (reader?.again === true) retried.add(rangeText(reader.pages));
  }
  for (const agent of agents) {
    const reader = runReader(run, agent.row.label);
    if (reader === null) continue;
    const key = rangeText(reader.pages);
    started.add(key);
    const span = spanOf(agent.row);
    if (span !== null) {
      first = first === null ? span.start : Math.min(first, span.start);
      if (span.end !== null && agent.state !== 'running') last = last === null ? span.end : Math.max(last, span.end);
    }
    // The script tries a group again only when its first reader gave no notes, whatever that reader's end.
    if (agent.state !== 'done' || (!reader.again && retried.has(key))) continue;
    if (!readGroups.has(key)) {
      readGroups.add(key);
      read += reader.pages.length;
    }
    if (span !== null && span.end !== null)
      ended.push({ start: span.start, end: span.end, pages: reader.pages.length });
  }
  const active = entry === null || entry.status === 'active';
  const head = !active && entry?.report !== undefined ? readPdfReportHead(entry.report) : null;
  if (head !== null) read = head.read;
  let rate: number | null = null;
  if (active && ended.length >= RATE_FROM) {
    const newest = [...ended].sort((a, b) => a.end - b.end).slice(-RATE_READERS);
    let pages = 0;
    let from = newest[0]!.start;
    for (const reader of newest) {
      pages += reader.pages;
      from = Math.min(from, reader.start);
    }
    const span = newest.at(-1)!.end - from;
    rate = span > 0 ? (pages * 1_000) / span : null;
  } else if (!active && read > 0 && first !== null && last !== null && last > first) {
    rate = (read * 1_000) / (last - first);
  }
  const firstStep = entry?.phases[0]?.title;
  const reading = active && input.running && firstStep !== undefined && entry?.currentPhase === firstStep;
  let leftMs: number | null = null;
  if (reading && rate !== null && rate > 0) {
    const since = last === null ? 0 : Math.max(0, input.now - last);
    leftMs = Math.max(0, ((total - read) * 1_000) / rate - since);
  }
  let notStarted: number | null = null;
  if (active) {
    notStarted = 0;
    for (const group of groups) if (!started.has(rangeText(group))) notStarted += group.length;
  }
  return { total, read, rate, leftMs, notStarted, unread: head === null ? null : head.unread };
}

function oneDecimal(n: number): string {
  return n >= 10 ? String(Math.round(n)) : n.toFixed(1);
}

/** Pages per second as words: per second from 1 a second, else per minute. */
export function rateWords(perSecond: number): string {
  return perSecond >= 1
    ? fill(CHAT.pdf.rateSecond, { n: oneDecimal(perSecond) })
    : fill(CHAT.pdf.rateMinute, { n: oneDecimal(perSecond * 60) });
}

const MINUTE_MS = 60_000;
const HOUR_MS = 3_600_000;

/** Time left as words: under a minute, else about so many minutes, from an hour about so many hours. */
export function leftWords(ms: number): string {
  const w = CHAT.pdf.left;
  if (ms < MINUTE_MS) return w.minute;
  const minutes = Math.round(ms / MINUTE_MS);
  if (minutes < 60) return fill(w.minutes, { n: minutes });
  return fill(w.hours, { n: Math.round(ms / HOUR_MS) });
}

/** The PDF line: the pages and the counts, then the rate and the time left when known; the pages not read apart. */
export interface PdfLine {
  text: string;
  /** In warning ink after the line. */
  unread: string | null;
}

/** The one PDF line of the block, its end and the Sessions page; null for a run that reads no page. */
export function pdfLine(run: PdfRun, progress: PdfProgress): PdfLine | null {
  if (progress.total === 0) return null;
  const p = CHAT.pdf;
  const range = rangeText(run.pages);
  const done = formatCount(progress.read);
  const parts = [
    progress.total === 1
      ? fill(p.lineOne, { range, done })
      : fill(p.line, { range, done, total: formatCount(progress.total) }),
  ];
  if (progress.rate !== null) parts.push(rateWords(progress.rate));
  if (progress.leftMs !== null) parts.push(leftWords(progress.leftMs));
  const unread = progress.unread ?? [];
  const list = rangeText(unread);
  return {
    text: parts.join(' · '),
    unread: unread.length === 0 ? null : fill(unread.length === 1 ? p.notReadOne : p.notRead, { list }),
  };
}

/** `168 pages not started`; null while none is left. */
export function notStartedWords(progress: PdfProgress): string | null {
  const n = progress.notStarted;
  if (n === null || n === 0) return null;
  return n === 1 ? CHAT.pdf.notStartedOne : fill(CHAT.pdf.notStarted, { n: formatCount(n) });
}

/** The engine's newest request that named a reader's session, on any model. */
export function readerRequest(requests: readonly EngineRequest[], session: string): EngineRequest | null {
  let found: EngineRequest | null = null;
  for (const request of requests)
    if (request.session === session && (found === null || request.startedAt >= found.startedAt)) found = request;
  return found;
}

/** What the engine does for a reader now: it waits, reads its pages (with counts once the engine has some), writes. */
export function readerFact(request: EngineRequest | null): string | null {
  if (request === null) return null;
  const f = CHAT.pdf.readerFacts;
  switch (request.phase) {
    case 'queued':
    case 'slot':
      return f.waiting;
    case 'prepare':
      return f.reading;
    case 'prefill': {
      const done = request.prefillDone;
      const total = request.prefillTotal;
      if (done === null || total === null || total <= 0 || done <= 0) return f.reading;
      return fill(f.readingOf, { done: formatCount(Math.min(done, total)), total: formatCount(total) });
    }
    case 'decode':
      return f.writing;
    default:
      return null;
  }
}

const PAGE_FILE = 'page-';
const PAGE_EXT = '.jpg';

/** The page a reader's open `read_file` names (`Read \`…/page-043.jpg\``); null for any other title. */
export function pageOpened(title: string): number | null {
  const at = title.lastIndexOf(PAGE_FILE);
  if (at === -1) return null;
  const start = at + PAGE_FILE.length;
  if (title.slice(start + 3, start + 3 + PAGE_EXT.length) !== PAGE_EXT) return null;
  const n = digits(title.slice(start, start + 3));
  return n === null || n < 1 ? null : n;
}

export function openedWords(page: number): string {
  return fill(CHAT.pdf.readerOpened, { n: page });
}

/** A PDF a message sent to the tool whose pages no run read. */
export interface UnreadPdf {
  id: string;
  name: string;
}

/**
 * The PDFs of the message at `at` that went to the PDF tool and that no run of the tool read: shown once its turn
 * ended, so never while an answer of the conversation runs (a Read it now that starts runs as one), and never for a
 * note another conversation sent (a fork's copy of the original's message: its page folder is the original's).
 * `started`: the ids of the conversation's PDF runs; `session`: the conversation's id.
 */
export function pdfUnread(
  entries: readonly TranscriptEntry[],
  at: number,
  running: boolean,
  started: ReadonlySet<string>,
  session: string | null,
): UnreadPdf[] {
  const entry = entries[at];
  if (running || entry?.kind !== 'user' || entry.attachments === undefined) return [];
  const out: UnreadPdf[] = [];
  for (const item of entry.attachments) {
    if (item.kind !== 'pdf' || item.way !== 'tool' || item.id === undefined || started.has(item.id)) continue;
    if (item.session !== undefined && item.session !== session) continue;
    out.push({ id: item.id, name: item.name });
  }
  return out;
}

/** The name of each PDF a message of the conversation sent to the tool, by its id. */
export function pdfNames(entries: readonly TranscriptEntry[]): ReadonlyMap<string, string> {
  const names = new Map<string, string>();
  for (const entry of entries) {
    if (entry.kind !== 'user' || entry.attachments === undefined) continue;
    for (const item of entry.attachments)
      if (item.kind === 'pdf' && item.way === 'tool' && item.id !== undefined && !names.has(item.id))
        names.set(item.id, item.name);
  }
  return names;
}

/**
 * The words a `/workflow read-pdf` prompt shows in place of its command: `You started reading {file}` for Read it now,
 * `Started reading {file}` for the app's own start; null for any other text.
 */
export function hostStartWords(text: string, names: ReadonlyMap<string, string>): string | null {
  const args = hostStartOf(text);
  if (args === null) return null;
  const name = names.get(args.pdf);
  const p = CHAT.pdf;
  if (args.by === 'app') return name === undefined ? p.appStartBare : fill(p.appStart, { name });
  return name === undefined ? p.hostStartBare : fill(p.hostStart, { name });
}
