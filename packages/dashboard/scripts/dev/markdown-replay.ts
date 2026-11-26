/**
 * Replays recorded answers into the chat Markdown parser the way a streaming answer arrives: 4-24 characters per
 * paint, seeded. Prints the parse time per paint by answer size, how many top-level nodes kept their object (the
 * renderer skips those), how often text already shown changed above the last line (a flash), and how the finished
 * text differs from a baseline run of the old parser.
 *
 *   oxnode packages/dashboard/scripts/dev/markdown-replay.ts [--fresh-corpus]
 *   oxnode packages/dashboard/scripts/dev/markdown-replay.ts --hostile
 *
 * `--hostile` replays two inputs a looping model can write, with and without ChatMarkdown's hold (`paceHoldMs`),
 * and prints the main-thread time per update and the share of the time left free.
 *
 * Reads the Agent and Chat session homes, never writes them. The corpus is kept in the research cache so a baseline
 * and a later run see the same answers.
 */

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join } from 'node:path';

import { getMarkdown, parseMarkdownToStructure, type MarkdownIt, type ParsedNode } from 'stream-markdown-parser';

import { AgentSessionsReader } from '../../src/agent-sessions/reader.js';
import { chatPaths } from '../../src/chat/paths.js';
import { visibleText } from '../../ui/src/lib/link-rule.js';
import { CHAT_PARSE_OPTIONS, chatMarkdownIt, paceHoldMs } from '../../ui/src/lib/markstream-config.js';

export const RESEARCH_DIR = join(homedir(), '.cache', 'mlx-node-research', 'ui-polish-2');
export const CORPUS_FILE = join(RESEARCH_DIR, 'markstream-corpus.json');
export const BASELINE_FILE = join(RESEARCH_DIR, 'markstream-baseline.json');
export const RESULT_FILE = join(RESEARCH_DIR, 'markstream-replay.json');
export const HOSTILE_FILE = join(RESEARCH_DIR, 'markstream-hostile.json');

export interface Doc {
  id: string;
  text: string;
}

/** One replayed doc: what a baseline keeps to compare a later run against. */
export interface DocResult {
  id: string;
  hash: string;
  chars: number;
  paints: number;
  flashes: number;
  /** The shown text of the finished answer. */
  final: string;
}

export interface Timing {
  /** Size of the text at that paint. */
  chars: number;
  /** Wall time. */
  ms: number;
  /** Main-thread CPU time: what the parse costs when other processes load the machine. */
  cpuMs?: number;
}

/** CPU time of this thread so far, in ms. */
export function threadCpuMs(): number {
  const usage = process.threadCpuUsage();
  return (usage.user + usage.system) / 1_000;
}

export const BUCKETS: ReadonlyArray<readonly [string, number, number]> = [
  ['<10K', 0, 10_000],
  ['10-50K', 10_000, 50_000],
  ['50-100K', 50_000, 100_000],
  ['100-200K', 100_000, 200_000],
  ['>=200K', 200_000, Number.POSITIVE_INFINITY],
];

/** mulberry32: a small seeded generator, so every run cuts the same chunks. */
export function seeded(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4_294_967_296;
  };
}

/** FNV-1a of the text, as hex. */
export function hashText(text: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16);
}

/** The end of each paint: 4 to 24 characters after the one before, the last at the end of the text. */
export function chunkEnds(text: string, seed: number): number[] {
  const random = seeded(seed);
  const ends: number[] = [];
  let at = 0;
  while (at < text.length) {
    at = Math.min(text.length, at + 4 + Math.floor(random() * 21));
    ends.push(at);
  }
  return ends;
}

/** Text shown before this paint changed somewhere above its last line. */
export function isFlash(before: string, after: string): boolean {
  const cut = before.lastIndexOf('\n');
  if (cut < 0) return false;
  return !after.startsWith(before.slice(0, cut));
}

export function percentile(sorted: readonly number[], q: number): number {
  if (sorted.length === 0) return 0;
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(q * sorted.length) - 1))]!;
}

export interface BucketStats {
  bucket: string;
  paints: number;
  p50: number;
  p95: number;
  max: number;
}

export function bucketStats(timings: readonly Timing[], cpu = false): BucketStats[] {
  const out: BucketStats[] = [];
  for (const [bucket, from, to] of BUCKETS) {
    const ms = timings
      .filter((t) => t.chars >= from && t.chars < to)
      .map((t) => (cpu ? (t.cpuMs ?? t.ms) : t.ms))
      .sort((a, b) => a - b);
    if (ms.length === 0) continue;
    out.push({ bucket, paints: ms.length, p50: percentile(ms, 0.5), p95: percentile(ms, 0.95), max: ms.at(-1)! });
  }
  return out;
}

export function printBuckets(title: string, stats: readonly BucketStats[]): void {
  console.log(title);
  for (const s of stats) {
    console.log(
      `  ${s.bucket.padEnd(9)} paints ${String(s.paints).padStart(6)}  p50 ${s.p50.toFixed(3)} ms  p95 ${s.p95.toFixed(3)} ms  max ${s.max.toFixed(3)} ms`,
    );
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Corpus

function assistantTexts(agentHome: string): string[][] {
  const reader = new AgentSessionsReader({ agentHome });
  const index = reader.scan();
  const sessions: string[][] = [];
  for (const id of index.sessions.keys()) {
    const detail = reader.detail(id);
    if (detail === null) continue;
    const texts: string[] = [];
    for (const entry of detail.transcript)
      if (entry.kind === 'assistant' && entry.text.trim() !== '') texts.push(entry.text);
    if (texts.length > 0) sessions.push(texts);
  }
  return sessions;
}

const WORDS = [
  'cache',
  'block',
  'prefill',
  'decode',
  'lane',
  'token',
  'model',
  'stream',
  'parser',
  'render',
  'layout',
  'kernel',
  'window',
  'metal',
  'batch',
  'queue',
];

/** A long answer that uses every construct a model writes, built from a seed. */
export function syntheticDoc(chars: number, seed: number): string {
  const random = seeded(seed);
  const pick = (list: readonly string[]): string => list[Math.floor(random() * list.length)]!;
  const sentence = (): string => {
    const n = 6 + Math.floor(random() * 14);
    const words: string[] = [];
    for (let i = 0; i < n; i += 1) {
      const word = pick(WORDS);
      const roll = random();
      if (roll < 0.05) words.push(`**${word}**`);
      else if (roll < 0.08) words.push(`*${word}*`);
      else if (roll < 0.12) words.push(`\`${word}_${i}()\``);
      else if (roll < 0.14) words.push(`[${word}](https://docs.example/${word}/${i})`);
      else if (roll < 0.15) words.push(`https://ref.example/${word}?q=${i}`);
      else if (roll < 0.17) words.push(`[${1 + Math.floor(random() * 9)}]`);
      else if (roll < 0.175) words.push(`<b>${word}</b>`);
      else if (roll < 0.18) words.push('&amp;');
      else words.push(word);
    }
    return `${words.join(' ')}.`;
  };
  const parts: string[] = [];
  let size = 0;
  let section = 0;
  while (size < chars) {
    const roll = random();
    let part: string;
    if (roll < 0.08) part = `## ${section++}. ${sentence()}`;
    else if (roll < 0.4) part = `${sentence()} ${sentence()}\n${sentence()}`;
    else if (roll < 0.55)
      part = Array.from({ length: 2 + Math.floor(random() * 5) }, (_, i) =>
        random() < 0.2 ? `  - ${sentence()}` : `${random() < 0.5 ? `${i + 1}.` : '-'} ${sentence()}`,
      ).join('\n');
    else if (roll < 0.65) {
      const rows = Array.from(
        { length: 2 + Math.floor(random() * 8) },
        (_, i) => `| ${pick(WORDS)} | ${i} | ${sentence()} |`,
      );
      part = ['| name | n | note |', '|:--|--:|---|', ...rows].join('\n');
    } else if (roll < 0.75) {
      const lines = Array.from(
        { length: 3 + Math.floor(random() * 12) },
        (_, i) => `const ${pick(WORDS)}${i} = ${i} * 2; // ${pick(WORDS)}`,
      );
      part = ['```ts', ...lines, '```'].join('\n');
    } else if (roll < 0.78) part = ['```diff', '-old line', '+new line', ' same', '```'].join('\n');
    else if (roll < 0.85) part = `> ${sentence()}\n> ${sentence()}`;
    else if (roll < 0.87) part = '---';
    else part = sentence();
    parts.push(part);
    size += part.length + 2;
  }
  return parts.join('\n\n');
}

/** Real answers, longest first, cut into docs: single answers for the small buckets, joined answers past 50K. */
export function buildCorpus(): Doc[] {
  const homes = [join(homedir(), '.mlx-agent'), chatPaths().chatAgent];
  const seen = new Set<string>();
  const answers: string[] = [];
  for (const home of homes) {
    for (const texts of assistantTexts(home)) {
      for (const text of texts) {
        if (seen.has(text)) continue;
        seen.add(text);
        answers.push(text);
      }
    }
  }
  answers.sort((a, b) => b.length - a.length);
  const docs: Doc[] = [];
  const singles = answers.filter((a) => a.length >= 2_000).slice(0, 40);
  for (const [i, text] of singles.entries()) docs.push({ id: `real-${i}`, text });
  const joined = answers.filter((a) => a.length >= 500);
  let next = 0;
  const joinUpTo = (target: number, id: string): void => {
    const parts: string[] = [];
    let size = 0;
    while (size < target && joined.length > 0) {
      const text = joined[next % joined.length]!;
      next += 1;
      parts.push(text);
      size += text.length + 2;
    }
    if (size >= target) docs.push({ id, text: parts.join('\n\n') });
  };
  for (let i = 0; i < 6; i += 1) joinUpTo(50_000, `joined-50K-${i}`);
  for (let i = 0; i < 2; i += 1) joinUpTo(100_000, `joined-100K-${i}`);
  joinUpTo(200_000, 'joined-200K');
  docs.push({ id: 'synthetic-50K', text: syntheticDoc(50_000, 50) });
  docs.push({ id: 'synthetic-200K', text: syntheticDoc(200_000, 200) });
  return docs;
}

export function loadCorpus(fresh: boolean): Doc[] {
  if (!fresh) {
    try {
      return JSON.parse(readFileSync(CORPUS_FILE, 'utf8')) as Doc[];
    } catch {
      // No corpus yet: build one.
    }
  }
  const docs = buildCorpus();
  writeJson(CORPUS_FILE, docs);
  return docs;
}

export function writeJson(path: string, value: unknown): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(value)}\n`);
}

// ---------------------------------------------------------------------------------------------------------------
// The words of a finished diff, to say what changed

/** Words of `text`, split on spaces, tabs and line ends. */
export function words(text: string): string[] {
  const out: string[] = [];
  let start = -1;
  for (let i = 0; i <= text.length; i += 1) {
    const ch = text[i];
    const space = ch === undefined || ch === ' ' || ch === '\n' || ch === '\t' || ch === '\r';
    if (space) {
      if (start >= 0) out.push(text.slice(start, i));
      start = -1;
    } else if (start < 0) start = i;
  }
  return out;
}

/** Changed runs between two word lists (Myers), as "old → new"; null when more than `limit` edits. */
export function wordChanges(a: readonly string[], b: readonly string[], limit = 4_000): string[] | null {
  let lo = 0;
  while (lo < a.length && lo < b.length && a[lo] === b[lo]) lo += 1;
  let ha = a.length;
  let hb = b.length;
  while (ha > lo && hb > lo && a[ha - 1] === b[hb - 1]) {
    ha -= 1;
    hb -= 1;
  }
  const x = a.slice(lo, ha);
  const y = b.slice(lo, hb);
  const n = x.length;
  const m = y.length;
  if (n === 0 && m === 0) return [];
  const max = n + m;
  const offset = max + 1;
  const v = new Int32Array(2 * max + 3);
  const trace: Int32Array[] = [];
  let found = -1;
  for (let d = 0; d <= Math.min(max, limit); d += 1) {
    trace.push(v.slice());
    for (let k = -d; k <= d; k += 2) {
      let px: number;
      if (k === -d || (k !== d && v[offset + k - 1]! < v[offset + k + 1]!)) px = v[offset + k + 1]!;
      else px = v[offset + k - 1]! + 1;
      let py = px - k;
      while (px < n && py < m && x[px] === y[py]) {
        px += 1;
        py += 1;
      }
      v[offset + k] = px;
      if (px >= n && py >= m) {
        found = d;
        break;
      }
    }
    if (found >= 0) break;
  }
  if (found < 0) return null;
  // Walk back: mark kept pairs, then group the rest into runs.
  const keepA = new Uint8Array(n);
  const keepB = new Uint8Array(m);
  let px = n;
  let py = m;
  for (let d = found; d >= 0; d -= 1) {
    const vd = trace[d]!;
    const k = px - py;
    const prevK = k === -d || (k !== d && vd[offset + k - 1]! < vd[offset + k + 1]!) ? k + 1 : k - 1;
    const prevX = vd[offset + prevK]!;
    const prevY = prevX - prevK;
    while (px > prevX && py > prevY && px > 0 && py > 0) {
      px -= 1;
      py -= 1;
      keepA[px] = 1;
      keepB[py] = 1;
    }
    px = prevX;
    py = prevY;
  }
  const changes: string[] = [];
  let i = 0;
  let j = 0;
  while (i < n || j < m) {
    if (i < n && j < m && keepA[i] === 1 && keepB[j] === 1) {
      i += 1;
      j += 1;
      continue;
    }
    const from: string[] = [];
    const to: string[] = [];
    while (i < n && keepA[i] !== 1) from.push(x[i++]!);
    while (j < m && keepB[j] !== 1) to.push(y[j++]!);
    changes.push(`${from.join(' ')} → ${to.join(' ')}`);
  }
  return changes;
}

// ---------------------------------------------------------------------------------------------------------------
// What markstream shows

/** A field that should be a string, or nothing. */
function str(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function inlineShown(nodes: readonly ParsedNode[] | undefined): string {
  if (nodes === undefined) return '';
  let out = '';
  for (const node of nodes) {
    const any = node as ParsedNode & {
      content?: unknown;
      code?: unknown;
      alt?: unknown;
      id?: unknown;
      children?: unknown;
    };
    switch (node.type) {
      case 'text':
      case 'html_inline':
      case 'math_inline':
        out += str(any.content);
        break;
      case 'inline_code':
        out += str(any.code);
        break;
      case 'image':
        out += str(any.alt);
        break;
      case 'reference':
        out += `[${str(any.id)}]`;
        break;
      case 'hardbreak':
        out += '\n';
        break;
      default:
        out += Array.isArray(any.children) ? inlineShown(any.children as ParsedNode[]) : String(node.raw ?? '');
    }
  }
  return out;
}

function codeShown(node: ParsedNode & { code?: unknown; diff?: unknown; loading?: boolean }): string {
  const code = String(node.diff === true ? node.raw : (node.code ?? ''));
  return code.endsWith('\n') ? code.slice(0, -1) : code;
}

export function blockShown(node: ParsedNode): string {
  const any = node as ParsedNode & Record<string, unknown>;
  switch (node.type) {
    case 'paragraph':
    case 'heading':
      return inlineShown(any.children as ParsedNode[]);
    case 'code_block':
      return codeShown(node);
    case 'blockquote':
      return (any.children as ParsedNode[]).map(blockShown).join('\n');
    case 'list':
      return (any.items as Array<{ children: ParsedNode[] }>)
        .map((item) => item.children.map(blockShown).join('\n'))
        .join('\n');
    case 'thematic_break':
      return '';
    case 'table': {
      const rows = [any.header, ...(any.rows as unknown[])] as Array<{ cells: Array<{ children: ParsedNode[] }> }>;
      return rows.map((row) => row.cells.map((cell) => inlineShown(cell.children)).join(' | ')).join('\n');
    }
    case 'html_block':
    case 'math_block':
      return str(any.content);
    default:
      return Array.isArray(any.children) ? inlineShown(any.children as ParsedNode[]) : String(node.raw ?? '');
  }
}

export function shownOf(nodes: readonly ParsedNode[]): string {
  return nodes.map(blockShown).join('\n');
}

/** The renderer's keep rule (markstream-react NodeRenderer): same type, raw, loading, autoClosed, lines and diff. */
function sameNode(a: ParsedNode, b: ParsedNode): boolean {
  const x = a as ParsedNode & { autoClosed?: unknown; diff?: unknown };
  const y = b as ParsedNode & { autoClosed?: unknown; diff?: unknown };
  return (
    x.type === y.type &&
    x.raw === y.raw &&
    x.loading === y.loading &&
    x.autoClosed === y.autoClosed &&
    x.sourceMap?.startLine === y.sourceMap?.startLine &&
    x.sourceMap?.endLine === y.sourceMap?.endLine &&
    ((x.type !== 'code_block' && y.type !== 'code_block') || x.diff === y.diff)
  );
}

export function chatMd(): MarkdownIt {
  return chatMarkdownIt(getMarkdown('renderer-mlx-chat'));
}

export interface ReplayRun {
  docs: DocResult[];
  timings: Timing[];
  kept: number;
  nodes: number;
  /** Paints whose top-level nodes all kept their objects. */
  quietPaints: number;
  mdCreateMs: number[];
  finalMs: Array<{ chars: number; ms: number }>;
  /** Per doc: the end of each paint and its main-thread cpu ms, in order. */
  paints: Array<{ ends: number[]; cpuMs: number[]; finalMs: number }>;
}

function replayNew(docs: readonly Doc[]): ReplayRun {
  const run: ReplayRun = {
    docs: [],
    timings: [],
    kept: 0,
    nodes: 0,
    quietPaints: 0,
    mdCreateMs: [],
    finalMs: [],
    paints: [],
  };
  for (const [index, doc] of docs.entries()) {
    const full = visibleText(doc.text);
    let started = performance.now();
    const md = chatMd();
    run.mdCreateMs.push(performance.now() - started);
    const options = { ...CHAT_PARSE_OPTIONS, final: false };
    let previous: ParsedNode[] = [];
    let shown = '';
    let flashes = 0;
    const ends = chunkEnds(full, 1_000 + index);
    const cpuMs: number[] = [];
    for (const end of ends) {
      const text = full.slice(0, end);
      const cpu = threadCpuMs();
      started = performance.now();
      const parsed = parseMarkdownToStructure(text, md, options);
      let kept = 0;
      const nodes = parsed.map((node, i) => {
        const before = previous[i];
        if (before !== undefined && sameNode(before, node)) {
          kept += 1;
          return before;
        }
        return node;
      });
      const timing = { chars: text.length, ms: performance.now() - started, cpuMs: threadCpuMs() - cpu };
      run.timings.push(timing);
      cpuMs.push(timing.cpuMs);
      run.kept += kept;
      run.nodes += nodes.length;
      if (kept === nodes.length && nodes.length === previous.length) run.quietPaints += 1;
      previous = nodes;
      const next = shownOf(nodes);
      if (isFlash(shown, next)) flashes += 1;
      shown = next;
    }
    started = performance.now();
    const finished = parseMarkdownToStructure(full, md, { ...CHAT_PARSE_OPTIONS, final: true });
    run.finalMs.push({ chars: full.length, ms: performance.now() - started });
    run.paints.push({ ends, cpuMs, finalMs: run.finalMs.at(-1)!.ms });
    run.docs.push({
      id: doc.id,
      hash: hashText(doc.text),
      chars: full.length,
      paints: ends.length,
      flashes,
      final: shownOf(finished),
    });
  }
  return run;
}

// ---------------------------------------------------------------------------------------------------------------
// The main thread over a streaming answer, with and without ChatMarkdown's hold

/** Text a looping model can write: a bracket run (markdown-it's link scan is quadratic on it) and a deep `*` nest. */
export const HOSTILE_DOCS: readonly Doc[] = [
  { id: 'hostile-brackets', text: `${'['.repeat(40_000)}${'[a](x)'.repeat(40_000)}` },
  { id: 'hostile-stars', text: `${'*'.repeat(4_000)}x${'*'.repeat(4_000)}` },
];

/** A chunk every frame: faster than a local model writes, so the thread never waits for text. */
export const ARRIVAL_MS = 16;

export interface PaceRun {
  chunks: number;
  /** Parses the thread ran; chunks that arrived while it was busy or holding are taken together. */
  updates: number;
  /** Main-thread ms of each update. */
  costs: number[];
  busyMs: number;
  /** From the first chunk to the end of the final update. */
  wallMs: number;
  /** Longest time from a chunk's arrival to the update that shows it. */
  maxLagMs: number;
}

/**
 * The thread on a clock that moves by each update's cost. Chunk `i` arrives at `i * ARRIVAL_MS` and the answer ends
 * one step after the last. An update starts when the thread is free, a chunk is waiting and (paced) the hold is over,
 * and takes every chunk arrived by then; the finished text runs as soon as the thread is free. `cost(end, final)`
 * does the work for the text up to `end` and returns its ms.
 */
export function paceTimeline(
  ends: readonly number[],
  cost: (end: number, final: boolean) => number,
  paced: boolean,
): PaceRun {
  const finishAt = ends.length * ARRIVAL_MS;
  const costs: number[] = [];
  let shown = 0;
  let free = 0;
  let holdUntil = 0;
  let maxLagMs = 0;
  while (shown < ends.length) {
    const start = Math.max(free, paced ? holdUntil : 0, shown * ARRIVAL_MS);
    if (start >= finishAt) break;
    maxLagMs = Math.max(maxLagMs, start - shown * ARRIVAL_MS);
    shown = Math.min(ends.length, Math.floor(start / ARRIVAL_MS) + 1);
    const ms = cost(ends[shown - 1]!, false);
    costs.push(ms);
    free = start + ms;
    holdUntil = free + paceHoldMs(ms);
  }
  const start = Math.max(free, finishAt);
  if (shown < ends.length) maxLagMs = Math.max(maxLagMs, start - shown * ARRIVAL_MS);
  const ms = cost(ends.at(-1) ?? 0, true);
  costs.push(ms);
  const busyMs = costs.reduce((a, b) => a + b, 0);
  return { chunks: ends.length, updates: costs.length, costs, busyMs, wallMs: start + ms, maxLagMs };
}

/**
 * The real parse of each text the timeline hands in, on one parser instance as the renderer keeps it. A parse that
 * overflows the stack costs what it ran; the renderer then shows plain text (`PlainOnError`), which costs nothing here.
 */
function parseCost(full: string): { cost: (end: number, final: boolean) => number; threw: () => boolean } {
  const md = chatMd();
  let threw = false;
  const cost = (end: number, final: boolean): number => {
    if (threw) return 0;
    const cpu = threadCpuMs();
    try {
      parseMarkdownToStructure(full.slice(0, end), md, { ...CHAT_PARSE_OPTIONS, final });
    } catch (error) {
      if (!(error instanceof RangeError)) throw error;
      threw = true;
    }
    return threadCpuMs() - cpu;
  };
  return { cost, threw: () => threw };
}

function printPace(label: string, run: PaceRun, threw: boolean): void {
  const sorted = [...run.costs].sort((a, b) => a - b);
  const free = 100 * (1 - run.busyMs / run.wallMs);
  console.log(
    `  ${label.padEnd(7)} updates ${String(run.updates).padStart(6)}/${run.chunks}  per update p50 ${percentile(sorted, 0.5).toFixed(1)} p95 ${percentile(sorted, 0.95).toFixed(1)} max ${sorted.at(-1)!.toFixed(1)} ms  busy per chunk ${(run.busyMs / run.chunks).toFixed(2)} ms  busy ${(run.busyMs / 1_000).toFixed(1)} s of ${(run.wallMs / 1_000).toFixed(1)} s  free ${free.toFixed(1)} %  max lag ${(run.maxLagMs / 1_000).toFixed(2)} s${threw ? '  (stack overflow: plain text after)' : ''}`,
  );
}

function hostileMain(): void {
  const results: Record<string, unknown> = {};
  for (const [index, doc] of HOSTILE_DOCS.entries()) {
    const full = visibleText(doc.text);
    const ends = chunkEnds(full, 9_000 + index);
    console.log(`${doc.id}: ${full.length} chars, ${ends.length} chunks, one every ${ARRIVAL_MS} ms`);
    const row: Record<string, unknown> = {};
    for (const paced of [false, true]) {
      const parse = parseCost(full);
      const run = paceTimeline(ends, parse.cost, paced);
      printPace(paced ? 'paced' : 'before', run, parse.threw());
      const { costs: _costs, ...rest } = run;
      row[paced ? 'paced' : 'before'] = { ...rest, threw: parse.threw() };
    }
    results[doc.id] = row;
  }
  writeJson(HOSTILE_FILE, results);
}

interface Baseline {
  docs: DocResult[];
  buckets: BucketStats[];
}

function main(): void {
  const fresh = process.argv.includes('--fresh-corpus');
  const docs = loadCorpus(fresh);
  console.log(`corpus: ${docs.length} docs, ${docs.reduce((n, d) => n + d.text.length, 0)} chars (${CORPUS_FILE})`);
  for (const doc of docs) console.log(`  ${doc.id.padEnd(16)} ${doc.text.length}`);
  const run = replayNew(docs);
  const stats = bucketStats(run.timings);
  const cpuStats = bucketStats(run.timings, true);
  printBuckets('markstream: parse per paint, wall', stats);
  printBuckets('markstream: parse per paint, main-thread cpu', cpuStats);
  const creates = [...run.mdCreateMs].sort((a, b) => a - b);
  console.log(`  parser instance: p50 ${percentile(creates, 0.5).toFixed(2)} ms, max ${creates.at(-1)!.toFixed(2)} ms`);
  for (const f of run.finalMs.filter((f) => f.chars >= 50_000))
    console.log(`  final parse at ${f.chars} chars: ${f.ms.toFixed(1)} ms`);
  console.log(
    `  top-level node reuse ${((100 * run.kept) / Math.max(1, run.nodes)).toFixed(1)} % (${run.kept}/${run.nodes}); paints with every node kept ${run.quietPaints}`,
  );
  // The hold on the recorded answers: chunks that did not get their own update.
  let held = 0;
  let paints = 0;
  for (const doc of run.paints) {
    const cpuAt = new Map(doc.ends.map((end, i) => [end, doc.cpuMs[i]!]));
    const paced = paceTimeline(doc.ends, (end, final) => (final ? doc.finalMs : (cpuAt.get(end) ?? 0)), true);
    held += doc.ends.length + 1 - paced.updates;
    paints += doc.ends.length;
  }
  const overFrame = run.timings.filter((t) => paceHoldMs(t.ms) > 0).length;
  console.log(`  paced: ${held} of ${paints} chunks held (cpu); ${overFrame} paints over one frame (wall)`);
  // At 50K and up: p95 <= 4 ms in every bucket; max <= 16 ms up to 100K, <= 25 ms past it (a rare full re-parse of
  // an open list or quote costs a frame or two on answers that long).
  const verdict = (list: readonly BucketStats[]): string =>
    list
      .filter((s) => s.bucket !== '<10K' && s.bucket !== '10-50K')
      .every((s) => s.p95 <= 4 && s.max <= (s.bucket === '50-100K' ? 16 : 25))
      ? 'PASS'
      : 'FAIL';
  console.log(
    `gate (>=50K: p95 <= 4 ms; max <= 16 ms to 100K, 25 ms past): wall ${verdict(stats)}, cpu ${verdict(cpuStats)}`,
  );

  let baseline: Baseline | null = null;
  try {
    baseline = JSON.parse(readFileSync(BASELINE_FILE, 'utf8')) as Baseline;
  } catch {
    console.log(`no baseline at ${BASELINE_FILE}`);
  }
  const flashesNew = run.docs.reduce((n, d) => n + d.flashes, 0);
  if (baseline !== null) {
    printBuckets('old parser (baseline): parse per paint', baseline.buckets);
    const old = new Map(baseline.docs.map((d) => [d.id, d]));
    let flashesOld = 0;
    let compared = 0;
    const changes = new Map<string, number>();
    let tooDifferent = 0;
    for (const doc of run.docs) {
      const before = old.get(doc.id);
      if (before === undefined || before.hash !== doc.hash) continue;
      compared += 1;
      flashesOld += before.flashes;
      const diff = wordChanges(words(before.final), words(doc.final));
      if (diff === null) {
        tooDifferent += 1;
        continue;
      }
      for (const change of diff) changes.set(change, (changes.get(change) ?? 0) + 1);
    }
    console.log(`flashes over ${compared} docs: markstream ${flashesNew}, old ${flashesOld}`);
    for (const doc of run.docs) {
      const before = old.get(doc.id);
      if (before !== undefined && (doc.flashes > 0 || before.flashes > 0))
        console.log(`  ${doc.id.padEnd(16)} markstream ${doc.flashes}  old ${before.flashes}`);
    }
    const top = [...changes.entries()].sort((a, b) => b[1] - a[1]);
    console.log(
      `final text: ${top.length} distinct word changes${tooDifferent > 0 ? `, ${tooDifferent} docs past the diff limit` : ''}`,
    );
    for (const [change, count] of top.slice(0, 60))
      console.log(`  ${String(count).padStart(5)}  ${change.slice(0, 160)}`);
  } else console.log(`flashes: markstream ${flashesNew}`);
  writeJson(RESULT_FILE, {
    buckets: stats,
    cpuBuckets: cpuStats,
    docs: run.docs.map(({ final: _final, ...rest }) => rest),
  });
}

if (import.meta.main) {
  if (process.argv.includes('--hostile')) hostileMain();
  else main();
}
