/**
 * One turn of a session as lanes of spans: the main agent and the subagents it handed work to. Pure data → data: the
 * reader opens the files, this module only reads what it is given. Positions are recorded event times; nothing is moved
 * or predicted, and every edge that was not recorded is flagged as inferred.
 */

import type { ParsedUpdates } from './transcript.js';
import type {
  SessionTimelineResponse,
  TimelineHandoff,
  TimelineLane,
  TimelineLaneStatus,
  TimelineNoSteps,
  TimelineSpan,
} from './types.js';

/** Subagent lanes drawn in detail; the rest fold into one lane. */
export const MAX_TIMELINE_LANES = 8;
/** Spans kept per lane (the longest); the rest are counted in `notes.droppedSpans`. */
export const MAX_LANE_SPANS = 2000;
/** Tool names that spawn a subagent. */
export const SPAWN_TOOL_NAMES: ReadonlySet<string> = new Set(['task', 'Task', 'spawn_subagent']);

/** A transcript entry without its text: kinds, times, tool names, statuses and usage only. */
export type DigestEntry =
  | { kind: 'user'; ts: number; interjection: boolean }
  /** A hidden prompt: a background event woke the agent. */
  | { kind: 'wake'; ts: number }
  | { kind: 'assistant' | 'thinking'; ts: number; endTs: number }
  | { kind: 'tool'; ts: number; endTs: number | null; name: string; failed: boolean }
  | { kind: 'plan' | 'model'; ts: number }
  | {
      kind: 'subagent';
      ts: number;
      /** Time of the finish line. */
      endTs: number | null;
      childId: string;
      type: string;
      description: string;
      status: string | null;
      failed: boolean;
      durationMs: number | null;
      noSpawn: boolean;
    }
  | {
      kind: 'turn-end';
      ts: number;
      stopReason: string;
      elapsedMs: number | null;
      tokens: number;
      model: string | null;
    };

export type RetryLine = ParsedUpdates['retries'][number];

/** What one session file gives a lane. */
export interface LaneSource {
  entries: DigestEntry[];
  retries: RetryLine[];
  lastTs: number;
}

/** Facts about one child id from the parent's folder, known without reading the child's events. */
export interface TimelineChildInfo {
  /** `meta.json` status; `running` while the child has not finished. null without a meta. */
  status: string | null;
  type: string;
  description: string;
  model: string | null;
  /** The child's session folder is indexed. */
  exists: boolean;
  /** Subagents the child spawned itself. */
  subagents: number;
}

export type ChildEvents =
  | { ok: true; source: LaneSource }
  | { ok: false; noSteps: 'missing' | 'empty' | 'too-large' | 'unreadable'; fileBytes: number | null };

export interface TimelineInput {
  /** 1-based turn number. */
  turn: number;
  parent: LaneSource;
  live: boolean;
  asOfMs: number;
  /** `model` is the fallback when the turn's usage names none. */
  main: { name: string; type: string | null; model: string | null; sessionId: string };
  children: ReadonlyMap<string, TimelineChildInfo>;
}

type Entry<K extends DigestEntry['kind']> = Extract<DigestEntry, { kind: K }>;

/** `parseUpdates` output → the digest a lane needs; no text survives. */
export function digestUpdates(parsed: ParsedUpdates): LaneSource {
  const entries: DigestEntry[] = [];
  let ended = 0;
  for (const e of parsed.transcript) {
    switch (e.kind) {
      case 'user':
        entries.push({ kind: 'user', ts: e.ts, interjection: e.interjection === true });
        break;
      case 'wake':
        entries.push({ kind: 'wake', ts: e.ts });
        break;
      case 'assistant':
      case 'thinking':
        entries.push({ kind: e.kind, ts: e.ts, endTs: e.endTs ?? e.ts });
        break;
      case 'tool':
        entries.push({
          kind: 'tool',
          ts: e.ts,
          endTs: e.endTs ?? null,
          name: parsed.toolNames.get(e.id) ?? 'tool',
          failed: e.status === 'failed',
        });
        break;
      case 'plan':
      case 'model':
        entries.push({ kind: e.kind, ts: e.ts });
        break;
      case 'subagent':
        entries.push({
          kind: 'subagent',
          ts: e.ts,
          endTs: e.endTs ?? null,
          childId: e.childId,
          type: e.type,
          description: e.description,
          status: e.status ?? null,
          failed: e.status === 'failed' || e.error !== undefined,
          durationMs: e.durationMs ?? null,
          noSpawn: e.noSpawn === true,
        });
        break;
      case 'turn-end': {
        const metric = parsed.turns[ended];
        ended += 1;
        entries.push({
          kind: 'turn-end',
          ts: e.ts,
          stopReason: e.stopReason,
          elapsedMs: e.elapsedMs ?? null,
          tokens: metric === undefined ? 0 : metric.inputTokens + metric.outputTokens,
          model: metric?.model ?? null,
        });
        break;
      }
    }
  }
  return { entries, retries: parsed.retries, lastTs: parsed.lastTs };
}

export interface TurnWindow<E> {
  /** 1-based; 0 for entries that precede every turn and belong to none. */
  number: number;
  /** Without the closing turn-end. */
  entries: E[];
  /** null = still open. */
  end: E | null;
}

/**
 * The transcript partitioned into turns exactly as the SPA's `groupTurns` does: leading entries before the first prompt
 * (typed or a wake) that hold more than model switches are window 0, the k-th turn-end closes window k, and entries
 * after the last turn-end form the open window.
 */
export function turnWindows<E extends { kind: string }>(entries: readonly E[]): TurnWindow<E>[] {
  const windows: TurnWindow<E>[] = [];
  let start = 0;
  const firstPrompt = entries.findIndex((e) => e.kind === 'user' || e.kind === 'wake');
  const firstEnd = entries.findIndex((e) => e.kind === 'turn-end');
  if (firstPrompt > 0 && (firstEnd === -1 || firstPrompt < firstEnd)) {
    const leading = entries.slice(0, firstPrompt);
    if (leading.some((e) => e.kind !== 'model')) {
      windows.push({ number: 0, entries: leading, end: null });
      start = firstPrompt;
    }
  }
  let current: E[] = [];
  let closed = 0;
  for (let i = start; i < entries.length; i += 1) {
    const entry = entries[i]!;
    if (entry.kind !== 'turn-end') {
      current.push(entry);
      continue;
    }
    closed += 1;
    windows.push({ number: closed, entries: current, end: entry });
    current = [];
  }
  if (current.length > 0) windows.push({ number: closed + 1, entries: current, end: null });
  return windows;
}

/** Whether trailing entries are a turn: at least one prompt (typed or a wake), text or tool entry. */
export function hasTurnWork(entries: readonly { kind: string }[]): boolean {
  return entries.some(
    (e) =>
      e.kind === 'user' || e.kind === 'wake' || e.kind === 'assistant' || e.kind === 'thinking' || e.kind === 'tool',
  );
}

// ---------------------------------------------------------------------------------------------------------------
// Spans of one lane
// ---------------------------------------------------------------------------------------------------------------

type DraftKind = 'generate' | 'tool' | 'wait-subagent' | 'wait-model' | 'run' | 'idle';

/** A span in epoch ms; `end` is `Infinity` while it runs. */
export interface LaneSpan {
  kind: DraftKind;
  start: number;
  end: number;
  inferredStart: boolean;
  inferredEnd: boolean;
  /** The turn's window cut this side: the span went on past it. */
  cutStart: boolean;
  cutEnd: boolean;
  status: TimelineSpan['status'];
  /** Recorded with zero length: kept, never cut away. */
  zero: boolean;
  phase: 'reasoning' | 'answer';
  until: 'text' | 'tool' | 'open';
  tools: string[];
  calls: number;
  failedCalls: number;
  noSteps: TimelineNoSteps;
  retry: { attempt: number; max: number } | null;
}

export interface LaneBounds {
  /** Where the cursor starts: the window start (main lane) or the spawn time (child run). */
  fromMs: number;
  fromExact: boolean;
  /** Where a span with no recorded end and nothing after it stops; null = the lane is still running. */
  end: { ms: number; exact: boolean } | null;
  /** Applied in order; `LaneResult.cut` says which of them cut a span. `window`: the turn's window. */
  clips: ReadonlyArray<{ lo: number; hi: number; window: boolean }>;
  /**
   * The lane's first prompt, text or tool time after the given entries (a later turn's prompt); null = none. A child
   * that returns after it was not waited for.
   */
  nextOwnMs: number | null;
}

export interface LaneResult {
  spans: LaneSpan[];
  cut: boolean[];
  unplacedRetries: number;
}

/** Lower tiers are drawn over higher ones. */
const TIER: Record<DraftKind, number> = {
  generate: 0,
  tool: 1,
  'wait-subagent': 2,
  'wait-model': 3,
  run: 4,
  idle: 5,
};

function draft(kind: DraftKind, start: number, end: number, extra: Partial<LaneSpan> = {}): LaneSpan {
  return {
    kind,
    start,
    end,
    inferredStart: false,
    inferredEnd: false,
    cutStart: false,
    cutEnd: false,
    status: end === Infinity ? 'running' : 'ok',
    zero: start === end,
    phase: 'answer',
    until: 'tool',
    tools: [],
    calls: 0,
    failedCalls: 0,
    noSteps: 'missing',
    retry: null,
    ...extra,
  };
}

function byStart(a: LaneSpan, b: LaneSpan): number {
  return a.start - b.start || a.end - b.end || TIER[a.kind] - TIER[b.kind];
}

/** A span that no longer ends at `Infinity` cannot stay `running`. */
function settle(span: LaneSpan): void {
  if (span.status === 'running' && span.end !== Infinity) span.status = 'ok';
}

/** Members sorted by start → one span with the merged facts. */
function mergeSpans(members: readonly LaneSpan[]): LaneSpan {
  const first = members[0]!;
  let start = Infinity;
  let end = -Infinity;
  for (const m of members) {
    start = Math.min(start, m.start);
    end = Math.max(end, m.end);
  }
  const inferredStart = members.every((m) => m.start !== start || m.inferredStart);
  const inferredEnd = members.every((m) => m.end !== end || m.inferredEnd);
  const tools: string[] = [];
  let calls = 0;
  let failedCalls = 0;
  let failed = false;
  for (const m of members) {
    calls += m.calls;
    failedCalls += m.failedCalls;
    if (m.status === 'failed') failed = true;
    for (const name of m.tools) if (tools.length < 3 && !tools.includes(name)) tools.push(name);
  }
  const status: LaneSpan['status'] =
    end === Infinity ? 'running' : failed || failedCalls > 0 ? 'failed' : inferredEnd ? 'unfinished' : 'ok';
  const cutStart = members.some((m) => m.start === start && m.cutStart);
  const cutEnd = members.some((m) => m.end === end && m.cutEnd);
  return {
    ...first,
    start,
    end,
    inferredStart,
    inferredEnd,
    cutStart,
    cutEnd,
    status,
    zero: start === end,
    tools,
    calls,
    failedCalls,
  };
}

/** Spans sorted by start that touch or overlap merge. */
function mergeTouching(spans: readonly LaneSpan[]): LaneSpan[] {
  const out: LaneSpan[] = [];
  let group: LaneSpan[] = [];
  let groupEnd = -Infinity;
  for (const span of [...spans].sort(byStart)) {
    if (group.length > 0 && span.start <= groupEnd) {
      group.push(span);
      groupEnd = Math.max(groupEnd, span.end);
      continue;
    }
    if (group.length > 0) out.push(mergeSpans(group));
    group = [span];
    groupEnd = span.end;
  }
  if (group.length > 0) out.push(mergeSpans(group));
  return out;
}

type Interval = [number, number];

function byInterval(x: Interval, y: Interval): number {
  if (x[0] !== y[0]) return x[0] < y[0] ? -1 : 1;
  if (x[1] !== y[1]) return x[1] < y[1] ? -1 : 1;
  return 0;
}

/**
 * The first interval that ends after `x`. The intervals of `stack` never overlap and no point lies inside one, so
 * sorted by start they are sorted by end too.
 */
function firstEndAfter(list: readonly Interval[], x: number): number {
  return firstIndex(list, (item) => item[1] > x);
}

/** The first index where `pred` holds, for a `pred` that once true stays true along `list`; `list.length` if none. */
function firstIndex<T>(list: readonly T[], pred: (item: T) => boolean): number {
  let lo = 0;
  let hi = list.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (pred(list[mid]!)) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

function strictlyInside(list: readonly Interval[], x: number): boolean {
  const i = firstEndAfter(list, x);
  return i < list.length && list[i]![0] < x;
}

/** `[p0, p1]` with every interval of `list` taken out; a point inside it splits it. */
function freeParts(list: readonly Interval[], p0: number, p1: number, out: Interval[]): void {
  let at = p0;
  for (let i = firstEndAfter(list, p0); i < list.length && list[i]![0] < p1; i += 1) {
    const [a, b] = list[i]!;
    if (a > at) out.push([at, a]);
    if (b > at) at = b;
  }
  if (at < p1) out.push([at, p1]);
}

function insertInterval(list: Interval[], item: Interval): void {
  let lo = 0;
  let hi = list.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (byInterval(list[mid]!, item) <= 0) lo = mid + 1;
    else hi = mid;
  }
  if (lo === list.length) list.push(item);
  else list.splice(lo, 0, item);
}

/** A lane never draws two spans over each other: a lower tier is cut where a higher one stands. */
export function stack(spans: readonly LaneSpan[]): LaneSpan[] {
  const order = [...spans].sort((a, b) => TIER[a.kind] - TIER[b.kind] || byStart(a, b));
  // Intervals of the tiers above, and of the tier being placed: a tier inserts into its own short list.
  let above: Interval[] = [];
  let same: Interval[] = [];
  let tier = -1;
  const out: LaneSpan[] = [];
  for (const span of order) {
    if (TIER[span.kind] !== tier) {
      above = [...above, ...same].sort(byInterval);
      same = [];
      tier = TIER[span.kind];
    }
    if (span.zero) {
      if (strictlyInside(above, span.start) || strictlyInside(same, span.start)) continue;
      out.push(span);
      insertInterval(same, [span.start, span.start]);
      continue;
    }
    const outside: Interval[] = [];
    freeParts(above, span.start, span.end, outside);
    const pieces: Interval[] = [];
    for (const [p0, p1] of outside) freeParts(same, p0, p1, pieces);
    for (const [p0, p1] of pieces) {
      if (p1 - p0 < 1) continue;
      const piece: LaneSpan = {
        ...span,
        start: p0,
        end: p1,
        inferredStart: span.inferredStart && p0 === span.start,
        inferredEnd: span.inferredEnd && p1 === span.end,
        cutStart: span.cutStart && p0 === span.start,
        cutEnd: span.cutEnd && p1 === span.end,
        tools: [...span.tools],
      };
      settle(piece);
      out.push(piece);
      insertInterval(same, [p0, p1]);
    }
  }
  return out.sort(byStart);
}

/** Cut to `[lo, hi]`; `cut` is true when a span crossed an edge. `window`: the edges are the turn's window. */
function clipSpans(
  spans: readonly LaneSpan[],
  lo: number,
  hi: number,
  window: boolean,
): { spans: LaneSpan[]; cut: boolean } {
  const out: LaneSpan[] = [];
  let cut = false;
  for (const span of spans) {
    if (span.zero) {
      if (span.start >= lo && span.start <= hi) out.push(span);
      continue;
    }
    if (span.end <= lo || span.start >= hi) continue;
    const p0 = Math.max(span.start, lo);
    const p1 = Math.min(span.end, hi);
    if (p0 > span.start || p1 < span.end) cut = true;
    if (p1 - p0 < 1) continue;
    const piece: LaneSpan = {
      ...span,
      start: p0,
      end: p1,
      inferredStart: span.inferredStart && p0 === span.start,
      inferredEnd: span.inferredEnd && p1 === span.end,
      cutStart: span.cutStart || (window && p0 > span.start),
      cutEnd: span.cutEnd || (window && p1 < span.end),
    };
    settle(piece);
    out.push(piece);
  }
  return { spans: out, cut };
}

/** A retry goes to the model span that holds it, else to the next one; never to a tool or a wait for subagents. */
export function placeRetries(spans: readonly LaneSpan[], retries: readonly RetryLine[]): number {
  // Stacked spans never overlap, so sorted by start they are sorted by end too.
  const model = spans.filter((s) => s.kind === 'generate' || s.kind === 'wait-model').sort(byStart);
  let unplaced = 0;
  for (const retry of [...retries].sort((a, b) => a.ts - b.ts)) {
    const holder = firstIndex(model, (s) => s.end >= retry.ts);
    const at =
      holder < model.length && model[holder]!.start <= retry.ts
        ? holder
        : firstIndex(model, (s) => s.start >= retry.ts);
    const target = model[at];
    if (target === undefined) {
      unplaced += 1;
      continue;
    }
    if (target.retry === null || retry.attempt > target.retry.attempt) {
      target.retry = { attempt: retry.attempt, max: retry.max };
    }
  }
  return unplaced;
}

function isActivityOrPrompt(e: DigestEntry): boolean {
  return e.kind === 'user' || e.kind === 'wake' || e.kind === 'assistant' || e.kind === 'thinking' || e.kind === 'tool';
}

interface Activity {
  type: 'text' | 'block' | 'wait' | 'user';
  start: number;
  end: number;
  endInferred: boolean;
  order: number;
}

/**
 * The spans of one lane from its own entries, by one cursor walk: the model was asked (`request`) or was streaming
 * (`stream`). Retries are model request retries of the same lane.
 */
export function laneSpans(
  entries: readonly DigestEntry[],
  retries: readonly RetryLine[],
  bounds: LaneBounds,
): LaneResult {
  const timed = entries.filter((e) => e.ts > 0).sort((a, b) => a.ts - b.ts);

  const nextNonTool: Array<number | null> = Array.from({ length: timed.length }, () => null);
  const nextNonSpawn: Array<number | null> = Array.from({ length: timed.length }, () => null);
  const nextOwn: Array<number | null> = Array.from({ length: timed.length }, () => null);
  let lastWork = -1;
  for (let i = timed.length - 1; i >= 0; i -= 1) {
    const after = timed[i + 1];
    nextNonTool[i] = after === undefined ? null : after.kind === 'tool' ? nextNonTool[i + 1]! : after.ts;
    // A spawn line is part of the spawn call that runs around it: it never ends a spawn tool.
    nextNonSpawn[i] =
      after === undefined ? null : after.kind === 'tool' || after.kind === 'subagent' ? nextNonSpawn[i + 1]! : after.ts;
    nextOwn[i] = after === undefined ? null : isActivityOrPrompt(after) ? after.ts : nextOwn[i + 1]!;
    const kind = timed[i]!.kind;
    if (lastWork === -1 && (isActivityOrPrompt(timed[i]!) || kind === 'subagent' || kind === 'turn-end')) lastWork = i;
  }
  // A lane whose last line is its own turn end has nothing under way, even while its session runs: it ends there.
  const stopMs = lastWork !== -1 && timed[lastWork]!.kind === 'turn-end' ? timed[lastWork]!.ts : null;
  const laneStop = bounds.end ?? (stopMs === null ? null : { ms: stopMs, exact: true });
  const running = laneStop === null;
  const laneEnd = laneStop === null ? Infinity : laneStop.ms;
  const laneEndExact = laneStop === null || laneStop.exact;

  const generate: LaneSpan[] = [];
  const tools: Array<{ span: LaneSpan; spawn: boolean; order: number }> = [];
  const activities: Activity[] = [];
  const lineWaits: LaneSpan[] = [];

  timed.forEach((e, i) => {
    if (e.kind === 'assistant' || e.kind === 'thinking') {
      const open = running && i === lastWork;
      const end = open ? Infinity : Math.max(e.ts, e.endTs);
      generate.push(draft('generate', e.ts, end, { phase: e.kind === 'thinking' ? 'reasoning' : 'answer' }));
      activities.push({ type: 'text', start: e.ts, end, endInferred: false, order: i });
    } else if (e.kind === 'tool') {
      const spawn = SPAWN_TOOL_NAMES.has(e.name);
      const next = spawn ? nextNonSpawn[i]! : nextNonTool[i]!;
      let end: number;
      let inferredEnd = false;
      if (e.endTs !== null) {
        end = Math.max(e.ts, e.endTs);
      } else if (next !== null) {
        end = Math.max(e.ts, next);
        inferredEnd = true;
      } else if (running) {
        end = Infinity;
      } else {
        end = Math.max(e.ts, laneEnd);
        inferredEnd = true;
      }
      const status: LaneSpan['status'] =
        end === Infinity ? 'running' : e.failed ? 'failed' : inferredEnd ? 'unfinished' : 'ok';
      const span = draft(spawn ? 'wait-subagent' : 'tool', e.ts, end, {
        inferredEnd,
        status,
        tools: [e.name],
        calls: 1,
        failedCalls: e.failed ? 1 : 0,
      });
      tools.push({ span, spawn, order: i });
    } else if (e.kind === 'user' || e.kind === 'wake') {
      activities.push({ type: 'user', start: e.ts, end: e.ts, endInferred: false, order: i });
    }
  });

  // Tool blocks: tools whose intervals touch or overlap; regular and spawn tools merge apart.
  const blockSpans: LaneSpan[] = [];
  let block: typeof tools = [];
  let blockEnd = -Infinity;
  const closeBlock = (): void => {
    if (block.length === 0) return;
    const all = mergeSpans(block.map((t) => t.span));
    activities.push({
      type: 'block',
      start: all.start,
      end: all.end,
      endInferred: all.inferredEnd,
      order: block[0]!.order,
    });
    const regular = block.filter((t) => !t.spawn).map((t) => t.span);
    const spawns = block.filter((t) => t.spawn).map((t) => t.span);
    if (regular.length > 0) blockSpans.push(mergeSpans(regular));
    if (spawns.length > 0) lineWaits.push(mergeSpans(spawns));
    block = [];
  };
  for (const tool of tools) {
    if (block.length > 0 && tool.span.start > blockEnd) closeBlock();
    if (block.length === 0) blockEnd = tool.span.end;
    block.push(tool);
    blockEnd = Math.max(blockEnd, tool.span.end);
  }
  closeBlock();

  // Spawn lines with no spawn tool around them: a wait when the lane did nothing of its own until the child returned.
  const spawnTools = tools.filter((t) => t.spawn).map((t) => t.span);
  const spawnReach: number[] = [];
  for (const t of spawnTools) spawnReach.push(Math.max(spawnReach.at(-1) ?? -Infinity, t.end));
  timed.forEach((e, i) => {
    if (e.kind !== 'subagent' || e.noSpawn || e.ts < bounds.fromMs) return;
    const s = e.ts;
    const started = firstIndex(spawnTools, (t) => t.start > s);
    if (started > 0 && spawnReach[started - 1]! >= s) return;
    const r = e.endTs !== null && e.endTs >= s ? e.endTs : null;
    const n = nextOwn[i]!;
    const nextAny = n ?? bounds.nextOwnMs;
    let wait: LaneSpan | null = null;
    if (r !== null && (nextAny === null || r <= nextAny)) {
      wait = draft('wait-subagent', s, r);
    } else if (r === null && n === null) {
      wait = running
        ? draft('wait-subagent', s, Infinity)
        : draft('wait-subagent', s, Math.max(s, laneEnd), {
            inferredEnd: !laneEndExact,
            status: laneEndExact ? 'ok' : 'unfinished',
          });
    }
    if (wait === null) return;
    lineWaits.push(wait);
    activities.push({ type: 'wait', start: wait.start, end: wait.end, endInferred: wait.inferredEnd, order: i });
  });

  // The cursor walk adds the waits for the model.
  const modelWaits: LaneSpan[] = [];
  let cursor = bounds.fromMs;
  let exact = bounds.fromExact;
  let after: 'request' | 'stream' = 'request';
  for (const act of activities.sort((a, b) => a.start - b.start || a.order - b.order)) {
    if (act.type === 'user') {
      if (act.start >= cursor) {
        cursor = act.start;
        exact = true;
        after = 'request';
      }
      continue;
    }
    if (after === 'request' && act.start > cursor) {
      modelWaits.push(
        draft('wait-model', cursor, act.start, { inferredStart: !exact, until: act.type === 'text' ? 'text' : 'tool' }),
      );
    }
    if (act.end >= cursor) {
      cursor = act.end;
      exact = !act.endInferred;
      after = act.type === 'text' ? 'stream' : 'request';
    }
  }
  if (running && after === 'request' && cursor !== Infinity) {
    modelWaits.push(draft('wait-model', cursor, Infinity, { inferredStart: !exact, until: 'open' }));
  }

  let spans = stack([...generate, ...blockSpans, ...mergeTouching(lineWaits), ...modelWaits]);
  const cut: boolean[] = [];
  for (const clip of bounds.clips) {
    const clipped = clipSpans(spans, clip.lo, clip.hi, clip.window);
    spans = clipped.spans;
    cut.push(clipped.cut);
  }
  const unplacedRetries = placeRetries(spans, retries);
  return { spans, cut, unplacedRetries };
}

// ---------------------------------------------------------------------------------------------------------------
// The turn
// ---------------------------------------------------------------------------------------------------------------

interface WindowTimes {
  /** Axis origin: `turnStartMs`, or earlier for a woken turn with background runs in the gap before it. */
  startMs: number;
  turnStartMs: number;
  endMs: number | null;
  startExact: boolean;
  leadMs: number;
}

const isPrompt = (e: DigestEntry): boolean => (e.kind === 'user' && !e.interjection) || e.kind === 'wake';

/** The smallest time > 0 of the entries `keep` accepts; null = none. */
function earliest(entries: readonly DigestEntry[], keep: (e: DigestEntry) => boolean): number | null {
  let min: number | null = null;
  for (const e of entries) if (keep(e) && e.ts > 0 && (min === null || e.ts < min)) min = e.ts;
  return min;
}

/**
 * The start of a window's turn: the prompt or the start the agent measured, whichever is earlier. Without a timed
 * prompt the earliest interjection or own work gives it, spawn lines last. null = no usable time.
 */
function windowTimes(window: TurnWindow<DigestEntry>): Omit<WindowTimes, 'startMs'> | null {
  let promptMs = window.entries.find((e) => isPrompt(e) && e.ts > 0)?.ts ?? null;
  promptMs ??= earliest(
    window.entries,
    (e) =>
      e.kind === 'user' || e.kind === 'assistant' || e.kind === 'thinking' || e.kind === 'tool' || e.kind === 'plan',
  );
  promptMs ??= earliest(window.entries, (e) => e.kind === 'subagent');
  const end = window.end !== null && window.end.kind === 'turn-end' ? window.end : null;
  const elapsed = end?.elapsedMs ?? null;
  const measuredMs =
    end !== null && end.ts > 0 && elapsed !== null && Number.isFinite(elapsed) && elapsed > 0 ? end.ts - elapsed : null;
  if (promptMs === null && measuredMs === null) return null;
  const turnStartMs = promptMs === null ? measuredMs! : measuredMs === null ? promptMs : Math.min(promptMs, measuredMs);
  const endMs = end === null ? null : end.ts;
  if (endMs !== null && endMs < turnStartMs) return null;
  return {
    turnStartMs,
    endMs,
    startExact: measuredMs !== null && (promptMs === null || promptMs >= measuredMs),
    leadMs: measuredMs === null ? 0 : measuredMs - turnStartMs,
  };
}

/**
 * Where the axis of a woken turn starts: the earliest moment after the previous turn's end at which a run spawned
 * before the turn was active. null = no such run, or the turn was typed.
 */
function stretchStart(
  window: TurnWindow<DigestEntry>,
  prevEndMs: number | null,
  turnStartMs: number,
  runs: readonly Run[],
): number | null {
  if (prevEndMs === null || window.entries.find(isPrompt)?.kind !== 'wake') return null;
  let start: number | null = null;
  for (const run of runs) {
    const s = run.spawn;
    if (s === null || s >= turnStartMs) continue;
    const active = run.ret !== null ? run.ret > prevEndMs : s > prevEndMs;
    if (!active) continue;
    const from = Math.max(s, prevEndMs);
    if (start === null || from < start) start = from;
  }
  return start !== null && start < turnStartMs ? start : null;
}

interface Run {
  childId: string;
  entry: Entry<'subagent'>;
  spawn: number | null;
  ret: number | null;
  /** A finish line was recorded. */
  finished: boolean;
}

function runsOf(entries: readonly DigestEntry[]): Run[] {
  const runs: Run[] = [];
  for (const e of entries) {
    if (e.kind !== 'subagent' || e.ts <= 0) continue;
    const spawn = e.noSpawn ? null : e.ts;
    const ret = e.endTs !== null && e.endTs > 0 && (spawn === null || e.endTs >= spawn) ? e.endTs : null;
    runs.push({ childId: e.childId, entry: e, spawn, ret, finished: e.endTs !== null });
  }
  return runs;
}

interface Plan {
  window: TurnWindow<DigestEntry>;
  turns: number;
  open: boolean;
  times: WindowTimes | null;
  /** Where a lane that never ended is cut. */
  closeMs: number;
  /** Child ids in lane order with their runs in this window. */
  lanes: Array<{ childId: string; runs: Run[] }>;
  folded: Array<{ childId: string; runs: Run[] }>;
  /** Ordinal of every child id of the whole transcript. */
  ordinals: Map<string, number>;
}

function plan(input: TimelineInput): Plan | 'no-turn' {
  const entries = input.parent.entries;
  const windows = turnWindows(entries);
  const turns = entries.filter((e) => e.kind === 'turn-end').length;
  const window = windows.find((w) => w.number === input.turn && w.number > 0);
  if (window === undefined) return 'no-turn';
  const open = window.end === null;
  if (open && !hasTurnWork(window.entries)) return 'no-turn';

  const ordinals = new Map<string, number>();
  for (const e of entries)
    if (e.kind === 'subagent' && !ordinals.has(e.childId)) ordinals.set(e.childId, ordinals.size);

  const turnTimes = windowTimes(window);
  if (turnTimes === null) return { window, turns, open, times: null, closeMs: 0, lanes: [], folded: [], ordinals };
  const allRuns = runsOf(entries);
  const prevEnd = windows.find((w) => w.number === input.turn - 1)?.end ?? null;
  const prevEndMs = prevEnd !== null && prevEnd.ts > 0 ? prevEnd.ts : null;
  const stretch = stretchStart(window, prevEndMs, turnTimes.turnStartMs, allRuns);
  const times: WindowTimes = { ...turnTimes, startMs: stretch ?? turnTimes.turnStartMs };

  let closeMs = times.endMs ?? times.turnStartMs;
  if (open) {
    for (const e of window.entries) {
      if (e.ts <= 0) continue;
      closeMs = Math.max(closeMs, e.ts);
      if ((e.kind === 'tool' || e.kind === 'subagent') && e.endTs !== null) closeMs = Math.max(closeMs, e.endTs);
      if (e.kind === 'assistant' || e.kind === 'thinking') closeMs = Math.max(closeMs, e.endTs);
    }
    closeMs = Math.max(closeMs, input.parent.lastTs);
  }

  const S = times.startMs;
  const E = times.endMs;
  const inWindow = (t: number): boolean => t >= S && (E === null || t <= E);
  const keyOf = (run: Run): number => {
    if (run.spawn === null) return Math.max(S, run.ret! - (run.entry.durationMs ?? 0));
    return Math.max(S, run.spawn);
  };
  const byChild = new Map<string, Run[]>();
  for (const run of allRuns) {
    let meets: boolean;
    if (run.spawn === null) meets = run.ret !== null && inWindow(run.ret);
    else if (inWindow(run.spawn)) meets = true;
    else if (run.spawn > S) meets = false;
    else if (run.ret !== null) meets = run.ret > S;
    else meets = open && input.live && input.children.get(run.childId)?.status === 'running';
    if (!meets) continue;
    const list = byChild.get(run.childId);
    if (list === undefined) byChild.set(run.childId, [run]);
    else list.push(run);
  }
  const all = [...byChild].map(([childId, runs]) => ({ childId, runs: runs.sort((a, b) => keyOf(a) - keyOf(b)) }));
  all.sort((a, b) => {
    const d = keyOf(a.runs[0]!) - keyOf(b.runs[0]!);
    return d !== 0 ? d : a.childId < b.childId ? -1 : a.childId > b.childId ? 1 : 0;
  });
  // A run of the idle stretch alone must not take a lane from a child the turn itself spawned or waits on.
  const U = times.turnStartMs;
  const meetsTurn = (child: { runs: Run[] }): boolean => child.runs.some((run) => run.ret === null || run.ret >= U);
  const kept = new Set([...all.filter(meetsTurn), ...all.filter((c) => !meetsTurn(c))].slice(0, MAX_TIMELINE_LANES));
  return {
    window,
    turns,
    open,
    times,
    closeMs,
    lanes: all.filter((c) => kept.has(c)),
    folded: all.filter((c) => !kept.has(c)),
    ordinals,
  };
}

/** Child ids whose event files `buildTimeline` reads: drawn lanes whose folder exists and that hold a spawned run. */
export function timelineChildReads(input: TimelineInput): string[] | 'no-turn' {
  const p = plan(input);
  if (p === 'no-turn') return p;
  return p.lanes
    .filter((lane) => input.children.get(lane.childId)?.exists === true && lane.runs.some((r) => r.spawn !== null))
    .map((lane) => lane.childId);
}

function laneStatusOf(run: Run, running: boolean): TimelineLaneStatus {
  if (!run.finished) return running ? 'running' : 'unknown';
  if (run.entry.failed) return 'failed';
  const status = run.entry.status;
  if (status === 'completed' || status === 'cancelled') return status;
  return 'unknown';
}

function toWire(span: LaneSpan, lane: string, index: number, origin: number): TimelineSpan {
  const base = {
    id: `${lane}:${String(index)}`,
    lane,
    startMs: span.start - origin,
    endMs: span.end === Infinity ? null : span.end - origin,
    status: span.status,
    inferredStart: span.inferredStart,
    inferredEnd: span.inferredEnd,
    cutStart: span.cutStart,
    cutEnd: span.cutEnd,
    retry: span.kind === 'generate' || span.kind === 'wait-model' ? span.retry : null,
  };
  switch (span.kind) {
    case 'generate':
      return { ...base, kind: 'generate', phase: span.phase };
    case 'tool':
      return { ...base, kind: 'tool', tools: span.tools, calls: span.calls, failedCalls: span.failedCalls };
    case 'wait-model':
      return { ...base, kind: 'wait', reason: 'model', until: span.until };
    case 'wait-subagent':
      return {
        ...base,
        kind: 'wait',
        reason: 'subagent',
        tools: span.tools,
        calls: span.calls,
        failedCalls: span.failedCalls,
      };
    case 'run':
      return { ...base, kind: 'run', noSteps: span.noSteps };
    case 'idle':
      return { ...base, kind: 'idle' };
  }
}

function spanLength(span: LaneSpan): number {
  return span.end === Infinity ? Number.MAX_VALUE : span.end - span.start;
}

/** Keeps the `MAX_LANE_SPANS` longest spans of a lane, in start order. */
function capSpans(spans: LaneSpan[]): { spans: LaneSpan[]; dropped: number } {
  if (spans.length <= MAX_LANE_SPANS) return { spans, dropped: 0 };
  const kept = [...spans]
    .sort((a, b) => spanLength(b) - spanLength(a) || byStart(a, b))
    .slice(0, MAX_LANE_SPANS)
    .sort(byStart);
  return { spans: kept, dropped: spans.length - MAX_LANE_SPANS };
}

function emptyResponse(input: TimelineInput, p: Plan, untimed: number): SessionTimelineResponse {
  return {
    turn: input.turn,
    turns: p.turns,
    open: p.open,
    live: input.live,
    stopReason: p.window.end?.kind === 'turn-end' ? p.window.end.stopReason : null,
    asOfMs: input.asOfMs,
    range: null,
    lanes: [],
    spans: [],
    handoffs: [],
    notes: { untimedEvents: untimed, foldedSubagents: 0, droppedSpans: 0, unplacedRetries: 0 },
  };
}

/** One turn as lanes, spans and handoffs; `events` holds the child files that `timelineChildReads` named. */
export function buildTimeline(
  input: TimelineInput,
  events: ReadonlyMap<string, ChildEvents>,
): SessionTimelineResponse | 'no-turn' {
  const p = plan(input);
  if (p === 'no-turn') return p;
  const untimed = p.window.entries.filter((e) => e.ts <= 0).length;
  if (p.times === null) return emptyResponse(input, p, untimed);
  const { startMs: S, turnStartMs: U, endMs: E } = p.times;
  const winHi = E ?? Infinity;
  const running = p.open && input.live;
  const end = p.window.end?.kind === 'turn-end' ? p.window.end : null;

  const lanes: TimelineLane[] = [];
  const laneSpanLists: LaneSpan[][] = [];
  const handoffs: TimelineHandoff[] = [];
  let unplacedRetries = 0;
  let lastEventMs = S;
  for (const e of p.window.entries) {
    if (e.ts <= 0) continue;
    lastEventMs = Math.max(lastEventMs, e.ts);
    if ((e.kind === 'tool' || e.kind === 'subagent') && e.endTs !== null) lastEventMs = Math.max(lastEventMs, e.endTs);
    if (e.kind === 'assistant' || e.kind === 'thinking') lastEventMs = Math.max(lastEventMs, e.endTs);
  }
  if (p.open) lastEventMs = Math.max(lastEventMs, input.parent.lastTs);

  // Main lane. Whether it waited for a spawn with no spawn tool can depend on the next turn's prompt.
  let nextOwnMs: number | null = null;
  const endIndex = p.window.end === null ? -1 : input.parent.entries.indexOf(p.window.end);
  if (endIndex !== -1) {
    for (let i = endIndex + 1; i < input.parent.entries.length; i += 1) {
      const e = input.parent.entries[i]!;
      if (e.ts > 0 && isActivityOrPrompt(e) && (nextOwnMs === null || e.ts < nextOwnMs)) nextOwnMs = e.ts;
    }
  }
  const mainEnd = E !== null ? { ms: E, exact: true } : running ? null : { ms: p.closeMs, exact: false };
  // The main lane starts at the turn; before it, on the axis of a woken turn, no turn was open.
  const main = laneSpans(
    p.window.entries,
    input.parent.retries.filter((r) => r.ts >= U && r.ts <= winHi),
    {
      fromMs: U,
      fromExact: p.times.startExact,
      end: mainEnd,
      clips: [{ lo: U, hi: E ?? (running ? Infinity : p.closeMs), window: true }],
      nextOwnMs,
    },
  );
  unplacedRetries += main.unplacedRetries;
  if (S < U) main.spans.unshift(draft('idle', S, U));
  const mainStatus: TimelineLaneStatus =
    end === null
      ? running
        ? 'running'
        : 'unknown'
      : end.stopReason === 'error'
        ? 'failed'
        : end.stopReason === 'cancelled'
          ? 'cancelled'
          : 'completed';
  lanes.push({
    id: 'main',
    role: 'main',
    name: input.main.name,
    type: input.main.type,
    model: end?.model ?? input.main.model,
    slot: null,
    sessionId: input.main.sessionId,
    status: mainStatus,
    carriedIn: false,
    carriedOut: false,
    noSteps: null,
    fileBytes: null,
    clamped: false,
    subagents: p.lanes.length + p.folded.length,
    tokenReports: end !== null && end.tokens > 0 ? [{ atMs: E! - S, tokens: end.tokens }] : [],
  });
  laneSpanLists.push(main.spans);

  // Subagent lanes.
  const usedSlots = new Set<number>();
  for (const { childId, runs } of p.lanes) {
    const info = input.children.get(childId);
    const childRunning = running && info?.status === 'running';
    const source = events.get(childId);
    const spans: LaneSpan[] = [];
    const tokenReports: Array<{ atMs: number; tokens: number }> = [];
    let clamped = false;
    let barReason: TimelineNoSteps | null = null;
    let fileBytes: number | null = null;
    let allBars = true;
    runs.forEach((run, index) => {
      const runRunning = childRunning && !run.finished;
      const runHi = run.ret ?? (runRunning ? Infinity : p.closeMs);
      const bar = (noSteps: TimelineNoSteps): void => {
        const start = run.spawn ?? Math.max(S, run.ret! - (run.entry.durationMs ?? 0));
        const stop = run.ret ?? (runRunning ? Infinity : Math.max(start, p.closeMs));
        spans.push(
          draft('run', start, stop, {
            noSteps,
            inferredStart: run.spawn === null,
            inferredEnd: run.ret === null && !runRunning,
            status: runRunning ? 'running' : run.entry.failed ? 'failed' : run.ret === null ? 'unfinished' : 'ok',
          }),
        );
        if (barReason === null) barReason = noSteps;
      };
      if (run.spawn === null) bar('no-spawn');
      else if (info === undefined || !info.exists) bar('missing');
      else if (source === undefined) bar('unreadable');
      else if (!source.ok) {
        bar(source.noSteps);
        if (source.fileBytes !== null) fileBytes = source.fileBytes;
      } else if (source.source.entries.length === 0) {
        bar('empty');
      } else {
        const spawn = run.spawn;
        const inRun = source.source.entries.filter((e) => {
          if (e.ts <= 0) return false;
          const stop = 'endTs' in e && e.endTs !== null ? Math.max(e.ts, e.endTs) : e.ts;
          return e.ts <= runHi && stop >= spawn;
        });
        if (inRun.length === 0) {
          bar('out-of-range');
        } else {
          allBars = false;
          const runEnd =
            run.ret !== null ? { ms: run.ret, exact: true } : runRunning ? null : { ms: p.closeMs, exact: false };
          const lane = laneSpans(
            inRun,
            source.source.retries.filter((r) => r.ts >= spawn && r.ts <= runHi && r.ts >= S && r.ts <= winHi),
            {
              fromMs: spawn,
              fromExact: true,
              end: runEnd,
              clips: [
                { lo: spawn, hi: runHi, window: false },
                { lo: S, hi: E ?? (running ? Infinity : p.closeMs), window: true },
              ],
              nextOwnMs: null,
            },
          );
          if (lane.cut[0] === true) clamped = true;
          unplacedRetries += lane.unplacedRetries;
          spans.push(...lane.spans);
          for (const e of inRun) {
            if (e.kind === 'turn-end' && e.tokens > 0 && e.ts >= S && e.ts <= winHi) {
              tokenReports.push({ atMs: e.ts - S, tokens: e.tokens });
            }
          }
        }
      }
      if (run.spawn !== null && run.spawn >= S && run.spawn <= winHi) {
        handoffs.push({
          id: `${childId}:spawn:${String(index)}`,
          lane: childId,
          kind: 'spawn',
          atMs: run.spawn - S,
          failed: false,
        });
      }
      if (run.ret !== null && run.ret >= S && run.ret <= winHi) {
        handoffs.push({
          id: `${childId}:return:${String(index)}`,
          lane: childId,
          kind: 'return',
          atMs: run.ret - S,
          failed: run.entry.failed,
        });
      }
    });
    const drawn = stack(clipSpans(spans, S, E ?? (running ? Infinity : p.closeMs), true).spans);
    laneSpanLists.push(drawn);
    for (const span of drawn) if (span.end !== Infinity) lastEventMs = Math.max(lastEventMs, span.end);

    let slot = (p.ordinals.get(childId) ?? 0) % MAX_TIMELINE_LANES;
    while (usedSlots.has(slot)) slot = (slot + 1) % MAX_TIMELINE_LANES;
    usedSlots.add(slot);
    const first = runs[0]!;
    const last = runs[runs.length - 1]!;
    const description = runs.map((r) => r.entry.description).find((d) => d !== '') ?? info?.description ?? '';
    const type = runs.map((r) => r.entry.type).find((t) => t !== '') ?? info?.type ?? '';
    lanes.push({
      id: childId,
      role: 'subagent',
      name: description !== '' ? description : type !== '' ? type : 'Subagent',
      type: type !== '' ? type : null,
      model: info?.model ?? null,
      slot,
      sessionId: info?.exists === true ? childId : null,
      status: laneStatusOf(last, childRunning),
      carriedIn: first.spawn !== null && first.spawn < S,
      carriedOut: E !== null && last.ret !== null && last.ret > E,
      noSteps: allBars ? barReason : null,
      fileBytes: allBars && barReason === 'too-large' ? fileBytes : null,
      clamped,
      subagents: info?.subagents ?? 0,
      tokenReports: tokenReports.sort((a, b) => a.atMs - b.atMs),
    });
  }

  // Folded lane: one bar per run, no handoffs, no files read.
  if (p.folded.length > 0) {
    const bars: LaneSpan[] = [];
    for (const { childId, runs } of p.folded) {
      const childRunning = running && input.children.get(childId)?.status === 'running';
      for (const run of runs) {
        const runRunning = childRunning && !run.finished;
        const start = run.spawn ?? Math.max(S, run.ret! - (run.entry.durationMs ?? 0));
        const stop = run.ret ?? (runRunning ? Infinity : Math.max(start, p.closeMs));
        bars.push(
          draft('run', start, stop, {
            noSteps: 'folded',
            inferredStart: run.spawn === null,
            inferredEnd: run.ret === null && !runRunning,
            status: runRunning ? 'running' : run.entry.failed ? 'failed' : run.ret === null ? 'unfinished' : 'ok',
          }),
        );
      }
    }
    laneSpanLists.push(clipSpans(mergeTouching(bars), S, E ?? (running ? Infinity : p.closeMs), true).spans);
    lanes.push({
      id: 'folded',
      role: 'folded',
      name: `${String(p.folded.length)} more subagents`,
      type: null,
      model: null,
      slot: null,
      sessionId: null,
      status: 'unknown',
      carriedIn: false,
      carriedOut: false,
      noSteps: 'folded',
      fileBytes: null,
      clamped: false,
      subagents: p.folded.length,
      tokenReports: [],
    });
  }

  const spans: TimelineSpan[] = [];
  let droppedSpans = 0;
  lanes.forEach((lane, i) => {
    const capped = capSpans(laneSpanLists[i]!);
    droppedSpans += capped.dropped;
    capped.spans.forEach((span, index) => spans.push(toWire(span, lane.id, index, S)));
  });
  const laneOrder = new Map(lanes.map((lane, i) => [lane.id, i]));
  handoffs.sort((a, b) => laneOrder.get(a.lane)! - laneOrder.get(b.lane)! || a.atMs - b.atMs);

  return {
    turn: input.turn,
    turns: p.turns,
    open: p.open,
    live: input.live,
    stopReason: end?.stopReason ?? null,
    asOfMs: input.asOfMs,
    range: {
      startMs: S,
      turnStartMs: U,
      endMs: E,
      lastEventMs: E === null ? lastEventMs : Math.min(lastEventMs, E),
      startExact: p.times.startExact,
      leadMs: p.times.leadMs,
    },
    lanes,
    spans,
    handoffs,
    notes: { untimedEvents: untimed, foldedSubagents: p.folded.length, droppedSpans, unplacedRetries },
  };
}
