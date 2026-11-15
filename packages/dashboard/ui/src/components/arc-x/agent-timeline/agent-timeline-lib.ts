import { fitLeft, fitTop, type Box } from '../../../lib/float-box';
import { niceTicks } from '../chart/scale';

/** 'idle': the main agent between two turns, drawn before a turn that a background event started. */
export type AgentSpanKind = 'generate' | 'tool' | 'wait' | 'run' | 'idle';
export type AgentSpanStatus = 'ok' | 'failed' | 'running' | 'unfinished';

export interface AgentLane {
  id: string;
  name: string;
  /** Model id for the second line; printed after its family mark. */
  model?: string;
  /** Plain text after the model: "+2 subagents". */
  extra?: string;
  /** CSS color expression: 'var(--lane-3)'. */
  color: string;
  /** Route of the lane's own session; the name is then a link. */
  to?: string;
  /** Word after the name when the lane did not end well. */
  outcome?: { text: string; tone: 'danger' | 'warning' };
}

export interface AgentSpan {
  id: string;
  laneId: string;
  kind: AgentSpanKind;
  startMs: number;
  /** null = running: drawn to the clock (live) or to the stop point (stale). */
  endMs: number | null;
  status: AgentSpanStatus;
  /** The edge whose time was inferred. */
  open?: 'start' | 'end' | 'both';
  /** The edge where the turn cut a step that went on past it: square, no fade. */
  cut?: 'start' | 'end' | 'both';
  /** "Retry 1 of 3": draws the retry glyph and joins the accessible name. */
  retry?: string;
  /** Card label and accessible name: "Tool call · run_terminal_command". */
  label: string;
  /** Short facts for the card's meta line, after the duration: "3 calls", "1 failed". */
  facts?: readonly string[];
  /** One sentence under the meta line: why an edge is open, why there are no steps. */
  note?: string;
  /** A main-lane wait on subagents or idle stretch: the handoffs inside it belong to its selection path. */
  ownsHandoffs?: boolean;
}

export interface AgentHandoff {
  id: string;
  /** The child lane; the other end is lanes[0]. */
  laneId: string;
  kind: 'spawn' | 'return';
  atMs: number;
  failed?: boolean;
}

export interface AgentTokenReport {
  laneId: string;
  atMs: number;
  tokens: number;
}

export type AgentTimelineMode =
  /** A finished record: opens settled, can be replayed and scrubbed. */
  | { kind: 'recorded' }
  /** Follows the clock: now = clock.now() − originMs. No replay, no scrub. */
  | { kind: 'live'; originMs: number }
  /** A live turn whose last body is old: stands still at atMs. No replay, no scrub. */
  | { kind: 'stale'; atMs: number; asOfText: string };

export interface AgentOutcome {
  /** "Finished", "Failed", "Cancelled", "Ended · max_tokens", "Not finished". */
  word: string;
  tone: 'success' | 'danger' | 'warning' | 'neutral';
  /** Replaces the total in the settled status line and the final summary: "last event 14:32:10". */
  detail?: string;
}

export interface AgentClock {
  now(): number;
  requestFrame(callback: (time: number) => void): number;
  cancelFrame(handle: number): void;
}

export interface AxisUnit {
  unitMs: number;
  format: (offsetMs: number) => string;
}

/** Bars drawn at most; the hidden table lists every span. */
export const MAX_DRAWN_SPANS = 400;
/** A replay at the first speed lasts at most this long. */
export const REPLAY_MAX_MS = 20_000;

const REPLAY_RATES = [1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200, 1800, 3600] as const;
const SCRUB_STEPS_MS = [
  100, 200, 500, 1000, 2000, 5000, 10_000, 15_000, 30_000, 60_000, 120_000, 300_000, 600_000,
] as const;
const LIVE_AXIS_MS = [
  10_000, 20_000, 50_000, 100_000, 120_000, 300_000, 600_000, 1_200_000, 3_000_000, 6_000_000,
] as const;
const MINUTE_MS = 60_000;

/** The first speed of a replay: a true multiplier of recorded time that keeps the replay within 20 s. */
export function replayRate(durationMs: number): number {
  const length = Math.max(1, durationMs);
  for (const rate of REPLAY_RATES) if (length / rate <= REPLAY_MAX_MS) return rate;
  return Math.ceil(length / REPLAY_MAX_MS);
}

/** One arrow-key step of the scrubber, in ms: about a sixtieth of the record on a round value. */
export function scrubStep(durationMs: number): number {
  const target = Math.max(1, durationMs) / 60;
  for (const step of SCRUB_STEPS_MS) if (step >= target) return step;
  return Math.ceil(target / MINUTE_MS) * MINUTE_MS;
}

/** The live axis length for a clock offset `x`: the first 1-2-5 step that leaves a tenth of room. */
export function liveAxisEnd(x: number): number {
  const need = Math.max(0, x) / 0.9;
  for (const end of LIVE_AXIS_MS) if (end >= need) return end;
  if (!Number.isFinite(need)) return LIVE_AXIS_MS[LIVE_AXIS_MS.length - 1]!;
  let decade = 100 * MINUTE_MS;
  for (;;) {
    for (const factor of [2, 5, 10]) if (decade * factor >= need) return decade * factor;
    decade *= 10;
  }
}

/** The live playhead: the elapsed time itself, or whole seconds when nothing may travel. */
export function liveNow(elapsedMs: number, staticMotion: boolean): number {
  const elapsed = Math.max(0, elapsedMs);
  return staticMotion ? Math.floor(elapsed / 1000) * 1000 : elapsed;
}

/** Ticks of an axis `axisMs` long, round in `unitMs`, never past the axis end. */
export function axisTicks(axisMs: number, unitMs: number, target = 5): number[] {
  const length = Math.max(1, axisMs);
  const unit = unitMs > 0 ? unitMs : 1;
  const ticks: number[] = [];
  for (const tick of niceTicks(0, length / unit, target).ticks) {
    const ms = Number((tick * unit).toPrecision(12));
    if (ms >= 0 && ms <= length) ticks.push(ms);
  }
  return ticks;
}

export interface TokensAt {
  /** The first lane's reported tokens; null before its first report. */
  main: number | null;
  /** The other lanes' reports from the turn start on; null before any. */
  sub: number | null;
  /** Lanes other than the first with such a report by then. */
  subLanes: number;
  /** The other lanes' reports before the turn start; null when none. */
  before: number | null;
  beforeLanes: number;
}

/** What had been reported by `t`: the first lane's figure, and the others' split at `turnStartMs`. */
export function tokensAt(
  reports: readonly AgentTokenReport[],
  mainLaneId: string,
  t: number,
  turnStartMs = 0,
): TokensAt {
  let main: number | null = null;
  let sub: number | null = null;
  let before: number | null = null;
  const lanes = new Set<string>();
  const beforeLanes = new Set<string>();
  for (const report of reports) {
    if (report.atMs > t) continue;
    if (report.laneId === mainLaneId) main = (main ?? 0) + report.tokens;
    else if (report.atMs < turnStartMs) {
      before = (before ?? 0) + report.tokens;
      beforeLanes.add(report.laneId);
    } else {
      sub = (sub ?? 0) + report.tokens;
      lanes.add(report.laneId);
    }
  }
  return { main, sub, subLanes: lanes.size, before, beforeLanes: beforeLanes.size };
}

/** Lanes with a span under way at `t`; an idle agent is not working. */
export function activeLanes(spans: readonly AgentSpan[], t: number): number {
  const lanes = new Set<string>();
  for (const span of spans) {
    if (span.kind === 'idle') continue;
    if (span.startMs <= t && (span.endMs === null || span.endMs > t)) lanes.add(span.laneId);
  }
  return lanes.size;
}

export interface SelectionPath {
  spans: ReadonlySet<string>;
  handoffs: ReadonlySet<string>;
}

function spanEnd(span: AgentSpan): number {
  return span.endMs ?? Number.POSITIVE_INFINITY;
}

/** The runs of a lane: spawn → return pairs of its handoffs; a missing edge is open. */
function laneRuns(
  laneId: string,
  spans: readonly AgentSpan[],
  handoffs: readonly AgentHandoff[],
): Array<[number, number]> {
  const own = handoffs.filter((handoff) => handoff.laneId === laneId).sort((a, b) => a.atMs - b.atMs);
  const runs: Array<[number, number]> = [];
  let open: number | null = null;
  for (const handoff of own) {
    if (handoff.kind === 'spawn') {
      if (open !== null) runs.push([open, Number.POSITIVE_INFINITY]);
      open = handoff.atMs;
    } else {
      runs.push([open ?? Number.NEGATIVE_INFINITY, handoff.atMs]);
      open = null;
    }
  }
  if (open !== null) runs.push([open, Number.POSITIVE_INFINITY]);
  if (runs.length > 0) return runs;
  const laneSpans = spans.filter((span) => span.laneId === laneId);
  if (laneSpans.length === 0) return [];
  let start = Number.POSITIVE_INFINITY;
  let end = Number.NEGATIVE_INFINITY;
  for (const span of laneSpans) {
    start = Math.min(start, span.startMs);
    end = Math.max(end, spanEnd(span));
  }
  return [[start, end]];
}

/** What stays at full color around a selected span (the rest dims). */
export function selectionPath(
  selectedId: string | null,
  spans: readonly AgentSpan[],
  handoffs: readonly AgentHandoff[],
  mainLaneId: string,
): SelectionPath | null {
  if (selectedId === null) return null;
  const selected = spans.find((span) => span.id === selectedId);
  if (selected === undefined) return null;
  const pathSpans = new Set<string>([selected.id]);
  const pathHandoffs = new Set<string>();
  if (selected.laneId !== mainLaneId) {
    const lane = selected.laneId;
    for (const span of spans) if (span.laneId === lane) pathSpans.add(span.id);
    for (const handoff of handoffs) if (handoff.laneId === lane) pathHandoffs.add(handoff.id);
    const runs = laneRuns(lane, spans, handoffs);
    for (const span of spans) {
      if (span.laneId !== mainLaneId || span.kind !== 'wait') continue;
      if (runs.some(([from, to]) => span.startMs < to && spanEnd(span) > from)) pathSpans.add(span.id);
    }
    return { spans: pathSpans, handoffs: pathHandoffs };
  }
  if (selected.ownsHandoffs !== true) return { spans: pathSpans, handoffs: pathHandoffs };
  for (const handoff of handoffs) {
    if (handoff.atMs < selected.startMs || handoff.atMs > spanEnd(selected)) continue;
    pathHandoffs.add(handoff.id);
    const child = spans.filter((span) => span.laneId === handoff.laneId);
    if (child.length === 0) continue;
    if (handoff.kind === 'spawn') {
      const first = child.find((span) => span.startMs >= handoff.atMs) ?? child[0]!;
      pathSpans.add(first.id);
    } else {
      let last = child[child.length - 1]!;
      for (const span of child) if (spanEnd(span) <= handoff.atMs) last = span;
      pathSpans.add(last.id);
    }
  }
  return { spans: pathSpans, handoffs: pathHandoffs };
}

/** At most `cap` bars: over it each lane keeps its longest spans (budget = cap ÷ lanes), in their order. */
export function drawnSpans(spans: readonly AgentSpan[], cap = MAX_DRAWN_SPANS): { drawn: AgentSpan[]; hidden: number } {
  if (spans.length <= cap) return { drawn: [...spans], hidden: 0 };
  const byLane = new Map<string, AgentSpan[]>();
  for (const span of spans) {
    const list = byLane.get(span.laneId);
    if (list === undefined) byLane.set(span.laneId, [span]);
    else list.push(span);
  }
  const budget = Math.max(1, Math.floor(cap / byLane.size));
  const kept = new Set<string>();
  for (const list of byLane.values()) {
    const length = (span: AgentSpan): number => (span.endMs === null ? Number.MAX_VALUE : span.endMs - span.startMs);
    const longest = [...list].sort((a, b) => length(b) - length(a));
    for (const span of longest.slice(0, budget)) kept.add(span.id);
  }
  const drawn = spans.filter((span) => kept.has(span.id));
  return { drawn, hidden: spans.length - drawn.length };
}

/** A span whose start the playhead has not passed. A settled timeline shows every span. */
export function isPending(span: AgentSpan, t: number, settled: boolean): boolean {
  return !settled && !(span.startMs < t);
}

/**
 * The plot's one tab stop: the active span when shown; when it is pending, the last shown span before it on its lane,
 * else the first shown span. null = no span is shown: the plot element itself.
 */
export function tabStop(
  activeId: string | null,
  spans: readonly AgentSpan[],
  t: number,
  settled: boolean,
): string | null {
  const shown = spans.filter((span) => !isPending(span, t, settled));
  if (shown.length === 0) return null;
  const active = activeId === null ? undefined : spans.find((span) => span.id === activeId);
  if (active === undefined) return shown[0]!.id;
  if (!isPending(active, t, settled)) return active.id;
  let before: AgentSpan | undefined;
  for (const span of shown) {
    if (span.laneId === active.laneId && span.startMs <= active.startMs) before = span;
  }
  return (before ?? shown[0]!).id;
}

export type FocusKey = 'ArrowLeft' | 'ArrowRight' | 'ArrowUp' | 'ArrowDown' | 'Home' | 'End';

/**
 * Where a key moves focus among the shown spans (ordered by lane, then start). `whole`: ⌘/Ctrl held, so Home / End go to
 * the first / last span of the timeline. null = nowhere to go.
 */
export function moveFocus(
  currentId: string,
  key: FocusKey,
  shown: readonly AgentSpan[],
  laneOrder: readonly string[],
  whole = false,
): string | null {
  const here = shown.find((span) => span.id === currentId);
  if (here === undefined) return shown[0]?.id ?? null;
  const inLane = shown.filter((span) => span.laneId === here.laneId);
  const at = inLane.indexOf(here);
  const nearestIn = (laneId: string): string | null => {
    let best: AgentSpan | null = null;
    for (const span of shown) {
      if (span.laneId !== laneId) continue;
      if (best === null || Math.abs(span.startMs - here.startMs) < Math.abs(best.startMs - here.startMs)) best = span;
    }
    return best?.id ?? null;
  };
  const lanes = laneOrder.filter((lane) => shown.some((span) => span.laneId === lane));
  const laneAt = lanes.indexOf(here.laneId);
  switch (key) {
    case 'ArrowRight':
      return inLane[Math.min(inLane.length - 1, at + 1)]!.id;
    case 'ArrowLeft':
      return inLane[Math.max(0, at - 1)]!.id;
    case 'ArrowDown':
      return laneAt < lanes.length - 1 ? nearestIn(lanes[laneAt + 1]!) : here.id;
    case 'ArrowUp':
      return laneAt > 0 ? nearestIn(lanes[laneAt - 1]!) : here.id;
    case 'Home':
      return whole ? shown[0]!.id : inLane[0]!.id;
    case 'End':
      return whole ? shown[shown.length - 1]!.id : inLane[inLane.length - 1]!.id;
  }
}

export interface CardAnchor {
  /** The span's visible left edge, top and bottom, in px from the window's top-left corner. */
  left: number;
  top: number;
  bottom: number;
}

/**
 * The detail card's place in window px: above its span (6 px gap) when the whole card fits, else below, else on the
 * side with more room, moved inside `limits` (the visible page area). Left edge: the span's left − 12 px. Always 8 px
 * inside `limits`.
 */
export function cardPlace(
  anchor: CardAnchor,
  card: { width: number; height: number },
  limits: Box,
): { x: number; y: number; below: boolean } {
  const x = fitLeft(anchor.left - 12, card.width, limits);
  const { y, below } = fitTop(anchor.top - 6 - card.height, anchor.bottom + 6, card.height, limits);
  return { x, y, below };
}

/** The axis unit by default: seconds, printed with the duration formatter. */
export function secondsAxis(format: (ms: number) => string): (spanMs: number) => AxisUnit {
  return () => ({ unitMs: 1000, format });
}
