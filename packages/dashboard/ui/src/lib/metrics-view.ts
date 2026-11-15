/** Pure transforms for the Metrics page, the Overview numbers and the turn drawer. React-free. */

import type { ChartPoint, ChartSeries } from '../components/arc-x/chart/types';
import type { HistogramBin } from '../components/arc-x/histogram/histogram';
import type { TimelineMarker, TimelineRow, TimelineSpan } from '../components/arc-x/span-timeline/span-timeline';
import type { BarSegment } from '../components/arc-x/split-bar/split-bar';
import type { StatDelta } from '../components/arc-x/stat-tile/stat-tile';
import {
  formatCount,
  formatCountNoun,
  formatDuration,
  formatDurationParts,
  formatNumber,
  formatRate,
  formatTps,
} from './format';
import { PREFILL_UNDER_TEXT } from './prefill';
import type { AlignedRange, RangeKey } from './range';
import { pivotMeasures } from './timeseries';
import {
  LOW_SAMPLE_TAIL_N,
  type Histogram,
  type MetricsBucket,
  type MetricsSpeculative,
  type MetricsSummary,
  type MetricsTimeseries,
  type MetricsTotals,
  type TurnRow,
} from './types';

type TsPoint = MetricsTimeseries['points'][number];

/** The one hint every empty metrics region shares. */
export const RECORD_HINT =
  'Turns are recorded when a client uses the local server. Records older than 30 days are removed.';

export type MetricsTab = 'summary' | 'speed' | 'usage' | 'speculation' | 'failures' | 'turns';
export type KpiKey = 'decode' | 'ttft' | 'turns' | 'tokens' | 'failed';
export type OutcomeFilter = 'all' | 'ok' | 'error' | 'cancelled';

export const METRICS_TABS: ReadonlyArray<{ value: MetricsTab; label: string }> = [
  { value: 'summary', label: 'Summary' },
  { value: 'speed', label: 'Speed' },
  { value: 'usage', label: 'Usage' },
  { value: 'speculation', label: 'Speculation' },
  { value: 'failures', label: 'Failures' },
  { value: 'turns', label: 'Turns' },
];

export const KPI_KEYS: readonly KpiKey[] = ['decode', 'ttft', 'turns', 'tokens', 'failed'];

export function isMetricsTab(value: string): value is MetricsTab {
  return (
    value === 'summary' ||
    value === 'speed' ||
    value === 'usage' ||
    value === 'speculation' ||
    value === 'failures' ||
    value === 'turns'
  );
}

export function isKpiKey(value: string): value is KpiKey {
  return value === 'decode' || value === 'ttft' || value === 'turns' || value === 'tokens' || value === 'failed';
}

export function isOutcomeFilter(value: string): value is OutcomeFilter {
  return value === 'all' || value === 'ok' || value === 'error' || value === 'cancelled';
}

/**
 * A value already divided into its axis unit (`timeAxis`, `byteAxis`), printed without the unit: the chart prints the
 * unit beside it. Three significant digits, grouped from 1,000, trailing zeros dropped; 0 prints `0`.
 */
export function axisNumber(value: number): string {
  if (!Number.isFinite(value)) return '—';
  const size = Math.abs(value);
  if (size >= 100) return formatNumber(value);
  let text = value.toFixed(size >= 10 ? 1 : size >= 1 ? 2 : 3);
  if (text.includes('.')) {
    let end = text.length;
    while (end > 0 && text[end - 1] === '0') end -= 1;
    if (end > 0 && text[end - 1] === '.') end -= 1;
    text = text.slice(0, end);
  }
  return text === '-0' || text === '' ? '0' : text;
}

/** Axis unit for a time chart: max < 2000 ms -> ms (div 1); < 120 s -> s (div 1000); else min (div 60000). */
export function timeAxis(maxMs: number): { unit: 'ms' | 's' | 'min'; div: number } {
  if (!(maxMs >= 2000)) return { unit: 'ms', div: 1 };
  if (maxMs < 120_000) return { unit: 's', div: 1000 };
  return { unit: 'min', div: 60_000 };
}

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;

/** Axis unit for bytes: MB below 2 GiB, else GB. Base 1024. */
export function byteAxis(maxBytes: number): { unit: 'MB' | 'GB'; div: number } {
  return maxBytes >= 2 * GIB ? { unit: 'GB', div: GIB } : { unit: 'MB', div: MIB };
}

/** API histogram (lower bounds + counts) -> HistogramBin[]; last bin to = Infinity. */
export function toBins(edges: readonly number[], histogram: Histogram | undefined): HistogramBin[] {
  return edges.map((from, index) => ({
    from,
    to: index + 1 < edges.length ? edges[index + 1] : Infinity,
    count: histogram?.counts[index] ?? 0,
  }));
}

/** Phase durations of one turn, each clamped so they sum to preDispatch + duration. */
export interface TurnPhases {
  handlingMs: number;
  resolveMs: number;
  loadWaitMs: number;
  queueMs: number;
  /** The model was busy with other requests: before the time to first token. */
  waitMs: number;
  /** No lane or memory to start: inside the time to first token, or before the end of a turn that never had one. */
  slotMs: number;
  /** Prompt preparation before the prefill: placed as `slotMs`. */
  prepareMs: number;
  /** Time to first token of a turn without a prefill window, net of slot and prepare; 0 when measured. */
  firstTokenMs: number;
  /** Time to first token outside the prefill window, slot and prepare; 0 when not measured. */
  setupMs: number;
  /** The measured prefill window, capped at time to first token; 0 when not measured. */
  prefillMs: number;
  decodeMs: number;
  tailMs: number;
  totalMs: number;
}

function nonNegative(value: number | null | undefined): number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : 0;
}

export function turnPhases(row: TurnRow): TurnPhases {
  const pre = nonNegative(row.preDispatchMs);
  const queueMs = Math.min(nonNegative(row.queueMs), pre);
  const loadWaitMs = Math.min(nonNegative(row.loadWaitMs), pre - queueMs);
  const resolveMs = Math.min(nonNegative(row.resolveMs), pre - queueMs - loadWaitMs);
  const handlingMs = pre - queueMs - loadWaitMs - resolveMs;
  const dur = nonNegative(row.durationMs);
  const measured = row.prefillMs !== null;
  const ttft = nonNegative(row.ttftMs);
  const lead = Math.min(ttft > 0 ? ttft : measured ? nonNegative(row.prefillMs) : 0, dur);
  const prefillMs = measured ? Math.min(nonNegative(row.prefillMs), lead) : 0;
  const waitMs = Math.min(nonNegative(row.waitMs), dur - lead);
  const decodeMs = Math.min(nonNegative(row.decodeMs), dur - lead - waitMs);
  // A turn with no first token has no lead: its slot and prepare time sits in the rest of the duration.
  const inLead = lead > 0;
  const room = inLead ? lead - prefillMs : dur - waitMs - decodeMs;
  const slotMs = Math.min(nonNegative(row.slotMs), room);
  const prepareMs = Math.min(nonNegative(row.prepareMs), room - slotMs);
  const beforePrefill = inLead ? slotMs + prepareMs : 0;
  const setupMs = measured ? lead - prefillMs - beforePrefill : 0;
  const firstTokenMs = measured ? 0 : lead - beforePrefill;
  const tailMs = dur - lead - waitMs - decodeMs - (inLead ? 0 : slotMs + prepareMs);
  return {
    handlingMs,
    resolveMs,
    loadWaitMs,
    queueMs,
    waitMs,
    slotMs,
    prepareMs,
    firstTokenMs,
    setupMs,
    prefillMs,
    decodeMs,
    tailMs,
    totalMs: pre + dur,
  };
}

/** One order everywhere: bar, legend, timeline, table. */
export const PHASES: ReadonlyArray<{ key: string; label: string; color: string; pattern: 'solid' | 'hatch' }> = [
  { key: 'queue', label: 'Queue', color: 'var(--phase-queue)', pattern: 'hatch' },
  { key: 'loadWait', label: 'Waiting for a model load', color: 'var(--phase-wait)', pattern: 'hatch' },
  { key: 'resolve', label: 'Loading the model', color: 'var(--phase-wait)', pattern: 'solid' },
  { key: 'wait', label: 'Model busy', color: 'var(--phase-queue)', pattern: 'solid' },
  { key: 'slot', label: 'Waiting for a slot', color: 'var(--phase-slot)', pattern: 'hatch' },
  { key: 'prepare', label: 'Preparing', color: 'var(--phase-prepare)', pattern: 'solid' },
  { key: 'first', label: 'To first token', color: 'var(--phase-prefill)', pattern: 'hatch' },
  { key: 'setup', label: 'Setup', color: 'var(--phase-setup)', pattern: 'solid' },
  { key: 'prefill', label: 'Prefill', color: 'var(--phase-prefill)', pattern: 'solid' },
  { key: 'decode', label: 'Decode', color: 'var(--phase-decode)', pattern: 'solid' },
  { key: 'other', label: 'Other', color: 'var(--border-strong)', pattern: 'solid' },
];

/** The phases as chart series: hatch reads as `alt` in a bar chart. */
export const PHASE_SERIES: ChartSeries[] = PHASES.map((phase) => ({
  key: phase.key,
  label: phase.label,
  color: phase.color,
  pattern: phase.pattern === 'hatch' ? 'alt' : 'solid',
}));

function isTotals(value: TurnPhases | MetricsTotals): value is MetricsTotals {
  return 'totalWallMs' in value;
}

/** Phases a record may lack: left out of a bar at 0, so a record from an older engine shows no empty part. */
const OPTIONAL_PHASES: ReadonlySet<string> = new Set(['first', 'wait', 'slot', 'prepare']);

/**
 * Segments for a SplitBar: handling and tail are merged into `other`. `first`,
 * `wait`, `slot` and `prepare` are left out at 0; `setup` and `prefill` are
 * left out when both are 0 and `first` is not.
 */
export function phaseSegments(phases: TurnPhases | MetricsTotals): BarSegment[] {
  const values: Record<string, number> = isTotals(phases)
    ? {
        queue: phases.queueWallMs,
        loadWait: phases.loadWaitWallMs,
        resolve: phases.resolveWallMs,
        wait: phases.waitWallMs,
        slot: phases.slotWallMs,
        prepare: phases.prepareWallMs,
        first: phases.firstTokenWallMs,
        setup: phases.setupWallMs,
        prefill: phases.prefillWallMs,
        decode: phases.decodeWallMs,
        other: phases.otherWallMs,
      }
    : {
        queue: phases.queueMs,
        loadWait: phases.loadWaitMs,
        resolve: phases.resolveMs,
        wait: phases.waitMs,
        slot: phases.slotMs,
        prepare: phases.prepareMs,
        first: phases.firstTokenMs,
        setup: phases.setupMs,
        prefill: phases.prefillMs,
        decode: phases.decodeMs,
        other: phases.handlingMs + phases.tailMs,
      };
  const value = (key: string): number => nonNegative(values[key]);
  const unsplitOnly = value('first') > 0 && value('setup') === 0 && value('prefill') === 0;
  return PHASES.filter((phase) => {
    if (OPTIONAL_PHASES.has(phase.key)) return value(phase.key) > 0;
    if (phase.key === 'setup' || phase.key === 'prefill') return !unsplitOnly;
    return true;
  }).map((phase) => ({
    key: phase.key,
    label: phase.label,
    value: value(phase.key),
    color: phase.color,
    pattern: phase.pattern,
  }));
}

/** Mean wall time per turn and phase per bucket; a bucket with no turn is all null. */
export function phaseMeans(points: readonly TsPoint[], range: AlignedRange): ChartPoint[] {
  const mean = (pick: (t: MetricsTotals) => number) => (p: TsPoint) =>
    p.totals.turns > 0 ? pick(p.totals) / p.totals.turns : null;
  const data = pivotMeasures(
    points,
    range,
    {
      queue: mean((t) => t.queueWallMs),
      loadWait: mean((t) => t.loadWaitWallMs),
      resolve: mean((t) => t.resolveWallMs),
      wait: mean((t) => t.waitWallMs),
      slot: mean((t) => t.slotWallMs),
      prepare: mean((t) => t.prepareWallMs),
      first: mean((t) => t.firstTokenWallMs),
      setup: mean((t) => t.setupWallMs),
      prefill: mean((t) => t.prefillWallMs),
      decode: mean((t) => t.decodeWallMs),
      other: mean((t) => t.otherWallMs),
    },
    null,
  );
  const turns = pivotMeasures(points, range, { turns: (p) => p.totals.turns }, 0);
  return data.map((point, index) => {
    const n = turns[index]?.values.turns ?? 0;
    const samples: Record<string, number> = {};
    for (const phase of PHASES) samples[phase.key] = n ?? 0;
    return { ...point, samples };
  });
}

const FAIL_PHASE_TEXT: Record<string, string> = {
  queued: 'while waiting',
  slot: 'while waiting for a slot',
  prepare: 'while preparing',
  prefill: 'during prefill',
  decode: 'during decode',
};

const ERROR_KIND_TEXT: Record<string, string> = {
  finish_error: 'model reported an error',
  exception: 'exception',
};

export function failPhaseText(phase: string | null): string | null {
  return phase === null ? null : (FAIL_PHASE_TEXT[phase] ?? null);
}

export function errorKindText(kind: string | null): string | null {
  return kind === null ? null : (ERROR_KIND_TEXT[kind] ?? kind);
}

export function outcomeText(outcome: string): string {
  return outcome === 'error' ? 'Failed' : outcome === 'cancelled' ? 'Cancelled' : 'OK';
}

/** `Cancelled · during decode`, `Failed · exception · prefill`: null parts are left out. */
export function failureLabel(row: MetricsSummary['failures'][number]): string {
  const parts = [outcomeText(row.outcome), errorKindText(row.errorKind), failPhaseText(row.failPhase)];
  return parts.filter((part): part is string => part !== null).join(' · ');
}

/** `Stopped during decode.` for a failed or cancelled turn with a known phase, else null. */
export function stopSentence(row: Pick<TurnRow, 'outcome' | 'failPhase'>): string | null {
  if (row.outcome === 'ok') return null;
  const text = failPhaseText(row.failPhase);
  return text === null ? null : `Stopped ${text}.`;
}

const SPAN_LABEL = {
  handling: 'Request handling',
  resolve: 'Loading the model',
  loadWait: 'Waiting for a model load',
  queue: 'Queue',
  wait: 'Model busy',
  slot: 'Waiting for a slot',
  prepare: 'Preparing',
  first: 'To first token',
  setup: 'Setup',
  prefill: 'Prefill',
  decode: 'Decode',
  tail: 'Other',
} as const;

/** The waterfall of one turn, origin = handler start. Pre-dispatch spans are placed back to back ending at dispatch. */
export function turnTimeline(row: TurnRow): {
  rows: TimelineRow[];
  spans: TimelineSpan[];
  markers: TimelineMarker[];
  domainMs: [number, number];
} {
  const phases = turnPhases(row);
  const pre = phases.handlingMs + phases.resolveMs + phases.loadWaitMs + phases.queueMs;
  const rowId = 'turn';
  const rows: TimelineRow[] = [{ id: rowId, label: 'Turn' }];
  const stop = row.outcome === 'ok' ? null : row.failPhase;
  const beforePrefill = stop === 'queued' || stop === 'slot' || stop === 'prepare';
  const spans: TimelineSpan[] = [];
  let at = 0;
  const push = (
    id: keyof typeof SPAN_LABEL,
    kind: TimelineSpan['kind'],
    length: number,
    estimated: boolean,
    detail?: TimelineSpan['detail'],
  ): void => {
    const start = at;
    at += length;
    if (length <= 0) return;
    spans.push({
      id,
      rowId,
      kind,
      startMs: start,
      endMs: start + length,
      label: SPAN_LABEL[id],
      estimated,
      ...(detail === undefined ? {} : { detail }),
    });
  };
  push('handling', 'other', phases.handlingMs, true);
  push('resolve', 'load', phases.resolveMs, true);
  push('loadWait', 'wait', phases.loadWaitMs, true);
  push('queue', 'queue', phases.queueMs, true);
  push('wait', 'busy', phases.waitMs, false, [{ label: 'Waiting for', value: 'other requests on this model' }]);
  push('slot', 'slot', phases.slotMs, false, [{ label: 'Waiting for', value: 'a free lane or memory' }]);
  push('prepare', 'prepare', phases.prepareMs, false, [
    { label: 'Cached tokens', value: formatNumber(row.cachedTokens) },
    { label: 'Work', value: 'prompt, prefix match, cache restore' },
  ]);
  if (!beforePrefill) {
    if (row.prefillMs !== null) {
      push('setup', 'setup', phases.setupMs, true, [
        {
          label: 'Before prefill',
          value:
            row.prepareMs !== null && row.prepareMs > 0
              ? 'other requests on this model'
              : 'prefix lookup, cache restore, allocation',
        },
        { label: 'After prefill', value: 'first decode step' },
      ]);
      push('prefill', 'prefill', phases.prefillMs, false, [
        { label: 'Prefill tokens', value: formatNumber(row.prefillTokens) },
        {
          label: 'Prefill speed',
          value: row.prefillTps === null ? PREFILL_UNDER_TEXT : `${formatCount(row.prefillTps)} tok/s`,
        },
      ]);
    } else {
      const newTokens =
        row.promptTokens === null ? null : Math.max(0, row.promptTokens - nonNegative(row.cachedTokens));
      push('first', 'first', phases.firstTokenMs, false, [
        { label: 'New prompt tokens', value: formatNumber(newTokens) },
        { label: 'Prefill', value: 'not measured for this turn' },
      ]);
    }
    if (stop !== 'prefill') {
      push('decode', 'decode', phases.decodeMs, true, [
        { label: 'Output tokens', value: formatNumber(row.outputTokens) },
        { label: 'Decode speed', value: row.decodeTps === null ? '—' : `${formatTps(row.decodeTps)} tok/s` },
      ]);
      push('tail', 'other', phases.tailMs, false);
    }
  }
  const last = spans[spans.length - 1];
  if (last !== undefined && row.outcome !== 'ok') {
    spans[spans.length - 1] = { ...last, status: row.outcome };
  }
  const markers: TimelineMarker[] =
    row.ttftMs !== null && !beforePrefill
      ? [
          {
            id: 'first-token',
            rowId,
            atMs:
              pre +
              phases.waitMs +
              phases.slotMs +
              phases.prepareMs +
              phases.setupMs +
              phases.prefillMs +
              phases.firstTokenMs,
            label: 'First token',
          },
        ]
      : [];
  return { rows, spans, markers, domainMs: [0, Math.max(phases.totalMs, 1)] };
}

/** Moved from the Metrics page, unchanged. */
export function modelShareLabel(turns: number, total: number): string {
  return `${formatCountNoun(turns, 'turn', 'turns')} · ${formatRate(turns, total)}`;
}

/** Share of prompt tokens served from the cache: formatRate(cached, cached + input); 0 prompt tokens is unknown. */
export function cacheReuse(cached: number, input: number): { text: string; fraction: number | null } {
  const total = cached + input;
  return { text: formatRate(cached, total), fraction: total > 0 ? cached / total : null };
}

const BUCKET_WORD: Record<MetricsBucket, string> = { '5m': '5 minutes', hour: 'hour', day: 'day' };

/** `5 minutes`, `hour`, `day`: the word after "per". */
export function bucketWord(bucket: MetricsBucket): string {
  return BUCKET_WORD[bucket];
}

const RANGE_WORDS: Record<RangeKey, { label: string; window: string }> = {
  '1h': { label: 'hour', window: 'the last hour' },
  '24h': { label: '24 hours', window: 'the last 24 hours' },
  '7d': { label: '7 days', window: 'the last 7 days' },
  '30d': { label: '30 days', window: 'the last 30 days' },
};

/** `vs previous 7 days`, `vs previous hour`. */
export function versusText(key: RangeKey): string {
  return `vs previous ${RANGE_WORDS[key].label}`;
}

/** `Previous 7 days`: the ghost line's legend key. */
export function previousLabel(key: RangeKey): string {
  return `Previous ${RANGE_WORDS[key].label}`;
}

export const KPI_LABEL: Record<KpiKey, string> = {
  decode: 'Decode speed',
  ttft: 'First token',
  turns: 'Turns',
  tokens: 'Output tokens',
  failed: 'Failed turns',
};

/** One sentence: what the number is and over which window. */
export function kpiDescription(key: KpiKey, range: RangeKey): string {
  const window = RANGE_WORDS[range].window;
  switch (key) {
    case 'turns':
      return `Turns recorded in ${window}.`;
    case 'tokens':
      return `Tokens generated in ${window}.`;
    case 'decode':
      return `Median decode speed of completed turns in ${window}.`;
    case 'ttft':
      return `Median time to the first token of completed turns in ${window}.`;
    case 'failed':
      return `Turns that ended in an error in ${window}.`;
  }
}

/** The measure of one timeseries row for a KPI: the median for speeds, the bucket total for counts. */
export function kpiPointValue(key: KpiKey, point: TsPoint): number | null {
  switch (key) {
    case 'decode':
      return point.decodeTps.p50;
    case 'ttft':
      return point.ttftMs.p50;
    case 'turns':
      return point.totals.turns;
    case 'tokens':
      return point.totals.outputTokens;
    case 'failed':
      return point.totals.error;
  }
}

/** A bucket with no row: 0 for counts, null (a gap) for medians. */
export function kpiEmpty(key: KpiKey): 0 | null {
  return key === 'decode' || key === 'ttft' ? null : 0;
}

/** One value per bucket of the range, nulls kept (a tile spark or a drill-down trend). */
export function kpiSeries(key: KpiKey, points: readonly TsPoint[], range: AlignedRange): (number | null)[] {
  return pivotMeasures(points, range, { value: (p) => kpiPointValue(key, p) }, kpiEmpty(key)).map(
    (point) => point.values.value ?? null,
  );
}

/** The window of equal length before `range`, bucket for bucket (local days for day buckets). */
export function previousRange(range: AlignedRange): AlignedRange {
  const n = range.starts.length;
  if (range.bucket === 'day') {
    const starts = range.starts.map((start) => {
      const day = new Date(start);
      return new Date(day.getFullYear(), day.getMonth(), day.getDate() - n).getTime();
    });
    return { from: starts[0] ?? range.from, to: range.from, bucket: range.bucket, starts };
  }
  const span = range.to - range.from;
  return {
    from: range.from - span,
    to: range.from,
    bucket: range.bucket,
    starts: range.starts.map((start) => start - span),
  };
}

type ByModel = MetricsSummary['byModel'][number];

/** The KPI's measure per model, with its sample count; rows with no turn or no value are left out. */
export function kpiByModel(
  key: KpiKey,
  byModel: readonly ByModel[],
): Array<{ model: string; value: number; samples?: number }> {
  const rows: Array<{ model: string; value: number; samples?: number }> = [];
  for (const row of byModel) {
    if (row.totals.turns <= 0) continue;
    if (key === 'decode' || key === 'ttft') {
      const quantiles = key === 'decode' ? row.latency.decodeTps : row.latency.ttftMs;
      if (quantiles.p50 === null) continue;
      rows.push({ model: row.model, value: quantiles.p50, samples: quantiles.n });
    } else {
      const value = key === 'turns' ? row.totals.turns : key === 'tokens' ? row.totals.outputTokens : row.totals.error;
      rows.push({ model: row.model, value });
    }
  }
  return rows;
}

export interface KpiTileView {
  key: KpiKey;
  label: string;
  value: string | null;
  unit?: string;
  delta: StatDelta;
  hint: string;
  sampleCount?: number;
}

function measured(text: string): string | null {
  return text === '—' ? null : text;
}

function errorShare(totals: MetricsTotals | undefined): number | null {
  return totals === undefined || totals.turns <= 0 ? null : totals.error / totals.turns;
}

/**
 * The five KPI tiles of a summary. `decodeHint`: 'tails' prints p05 and p95 once there are 20 turns (Metrics),
 * 'median' always prints the sample (Overview).
 */
export function kpiTiles(summary: MetricsSummary, versus: string, decodeHint: 'tails' | 'median'): KpiTileView[] {
  const { totals, latency, previous } = summary;
  const decode = latency.decodeTps;
  const ttft = latency.ttftMs;
  const ttftParts = formatDurationParts(ttft.p50);
  const decodeText =
    decodeHint === 'tails' && decode.n >= LOW_SAMPLE_TAIL_N
      ? `p05 ${formatTps(decode.p05)} · p95 ${formatTps(decode.p95)}`
      : `median of ${formatCountNoun(decode.n, 'turn', 'turns')}`;
  const ttftText =
    ttft.n >= LOW_SAMPLE_TAIL_N
      ? `p95 ${formatDuration(ttft.p95)}`
      : `median of ${formatCountNoun(ttft.n, 'turn', 'turns')}`;
  return [
    {
      key: 'decode',
      label: KPI_LABEL.decode,
      value: measured(formatTps(decode.p50)),
      unit: 'tok/s',
      delta: { current: decode.p50, previous: previous?.latency.decodeTps.p50 ?? null, goodWhen: 'up', versus },
      hint: decodeText,
      sampleCount: decode.n,
    },
    {
      key: 'ttft',
      label: KPI_LABEL.ttft,
      value: measured(ttftParts.value),
      unit: ttftParts.unit === '' ? undefined : ttftParts.unit,
      delta: { current: ttft.p50, previous: previous?.latency.ttftMs.p50 ?? null, goodWhen: 'down', versus },
      hint: ttftText,
      sampleCount: ttft.n,
    },
    {
      key: 'turns',
      label: KPI_LABEL.turns,
      value: formatCount(totals.turns),
      delta: { current: totals.turns, previous: previous?.totals.turns ?? null, goodWhen: 'neutral', versus },
      hint: `${formatNumber(totals.ok)} ok · ${formatNumber(totals.error)} failed · ${formatNumber(totals.cancelled)} cancelled`,
    },
    {
      key: 'tokens',
      label: KPI_LABEL.tokens,
      value: formatCount(totals.outputTokens),
      delta: {
        current: totals.outputTokens,
        previous: previous?.totals.outputTokens ?? null,
        goodWhen: 'neutral',
        versus,
      },
      hint: `${formatCount(totals.inputTokens)} new input · ${formatCount(totals.cachedTokens)} cached`,
    },
    {
      key: 'failed',
      label: KPI_LABEL.failed,
      value: formatCount(totals.error),
      delta: { current: errorShare(totals), previous: errorShare(previous?.totals), goodWhen: 'down', versus },
      hint: `${formatRate(totals.error, totals.turns)} of turns`,
      sampleCount: totals.turns,
    },
  ];
}

const DISPATCH_TEXT: Record<string, string> = {
  start: 'New conversation',
  continue: 'Next message',
  continue_tool: 'After a tool call',
};

export function dispatchText(dispatch: string): string {
  return DISPATCH_TEXT[dispatch] ?? dispatch;
}

const MAX_POSITIONS = 8;

export interface SpeculationView {
  /** Models with at least one MTP turn. */
  models: MetricsSpeculative['models'];
  cycles: number;
  tokensPerCycle: number | null;
  draftsPerCycle: number | null;
  meanDepth: number | null;
  mtpTurns: number;
  turns: number;
  /** Cycle-weighted acceptance per draft position 1…8; null where no model reached the position. */
  rates: (number | null)[];
  samples: number[];
}

function weighted(
  rows: MetricsSpeculative['models'],
  pick: (row: MetricsSpeculative['models'][number]) => number | null,
): number | null {
  let sum = 0;
  let weight = 0;
  for (const row of rows) {
    const value = pick(row);
    if (value === null || row.cycles <= 0) continue;
    sum += value * row.cycles;
    weight += row.cycles;
  }
  return weight > 0 ? sum / weight : null;
}

/** The Speculation tab's numbers: every mean is weighted by cycles over the models that used MTP. */
export function speculationView(all: MetricsSpeculative['models']): SpeculationView {
  const models = all.filter((row) => row.mtpTurns > 0);
  const rates: (number | null)[] = [];
  const samples: number[] = [];
  for (let pos = 1; pos <= MAX_POSITIONS; pos += 1) {
    let sum = 0;
    let cycles = 0;
    for (const row of models) {
      const entry = row.acceptByPos.find((item) => item.pos === pos - 1);
      if (entry === undefined || entry.cycles <= 0) continue;
      sum += entry.rate * entry.cycles;
      cycles += entry.cycles;
    }
    rates.push(cycles > 0 ? sum / cycles : null);
    samples.push(cycles);
  }
  let last = rates.length;
  while (last > 0 && rates[last - 1] === null) last -= 1;
  return {
    models,
    cycles: models.reduce((sum, row) => sum + row.cycles, 0),
    tokensPerCycle: weighted(models, (row) => row.meanAccepted),
    draftsPerCycle: weighted(models, (row) => row.meanDrafts),
    meanDepth: weighted(models, (row) => row.meanDepth),
    mtpTurns: models.reduce((sum, row) => sum + row.mtpTurns, 0),
    turns: models.reduce((sum, row) => sum + row.turns, 0),
    rates: rates.slice(0, last),
    samples: samples.slice(0, last),
  };
}
