/**
 * Gallery sections for the chart kit and the live components, on fixture data. Each body replays from its first
 * frame when its Section remounts it, and moves to a second data set on a timer to show reshape, roll and swap.
 */

import { type ReactElement, useEffect, useState } from 'react';

import {
  AgentTimeline,
  AgentTimelineLoading,
  type AgentHandoff,
  type AgentLane,
  type AgentSpan,
  type AgentTimelineMode,
  type AgentTimelineProps,
  type AgentTokenReport,
} from '../../../ui/src/components/arc-x/agent-timeline/agent-timeline';
import { BarList, type BarListRow } from '../../../ui/src/components/arc-x/bar-list/bar-list';
import { CellGrid } from '../../../ui/src/components/arc-x/cell-grid/cell-grid';
import { ChartCard, type ChartState } from '../../../ui/src/components/arc-x/chart-card/chart-card';
import type { ChartPoint, ChartSeries } from '../../../ui/src/components/arc-x/chart/types';
import { DraftSweep } from '../../../ui/src/components/arc-x/draft-sweep/draft-sweep';
import { Histogram, type HistogramBin } from '../../../ui/src/components/arc-x/histogram/histogram';
import { KpiDrilldown, type KpiDetail } from '../../../ui/src/components/arc-x/kpi-drilldown/kpi-drilldown';
import { LiveStream, type LiveStreamPoint } from '../../../ui/src/components/arc-x/live-stream/live-stream';
import { LiveTurnList, LiveTurnRow } from '../../../ui/src/components/arc-x/live-turn-row/live-turn-row';
import { MetricTicker, type TickerItem } from '../../../ui/src/components/arc-x/metric-ticker/metric-ticker';
import { MiniSpark } from '../../../ui/src/components/arc-x/mini-spark/mini-spark';
import { ModelBadge } from '../../../ui/src/components/arc-x/model-badge/model-badge';
import { QuantileBar } from '../../../ui/src/components/arc-x/quantile-bar/quantile-bar';
import { ScatterChart, type ScatterPoint } from '../../../ui/src/components/arc-x/scatter-chart/scatter-chart';
import { SpanTimeline, type TimelineSpan } from '../../../ui/src/components/arc-x/span-timeline/span-timeline';
import { Meter, SplitBar } from '../../../ui/src/components/arc-x/split-bar/split-bar';
import { StackedBarChart } from '../../../ui/src/components/arc-x/stacked-bar-chart/stacked-bar-chart';
import { StatGroup, StatTile } from '../../../ui/src/components/arc-x/stat-tile/stat-tile';
import { TrendChart } from '../../../ui/src/components/arc-x/trend-chart/trend-chart';
import { timelineAxis } from '../../../ui/src/lib/agent-timeline';
import { formatClock, formatCount, formatDuration, formatNumber } from '../../../ui/src/lib/format';
import { TOKEN_SERIES } from '../../../ui/src/lib/series';
import { TimelineLoadError, TimelineNoTimes } from '../../../ui/src/pages/session-detail/turn-timeline';

const MODELS = ['qwen3.6-27b', 'gemma-4-e4b', 'lfm2.5-8b'];
/** Fixed slots: the shared registry in this origin's storage may already hold other names. */
const SERIES: ChartSeries[] = MODELS.map((name, index) => ({
  key: name,
  label: name,
  color: `var(--model-${String(index + 1)})`,
}));

/** 0 for the first `everyMs`, then 1, 2 … up to `max`. */
function useStep(everyMs: number, max: number): number {
  const [step, setStep] = useState(0);
  useEffect(() => {
    if (step >= max) return undefined;
    const timer = setTimeout(() => setStep((value) => value + 1), everyMs);
    return () => clearTimeout(timer);
  }, [step, everyMs, max]);
  return step;
}

/** A clock that ticks every `everyMs` while mounted. */
function useTick(everyMs: number): number {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const timer = setInterval(() => setTick((value) => value + 1), everyMs);
    return () => clearInterval(timer);
  }, [everyMs]);
  return tick;
}

function wave(i: number, seed: number, base: number, swing: number): number {
  return Math.round((base + swing * Math.sin((i + seed) / 2.6) + swing * 0.4 * Math.cos((i * seed) / 3.1)) * 10) / 10;
}

const HOUR = 3_600_000;
const START = Date.UTC(2026, 10, 10, 0, 0, 0);

function hourly(seed: number): ChartPoint[] {
  return Array.from({ length: 24 }, (_, i) => {
    const t = START + i * HOUR;
    const gap = seed === 1 && (i === 6 || i === 7);
    return {
      key: String(t),
      label: `Nov 10, ${String(i).padStart(2, '0')}:00${i === 23 ? ', so far' : ''}`,
      axisLabel: `${String(i).padStart(2, '0')}:00`,
      values: {
        [MODELS[0]]: gap ? null : wave(i, seed, 42, 6),
        [MODELS[1]]: wave(i, seed + 3, 78, 9),
        [MODELS[2]]: i < 4 ? null : wave(i, seed + 7, 120, 14),
      },
      samples: { [MODELS[0]]: i === 12 ? 3 : 40, [MODELS[1]]: 30, [MODELS[2]]: 25 },
    };
  });
}

function singleWithBand(seed: number): ChartPoint[] {
  return hourly(seed).map((point) => {
    const p50 = point.values[MODELS[1]] ?? null;
    return {
      ...point,
      values: { current: p50, low: p50 === null ? null : p50 - 12, high: p50 === null ? null : p50 + 15 },
      samples: { current: 30 },
    };
  });
}

export function TrendSection(): ReactElement {
  const step = useStep(2600, 1);
  const data = hourly(step === 0 ? 1 : 4);
  const single = singleWithBand(step === 0 ? 2 : 5);
  return (
    <div className="gallery-grid">
      <TrendChart label="Decode speed by model" unit="tok/s" data={data} series={SERIES} curve="monotone" />
      <TrendChart
        label="Decode speed with band"
        unit="tok/s"
        data={single}
        series={[{ key: 'current', label: 'Decode speed', color: 'var(--series-1)' }]}
        band={{ lowKey: 'low', highKey: 'high' }}
        ghost={{ label: 'Previous 24 hours', values: singleWithBand(9).map((point) => point.values.current ?? null) }}
        curve="linear"
        selectedKey={single[15].key}
      />
    </div>
  );
}

function tokenColumns(seed: number): ChartPoint[] {
  return Array.from({ length: 14 }, (_, i) => ({
    key: `d${String(i)}`,
    label: `Nov ${String(i + 1)}`,
    axisLabel: `Nov ${String(i + 1)}`,
    values:
      i === 3
        ? { cached: null, input: null, answer: null, reasoning: null }
        : {
            cached: Math.round(wave(i, seed, 40, 20) * 1000),
            input: Math.round(wave(i, seed + 2, 30, 10) * 1000),
            answer: Math.round(wave(i, seed + 4, 18, 6) * 1000),
            reasoning: Math.round(wave(i, seed + 6, 8, 4) * 1000),
          },
  }));
}

export function StackedSection(): ReactElement {
  const step = useStep(2600, 1);
  const [selected, setSelected] = useState<string | null>(null);
  const data = tokenColumns(step === 0 ? 1 : 3);
  return (
    <div className="gallery-grid">
      <StackedBarChart
        label="Tokens per day"
        unit="tokens"
        data={data}
        series={TOKEN_SERIES}
        selectedKey={selected}
        onSelect={(point) => setSelected(point.key)}
      />
      <StackedBarChart
        label="Turns per day by model"
        unit="turns"
        mode="grouped"
        data={hourly(step === 0 ? 2 : 6).slice(0, 10)}
        series={SERIES}
      />
    </div>
  );
}

function bins(seed: number): HistogramBin[] {
  const edges = [0, 100, 200, 300, 400, 600, 800, 1200, 1600, 2400, 3200];
  return edges.map((from, i) => ({
    from,
    to: edges[i + 1] ?? from * 2,
    count: Math.max(0, Math.round(80 * Math.exp(-((i - 3 - seed * 0.4) ** 2) / 5))),
  }));
}

export function HistogramSection(): ReactElement {
  const step = useStep(2600, 1);
  return (
    <Histogram
      label="Time to first token"
      unit="ms"
      countLabel={{ one: 'turn', other: 'turns' }}
      bins={bins(step)}
      markers={[
        { value: step === 0 ? 380 : 450, label: 'p50' },
        { value: step === 0 ? 1450 : 1700, label: 'p95' },
      ]}
    />
  );
}

function barRows(seed: number): BarListRow[] {
  const names = [
    ...MODELS,
    'nemotron-3.5',
    'qwen3-8b',
    'muse-glimmer-30b',
    'deepseek-v4.1-flash',
    'qwen3.6-35b-a3b',
    'gemma-4-31b',
  ];
  return names.map((name, i) => ({
    key: name,
    label: <ModelBadge name={name} size="sm" />,
    text: name,
    value: i === 5 ? null : Math.round(wave(i, seed, 60, 30)),
    samples: i === 2 ? 3 : 40,
    color: `var(--model-${String((i % 8) + 1)})`,
    pattern: i >= 8 ? 'alt' : 'solid',
    hint: `${formatCount(40 + i * 17)} turns`,
  }));
}

export function BarListSection(): ReactElement {
  const step = useStep(2600, 1);
  return <BarList label="Decode speed by model" unit="tok/s" rows={barRows(step === 0 ? 1 : 5)} limit={6} />;
}

const CARD_STATES: readonly ChartState[] = ['loading', 'ready', 'held', 'ready'];

export function ChartCardSection(): ReactElement {
  const step = useStep(1500, CARD_STATES.length - 1);
  const state = CARD_STATES[step];
  return (
    <div className="gallery-grid">
      <ChartCard
        title="Decode speed"
        subtitle="Median per hour · tok/s"
        state={state}
        height={180}
        footnote="412 turns · 3 models"
      >
        <TrendChart label="Decode speed" unit="tok/s" data={hourly(step < 2 ? 1 : 4)} series={SERIES} height={180} />
      </ChartCard>
      <ChartCard
        title="Failures"
        subtitle="Turns per hour · turns"
        state={step === 0 ? 'loading' : 'error'}
        height={180}
        error={{ message: 'Failed to load the timeseries', retry: () => undefined }}
      >
        {null}
      </ChartCard>
      <ChartCard
        title="Speculation"
        subtitle="Drafts kept per cycle"
        state={step === 0 ? 'loading' : 'empty'}
        height={180}
        empty={{ title: 'No MTP turns in this range', hint: 'Turns with speculative decoding show up here.' }}
      >
        {null}
      </ChartCard>
    </div>
  );
}

const SPARK = Array.from({ length: 48 }, (_, i) => (i === 20 || i === 21 ? null : wave(i, 2, 40, 8)));

export function StatSection(): ReactElement {
  const step = useStep(2600, 2);
  const [selected, setSelected] = useState('decode');
  return (
    <div className="gallery-body">
      <StatGroup label="Key numbers">
        <StatTile
          label="Decode speed"
          value={step === 0 ? '42.7' : step === 1 ? '44.1' : '43.9'}
          unit="tok/s"
          delta={{ current: step === 0 ? 42.7 : 44.1, previous: 41, goodWhen: 'up', versus: 'vs previous 24 hours' }}
          spark={{ data: SPARK, label: 'Decode speed, last 24 hours' }}
          onSelect={() => setSelected('decode')}
          selected={selected === 'decode'}
        />
        <StatTile
          label="First token"
          value={step === 0 ? '412' : '398'}
          unit="ms"
          delta={{ current: step === 0 ? 412 : 398, previous: 380, goodWhen: 'down', versus: 'vs previous 24 hours' }}
          onSelect={() => setSelected('ttft')}
          selected={selected === 'ttft'}
        />
        <StatTile
          label="Turns"
          value={step === 0 ? '2,904' : '2,951'}
          hint="31 failed"
          onSelect={() => setSelected('turns')}
          selected={selected === 'turns'}
        />
        <StatTile
          label="Tokens"
          value={step === 0 ? '8.4M' : '8.5M'}
          sampleCount={3}
          onSelect={() => setSelected('tokens')}
          selected={selected === 'tokens'}
        />
      </StatGroup>
      <StatGroup label="Storage">
        <StatTile label="Models on disk" value={null} loading={step === 0} />
        <StatTile
          label="Cold cache"
          value="21.6"
          unit="GB"
          footer={<Meter label="Cold cache quota" value={21.6} max={32} />}
        />
        <StatTile label="Downloads" value={null} error="Not reachable" />
      </StatGroup>
      <div className="gallery-row">
        <MiniSpark data={SPARK} label="Decode speed, last 24 hours" />
        <MiniSpark
          data={SPARK.map((value) => (value === null ? null : value * (step === 0 ? 1 : 1.2)))}
          label="Spark"
          width={120}
          height={32}
        />
      </div>
    </div>
  );
}

export function SplitSection(): ReactElement {
  const step = useStep(2600, 1);
  const used = step === 0 ? 61 : 74;
  return (
    <div className="gallery-body">
      <SplitBar
        label="Memory"
        unit="GB"
        size="tall"
        total={128}
        segments={[
          { key: 'active', label: 'MLX active', value: used * 0.7, color: 'var(--model-1)' },
          { key: 'cache', label: 'MLX cache', value: used * 0.3, color: 'var(--model-2)' },
          { key: 'other', label: 'Other apps', value: 12, color: 'var(--model-other)', pattern: 'hatch' },
        ]}
        threshold={{ value: 96, label: 'Wired limit' }}
      />
      <SplitBar
        label="Disk"
        unit="GB"
        total={null}
        segments={[
          { key: 'models', label: 'Models', value: 412, color: 'var(--model-3)' },
          { key: 'cache', label: 'Cache', value: 88, color: 'var(--model-4)' },
        ]}
      />
      <div className="gallery-grid">
        <Meter label="Context" value={step === 0 ? 0.42 : 0.86} max={1} />
        <Meter label="Prefill" value={step === 0 ? 1200 : 9800} max={31000} size="dense" busy warnAt={2} dangerAt={2} />
        <Meter label="Quota" value={0.999} max={1} />
      </div>
    </div>
  );
}

export function TimelineSection(): ReactElement {
  const spans: TimelineSpan[] = [
    { id: 'q1', rowId: 't1', kind: 'queue', startMs: 0, endMs: 38, label: 'Queue' },
    { id: 'l1', rowId: 't1', kind: 'load', startMs: 38, endMs: 240, label: 'Loading the model' },
    {
      id: 'p1',
      rowId: 't1',
      kind: 'prefill',
      startMs: 240,
      endMs: 650,
      label: 'Prefill',
      estimated: true,
      detail: [{ label: 'Tokens', value: '12,400' }],
    },
    { id: 'd1', rowId: 't1', kind: 'decode', startMs: 650, endMs: 1300, label: 'Decode' },
    { id: 'q2', rowId: 't2', kind: 'queue', startMs: 200, endMs: 380, label: 'Queue' },
    { id: 'p2', rowId: 't2', kind: 'prefill', startMs: 380, endMs: 900, label: 'Prefill', status: 'error' },
    { id: 'q3', rowId: 't3', kind: 'wait', startMs: 500, endMs: 700, label: 'Waiting for a model load' },
    { id: 'd3', rowId: 't3', kind: 'decode', startMs: 700, endMs: 1600, label: 'Decode' },
  ];
  const [selected, setSelected] = useState<string | null>(null);
  return (
    <div className="gallery-body">
      <SpanTimeline
        label="Session timeline"
        rows={[
          { id: 't1', label: 'Turn 1', sublabel: 'qwen3.6-27b · 1.2K out', color: SERIES[0].color },
          { id: 't2', label: 'Turn 2', sublabel: 'gemma-4-e4b', color: SERIES[1].color },
          { id: 't3', label: 'Turn 3', sublabel: 'lfm2.5-8b', color: SERIES[2].color },
        ]}
        spans={spans}
        markers={[{ id: 'ft', rowId: 't1', atMs: 680, label: 'First token' }]}
        domainMs={[0, 2000]}
        selectedId={selected}
        onSelect={(span) => setSelected(span?.id ?? null)}
      />
      <SpanTimeline label="One turn" rows={[{ id: 't1', label: 'Turn' }]} spans={spans.slice(0, 4)} compact />
    </div>
  );
}

type SpanSeed = Omit<AgentSpan, 'id' | 'laneId'>;

function laneSpans(laneId: string, seeds: readonly SpanSeed[]): AgentSpan[] {
  return seeds.map((seed, i) => ({ ...seed, id: `${laneId}-${String(i)}`, laneId }));
}

const FIRST_TOKEN: SpanSeed = {
  kind: 'wait',
  startMs: 0,
  endMs: 0,
  status: 'ok',
  label: 'Waiting for the first token',
};
const MODEL_WAIT: SpanSeed = { kind: 'wait', startMs: 0, endMs: 0, status: 'ok', label: 'Waiting for the model' };

function at(seed: SpanSeed, startMs: number, endMs: number | null, extra: Partial<SpanSeed> = {}): SpanSeed {
  return { ...seed, startMs, endMs, ...extra };
}

function generate(startMs: number, endMs: number | null, answer = false, extra: Partial<SpanSeed> = {}): SpanSeed {
  return {
    kind: 'generate',
    startMs,
    endMs,
    status: 'ok',
    label: answer ? 'Writing the answer' : 'Reasoning',
    ...extra,
  };
}

function toolCall(startMs: number, endMs: number | null, name: string, extra: Partial<SpanSeed> = {}): SpanSeed {
  return { kind: 'tool', startMs, endMs, status: 'ok', label: `Tool call · ${name}`, ...extra };
}

/** Alternating model waits, reasoning and tool calls from `from` to `to`, on a fixed pattern. */
export function workSpans(from: number, to: number, stepMs: number, seed: number): SpanSeed[] {
  const tools = ['read_file', 'grep', 'run_terminal_command', 'edit_file'];
  const out: SpanSeed[] = [at(FIRST_TOKEN, from, from + stepMs * 0.2)];
  let t = from + stepMs * 0.2;
  let i = seed;
  while (t + stepMs < to) {
    const think = stepMs * (0.25 + ((i * 7) % 5) * 0.05);
    const call = Math.min(stepMs * (0.3 + ((i * 3) % 4) * 0.12), stepMs * 0.9 - think);
    out.push(generate(t, t + think));
    out.push(toolCall(t + think, t + think + call, tools[i % tools.length]!));
    out.push(at(MODEL_WAIT, t + think + call, t + stepMs));
    t += stepMs;
    i += 1;
  }
  out.push(generate(t, to, true));
  return out;
}

const AGENT_LANES: AgentLane[] = [
  { id: 'main', name: 'Main agent', model: MODELS[0], extra: '+4 subagents', color: 'var(--lane-main)' },
  { id: 'a', name: 'Map the scheduler queues', model: MODELS[0], color: 'var(--lane-1)', to: '/sessions/gallery-a' },
  {
    id: 'b',
    name: 'Bisect the cache regression',
    model: MODELS[1],
    color: 'var(--lane-2)',
    outcome: { text: 'Failed', tone: 'danger' },
  },
  { id: 'c', name: 'Check the docs', color: 'var(--lane-3)' },
  { id: 'd', name: 'Write the planner tests', model: MODELS[2], color: 'var(--lane-4)' },
];

const AGENT_SPANS: AgentSpan[] = [
  ...laneSpans('main', [
    at(FIRST_TOKEN, 0, 800),
    generate(800, 2400, false, { retry: 'Retry 1 of 3' }),
    toolCall(2400, 3600, 'read_file'),
    at(MODEL_WAIT, 3600, 4200),
    toolCall(4200, 5000, 'run_terminal_command', { status: 'failed' }),
    at(MODEL_WAIT, 5000, 5600),
    {
      kind: 'wait',
      startMs: 5600,
      endMs: 22_000,
      status: 'ok',
      label: 'Waiting for subagents',
      facts: ['4 spawn calls', '1 failed'],
      ownsHandoffs: true,
    },
    at(MODEL_WAIT, 22_000, 22_800),
    generate(22_800, 26_000, true),
  ]),
  ...laneSpans('a', workSpans(5_700, 15_000, 2_400, 1)),
  ...laneSpans('b', [
    ...workSpans(5_750, 17_000, 2_800, 2),
    toolCall(17_000, 21_500, 'run_terminal_command', { status: 'failed' }),
  ]),
  ...laneSpans('c', [
    {
      kind: 'run',
      startMs: 5_800,
      endMs: 12_000,
      status: 'ok',
      label: 'Ran, no step record',
      note: "No step record: this subagent's session folder is gone.",
    },
  ]),
  ...laneSpans('d', workSpans(5_850, 19_000, 3_100, 3)),
];

const AGENT_HANDOFFS: AgentHandoff[] = [
  { id: 'a-s', laneId: 'a', kind: 'spawn', atMs: 5_650 },
  { id: 'b-s', laneId: 'b', kind: 'spawn', atMs: 5_700 },
  { id: 'c-s', laneId: 'c', kind: 'spawn', atMs: 5_750 },
  { id: 'd-s', laneId: 'd', kind: 'spawn', atMs: 5_800 },
  { id: 'c-r', laneId: 'c', kind: 'return', atMs: 12_000 },
  { id: 'a-r', laneId: 'a', kind: 'return', atMs: 15_100 },
  { id: 'd-r', laneId: 'd', kind: 'return', atMs: 19_200 },
  { id: 'b-r', laneId: 'b', kind: 'return', atMs: 21_800, failed: true },
];

const AGENT_REPORTS: AgentTokenReport[] = [
  { laneId: 'a', atMs: 15_000, tokens: 6_200 },
  { laneId: 'd', atMs: 19_000, tokens: 4_300 },
  { laneId: 'b', atMs: 21_500, tokens: 9_100 },
  { laneId: 'main', atMs: 26_000, tokens: 18_400 },
];

const FINISHED: AgentTimelineProps = {
  label: 'Turn 6 timeline',
  title: 'Find why the scheduler drops the second queue',
  lanes: AGENT_LANES,
  spans: AGENT_SPANS,
  handoffs: AGENT_HANDOFFS,
  reports: AGENT_REPORTS,
  durationMs: 26_000,
  mode: { kind: 'recorded' },
  outcome: { word: 'Finished', tone: 'success' },
  formatDuration,
  axisUnit: timelineAxis,
};

const LONG_LANES: AgentLane[] = [
  { id: 'main', name: 'Main agent', model: MODELS[0], extra: '+2 subagents', color: 'var(--lane-main)' },
  { id: 'a', name: 'Survey the crates', model: MODELS[0], color: 'var(--lane-1)' },
  { id: 'b', name: 'Run the long benchmark', model: MODELS[2], color: 'var(--lane-2)' },
];

const LONG: AgentTimelineProps = {
  ...FINISHED,
  label: 'Turn 30 timeline',
  title: 'Benchmark every crate and write the report',
  lanes: LONG_LANES,
  spans: [
    ...laneSpans('main', [
      ...workSpans(0, 200_000, 24_000, 1),
      {
        kind: 'wait',
        startMs: 200_000,
        endMs: 640_000,
        status: 'ok',
        label: 'Waiting for subagents',
        facts: ['2 spawn calls'],
        ownsHandoffs: true,
      },
      at(MODEL_WAIT, 640_000, 646_000),
      generate(646_000, 720_000, true),
    ]),
    ...laneSpans('a', workSpans(201_000, 420_000, 31_000, 2)),
    ...laneSpans('b', workSpans(202_000, 636_000, 52_000, 3)),
  ],
  handoffs: [
    { id: 'a-s', laneId: 'a', kind: 'spawn', atMs: 200_500 },
    { id: 'b-s', laneId: 'b', kind: 'spawn', atMs: 201_500 },
    { id: 'a-r', laneId: 'a', kind: 'return', atMs: 421_000 },
    { id: 'b-r', laneId: 'b', kind: 'return', atMs: 638_000 },
  ],
  reports: [
    { laneId: 'a', atMs: 420_000, tokens: 21_800 },
    { laneId: 'b', atMs: 636_000, tokens: 40_100 },
    { laneId: 'main', atMs: 720_000, tokens: 52_600 },
  ],
  durationMs: 720_000,
};

const LIVE_LANES: AgentLane[] = [
  { id: 'main', name: 'Main agent', model: MODELS[0], extra: '+2 subagents', color: 'var(--lane-main)' },
  { id: 'a', name: 'Run the upload tests', model: MODELS[1], color: 'var(--lane-1)' },
  { id: 'b', name: 'Read the upload client', model: MODELS[0], color: 'var(--lane-2)' },
];

const LIVE_SPANS: AgentSpan[] = [
  ...laneSpans('main', [
    at(FIRST_TOKEN, 0, 1_000),
    generate(1_000, 3_000),
    toolCall(3_000, 4_000, 'read_file'),
    at(MODEL_WAIT, 4_000, 5_000),
    {
      kind: 'wait',
      startMs: 5_000,
      endMs: null,
      status: 'running',
      label: 'Waiting for subagents',
      facts: ['2 spawn calls'],
      ownsHandoffs: true,
    },
  ]),
  ...laneSpans('a', [
    at(FIRST_TOKEN, 5_200, 6_000),
    generate(6_000, 9_000),
    toolCall(9_000, null, 'run_terminal_command', { status: 'running' }),
  ]),
  ...laneSpans('b', workSpans(5_300, 14_000, 2_900, 2)),
];

const ONE_LANE: AgentTimelineProps = {
  ...FINISHED,
  label: 'Turn 7 timeline',
  title: 'Rename the queue type',
  lanes: [{ id: 'main', name: 'Main agent', model: MODELS[1], color: 'var(--lane-main)' }],
  spans: laneSpans('main', [
    at(FIRST_TOKEN, 0, 1_000),
    generate(1_000, 2_400),
    {
      kind: 'tool',
      startMs: 3_500,
      endMs: 4_200,
      status: 'ok',
      label: '3 tool calls · read_file, grep +1',
      facts: ['3 calls'],
    },
    at(MODEL_WAIT, 4_200, 5_300),
    generate(5_300, 7_900, true),
  ]),
  handoffs: [],
  reports: [{ laneId: 'main', atMs: 7_900, tokens: 3_400 }],
  durationMs: 10_200,
};

const HONEST: AgentTimelineProps = {
  ...FINISHED,
  label: 'Turn 12 timeline',
  title: 'Retry the flaky upload test',
  lanes: [
    { id: 'main', name: 'Main agent', model: MODELS[0], extra: '+10 subagents', color: 'var(--lane-main)' },
    { id: 'a', name: 'Collect the benchmark logs', color: 'var(--lane-1)' },
    { id: 'b', name: 'Trace the loader', model: MODELS[2], color: 'var(--lane-2)' },
    { id: 'f', name: '2 more subagents', extra: '+2 subagents', color: 'var(--lane-main)' },
  ],
  spans: [
    ...laneSpans('main', [
      at(FIRST_TOKEN, 0, 900),
      generate(900, 1_600),
      toolCall(1_600, 3_000, 'task', { open: 'end', note: 'End not recorded. Drawn to the next recorded event.' }),
      {
        kind: 'wait',
        startMs: 3_000,
        endMs: 9_500,
        status: 'ok',
        label: 'Waiting for subagents',
        facts: ['3 spawn calls'],
        ownsHandoffs: true,
      },
      toolCall(9_800, 12_400, 'run_terminal_command', {
        status: 'unfinished',
        open: 'end',
        note: 'End not recorded. Drawn to the next recorded event.',
      }),
    ]),
    ...laneSpans('a', [
      {
        kind: 'run',
        startMs: 3_100,
        endMs: 7_000,
        status: 'ok',
        label: 'Ran, no step record',
        note: 'No step record: the subagent wrote no events.',
      },
    ]),
    ...laneSpans('b', [
      at(MODEL_WAIT, 3_200, 4_400, {
        open: 'start',
        note: 'Start not recorded. Drawn from the nearest recorded event.',
      }),
      generate(4_400, 6_100),
      toolCall(6_100, 8_900, 'grep'),
    ]),
    ...laneSpans('f', [
      {
        kind: 'run',
        startMs: 3_300,
        endMs: 8_000,
        status: 'ok',
        label: 'Ran, no step record',
        note: 'Not drawn in detail: only the first 8 subagents get a lane.',
      },
      {
        kind: 'run',
        startMs: 3_400,
        endMs: 9_200,
        status: 'ok',
        label: 'Ran, no step record',
        note: 'Not drawn in detail: only the first 8 subagents get a lane.',
      },
    ]),
  ],
  handoffs: [
    { id: 'a-s', laneId: 'a', kind: 'spawn', atMs: 3_100 },
    { id: 'b-s', laneId: 'b', kind: 'spawn', atMs: 3_150 },
    { id: 'a-r', laneId: 'a', kind: 'return', atMs: 7_000 },
    { id: 'b-r', laneId: 'b', kind: 'return', atMs: 9_000 },
  ],
  reports: [{ laneId: 'b', atMs: 8_900, tokens: 2_100 }],
  durationMs: 12_400,
  outcome: { word: 'Not finished', tone: 'neutral', detail: 'last event 14:32:10' },
};

const WOKEN_START = 20 * 60_000;
const WOKEN_END = WOKEN_START + 252_000;

const WOKEN: AgentTimelineProps = {
  ...FINISHED,
  label: 'Turn 27 timeline',
  title: 'Woken by workflow deep-research',
  lanes: [
    { id: 'main', name: 'Main agent', model: MODELS[0], extra: '+1 subagent', color: 'var(--lane-main)' },
    { id: 'a', name: 'Survey the prior art', model: MODELS[0], extra: 'from an earlier turn', color: 'var(--lane-1)' },
    { id: 'b', name: 'Verify the sources', model: MODELS[2], color: 'var(--lane-2)' },
    { id: 'c', name: 'Write the report', model: MODELS[1], color: 'var(--lane-3)' },
  ],
  spans: [
    ...laneSpans('main', [
      {
        kind: 'idle',
        startMs: 0,
        endMs: WOKEN_START,
        status: 'ok',
        label: 'Idle between turns',
        note: 'No turn was open. Handoffs here are background work, not steps of this agent. The turn starts where this line ends.',
        ownsHandoffs: true,
      },
      ...workSpans(WOKEN_START, WOKEN_START + 50_000, 20_000, 4),
      {
        kind: 'wait',
        startMs: WOKEN_START + 50_000,
        endMs: WOKEN_START + 200_000,
        status: 'ok',
        label: 'Waiting for subagents',
        facts: ['1 spawn call'],
        ownsHandoffs: true,
      },
      at(MODEL_WAIT, WOKEN_START + 200_000, WOKEN_START + 204_000),
      generate(WOKEN_START + 204_000, WOKEN_END, true),
    ]),
    ...laneSpans(
      'a',
      workSpans(0, 300_000, 60_000, 1).map((seed, i) =>
        i === 0
          ? { ...seed, cut: 'start' as const, note: 'Began before this turn; drawn from the start of the axis.' }
          : seed,
      ),
    ),
    ...laneSpans('b', workSpans(311_000, 820_000, 90_000, 2)),
    ...laneSpans('c', workSpans(WOKEN_START + 51_000, WOKEN_START + 198_000, 30_000, 3)),
  ],
  handoffs: [
    { id: 'a-r', laneId: 'a', kind: 'return', atMs: 301_000 },
    { id: 'b-s', laneId: 'b', kind: 'spawn', atMs: 310_000 },
    { id: 'b-r', laneId: 'b', kind: 'return', atMs: 822_000 },
    { id: 'c-s', laneId: 'c', kind: 'spawn', atMs: WOKEN_START + 50_500 },
    { id: 'c-r', laneId: 'c', kind: 'return', atMs: WOKEN_START + 199_000 },
  ],
  reports: [
    { laneId: 'a', atMs: 299_000, tokens: 122_400 },
    { laneId: 'b', atMs: 820_000, tokens: 345_700 },
    { laneId: 'c', atMs: WOKEN_START + 198_000, tokens: 39_200 },
    { laneId: 'main', atMs: WOKEN_END, tokens: 1_502_100 },
  ],
  durationMs: WOKEN_END,
  turnStartMs: WOKEN_START,
};

const LIVE_FOR_MS = 10_000;

function LiveAgentTimeline(): ReactElement {
  const [mountedAt] = useState(() => Date.now());
  const [stale, setStale] = useState(false);
  useEffect(() => {
    const timer = setTimeout(() => setStale(true), LIVE_FOR_MS);
    return () => clearTimeout(timer);
  }, []);
  const mode: AgentTimelineMode = stale
    ? { kind: 'stale', atMs: 20_000 + LIVE_FOR_MS, asOfText: formatClock(mountedAt + LIVE_FOR_MS, true) }
    : { kind: 'live', originMs: mountedAt - 20_000 };
  return (
    <AgentTimeline
      {...FINISHED}
      label="Turn 9 timeline"
      title="Fix the upload retry"
      lanes={LIVE_LANES}
      spans={LIVE_SPANS}
      handoffs={[
        { id: 'a-s', laneId: 'a', kind: 'spawn', atMs: 5_100 },
        { id: 'b-s', laneId: 'b', kind: 'spawn', atMs: 5_200 },
        { id: 'b-r', laneId: 'b', kind: 'return', atMs: 14_100 },
      ]}
      reports={[{ laneId: 'b', atMs: 14_000, tokens: 5_600 }]}
      durationMs={19_000}
      mode={mode}
      outcome={{ word: 'In progress', tone: 'neutral' }}
    />
  );
}

export function AgentTimelineSection({
  variant,
}: {
  variant: 'finished' | 'long' | 'live' | 'one-lane' | 'honest' | 'woken' | 'narrow' | 'states';
}): ReactElement {
  switch (variant) {
    case 'finished':
      return <AgentTimeline {...FINISHED} />;
    case 'woken':
      return <AgentTimeline {...WOKEN} />;
    case 'long':
      return <AgentTimeline {...LONG} />;
    case 'live':
      return <LiveAgentTimeline />;
    case 'one-lane':
      return <AgentTimeline {...ONE_LANE} />;
    case 'honest':
      return <AgentTimeline {...HONEST} />;
    case 'narrow':
      return (
        <div style={{ width: 360, maxWidth: '100%' }}>
          <AgentTimeline {...FINISHED} />
        </div>
      );
    case 'states':
      return (
        <div className="gallery-body">
          <AgentTimelineLoading label="Turn 3 timeline" title="Map the scheduler queues" lanes={3} />
          <TimelineLoadError message="The worker did not answer in time." onRetry={() => undefined} />
          <TimelineNoTimes label="Turn 20 timeline" title="Summarise the import log" />
        </div>
      );
  }
}

function scatterPoints(seed: number): ScatterPoint[] {
  return Array.from({ length: 160 }, (_, i) => {
    const model = MODELS[i % 3];
    const x = Math.round(((i * 7919 + seed * 31) % 16000) + 200);
    return {
      key: `trace-${String(seed > 2 ? i + 500 : i)}`,
      x,
      y: Math.round(x * (0.04 + (i % 3) * 0.03) + ((i * 37) % 120)),
      series: model,
      hollow: i % 11 === 0,
      detail: [{ label: 'Turn', value: `#${String(i)}` }],
    };
  });
}

export function ScatterSection(): ReactElement {
  const step = useStep(2600, 2);
  return (
    <ScatterChart
      label="Prefill time against new prompt tokens"
      xLabel="New prompt tokens"
      xUnit="tokens"
      yUnit="ms"
      points={scatterPoints(step === 0 ? 1 : step === 1 ? 2 : 3)}
      series={SERIES}
      hollowLabel="Cache restore"
      formatX={(value) => formatNumber(value)}
    />
  );
}

export function QuantileSection(): ReactElement {
  const step = useStep(2600, 1);
  return (
    <div className="gallery-row">
      <QuantileBar
        label="Decode speed, qwen3.6-27b"
        quantiles={{ n: 412, p05: 30, p50: step === 0 ? 42 : 47, p95: 56 }}
        min={0}
        max={140}
        unit="tok/s"
      />
      <QuantileBar
        label="Decode speed, gemma-4-e4b"
        quantiles={{ n: 12, p05: 60, p50: 78, p95: 94 }}
        min={0}
        max={140}
        unit="tok/s"
      />
      <QuantileBar
        label="Decode speed, lfm2.5-8b"
        quantiles={{ n: 3, p05: 100, p50: 118, p95: 130 }}
        min={0}
        max={140}
        unit="tok/s"
      />
    </div>
  );
}

const LIVE_START = 1_000_000;

function livePoints(seconds: number): LiveStreamPoint[] {
  const out: LiveStreamPoint[] = [];
  for (let s = 0; s <= seconds; s += 1) {
    if (s > 100 && s < 130) continue;
    out.push({ t: LIVE_START + s * 1000, value: s > 200 && s < 215 ? 0 : wave(s, 3, 42, 9) });
  }
  return out;
}

export function LiveStreamSection(): ReactElement {
  const tick = useTick(1000);
  const seconds = 400 + tick;
  return (
    <div className="gallery-grid">
      <LiveStream
        label="Output rate"
        unit="tok/s"
        points={livePoints(seconds)}
        nowMs={LIVE_START + seconds * 1000}
        live
        status="Idle"
      />
      <LiveStream
        label="Stopped stream"
        unit="tok/s"
        points={[]}
        nowMs={LIVE_START}
        live={false}
        status="Not running"
      />
    </div>
  );
}

export function CellGridSection(): ReactElement {
  const step = useStep(2200, 3);
  const used = [2400, 3100, 4600, 3900][step];
  return (
    <CellGrid
      label="KV block pool, qwen3.6-27b"
      capacity={8000}
      unit={{ one: 'block', other: 'blocks' }}
      segments={[
        { key: 'used', label: 'In use', value: used, color: 'var(--model-1)' },
        { key: 'reclaim', label: 'Reclaimable', value: 900, color: 'var(--model-1)', pattern: 'hatch' },
        { key: 'pinned', label: 'Pinned', value: 6, color: 'var(--model-4)' },
      ]}
      reserved={{ value: 600, label: 'Reserved' }}
      threshold={{ value: 6400, label: 'Eviction starts' }}
    />
  );
}

export function LiveTurnSection(): ReactElement {
  const tick = useTick(1000);
  const prefillDone = Math.min(31_000, 2_000 + tick * 2_400);
  const decoding = prefillDone >= 31_000;
  return (
    <LiveTurnList label="In flight">
      <LiveTurnRow
        key="a"
        model="qwen3.6-27b"
        alias="claude-haiku-4"
        dispatchLabel="continue"
        phase={decoding ? 'decode' : 'prefill'}
        elapsedMs={3_000 + tick * 1000}
        prefillDone={prefillDone}
        prefillTotal={31_000}
        generatedTokens={decoding ? (tick - 12) * 34 : null}
        promptTokens={31_000}
        ctxWindow={131_072}
      />
      <LiveTurnRow
        key="b"
        model="gemma-4-e4b"
        dispatchLabel="start"
        phase="decode"
        elapsedMs={12_000 + tick * 1000}
        prefillDone={null}
        prefillTotal={null}
        generatedTokens={1204 + tick * 41}
        promptTokens={3_900}
        ctxWindow={32_768}
      />
      {tick % 8 < 5 ? (
        <LiveTurnRow
          key="c"
          model="lfm2.5-8b"
          dispatchLabel="start"
          phase="queued"
          elapsedMs={1_000 + (tick % 8) * 1000}
          prefillDone={null}
          prefillTotal={null}
          generatedTokens={null}
          promptTokens={null}
          ctxWindow={null}
        />
      ) : null}
      <LiveTurnRow
        key="d"
        model="qwen3-8b"
        phase={null}
        elapsedMs={3_600_000 + 7_000 + tick * 1000}
        prefillDone={null}
        prefillTotal={null}
        generatedTokens={null}
        promptTokens={2_000}
        ctxWindow={40_960}
        density="compact"
        live={false}
      />
    </LiveTurnList>
  );
}

const DETAILS: Record<string, KpiDetail> = {
  decode: {
    label: 'Decode speed',
    value: '42.7',
    unit: 'tok/s',
    delta: { current: 42.7, previous: 41, goodWhen: 'up', versus: 'vs previous 24 hours' },
    description: 'Median decode speed of completed turns in the last 24 hours.',
    children: (
      <>
        <TrendChart
          label="Decode speed, this and the previous period"
          unit="tok/s"
          height={180}
          curve="linear"
          data={singleWithBand(2)}
          series={[{ key: 'current', label: 'Decode speed', color: 'var(--series-1)' }]}
          ghost={{ label: 'Previous 24 hours', values: singleWithBand(9).map((point) => point.values.current ?? null) }}
        />
        <BarList label="By model" unit="tok/s" rows={barRows(1)} limit={6} />
      </>
    ),
    footer: <a href="#metrics">Open in Metrics</a>,
  },
  turns: {
    label: 'Turns',
    value: '2,904',
    sampleCount: 3,
    description: 'Turns that finished in the last 24 hours.',
    children: (
      <>
        <StackedBarChart
          label="Tokens per day"
          unit="tokens"
          data={tokenColumns(1)}
          series={TOKEN_SERIES}
          height={180}
        />
        <BarList label="By model" unit="turns" rows={barRows(3)} limit={6} />
      </>
    ),
  },
};

export function DrilldownSection(): ReactElement {
  const [open, setOpen] = useState<string | null>(null);
  return (
    <KpiDrilldown
      label="Last 24 hours"
      open={open}
      onOpenChange={setOpen}
      detail={(key) => DETAILS[key] ?? null}
      className="gallery-stage"
    >
      <StatGroup label="Last 24 hours">
        <StatTile label="Decode speed" value="42.7" unit="tok/s" detailKey="decode" delta={DETAILS.decode.delta} />
        <StatTile label="Turns" value="2,904" detailKey="turns" />
        <StatTile label="First token" value="412" unit="ms" />
        <StatTile label="Tokens" value="8.4M" />
      </StatGroup>
      <TrendChart label="Turns per hour" unit="turns" data={hourly(1)} series={SERIES} height={220} />
    </KpiDrilldown>
  );
}

function tickerItems(tick: number): TickerItem[] {
  const rate = wave(tick, 2, 42, 6);
  return [
    {
      key: 'rate',
      label: 'Output rate',
      value: rate.toFixed(1),
      unit: 'tok/s',
      delta: { text: tick % 3 === 0 ? '+4.2' : '−1.3', direction: tick % 3 === 0 ? 'up' : 'down', goodWhen: 'up' },
      spark: { data: SPARK.slice(tick % 10, (tick % 10) + 38), label: 'Output rate, last 10 minutes' },
    },
    {
      key: 'flight',
      label: 'In flight',
      value: String(1 + (tick % 3)),
      delta: { text: '0', direction: 'flat', goodWhen: 'neutral' },
      facts: [
        { label: 'In flight', value: String(1 + (tick % 3)) },
        { label: 'Max at once', value: '4' },
      ],
    },
    {
      key: 'queued',
      label: 'Queued',
      value: String(tick % 2),
      delta: { text: '+1', direction: 'up', goodWhen: 'down' },
    },
    {
      key: 'mem',
      label: 'MLX memory',
      value: (18.4 + (tick % 4) * 0.1).toFixed(1),
      unit: 'GB',
      delta: { text: '+0.3', direction: 'up', goodWhen: 'neutral' },
      footer: <Meter label="Wired limit" value={18.4} max={96} />,
    },
    { key: 'turns', label: 'Turns since start', value: formatCount(1204 + tick * 3) },
    { key: 'uptime', label: 'Uptime', value: `3h ${String(12 + Math.floor(tick / 6))}m` },
  ];
}

export function TickerSection(): ReactElement {
  const tick = useTick(2000);
  const loading = tick === 0;
  return (
    <div className="gallery-body">
      <MetricTicker label="Now" items={loading ? [] : tickerItems(tick)} loading={loading} />
      <div style={{ maxWidth: 560 }}>
        <MetricTicker label="Now, narrow" items={tickerItems(tick + 1)} />
      </div>
      <MetricTicker label="Now, stale" items={tickerItems(1)} stale="12 s ago" />
    </div>
  );
}

export function SweepSection(): ReactElement {
  const step = useStep(2600, 1);
  return (
    <div className="gallery-row">
      <DraftSweep
        label="Share of drafts kept at each position"
        rates={step === 0 ? [0.92, 0.71, 0.48, 0.22, null] : [0.95, 0.8, 0.52, 0.3, 0.12]}
        samples={[1204, 1204, 980, 410, 3]}
      />
      <DraftSweep
        label="Ten positions"
        rates={[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]}
        color="var(--model-3)"
      />
    </div>
  );
}
