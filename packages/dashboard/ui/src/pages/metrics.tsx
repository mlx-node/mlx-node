import { Card, CardContent } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import {
  AXIS_TICK,
  ChartBody,
  ChartCard,
  ChartEmpty,
  OTHER_SERIES_COLOR,
  TOOLTIP_CONTENT_STYLE,
  buildSeriesColorMap,
  categoryChartHeight,
  categoryLabels,
} from '@/lib/chart';
import { formatCount, formatNumber, formatRate } from '@/lib/format';
import type {
  MetricsOverviewResponse,
  ModelShareRow,
  MtpByModelRow,
  ThroughputByModelRow,
  ThroughputTrendPoint,
  TokensByDayRow,
} from '@/lib/types';
import { useJson } from '@/lib/use-api';
import { AlertCircle, Gauge, Layers, PieChart as PieIcon, Timer, Zap } from 'lucide-react';
import { useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const DAY_MS = 24 * 60 * 60 * 1000;

const RANGE_OPTIONS: Array<{ value: string; label: string; days: number }> = [
  { value: '7', label: 'Last 7 days', days: 7 },
  { value: '30', label: 'Last 30 days', days: 30 },
  { value: '90', label: 'Last 90 days', days: 90 },
];

/** Empty-state hint shared by every chart: what to run to record data. */
const RECORD_HINT = 'Run mlx agent to record metrics. Traces older than 30 days are pruned.';

/** UTC-day string (`YYYY-MM-DD`) → short local-independent label (`Jul 20`). */
function formatDay(day: string): string {
  const parsed = new Date(`${day}T00:00:00Z`);
  if (Number.isNaN(parsed.getTime())) return day;
  return parsed.toLocaleDateString(undefined, { month: 'short', day: 'numeric', timeZone: 'UTC' });
}

/**
 * Tooltip text for one model-usage-share slice.
 *
 * A named export rather than an inline closure because the percentage here is
 * another call site of the rounding rule the Cache page's hit rate carried: a
 * bare `Math.round((turns / total) * 100)` renders 9999 turns out of 10000 as a
 * flat "100%" while the legend right beside it lists two models. `formatRate`
 * decides both endpoints from the raw counts, so "100%" requires every other
 * slice to be empty.
 */
export function modelShareLabel(turns: number, total: number): string {
  return `${formatNumber(turns)} turns · ${formatRate(turns, total)}`;
}

/** Stable trend pickers (module-level so they never re-trigger the pivot memo). */
const pickDecodeTps = (p: ThroughputTrendPoint): number | null => p.decodeTps;
const pickTtftMs = (p: ThroughputTrendPoint): number | null => p.ttftMs;

interface ModelValue {
  model: string;
  value: number;
}

function toModelValues<T extends { model: string }>(rows: T[], pick: (row: T) => number | null): ModelValue[] {
  return rows
    .map((row) => ({ model: row.model, value: pick(row) }))
    .filter((row): row is ModelValue => row.value !== null && Number.isFinite(row.value))
    .sort((a, b) => b.value - a.value);
}

export default function Metrics() {
  const [rangeKey, setRangeKey] = useState('30');
  const range = useMemo(() => {
    const days = RANGE_OPTIONS.find((o) => o.value === rangeKey)?.days ?? 30;
    const to = Date.now();
    return { from: to - days * DAY_MS, to };
  }, [rangeKey]);

  const path = `/metrics/overview?from=${range.from}&to=${range.to}`;
  const metrics = useJson<MetricsOverviewResponse>(path);
  const { loading, error } = metrics;

  const tokensByDay: TokensByDayRow[] = useMemo(() => metrics.data?.tokensByDay ?? [], [metrics.data]);
  const throughput: ThroughputByModelRow[] = useMemo(() => metrics.data?.throughputByModel ?? [], [metrics.data]);
  const throughputTrend: ThroughputTrendPoint[] = useMemo(() => metrics.data?.throughputTrend ?? [], [metrics.data]);
  const mtp: MtpByModelRow[] = useMemo(() => metrics.data?.mtpByModel ?? [], [metrics.data]);
  const modelShare: ModelShareRow[] = useMemo(() => metrics.data?.modelShare ?? [], [metrics.data]);

  // Every model this page mentions, in a stable usage rank (share turns desc,
  // then throughput, then mtp) so both the colour and the label that follow are
  // properties of the model rather than of its position in any single chart.
  const orderedModels = useMemo(() => {
    const seen = new Set<string>();
    const ordered: string[] = [];
    for (const rows of [modelShare, throughput, mtp, throughputTrend] as Array<Array<{ model: string }>>) {
      for (const row of rows) {
        if (!seen.has(row.model)) {
          seen.add(row.model);
          ordered.push(row.model);
        }
      }
    }
    return ordered;
  }, [modelShare, throughput, mtp, throughputTrend]);

  // One colour per model, held constant across every chart.
  const colorFor = useMemo(() => {
    const map = buildSeriesColorMap(orderedModels);
    return (model: string): string => map.get(model) ?? OTHER_SERIES_COLOR;
  }, [orderedModels]);

  // One short label per model, likewise held constant across every chart and
  // legend, and computed over the WHOLE page so two models can never collapse
  // onto the same name — including in charts that list only one of them.
  const labelFor = useMemo(() => {
    const map = categoryLabels(orderedModels);
    return (model: string): string => map.get(model) ?? model;
  }, [orderedModels]);

  const tokensData = useMemo(
    () =>
      tokensByDay.map((row) => ({
        day: formatDay(row.day),
        input: row.input,
        output: row.output,
        cached: row.cached,
      })),
    [tokensByDay],
  );
  const hasTokens = tokensByDay.some((row) => row.input + row.output + row.cached > 0);

  const decodeData = useMemo(() => toModelValues(throughput, (r) => r.avgDecodeTps), [throughput]);
  const ttftData = useMemo(() => toModelValues(throughput, (r) => r.avgTtftMs), [throughput]);
  const mtpData = useMemo(() => toModelValues(mtp, (r) => r.meanAccepted), [mtp]);

  // Ordered model list for the temporal trend series (most-sampled first, name
  // tie-break) so both trend charts share one stable legend order and colour.
  const trendModels = useMemo(() => {
    const totals = new Map<string, number>();
    for (const p of throughputTrend) {
      totals.set(p.model, (totals.get(p.model) ?? 0) + (Number.isFinite(p.samples) ? p.samples : 0));
    }
    return [...totals.entries()]
      .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
      .map(([model]) => model);
  }, [throughputTrend]);
  const hasDecodeTrend = throughputTrend.some((p) => Number.isFinite(p.decodeTps));
  const hasTtftTrend = throughputTrend.some((p) => Number.isFinite(p.ttftMs));
  const shareData = useMemo(() => modelShare.filter((row) => row.turns > 0), [modelShare]);
  const shareTotal = useMemo(() => shareData.reduce((sum, row) => sum + row.turns, 0), [shareData]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Metrics</h1>
          <p className="text-muted-foreground text-sm">Throughput, token, and acceptance trends across sessions.</p>
        </div>
        <Select value={rangeKey} onValueChange={setRangeKey}>
          <SelectTrigger className="w-40" aria-label="Date range">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {RANGE_OPTIONS.map((o) => (
              <SelectItem key={o.value} value={o.value}>
                {o.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {error !== undefined && (
        <Card>
          <CardContent className="text-destructive flex items-center gap-2 py-6 text-sm">
            <AlertCircle className="size-4 shrink-0" aria-hidden />
            {error.message}
          </CardContent>
        </Card>
      )}

      <ChartCard title="Tokens per day" subtitle="Input, output, and cached tokens by day (UTC)" heightClass="h-64">
        <ChartBody
          loading={loading}
          error={error}
          isEmpty={!hasTokens}
          empty={<ChartEmpty icon={Layers} message="No token usage in this range." hint={RECORD_HINT} />}
        >
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={tokensData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" vertical={false} />
              <XAxis dataKey="day" tick={AXIS_TICK} tickLine={false} axisLine={false} minTickGap={16} />
              <YAxis
                tick={AXIS_TICK}
                tickLine={false}
                axisLine={false}
                width={48}
                tickFormatter={(v) => formatCount(Number(v))}
              />
              <Tooltip
                cursor={{ fill: 'var(--color-muted)', opacity: 0.35 }}
                contentStyle={TOOLTIP_CONTENT_STYLE}
                formatter={(value) => formatNumber(Number(value))}
              />
              <Legend
                wrapperStyle={{ fontSize: 12 }}
                formatter={(value) => <span className="text-foreground">{value}</span>}
              />
              <Bar
                dataKey="input"
                stackId="tokens"
                name="Input"
                fill="var(--viz-input)"
                stroke="var(--color-card)"
                strokeWidth={1}
                isAnimationActive={false}
              />
              <Bar
                dataKey="output"
                stackId="tokens"
                name="Output"
                fill="var(--viz-output)"
                stroke="var(--color-card)"
                strokeWidth={1}
                isAnimationActive={false}
              />
              <Bar
                dataKey="cached"
                stackId="tokens"
                name="Cached"
                fill="var(--viz-cached)"
                stroke="var(--color-card)"
                strokeWidth={1}
                radius={[4, 4, 0, 0]}
                isAnimationActive={false}
              />
            </BarChart>
          </ResponsiveContainer>
        </ChartBody>
      </ChartCard>

      <ChartCard
        title="Decode throughput over time"
        subtitle="Average decode tokens/second per day, per model (UTC)"
        heightClass="h-64"
      >
        <ChartBody
          loading={loading}
          error={error}
          isEmpty={!hasDecodeTrend}
          empty={<ChartEmpty icon={Zap} message="No decode throughput in this range." hint={RECORD_HINT} />}
        >
          <ModelTrendChart
            points={throughputTrend}
            models={trendModels}
            pick={pickDecodeTps}
            unit=" tok/s"
            valueFormat={(v) => v.toFixed(0)}
            colorFor={colorFor}
            labelFor={labelFor}
          />
        </ChartBody>
      </ChartCard>

      <ChartCard
        title="Time to first token over time"
        subtitle="Average TTFT in milliseconds per day, per model (UTC)"
        heightClass="h-64"
      >
        <ChartBody
          loading={loading}
          error={error}
          isEmpty={!hasTtftTrend}
          empty={<ChartEmpty icon={Timer} message="No TTFT samples in this range." hint={RECORD_HINT} />}
        >
          <ModelTrendChart
            points={throughputTrend}
            models={trendModels}
            pick={pickTtftMs}
            unit=" ms"
            valueFormat={(v) => Math.round(v).toString()}
            colorFor={colorFor}
            labelFor={labelFor}
          />
        </ChartBody>
      </ChartCard>

      <div className="grid gap-4 lg:grid-cols-2">
        <ChartCard
          title="Decode throughput per model"
          subtitle="Average decode tokens/second (per model)"
          heightPx={categoryChartHeight(decodeData.length)}
        >
          <ChartBody
            loading={loading}
            error={error}
            isEmpty={decodeData.length === 0}
            empty={<ChartEmpty icon={Zap} message="No throughput samples in this range." hint={RECORD_HINT} />}
          >
            <ModelBarChart
              data={decodeData}
              unit=" tok/s"
              valueFormat={(v) => v.toFixed(0)}
              colorFor={colorFor}
              labelFor={labelFor}
            />
          </ChartBody>
        </ChartCard>

        <ChartCard
          title="Time to first token per model"
          subtitle="Average TTFT in milliseconds (per model)"
          heightPx={categoryChartHeight(ttftData.length)}
        >
          <ChartBody
            loading={loading}
            error={error}
            isEmpty={ttftData.length === 0}
            empty={<ChartEmpty icon={Timer} message="No TTFT samples in this range." hint={RECORD_HINT} />}
          >
            <ModelBarChart
              data={ttftData}
              unit=" ms"
              valueFormat={(v) => Math.round(v).toString()}
              colorFor={colorFor}
              labelFor={labelFor}
            />
          </ChartBody>
        </ChartCard>

        <ChartCard
          title="MTP mean accepted per model"
          subtitle="Average speculative-decoding tokens accepted per cycle"
          heightPx={categoryChartHeight(mtpData.length)}
        >
          <ChartBody
            loading={loading}
            error={error}
            isEmpty={mtpData.length === 0}
            empty={
              <ChartEmpty
                icon={Gauge}
                message="No speculative-decoding acceptance recorded."
                hint="Enable MTP (enableMtp) when running a model to record acceptance."
              />
            }
          >
            <ModelBarChart
              data={mtpData}
              unit=""
              valueFormat={(v) => v.toFixed(2)}
              colorFor={colorFor}
              labelFor={labelFor}
            />
          </ChartBody>
        </ChartCard>

        <ChartCard title="Model usage share" subtitle="Share of recorded turns by model">
          <ChartBody
            loading={loading}
            error={error}
            isEmpty={shareData.length === 0}
            empty={<ChartEmpty icon={PieIcon} message="No turns recorded in this range." hint={RECORD_HINT} />}
          >
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Tooltip
                  contentStyle={TOOLTIP_CONTENT_STYLE}
                  formatter={(value, _name, item) => {
                    const model = String((item as { payload?: { model?: string } }).payload?.model ?? '');
                    return [modelShareLabel(Number(value), shareTotal), model];
                  }}
                />
                <Legend
                  layout="vertical"
                  align="right"
                  verticalAlign="middle"
                  wrapperStyle={{ fontSize: 12, maxWidth: '45%' }}
                  formatter={(value) => (
                    <span className="text-foreground" title={String(value)}>
                      {labelFor(String(value))}
                    </span>
                  )}
                />
                <Pie
                  data={shareData}
                  dataKey="turns"
                  nameKey="model"
                  cx="42%"
                  innerRadius={52}
                  outerRadius={82}
                  paddingAngle={2}
                  stroke="var(--color-card)"
                  strokeWidth={2}
                  isAnimationActive={false}
                >
                  {shareData.map((row) => (
                    <Cell key={row.model} fill={colorFor(row.model)} />
                  ))}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
          </ChartBody>
        </ChartCard>
      </div>
    </div>
  );
}

interface ModelBarChartProps {
  data: ModelValue[];
  unit: string;
  valueFormat: (value: number) => string;
  colorFor: (model: string) => string;
  labelFor: (model: string) => string;
}

/**
 * Horizontal bar of a single measure across models — one bar per model, coloured
 * by the page-wide model palette so a reader can cross-reference the usage-share
 * legend. The category axis labels each model (no legend needed) and the value is
 * direct-labelled at the bar end (the sub-3:1 relief channel).
 *
 * Three properties every row on this chart depends on, none of them defaults:
 *  - `interval={0}` draws EVERY category tick. recharts' default hides the ones
 *    it measures as overlapping, which on a numeric axis is a fine economy and
 *    on a category axis is a row with a bar and no name.
 *  - the plot height comes from {@link categoryChartHeight} at the call site, so
 *    the rows actually have the space `interval={0}` assumes.
 *  - `width="auto"` lets recharts measure the rendered ticks and size the axis to
 *    them. A fixed width silently clips: the ticks are anchored at the axis's
 *    right edge, so anything wider than the reservation runs off the left of the
 *    SVG and is cut off mid-name.
 */
export function ModelBarChart({ data, unit, valueFormat, colorFor, labelFor }: ModelBarChartProps) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart layout="vertical" data={data} margin={{ top: 4, right: 44, bottom: 0, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" horizontal={false} />
        <XAxis
          type="number"
          tick={AXIS_TICK}
          tickLine={false}
          axisLine={false}
          tickFormatter={(v) => valueFormat(Number(v))}
        />
        <YAxis
          type="category"
          dataKey="model"
          tick={AXIS_TICK}
          tickLine={false}
          axisLine={false}
          width="auto"
          interval={0}
          tickFormatter={(v) => labelFor(String(v))}
        />
        <Tooltip
          cursor={{ fill: 'var(--color-muted)', opacity: 0.35 }}
          contentStyle={TOOLTIP_CONTENT_STYLE}
          formatter={(value) => `${valueFormat(Number(value))}${unit}`}
        />
        <Bar dataKey="value" radius={[0, 4, 4, 0]} stroke="var(--color-card)" strokeWidth={1} isAnimationActive={false}>
          {data.map((row) => (
            <Cell key={row.model} fill={colorFor(row.model)} />
          ))}
          <LabelList
            dataKey="value"
            position="right"
            className="fill-muted-foreground"
            fontSize={12}
            formatter={(value) => `${valueFormat(Number(value))}${unit}`}
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

interface ModelTrendChartProps {
  points: ThroughputTrendPoint[];
  /** Ordered model list (legend + z-order); only models with finite data are drawn. */
  models: string[];
  pick: (point: ThroughputTrendPoint) => number | null;
  unit: string;
  valueFormat: (value: number) => string;
  colorFor: (model: string) => string;
  labelFor: (model: string) => string;
}

/** One pivoted day row: a formatted day label plus one numeric column per model. */
type TrendRow = { day: string } & Record<string, number | string>;

/**
 * Temporal multi-series line: x = UTC day, one line per model, coloured by the
 * page-wide model palette. Mirrors the tokens/day chart's day bucketing, pivoting
 * the flat `(model, day)` trend points into one row per day. Days are sorted by
 * their raw `YYYY-MM-DD` key before formatting so the axis stays chronological,
 * and only models carrying a finite value for `pick` become series.
 */
function ModelTrendChart({ points, models, pick, unit, valueFormat, colorFor, labelFor }: ModelTrendChartProps) {
  const { rows, series } = useMemo(() => {
    const byDay = new Map<string, TrendRow>();
    const present = new Set<string>();
    const sorted = [...points].sort((a, b) => (a.day < b.day ? -1 : a.day > b.day ? 1 : 0));
    for (const p of sorted) {
      // Create the day row unconditionally so a day whose metric is null still
      // occupies an x position; leaving the series key absent lets connectNulls=false
      // draw a gap there instead of bridging neighbours into false continuity.
      let row = byDay.get(p.day);
      if (!row) {
        row = { day: formatDay(p.day) };
        byDay.set(p.day, row);
      }
      const value = pick(p);
      if (value === null || !Number.isFinite(value)) continue;
      present.add(p.model);
      row[p.model] = value;
    }
    // Map insertion order follows the day-sorted scan, so values() is chronological.
    return { rows: [...byDay.values()], series: models.filter((model) => present.has(model)) };
  }, [points, models, pick]);

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" vertical={false} />
        <XAxis dataKey="day" tick={AXIS_TICK} tickLine={false} axisLine={false} minTickGap={16} />
        <YAxis
          tick={AXIS_TICK}
          tickLine={false}
          axisLine={false}
          width={48}
          tickFormatter={(v) => valueFormat(Number(v))}
        />
        <Tooltip
          cursor={{ stroke: 'var(--color-muted-foreground)', strokeWidth: 1 }}
          contentStyle={TOOLTIP_CONTENT_STYLE}
          formatter={(value, name) => [`${valueFormat(Number(value))}${unit}`, labelFor(String(name))]}
        />
        <Legend
          wrapperStyle={{ fontSize: 12 }}
          formatter={(value) => (
            <span className="text-foreground" title={String(value)}>
              {labelFor(String(value))}
            </span>
          )}
        />
        {series.map((model) => (
          <Line
            key={model}
            type="monotone"
            dataKey={model}
            name={model}
            stroke={colorFor(model)}
            strokeWidth={2}
            dot={{ r: 2 }}
            activeDot={{ r: 5 }}
            connectNulls={false}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
