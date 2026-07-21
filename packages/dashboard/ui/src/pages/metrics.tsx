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
} from '@/lib/chart';
import { formatCount, formatNumber } from '@/lib/format';
import type {
  MetricsOverviewResponse,
  ModelShareRow,
  MtpByModelRow,
  ThroughputByModelRow,
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

/** Truncate a long model id for an axis/legend tick; the full name lives in the tooltip. */
function shortModel(model: string): string {
  return model.length > 22 ? `${model.slice(0, 21)}…` : model;
}

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
  const mtp: MtpByModelRow[] = useMemo(() => metrics.data?.mtpByModel ?? [], [metrics.data]);
  const modelShare: ModelShareRow[] = useMemo(() => metrics.data?.modelShare ?? [], [metrics.data]);

  // One colour per model, held constant across every chart. Ordering is a stable
  // usage rank (share turns desc, then throughput, then mtp) so a colour follows
  // the model rather than its position within any single chart.
  const colorFor = useMemo(() => {
    const seen = new Set<string>();
    const ordered: string[] = [];
    for (const rows of [modelShare, throughput, mtp] as Array<Array<{ model: string }>>) {
      for (const row of rows) {
        if (!seen.has(row.model)) {
          seen.add(row.model);
          ordered.push(row.model);
        }
      }
    }
    const map = buildSeriesColorMap(ordered);
    return (model: string): string => map.get(model) ?? OTHER_SERIES_COLOR;
  }, [modelShare, throughput, mtp]);

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

      <div className="grid gap-4 lg:grid-cols-2">
        <ChartCard title="Decode throughput per model" subtitle="Average decode tokens/second (per model)">
          <ChartBody
            loading={loading}
            error={error}
            isEmpty={decodeData.length === 0}
            empty={<ChartEmpty icon={Zap} message="No throughput samples in this range." hint={RECORD_HINT} />}
          >
            <ModelBarChart data={decodeData} unit=" tok/s" valueFormat={(v) => v.toFixed(0)} colorFor={colorFor} />
          </ChartBody>
        </ChartCard>

        <ChartCard title="Time to first token per model" subtitle="Average TTFT in milliseconds (per model)">
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
            />
          </ChartBody>
        </ChartCard>

        <ChartCard
          title="MTP mean accepted per model"
          subtitle="Average speculative-decoding tokens accepted per cycle"
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
            <ModelBarChart data={mtpData} unit="" valueFormat={(v) => v.toFixed(2)} colorFor={colorFor} />
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
                    const turns = Number(value);
                    const pct = shareTotal > 0 ? Math.round((turns / shareTotal) * 100) : 0;
                    const model = String((item as { payload?: { model?: string } }).payload?.model ?? '');
                    return [`${formatNumber(turns)} turns · ${pct}%`, model];
                  }}
                />
                <Legend
                  layout="vertical"
                  align="right"
                  verticalAlign="middle"
                  wrapperStyle={{ fontSize: 12, maxWidth: '45%' }}
                  formatter={(value) => (
                    <span className="text-foreground" title={String(value)}>
                      {shortModel(String(value))}
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
}

/**
 * Horizontal bar of a single measure across models — one bar per model, coloured
 * by the page-wide model palette so a reader can cross-reference the usage-share
 * legend. The category axis labels each model (no legend needed) and the value is
 * direct-labelled at the bar end (the sub-3:1 relief channel).
 */
function ModelBarChart({ data, unit, valueFormat, colorFor }: ModelBarChartProps) {
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
          width={140}
          tickFormatter={(v) => shortModel(String(v))}
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
