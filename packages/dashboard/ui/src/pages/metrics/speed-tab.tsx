import { useEffect, useMemo, type ReactElement } from 'react';

import { ChartCard } from '../../components/arc-x/chart-card/chart-card';
import type { ChartPoint, ChartSeries } from '../../components/arc-x/chart/types';
import { Histogram } from '../../components/arc-x/histogram/histogram';
import { ScatterChart, type ScatterPoint } from '../../components/arc-x/scatter-chart/scatter-chart';
import { StackedBarChart } from '../../components/arc-x/stacked-bar-chart/stacked-bar-chart';
import { StatGroup, StatTile } from '../../components/arc-x/stat-tile/stat-tile';
import { TrendChart } from '../../components/arc-x/trend-chart/trend-chart';
import {
  formatCount,
  formatCountNoun,
  formatDuration,
  formatDurationParts,
  formatNumber,
  formatTps,
} from '../../lib/format';
import { axisNumber, bucketWord, phaseMeans, PHASE_SERIES, timeAxis, toBins } from '../../lib/metrics-view';
import { modelColor, modelSeries } from '../../lib/model-colors';
import {
  PREFILL_MIN_TOKENS,
  PREFILL_NONE_HINT,
  PREFILL_NONE_TITLE,
  PREFILL_UNDER_TEXT,
  prefillCoverage,
  prefillEmpty,
} from '../../lib/prefill';
import { moreInTableLabel, pivotByModel, pivotMeasures, topModels } from '../../lib/timeseries';
import {
  LOW_SAMPLE_TAIL_N,
  type MetricsDistributions,
  type MetricsSummary,
  type MetricsTimeseries,
  type MetricsTurns,
  type Quantiles,
} from '../../lib/types';
import { PageGrid } from '../../shell/page';
import { RefreshAlert, regionError, regionState, tileState, useHeldQuery, useMetricsScope, withModel } from './shared';

const SCATTER_LIMIT = 1000;

function tailHint(q: Quantiles | undefined, tail: (q: Quantiles) => string): string {
  if (q === undefined) return '';
  return q.n >= LOW_SAMPLE_TAIL_N ? tail(q) : `median of ${formatCountNoun(q.n, 'turn', 'turns')}`;
}

function markers(q: Quantiles | undefined): Array<{ value: number; label: string }> {
  if (q === undefined) return [];
  const list: Array<{ value: number; label: string }> = [];
  if (q.p50 !== null) list.push({ value: q.p50, label: 'p50' });
  if (q.p95 !== null && q.n >= LOW_SAMPLE_TAIL_N) list.push({ value: q.p95, label: 'p95' });
  return list;
}

function durationTile(q: Quantiles | undefined): { value: string | null; unit?: string } {
  const parts = formatDurationParts(q?.p50 ?? null);
  return parts.value === '—'
    ? { value: null }
    : { value: parts.value, unit: parts.unit === '' ? undefined : parts.unit };
}

/** A histogram bound of a duration axis: the first edge is 0, not "<1 ms". */
function durationBound(ms: number): string {
  return ms === 0 ? '0' : formatDuration(ms);
}

export function SpeedTab(): ReactElement {
  const scope = useMetricsScope();
  const { q, range, model } = scope;
  const summary = useHeldQuery<MetricsSummary>(`/metrics/summary?${q}`);
  const dist = useHeldQuery<MetricsDistributions>(`/metrics/distributions?${q}&groupBy=none`);
  const byModelTs = useHeldQuery<MetricsTimeseries>(`/metrics/timeseries?${q}&bucket=${range.bucket}&groupBy=model`, {
    range,
  });
  const noneTs = useHeldQuery<MetricsTimeseries>(
    `/metrics/timeseries?${q}&bucket=${range.bucket}&groupBy=${model === null ? 'none' : 'model'}`,
    { range },
  );
  const byModelRange = byModelTs.range ?? range;
  const noneRange = noneTs.range ?? range;
  const scatter = useHeldQuery<MetricsTurns>(`/metrics/turns?${q}&outcome=ok&limit=${String(SCATTER_LIMIT)}`);
  const latency = summary.data?.latency;
  const tile = tileState(summary);
  const histColor = model === null ? undefined : modelColor(model).color;
  const series = dist.data?.series[0];
  const edges = dist.data?.edges;

  const { reportTurns } = scope;
  useEffect(() => {
    if (scatter.data !== undefined) reportTurns(scatter.data.rows);
  }, [scatter.data, reportTurns]);

  const phases = useMemo(() => {
    const raw = phaseMeans(noneTs.data?.points ?? [], noneRange);
    let max = 0;
    for (const point of raw) {
      let sum = 0;
      for (const value of Object.values(point.values)) sum += value ?? 0;
      max = Math.max(max, sum);
    }
    const axis = timeAxis(max);
    const shown = new Set<string>();
    const data: ChartPoint[] = raw.map((point) => {
      const values: Record<string, number | null> = {};
      for (const [key, value] of Object.entries(point.values)) {
        values[key] = value === null ? null : value / axis.div;
        if (value !== null && value > 0) shown.add(key);
      }
      return { ...point, values };
    });
    return { data, axis, series: PHASE_SERIES.filter((phase) => shown.has(phase.key)) };
  }, [noneTs.data, noneRange]);

  const scatterView = useMemo(() => {
    const rows = scatter.data?.rows ?? [];
    const usable = rows.filter((row) => row.prefillMs !== null && row.prefillTokens !== null);
    let max = 0;
    for (const row of usable) max = Math.max(max, row.prefillMs ?? 0);
    const axis = timeAxis(max);
    const counts = new Map<string, number>();
    for (const row of usable) counts.set(row.model, (counts.get(row.model) ?? 0) + 1);
    const top = [...counts.entries()]
      .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
      .slice(0, 8)
      .map(([name]) => name);
    const drawn = new Set(top);
    const points: ScatterPoint[] = usable
      .filter((row) => drawn.has(row.model))
      .map((row) => ({
        key: row.traceId,
        x: row.prefillTokens ?? 0,
        y: (row.prefillMs ?? 0) / axis.div,
        series: row.model,
        hollow: (row.concurrencyPeak ?? 1) > 1,
        detail: [
          { label: 'Setup', value: formatDuration(row.setupMs) },
          {
            label: 'Prefill speed',
            value: row.prefillTps === null ? PREFILL_UNDER_TEXT : `${formatCount(row.prefillTps)} tok/s`,
          },
        ],
      }));
    return {
      points,
      axis,
      series: modelSeries(top),
      total: scatter.data?.total ?? 0,
      rows: rows.length,
      unmeasured: rows.length - usable.length,
    };
  }, [scatter.data]);

  const prefill = useMemo((): { data: ChartPoint[]; series: ChartSeries[]; dropped: string[] } => {
    const points = byModelTs.data?.points ?? [];
    if (model !== null) {
      const data = pivotMeasures(
        points,
        byModelRange,
        {
          [model]: (p) => p.prefillTps.p50,
          p05: (p) => (p.prefillTps.n >= LOW_SAMPLE_TAIL_N ? p.prefillTps.p05 : null),
          p95: (p) => (p.prefillTps.n >= LOW_SAMPLE_TAIL_N ? p.prefillTps.p95 : null),
        },
        null,
      );
      const counts = pivotMeasures(points, byModelRange, { n: (p) => p.prefillTps.n }, 0);
      return {
        data: data.map((point, index) => ({ ...point, samples: { [model]: counts[index]?.values.n ?? 0 } })),
        series: [{ key: model, label: model, color: modelColor(model).color }],
        dropped: [],
      };
    }
    const top = topModels(summary.data?.byModel ?? []);
    const pivot = pivotByModel(points, byModelRange, (p) => p.prefillTps.p50, {
      empty: null,
      samples: (p) => p.prefillTps.n,
      models: top,
      fold: 'drop',
    });
    return { data: pivot.data, series: modelSeries(top), dropped: pivot.dropped };
  }, [byModelTs.data, summary.data, model, byModelRange]);

  const ok = summary.data?.totals.ok ?? 0;
  const noOk = summary.data !== undefined && ok === 0;
  const measured = summary.data?.totals.prefillMeasured ?? 0;
  const rated = latency?.prefillTps.n ?? 0;
  const unmeasured = ok - measured;

  const ttft = durationTile(latency?.ttftMs);
  const setup = durationTile(latency?.setupMs);
  const total = durationTile(latency?.totalMs);
  const queue = durationTile(latency?.queueMs);
  const decodeText = formatTps(latency?.decodeTps.p50 ?? null);
  const prefillText = formatCount(latency?.prefillTps.p50 ?? null);

  const histogram = (
    title: string,
    subtitle: string,
    key: 'decodeTps' | 'ttftMs' | 'totalMs',
    unit: string,
    formatBound: (value: number) => string,
    quantiles: Quantiles | undefined,
    span: string,
  ): ReactElement => (
    <ChartCard
      className={span}
      title={title}
      subtitle={withModel(subtitle, model)}
      height={160}
      state={regionState(dist, noOk || (series?.[key].n ?? 0) === 0)}
      empty={{ title: 'No completed turns in this range' }}
      error={regionError(dist)}
    >
      <Histogram
        label={title}
        unit={unit}
        height={160}
        countLabel={{ one: 'turn', other: 'turns' }}
        bins={toBins(edges?.[key] ?? [], series?.[key])}
        markers={markers(quantiles)}
        formatBound={formatBound}
        color={histColor}
        emptyLabel="No completed turns in this range"
      />
    </ChartCard>
  );

  const scatterUnmeasured =
    scatterView.unmeasured > 0 ? `${formatNumber(scatterView.unmeasured)} without a prefill measurement · ` : '';
  const scatterFootnote =
    scatterView.total > SCATTER_LIMIT
      ? `Newest 1,000 of ${formatNumber(scatterView.total)} · ${scatterUnmeasured}hollow = ran beside other turns`
      : `${formatNumber(scatterView.points.length)} of ${formatCountNoun(scatterView.total, 'completed turn', 'completed turns')} · ${scatterUnmeasured}hollow = ran beside other turns`;

  const setupHint =
    latency !== undefined && latency.setupMs.n === 0
      ? 'no measured prefill'
      : tailHint(latency?.setupMs, (v) => `p95 ${formatDuration(v.p95)}`);
  let prefillHint = tailHint(latency?.prefillTps, (v) => `p05 ${formatCount(v.p05)}`);
  if (latency !== undefined && rated === 0) {
    prefillHint = measured === 0 ? 'no measured prefill' : `no prefill of ${String(PREFILL_MIN_TOKENS)}+ tokens`;
  }

  const phaseFootnote =
    'Setup: before prefill, prefix lookup, cache restore and allocation; after it, the first decode step.' +
    (summary.data !== undefined && unmeasured > 0
      ? ` To first token: ${formatCountNoun(unmeasured, 'turn', 'turns')} without a prefill measurement, setup and prefill together.`
      : '');

  const coverage = summary.data === undefined ? null : prefillCoverage(ok, measured, rated);
  const more = prefill.dropped.length > 0 ? moreInTableLabel(prefill.dropped.length) : null;
  const prefillFootnote = [coverage, more].filter((part): part is string => part !== null).join(' · ');
  const prefillEmptyState = noOk ? { title: 'No completed turns in this range' } : prefillEmpty(measured);

  return (
    <div className="grid gap-3 pt-3">
      <RefreshAlert queries={[summary, dist, byModelTs, noneTs, scatter]} />
      <StatGroup label="Percentiles" maxColumns={3}>
        <StatTile
          label="First token"
          {...ttft}
          hint={tailHint(latency?.ttftMs, (v) => `p95 ${formatDuration(v.p95)}`)}
          sampleCount={latency?.ttftMs.n}
          {...tile}
        />
        <StatTile label="Setup" {...setup} hint={setupHint} sampleCount={latency?.setupMs.n} {...tile} />
        <StatTile
          label="Prefill speed"
          value={prefillText === '—' ? null : prefillText}
          unit="tok/s"
          hint={prefillHint}
          sampleCount={latency?.prefillTps.n}
          {...tile}
        />
        <StatTile
          label="Decode speed"
          value={decodeText === '—' ? null : decodeText}
          unit="tok/s"
          hint={tailHint(latency?.decodeTps, (v) => `slowest 5%: ${formatTps(v.p05)} tok/s`)}
          sampleCount={latency?.decodeTps.n}
          {...tile}
        />
        <StatTile
          label="Whole turn"
          {...total}
          hint={tailHint(latency?.totalMs, (v) => `p95 ${formatDuration(v.p95)}`)}
          sampleCount={latency?.totalMs.n}
          {...tile}
        />
        <StatTile
          label="Queue wait"
          {...queue}
          hint={tailHint(latency?.queueMs, (v) => `p95 ${formatDuration(v.p95)}`)}
          sampleCount={latency?.queueMs.n}
          {...tile}
        />
      </StatGroup>
      <PageGrid>
        {histogram(
          'Decode speed',
          'Completed turns per speed range · tok/s',
          'decodeTps',
          'tok/s',
          axisNumber,
          latency?.decodeTps,
          'col-span-12 @md:col-span-6',
        )}
        {histogram(
          'Time to first token',
          'Completed turns per time range',
          'ttftMs',
          '',
          durationBound,
          latency?.ttftMs,
          'col-span-12 @md:col-span-6',
        )}
        <ChartCard
          className="col-span-12 @md:col-span-7"
          title="Average turn, by phase"
          subtitle={withModel(
            `Mean wall time per turn, per ${bucketWord(noneRange.bucket)} · ${phases.axis.unit}`,
            model,
          )}
          height={200}
          state={regionState(noneTs, summary.data?.totals.turns === 0)}
          empty={{ title: 'No turns in this range' }}
          error={regionError(noneTs)}
          footnote={phaseFootnote}
        >
          <StackedBarChart
            label="Average turn, by phase"
            unit={phases.axis.unit}
            height={200}
            data={phases.data}
            series={phases.series}
            lowSampleBelow={5}
            formatValue={axisNumber}
          />
        </ChartCard>
        {histogram(
          'Whole turn',
          'Turns per duration range',
          'totalMs',
          '',
          durationBound,
          latency?.totalMs,
          'col-span-12 @md:col-span-5',
        )}
        <ChartCard
          className="col-span-12 @md:col-span-7"
          title="Prefill time against prefill tokens"
          subtitle={withModel(`Newest 1,000 completed turns · ${scatterView.axis.unit}`, model)}
          height={264}
          state={regionState(scatter, scatterView.points.length === 0)}
          empty={
            scatterView.rows === 0
              ? { title: 'No completed turns in this range' }
              : { title: PREFILL_NONE_TITLE, hint: PREFILL_NONE_HINT }
          }
          error={regionError(scatter)}
          footnote={scatterFootnote}
        >
          <div className="grid gap-2">
            <ScatterChart
              label="Prefill time against prefill tokens"
              xLabel="Prefill tokens"
              xUnit="tokens"
              yLabel="Prefill time"
              yUnit={scatterView.axis.unit}
              height={240}
              points={scatterView.points}
              series={scatterView.series}
              hollowLabel="Ran beside other turns"
              formatX={formatCount}
              formatY={axisNumber}
              onSelect={(point) => scope.setParams({ turn: point.key })}
            />
            <p className="text-caption text-secondary">
              Points on one line share a prefill speed. Points above it took longer than their token count explains.
            </p>
          </div>
        </ChartCard>
        <ChartCard
          className="col-span-12 @md:col-span-5"
          title="Prefill speed"
          subtitle={withModel(
            `Median per ${bucketWord(byModelRange.bucket)}, per model · tok/s · prefills of ${String(PREFILL_MIN_TOKENS)}+ tokens`,
            model,
          )}
          height={240}
          state={regionState(byModelTs, noOk || (latency !== undefined && rated === 0))}
          empty={prefillEmptyState}
          error={regionError(byModelTs)}
          footnote={prefillFootnote === '' ? undefined : prefillFootnote}
        >
          <TrendChart
            label="Prefill speed"
            unit="tok/s"
            height={240}
            curve="linear"
            data={prefill.data}
            series={prefill.series}
            band={model === null ? undefined : { lowKey: 'p05', highKey: 'p95' }}
            lowSampleBelow={5}
            formatValue={formatCount}
            formatTick={axisNumber}
          />
        </ChartCard>
      </PageGrid>
    </div>
  );
}
