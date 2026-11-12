import { useMemo, useRef, type ReactElement } from 'react';
import { useSearchParams } from 'react-router-dom';

import { ChartCard } from '../../components/arc-x/chart-card/chart-card';
import type { ChartPoint, ChartSeries } from '../../components/arc-x/chart/types';
import { useChartSize } from '../../components/arc-x/chart/use-chart-size';
import { DataTable, type DataColumn } from '../../components/arc-x/data-table/data-table';
import { KpiDrilldown } from '../../components/arc-x/kpi-drilldown/kpi-drilldown';
import { ModelBadge } from '../../components/arc-x/model-badge/model-badge';
import { Panel } from '../../components/arc-x/panel/panel';
import { QuantileBar } from '../../components/arc-x/quantile-bar/quantile-bar';
import { Meter, SplitBar } from '../../components/arc-x/split-bar/split-bar';
import { StackedBarChart } from '../../components/arc-x/stacked-bar-chart/stacked-bar-chart';
import { StatGroup, StatTile } from '../../components/arc-x/stat-tile/stat-tile';
import { TrendChart } from '../../components/arc-x/trend-chart/trend-chart';
import { Button } from '../../components/arc/button/button';
import {
  formatCount,
  formatCountNoun,
  formatDuration,
  formatNumber,
  formatRate,
  formatRelativeTime,
  formatTps,
} from '../../lib/format';
import {
  axisNumber,
  bucketWord,
  isKpiKey,
  kpiSeries,
  kpiTiles,
  phaseSegments,
  RECORD_HINT,
  timeAxis,
  versusText,
  type KpiKey,
} from '../../lib/metrics-view';
import { modelColor, modelSeries } from '../../lib/model-colors';
import { findLocalModel } from '../../lib/model-usage';
import { OUTCOME_SERIES } from '../../lib/series';
import { moreInTableLabel, OTHER_KEY, otherLabel, pivotByModel, pivotMeasures, topModels } from '../../lib/timeseries';
import {
  LOW_SAMPLE_TAIL_N,
  type MetricsSummary,
  type MetricsTimeseries,
  type ModelsResponse,
  type Quantiles,
} from '../../lib/types';
import { useJson } from '../../lib/use-api';
import { kpiDetail, kpiDetailLoading, ShowLink } from './kpi-detail';
import {
  MedianCell,
  quantileMax,
  RefreshAlert,
  regionError,
  regionState,
  tileState,
  useHeldQuery,
  useMetricsScope,
  useReportBusy,
  withModel,
} from './shared';

type TsPoint = MetricsTimeseries['points'][number];
type ByModelRow = MetricsSummary['byModel'][number];

/** Medians with the p05–p95 band of one model; band keys are null below LOW_SAMPLE_TAIL_N. */
function bandPoints(
  points: readonly TsPoint[],
  range: Parameters<typeof pivotMeasures>[1],
  key: string,
  pick: (p: TsPoint) => Quantiles,
): ChartPoint[] {
  const data = pivotMeasures(
    points,
    range,
    {
      [key]: (p) => pick(p).p50,
      p05: (p) => (pick(p).n >= LOW_SAMPLE_TAIL_N ? pick(p).p05 : null),
      p95: (p) => (pick(p).n >= LOW_SAMPLE_TAIL_N ? pick(p).p95 : null),
    },
    null,
  );
  const counts = pivotMeasures(points, range, { n: (p) => pick(p).n }, 0);
  return data.map((point, index) => ({ ...point, samples: { [key]: counts[index]?.values.n ?? 0 } }));
}

function scaleValues(data: readonly ChartPoint[], div: number): ChartPoint[] {
  if (div === 1) return [...data];
  return data.map((point) => {
    const values: Record<string, number | null> = {};
    for (const [key, value] of Object.entries(point.values)) values[key] = value === null ? null : value / div;
    return { ...point, values };
  });
}

function maxValue(data: readonly ChartPoint[]): number {
  let max = 0;
  for (const point of data)
    for (const value of Object.values(point.values)) if (value !== null) max = Math.max(max, value);
  return max;
}

export function SummaryTab(): ReactElement {
  const scope = useMetricsScope();
  const { range, model, q, rangeKey } = scope;
  const [params] = useSearchParams();
  const rawKpi = params.get('kpi');
  const kpi: KpiKey = rawKpi !== null && isKpiKey(rawKpi) ? rawKpi : 'decode';
  const rawDetail = params.get('detail');
  const detailKey = rawDetail !== null && isKpiKey(rawDetail) ? rawDetail : null;

  const summary = useHeldQuery<MetricsSummary>(`/metrics/summary?${q}`);
  const byModelTs = useHeldQuery<MetricsTimeseries>(`/metrics/timeseries?${q}&bucket=${range.bucket}&groupBy=model`, {
    range,
  });
  const noneTs = useHeldQuery<MetricsTimeseries>(
    `/metrics/timeseries?${q}&bucket=${range.bucket}&groupBy=${model === null ? 'none' : 'model'}`,
    { range },
  );
  const byModelRange = byModelTs.range ?? range;
  const noneRange = noneTs.range ?? range;
  const locals = useJson<ModelsResponse>('/models');
  useReportBusy(locals);
  const data = summary.data;
  const tiles = useMemo(
    () => (data === undefined ? null : kpiTiles(data, versusText(rangeKey), 'tails')),
    [data, rangeKey],
  );
  const nonePoints = noneTs.data?.points;
  const top = useMemo(
    () => (data === undefined ? [] : model === null ? topModels(data.byModel) : [model]),
    [data, model],
  );
  const tile = tileState(summary);

  const chart = useMemo(() => {
    const points = byModelTs.data?.points ?? [];
    const word = bucketWord((kpi === 'failed' ? noneRange : byModelRange).bucket);
    if (kpi === 'decode' || kpi === 'ttft') {
      const pick = (p: TsPoint): Quantiles => (kpi === 'decode' ? p.decodeTps : p.ttftMs);
      let raw: ChartPoint[];
      let series: ChartSeries[];
      let dropped: string[] = [];
      if (model !== null) {
        raw = bandPoints(points, byModelRange, model, pick);
        series = [{ key: model, label: model, color: modelColor(model).color }];
      } else {
        const pivot = pivotByModel(points, byModelRange, (p) => pick(p).p50, {
          empty: null,
          samples: (p) => pick(p).n,
          models: top,
          fold: 'drop',
        });
        raw = pivot.data;
        dropped = pivot.dropped;
        series = modelSeries(top);
      }
      const axis = kpi === 'ttft' ? timeAxis(maxValue(raw)) : { unit: 'tok/s', div: 1 };
      const title = kpi === 'decode' ? 'Decode speed' : 'Time to first token';
      const subtitle =
        model === null
          ? `Median per ${word}, per model · ${axis.unit}`
          : withModel(`Median with the p05–p95 range per ${word} · ${axis.unit}`, model);
      return {
        kind: 'line' as const,
        title,
        subtitle,
        unit: axis.unit,
        div: axis.div,
        data: scaleValues(raw, axis.div),
        series,
        dropped,
      };
    }
    if (kpi === 'failed') {
      const raw = pivotMeasures(
        nonePoints ?? [],
        noneRange,
        { error: (p) => p.totals.error, cancelled: (p) => p.totals.cancelled },
        0,
      );
      return {
        kind: 'bars' as const,
        title: 'Failed and cancelled turns',
        subtitle: withModel(`Per ${word} · turns`, model),
        unit: 'turns',
        div: 1,
        data: raw,
        series: OUTCOME_SERIES,
        dropped: [] as string[],
      };
    }
    const pick = (p: TsPoint): number => (kpi === 'turns' ? p.totals.turns : p.totals.outputTokens);
    const pivot = pivotByModel(points, byModelRange, pick, { empty: 0, models: top, fold: 'sum' });
    const series = modelSeries(top);
    if (pivot.dropped.length > 0) {
      series.push({
        key: OTHER_KEY,
        label: otherLabel(pivot.dropped.length),
        color: 'var(--model-other)',
      });
    }
    return {
      kind: 'bars' as const,
      title: kpi === 'turns' ? 'Turns' : 'Output tokens',
      subtitle: withModel(`Per ${word}, by model · ${kpi === 'turns' ? 'turns' : 'tokens'}`, model),
      unit: kpi === 'turns' ? 'turns' : 'tokens',
      div: 1,
      data: pivot.data,
      series,
      dropped: [] as string[],
    };
  }, [byModelTs.data, nonePoints, kpi, model, byModelRange, noneRange, top]);

  const totals = data?.totals;
  const noTurns = totals !== undefined && totals.turns === 0;
  const noValue =
    data !== undefined &&
    (kpi === 'decode' ? data.latency.decodeTps.n === 0 : kpi === 'ttft' ? data.latency.ttftMs.n === 0 : false);
  const chartSource = chart.kind === 'line' || kpi !== 'failed' ? byModelTs : noneTs;
  const mainState =
    summary.data === undefined ? regionState(summary, false) : regionState(chartSource, noTurns || noValue);
  const footnote =
    totals === undefined
      ? undefined
      : [
          `${formatCountNoun(totals.ok, 'completed turn', 'completed turns')} · ${formatCountNoun(data?.byModel.filter((row) => row.totals.turns > 0).length ?? 0, 'model', 'models')}`,
          chart.dropped.length > 0 ? moreInTableLabel(chart.dropped.length) : null,
          chart.kind === 'line' ? 'hollow marker = fewer than 5 turns' : null,
        ]
          .filter((part): part is string => part !== null)
          .join(' · ');

  const formatLine = kpi === 'ttft' ? axisNumber : (value: number) => formatTps(value);

  return (
    <div className="grid gap-3 pt-3">
      <RefreshAlert queries={[summary, byModelTs, noneTs]} />
      <KpiDrilldown
        label="Key numbers"
        open={detailKey}
        onOpenChange={(key) => scope.setParams({ detail: key })}
        detail={(key) => {
          if (!isKpiKey(key)) return null;
          const footer = (
            <ShowLink
              to={scope.urlWith({ tab: 'turns', detail: null, kpi: null, outcome: key === 'failed' ? 'error' : null })}
            >
              Show turns
            </ShowLink>
          );
          const view = tiles?.find((item) => item.key === key);
          if (data === undefined || view === undefined) return kpiDetailLoading(key, rangeKey, footer);
          return kpiDetail({
            key,
            tile: view,
            rangeKey,
            range,
            model,
            summary: data,
            points: nonePoints ?? [],
            pointsRange: noneRange,
            pointsLoading: nonePoints === undefined,
            pointsError: nonePoints === undefined ? regionError(noneTs) : undefined,
            footer,
          });
        }}
      >
        <div className="grid gap-3">
          <StatGroup label="Key numbers">
            {(tiles ?? kpiPlaceholders()).map((view) => (
              <StatTile
                key={view.key}
                label={view.label}
                value={view.value}
                unit={view.unit}
                delta={view.delta}
                hint={view.hint}
                sampleCount={view.sampleCount}
                spark={
                  nonePoints === undefined
                    ? undefined
                    : {
                        data: kpiSeries(view.key, nonePoints, noneRange).slice(-30),
                        label: `${view.label} per ${bucketWord(noneRange.bucket)}`,
                      }
                }
                selected={kpi === view.key}
                onSelect={() => scope.setParams({ kpi: view.key === 'decode' ? null : view.key })}
                detailKey={view.key}
                {...tile}
              />
            ))}
          </StatGroup>
          <ChartCard
            title={chart.title}
            subtitle={chart.subtitle}
            height={240}
            state={mainState}
            empty={
              noTurns || data === undefined
                ? { title: 'No turns in this range', hint: RECORD_HINT }
                : { title: 'No completed turns in this range', hint: RECORD_HINT }
            }
            error={regionError(summary.data === undefined ? summary : chartSource)}
            footnote={footnote}
          >
            {chart.kind === 'line' ? (
              <TrendChart
                label={chart.title}
                unit={chart.unit}
                height={240}
                curve="linear"
                data={chart.data}
                series={chart.series}
                band={model === null ? undefined : { lowKey: 'p05', highKey: 'p95' }}
                lowSampleBelow={5}
                formatValue={formatLine}
                formatTick={axisNumber}
              />
            ) : (
              <StackedBarChart
                label={chart.title}
                unit={chart.unit}
                height={240}
                data={chart.data}
                series={chart.series}
                formatValue={formatNumber}
                formatTick={formatCount}
                onSelect={() =>
                  scope.setParams({ tab: 'turns', kpi: null, detail: null, outcome: kpi === 'failed' ? 'error' : null })
                }
              />
            )}
          </ChartCard>
        </div>
      </KpiDrilldown>
      <TimePanel summary={summary.data} />
      <ModelsTable summary={summary} locals={locals.data} />
    </div>
  );
}

function kpiPlaceholders(): ReturnType<typeof kpiTiles> {
  const labels: Array<[KpiKey, string]> = [
    ['decode', 'Decode speed'],
    ['ttft', 'First token'],
    ['turns', 'Turns'],
    ['tokens', 'Output tokens'],
    ['failed', 'Failed turns'],
  ];
  return labels.map(([key, label]) => ({
    key,
    label,
    value: null,
    hint: '',
    delta: { current: null, previous: null, goodWhen: 'neutral', versus: '' },
  }));
}

function TimePanel({ summary }: { summary: MetricsSummary | undefined }): ReactElement {
  const totals = summary?.totals;
  const description =
    totals !== undefined && totals.firstTokenWallMs > 0
      ? 'Share of wall time across all turns in the range. To first token: turns without a prefill measurement, setup and prefill together.'
      : 'Share of wall time across all turns in the range.';
  return (
    <Panel title="Where the time goes" description={description} busy={summary === undefined}>
      {totals === undefined ? (
        <SplitBar size="tall" label="Wall time by phase" segments={[]} total={null} unknownLabel="" />
      ) : totals.totalWallMs === 0 ? (
        <p className="text-body text-secondary">No turns in this range</p>
      ) : (
        <SplitBar
          size="tall"
          label="Wall time by phase"
          segments={phaseSegments(totals)}
          total={totals.totalWallMs}
          formatValue={formatDuration}
        />
      )}
    </Panel>
  );
}

/** A latency cell: the median (nowrap) + gap + the bar. */
const LATENCY_BAR = 76;
const LATENCY_COLUMN = 166;

function ModelsTable({
  summary,
  locals,
}: {
  summary: ReturnType<typeof useHeldQuery<MetricsSummary>>;
  locals: ModelsResponse | undefined;
}): ReactElement {
  const scope = useMetricsScope();
  const rows = useMemo(() => summary.data?.byModel ?? [], [summary.data]);
  const allTurns = summary.data?.totals.turns ?? 0;
  const decodeMax = quantileMax(rows.map((row) => row.latency.decodeTps));
  const ttftMax = quantileMax(rows.map((row) => row.latency.ttftMs));
  const box = useRef<HTMLDivElement>(null);
  const bars = useChartSize(box, 916) >= 640;

  const columns: DataColumn<ByModelRow>[] = [
    {
      id: 'model',
      header: 'Model',
      flex: true,
      minWidth: 140,
      cell: (row) => {
        const quant = findLocalModel(locals?.models, row.model)?.quant;
        return (
          <span className="grid min-w-0 justify-items-start">
            <ModelBadge name={row.model} size="sm" fit="width" />
            {quant === undefined || quant === null || quant === '' ? null : (
              <span className="max-w-full truncate text-caption text-muted">{quant}</span>
            )}
          </span>
        );
      },
      sortValue: (row) => row.model,
      truncate: true,
    },
    {
      id: 'turns',
      header: 'Turns',
      align: 'end',
      width: 84,
      cell: (row) => (
        <span className="whitespace-nowrap tabular-nums">
          {formatNumber(row.totals.turns)} <span className="text-muted">{formatRate(row.totals.turns, allTurns)}</span>
        </span>
      ),
      sortValue: (row) => row.totals.turns,
    },
    {
      id: 'decode',
      header: 'Decode tok/s',
      width: bars ? LATENCY_COLUMN : undefined,
      cell: (row) => (
        <span className="inline-flex items-center gap-2">
          <MedianCell text={formatTps(row.latency.decodeTps.p50)} n={row.latency.decodeTps.n} />
          {bars ? (
            <QuantileBar
              label={`Decode speed of ${row.model}`}
              quantiles={row.latency.decodeTps}
              min={0}
              max={decodeMax}
              unit="tok/s"
              formatValue={formatTps}
              color={modelColor(row.model).color}
              width={LATENCY_BAR}
            />
          ) : null}
        </span>
      ),
      sortValue: (row) => row.latency.decodeTps.p50,
    },
    {
      id: 'ttft',
      header: 'First token',
      width: bars ? LATENCY_COLUMN : undefined,
      cell: (row) => (
        <span className="inline-flex items-center gap-2">
          <MedianCell text={formatDuration(row.latency.ttftMs.p50)} n={row.latency.ttftMs.n} />
          {bars ? (
            <QuantileBar
              label={`Time to first token of ${row.model}`}
              quantiles={row.latency.ttftMs}
              min={0}
              max={ttftMax}
              unit=""
              formatValue={formatDuration}
              color={modelColor(row.model).color}
              width={LATENCY_BAR}
            />
          ) : null}
        </span>
      ),
      sortValue: (row) => row.latency.ttftMs.p50,
    },
    {
      id: 'prefill',
      header: 'Prefill tok/s',
      align: 'end',
      width: 90,
      minTableWidth: 760,
      cell: (row) => <MedianCell text={formatCount(row.latency.prefillTps.p50)} n={row.latency.prefillTps.n} />,
      sortValue: (row) => row.latency.prefillTps.p50,
    },
    {
      id: 'failed',
      header: 'Failed',
      align: 'end',
      width: 72,
      cell: (row) => (
        <span className="whitespace-nowrap tabular-nums">
          {formatRate(row.totals.error, row.totals.turns)}
          {row.totals.error > 0 ? <span className="text-muted">{` (${formatNumber(row.totals.error)})`}</span> : null}
        </span>
      ),
      sortValue: (row) => (row.totals.turns > 0 ? row.totals.error / row.totals.turns : 0),
    },
    {
      id: 'context',
      header: 'Context',
      align: 'end',
      width: 96,
      minTableWidth: 640,
      cell: (row) => {
        const fill = row.latency.ctxFillPct;
        const tail = fill.n >= LOW_SAMPLE_TAIL_N && fill.p95 !== null;
        const pct = tail ? fill.p95 : fill.p50;
        if (pct === null) return <span className="text-muted">—</span>;
        return (
          <span className="grid justify-items-end gap-1">
            <span className="whitespace-nowrap tabular-nums" title={`n=${formatNumber(fill.n)}`}>
              {`${tail ? 'p95' : 'median'} ${String(Math.round(pct))}%`}
            </span>
            <Meter
              size="dense"
              label={`Context window in use by ${row.model}`}
              value={pct}
              max={100}
              className="w-full"
            />
          </span>
        );
      },
      sortValue: (row) => row.latency.ctxFillPct.p95 ?? row.latency.ctxFillPct.p50,
    },
    {
      id: 'output',
      header: 'Output tokens',
      align: 'end',
      width: 100,
      minTableWidth: 980,
      cell: (row) => <span className="tabular-nums">{formatCount(row.totals.outputTokens)}</span>,
      sortValue: (row) => row.totals.outputTokens,
    },
    {
      id: 'last',
      header: 'Last used',
      align: 'end',
      width: 84,
      minTableWidth: 660,
      cell: (row) => <span className="whitespace-nowrap tabular-nums">{formatRelativeTime(row.lastTs)}</span>,
      sortValue: (row) => row.lastTs,
    },
  ];

  return (
    <div ref={box}>
      <DataTable
        caption="Models"
        columns={columns}
        rows={rows}
        rowKey={(row) => row.model}
        rowAttrs={(row) => ({ 'data-model': row.model })}
        rowTo={(row) => scope.urlWith({ model: row.model, detail: null })}
        defaultSort={{ id: 'turns', direction: 'desc' }}
        loading={summary.data === undefined && summary.state.error === undefined}
        error={summary.data === undefined ? regionError(summary) : undefined}
        empty={{ title: 'No turns in this range', description: RECORD_HINT }}
        footer={
          scope.model === null ? undefined : (
            <Button variant="ghost" size="sm" onClick={() => scope.setParams({ model: null })}>
              Show all models
            </Button>
          )
        }
      />
    </div>
  );
}
