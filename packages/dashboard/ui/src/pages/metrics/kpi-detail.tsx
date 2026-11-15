import type { ReactElement, ReactNode } from 'react';

import { BarList, type BarListRow } from '../../components/arc-x/bar-list/bar-list';
import { ButtonLink } from '../../components/arc-x/button-link/button-link';
import type { ChartPoint } from '../../components/arc-x/chart/types';
import { KeyValue } from '../../components/arc-x/key-value/key-value';
import type { KpiDetail } from '../../components/arc-x/kpi-drilldown/kpi-drilldown';
import { ModelBadge } from '../../components/arc-x/model-badge/model-badge';
import { Placeholder } from '../../components/arc-x/placeholder/placeholder';
import { TrendChart } from '../../components/arc-x/trend-chart/trend-chart';
import { Button } from '../../components/arc/button/button';
import { formatCount, formatDuration, formatNumber, formatTps } from '../../lib/format';
import {
  axisNumber,
  bucketWord,
  kpiByModel,
  kpiDescription,
  KPI_LABEL,
  kpiEmpty,
  kpiPointValue,
  kpiSeries,
  previousLabel,
  previousRange,
  timeAxis,
  type KpiKey,
  type KpiTileView,
} from '../../lib/metrics-view';
import { modelColor } from '../../lib/model-colors';
import type { AlignedRange, RangeKey } from '../../lib/range';
import { pivotMeasures } from '../../lib/timeseries';
import type { MetricsSummary, MetricsTimeseries, Quantiles } from '../../lib/types';
import { useJson } from '../../lib/use-api';
import { modelPaint, useReportBusy } from './shared';

type TsPoint = MetricsTimeseries['points'][number];

export interface KpiDetailInput {
  key: KpiKey;
  tile: KpiTileView;
  rangeKey: RangeKey;
  range: AlignedRange;
  model: string | null;
  summary: MetricsSummary;
  /** The page's `groupBy=none` series; undefined: the body fetches it while open. */
  points: readonly TsPoint[] | undefined;
  /** The window `points` answers (a held answer keeps the window it was asked with). Default `range`. */
  pointsRange?: AlignedRange;
  /** With `points` given: the page's query has not answered yet. */
  pointsLoading?: boolean;
  /** With `points` given: the page's query failed and holds nothing. */
  pointsError?: { message: string; retry: () => void };
  footer: ReactNode;
}

/** `key` is reserved by React, so the bodies take the KPI as `kpi`. */
type BodyProps = Omit<KpiDetailInput, 'key'> & { kpi: KpiKey };

function bodyProps({ key, ...rest }: KpiDetailInput): BodyProps {
  return { ...rest, kpi: key };
}

/** The panel of one KPI tile: the trend with the previous period as a ghost, and the measure by model. */
export function kpiDetail(input: KpiDetailInput): KpiDetail {
  return {
    label: input.tile.label,
    value: input.tile.value,
    unit: input.tile.unit,
    delta: input.tile.delta,
    sampleCount: input.tile.sampleCount,
    description: kpiDescription(input.key, input.rangeKey),
    children:
      input.points === undefined ? (
        <FetchedBody {...bodyProps(input)} />
      ) : (
        <DetailBody {...bodyProps(input)} points={input.points} pointsLoading={input.pointsLoading === true} />
      ),
    footer: input.footer,
  };
}

function noneQuery(range: AlignedRange, model: string | null): string {
  const base = `/metrics/timeseries?from=${String(range.from)}&to=${String(range.to)}&bucket=${range.bucket}&groupBy=none`;
  return model === null ? base : `${base}&model=${encodeURIComponent(model)}`;
}

function FetchedBody(props: BodyProps): ReactElement {
  const current = useJson<MetricsTimeseries>(noneQuery(props.range, props.model));
  useReportBusy(current);
  return <DetailBody {...props} points={current.data?.points ?? []} pointsLoading={current.data === undefined} />;
}

function formatFor(key: KpiKey): (value: number) => string {
  if (key === 'decode') return formatTps;
  if (key === 'ttft') return formatDuration;
  return formatNumber;
}

function DetailBody(props: BodyProps & { points: readonly TsPoint[]; pointsLoading: boolean }): ReactElement {
  const failed = props.pointsError;
  const { kpi: key, model } = props;
  const range = props.pointsRange ?? props.range;
  const before = previousRange(props.range);
  const previous = useJson<MetricsTimeseries>(noneQuery(before, model));
  useReportBusy(previous);
  const sameWindow = range.from === props.range.from && range.to === props.range.to;
  const ghostRaw = previous.data === undefined || !sameWindow ? null : kpiSeries(key, previous.data.points, before);
  const data: ChartPoint[] = pivotMeasures(
    props.points,
    range,
    { current: (p) => kpiPointValue(key, p) },
    kpiEmpty(key),
  );
  let unit =
    props.tile.unit ?? (key === 'turns' ? 'turns' : key === 'tokens' ? 'tokens' : key === 'failed' ? 'turns' : '');
  let div = 1;
  if (key === 'ttft') {
    let max = 0;
    for (const point of data) max = Math.max(max, point.values.current ?? 0);
    for (const value of ghostRaw ?? []) max = Math.max(max, value ?? 0);
    const axis = timeAxis(max);
    unit = axis.unit;
    div = axis.div;
  }
  const scaled =
    div === 1
      ? data
      : data.map((point) => ({
          ...point,
          values: { current: point.values.current === null ? null : point.values.current / div },
        }));
  const ghost =
    ghostRaw === null
      ? undefined
      : {
          label: previousLabel(props.rangeKey),
          values: ghostRaw.map((value) => (value === null ? null : value / div)),
        };
  const empty = props.summary.totals.turns === 0;
  return (
    <>
      <section className="grid content-start gap-2" aria-label="Trend">
        <h4 className="text-body font-medium">{`Per ${bucketWord(range.bucket)} · ${unit}`}</h4>
        {failed !== undefined ? (
          <div className="grid place-content-center justify-items-center gap-2" style={{ height: 180 }} role="alert">
            <span className="text-body text-secondary">{failed.message}</span>
            <Button variant="secondary" size="sm" onClick={failed.retry}>
              Try again
            </Button>
          </div>
        ) : props.pointsLoading ? (
          <Placeholder height={180} radius="control" />
        ) : (
          <TrendChart
            label={props.tile.label}
            unit={unit}
            height={180}
            curve="linear"
            data={scaled}
            series={[
              {
                key: 'current',
                label: props.tile.label,
                color: model === null ? 'var(--series-1)' : modelColor(model).color,
              },
            ]}
            ghost={ghost}
            formatValue={key === 'ttft' ? axisNumber : formatFor(key)}
            formatTick={key === 'decode' || key === 'ttft' ? axisNumber : formatCount}
            emptyLabel={empty ? 'No turns in this range' : undefined}
          />
        )}
      </section>
      <ByModel {...props} />
    </>
  );
}

function quantileItems(label: string, q: Quantiles, format: (value: number) => string) {
  const text = (value: number | null): string | null => (value === null ? null : format(value));
  return [
    { key: 'p05', label: `${label} p05`, value: text(q.p05), numeric: true },
    { key: 'p50', label: `${label} median`, value: text(q.p50), numeric: true },
    { key: 'p95', label: `${label} p95`, value: text(q.p95), numeric: true },
    { key: 'n', label: 'Turns measured', value: formatNumber(q.n), numeric: true },
  ];
}

function ByModel(props: BodyProps): ReactElement {
  const { kpi: key, model, summary } = props;
  if (model !== null) {
    const row = summary.byModel.find((item) => item.model === model);
    const items =
      row === undefined
        ? [{ key: 'none', label: 'Turns', value: null }]
        : key === 'decode'
          ? quantileItems('Decode', row.latency.decodeTps, (v) => `${formatTps(v)} tok/s`)
          : key === 'ttft'
            ? quantileItems('First token', row.latency.ttftMs, formatDuration)
            : [
                { key: 'turns', label: 'Turns', value: formatNumber(row.totals.turns), numeric: true },
                { key: 'ok', label: 'Completed', value: formatNumber(row.totals.ok), numeric: true },
                { key: 'error', label: 'Failed', value: formatNumber(row.totals.error), numeric: true },
                { key: 'output', label: 'Output tokens', value: formatNumber(row.totals.outputTokens), numeric: true },
              ];
    return (
      <section className="grid content-start gap-2" aria-label={model}>
        <h4 className="text-body font-medium">{model}</h4>
        <KeyValue layout="rows" items={items} ariaLabel={`${model} numbers`} />
      </section>
    );
  }
  const format = formatFor(key);
  const rows: BarListRow[] = kpiByModel(key, summary.byModel).map((row) => {
    return {
      key: row.model,
      label: <ModelBadge name={row.model} size="sm" />,
      text: row.model,
      value: row.value,
      samples: row.samples,
      ...modelPaint(row.model),
    };
  });
  return (
    <section className="grid content-start gap-2" aria-label="By model">
      <h4 className="text-body font-medium">By model</h4>
      <BarList
        label={`${props.tile.label} by model`}
        unit={key === 'ttft' ? '' : (props.tile.unit ?? '')}
        rows={rows}
        limit={6}
        formatValue={format}
        lowSampleBelow={key === 'decode' || key === 'ttft' ? 5 : undefined}
        emptyLabel="No turns in this range"
      />
    </section>
  );
}

export function ShowLink({ to, children }: { to: string; children: ReactNode }): ReactElement {
  return (
    <ButtonLink to={to} variant="ghost" size="sm">
      {children}
    </ButtonLink>
  );
}

/** The panel of a known key whose summary is still loading: it opens at once (a deep link keeps `?detail=`). */
export function kpiDetailLoading(key: KpiKey, rangeKey: RangeKey, footer: ReactNode): KpiDetail {
  return {
    label: KPI_LABEL[key],
    value: null,
    description: kpiDescription(key, rangeKey),
    children: <Placeholder height={180} radius="control" />,
    footer,
  };
}
