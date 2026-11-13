import { useEffect, useState, type ReactElement } from 'react';
import { Link } from 'react-router-dom';

import { BarList, type BarListRow } from '../../components/arc-x/bar-list/bar-list';
import { ChartCard } from '../../components/arc-x/chart-card/chart-card';
import { DataTable, type DataColumn } from '../../components/arc-x/data-table/data-table';
import { ModelBadge } from '../../components/arc-x/model-badge/model-badge';
import { Panel } from '../../components/arc-x/panel/panel';
import { StackedBarChart } from '../../components/arc-x/stacked-bar-chart/stacked-bar-chart';
import { StatGroup, StatTile } from '../../components/arc-x/stat-tile/stat-tile';
import SegmentedControl, { type Segment } from '../../components/arc/segmented-control/segmented-control';
import { formatClock, formatCount, formatDay, formatDuration, formatNumber, formatRate } from '../../lib/format';
import { bucketWord, errorKindText, failPhaseText, failureLabel, versusText } from '../../lib/metrics-view';
import { OUTCOME_SERIES } from '../../lib/series';
import { pivotMeasures } from '../../lib/timeseries';
import type { MetricsSummary, MetricsTimeseries, MetricsTotals, MetricsTurns, TurnRow } from '../../lib/types';
import { LIVE_REFETCH_FAST_MS, useJson } from '../../lib/use-api';
import { PageGrid } from '../../shell/page';
import {
  RefreshAlert,
  regionError,
  regionState,
  tileState,
  useHeldQuery,
  useMetricsScope,
  useReportBusy,
  withModel,
} from './shared';

const OUTCOME_OPTIONS: Segment[] = [
  { value: 'error', label: 'Failed' },
  { value: 'cancelled', label: 'Cancelled' },
];

function share(totals: MetricsTotals | undefined, pick: (t: MetricsTotals) => number): number | null {
  return totals === undefined || totals.turns <= 0 ? null : pick(totals) / totals.turns;
}

function whenText(ts: number): string {
  const today = new Date();
  const day = new Date(ts);
  const same =
    today.getFullYear() === day.getFullYear() &&
    today.getMonth() === day.getMonth() &&
    today.getDate() === day.getDate();
  return same ? formatClock(ts) : `${formatDay(ts)} ${formatClock(ts)}`;
}

export function FailuresTab(): ReactElement {
  const scope = useMetricsScope();
  const { q, range, model, rangeKey } = scope;
  const summary = useHeldQuery<MetricsSummary>(`/metrics/summary?${q}`);
  const noneTs = useHeldQuery<MetricsTimeseries>(
    `/metrics/timeseries?${q}&bucket=${range.bucket}&groupBy=${model === null ? 'none' : 'model'}`,
    { range },
  );
  const tsRange = noneTs.range ?? range;
  const [outcome, setOutcome] = useState<'error' | 'cancelled'>('error');
  const turns = useJson<MetricsTurns>(`/metrics/turns?${q}&outcome=${outcome}&limit=200`, {
    live: { minIntervalMs: LIVE_REFETCH_FAST_MS },
  });
  useReportBusy(turns);
  const { reportTurns } = scope;
  useEffect(() => {
    if (turns.data !== undefined) reportTurns(turns.data.rows);
  }, [turns.data, reportTurns]);

  const totals = summary.data?.totals;
  const previous = summary.data?.previous?.totals;
  const versus = versusText(rangeKey);
  const tile = tileState(summary);

  const chartData = pivotMeasures(
    noneTs.data?.points ?? [],
    tsRange,
    { error: (p) => p.totals.error, cancelled: (p) => p.totals.cancelled },
    0,
  );
  const failedSum = totals === undefined ? 0 : totals.error + totals.cancelled;

  const stopRows: BarListRow[] = (summary.data?.failures ?? []).map((row, index) => ({
    key: `${row.outcome}-${row.errorKind ?? 'none'}-${row.failPhase ?? 'none'}-${String(index)}`,
    label: failureLabel(row),
    text: failureLabel(row),
    value: row.turns,
    color: row.outcome === 'error' ? 'var(--model-8)' : 'var(--model-other)',
    pattern: row.outcome === 'error' ? 'solid' : 'alt',
  }));
  const finishRows: BarListRow[] = (summary.data?.finishReasons ?? []).map((row) => ({
    key: row.reason,
    label: <span className="font-mono">{row.reason}</span>,
    text: row.reason,
    value: row.turns,
  }));

  const columns: DataColumn<TurnRow>[] = [
    {
      id: 'when',
      header: 'When',
      width: 112,
      cell: (row) => (
        <span className="whitespace-nowrap tabular-nums" title={new Date(row.ts).toISOString()}>
          {whenText(row.ts)}
        </span>
      ),
    },
    {
      id: 'model',
      header: 'Model',
      width: '36%',
      cell: (row) => <ModelBadge name={row.model} size="sm" />,
      truncate: true,
    },
    {
      id: 'kind',
      header: 'Kind',
      truncate: true,
      cell: (row) => {
        const text = errorKindText(row.errorKind) ?? '—';
        return <span title={text}>{text}</span>;
      },
    },
    {
      id: 'phase',
      header: 'Phase',
      truncate: true,
      minTableWidth: 560,
      cell: (row) => {
        const text = failPhaseText(row.failPhase) ?? '—';
        return <span title={text}>{text}</span>;
      },
    },
    {
      id: 'out',
      header: 'Out',
      align: 'end',
      width: 64,
      cell: (row) => formatCount(row.outputTokens),
      minTableWidth: 680,
    },
    {
      id: 'duration',
      header: 'Duration',
      align: 'end',
      cell: (row) =>
        row.durationMs === null && row.preDispatchMs === null
          ? '—'
          : formatDuration((row.preDispatchMs ?? 0) + (row.durationMs ?? 0)),
    },
  ];

  const tableRows = turns.data?.rows ?? [];
  const tableTotal = turns.data?.total ?? 0;

  return (
    <div className="grid gap-3 pt-3">
      <RefreshAlert queries={[summary, noneTs]} />
      <StatGroup label="Reliability">
        <StatTile
          label="Failed turns"
          value={totals === undefined ? null : formatCount(totals.error)}
          hint={totals === undefined ? undefined : `${formatRate(totals.error, totals.turns)} of turns`}
          delta={{
            current: share(totals, (t) => t.error),
            previous: share(previous, (t) => t.error),
            goodWhen: 'down',
            versus,
          }}
          sampleCount={totals?.turns}
          {...tile}
        />
        <StatTile
          label="Cancelled turns"
          value={totals === undefined ? null : formatCount(totals.cancelled)}
          hint={totals === undefined ? undefined : `${formatRate(totals.cancelled, totals.turns)} of turns`}
          delta={{
            current: share(totals, (t) => t.cancelled),
            previous: share(previous, (t) => t.cancelled),
            goodWhen: 'neutral',
            versus,
          }}
          {...tile}
        />
        <StatTile
          label="Completed"
          value={totals === undefined ? null : formatCount(totals.ok)}
          hint={totals === undefined ? undefined : `${formatRate(totals.ok, totals.turns)} of turns`}
          {...tile}
        />
      </StatGroup>
      <PageGrid>
        <ChartCard
          className="col-span-12 @md:col-span-7"
          title="Failed and cancelled turns"
          subtitle={withModel(`Per ${bucketWord(tsRange.bucket)} · turns`, model)}
          height={200}
          state={summary.data === undefined ? regionState(summary, false) : regionState(noneTs, failedSum === 0)}
          empty={{ title: 'No failed or cancelled turns', hint: 'Nothing went wrong in this range.' }}
          error={regionError(summary.data === undefined ? summary : noneTs)}
        >
          <StackedBarChart
            label="Failed and cancelled turns"
            unit="turns"
            height={200}
            data={chartData}
            series={OUTCOME_SERIES}
            formatValue={formatNumber}
            formatTick={formatCount}
            onSelect={() => scope.setParams({ tab: 'turns', outcome: 'error' })}
          />
        </ChartCard>
        <Panel className="col-span-12 @md:col-span-5" title="Where turns stop" busy={summary.data === undefined}>
          <BarList
            label="Where turns stop"
            unit="turns"
            rows={stopRows}
            limit={8}
            emptyLabel="No failed or cancelled turns"
          />
        </Panel>
        <Panel
          className="col-span-12 @md:col-span-5"
          title="How turns end"
          description="Finish reason of every turn."
          busy={summary.data === undefined}
        >
          <div className="grid gap-2">
            <BarList label="How turns end" unit="turns" rows={finishRows} emptyLabel="No turns in this range" />
            <p className="text-caption text-secondary">
              <span className="font-mono">length</span> means the answer hit the output limit.
            </p>
          </div>
        </Panel>
        <Panel
          className="col-span-12 @md:col-span-7"
          title={outcome === 'error' ? 'Failed turns' : 'Cancelled turns'}
          padding="none"
          actions={
            <SegmentedControl
              label="Outcome"
              options={OUTCOME_OPTIONS}
              value={outcome}
              onValueChange={(value) => {
                if (value === 'error' || value === 'cancelled') setOutcome(value);
              }}
            />
          }
        >
          <DataTable
            caption={outcome === 'error' ? 'Failed turns' : 'Cancelled turns'}
            columns={columns}
            rows={tableRows}
            rowKey={(row) => row.traceId}
            rowTo={(row) => scope.urlWith({ turn: row.traceId })}
            density="compact"
            loading={turns.loading}
            error={
              turns.error === undefined || turns.data !== undefined
                ? undefined
                : { message: turns.error.message, retry: turns.reload }
            }
            empty={{
              title: outcome === 'error' ? 'No failed turns in this range' : 'No cancelled turns in this range',
              description: 'Nothing went wrong in this range.',
            }}
            footer={
              tableTotal > tableRows.length
                ? `Showing the newest ${formatNumber(tableRows.length)} of ${formatNumber(tableTotal)}`
                : undefined
            }
          />
        </Panel>
      </PageGrid>
      <p className="text-caption text-secondary">
        Requests turned away because a queue was full never became turns. They are counted on the{' '}
        <Link to="/engine" className="link">
          Engine page
        </Link>
        .
      </p>
    </div>
  );
}
