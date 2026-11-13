import { useMemo, type ReactElement, type ReactNode } from 'react';

import { ChartCard, type ChartState } from '../../components/arc-x/chart-card/chart-card';
import { DataTable, type DataColumn } from '../../components/arc-x/data-table/data-table';
import { ModelBadge } from '../../components/arc-x/model-badge/model-badge';
import { Panel } from '../../components/arc-x/panel/panel';
import { StatGroup, StatTile } from '../../components/arc-x/stat-tile/stat-tile';
import { TrendChart } from '../../components/arc-x/trend-chart/trend-chart';
import { Badge } from '../../components/arc/badge/badge';
import { flightPoints, historyTotals, memoryPoints, procKeyOf, runtimeBucketWord } from '../../lib/engine-view';
import { formatBytes, formatClock, formatDay, formatDuration, formatNumber, formatPercent } from '../../lib/format';
import { axisNumber } from '../../lib/metrics-view';
import { SERIES_A, SERIES_B, SERIES_REF } from '../../lib/series';
import type { RuntimeLive, RuntimeSeries } from '../../lib/types';
import { PageGrid, PageSection } from '../../shell/page';

const REJECTED_HINT = 'HTTP 429: a chat, model-load, audio or decision queue was full';

type Load = RuntimeSeries['loads'][number];
type Proc = RuntimeSeries['procs'][number];
type LiveProcess = RuntimeLive['processes'][number];

function stamp(ts: number): string {
  return `${formatDay(ts)} ${formatClock(ts)}`;
}

function maxOf(values: readonly (number | null)[]): number | null {
  let best: number | null = null;
  for (const value of values) if (value !== null && (best === null || value > best)) best = value;
  return best;
}

/** The Model loads columns, in priority order: the last one goes first when the card narrows. */
export function loadColumnsOf(): DataColumn<Load>[] {
  return [
    { id: 'when', header: 'When', width: 116, cell: (row) => <span className="tabular-nums">{stamp(row.ts)}</span> },
    {
      id: 'model',
      header: 'Model',
      flex: true,
      minWidth: 132,
      truncate: true,
      cell: (row) => <ModelBadge name={row.model} size="sm" fit="width" />,
    },
    {
      id: 'result',
      header: 'Result',
      width: 88,
      cell: (row) => (
        <Badge size="sm" tone={row.ok ? 'neutral' : 'danger'}>
          {row.ok ? 'Loaded' : 'Failed'}
        </Badge>
      ),
    },
    {
      id: 'took',
      header: 'Took',
      align: 'end',
      width: 80,
      minTableWidth: 420,
      cell: (row) => formatDuration(row.ms),
    },
    { id: 'kind', header: 'Kind', width: 72, minTableWidth: 540, cell: (row) => row.capability ?? '—' },
  ];
}

/** The Processes columns, in priority order; `liveByKey` gives the state of a process that runs now. */
export function procColumnsOf(liveByKey: ReadonlyMap<string, LiveProcess>): DataColumn<Proc>[] {
  return [
    {
      id: 'started',
      header: 'Started',
      width: 112,
      cell: (row) => <span className="tabular-nums">{stamp(row.startedAt)}</span>,
    },
    {
      id: 'ran',
      header: 'Ran for',
      align: 'end',
      width: 88,
      cell: (row) => formatDuration(Math.max(0, (row.exitTs ?? row.lastTs) - row.startedAt)),
    },
    {
      id: 'state',
      header: 'State',
      width: 132,
      cell: (row) => {
        const current = liveByKey.get(row.procKey);
        if (current?.status === 'running') {
          return (
            <Badge size="sm" tone="success">
              Running
            </Badge>
          );
        }
        if (current?.status === 'unresponsive') {
          return (
            <Badge size="sm" tone="warning">
              Not responding
            </Badge>
          );
        }
        return row.exitTs !== null ? (
          <Badge size="sm" tone="neutral">
            Exited
          </Badge>
        ) : (
          <Badge size="sm" tone="danger">
            No exit record
          </Badge>
        );
      },
    },
    {
      id: 'host',
      header: 'Host',
      width: 84,
      minTableWidth: 420,
      cell: (row) => (
        <span title={row.port === null ? undefined : `${row.host ?? '—'} · port ${String(row.port)}`}>
          {row.host ?? '—'}
        </span>
      ),
    },
    {
      id: 'version',
      header: 'Version',
      width: 76,
      minTableWidth: 540,
      truncate: true,
      cell: (row) => (row.version === null ? '—' : <span title={`v${row.version}`}>{`v${row.version}`}</span>),
    },
    {
      id: 'chip',
      header: 'Chip',
      width: 128,
      minTableWidth: 700,
      truncate: true,
      cell: (row) => (row.chip === null ? '—' : <span title={row.chip}>{row.chip}</span>),
    },
  ];
}

export function HistorySection({
  series,
  state,
  error,
  live,
  starts,
  rangeControl,
}: {
  series: RuntimeSeries | undefined;
  state: ChartState;
  error?: { message: string; retry: () => void };
  live: RuntimeLive | undefined;
  starts: readonly number[];
  rangeControl: ReactNode;
}): ReactElement {
  const alive = useMemo(
    () => new Set((live?.processes ?? []).filter((p) => p.status !== 'dead').map((p) => procKeyOf(p))),
    [live],
  );
  const now = live?.now ?? Date.now();
  const totals = series === undefined ? null : historyTotals(series, now, alive);
  const loading = series === undefined && error === undefined;
  const tileError = series === undefined && error !== undefined ? 'Could not load' : undefined;
  const held = state === 'held';
  const word = series === undefined ? 'minute' : runtimeBucketWord(series.bucketMs);

  const newestProc = series === undefined ? undefined : [...series.procs].sort((a, b) => b.startedAt - a.startedAt)[0];
  const memory = useMemo(
    () => (series === undefined ? null : memoryPoints(series, starts, newestProc?.wiredLimitBytes ?? null)),
    [series, starts, newestProc],
  );
  const flight = useMemo(() => (series === undefined ? [] : flightPoints(series, starts)), [series, starts]);
  const noSamples = series !== undefined && series.points.every((point) => point.samples === 0);

  const loads = useMemo(() => [...(series?.loads ?? [])].sort((a, b) => b.ts - a.ts), [series]);
  const procs = useMemo(() => [...(series?.procs ?? [])].sort((a, b) => b.startedAt - a.startedAt), [series]);
  const liveByKey = new Map((live?.processes ?? []).map((p) => [procKeyOf(p), p]));

  const loadColumns = loadColumnsOf();
  const procColumns = procColumnsOf(liveByKey);

  const busyText =
    totals === null || totals.coveredMs <= 0 ? null : formatPercent(Math.min(1, totals.busyMs / totals.coveredMs));

  return (
    <PageSection title="History" description="All inference processes on this machine." actions={rangeControl}>
      <div className="grid gap-3">
        <StatGroup label="History">
          <StatTile
            label="Busy time"
            value={busyText}
            hint={
              totals === null || totals.coveredMs <= 0
                ? undefined
                : `${formatDuration(totals.busyMs)} of ${formatDuration(totals.coveredMs)} observed`
            }
            loading={loading}
            held={held}
            error={tileError}
          />
          <StatTile
            label="Peak in flight"
            value={totals?.peakInFlight == null ? null : formatNumber(totals.peakInFlight)}
            hint={totals === null ? undefined : `peak queue ${formatNumber(totals.peakQueued)}`}
            loading={loading}
            held={held}
            error={tileError}
          />
          <StatTile
            label="Rejected requests"
            value={totals === null ? null : formatNumber(totals.rejected)}
            hint={
              totals === null ? undefined : (
                <span className="flex min-w-0 items-center gap-1.5" title={REJECTED_HINT}>
                  {totals.rejected > 0 ? (
                    <Badge size="sm" tone="warning" className="shrink-0">
                      Queue full
                    </Badge>
                  ) : null}
                  <span className="min-w-0 truncate">{REJECTED_HINT}</span>
                </span>
              )
            }
            loading={loading}
            held={held}
            error={tileError}
          />
          <StatTile
            label="Model loads"
            value={totals === null ? null : formatNumber(totals.loads)}
            hint={
              totals === null || totals.loads === 0
                ? undefined
                : [
                    `slowest ${formatDuration(totals.slowestLoadMs)}`,
                    totals.failedLoads > 0 ? `${formatNumber(totals.failedLoads)} failed` : null,
                  ]
                    .filter((part): part is string => part !== null)
                    .join(' · ')
            }
            loading={loading}
            held={held}
            error={tileError}
          />
          <StatTile
            label="Process starts"
            value={totals === null ? null : formatNumber(totals.starts)}
            hint={totals === null ? undefined : `${formatNumber(totals.unclean)} ended without a clean exit`}
            loading={loading}
            held={held}
            error={tileError}
          />
        </StatGroup>
        <PageGrid>
          <ChartCard
            className="col-span-12 @md:col-span-7"
            title="Memory over time"
            subtitle={`Mean per ${word} · ${memory?.unit ?? 'GB'}`}
            height={200}
            state={state === 'ready' || state === 'held' ? (noSamples ? 'empty' : state) : state}
            empty={{
              title: 'No runtime samples in this range',
              hint: 'Samples are written while inference runs. They are kept 14 days.',
            }}
            error={error}
            footnote={
              series === undefined
                ? undefined
                : `Peak ${formatBytes(maxOf(series.points.map((p) => p.memPeakMax)))} · Node RSS up to ${formatBytes(maxOf(series.points.map((p) => p.rssMax)))}`
            }
          >
            <TrendChart
              label="Memory over time"
              unit={memory?.unit ?? 'GB'}
              height={200}
              data={memory?.data ?? []}
              series={[
                { key: 'active', label: 'MLX active', color: SERIES_A },
                { key: 'cache', label: 'MLX cache', color: SERIES_B },
                { key: 'limit', label: 'Wired limit', ...SERIES_REF },
              ]}
              formatValue={axisNumber}
            />
          </ChartCard>
          <ChartCard
            className="col-span-12 @md:col-span-5"
            title="Turns in flight and queued"
            subtitle={`Highest per ${word} · turns`}
            height={200}
            state={state === 'ready' || state === 'held' ? (noSamples ? 'empty' : state) : state}
            empty={{
              title: 'No runtime samples in this range',
              hint: 'Samples are written while inference runs. They are kept 14 days.',
            }}
            error={error}
          >
            <TrendChart
              label="Turns in flight and queued"
              unit="turns"
              height={200}
              data={flight}
              series={[
                { key: 'inFlight', label: 'In flight', color: SERIES_A },
                { key: 'queued', label: 'Queued', color: SERIES_B },
              ]}
              formatValue={formatNumber}
              formatTick={formatNumber}
            />
          </ChartCard>
          <Panel className="col-span-12 @md:col-span-6" title="Model loads" padding="none">
            <DataTable
              caption="Model loads"
              density="compact"
              columns={loadColumns}
              rows={loads}
              rowKey={(row) => `${row.procKey}-${String(row.ts)}-${row.model}`}
              loading={loading}
              error={series === undefined ? error : undefined}
              empty={{
                title: 'No model was loaded in this range',
                description: 'Loads are recorded when a model is first used.',
              }}
            />
          </Panel>
          <Panel className="col-span-12 @md:col-span-6" title="Processes" padding="none">
            <DataTable
              caption="Processes"
              density="compact"
              columns={procColumns}
              rows={procs}
              rowKey={(row) => row.procKey}
              rowAttrs={(row) => ({ 'data-pid': String(row.pid) })}
              loading={loading}
              error={series === undefined ? error : undefined}
              empty={{
                title: 'No inference process in this range',
                description: 'Processes are recorded while inference runs.',
              }}
            />
          </Panel>
        </PageGrid>
      </div>
    </PageSection>
  );
}
