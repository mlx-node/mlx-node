import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactElement, type RefObject } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';

import { BarList, type BarListRow } from '../components/arc-x/bar-list/bar-list';
import { ButtonLink } from '../components/arc-x/button-link/button-link';
import { ChartCard } from '../components/arc-x/chart-card/chart-card';
import { DataTable, type DataColumn } from '../components/arc-x/data-table/data-table';
import { KeyValue } from '../components/arc-x/key-value/key-value';
import { KpiDrilldown } from '../components/arc-x/kpi-drilldown/kpi-drilldown';
import { LiveDot, type LiveDotProps } from '../components/arc-x/live-dot/live-dot';
import { LiveTurnList } from '../components/arc-x/live-turn-row/live-turn-row';
import { MiniSpark } from '../components/arc-x/mini-spark/mini-spark';
import { ModelBadge } from '../components/arc-x/model-badge/model-badge';
import { NumberValue } from '../components/arc-x/number-value/number-value';
import { Panel } from '../components/arc-x/panel/panel';
import { Placeholder } from '../components/arc-x/placeholder/placeholder';
import { Meter, SplitBar } from '../components/arc-x/split-bar/split-bar';
import { StackedBarChart } from '../components/arc-x/stacked-bar-chart/stacked-bar-chart';
import { StatGroup, StatTile } from '../components/arc-x/stat-tile/stat-tile';
import { SwapText } from '../components/arc-x/swap-text/swap-text';
import { Alert } from '../components/arc/alert/alert';
import { Badge } from '../components/arc/badge/badge';
import { useTurnDrawer } from '../components/turn-drawer';
import { coldWindowTotals } from '../lib/cold-tier';
import { engineSummary, pickProcess, recentSpark } from '../lib/engine-view';
import {
  formatBytes,
  formatBytesParts,
  formatClock,
  formatCount,
  formatCountNoun,
  formatDay,
  formatDuration,
  formatNumber,
  formatPercent,
  formatRate,
  formatRelativeTime,
  formatTps,
} from '../lib/format';
import { healthChecks, type CheckTone } from '../lib/health-checks';
import { useHeld } from '../lib/held';
import { useDataVersion, useRuntimeLive } from '../lib/live';
import { LIVE_GAP_MS, useLiveHistory } from '../lib/live-history';
import {
  isKpiKey,
  KPI_LABEL,
  kpiTiles,
  outcomeText,
  phaseSegments,
  RECORD_HINT,
  turnPhases,
  versusText,
  type KpiKey,
} from '../lib/metrics-view';
import { modelSeries } from '../lib/model-colors';
import { findLocalModel } from '../lib/model-usage';
import { alignedRange } from '../lib/range';
import { OTHER_KEY, otherLabel, pivotByModel, topModels } from '../lib/timeseries';
import type {
  CacheResponse,
  DownloadsResponse,
  EngineStatus,
  MetricsSummary,
  MetricsTimeseries,
  MetricsTurns,
  ModelsResponse,
  RuntimeLive,
  RuntimeProcessLive,
  TurnRow,
} from '../lib/types';
import { LIVE_REFETCH_FAST_MS, useJson } from '../lib/use-api';
import { Page, PageGrid, PageHeader } from '../shell/page';
import { TurnRowFor } from './engine/flight-section';
import { kpiDetail, kpiDetailLoading } from './metrics/kpi-detail';
import { modelPaint, PENDING_DELTA } from './metrics/shared';

const STATUS_DOT: Record<EngineStatus, { tone: LiveDotProps['tone']; pulse: boolean; label: string }> = {
  running: { tone: 'success', pulse: true, label: 'Running' },
  loading: { tone: 'accent', pulse: true, label: 'Loading a model' },
  starting: { tone: 'accent', pulse: true, label: 'Starting' },
  restarting: { tone: 'warning', pulse: true, label: 'Restarting' },
  unresponsive: { tone: 'warning', pulse: false, label: 'Not responding' },
  failed: { tone: 'danger', pulse: false, label: 'Failed' },
  untrusted: { tone: 'danger', pulse: false, label: 'Output not trustworthy' },
  stopped: { tone: 'neutral', pulse: false, label: 'Stopped' },
};

const FAILED_TITLE: Partial<Record<EngineStatus, string>> = {
  failed: 'Inference failed',
  untrusted: 'Output not trustworthy',
};

const CHECK_DOT: Record<CheckTone, LiveDotProps['tone']> = {
  success: 'success',
  warning: 'warning',
  danger: 'danger',
  neutral: 'neutral',
};

const OUTCOME_TONE = { ok: 'neutral', error: 'danger', cancelled: 'warning' } as const;

const SPARK_SAMPLES = 60;

const TILE_ORDER: readonly KpiKey[] = ['turns', 'tokens', 'decode', 'ttft', 'failed'];

const METRICS_LINK: Record<KpiKey, string> = {
  turns: '/metrics?range=24h&kpi=turns',
  tokens: '/metrics?range=24h&kpi=tokens',
  decode: '/metrics?range=24h&kpi=decode',
  ttft: '/metrics?range=24h&kpi=ttft',
  failed: '/metrics?range=24h&tab=failures',
};

function sum(values: readonly number[]): number {
  let total = 0;
  for (const value of values) total += value;
  return total;
}

function isToday(ts: number): boolean {
  const now = new Date();
  const day = new Date(ts);
  return (
    now.getFullYear() === day.getFullYear() && now.getMonth() === day.getMonth() && now.getDate() === day.getDate()
  );
}

/** Tailwind's `@md` container width (`--container-md`). */
const MD_REM = 52;

/** True when the enclosing page container is at least `@md` wide; true while nothing was measured. */
function usePageAtLeastMd(ref: RefObject<Element | null>): boolean {
  const [wide, setWide] = useState(true);
  useLayoutEffect(() => {
    const page = ref.current?.closest('[data-page]');
    if (page === null || page === undefined) return;
    const rem = Number.parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
    const apply = (width: number): void => {
      if (width > 0) setWide(width >= MD_REM * rem);
    };
    const style = getComputedStyle(page);
    apply(
      page.clientWidth - (Number.parseFloat(style.paddingLeft) || 0) - (Number.parseFloat(style.paddingRight) || 0),
    );
    if (typeof ResizeObserver !== 'function') return;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[entries.length - 1];
      if (entry !== undefined) apply(entry.contentRect.width);
    });
    observer.observe(page);
    return () => observer.disconnect();
  }, [ref]);
  return wide;
}

function EnginePanel({
  live,
  latest,
  locals,
  history,
}: {
  live: RuntimeLive | undefined;
  latest: TurnRow | undefined;
  locals: ModelsResponse | undefined;
  history: ReturnType<typeof useLiveHistory>;
}): ReactElement {
  const process = pickProcess(live, null);
  const open = (
    <ButtonLink to="/engine" variant="ghost" size="sm">
      Open Engine
    </ButtonLink>
  );
  if (live === undefined) {
    return (
      <Panel title="Engine" actions={open} busy className="col-span-12 @md:col-span-8">
        <div className="grid gap-2">
          <Placeholder width="60%" />
          <Placeholder width="40%" />
          <Placeholder width="80%" />
        </div>
      </Panel>
    );
  }
  const dot = STATUS_DOT[live.status];
  const failedTitle = FAILED_TITLE[live.status];
  const shimmer = live.status === 'loading' || live.status === 'starting' || live.status === 'restarting';
  const statusRow = (
    <div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1">
      <LiveDot tone={dot.tone} pulse={dot.pulse} label={dot.label} showLabel shimmer={shimmer} />
      {process === null ? null : (
        <span className="text-body text-secondary tabular-nums">
          {`${process.host} · :${String(process.port)} · up `}
          <SwapText text={formatDuration(Math.max(0, live.now - process.startedAt))} granularity="char" />
        </span>
      )}
    </div>
  );

  let body: ReactElement;
  if (failedTitle !== undefined) {
    body = (
      <Alert tone="danger" title={failedTitle}>
        <Link to="/engine" className="link">
          Open Engine
        </Link>
      </Alert>
    );
  } else if (live.status === 'stopped' || process === null) {
    const exit = live.supervisor?.lastExit ?? null;
    body = (
      <div className="grid gap-1">
        <p className="text-body">Inference is not running.</p>
        <p className="text-body text-secondary">Start it from the menu bar, or run mlx serve in a terminal.</p>
        {exit?.verdict === 'crash' ? (
          <p className="text-body text-secondary">{`Last exit: crash at ${formatClock(exit.atMs)}.`}</p>
        ) : null}
      </div>
    );
  } else if (live.status === 'starting' || live.status === 'restarting') {
    body = <p className="text-body text-secondary">Waiting for the inference process.</p>;
  } else if (process.status === 'unresponsive') {
    body = <p className="text-body text-secondary">{`No heartbeat for ${formatDuration(process.ageMs)}.`}</p>;
  } else {
    body = <ProcessBody live={live} process={process} latest={latest} locals={locals} history={history} />;
  }

  return (
    <Panel title="Engine" actions={open} className="col-span-12 @md:col-span-8">
      <div className="grid gap-3">
        {statusRow}
        {live.status === 'loading' && process != null && process.loading.length > 0 ? (
          <p className="text-body text-secondary">{`Loading ${process.loading.map((load) => load.model).join(', ')}…`}</p>
        ) : null}
        {body}
      </div>
    </Panel>
  );
}

function ProcessBody({
  live,
  process,
  latest,
  locals,
  history,
}: {
  live: RuntimeLive;
  process: RuntimeProcessLive;
  latest: TurnRow | undefined;
  locals: ModelsResponse | undefined;
  history: ReturnType<typeof useLiveHistory>;
}): ReactElement {
  const tps = process.decodeTps ?? 0;
  const spark = recentSpark(history, (sample) => sample.outTps, process.pid, SPARK_SAMPLES, LIVE_GAP_MS);
  const inFlight = sum(process.models.map((model) => model.inFlight));
  const queued = sum(process.models.map((model) => model.queued));
  const turns = [...process.turns].sort((a, b) => a.startedAt - b.startedAt);
  const body = useRef<HTMLDivElement>(null);
  const shown = turns.slice(0, usePageAtLeastMd(body) ? 3 : 2);
  const running = process.status === 'running';
  return (
    <>
      <div ref={body} className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <span className="inline-flex min-w-[7ch] items-baseline gap-1.5 font-heading text-hero">
          <span className="tabular-nums" data-stat-value="">
            <NumberValue text={formatTps(tps)} />
          </span>
          <span className="font-sans text-body text-secondary">tok/s</span>
        </span>
        <MiniSpark data={spark} label="Output rate, last minutes" width={96} height={24} />
        <span className="text-body text-secondary tabular-nums">
          <SwapText text={`${formatNumber(inFlight)} in flight · ${formatNumber(queued)} queued`} />
        </span>
      </div>
      {turns.length === 0 ? (
        <p className="text-body text-secondary">
          {latest === undefined ? 'Idle.' : `Idle. Last turn finished ${formatRelativeTime(latest.ts)}.`}
        </p>
      ) : (
        <div className="grid gap-1">
          <LiveTurnList label="Turns in flight">
            {shown.map((turn) => (
              <TurnRowFor key={turn.traceId} turn={turn} now={live.now} live={running} density="compact" />
            ))}
          </LiveTurnList>
          {turns.length > shown.length ? (
            <Link to="/engine" className="link text-body">
              {`+${formatNumber(turns.length - shown.length)} more`}
            </Link>
          ) : null}
        </div>
      )}
      {process.models.length === 0 ? null : (
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <span className="text-body text-secondary">Loaded</span>
          {process.models.map((model) => {
            const local = findLocalModel(locals?.models, model.name);
            return (
              <ModelBadge
                key={model.name}
                name={model.name}
                size="sm"
                to={local === undefined ? undefined : `/models?focus=${encodeURIComponent(local.name)}`}
              />
            );
          })}
        </div>
      )}
    </>
  );
}

export default function Overview(): ReactElement {
  const [params, setSearch] = useSearchParams();
  const navigate = useNavigate();
  const live = useRuntimeLive();
  const history = useLiveHistory();
  const dataVersion = useDataVersion();
  // A data event re-aligns the window; inside one hour the request paths stay equal.
  const range = useMemo(() => alignedRange('24h'), [dataVersion]);
  const coldRange = useMemo(() => alignedRange('30d'), [dataVersion]);
  const q = `from=${String(range.from)}&to=${String(range.to)}`;

  const summaryState = useJson<MetricsSummary>(`/metrics/summary?${q}`, { live: true });
  const summary = useHeld(summaryState);
  const seriesState = useJson<MetricsTimeseries>(`/metrics/timeseries?${q}&bucket=hour&groupBy=model`, { live: true });
  const series = useHeld(seriesState);
  const turns = useJson<MetricsTurns>('/metrics/turns?limit=6', { live: { minIntervalMs: LIVE_REFETCH_FAST_MS } });
  const models = useJson<ModelsResponse>('/models');
  const cache = useJson<CacheResponse>('/cache', { live: { minIntervalMs: LIVE_REFETCH_FAST_MS } });
  const downloads = useJson<DownloadsResponse>('/downloads');
  const drawer = useTurnDrawer(turns.data?.rows ?? []);

  const rawDetail = params.get('detail');
  const detailKey = rawDetail !== null && isKpiKey(rawDetail) ? rawDetail : null;
  const setDetail = (key: string | null): void => {
    setSearch(
      (current) => {
        const next = new URLSearchParams(current);
        if (key === null) next.delete('detail');
        else next.set('detail', key);
        return next;
      },
      { replace: true },
    );
  };

  const [status, setStatus] = useState<{ status: EngineStatus | null; text: string }>({ status: null, text: '' });
  useEffect(() => {
    if (live === undefined || live.status === status.status) return;
    setStatus({ status: live.status, text: engineSummary(live, null) });
  }, [live, status.status]);

  const data = summary.data;
  const tiles = useMemo(
    () =>
      data === undefined
        ? null
        : kpiTiles(data, versusText('24h'), 'median').map((tile) =>
            tile.key === 'decode' || tile.key === 'ttft' ? tile : { ...tile, sampleCount: undefined },
          ),
    [data],
  );
  const top = useMemo(() => (data === undefined ? [] : topModels(data.byModel)), [data]);
  const points = series.data?.points;
  const chart = useMemo(() => {
    const pivot = pivotByModel(points ?? [], range, (p) => p.totals.turns, { empty: 0, models: top, fold: 'sum' });
    const chartSeries = modelSeries(top);
    if (pivot.dropped.length > 0) {
      chartSeries.push({
        key: OTHER_KEY,
        label: otherLabel(pivot.dropped.length),
        color: 'var(--model-other)',
      });
    }
    return { data: pivot.data, series: chartSeries };
  }, [points, range, top]);
  const hourly = (pick: (p: MetricsTimeseries['points'][number]) => number): (number | null)[] | undefined =>
    points === undefined
      ? undefined
      : pivotByModel(points, range, pick, { empty: 0, models: [], fold: 'sum' }).data.map(
          (point) => point.values[OTHER_KEY] ?? 0,
        );

  const checks = healthChecks({ live, totals: data?.totals, cache: cache.data, models: models.data });
  const healthLoading =
    live === undefined &&
    summary.data === undefined &&
    cache.data === undefined &&
    models.data === undefined &&
    summaryState.error === undefined &&
    cache.error === undefined &&
    models.error === undefined;

  const usedRows: BarListRow[] = (data?.byModel ?? [])
    .filter((row) => row.totals.turns > 0)
    .map((row) => ({
      key: row.model,
      label: <ModelBadge name={row.model} size="sm" />,
      text: row.model,
      value: row.totals.turns,
      hint: formatRate(row.totals.turns, data?.totals.turns ?? 0),
      to: `/metrics?range=24h&model=${encodeURIComponent(row.model)}`,
      ...modelPaint(row.model),
    }));

  const companions = models.data?.companions ?? [];
  const modelBytes = [...(models.data?.models ?? []), ...companions].reduce(
    (total, model) => total + model.sizeBytes,
    0,
  );
  const disk = cache.data?.disk;
  const quota = disk?.quotaBytes ?? 0;
  const coldBytes = disk === undefined ? null : formatBytesParts(disk.totalBytes);
  const coldParts = coldBytes === null || coldBytes.value === '—' ? null : coldBytes;
  const coldWindow = coldWindowTotals(cache.data?.trend ?? [], coldRange.starts);
  const lookups = { hits: coldWindow.hits, all: coldWindow.hits + coldWindow.misses };
  const jobs = downloads.data?.jobs ?? [];
  const running = jobs.filter((job) => job.state === 'running').length;
  const done = jobs.filter((job) => job.state === 'done').length;

  const turnColumns: DataColumn<TurnRow>[] = [
    {
      id: 'when',
      header: 'When',
      width: 116,
      cell: (row) => (
        <span className="tabular-nums" title={new Date(row.ts).toISOString()}>
          {isToday(row.ts) ? formatClock(row.ts) : `${formatDay(row.ts)} ${formatClock(row.ts)}`}
        </span>
      ),
    },
    {
      id: 'model',
      header: 'Model',
      flex: true,
      minWidth: 140,
      truncate: true,
      cell: (row) => <ModelBadge name={row.model} size="sm" fit="width" />,
    },
    {
      id: 'result',
      header: 'Result',
      width: 96,
      cell: (row) => (
        <Badge size="sm" tone={OUTCOME_TONE[row.outcome]}>
          {outcomeText(row.outcome)}
        </Badge>
      ),
    },
    {
      id: 'tokens',
      header: 'Tokens in → out',
      align: 'end',
      cell: (row) => (
        <span className="tabular-nums">{`${formatCount(row.promptTokens)} → ${formatCount(row.outputTokens)}`}</span>
      ),
    },
    { id: 'ttft', header: 'First token', align: 'end', minTableWidth: 700, cell: (row) => formatDuration(row.ttftMs) },
    {
      id: 'decode',
      header: 'Decode',
      align: 'end',
      minTableWidth: 660,
      cell: (row) => (row.decodeTps === null ? '—' : `${formatTps(row.decodeTps)} tok/s`),
    },
    {
      id: 'phases',
      header: 'Phases',
      width: 96,
      minTableWidth: 480,
      cell: (row) => {
        const phases = turnPhases(row);
        return (
          <SplitBar
            size="dense"
            legend={false}
            label={`Phases of the turn at ${formatClock(row.ts)}`}
            segments={phaseSegments(phases)}
            total={phases.totalMs}
            formatValue={formatDuration}
          />
        );
      },
    },
  ];

  const turnUrl = (traceId: string): string => {
    const next = new URLSearchParams(params);
    next.set('turn', traceId);
    return `/?${next.toString()}`;
  };

  const summaryTile =
    data === undefined && summaryState.error !== undefined
      ? { loading: false, held: false, error: 'Could not load' }
      : { loading: data === undefined, held: summary.held };

  return (
    <Page busy={summaryState.loading || seriesState.loading}>
      <PageHeader
        title="Overview"
        description="Inference status and the last 24 hours."
        actions={
          models.data !== undefined && models.error === undefined && models.data.models.length === 0 ? (
            <ButtonLink to="/welcome" variant="primary">
              Choose your first model
            </ButtonLink>
          ) : undefined
        }
      />
      <p className="sr-only" aria-live="polite">
        {status.text}
      </p>
      <PageGrid>
        <EnginePanel live={live} latest={turns.data?.rows[0]} locals={models.data} history={history} />
        <Panel title="Health" className="col-span-12 @md:col-span-4">
          <KeyValue
            layout="rows"
            ariaLabel="Health"
            loading={healthLoading}
            items={checks.map((check) => ({
              key: check.key,
              label: check.label,
              value: (
                <Link to={check.to} className="text-foreground hover:underline">
                  <LiveDot tone={CHECK_DOT[check.tone]} label={check.text} showLabel />
                </Link>
              ),
            }))}
          />
        </Panel>
        <KpiDrilldown
          className="col-span-12"
          label="Last 24 hours"
          open={detailKey}
          onOpenChange={setDetail}
          detail={(key) => {
            if (!isKpiKey(key)) return null;
            const footer = (
              <ButtonLink to={METRICS_LINK[key]} variant="ghost" size="sm">
                Open in Metrics
              </ButtonLink>
            );
            const view = tiles?.find((item) => item.key === key);
            if (data === undefined || view === undefined) return kpiDetailLoading(key, '24h', footer);
            return kpiDetail({
              key,
              tile: view,
              rangeKey: '24h',
              range,
              model: null,
              summary: data,
              points: undefined,
              footer,
            });
          }}
        >
          <div className="grid gap-3">
            <StatGroup label="Last 24 hours">
              {TILE_ORDER.map((key) => {
                const view = tiles?.find((item) => item.key === key);
                const label = KPI_LABEL[key];
                const spark =
                  key === 'turns'
                    ? hourly((p) => p.totals.turns)
                    : key === 'tokens'
                      ? hourly((p) => p.totals.outputTokens)
                      : undefined;
                return (
                  <StatTile
                    key={key}
                    label={label}
                    value={view?.value ?? null}
                    unit={view?.unit}
                    delta={view?.delta ?? PENDING_DELTA}
                    hint={view?.hint ?? ''}
                    sampleCount={view?.sampleCount}
                    spark={spark === undefined ? undefined : { data: spark, label: `${label} per hour` }}
                    detailKey={key}
                    {...summaryTile}
                  />
                );
              })}
            </StatGroup>
            <PageGrid>
              <ChartCard
                className="col-span-12 @md:col-span-8"
                title="Turns per hour"
                subtitle="By model · turns"
                height={200}
                state={
                  data === undefined
                    ? summaryState.error !== undefined
                      ? 'error'
                      : 'loading'
                    : data.totals.turns === 0
                      ? 'empty'
                      : series.data === undefined
                        ? seriesState.error !== undefined
                          ? 'error'
                          : 'loading'
                        : summary.held || series.held
                          ? 'held'
                          : 'ready'
                }
                empty={{ title: 'No turns in the last 24 hours', hint: RECORD_HINT }}
                error={
                  summaryState.error !== undefined
                    ? { message: summaryState.error.message, retry: summaryState.reload }
                    : seriesState.error !== undefined
                      ? { message: seriesState.error.message, retry: seriesState.reload }
                      : undefined
                }
                footnote={
                  data === undefined
                    ? undefined
                    : `${formatCountNoun(data.totals.turns, 'turn', 'turns')} · ${formatCountNoun(data.byModel.filter((row) => row.totals.turns > 0).length, 'model', 'models')}`
                }
              >
                <StackedBarChart
                  label="Turns per hour"
                  unit="turns"
                  height={200}
                  data={chart.data}
                  series={chart.series}
                  formatValue={formatNumber}
                  formatTick={formatCount}
                  onSelect={() => navigate('/metrics?range=24h')}
                />
              </ChartCard>
              <Panel
                className="col-span-12 @md:col-span-4"
                title="Models used"
                description="Turns · last 24 hours"
                busy={data === undefined}
              >
                <BarList
                  label="Turns by model"
                  unit="turns"
                  rows={usedRows}
                  limit={5}
                  formatValue={formatNumber}
                  emptyLabel="No turns in the last 24 hours"
                />
              </Panel>
            </PageGrid>
          </div>
        </KpiDrilldown>
        <div className="col-span-12">
          <StatGroup label="Storage">
            <StatTile
              label="Local models"
              value={models.data === undefined ? null : formatCount(models.data.models.length)}
              hint={
                models.data === undefined
                  ? ''
                  : `${formatBytes(modelBytes)} on disk${companions.length > 0 ? ', including companions' : ''}`
              }
              to="/models"
              loading={models.loading}
              error={models.error === undefined ? undefined : 'Failed to load models'}
            />
            <StatTile
              label="Cold cache"
              value={coldParts === null ? null : coldParts.value}
              unit={coldParts === null || coldParts.unit === '' ? undefined : coldParts.unit}
              hint={
                disk === undefined
                  ? ''
                  : `${quota > 0 ? `${formatPercent(Math.min(1, disk.totalBytes / quota))} of ${formatBytes(quota)}` : 'no quota'}${lookups.all > 0 ? ` · hit rate ${formatRate(lookups.hits, lookups.all)}` : ''}`
              }
              footer={
                <Placeholder loading={disk === undefined} height={6} radius="pill">
                  {disk === undefined ? null : (
                    <Meter
                      size="thin"
                      label="Cold cache quota"
                      value={disk.totalBytes}
                      max={quota > 0 ? quota : null}
                      unknownLabel="no quota"
                      formatValue={formatBytes}
                    />
                  )}
                </Placeholder>
              }
              to="/cache"
              loading={cache.loading}
              error={cache.error === undefined || cache.data !== undefined ? undefined : 'Failed to load cache'}
            />
            <StatTile
              label="Active downloads"
              value={downloads.data === undefined ? null : formatCount(running)}
              hint={
                downloads.data === undefined ? (
                  ''
                ) : running > 0 || done > 0 ? (
                  <span>
                    {done > 0 ? `${formatCount(done)} completed · ` : ''}
                    <Link to="/models" className="link">
                      manage
                    </Link>
                  </span>
                ) : (
                  <span>
                    None running ·{' '}
                    <Link to="/models" className="link">
                      install a model
                    </Link>
                  </span>
                )
              }
              loading={downloads.loading}
              error={downloads.error === undefined ? undefined : 'Failed to load downloads'}
            />
          </StatGroup>
        </div>
        <Panel
          className="col-span-12"
          title="Latest turns"
          padding="none"
          actions={
            <ButtonLink to="/metrics?tab=turns" variant="ghost" size="sm">
              All turns
            </ButtonLink>
          }
        >
          <DataTable
            caption="Latest turns"
            density="compact"
            columns={turnColumns}
            rows={turns.data?.rows ?? []}
            rowKey={(row) => row.traceId}
            rowAttrs={(row) => ({ 'data-model': row.model })}
            rowTo={(row) => turnUrl(row.traceId)}
            loading={turns.loading}
            loadingRows={6}
            error={
              turns.error === undefined || turns.data !== undefined
                ? undefined
                : { message: turns.error.message, retry: turns.reload }
            }
            empty={{ title: 'No turns recorded yet', description: RECORD_HINT }}
          />
        </Panel>
      </PageGrid>
      {drawer.element}
    </Page>
  );
}
