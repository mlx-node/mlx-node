import { ExternalLink } from 'lucide-react';
import { useLayoutEffect, useRef, useState, type ReactElement, type ReactNode, type RefObject } from 'react';
import { Link } from 'react-router-dom';

import type { ChartPoint, ChartSeries } from '../../components/arc-x/chart/types';
import { CompactField } from '../../components/arc-x/compact-field/compact-field';
import { ConfirmPopover } from '../../components/arc-x/confirm-popover/confirm-popover';
import { DataTable, type DataColumn } from '../../components/arc-x/data-table/data-table';
import { KeyValue, type KeyValueItem } from '../../components/arc-x/key-value/key-value';
import { LiveDot } from '../../components/arc-x/live-dot/live-dot';
import { PathText } from '../../components/arc-x/path-text/path-text';
import { Placeholder } from '../../components/arc-x/placeholder/placeholder';
import { Rule } from '../../components/arc-x/rule/rule';
import { Meter } from '../../components/arc-x/split-bar/split-bar';
import { StackedBarChart } from '../../components/arc-x/stacked-bar-chart/stacked-bar-chart';
import { StatGroup, StatTile } from '../../components/arc-x/stat-tile/stat-tile';
import { TrendChart } from '../../components/arc-x/trend-chart/trend-chart';
import { Alert } from '../../components/arc/alert/alert';
import { Badge } from '../../components/arc/badge/badge';
import { Button } from '../../components/arc/button/button';
import { CopyButton } from '../../components/arc/copy-button/copy-button';
import { Drawer, DrawerContent, useDrawerItem } from '../../components/arc/drawer/drawer';
import { Select } from '../../components/arc/select/select';
import { hasUpdate } from '../../lib/catalog';
import {
  formatBytes,
  formatClock,
  formatCount,
  formatCountNoun,
  formatDay,
  formatDayLong,
  formatDuration,
  formatDurationParts,
  formatNumber,
  formatRate,
  formatRelativeTime,
  formatTps,
} from '../../lib/format';
import { useHeld } from '../../lib/held';
import { truncateMiddle } from '../../lib/labels';
import { axisNumber, failPhaseText, outcomeText } from '../../lib/metrics-view';
import { modelColor } from '../../lib/model-colors';
import { usageKeys, type ModelUsage } from '../../lib/model-usage';
import type { AlignedRange } from '../../lib/range';
import {
  LOW_SAMPLE_TAIL_N,
  type CatalogItem,
  type LocalModel,
  type MetricsSummary,
  type MetricsTimeseries,
  type MetricsTurns,
  type RuntimeLive,
  type TurnRow,
} from '../../lib/types';
import { LIVE_REFETCH_FAST_MS, useJson } from '../../lib/use-api';
import { deleteContent, LiveBadge, liveStateOf, usageKeyOf } from './local-models';

export interface ModelDrawerProps {
  /** The folder named by ?focus=, or null when the drawer is closed. */
  model: LocalModel | null;
  usage: ReadonlyMap<string, ModelUsage>;
  catalog: readonly CatalogItem[];
  remoteRevisions: ReadonlyMap<string, string | null>;
  live: RuntimeLive | undefined;
  range: AlignedRange;
  onClose: () => void;
  onDeleted: (name: string) => void;
  /** Set by the page when the model shown was deleted: focus then goes to the scroll root, its row is gone. */
  deletedRef: RefObject<boolean>;
}

const NO_TURNS = 'No recorded turns for this model in the last 30 days.';
const NO_TURNS_HINT = 'Turns are recorded when the model serves a request through mlx serve, mlx launch or the app.';

/** The row of the table for `name`, else the element that opened the drawer, else the scroll root. */
function returnTarget(name: string | undefined, opener: HTMLElement | null): HTMLElement | null {
  for (const row of document.querySelectorAll<HTMLElement>('tr[data-model]')) {
    if (row.getAttribute('data-model') === name) return row;
  }
  if (opener !== null && opener.isConnected) return opener;
  return document.querySelector<HTMLElement>('[data-scroll-root]');
}

/** The right drawer of one checkpoint folder: disk, what is loaded right now, 30-day usage and the delete. */
export function ModelDrawer(props: ModelDrawerProps): ReactElement {
  const shown = useDrawerItem(props.model) ?? null;
  const open = props.model !== null;
  // Opened by the URL, not a trigger: Radix has nothing to return focus to, so the drawer remembers the opener.
  const opener = useRef<HTMLElement | null>(null);
  useLayoutEffect(() => {
    if (!open) return;
    const active = document.activeElement;
    opener.current = active instanceof HTMLElement && active !== document.body ? active : null;
  }, [open]);
  return (
    <Drawer
      open={open}
      onOpenChange={(next) => {
        if (!next) props.onClose();
      }}
    >
      <DrawerContent
        side="right"
        title={shown === null ? 'Model' : truncateMiddle(shown.name, 40)}
        description={shown === null ? undefined : `${shown.modelType} · ${shown.quant ?? 'full precision'}`}
        onCloseAutoFocus={(event) => {
          event.preventDefault();
          if (props.deletedRef.current === true) {
            props.deletedRef.current = false;
            document.querySelector<HTMLElement>('[data-scroll-root]')?.focus({ preventScroll: true });
            return;
          }
          returnTarget(shown?.name, opener.current)?.focus({ preventScroll: true });
        }}
      >
        {shown === null ? null : <ModelDrawerBody key={shown.name} {...props} model={shown} />}
      </DrawerContent>
    </Drawer>
  );
}

function Heading({ children, actions }: { children: string; actions?: ReactNode }): ReactElement {
  return (
    <div className="flex min-w-0 flex-wrap items-center justify-between gap-2">
      <h3 className="m-0 text-strong font-medium">{children}</h3>
      {actions}
    </div>
  );
}

function ChartBlock(props: { title: string; unit: string; height: number; loading: boolean; children: ReactNode }) {
  return (
    <div className="grid min-w-0 gap-1.5">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-body font-medium">{props.title}</span>
        <span className="text-caption text-secondary">{props.unit}</span>
      </div>
      {props.loading ? <Placeholder height={props.height} radius="panel" /> : props.children}
    </div>
  );
}

interface DayRow {
  ok: number;
  failed: number;
  turns: number;
  decode: MetricsTimeseries['points'][number]['decodeTps'] | null;
  mtpTurns: number;
  mtpSum: number;
}

function byDay(points: MetricsTimeseries['points'] | undefined, key: string): Map<number, DayRow> {
  const days = new Map<number, DayRow>();
  for (const point of points ?? []) {
    if (point.model !== key) continue;
    const row = days.get(point.t) ?? { ok: 0, failed: 0, turns: 0, decode: null, mtpTurns: 0, mtpSum: 0 };
    row.ok += point.totals.ok;
    row.failed += point.totals.error + point.totals.cancelled;
    row.turns += point.totals.turns;
    row.decode = point.decodeTps;
    if (point.mtpMeanAccepted !== null && point.mtpTurns > 0) {
      row.mtpTurns += point.mtpTurns;
      row.mtpSum += point.mtpMeanAccepted * point.mtpTurns;
    }
    days.set(point.t, row);
  }
  return days;
}

function outcomeTone(outcome: TurnRow['outcome']): 'success' | 'danger' | 'neutral' {
  if (outcome === 'ok') return 'success';
  if (outcome === 'error') return 'danger';
  return 'neutral';
}

export const TURN_COLUMNS: DataColumn<TurnRow>[] = [
  {
    id: 'when',
    header: 'When',
    width: 90,
    cell: (row) => <span title={new Date(row.ts).toISOString()}>{formatRelativeTime(row.ts)}</span>,
  },
  {
    id: 'outcome',
    header: 'Outcome',
    flex: true,
    minWidth: 104,
    truncate: true,
    cell: (row) => {
      const phase = row.outcome === 'error' ? failPhaseText(row.failPhase) : null;
      const words = phase === null ? outcomeText(row.outcome) : `${outcomeText(row.outcome)} · ${phase}`;
      return (
        <span className="flex min-w-0 items-center gap-1.5">
          <LiveDot tone={outcomeTone(row.outcome)} label={words} size={6} />
          <span className="min-w-0 truncate" title={words}>
            {words}
          </span>
        </span>
      );
    },
  },
  { id: 'output', header: 'Output', align: 'end', width: 72, cell: (row) => formatNumber(row.outputTokens) },
  {
    id: 'decode',
    header: 'Decode',
    align: 'end',
    width: 96,
    cell: (row) =>
      row.decodeTps === null ? (
        '—'
      ) : (
        <span className="whitespace-nowrap">
          {formatTps(row.decodeTps)} <span className="text-secondary">tok/s</span>
        </span>
      ),
  },
  {
    id: 'total',
    header: 'Total',
    align: 'end',
    width: 76,
    cell: (row) => formatDuration(row.durationMs === null ? null : (row.preDispatchMs ?? 0) + row.durationMs),
  },
];

function ModelDrawerBody(props: ModelDrawerProps & { model: LocalModel }): ReactElement {
  const { model, range } = props;
  const keys = usageKeys(model);
  const usage = props.usage.get(model.name);
  const [chosen, setChosen] = useState<string | null>(null);
  const usageId = chosen !== null && keys.includes(chosen) ? chosen : usageKeyOf(model, usage);
  const scope = `from=${String(range.from)}&to=${String(range.to)}&model=${encodeURIComponent(usageId)}`;
  const summary = useJson<MetricsSummary>(`/metrics/summary?${scope}`, { live: true });
  const held = useHeld(summary);
  const series = useJson<MetricsTimeseries>(`/metrics/timeseries?${scope}&bucket=day&groupBy=model`, { live: true });
  const turns = useJson<MetricsTurns>(`/metrics/turns?model=${encodeURIComponent(usageId)}&limit=5`, {
    live: { minIntervalMs: LIVE_REFETCH_FAST_MS },
  });

  const catalogItem = props.catalog.find((item) => item.slug === model.name);
  const update = catalogItem !== undefined && hasUpdate(catalogItem, props.remoteRevisions);
  const state = liveStateOf(model, props.live);
  const resident = (() => {
    for (const process of props.live?.processes ?? []) {
      if (process.status !== 'running') continue;
      const exact = process.models.find((entry) => entry.name === usageId);
      if (exact !== undefined) return exact;
    }
    return state.resident;
  })();
  const color = modelColor(usageId).color;

  const disk: KeyValueItem[] = [
    { key: 'path', label: 'Path', value: <PathText path={model.path} maxChars={42} />, mono: true, copy: model.path },
    { key: 'size', label: 'Size', value: formatBytes(model.sizeBytes), numeric: true },
    { key: 'files', label: 'Files', value: formatNumber(model.fileCount), numeric: true },
    {
      key: 'context',
      label: 'Context',
      value: model.contextWindow === null ? null : `${formatNumber(model.contextWindow)} tokens`,
      numeric: true,
    },
  ];
  if (catalogItem !== undefined) {
    disk.push({
      key: 'source',
      label: 'Source',
      value: (
        <a className="link" href={`https://huggingface.co/${catalogItem.hfRepo}`} target="_blank" rel="noreferrer">
          {catalogItem.hfRepo} <ExternalLink size={12} aria-hidden="true" className="inline" />
        </a>
      ),
    });
    if (catalogItem.localRevision !== null) {
      disk.push({ key: 'revision', label: 'Revision', value: catalogItem.localRevision.slice(0, 7), mono: true });
    }
  }

  const data = held.data;
  const totals = data?.totals;
  const latency = data?.latency;
  const loading = data === undefined && summary.error === undefined;
  const empty = totals !== undefined && totals.turns === 0;
  const ttft = formatDurationParts(latency?.ttftMs.p50 ?? null);
  const days = byDay(series.data?.points, usageId);
  const barData: ChartPoint[] = range.starts.map((t) => {
    const row = days.get(t);
    return {
      key: String(t),
      label: formatDayLong(t),
      axisLabel: formatDay(t),
      values: { ok: row?.ok ?? 0, failed: row?.failed ?? 0 },
    };
  });
  const barSeries: ChartSeries[] = [
    { key: 'ok', label: 'Completed', color },
    { key: 'failed', label: 'Failed or cancelled', color: 'var(--model-other)', pattern: 'alt' },
  ];
  const decodeData: ChartPoint[] = range.starts.map((t) => {
    const decode = days.get(t)?.decode ?? null;
    const tails = decode !== null && decode.n >= LOW_SAMPLE_TAIL_N;
    return {
      key: String(t),
      label: formatDayLong(t),
      axisLabel: formatDay(t),
      values: {
        p50: decode?.p50 ?? null,
        p05: tails ? decode.p05 : null,
        p95: tails ? decode.p95 : null,
      },
      samples: { p50: decode?.n ?? 0 },
    };
  });
  let mtpTurns = 0;
  let mtpSum = 0;
  for (const row of days.values()) {
    mtpTurns += row.mtpTurns;
    mtpSum += row.mtpSum;
  }
  const ctx = latency?.ctxFillPct;
  const ctxWindow = usage?.ctxWindow ?? model.contextWindow;

  return (
    <div className="grid min-w-0 gap-5">
      {state.resident !== undefined || state.loading || update ? (
        <div className="flex flex-wrap items-center gap-1.5">
          <LiveBadge state={state} />
          {update ? (
            <Badge tone="warning" size="sm">
              Update
            </Badge>
          ) : null}
        </div>
      ) : null}

      <section className="grid min-w-0 gap-2" aria-label="On disk">
        <Heading>On disk</Heading>
        <KeyValue layout="rows" items={disk} ariaLabel="On disk" />
      </section>

      {resident !== undefined ? (
        <>
          <Rule />
          <section className="grid min-w-0 gap-2" aria-label="Right now">
            <Heading>Right now</Heading>
            <KeyValue
              ariaLabel="Right now"
              items={[
                {
                  key: 'loaded',
                  label: 'Loaded',
                  value:
                    resident.loadedAt === null
                      ? null
                      : `${formatClock(resident.loadedAt)} (${formatRelativeTime(resident.loadedAt)})`,
                },
                { key: 'load', label: 'Load time', value: formatDuration(resident.loadMs) },
                { key: 'paged', label: 'Paged cache', value: resident.paged ? 'Yes' : 'No' },
                { key: 'max', label: 'Max turns', value: formatNumber(resident.maxConcurrent), numeric: true },
                { key: 'flight', label: 'In flight', value: formatNumber(resident.inFlight), numeric: true },
                {
                  key: 'queued',
                  label: 'Queued',
                  value:
                    resident.queueLimit === null
                      ? formatNumber(resident.queued)
                      : `${formatNumber(resident.queued)} of ${formatNumber(resident.queueLimit)}`,
                  numeric: true,
                },
              ]}
            />
            <div className="grid min-w-0 gap-1.5">
              <div className="flex items-baseline justify-between gap-2 text-body">
                <span className="text-secondary">KV pool at rest</span>
                {resident.pool !== null ? (
                  <span className="tabular-nums">
                    {`${formatNumber(resident.pool.allocatedBlocks)} of ${formatNumber(resident.pool.blockCapacity)}`}{' '}
                    <span className="text-secondary">blocks</span>
                  </span>
                ) : null}
              </div>
              <Meter
                label="KV pool at rest"
                value={resident.pool?.allocatedBlocks ?? null}
                max={resident.pool?.blockCapacity ?? null}
                unit="blocks"
                formatValue={formatNumber}
                unknownLabel="no snapshot yet"
              />
            </div>
            {resident.pool !== null ? (
              <span className="text-caption text-secondary">as of {formatClock(resident.pool.at)}</span>
            ) : null}
          </section>
        </>
      ) : null}

      <Rule />
      <section className="grid min-w-0 gap-3" aria-label="Usage">
        <Heading
          actions={
            keys.length > 1 ? (
              <CompactField className="w-48">
                <Select
                  label="Served as"
                  options={keys.map((key) => ({ value: key, label: truncateMiddle(key, 28) }))}
                  value={usageId}
                  onValueChange={setChosen}
                />
              </CompactField>
            ) : undefined
          }
        >
          Usage · last 30 days
        </Heading>
        {summary.error !== undefined && data === undefined ? (
          <Alert tone="warning" title="Usage could not be loaded">
            <span className="grid justify-items-start gap-2">
              <span>{summary.error.message}</span>
              <Button variant="secondary" size="sm" onClick={summary.reload}>
                Try again
              </Button>
            </span>
          </Alert>
        ) : empty ? (
          <div className="grid gap-1">
            <p className="m-0 text-body">{NO_TURNS}</p>
            <p className="m-0 text-caption text-secondary">{NO_TURNS_HINT}</p>
          </div>
        ) : (
          <>
            <StatGroup label="Usage, last 30 days">
              <StatTile
                label="Turns"
                value={totals === undefined ? null : formatCount(totals.turns)}
                loading={loading}
                held={held.held}
              />
              <StatTile
                label="Decode"
                value={latency?.decodeTps.p50 == null ? null : formatTps(latency.decodeTps.p50)}
                unit="tok/s"
                sampleCount={latency?.decodeTps.n}
                loading={loading}
                held={held.held}
              />
              <StatTile
                label="First token"
                value={latency?.ttftMs.p50 == null ? null : ttft.value}
                unit={ttft.unit}
                loading={loading}
                held={held.held}
              />
              <StatTile
                label="Failed"
                value={totals === undefined ? null : formatCount(totals.error)}
                hint={
                  totals === undefined
                    ? undefined
                    : `${formatRate(totals.error, totals.turns)} · ${formatCount(totals.cancelled)} cancelled`
                }
                loading={loading}
                held={held.held}
              />
            </StatGroup>
            <ChartBlock title="Turns per day" unit="turns" height={96} loading={series.data === undefined}>
              <StackedBarChart
                label="Turns per day"
                unit="turns"
                height={96}
                data={barData}
                series={barSeries}
                formatValue={formatNumber}
              />
            </ChartBlock>
            <ChartBlock
              title="Decode speed · median per day, p05–p95 band"
              unit="tok/s"
              height={120}
              loading={series.data === undefined}
            >
              <TrendChart
                label="Decode speed per day"
                unit="tok/s"
                height={120}
                data={decodeData}
                series={[{ key: 'p50', label: 'Median', color }]}
                band={{ lowKey: 'p05', highKey: 'p95' }}
                formatValue={formatTps}
                formatTick={axisNumber}
              />
            </ChartBlock>
            {ctx !== undefined && ctx.n > 0 ? (
              <div className="grid gap-2">
                <KeyValue
                  ariaLabel="Context in use"
                  items={[
                    {
                      key: 'p50',
                      label: 'Context in use, typical',
                      value: ctx.p50 === null ? null : `${Math.round(ctx.p50)}%`,
                      numeric: true,
                    },
                    {
                      key: 'p95',
                      label: 'High (p95)',
                      value: ctx.p95 === null ? null : `${Math.round(ctx.p95)}%`,
                      numeric: true,
                    },
                  ]}
                />
                <Meter label="Context in use, p95" value={ctx.p95} max={100} unit="%" warnAt={0.8} dangerAt={0.95} />
                {ctxWindow !== null ? (
                  <span className="text-caption text-secondary">of {formatNumber(ctxWindow)} tokens</span>
                ) : null}
              </div>
            ) : null}
            {mtpTurns > 0 ? (
              <KeyValue
                ariaLabel="Speculative decoding"
                items={[
                  {
                    key: 'mtp',
                    label: 'Speculative decoding',
                    value: (
                      <span className="flex flex-wrap items-center gap-2">
                        <span className="tabular-nums">
                          {(mtpSum / mtpTurns).toFixed(2)} tokens per cycle ·{' '}
                          {formatCountNoun(mtpTurns, 'turn', 'turns', formatCount)}
                        </span>
                        <Link className="link" to={`/metrics?model=${encodeURIComponent(usageId)}`}>
                          Open in Metrics
                        </Link>
                      </span>
                    ),
                  },
                ]}
              />
            ) : null}
            <div className="grid min-w-0 gap-1.5">
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-body font-medium">Recent turns</span>
                <Link className="link text-caption" to={`/metrics?model=${encodeURIComponent(usageId)}`}>
                  All turns
                </Link>
              </div>
              <DataTable
                caption="Recent turns"
                density="compact"
                columns={TURN_COLUMNS}
                rows={turns.data?.rows ?? []}
                rowKey={(row) => row.traceId}
                loading={turns.data === undefined && turns.error === undefined}
                loadingRows={5}
                error={turns.error === undefined ? undefined : { message: turns.error.message, retry: turns.reload }}
                empty={{ title: 'No recent turns', description: NO_TURNS_HINT }}
              />
            </div>
          </>
        )}
      </section>

      <Rule />
      <div className="flex flex-wrap items-center justify-between gap-2">
        <CopyButton value={model.name} label="Copy name" />
        <ConfirmPopover
          trigger={{ kind: 'button', label: 'Delete model', variant: 'danger' }}
          align="end"
          {...deleteContent(model, { loaded: state.resident !== undefined, onDeleted: props.onDeleted })}
        />
      </div>
    </div>
  );
}
