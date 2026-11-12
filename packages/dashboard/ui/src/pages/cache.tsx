import { RefreshCw } from 'lucide-react';
import { useEffect, useMemo, useRef, useState, type ReactElement, type ReactNode } from 'react';

import { ChartCard, type ChartState } from '../components/arc-x/chart-card/chart-card';
import type { ChartPoint, ChartSeries } from '../components/arc-x/chart/types';
import { ConfirmDialog } from '../components/arc-x/confirm-dialog/confirm-dialog';
import { ConfirmPopover, type ConfirmContent } from '../components/arc-x/confirm-popover/confirm-popover';
import { Disclosure } from '../components/arc-x/disclosure/disclosure';
import { IconButton } from '../components/arc-x/icon-button/icon-button';
import { Panel } from '../components/arc-x/panel/panel';
import { PathText } from '../components/arc-x/path-text/path-text';
import { Placeholder } from '../components/arc-x/placeholder/placeholder';
import { Rule } from '../components/arc-x/rule/rule';
import { Meter, SplitBar } from '../components/arc-x/split-bar/split-bar';
import { StackedBarChart } from '../components/arc-x/stacked-bar-chart/stacked-bar-chart';
import { StatGroup, StatTile } from '../components/arc-x/stat-tile/stat-tile';
import { SwapText } from '../components/arc-x/swap-text/swap-text';
import { TrendChart } from '../components/arc-x/trend-chart/trend-chart';
import { Alert } from '../components/arc/alert/alert';
import { Badge } from '../components/arc/badge/badge';
import { Button } from '../components/arc/button/button';
import { CopyButton } from '../components/arc/copy-button/copy-button';
import SegmentedControl, { type Segment } from '../components/arc/segmented-control/segmented-control';
import { mutate } from '../lib/api';
import { coldObjectCounts, coldWindowTotals, evictPreview, hitRateSeries, sidecarReading } from '../lib/cold-tier';
import {
  formatBytes,
  formatBytesParts,
  formatClock,
  formatCount,
  formatCountNoun,
  formatNumber,
  formatPercent,
  formatPercentValue,
  formatRate,
  formatRelativeTime,
} from '../lib/format';
import { useHeld } from '../lib/held';
import { useDataVersion } from '../lib/live';
import { axisNumber, byteAxis } from '../lib/metrics-view';
import { alignedRange } from '../lib/range';
import { LOOKUP_SERIES, SERIES_A, SERIES_B } from '../lib/series';
import { bucketPoint } from '../lib/timeseries';
import { toast } from '../lib/toast';
import type { CacheMutationResult, CacheResponse } from '../lib/types';
import { LIVE_REFETCH_FAST_MS, reloadMountedQueries, useJson } from '../lib/use-api';
import { useCommands, type AppCommand } from '../shell/commands';
import { Page, PageGrid, PageHeader } from '../shell/page';

type EvictDays = 1 | 7 | 30;
type Pending = { kind: 'clear' } | { kind: 'evict'; days: EvictDays } | null;

const EVICT_OPTIONS: Segment[] = [
  { value: '1', label: '1 day' },
  { value: '7', label: '7 days' },
  { value: '30', label: '30 days' },
];
const MEASURE_OPTIONS: Segment[] = [
  { value: 'objects', label: 'Objects' },
  { value: 'bytes', label: 'Bytes' },
];
const VIEW_OPTIONS: Segment[] = [
  { value: 'lookups', label: 'Lookups' },
  { value: 'rate', label: 'Hit rate' },
];

const KEPT_SERIES: ChartSeries[] = [
  { key: 'kept', label: 'Kept', color: SERIES_A },
  { key: 'evict', label: 'Would be evicted', color: SERIES_A, pattern: 'alt' },
];
const BYTES_SERIES: ChartSeries[] = [
  { key: 'written', label: 'Written', color: SERIES_A },
  { key: 'restored', label: 'Restored', color: SERIES_B },
];

const HEALTH_HELP =
  'Counters are per turn and belong to this cache root. A corruption always costs a miss, so never add the two. ' +
  'A decline is neither a hit nor a miss: a refused restore looks nothing up. Sidecars hold family state kept outside ' +
  'the paged pool; the sidecar figures on the queue tile are a share of the object counts beside them. Writes, ' +
  'evictions and write errors advance on a background thread, so those totals are approximate.';

function plural(n: number, one: string, many: string): string {
  return n === 1 ? one : many;
}

function evictDaysOf(value: string): EvictDays | null {
  return value === '1' ? 1 : value === '7' ? 7 : value === '30' ? 30 : null;
}

/** The start index of the age histogram an eviction of `days` removes from. */
function evictStart(days: EvictDays): number {
  return days >= 30 ? 3 : days >= 7 ? 2 : 1;
}

function Strong({ children }: { children: ReactNode }): ReactElement {
  return <span className="font-medium text-foreground">{children}</span>;
}

/** A value split into the tile's value and unit; '—' becomes null. */
function bytesTile(bytes: number | null | undefined): { value: string | null; unit?: string } {
  const parts = formatBytesParts(bytes);
  return parts.value === '—' ? { value: null } : { value: parts.value, unit: parts.unit };
}

/** A tile hint on one line: cut with an ellipsis, the full text in its title. */
function Alarm({ on, children }: { on: boolean; children: string }): ReactElement {
  return (
    <span className="flex min-w-0 items-center gap-1.5" title={children}>
      {on ? (
        <Badge size="sm" tone="danger" className="shrink-0">
          Check
        </Badge>
      ) : null}
      <span className="min-w-0 truncate">{children}</span>
    </span>
  );
}

export default function Cache(): ReactElement {
  const cache = useJson<CacheResponse>('/cache', { live: { minIntervalMs: LIVE_REFETCH_FAST_MS } });
  const held = useHeld(cache);
  const data = held.data;
  const disk = data?.disk;
  const scope = data?.scope;
  const health = data?.health;
  const loading = data === undefined;

  const [evictDays, setEvictDays] = useState<EvictDays>(7);
  const [announced, setAnnounced] = useState('');
  const [measure, setMeasure] = useState<'objects' | 'bytes'>('objects');
  const [view, setView] = useState<'lookups' | 'rate'>('lookups');
  const [pending, setPending] = useState<Pending>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [receivedAt, setReceivedAt] = useState<number | null>(null);
  const result = useRef<CacheMutationResult | null>(null);

  useEffect(() => {
    if (cache.data !== undefined) setReceivedAt(Date.now());
  }, [cache.data]);

  const totalBytes = disk?.totalBytes ?? 0;
  const quotaBytes = disk?.quotaBytes ?? 0;
  const sidecarBytes = disk?.sidecarBytes ?? 0;
  const blockBytes = Math.max(0, totalBytes - sidecarBytes);
  const { blocks, sidecars, objects } = coldObjectCounts(disk);
  const quotaFraction = quotaBytes > 0 ? Math.min(1, totalBytes / quotaBytes) : 0;

  const dataVersion = useDataVersion();
  const range = useMemo(() => alignedRange('30d'), [dataVersion]);
  const now = range.to - 1;

  const trend = useMemo(() => data?.trend ?? [], [data]);
  const totals = useMemo(() => coldWindowTotals(trend, range.starts), [trend, range]);
  const lookups = totals.hits + totals.misses;
  const declines = health?.restoreDeclines ?? 0;
  const sidecar = sidecarReading(health, scope, totals.hits);
  const excludedLookups =
    scope === undefined
      ? 0
      : scope.legacy.hits +
        scope.legacy.misses +
        scope.otherRoots.hits +
        scope.otherRoots.misses +
        scope.unattributed.hits +
        scope.unattributed.misses;

  const previewOf = (days: EvictDays): string => {
    const preview = disk === undefined ? { count: 0, bytes: 0 } : evictPreview(disk.ageHistogram, days);
    return preview.count === 0
      ? 'Nothing is that old.'
      : `${formatNumber(preview.count)} ${plural(preview.count, 'object', 'objects')} · ${formatBytes(preview.bytes)} would be removed`;
  };
  const evictP = disk === undefined ? { count: 0, bytes: 0 } : evictPreview(disk.ageHistogram, evictDays);
  const previewText = previewOf(evictDays);
  const evictDisabled = cache.loading || evictP.count === 0;
  const clearDisabled = cache.loading || objects === 0;

  const ageChart = useMemo(() => {
    const buckets = disk?.ageHistogram ?? [];
    const start = evictStart(evictDays);
    const axis = byteAxis(Math.max(0, ...buckets.map((bucket) => bucket.bytes)));
    const data: ChartPoint[] = buckets.map((bucket, index) => {
      const value = measure === 'objects' ? bucket.count : bucket.bytes / axis.div;
      const evicted = index >= start;
      return {
        key: bucket.label,
        label: `${bucket.label} · ${formatNumber(bucket.count)} ${plural(bucket.count, 'object', 'objects')} · ${formatBytes(bucket.bytes)}`,
        axisLabel: bucket.label,
        values: { kept: evicted ? 0 : value, evict: evicted ? value : 0 },
      };
    });
    return { data, unit: measure === 'objects' ? 'objects' : axis.unit };
  }, [disk, evictDays, measure]);

  const lookupChart = useMemo(() => {
    const days = hitRateSeries(trend, range.starts);
    const lookupData: ChartPoint[] = days.map((day) => {
      const point = bucketPoint(day.t, 'day', now);
      const rate = formatRate(day.hits, day.lookups);
      return {
        ...point,
        label: day.lookups > 0 ? `${point.label} · hit rate ${rate}` : point.label,
        values: { hits: day.hits, misses: day.lookups - day.hits },
      };
    });
    const rateData: ChartPoint[] = days.map((day) => ({
      ...bucketPoint(day.t, 'day', now),
      values: { rate: day.rate },
      samples: { rate: day.lookups },
    }));
    const inWindow = days.reduce((sum, day) => sum + day.lookups, 0);
    return { lookupData, rateData, inWindow };
  }, [trend, range, now]);

  const bytesChart = useMemo(() => {
    const byDay = new Map<number, { written: number; restored: number }>();
    for (const row of trend) {
      const seen = byDay.get(row.t);
      byDay.set(row.t, {
        written: (seen?.written ?? 0) + row.bytesWritten,
        restored: (seen?.restored ?? 0) + row.bytesRestored,
      });
    }
    let max = 0;
    let written = 0;
    let restored = 0;
    for (const t of range.starts) {
      const day = byDay.get(t);
      max = Math.max(max, day?.written ?? 0, day?.restored ?? 0);
      written += day?.written ?? 0;
      restored += day?.restored ?? 0;
    }
    const axis = byteAxis(max);
    const data: ChartPoint[] = range.starts.map((t) => {
      const day = byDay.get(t);
      return {
        ...bucketPoint(t, 'day', now),
        values: { written: (day?.written ?? 0) / axis.div, restored: (day?.restored ?? 0) / axis.div },
      };
    });
    return { data, unit: axis.unit, written, restored };
  }, [trend, range, now]);

  const chartState = (empty: boolean): ChartState =>
    loading ? 'loading' : empty ? 'empty' : held.held ? 'held' : 'ready';

  const toastResult = (title: string): void => {
    const done = result.current;
    toast.success(title, {
      description:
        done === null
          ? undefined
          : `${formatNumber(done.removed)} ${plural(done.removed, 'object', 'objects')} removed · ${formatBytes(done.freedBytes)} freed`,
    });
  };

  const evictContent = (days: EvictDays): ConfirmContent => {
    const preview = disk === undefined ? { count: 0, bytes: 0 } : evictPreview(disk.ageHistogram, days);
    const daysText = `${String(days)} ${plural(days, 'day', 'days')}`;
    return {
      title: 'Evict old cache objects?',
      description: (
        <>
          Removes cold-tier objects (blocks and state sidecars) older than <Strong>{daysText}</Strong>. This frees an
          estimated{' '}
          <Strong>
            {`${formatNumber(preview.count)} ${plural(preview.count, 'object', 'objects')} · ${formatBytes(preview.bytes)}`}
          </Strong>
          . Newer objects are kept.
        </>
      ),
      confirmLabel: 'Evict',
      workingLabel: 'Evicting…',
      successLabel: 'Evicted',
      tone: 'neutral',
      errorTitle: 'Failed to update cache',
      onConfirm: async () => {
        result.current = await mutate<CacheMutationResult>('DELETE', '/cache', { olderThanDays: days });
      },
      onDone: () => {
        cache.reload();
        toastResult(`Evicted cold-tier objects older than ${daysText}`);
      },
    };
  };

  const clearContent: ConfirmContent = {
    title: 'Clear the cold cache?',
    description: (
      <>
        Removes all <Strong>{`${formatNumber(objects)} ${plural(objects, 'object', 'objects')}`}</Strong> —{' '}
        {`${formatNumber(blocks)} prefix ${plural(blocks, 'block', 'blocks')} and ${formatNumber(sidecars)} state ${plural(sidecars, 'sidecar', 'sidecars')}, ${formatBytes(totalBytes)} in total`}{' '}
        — from the cold tier. This cannot be undone.
      </>
    ),
    confirmLabel: 'Clear all',
    workingLabel: 'Clearing…',
    successLabel: 'Cleared',
    tone: 'danger',
    errorTitle: 'Failed to update cache',
    onConfirm: async () => {
      result.current = await mutate<CacheMutationResult>('DELETE', '/cache', { all: true });
    },
    onDone: () => {
      cache.reload();
      toastResult('Cleared cold cache');
    },
  };

  const commands = useMemo((): AppCommand[] => {
    const list: AppCommand[] = [];
    if (!evictDisabled) {
      list.push({
        id: 'cache-evict',
        label: 'Evict old cache objects…',
        group: 'Cache',
        run: () => setPending({ kind: 'evict', days: evictDays }),
      });
    }
    if (!clearDisabled) {
      list.push({
        id: 'cache-clear',
        label: 'Clear cold cache…',
        group: 'Cache',
        run: () => setPending({ kind: 'clear' }),
      });
    }
    return list;
  }, [evictDisabled, clearDisabled, evictDays]);
  useCommands(commands);

  const refresh = async (): Promise<void> => {
    setRefreshing(true);
    try {
      await mutate('POST', '/ingest');
    } catch {
      // A failed ingest still leaves the last index to read.
    } finally {
      reloadMountedQueries();
      setRefreshing(false);
    }
  };

  const header = (
    <PageHeader
      title="Cache"
      description={
        <>
          <span className="block">Paged cold tier on disk: use, reuse and health.</span>
          {disk === undefined ? null : (
            <span className="mt-1 flex min-w-0 items-center gap-1">
              <span className="min-w-0 truncate text-caption text-secondary">
                <PathText path={disk.root} title={scope?.root ?? disk.root} />
                {disk.exists ? '' : ' · not created yet'}
              </span>
              <CopyButton value={disk.root} iconOnly variant="plain" label="Copy cache path" />
            </span>
          )}
        </>
      }
      actions={
        <IconButton
          label="Refresh data"
          variant="secondary"
          icon={<RefreshCw size={16} aria-hidden="true" />}
          loading={refreshing}
          onClick={() => void refresh()}
        />
      }
    />
  );

  if (cache.error !== undefined && data === undefined) {
    return (
      <Page>
        {header}
        <Alert tone="danger" title="Could not load the cache">
          <span className="grid justify-items-start gap-2">
            <span>{cache.error.message}</span>
            <Button variant="secondary" size="sm" onClick={cache.reload}>
              Try again
            </Button>
          </span>
        </Alert>
      </Page>
    );
  }

  const scopeSentences: string[] = [];
  if (scope !== undefined) {
    if (scope.legacy.turns > 0) {
      scopeSentences.push(
        `${formatNumber(scope.legacy.turns)} ${plural(scope.legacy.turns, 'turn', 'turns')} recorded before cache attribution existed (${formatCountNoun(scope.legacy.hits, 'hit', 'hits')} · ${formatCountNoun(scope.legacy.misses, 'miss', 'misses')}) ${plural(scope.legacy.turns, 'is', 'are')} excluded; they age out within ${String(scope.trendWindowDays)} days.`,
      );
    }
    if (scope.otherRoots.turns > 0) {
      scopeSentences.push(
        `${formatNumber(scope.otherRoots.turns)} ${plural(scope.otherRoots.turns, 'turn', 'turns')} ran against a different cache directory (${formatCountNoun(scope.otherRoots.hits, 'hit', 'hits')} · ${formatCountNoun(scope.otherRoots.misses, 'miss', 'misses')}) and ${plural(scope.otherRoots.turns, 'is', 'are')} excluded too.`,
      );
    }
    if (scope.unattributed.turns > 0) {
      scopeSentences.push(
        `${formatNumber(scope.unattributed.turns)} ${plural(scope.unattributed.turns, 'turn', 'turns')} ran with the cold tier on but recorded no cache directory (${formatCountNoun(scope.unattributed.hits, 'hit', 'hits')} · ${formatCountNoun(scope.unattributed.misses, 'miss', 'misses')}) and cannot be attributed.`,
      );
    }
  }

  const hitHint =
    lookups > 0
      ? `${formatCountNoun(totals.hits, 'hit', 'hits', formatCount)} · ${formatCountNoun(totals.misses, 'miss', 'misses', formatCount)} · this cache only`
      : declines > 0
        ? `${formatCount(declines)} ${plural(declines, 'restore', 'restores')} refused · no lookup was performed`
        : 'no lookups recorded for this cache';

  const restoreFamilies = data?.restoreFamilies ?? [];
  const ageEmptyHint =
    restoreFamilies.length > 0
      ? `Objects appear when a persistent paged cache is used by a restore-eligible family: ${restoreFamilies.join(', ')}.`
      : 'Objects appear when a persistent paged cache is used by a restore-eligible model family.';
  const trendEmptyHint =
    excludedLookups > 0
      ? `${formatCountNoun(excludedLookups, 'lookup was', 'lookups were')} recorded against a different (or unattributed) cache root — see the note above.`
      : (scope?.disabledTurns ?? 0) > 0
        ? `${formatNumber(scope?.disabledTurns ?? 0)} ${plural(scope?.disabledTurns ?? 0, 'turn', 'turns')} ran with the cold tier off, so nothing was looked up.`
        : 'Lookups are recorded per turn when a model runs with a persistent paged cache.';

  const corruptionsSeen = (health?.corruptions ?? 0) > 0 || (health?.corruptionsTotal ?? 0) > 0;
  const writeErrorsSeen = (health?.writeErrors ?? 0) > 0 || (health?.writeErrorsTotal ?? 0) > 0;
  const capturesHint =
    sidecar.captures === 0
      ? sidecar.unrooted > 0
        ? `${formatNumber(sidecar.unrooted)} ran with no cache attached — persistence off, or the tier never opened`
        : 'not reached — dense families keep no state outside the pool'
      : sidecar.stranded
        ? 'nothing written and nothing on disk — no prefix can restore state'
        : `${formatNumber(sidecar.written)} written · ${formatNumber(sidecar.persisted)} already on disk · ${formatNumber(sidecar.noAnchor)} found no anchor`;
  const installsHint = sidecar.replayed
    ? 'state was on disk and prefixes restored — every one re-derived by a full replay'
    : sidecar.installed > 0
      ? 'restored prefixes that reused family state instead of replaying it'
      : 'no restored prefix carried family state in this window';

  const usage = bytesTile(disk?.totalBytes);
  const restored = bytesTile(loading ? undefined : totals.bytesRestored);
  const hitRate = formatRate(totals.hits, lookups);
  const oldest = disk?.oldestMtime ?? null;

  return (
    <Page busy={cache.loading || cache.refreshing}>
      {header}
      {cache.error !== undefined ? (
        <Alert tone="warning" title="Could not refresh">
          <span className="grid justify-items-start gap-2">
            <span>{`Showing data from ${formatClock(receivedAt)}.`}</span>
            <Button variant="secondary" size="sm" onClick={cache.reload}>
              Try again
            </Button>
          </span>
        </Alert>
      ) : null}
      {scopeSentences.length > 0 ? (
        <Alert tone="info" title="Some turns are not counted here">
          {scopeSentences.join(' ')}
        </Alert>
      ) : null}

      <StatGroup label="Cold cache">
        <StatTile
          label="Usage"
          value={usage.value}
          unit={usage.unit}
          loading={loading}
          held={held.held}
          hint={`${quotaBytes > 0 ? `${formatPercent(quotaFraction)} of ${formatBytes(quotaBytes)} quota` : 'no quota available'} · ${formatBytes(blockBytes)} blocks + ${formatBytes(sidecarBytes)} sidecars`}
          footer={
            loading ? (
              <Placeholder height={6} radius="pill" />
            ) : quotaBytes > 0 ? (
              <Meter
                size="thin"
                label={`Cold cache using ${formatBytes(totalBytes)} of ${formatBytes(quotaBytes)} quota`}
                value={totalBytes}
                max={quotaBytes}
              />
            ) : undefined
          }
        />
        <StatTile
          label="Objects"
          value={loading ? null : formatCount(objects)}
          loading={loading}
          held={held.held}
          hint={`${formatCount(blocks)} prefix ${plural(blocks, 'block', 'blocks')} · ${formatCount(sidecars)} state ${plural(sidecars, 'sidecar', 'sidecars')}`}
        />
        <StatTile
          label="Hit rate"
          value={hitRate === '—' ? null : hitRate}
          loading={loading}
          held={held.held}
          hint={hitHint}
        />
        <StatTile
          label="Bytes restored"
          value={restored.value}
          unit={restored.unit}
          loading={loading}
          held={held.held}
          hint={`${formatBytes(totals.bytesWritten)} landed`}
        />
        <StatTile
          label="Oldest object"
          value={oldest === null ? null : formatRelativeTime(oldest)}
          loading={loading}
          held={held.held}
          hint={disk?.newestMtime != null ? `newest ${formatRelativeTime(disk.newestMtime)}` : 'empty tier'}
        />
      </StatGroup>

      <PageGrid>
        <ChartCard
          className="col-span-12 @md:col-span-7"
          title="Objects by age"
          height={200}
          subtitle={`Prefix blocks and state sidecars · ${ageChart.unit}`}
          state={chartState(objects === 0)}
          empty={{ title: 'No cold-tier objects persisted yet.', hint: ageEmptyHint }}
          actions={
            <SegmentedControl
              label="Measure"
              options={MEASURE_OPTIONS}
              value={measure}
              onValueChange={(value) => {
                if (value === 'objects' || value === 'bytes') setMeasure(value);
              }}
            />
          }
          footnote={`${formatCountNoun(objects, 'object', 'objects')} · ${formatBytes(totalBytes)}`}
        >
          <StackedBarChart
            label="Objects by age"
            unit={ageChart.unit}
            data={ageChart.data}
            series={KEPT_SERIES}
            formatValue={measure === 'objects' ? formatNumber : undefined}
            formatTick={measure === 'objects' ? formatCount : undefined}
          />
        </ChartCard>

        <Panel
          className="col-span-12 @md:col-span-5"
          title="Manage cold cache"
          description="Removes blocks and their state sidecars from disk. They are made again on demand; the next matching prompt pays full prefill once."
        >
          <div className="grid gap-3">
            <div className="grid justify-items-start gap-1.5">
              <span className="text-body text-secondary">Evict objects older than</span>
              <SegmentedControl
                label="Eviction age threshold"
                options={EVICT_OPTIONS}
                value={String(evictDays)}
                onValueChange={(value) => {
                  const days = evictDaysOf(value);
                  if (days === null) return;
                  setEvictDays(days);
                  if (!loading) setAnnounced(previewOf(days));
                }}
              />
            </div>
            <p className="text-body tabular-nums">
              <SwapText text={loading ? '' : previewText} granularity="char" />
            </p>
            <p className="sr-only" aria-live="polite">
              {announced}
            </p>
            <div className="flex flex-wrap items-center gap-2">
              <ConfirmPopover
                trigger={{ kind: 'button', label: 'Evict older', variant: 'secondary' }}
                align="start"
                disabled={evictDisabled}
                disabledReason={previewText}
                {...evictContent(evictDays)}
              />
            </div>
            <Rule />
            <div className="flex flex-wrap items-center gap-2">
              <ConfirmPopover
                trigger={{ kind: 'button', label: 'Clear all', variant: 'danger' }}
                align="start"
                disabled={clearDisabled}
                disabledReason={objects === 0 ? 'The tier is empty.' : undefined}
                {...clearContent}
              />
              {!loading && objects === 0 ? <span className="text-body text-secondary">The tier is empty.</span> : null}
            </div>
          </div>
        </Panel>

        <ChartCard
          className="col-span-12 @md:col-span-7"
          title="Cache hit / miss trend"
          height={200}
          subtitle={
            view === 'lookups'
              ? `Lookups against this cache root, per day · last ${String(scope?.trendWindowDays ?? 30)} days`
              : 'Hits ÷ lookups per day · % · axis does not start at 0'
          }
          state={chartState(lookupChart.inWindow === 0)}
          empty={{ title: 'No hits or misses recorded for this cache.', hint: trendEmptyHint }}
          actions={
            <SegmentedControl
              label="View"
              options={VIEW_OPTIONS}
              value={view}
              onValueChange={(value) => {
                if (value === 'lookups' || value === 'rate') setView(value);
              }}
            />
          }
          footnote={`${formatCountNoun(totals.hits, 'hit', 'hits', formatCount)} · ${formatCountNoun(totals.misses, 'miss', 'misses', formatCount)} · this cache only`}
        >
          {view === 'lookups' ? (
            <StackedBarChart
              label="Cache hit / miss trend"
              unit="lookups"
              data={lookupChart.lookupData}
              series={LOOKUP_SERIES}
              formatValue={formatNumber}
              formatTick={formatCount}
            />
          ) : (
            <TrendChart
              label="Cache hit rate per day"
              unit="%"
              yMin="auto"
              lowSampleBelow={5}
              data={lookupChart.rateData}
              series={[{ key: 'rate', label: 'Hit rate', color: 'var(--series-1)' }]}
              formatValue={(value) => formatPercentValue(value / 100)}
              formatTick={axisNumber}
            />
          )}
        </ChartCard>

        <ChartCard
          className="col-span-12 @md:col-span-5"
          title="Bytes written and restored"
          height={200}
          subtitle={`Per day · ${bytesChart.unit}`}
          state={chartState(bytesChart.written + bytesChart.restored === 0)}
          empty={{
            title: 'Nothing written or restored yet',
            hint: 'Bytes move when a prompt prefix is saved to disk or read back.',
          }}
          footnote={`${formatBytes(bytesChart.written)} written · ${formatBytes(bytesChart.restored)} restored`}
        >
          <StackedBarChart
            label="Bytes written and restored"
            unit={bytesChart.unit}
            mode="grouped"
            data={bytesChart.data}
            series={BYTES_SERIES}
          />
        </ChartCard>

        <Panel className="col-span-12" title="Cold-tier health">
          <div className="grid gap-3">
            <Disclosure title="How to read these counters">
              <p className="max-w-[72ch] text-body text-secondary">{HEALTH_HELP}</p>
            </Disclosure>
            <StatGroup label="Cold-tier health">
              <StatTile
                label="Corruptions"
                value={loading ? null : formatNumber(health?.corruptions ?? 0)}
                loading={loading}
                held={held.held}
                hint={
                  <Alarm on={corruptionsSeen}>
                    {corruptionsSeen
                      ? `cumulative max ${formatNumber(health?.corruptionsTotal ?? 0)} — investigate`
                      : 'none seen (acceptance bar: 0)'}
                  </Alarm>
                }
              />
              <StatTile
                label="Queue drops"
                value={loading ? null : formatNumber(health?.queueDrops ?? 0)}
                loading={loading}
                held={held.held}
                hint={
                  <Alarm on={(health?.queueDropsTotal ?? 0) > 0}>
                    {`cumulative max ${formatNumber(health?.queueDropsTotal ?? 0)} · ${formatNumber(health?.enqueued ?? 0)} objects enqueued · ${formatNumber(sidecar.written)} ${plural(sidecar.written, 'sidecar', 'sidecars')} among them, ${formatNumber(health?.sidecarQueueDrops ?? 0)} of the drops`}
                  </Alarm>
                }
              />
              <StatTile
                label="Write errors"
                value={loading ? null : formatNumber(health?.writeErrors ?? 0)}
                loading={loading}
                held={held.held}
                hint={
                  <Alarm on={writeErrorsSeen}>
                    {writeErrorsSeen
                      ? `cumulative max ${formatNumber(health?.writeErrorsTotal ?? 0)} — the cache root is refusing writes`
                      : 'none seen (writes are landing)'}
                  </Alarm>
                }
              />
              <StatTile
                label="Restore declines"
                value={loading ? null : formatNumber(declines)}
                loading={loading}
                held={held.held}
                hint="refused before lookup · not counted as a lookup"
              />
              <StatTile
                label="Sidecar captures"
                value={loading ? null : formatNumber(sidecar.captures)}
                loading={loading}
                held={held.held}
                hint={<Alarm on={sidecar.stranded}>{capturesHint}</Alarm>}
                footer={
                  loading ? (
                    <Placeholder height={6} radius="pill" />
                  ) : sidecar.captures > 0 && !sidecar.stranded ? (
                    <SplitBar
                      size="dense"
                      legend={false}
                      label="Capture outcomes"
                      total={sidecar.captures}
                      segments={[
                        { key: 'written', label: 'Written', value: sidecar.written, color: SERIES_A },
                        { key: 'persisted', label: 'Already on disk', value: sidecar.persisted, color: SERIES_B },
                        { key: 'noAnchor', label: 'No anchor', value: sidecar.noAnchor, color: 'var(--model-other)' },
                      ]}
                    />
                  ) : undefined
                }
              />
              <StatTile
                label="Sidecar installs"
                value={loading ? null : formatNumber(sidecar.installed)}
                loading={loading}
                held={held.held}
                hint={<Alarm on={sidecar.replayed}>{installsHint}</Alarm>}
              />
              <StatTile
                label="Evictions"
                value={loading ? null : formatNumber(health?.evictions ?? 0)}
                loading={loading}
                held={held.held}
                hint="objects removed by quota pressure"
              />
              <StatTile
                label="Turns with the tier off"
                value={loading ? null : formatNumber(scope?.disabledTurns ?? 0)}
                loading={loading}
                held={held.held}
                hint="ran without a persistent cache"
              />
            </StatGroup>
          </div>
        </Panel>
      </PageGrid>

      {pending === null ? null : (
        <ConfirmDialog
          open
          onOpenChange={(open) => {
            if (!open) setPending(null);
          }}
          {...(pending.kind === 'evict' ? evictContent(pending.days) : clearContent)}
        />
      )}
    </Page>
  );
}
