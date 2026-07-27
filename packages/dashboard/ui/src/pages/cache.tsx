import { StatTile } from '@/components/stat-tile';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Skeleton } from '@/components/ui/skeleton';
import { mutate } from '@/lib/api';
import { AXIS_TICK, ChartBody, ChartCard, ChartEmpty, TOOLTIP_CONTENT_STYLE } from '@/lib/chart';
import { coldObjectCounts, evictPreview } from '@/lib/cold-tier';
import {
  formatBytes,
  formatCount,
  formatNumber,
  formatPercent,
  formatRate,
  formatRelativeTime,
  percentInt,
} from '@/lib/format';
import type { CacheMutationResult, CacheResponse } from '@/lib/types';
import { useJson } from '@/lib/use-api';
import { cn } from '@/lib/utils';
import type { LucideIcon } from 'lucide-react';
import {
  AlertCircle,
  Clock,
  Database,
  HardDrive,
  Layers,
  Loader2,
  Percent,
  ShieldAlert,
  ShieldCheck,
  Trash2,
} from 'lucide-react';
import { useMemo, useState } from 'react';
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { toast } from 'sonner';

const EVICT_OPTIONS: Array<{ value: string; label: string; days: number }> = [
  { value: '1', label: 'Older than 1 day', days: 1 },
  { value: '7', label: 'Older than 7 days', days: 7 },
  { value: '30', label: 'Older than 30 days', days: 30 },
];

/** UTC-day string (`YYYY-MM-DD`) → short label (`Jul 20`). */
function formatDay(day: string): string {
  const parsed = new Date(`${day}T00:00:00Z`);
  if (Number.isNaN(parsed.getTime())) return day;
  return parsed.toLocaleDateString(undefined, { month: 'short', day: 'numeric', timeZone: 'UTC' });
}

function errMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

type PendingAction = { kind: 'clear' } | { kind: 'evict'; days: number } | null;

export default function Cache() {
  const cache = useJson<CacheResponse>('/cache');
  const disk = cache.data?.disk;
  const trend = useMemo(() => cache.data?.trend ?? [], [cache.data]);

  const [evictKey, setEvictKey] = useState('7');
  const [pending, setPending] = useState<PendingAction>(null);
  const [busy, setBusy] = useState(false);

  const evictDays = EVICT_OPTIONS.find((o) => o.value === evictKey)?.days ?? 7;

  const totalBytes = disk?.totalBytes ?? 0;
  const quotaBytes = disk?.quotaBytes ?? 0;
  const sidecarBytes = disk?.sidecarBytes ?? 0;
  const blockBytes = Math.max(0, totalBytes - sidecarBytes);
  // `blocks` is the KV-block count; `objects` is blocks + sidecars, which is
  // what the age histogram charts, what the quota measures, and what clear/evict
  // actually unlink. Using `blocks` for any of those is the F2 miscount.
  const { blocks, sidecars, objects } = coldObjectCounts(disk);
  const quotaFraction = quotaBytes > 0 ? Math.min(1, totalBytes / quotaBytes) : 0;
  const scope = cache.data?.scope;
  const health = cache.data?.health;
  const restoreFamilies = cache.data?.restoreFamilies ?? [];

  const trendTotals = useMemo(
    () =>
      trend.reduce(
        (acc, row) => ({
          hits: acc.hits + row.hits,
          misses: acc.misses + row.misses,
          bytesWritten: acc.bytesWritten + row.bytesWritten,
          bytesRestored: acc.bytesRestored + row.bytesRestored,
        }),
        { hits: 0, misses: 0, bytesWritten: 0, bytesRestored: 0 },
      ),
    [trend],
  );
  const lookups = trendTotals.hits + trendTotals.misses;

  const trendData = useMemo(
    () => trend.map((row) => ({ day: formatDay(row.day), hits: row.hits, misses: row.misses })),
    [trend],
  );
  const hasTrend = lookups > 0;
  // Gates the destructive controls and the age chart. Must count OBJECTS: a
  // root holding only sidecars still occupies quota, and gating on blocks alone
  // left those bytes visible in the Usage tile but impossible to clear.
  const hasObjects = objects > 0;

  const evictP = disk !== undefined ? evictPreview(disk.ageHistogram, evictDays) : { count: 0, bytes: 0 };
  const excludedLookups =
    scope !== undefined
      ? scope.legacy.hits +
        scope.legacy.misses +
        scope.otherRoots.hits +
        scope.otherRoots.misses +
        scope.unattributed.hits +
        scope.unattributed.misses
      : 0;
  const corruptionsSeen = (health?.corruptionsTotal ?? 0) > 0 || (health?.corruptions ?? 0) > 0;

  const runDelete = async (action: Exclude<PendingAction, null>): Promise<void> => {
    setBusy(true);
    try {
      const body = action.kind === 'evict' ? { olderThanDays: action.days } : { all: true };
      const result = await mutate<CacheMutationResult>('DELETE', '/cache', body);
      // The server unlinks blocks AND sidecars, so `removed` is an OBJECT count
      // — say "objects", not "blocks", or the toast disagrees with the dialog.
      toast.success(
        action.kind === 'evict'
          ? `Evicted cold-tier objects older than ${action.days} day${action.days === 1 ? '' : 's'}`
          : 'Cleared cold cache',
        {
          description: `${formatNumber(result.removed)} object${result.removed === 1 ? '' : 's'} removed · ${formatBytes(result.freedBytes)} freed`,
        },
      );
      setPending(null);
      cache.reload();
    } catch (err) {
      toast.error('Failed to update cache', { description: errMessage(err) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Cache</h1>
        <p className="text-muted-foreground text-sm">PagedAttention cold-tier usage and management.</p>
        {disk !== undefined && (
          <p className="text-muted-foreground mt-1 font-mono text-xs break-all" title={scope?.root ?? disk.root}>
            {disk.root}
            {!disk.exists && ' · not created yet'}
          </p>
        )}
        {/*
          Every number on this page is scoped to the root above. Rows the scope
          left out are named here rather than silently folded in (which was the
          bug) or silently dropped (which would make real history vanish).
        */}
        {scope !== undefined &&
          (scope.legacy.turns > 0 || scope.otherRoots.turns > 0 || scope.unattributed.turns > 0) && (
            <p className="text-muted-foreground mt-1 text-xs">
              {scope.legacy.turns > 0 && (
                <span>
                  {formatNumber(scope.legacy.turns)} turn{scope.legacy.turns === 1 ? '' : 's'} recorded before cache
                  attribution existed ({formatNumber(scope.legacy.hits)} hits · {formatNumber(scope.legacy.misses)}{' '}
                  misses) are excluded; they age out within {scope.trendWindowDays} days.
                </span>
              )}
              {scope.otherRoots.turns > 0 && (
                <span>
                  {scope.legacy.turns > 0 && ' '}
                  {formatNumber(scope.otherRoots.turns)} turn{scope.otherRoots.turns === 1 ? '' : 's'} ran against a
                  different cache directory ({formatNumber(scope.otherRoots.hits)} hits ·{' '}
                  {formatNumber(scope.otherRoots.misses)} misses) and are excluded too.
                </span>
              )}
              {/*
              The partition's catch-all arm: tier on, root unrecorded. The
              writer can no longer produce one, so this line should never
              appear — which is exactly why it must exist rather than let those
              lookups fall through every bucket and disappear.
            */}
              {scope.unattributed.turns > 0 && (
                <span>
                  {(scope.legacy.turns > 0 || scope.otherRoots.turns > 0) && ' '}
                  {formatNumber(scope.unattributed.turns)} turn{scope.unattributed.turns === 1 ? '' : 's'} ran with the
                  cold tier on but recorded no cache directory ({formatNumber(scope.unattributed.hits)} hits ·{' '}
                  {formatNumber(scope.unattributed.misses)} misses) and cannot be attributed.
                </span>
              )}
            </p>
          )}
      </div>

      {cache.error ? (
        <Card>
          <CardContent className="text-destructive flex items-center gap-2 py-6 text-sm">
            <AlertCircle className="size-4 shrink-0" aria-hidden />
            {cache.error.message}
          </CardContent>
        </Card>
      ) : (
        <>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <StatTile
              label="Usage"
              icon={HardDrive}
              value={cache.loading ? <Skeleton className="h-8 w-24" /> : formatBytes(totalBytes)}
              sub={
                cache.loading ? (
                  <Skeleton className="h-4 w-32" />
                ) : (
                  // Over half the tier's bytes can be sidecars, so the split is
                  // spelled out rather than folded into one opaque total.
                  <span>
                    {quotaBytes > 0
                      ? `${formatPercent(quotaFraction)} of ${formatBytes(quotaBytes)} quota`
                      : 'no quota available'}
                    {` · ${formatBytes(blockBytes)} blocks + ${formatBytes(sidecarBytes)} sidecars`}
                  </span>
                )
              }
              footer={
                cache.loading ? (
                  <Skeleton className="h-2 w-full rounded-full" />
                ) : quotaBytes > 0 ? (
                  <UsageMeter fraction={quotaFraction} totalBytes={totalBytes} quotaBytes={quotaBytes} />
                ) : undefined
              }
            />
            <StatTile
              label="Objects"
              icon={Database}
              value={cache.loading ? <Skeleton className="h-8 w-16" /> : formatCount(objects)}
              sub={
                cache.loading ? (
                  <Skeleton className="h-4 w-40" />
                ) : (
                  `${formatCount(blocks)} prefix block${blocks === 1 ? '' : 's'} · ${formatCount(sidecars)} state sidecar${sidecars === 1 ? '' : 's'}`
                )
              }
            />
            <StatTile
              label="Hit rate"
              icon={Percent}
              value={cache.loading ? <Skeleton className="h-8 w-16" /> : formatRate(trendTotals.hits, lookups)}
              sub={
                cache.loading ? (
                  <Skeleton className="h-4 w-32" />
                ) : hasTrend ? (
                  `${formatCount(trendTotals.hits)} hits · ${formatCount(trendTotals.misses)} misses · this cache only`
                ) : (
                  'no lookups recorded for this cache'
                )
              }
            />
            <StatTile
              label="Oldest object"
              icon={Clock}
              value={
                cache.loading ? (
                  <Skeleton className="h-8 w-20" />
                ) : disk?.oldestMtime != null ? (
                  formatRelativeTime(disk.oldestMtime)
                ) : (
                  '—'
                )
              }
              sub={
                cache.loading ? (
                  <Skeleton className="h-4 w-28" />
                ) : disk?.newestMtime != null ? (
                  `newest ${formatRelativeTime(disk.newestMtime)}`
                ) : (
                  'empty tier'
                )
              }
            />
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <ChartCard title="Objects by age" subtitle="Prefix blocks AND state sidecars — everything evict removes">
              <ChartBody
                loading={cache.loading}
                error={cache.error}
                isEmpty={!hasObjects}
                empty={
                  <ChartEmpty
                    icon={Layers}
                    message="No cold-tier objects persisted yet."
                    hint={
                      restoreFamilies.length > 0
                        ? `Objects appear when a persistent paged cache is used by a restore-eligible family: ${restoreFamilies.join(', ')}.`
                        : 'Objects appear when a persistent paged cache is used by a restore-eligible model family.'
                    }
                  />
                }
              >
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={disk?.ageHistogram ?? []} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" vertical={false} />
                    <XAxis dataKey="label" tick={AXIS_TICK} tickLine={false} axisLine={false} />
                    <YAxis
                      tick={AXIS_TICK}
                      tickLine={false}
                      axisLine={false}
                      width={40}
                      allowDecimals={false}
                      tickFormatter={(v) => formatCount(Number(v))}
                    />
                    <Tooltip
                      cursor={{ fill: 'var(--color-muted)', opacity: 0.35 }}
                      contentStyle={TOOLTIP_CONTENT_STYLE}
                      formatter={(value, _name, item) => {
                        const bytes = Number((item as { payload?: { bytes?: number } }).payload?.bytes ?? 0);
                        return [`${formatNumber(Number(value))} objects · ${formatBytes(bytes)}`, 'Objects'];
                      }}
                    />
                    <Bar
                      dataKey="count"
                      name="Objects"
                      fill="var(--viz-input)"
                      radius={[4, 4, 0, 0]}
                      isAnimationActive={false}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </ChartBody>
            </ChartCard>

            <ChartCard
              title="Cache hit / miss trend"
              subtitle={
                scope !== undefined
                  ? `Lookups against THIS cache root, by day · last ${scope.trendWindowDays} days`
                  : 'Cold-tier lookups against this cache root, by day'
              }
            >
              <ChartBody
                loading={cache.loading}
                error={cache.error}
                isEmpty={!hasTrend}
                empty={
                  <ChartEmpty
                    icon={Percent}
                    message="No hits or misses recorded for this cache."
                    hint={
                      excludedLookups > 0
                        ? `${formatNumber(excludedLookups)} lookups were recorded against a different (or unattributed) cache root — see the scope note above.`
                        : 'Cold-cache lookups are recorded per turn when you run mlx agent with a persistent paged cache.'
                    }
                  />
                }
              >
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={trendData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" vertical={false} />
                    <XAxis dataKey="day" tick={AXIS_TICK} tickLine={false} axisLine={false} minTickGap={16} />
                    <YAxis
                      tick={AXIS_TICK}
                      tickLine={false}
                      axisLine={false}
                      width={40}
                      allowDecimals={false}
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
                      dataKey="hits"
                      name="Hits"
                      fill="var(--viz-output)"
                      stroke="var(--color-card)"
                      strokeWidth={1}
                      radius={[4, 4, 0, 0]}
                      isAnimationActive={false}
                    />
                    <Bar
                      dataKey="misses"
                      name="Misses"
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
          </div>

          {/*
            Cold-tier counters the native `coldCacheStats()` has always exposed
            and nothing shipped ever read. `corruptions == 0` is the stated
            acceptance bar for admitting a family to the restore allowlist, and
            until now it could not be checked from anything the user runs.
          */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Cold-tier health</CardTitle>
              <CardDescription>
                Per-turn counters recorded against this cache root. Corruptions are a SUBSET of misses (a failed
                validation always costs a miss), so never add the two. Writes, evictions and part of the drop count
                advance on a background writer thread, making those totals approximate.
              </CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              <HealthStat
                label="Corruptions"
                value={formatNumber(health?.corruptions ?? 0)}
                hint={
                  corruptionsSeen
                    ? `cumulative max ${formatNumber(health?.corruptionsTotal ?? 0)} — investigate`
                    : 'none seen (acceptance bar: 0)'
                }
                icon={corruptionsSeen ? ShieldAlert : ShieldCheck}
                alarm={corruptionsSeen}
                loading={cache.loading}
              />
              <HealthStat
                label="Queue drops"
                value={formatNumber(health?.queueDrops ?? 0)}
                hint={`cumulative max ${formatNumber(health?.queueDropsTotal ?? 0)} · ${formatNumber(health?.enqueued ?? 0)} enqueued`}
                alarm={(health?.queueDropsTotal ?? 0) > 0}
                loading={cache.loading}
              />
              <HealthStat
                label="Evictions"
                value={formatNumber(health?.evictions ?? 0)}
                hint="objects removed by quota pressure"
                loading={cache.loading}
              />
              <HealthStat
                label="Bytes restored"
                value={formatBytes(trendTotals.bytesRestored)}
                hint={`${formatBytes(trendTotals.bytesWritten)} written`}
                loading={cache.loading}
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Manage cold cache</CardTitle>
              <CardDescription>
                Removes persisted paged blocks AND their state sidecars from disk — both occupy the quota, so both go.
                This is safe (they are re-created on demand) but a cleared cache means the next matching prefix pays
                full prefill again.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
              <div className="flex flex-wrap items-end gap-2">
                <div className="space-y-1.5">
                  <label className="text-muted-foreground text-xs font-medium" htmlFor="evict-threshold">
                    Evict objects
                  </label>
                  <Select value={evictKey} onValueChange={setEvictKey}>
                    <SelectTrigger id="evict-threshold" className="w-48" aria-label="Eviction age threshold">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {EVICT_OPTIONS.map((o) => (
                        <SelectItem key={o.value} value={o.value}>
                          {o.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <Button
                  variant="outline"
                  disabled={cache.loading || evictP.count === 0}
                  onClick={() => setPending({ kind: 'evict', days: evictDays })}
                >
                  Evict older
                </Button>
              </div>
              <Button
                variant="destructive"
                disabled={cache.loading || !hasObjects}
                onClick={() => setPending({ kind: 'clear' })}
              >
                <Trash2 className="size-4" />
                Clear all
              </Button>
            </CardContent>
          </Card>
        </>
      )}

      <Dialog open={pending !== null} onOpenChange={(open) => !open && !busy && setPending(null)}>
        <DialogContent>
          <DialogHeader>
            {/* "objects", not "blocks": the number under this title is
                `evictP.count`, an object count, and the body says so too. */}
            <DialogTitle>
              {pending?.kind === 'evict' ? 'Evict old cache objects?' : 'Clear the cold cache?'}
            </DialogTitle>
            <DialogDescription>
              {pending?.kind === 'evict' ? (
                <>
                  Removes cold-tier objects (blocks and state sidecars) older than{' '}
                  <span className="text-foreground font-medium">
                    {pending.days} day{pending.days === 1 ? '' : 's'}
                  </span>
                  . This frees an estimated{' '}
                  <span className="text-foreground font-medium">
                    {formatNumber(evictP.count)} object{evictP.count === 1 ? '' : 's'} · {formatBytes(evictP.bytes)}
                  </span>
                  . Newer objects are kept.
                </>
              ) : (
                <>
                  Removes all{' '}
                  <span className="text-foreground font-medium">
                    {formatNumber(objects)} object{objects === 1 ? '' : 's'}
                  </span>{' '}
                  — {formatNumber(blocks)} prefix block{blocks === 1 ? '' : 's'} and {formatNumber(sidecars)} state
                  sidecar{sidecars === 1 ? '' : 's'}, {formatBytes(totalBytes)} in total — from the cold tier. This
                  cannot be undone.
                </>
              )}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <DialogClose asChild>
              <Button variant="outline" disabled={busy}>
                Cancel
              </Button>
            </DialogClose>
            <Button variant="destructive" disabled={busy} onClick={() => pending !== null && void runDelete(pending)}>
              {busy && <Loader2 className="size-4 animate-spin" />}
              {pending?.kind === 'evict' ? 'Evict' : 'Clear all'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

/**
 * One cold-tier health counter. Deliberately plain: these are diagnostics, and
 * an `alarm` counter is tinted destructive so a non-zero corruption count reads
 * as a finding rather than as another number.
 */
function HealthStat({
  label,
  value,
  hint,
  icon: Icon,
  alarm = false,
  loading,
}: {
  label: string;
  value: string;
  hint: string;
  icon?: LucideIcon;
  alarm?: boolean;
  loading: boolean;
}) {
  return (
    <div className="space-y-1">
      <p className="text-muted-foreground flex items-center gap-1.5 text-xs font-medium">
        {Icon !== undefined && <Icon className={cn('size-3.5', alarm && 'text-destructive')} aria-hidden />}
        {label}
      </p>
      {loading ? (
        <Skeleton className="h-6 w-16" />
      ) : (
        <p className={cn('text-xl leading-none font-semibold', alarm && 'text-destructive')}>{value}</p>
      )}
      <p className="text-muted-foreground text-xs">{hint}</p>
    </div>
  );
}

/** Usage-vs-quota bar with an accessible progressbar role. */
function UsageMeter({
  fraction,
  totalBytes,
  quotaBytes,
}: {
  fraction: number;
  totalBytes: number;
  quotaBytes: number;
}) {
  // Saturated exactly like the visible label, so the announced value can never
  // say 100 while the tile reads `>99%`.
  const pct = percentInt(fraction);
  return (
    <div
      className="bg-secondary h-2 w-full overflow-hidden rounded-full"
      role="progressbar"
      aria-valuenow={pct}
      aria-valuetext={formatPercent(fraction)}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={`Cold cache using ${formatBytes(totalBytes)} of ${formatBytes(quotaBytes)} quota`}
    >
      <div className="bg-primary h-full rounded-full" style={{ width: `${pct}%` }} />
    </div>
  );
}
