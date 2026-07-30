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
  const writeErrorsSeen = (health?.writeErrorsTotal ?? 0) > 0 || (health?.writeErrors ?? 0) > 0;
  // The two ways reuse is refused, summed for the hit-rate tile only: both end
  // with the prefix recomputed, and at the tile's altitude the operator needs
  // to know a refusal happened at all. The health card below keeps them apart,
  // because the causes are unrelated (the walk found no backed boundary vs a
  // family threw away a prefix it had).
  const refusals = (health?.restoreDeclines ?? 0) + (health?.restoreSuppressed ?? 0);

  const sidecarCaptures = health?.sidecarCaptureReached ?? 0;
  const sidecarInstalled = health?.sidecarInstalled ?? 0;
  const sidecarPersisted = health?.sidecarAlreadyPersisted ?? 0;
  const sidecarWritten = health?.sidecarEnqueued ?? 0;
  // Chain-empty and boundary-skip are ONE number here on purpose: both end with
  // the capture holding no rung to anchor state under, and the operator's next
  // move is the same either way. Which of the two it was goes to the
  // `[MLX_TRACE] paged` line, and both stay separate over the wire.
  const sidecarNoAnchor = (health?.sidecarChainEmpty ?? 0) + (health?.sidecarBoundarySkips ?? 0);
  // The regression `installed` exists to expose. Prefixes WERE restored (real
  // hits) and the family's state WAS on disk (the capture re-found the same
  // chain), yet nothing was installed — so every restored turn silently
  // replayed the whole prefix. The replay produces CORRECT state, which is why
  // every other number on this page reads exactly like a healthy run.
  const sidecarReplayed = sidecarInstalled === 0 && sidecarPersisted > 0 && trendTotals.hits > 0;
  // The write-side dead end: the capture ran and never once wrote a sidecar nor
  // found one already there, so no prefix will ever restore state.
  // `alreadyPersisted` is the discriminator — without it the healthy steady
  // state (nothing enqueued because the FIRST turn wrote the chain every later
  // turn re-selects) carries the identical signature.
  const sidecarStranded = sidecarCaptures > 0 && sidecarWritten === 0 && sidecarPersisted === 0;
  // Captures that ran against NO cache root. The recorder sits above the
  // capture's `adapter.cold_tier()` guard, so a hybrid turn with persistence
  // off — or one whose tier failed to open — counts one and then carries no
  // root for the scope to match. Reading that zero as "dense family" names the
  // wrong cause for the one state these counters exist to catch, so it takes
  // the hint over.
  const sidecarUnrooted = scope?.unrootedSidecarCaptures ?? 0;

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
          {/*
            Every bar in this grid is `h-[1lh]` — one line box of the font its own
            wrapper carries. StatTile renders `value` inside `text-3xl leading-none`
            and `sub` inside `text-sm`, so the bars measure 30px and 20px without
            any call site restating those numbers, and they stay right if the type
            scale moves. The fixed heights they replace did not match the text
            (`h-8` is 2px over the value, `h-4` 4px under the sub-line), so all four
            tiles grew as the request landed and took the charts below with them.
            The meter is the exception and needs no `1lh`: UsageMeter is itself
            `h-2`, so the bar already stands in its exact box.
          */}
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <StatTile
              label="Usage"
              icon={HardDrive}
              value={cache.loading ? <Skeleton className="h-[1lh] w-24" /> : formatBytes(totalBytes)}
              sub={
                cache.loading ? (
                  <Skeleton className="h-[1lh] w-32" />
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
              value={cache.loading ? <Skeleton className="h-[1lh] w-16" /> : formatCount(objects)}
              sub={
                cache.loading ? (
                  <Skeleton className="h-[1lh] w-40" />
                ) : (
                  `${formatCount(blocks)} prefix block${blocks === 1 ? '' : 's'} · ${formatCount(sidecars)} state sidecar${sidecars === 1 ? '' : 's'}`
                )
              }
            />
            <StatTile
              label="Hit rate"
              icon={Percent}
              value={cache.loading ? <Skeleton className="h-[1lh] w-16" /> : formatRate(trendTotals.hits, lookups)}
              sub={
                cache.loading ? (
                  <Skeleton className="h-[1lh] w-32" />
                ) : hasTrend ? (
                  `${formatCount(trendTotals.hits)} hits · ${formatCount(trendTotals.misses)} misses · this cache only`
                ) : refusals > 0 ? (
                  // A refused restore performs no lookup, so it lands as 0/0
                  // and used to render as the empty state below — telling an
                  // operator "nothing ran" when what happened was "reuse was
                  // refused". Those are opposite diagnoses.
                  `${formatCount(refusals)} restore${refusals === 1 ? '' : 's'} refused · no lookup was performed`
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
                  <Skeleton className="h-[1lh] w-20" />
                ) : disk?.oldestMtime != null ? (
                  formatRelativeTime(disk.oldestMtime)
                ) : (
                  '—'
                )
              }
              sub={
                cache.loading ? (
                  <Skeleton className="h-[1lh] w-28" />
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
                validation always costs a miss), so never add the two. Declines are NEITHER — a refused restore performs
                no lookup, so it moves neither hits nor misses. Sidecars are the family state kept outside the paged
                pool; the sidecar figures on the queue tile are a SHARE of the object counts beside them, since one
                admission bumps both. Writes, evictions and write errors advance on a background writer thread, making
                those totals approximate.
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
                // The value and `enqueued` are object-scoped — blocks and family
                // state sidecars share this queue, so "objects" is what they
                // count. The two sidecar figures are the sidecar SHARE of those
                // same numbers, phrased "among them" / "of the drops" so nobody
                // reads them as a second population to add.
                hint={`cumulative max ${formatNumber(health?.queueDropsTotal ?? 0)} · ${formatNumber(health?.enqueued ?? 0)} objects enqueued · ${formatNumber(sidecarWritten)} sidecar${sidecarWritten === 1 ? '' : 's'} among them, ${formatNumber(health?.sidecarQueueDrops ?? 0)} of the drops`}
                alarm={(health?.queueDropsTotal ?? 0) > 0}
                loading={cache.loading}
              />
              <HealthStat
                label="Write errors"
                value={formatNumber(health?.writeErrors ?? 0)}
                hint={
                  writeErrorsSeen
                    ? `cumulative max ${formatNumber(health?.writeErrorsTotal ?? 0)} — the cache root is refusing writes`
                    : 'none seen (writes are landing)'
                }
                icon={writeErrorsSeen ? ShieldAlert : ShieldCheck}
                alarm={writeErrorsSeen}
                loading={cache.loading}
              />
              {/*
                The write side of the family state that lives OUTSIDE the paged
                pool. Only the hybrid families enter this capture, so a zero
                beside real block traffic is the normal dense-model reading and
                deliberately wears no alarm — but only once the rootless
                captures are known to be zero too, since a hybrid that never got
                a tier produces the same zero here for the opposite reason.
              */}
              <HealthStat
                label="Sidecar captures"
                value={formatNumber(sidecarCaptures)}
                hint={
                  sidecarCaptures === 0
                    ? sidecarUnrooted > 0
                      ? `${formatNumber(sidecarUnrooted)} ran with no cache attached — persistence off, or the tier never opened`
                      : 'not reached — dense families keep no state outside the pool'
                    : sidecarStranded
                      ? 'nothing written and nothing on disk — no prefix can restore state'
                      : `${formatNumber(sidecarWritten)} written · ${formatNumber(sidecarPersisted)} already on disk · ${formatNumber(sidecarNoAnchor)} found no anchor`
                }
                icon={sidecarStranded ? ShieldAlert : undefined}
                alarm={sidecarStranded}
                loading={cache.loading}
              />
              <HealthStat
                label="Restore declines"
                value={formatNumber(health?.restoreDeclines ?? 0)}
                hint={`${formatNumber(health?.restoreSuppressed ?? 0)} suppressed after restore · neither counts as a lookup`}
                loading={cache.loading}
              />
              {/*
                The one cold-tier counter nothing else on this page can stand in
                for. Every `install_*_cold_sidecar` early-return falls through to
                a full O(prefix) replay that produces CORRECT state, so a
                regression from "restored and INSTALLED" to "restored and
                silently re-derived" leaves the hit rate, the trend, the bytes
                and the corruption count all exactly where they were.
              */}
              <HealthStat
                label="Sidecar installs"
                value={formatNumber(sidecarInstalled)}
                hint={
                  sidecarReplayed
                    ? 'state was on disk and prefixes restored — every one re-derived by a full replay'
                    : sidecarInstalled > 0
                      ? 'restored prefixes that reused family state instead of replaying it'
                      : 'no restored prefix carried family state in this window'
                }
                icon={sidecarReplayed ? ShieldAlert : undefined}
                alarm={sidecarReplayed}
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
                hint={`${formatBytes(trendTotals.bytesWritten)} landed`}
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
      {/*
        The value keeps its own wrapper while loading rather than being replaced
        by a bare bar, so the placeholder stands in the box the number will occupy
        instead of one that merely looks similar: `1lh` resolves against this
        element's own line-height, which is what `text-xl leading-none` sets. The
        `h-6` bar it replaces was 4px taller than that line, and this card renders
        eight of these in a two-row grid, so both rows and everything under them
        stepped down when the request landed.

        A `div` rather than the `<p>` it was, for the same reason StatTile's value
        is one: a paragraph may not contain the bar's `div`, and under preflight
        the two elements produce the identical box anyway.
      */}
      <div className={cn('text-xl leading-none font-semibold', alarm && 'text-destructive')}>
        {loading ? <Skeleton className="h-[1lh] w-16" /> : value}
      </div>
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
