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
import { formatBytes, formatCount, formatNumber, formatPercent, formatRelativeTime } from '@/lib/format';
import type { CacheMutationResult, CacheResponse, ColdCacheDiskInfo } from '@/lib/types';
import { useJson } from '@/lib/use-api';
import { AlertCircle, Clock, Database, HardDrive, Layers, Loader2, Percent, Trash2 } from 'lucide-react';
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

/**
 * Predicted eviction for a day threshold, summed from the age histogram. The
 * evict thresholds (1/7/30) align to the histogram bucket boundaries, so this is
 * exact at scan time (the server removes by mtime at request time).
 */
function evictPreview(hist: ColdCacheDiskInfo['ageHistogram'], days: number): { count: number; bytes: number } {
  const startIndex = days >= 30 ? 3 : days >= 7 ? 2 : 1;
  let count = 0;
  let bytes = 0;
  for (let i = startIndex; i < hist.length; i++) {
    count += hist[i].count;
    bytes += hist[i].bytes;
  }
  return { count, bytes };
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
  const entryCount = disk?.entryCount ?? 0;
  const quotaFraction = quotaBytes > 0 ? Math.min(1, totalBytes / quotaBytes) : 0;

  const trendTotals = useMemo(
    () => trend.reduce((acc, row) => ({ hits: acc.hits + row.hits, misses: acc.misses + row.misses }), { hits: 0, misses: 0 }),
    [trend],
  );
  const lookups = trendTotals.hits + trendTotals.misses;
  const hitRate = lookups > 0 ? trendTotals.hits / lookups : null;

  const trendData = useMemo(() => trend.map((row) => ({ day: formatDay(row.day), hits: row.hits, misses: row.misses })), [trend]);
  const hasTrend = lookups > 0;
  const hasBlocks = entryCount > 0;

  const evictP = disk !== undefined ? evictPreview(disk.ageHistogram, evictDays) : { count: 0, bytes: 0 };

  const runDelete = async (action: Exclude<PendingAction, null>): Promise<void> => {
    setBusy(true);
    try {
      const body = action.kind === 'evict' ? { olderThanDays: action.days } : undefined;
      const result = await mutate<CacheMutationResult>('DELETE', '/cache', body);
      toast.success(action.kind === 'evict' ? `Evicted blocks older than ${action.days} day${action.days === 1 ? '' : 's'}` : 'Cleared cold cache', {
        description: `${formatNumber(result.removed)} block${result.removed === 1 ? '' : 's'} removed · ${formatBytes(result.freedBytes)} freed`,
      });
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
          <p className="text-muted-foreground mt-1 font-mono text-xs break-all" title={disk.root}>
            {disk.root}
            {!disk.exists && ' · not created yet'}
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
                ) : quotaBytes > 0 ? (
                  `${formatPercent(quotaFraction)} of ${formatBytes(quotaBytes)} quota`
                ) : (
                  'no quota available'
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
              label="Blocks"
              icon={Database}
              value={cache.loading ? <Skeleton className="h-8 w-16" /> : formatCount(entryCount)}
              sub={cache.loading ? <Skeleton className="h-4 w-28" /> : 'persisted prefix blocks'}
            />
            <StatTile
              label="Hit rate"
              icon={Percent}
              value={cache.loading ? <Skeleton className="h-8 w-16" /> : hitRate !== null ? formatPercent(hitRate) : '—'}
              sub={
                cache.loading ? (
                  <Skeleton className="h-4 w-32" />
                ) : hitRate !== null ? (
                  `${formatCount(trendTotals.hits)} hits · ${formatCount(trendTotals.misses)} misses`
                ) : (
                  'no lookups recorded'
                )
              }
            />
            <StatTile
              label="Oldest block"
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
            <ChartCard title="Blocks by age" subtitle="Persisted block count by age bucket">
              <ChartBody
                loading={cache.loading}
                error={cache.error}
                isEmpty={!hasBlocks}
                empty={
                  <ChartEmpty
                    icon={Layers}
                    message="No blocks persisted yet."
                    hint="Cold-tier blocks appear when a persistent paged cache is used (qwen3 dense)."
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
                        return [`${formatNumber(Number(value))} blocks · ${formatBytes(bytes)}`, 'Blocks'];
                      }}
                    />
                    <Bar dataKey="count" name="Blocks" fill="var(--viz-input)" radius={[4, 4, 0, 0]} isAnimationActive={false} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartBody>
            </ChartCard>

            <ChartCard title="Cache hit / miss trend" subtitle="Cold-tier lookups by day (from trace deltas)">
              <ChartBody
                loading={cache.loading}
                error={cache.error}
                isEmpty={!hasTrend}
                empty={
                  <ChartEmpty
                    icon={Percent}
                    message="No hits or misses recorded."
                    hint="Cold-cache lookups are recorded per turn when you run mlx agent with a persistent paged cache."
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
                    <Bar dataKey="hits" name="Hits" fill="var(--viz-output)" stroke="var(--color-card)" strokeWidth={1} radius={[4, 4, 0, 0]} isAnimationActive={false} />
                    <Bar dataKey="misses" name="Misses" fill="var(--viz-cached)" stroke="var(--color-card)" strokeWidth={1} radius={[4, 4, 0, 0]} isAnimationActive={false} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartBody>
            </ChartCard>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Manage cold cache</CardTitle>
              <CardDescription>
                Removes persisted paged blocks from disk. This is safe — blocks are re-created on demand — but a cleared
                cache means the next matching prefix pays full prefill again.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
              <div className="flex flex-wrap items-end gap-2">
                <div className="space-y-1.5">
                  <label className="text-muted-foreground text-xs font-medium" htmlFor="evict-threshold">
                    Evict blocks
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
                disabled={cache.loading || !hasBlocks}
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
            <DialogTitle>{pending?.kind === 'evict' ? 'Evict old cache blocks?' : 'Clear the cold cache?'}</DialogTitle>
            <DialogDescription>
              {pending?.kind === 'evict' ? (
                <>
                  Removes blocks older than{' '}
                  <span className="text-foreground font-medium">
                    {pending.days} day{pending.days === 1 ? '' : 's'}
                  </span>
                  . This frees an estimated{' '}
                  <span className="text-foreground font-medium">
                    {formatNumber(evictP.count)} block{evictP.count === 1 ? '' : 's'} · {formatBytes(evictP.bytes)}
                  </span>
                  . Newer blocks are kept.
                </>
              ) : (
                <>
                  Removes all{' '}
                  <span className="text-foreground font-medium">
                    {formatNumber(entryCount)} block{entryCount === 1 ? '' : 's'} · {formatBytes(totalBytes)}
                  </span>{' '}
                  from the cold tier. This cannot be undone.
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

/** Usage-vs-quota bar with an accessible progressbar role. */
function UsageMeter({ fraction, totalBytes, quotaBytes }: { fraction: number; totalBytes: number; quotaBytes: number }) {
  const pct = Math.round(fraction * 100);
  return (
    <div
      className="bg-secondary h-2 w-full overflow-hidden rounded-full"
      role="progressbar"
      aria-valuenow={pct}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={`Cold cache using ${formatBytes(totalBytes)} of ${formatBytes(quotaBytes)} quota`}
    >
      <div className="bg-primary h-full rounded-full" style={{ width: `${pct}%` }} />
    </div>
  );
}
