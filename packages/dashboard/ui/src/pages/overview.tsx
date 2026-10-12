import { StatTile } from '@/components/stat-tile';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { formatBytes, formatCount, formatPercent, formatRate, percentInt } from '@/lib/format';
import type { CacheResponse, DownloadsResponse, ModelsResponse } from '@/lib/types';
import { useJson } from '@/lib/use-api';
import { AlertCircle, ArrowRight, Boxes, Download, HardDrive } from 'lucide-react';
import { Link } from 'react-router-dom';

function TileError({ message }: { message: string }) {
  return (
    <span className="text-destructive flex items-center gap-1.5 text-sm">
      <AlertCircle className="size-4 shrink-0" aria-hidden />
      {message}
    </span>
  );
}

export default function Overview() {
  const models = useJson<ModelsResponse>('/models');
  const cache = useJson<CacheResponse>('/cache');
  const downloads = useJson<DownloadsResponse>('/downloads');

  const modelCount = models.data?.models.length ?? 0;
  const companions = models.data?.companions ?? [];
  const modelBytes = [...(models.data?.models ?? []), ...companions].reduce((sum, m) => sum + m.sizeBytes, 0);

  const disk = cache.data?.disk;
  const cacheBytes = disk?.totalBytes ?? 0;
  const quotaBytes = disk?.quotaBytes ?? 0;
  const quotaFraction = quotaBytes > 0 ? Math.min(1, cacheBytes / quotaBytes) : 0;
  // `trend` is scoped server-side to the cache root shown on the Cache page, so
  // this tile reports THAT cache's reuse — not a sum over every cache directory
  // the machine has ever used.
  const cacheTotals = cache.data?.trend.reduce(
    (acc, row) => ({ hits: acc.hits + row.hits, misses: acc.misses + row.misses }),
    { hits: 0, misses: 0 },
  );
  const cacheHits = cacheTotals?.hits ?? 0;
  const cacheLookups = cacheTotals ? cacheTotals.hits + cacheTotals.misses : 0;

  const jobs = downloads.data?.jobs ?? [];
  const jobsRunning = jobs.filter((job) => job.state === 'running').length;
  const jobsDone = jobs.filter((job) => job.state === 'done').length;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Overview</h1>
          <p className="text-muted-foreground text-sm">Models, downloads, and cache at a glance.</p>
        </div>
        {!models.loading && !models.error && models.data && modelCount === 0 && (
          <Button asChild>
            <Link to="/welcome">
              Choose your first model <ArrowRight className="size-4" aria-hidden />
            </Link>
          </Button>
        )}
      </div>

      {/*
        Every bar below is `h-[1lh]`: one line box of the font its OWN wrapper
        carries. StatTile renders `value` inside `text-3xl leading-none` and
        `sub` inside `text-sm`, so the bar resolves to 30px and 20px without
        anyone having to know that — and it stays right if either font size ever
        changes. The hand-picked heights it replaces did not measure the same as
        the text: `h-8` was 2px taller than the value and `h-4` 4px shorter than
        the sub-line, so all four tiles grew 6px when their request landed, the
        grid row grew with them, and the card underneath jumped four times over
        as the four independent requests resolved.
      */}
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        <StatTile
          label="Local models"
          icon={Boxes}
          value={models.loading ? <Skeleton className="h-[1lh] w-16" /> : formatCount(modelCount)}
          sub={
            models.error ? (
              <TileError message="Failed to load models" />
            ) : models.loading ? (
              <Skeleton className="h-[1lh] w-24" />
            ) : (
              `${formatBytes(modelBytes)} on disk${companions.length > 0 ? ', including companions' : ''}`
            )
          }
        />

        <StatTile
          label="Cold cache"
          icon={HardDrive}
          value={cache.loading ? <Skeleton className="h-[1lh] w-20" /> : formatBytes(cacheBytes)}
          sub={
            cache.error ? (
              <TileError message="Failed to load cache" />
            ) : cache.loading ? (
              <Skeleton className="h-[1lh] w-40" />
            ) : (
              <span>
                {quotaBytes > 0 ? `${formatPercent(quotaFraction)} of ${formatBytes(quotaBytes)}` : 'no quota'}
                {cacheLookups > 0 ? ` · hit rate ${formatRate(cacheHits, cacheLookups)}` : ''}
              </span>
            )
          }
          footer={
            cache.loading ? (
              <Skeleton className="h-2 w-full rounded-full" />
            ) : cache.error ? undefined : (
              <div className="bg-secondary h-2 w-full overflow-hidden rounded-full" aria-hidden>
                <div className="bg-primary h-full rounded-full" style={{ width: `${percentInt(quotaFraction)}%` }} />
              </div>
            )
          }
        />

        <StatTile
          label="Active downloads"
          icon={Download}
          value={downloads.loading ? <Skeleton className="h-[1lh] w-12" /> : formatCount(jobsRunning)}
          sub={
            downloads.error ? (
              <TileError message="Failed to load downloads" />
            ) : downloads.loading ? (
              <Skeleton className="h-[1lh] w-28" />
            ) : jobsRunning > 0 || jobsDone > 0 ? (
              <span>
                {jobsDone > 0 ? `${formatCount(jobsDone)} completed · ` : ''}
                <Link to="/models" className="underline underline-offset-2">
                  manage
                </Link>
              </span>
            ) : (
              <span>
                None running ·{' '}
                <Link to="/models" className="underline underline-offset-2">
                  install a model
                </Link>
              </span>
            )
          }
        />
      </div>
    </div>
  );
}
