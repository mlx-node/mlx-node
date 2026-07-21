import { StatTile } from '@/components/stat-tile';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { formatBytes, formatCount, formatPercent, formatRelativeTime } from '@/lib/format';
import type {
  CacheResponse,
  DownloadsResponse,
  MetricsOverviewResponse,
  ModelsResponse,
  SessionRow,
  SessionsResponse,
} from '@/lib/types';
import { useJson } from '@/lib/use-api';
import { AlertCircle, ArrowRight, Boxes, Download, HardDrive, Inbox, MessagesSquare } from 'lucide-react';
import { useMemo } from 'react';
import { Link } from 'react-router-dom';

const DAY_MS = 24 * 60 * 60 * 1000;
const RECENT_LIMIT = 6;

function TileError({ message }: { message: string }) {
  return (
    <span className="text-destructive flex items-center gap-1.5 text-sm">
      <AlertCircle className="size-4 shrink-0" aria-hidden />
      {message}
    </span>
  );
}

export default function Overview() {
  const sevenDaysAgo = useMemo(() => Date.now() - 7 * DAY_MS, []);
  const models = useJson<ModelsResponse>('/models');
  const sessions = useJson<SessionsResponse>('/sessions');
  const metrics = useJson<MetricsOverviewResponse>(`/metrics/overview?from=${sevenDaysAgo}`);
  const cache = useJson<CacheResponse>('/cache');
  const downloads = useJson<DownloadsResponse>('/downloads');

  const modelCount = models.data?.models.length ?? 0;
  const modelBytes = models.data?.models.reduce((sum, m) => sum + m.sizeBytes, 0) ?? 0;

  const sessionCount = sessions.data?.sessions.length ?? 0;
  const tokens7d = metrics.data ? metrics.data.totals.inputTokens + metrics.data.totals.outputTokens : 0;

  const disk = cache.data?.disk;
  const cacheBytes = disk?.totalBytes ?? 0;
  const quotaBytes = disk?.quotaBytes ?? 0;
  const quotaFraction = quotaBytes > 0 ? Math.min(1, cacheBytes / quotaBytes) : 0;
  const cacheTotals = cache.data?.trend.reduce(
    (acc, row) => ({ hits: acc.hits + row.hits, misses: acc.misses + row.misses }),
    { hits: 0, misses: 0 },
  );
  const cacheLookups = cacheTotals ? cacheTotals.hits + cacheTotals.misses : 0;
  const hitRate = cacheLookups > 0 ? cacheTotals!.hits / cacheLookups : null;

  const jobs = downloads.data?.jobs ?? [];
  const jobsRunning = jobs.filter((job) => job.state === 'running').length;
  const jobsDone = jobs.filter((job) => job.state === 'done').length;

  const recent = sessions.data?.sessions.slice(0, RECENT_LIMIT) ?? [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Overview</h1>
        <p className="text-muted-foreground text-sm">Models, sessions, tokens, and cache at a glance.</p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatTile
          label="Local models"
          icon={Boxes}
          value={models.loading ? <Skeleton className="h-8 w-16" /> : formatCount(modelCount)}
          sub={
            models.error ? (
              <TileError message="Failed to load models" />
            ) : models.loading ? (
              <Skeleton className="h-4 w-24" />
            ) : (
              `${formatBytes(modelBytes)} on disk`
            )
          }
        />

        <StatTile
          label="Sessions"
          icon={MessagesSquare}
          value={sessions.loading ? <Skeleton className="h-8 w-16" /> : formatCount(sessionCount)}
          sub={
            sessions.error ? (
              <TileError message="Failed to load sessions" />
            ) : metrics.error ? (
              <TileError message="Tokens unavailable" />
            ) : sessions.loading || metrics.loading ? (
              <Skeleton className="h-4 w-32" />
            ) : (
              `${formatCount(tokens7d)} tokens · last 7 days`
            )
          }
        />

        <StatTile
          label="Cold cache"
          icon={HardDrive}
          value={cache.loading ? <Skeleton className="h-8 w-20" /> : formatBytes(cacheBytes)}
          sub={
            cache.error ? (
              <TileError message="Failed to load cache" />
            ) : cache.loading ? (
              <Skeleton className="h-4 w-40" />
            ) : (
              <span>
                {quotaBytes > 0 ? `${formatPercent(quotaFraction)} of ${formatBytes(quotaBytes)}` : 'no quota'}
                {hitRate !== null ? ` · hit rate ${formatPercent(hitRate)}` : ''}
              </span>
            )
          }
          footer={
            cache.loading ? (
              <Skeleton className="h-2 w-full rounded-full" />
            ) : cache.error ? undefined : (
              <div className="bg-secondary h-2 w-full overflow-hidden rounded-full" aria-hidden>
                <div
                  className="bg-primary h-full rounded-full"
                  style={{ width: `${Math.round(quotaFraction * 100)}%` }}
                />
              </div>
            )
          }
        />

        <StatTile
          label="Active downloads"
          icon={Download}
          value={downloads.loading ? <Skeleton className="h-8 w-12" /> : formatCount(jobsRunning)}
          sub={
            downloads.error ? (
              <TileError message="Failed to load downloads" />
            ) : downloads.loading ? (
              <Skeleton className="h-4 w-28" />
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

      <Card>
        <CardHeader>
          <CardTitle>Recent sessions</CardTitle>
        </CardHeader>
        <CardContent className="px-0">
          {sessions.error ? (
            <div className="text-destructive flex items-center gap-2 px-6 text-sm">
              <AlertCircle className="size-4 shrink-0" aria-hidden />
              {sessions.error.message}
            </div>
          ) : sessions.loading ? (
            <div className="space-y-3 px-6">
              {Array.from({ length: 4 }).map((_, i) => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          ) : recent.length === 0 ? (
            <div className="text-muted-foreground flex flex-col items-center gap-2 px-6 py-10 text-sm">
              <Inbox className="size-6" aria-hidden />
              No sessions recorded yet.
            </div>
          ) : (
            <ul className="divide-border divide-y">
              {recent.map((session) => (
                <RecentSessionRow key={session.id} session={session} />
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function RecentSessionRow({ session }: { session: SessionRow }) {
  const title = session.name ?? session.firstMessage ?? session.id;
  const tokens = session.inputTokens + session.outputTokens;
  const shownModels = session.models.slice(0, 2);
  const extraModels = session.models.length - shownModels.length;

  return (
    <li>
      <Link
        to={`/sessions/${encodeURIComponent(session.id)}`}
        className="hover:bg-muted/50 group flex items-center gap-3 px-6 py-3 transition-colors"
      >
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium">{title}</p>
          <div className="text-muted-foreground mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs">
            <span>{formatRelativeTime(session.modified)}</span>
            {tokens > 0 && (
              <>
                <span aria-hidden>·</span>
                <span className="tabular-nums">{formatCount(tokens)} tokens</span>
              </>
            )}
            {shownModels.map((model) => (
              <Badge key={model} variant="secondary" className="font-normal">
                {model}
              </Badge>
            ))}
            {extraModels > 0 && <span>+{extraModels}</span>}
          </div>
        </div>
        <ArrowRight
          className="text-muted-foreground size-4 shrink-0 transition-transform group-hover:translate-x-0.5"
          aria-hidden
        />
      </Link>
    </li>
  );
}
