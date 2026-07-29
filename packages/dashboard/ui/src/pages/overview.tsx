import { StatTile } from '@/components/stat-tile';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { formatBytes, formatCount, formatPercent, formatRate, formatRelativeTime, percentInt } from '@/lib/format';
import type { CacheResponse, DownloadsResponse, ModelsResponse, SessionRow, SessionsResponse } from '@/lib/types';
import { useJson } from '@/lib/use-api';
import { cn } from '@/lib/utils';
import { AlertCircle, ArrowRight, Boxes, Download, HardDrive, Inbox, MessagesSquare } from 'lucide-react';
import { Link } from 'react-router-dom';

const RECENT_LIMIT = 6;

/**
 * The three class strings that decide a recent-session row's box.
 *
 * The loaded row and the row that stands in for it while `/sessions` is in
 * flight have to measure the same, so the padding, the gaps and the two font
 * sizes are written once and shared rather than retyped in both places — retyped
 * they drift the first time anyone adjusts the row, and the drift is invisible
 * until the list swaps and the card below it moves.
 */
const ROW_BOX = 'flex items-center gap-3 px-6 py-3';
const ROW_TITLE = 'truncate text-sm font-medium';
const ROW_META = 'text-muted-foreground mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs';

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
  const sessions = useJson<SessionsResponse>('/sessions');
  const cache = useJson<CacheResponse>('/cache');
  const downloads = useJson<DownloadsResponse>('/downloads');

  const modelCount = models.data?.models.length ?? 0;
  const modelBytes = models.data?.models.reduce((sum, m) => sum + m.sizeBytes, 0) ?? 0;

  // Both halves of the Sessions tile come from ONE response, so they describe
  // ONE set of sessions. `total` is the true match total (not the capped page
  // length) and `tokens` is the server-side total over exactly those sessions,
  // counting each distinct inference once. It is not the sum of the per-row
  // token columns: those are raw per-session sums, so a forked session's
  // inherited turns appear in both rows while the tile bills them once.
  //
  // The subtitle used to read `/metrics/overview`, which is machine-wide: it
  // sums `traces` from `~/.mlx-node/metrics/traces` regardless of which
  // `--session-dir` the dashboard is listing. That rendered "Sessions 3" beside
  // a 7-day token total from every session directory the machine had ever used
  // — a number that did not move when the dashboard was pointed somewhere else.
  // Same defect as the Cache page's unscoped trend, same fix: scope the number,
  // and say in the tile which sessions it covers.
  const sessionCount = sessions.data?.total ?? 0;
  const sessionTokens = sessions.data?.tokens ?? 0;

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

  const recent = sessions.data?.sessions.slice(0, RECENT_LIMIT) ?? [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Overview</h1>
        <p className="text-muted-foreground text-sm">Models, sessions, tokens, and cache at a glance.</p>
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
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
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
              `${formatBytes(modelBytes)} on disk`
            )
          }
        />

        <StatTile
          label="Sessions"
          icon={MessagesSquare}
          value={sessions.loading ? <Skeleton className="h-[1lh] w-16" /> : formatCount(sessionCount)}
          sub={
            sessions.error ? (
              <TileError message="Failed to load sessions" />
            ) : sessions.loading ? (
              <Skeleton className="h-[1lh] w-32" />
            ) : (
              `${formatCount(sessionTokens)} tokens · these sessions only`
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
            <ul className="divide-border divide-y">
              {Array.from({ length: RECENT_LIMIT }).map((_, i) => (
                <RecentSessionRowSkeleton key={i} />
              ))}
            </ul>
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

/**
 * The row {@link RecentSessionRow} will become, with its text blanked.
 *
 * It is the same `<li>` in the same `<ul>` behind the same divider, so the list
 * it forms is the height of the list that replaces it — the four free-standing
 * `h-12` bars this replaces stacked to 228px against roughly 400px of real rows,
 * and the card grew by a third of a screen the moment `/sessions` answered. The
 * row count is `RECENT_LIMIT` for the same reason: it is the most rows `recent`
 * can hold, so a full list swaps in without moving anything below it.
 *
 * The title sits in a `<div>` where the loaded row uses a `<p>` only because a
 * `Skeleton` renders a `<div>`, which is not phrasing content and may not live
 * inside a paragraph. Preflight zeroes a `<p>`'s margins, so the two boxes are
 * identical.
 *
 * The second bar in the meta line stands in for a model badge, which is a line
 * box plus `py-0.5` and a 1px border on each side — 0.375rem of chrome the
 * timestamp beside it does not carry.
 */
function RecentSessionRowSkeleton() {
  return (
    <li>
      <div className={ROW_BOX}>
        <div className="min-w-0 flex-1">
          <div className={ROW_TITLE}>
            <Skeleton className="h-[1lh] w-48" />
          </div>
          <div className={ROW_META}>
            <Skeleton className="h-[1lh] w-24" />
            <Skeleton className="h-[calc(1lh_+_0.375rem)] w-20" />
          </div>
        </div>
        <ArrowRight className="text-muted-foreground size-4 shrink-0 opacity-40" aria-hidden />
      </div>
    </li>
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
        className={cn('hover:bg-muted/50 group transition-colors', ROW_BOX)}
      >
        <div className="min-w-0 flex-1">
          <p className={ROW_TITLE}>{title}</p>
          <div className={ROW_META}>
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
