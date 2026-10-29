import { Transcript } from '@/components/transcript';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { AXIS_TICK, ChartCard, TOOLTIP_CONTENT_STYLE } from '@/lib/chart';
import { formatCount, formatNumber, formatRelativeTime } from '@/lib/format';
import { copyResumeCommand, isoMs, LIVE_HINT, sessionTitle } from '@/lib/sessions';
import type { SessionDetail as SessionDetailData, SessionDetailResponse, TurnMetric } from '@/lib/types';
import { useJson } from '@/lib/use-api';
import { AlertCircle, ArrowLeft, Copy, FileWarning, GitBranch, Inbox } from 'lucide-react';
import { useMemo } from 'react';
import { Link, useParams } from 'react-router-dom';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

function isFiniteNumber(value: number | null | undefined): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function mean(values: number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

interface TurnPoint {
  turn: number;
  input: number;
  output: number;
  cached: number;
  decodeTps: number | null;
}

function toPoints(turns: TurnMetric[]): TurnPoint[] {
  return turns.map((t, i) => ({
    turn: i + 1,
    input: t.inputTokens,
    output: t.outputTokens,
    cached: t.cachedTokens,
    decodeTps: isFiniteNumber(t.decodeTps) ? t.decodeTps : null,
  }));
}

function isSubagentKind(kind: string | null): boolean {
  return kind !== null && kind.startsWith('subagent');
}

export default function SessionDetail() {
  const { id = '' } = useParams();
  const detail = useJson<SessionDetailResponse>(`/sessions/${encodeURIComponent(id)}`);

  const data = detail.data;
  const session = data?.session;
  const turns = useMemo(() => data?.turns ?? [], [data]);
  const points = useMemo(() => toPoints(turns), [turns]);

  const models = useMemo(() => {
    const set = new Set<string>();
    if (session?.model != null && session.model !== '') set.add(session.model);
    for (const t of turns) if (t.model !== null && t.model !== '') set.add(t.model);
    return [...set];
  }, [session, turns]);

  const totals = useMemo(() => {
    let input = 0;
    let output = 0;
    let cached = 0;
    for (const t of turns) {
      input += t.inputTokens;
      output += t.outputTokens;
      cached += t.cachedTokens;
    }
    const avgDecode = mean(turns.map((t) => t.decodeTps).filter(isFiniteNumber));
    const avgTtft = mean(turns.map((t) => t.ttftMs).filter(isFiniteNumber));
    return { input, output, cached, avgDecode, avgTtft };
  }, [turns]);

  const missingChildren = useMemo(
    () => new Set((session?.subagents ?? []).filter((s) => !s.exists).map((s) => s.id)),
    [session],
  );

  const hasTokenData = points.some((p) => p.input + p.output + p.cached > 0);
  const hasDecodeData = points.some((p) => p.decodeTps !== null);

  return (
    <div className="space-y-6">
      <Link
        to="/sessions"
        className="text-muted-foreground hover:text-foreground inline-flex items-center gap-1.5 text-sm"
      >
        <ArrowLeft className="size-4" aria-hidden />
        Sessions
      </Link>

      {detail.error ? (
        <Card>
          <CardContent className="text-destructive flex items-center gap-2 py-6 text-sm">
            <AlertCircle className="size-4 shrink-0" aria-hidden />
            {detail.error.message}
          </CardContent>
        </Card>
      ) : detail.loading || session === undefined || data === undefined ? (
        /*
          The loaded header's own markup with the leaf text swapped for bars:
          same row, same `space-y-2` column, and each bar INSIDE the element
          carrying the font, so `1lh` measures that element's line box. The
          model badges are not reserved: a session that names no model would
          then collapse a row it never shows.
        */
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0 space-y-2">
            {/* The title's classes but not its tag: an empty heading is still announced. */}
            <div className="text-2xl font-semibold tracking-tight break-words">
              <Skeleton className="h-[1lh] w-96 max-w-full" />
            </div>
            <div className="text-muted-foreground flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
              <Skeleton className="h-[1lh] w-72 max-w-full" />
            </div>
          </div>
          <Skeleton className="h-9 w-44" />
        </div>
      ) : (
        <>
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="min-w-0 space-y-2">
              <h1 className="text-2xl font-semibold tracking-tight break-words">{sessionTitle(session)}</h1>
              <div className="text-muted-foreground flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
                <span className="font-mono text-xs break-all" title={session.cwd}>
                  {session.cwd}
                </span>
                {session.branch !== null && (
                  <>
                    <span aria-hidden>·</span>
                    <span className="inline-flex items-center gap-1 font-mono text-xs">
                      <GitBranch className="size-3.5" aria-hidden />
                      {session.branch}
                    </span>
                  </>
                )}
                <span aria-hidden>·</span>
                <span title={session.createdAt}>Created {formatRelativeTime(isoMs(session.createdAt))}</span>
                <span aria-hidden>·</span>
                <span title={session.lastActiveAt}>Active {formatRelativeTime(isoMs(session.lastActiveAt))}</span>
              </div>
              {(models.length > 0 || session.live || session.kind !== null) && (
                <div className="flex flex-wrap items-center gap-1">
                  {session.live && (
                    <Badge className="bg-emerald-600 text-white" title={LIVE_HINT}>
                      Live
                    </Badge>
                  )}
                  {session.kind !== null && (
                    <Badge variant="outline" className="font-normal">
                      {session.kind}
                    </Badge>
                  )}
                  {models.map((model) => (
                    <Badge key={model} variant="secondary" className="font-normal">
                      {model}
                    </Badge>
                  ))}
                </div>
              )}
              <ParentLink session={session} />
            </div>
            <Button variant="outline" onClick={() => copyResumeCommand(data.resumeCommand)}>
              <Copy className="size-4" />
              Copy resume command
            </Button>
          </div>

          {session.lastTurnSummary !== null && session.lastTurnSummary !== '' && (
            <p className="text-muted-foreground text-sm">
              <span className="text-foreground font-medium">Last turn:</span> {session.lastTurnSummary}
            </p>
          )}

          {turns.length > 0 && (
            <div className="flex flex-wrap gap-2">
              <MetricChip label="Turns" value={formatNumber(turns.length)} />
              <MetricChip label="Input" value={formatCount(totals.input)} unit="tokens" />
              <MetricChip label="Output" value={formatCount(totals.output)} unit="tokens" />
              <MetricChip label="Cached" value={formatCount(totals.cached)} unit="tokens" />
              {totals.avgTtft !== null && (
                <MetricChip label="TTFT" value={Math.round(totals.avgTtft).toString()} unit="ms avg" />
              )}
              {totals.avgDecode !== null && (
                <MetricChip label="Decode" value={totals.avgDecode.toFixed(1)} unit="tok/s avg" />
              )}
            </div>
          )}

          {(hasTokenData || hasDecodeData) && (
            <div className="grid gap-4 lg:grid-cols-2">
              {hasTokenData && (
                <ChartCard title="Tokens per turn" subtitle="Input, output, and cached tokens by turn">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={points} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" vertical={false} />
                      <XAxis dataKey="turn" tick={AXIS_TICK} tickLine={false} axisLine={false} />
                      <YAxis
                        tick={AXIS_TICK}
                        tickLine={false}
                        axisLine={false}
                        width={44}
                        tickFormatter={(v) => formatCount(Number(v))}
                      />
                      <Tooltip
                        cursor={{ fill: 'var(--color-muted)', opacity: 0.35 }}
                        contentStyle={TOOLTIP_CONTENT_STYLE}
                        labelFormatter={(label) => `Turn ${Number(label)}`}
                        formatter={(value) => formatNumber(Number(value))}
                      />
                      <Legend
                        wrapperStyle={{ fontSize: 12 }}
                        formatter={(value) => <span className="text-foreground">{value}</span>}
                      />
                      <Bar
                        dataKey="input"
                        stackId="tokens"
                        name="Input"
                        fill="var(--viz-input)"
                        stroke="var(--color-card)"
                        strokeWidth={1}
                      />
                      <Bar
                        dataKey="output"
                        stackId="tokens"
                        name="Output"
                        fill="var(--viz-output)"
                        stroke="var(--color-card)"
                        strokeWidth={1}
                      />
                      <Bar
                        dataKey="cached"
                        stackId="tokens"
                        name="Cached"
                        fill="var(--viz-cached)"
                        stroke="var(--color-card)"
                        strokeWidth={1}
                        radius={[4, 4, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </ChartCard>
              )}
              {hasDecodeData && (
                <ChartCard title="Decode throughput per turn" subtitle="Tokens per second (turns that reported it)">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={points} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" vertical={false} />
                      <XAxis dataKey="turn" tick={AXIS_TICK} tickLine={false} axisLine={false} />
                      <YAxis
                        tick={AXIS_TICK}
                        tickLine={false}
                        axisLine={false}
                        width={44}
                        tickFormatter={(v) => Number(v).toFixed(0)}
                      />
                      <Tooltip
                        cursor={{ stroke: 'var(--color-muted-foreground)', strokeWidth: 1 }}
                        contentStyle={TOOLTIP_CONTENT_STYLE}
                        labelFormatter={(label) => `Turn ${Number(label)}`}
                        formatter={(value) => `${Number(value).toFixed(1)} tok/s`}
                      />
                      <Line
                        type="monotone"
                        dataKey="decodeTps"
                        name="Decode tok/s"
                        stroke="var(--viz-input)"
                        strokeWidth={2}
                        dot={{ r: 3, fill: 'var(--viz-input)' }}
                        activeDot={{ r: 5 }}
                        connectNulls={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartCard>
              )}
            </div>
          )}

          {session.subagents.length > 0 && <SubagentsCard subagents={session.subagents} />}

          <Card>
            <CardHeader>
              <CardTitle>Transcript</CardTitle>
            </CardHeader>
            <CardContent>
              {data.transcriptError !== undefined && (
                <div className="text-muted-foreground bg-muted/50 mb-4 flex items-start gap-2 rounded-md border p-3 text-xs">
                  <FileWarning className="mt-0.5 size-4 shrink-0" aria-hidden />
                  Transcript could not be read: {data.transcriptError}
                </div>
              )}
              {data.transcript.length === 0 ? (
                <div className="text-muted-foreground flex flex-col items-center gap-2 py-10 text-sm">
                  <Inbox className="size-6" aria-hidden />
                  No messages in this session.
                </div>
              ) : (
                <Transcript entries={data.transcript} missingChildren={missingChildren} />
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}

function ParentLink({ session }: { session: SessionDetailData }) {
  const parent = session.parent;
  if (parent === null) return null;
  const label = isSubagentKind(session.kind) ? 'Subagent of' : 'Forked from';
  return (
    <p className="text-muted-foreground text-sm">
      {label}{' '}
      {parent.title === null ? (
        <span title={parent.id}>a session that no longer exists</span>
      ) : (
        <Link to={`/sessions/${encodeURIComponent(parent.id)}`} className="text-primary hover:underline">
          {sessionTitle({ id: parent.id, title: parent.title })}
        </Link>
      )}
    </p>
  );
}

function SubagentsCard({ subagents }: { subagents: SessionDetailData['subagents'] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Subagents</CardTitle>
      </CardHeader>
      <CardContent>
        <ul className="divide-y">
          {subagents.map((child) => {
            const title = sessionTitle(child);
            return (
              <li
                key={child.id}
                className="flex flex-wrap items-center gap-x-3 gap-y-1 py-2 text-sm first:pt-0 last:pb-0"
              >
                {child.exists ? (
                  <Link
                    to={`/sessions/${encodeURIComponent(child.id)}`}
                    className="min-w-0 flex-1 truncate font-medium hover:underline"
                    title={title}
                  >
                    {title}
                  </Link>
                ) : (
                  <span className="text-muted-foreground min-w-0 flex-1 truncate" title={`${title} (folder missing)`}>
                    {title}
                  </span>
                )}
                {child.type !== '' && (
                  <Badge variant="outline" className="font-normal">
                    {child.type}
                  </Badge>
                )}
                {child.model !== null && (
                  <Badge variant="secondary" className="font-normal">
                    {child.model}
                  </Badge>
                )}
                <span
                  className={child.status === 'failed' ? 'text-destructive text-xs' : 'text-muted-foreground text-xs'}
                >
                  {child.status}
                  {!child.exists && ' · folder missing'}
                </span>
              </li>
            );
          })}
        </ul>
      </CardContent>
    </Card>
  );
}

function MetricChip({ label, value, unit }: { label: string; value: string; unit?: string }) {
  return (
    <div className="bg-muted/40 flex items-baseline gap-1.5 rounded-md border px-3 py-1.5">
      <span className="text-muted-foreground text-xs font-medium">{label}</span>
      <span className="text-sm font-semibold tabular-nums">{value}</span>
      {unit !== undefined && <span className="text-muted-foreground text-xs">{unit}</span>}
    </div>
  );
}
