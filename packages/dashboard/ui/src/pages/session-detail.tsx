import { Transcript } from '@/components/transcript';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { formatBytes, formatCount, formatDateTime, formatNumber, formatRate } from '@/lib/format';
import { shQuote } from '@/lib/shell';
import type { SessionDetailResponse, SessionMetricsResponse, SessionTurnMetric } from '@/lib/types';
import { useJson } from '@/lib/use-api';
import { AlertCircle, ArrowLeft, Copy, FileWarning, Inbox } from 'lucide-react';
import { type ReactNode, useMemo } from 'react';
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
import { toast } from 'sonner';

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
  ttftMs: number | null;
}

function toPoints(turns: SessionTurnMetric[]): TurnPoint[] {
  return turns.map((t, i) => ({
    turn: i + 1,
    input: t.inputTokens ?? 0,
    output: t.outputTokens ?? 0,
    cached: t.cachedTokens ?? 0,
    decodeTps: isFiniteNumber(t.decodeTps) ? t.decodeTps : null,
    ttftMs: isFiniteNumber(t.ttftMs) ? t.ttftMs : null,
  }));
}

const AXIS_TICK = { fill: 'var(--color-muted-foreground)', fontSize: 12 } as const;
const TOOLTIP_CONTENT_STYLE = {
  background: 'var(--color-popover)',
  border: '1px solid var(--color-border)',
  borderRadius: 8,
  fontSize: 12,
  color: 'var(--color-popover-foreground)',
} as const;

export default function SessionDetail() {
  const { id = '' } = useParams();
  const encodedId = encodeURIComponent(id);
  const detail = useJson<SessionDetailResponse>(`/sessions/${encodedId}`);
  const metrics = useJson<SessionMetricsResponse>(`/sessions/${encodedId}/metrics`);

  const session = detail.data?.session;
  const turns = useMemo(() => metrics.data?.turns ?? [], [metrics.data]);
  const traces = useMemo(() => metrics.data?.traces ?? [], [metrics.data]);
  const points = useMemo(() => toPoints(turns), [turns]);

  const models = useMemo(() => {
    const set = new Set<string>();
    for (const t of turns) if (t.model !== null && t.model !== '') set.add(t.model);
    for (const tr of traces) if (tr.model !== null && tr.model !== '') set.add(tr.model);
    return [...set];
  }, [turns, traces]);

  const totals = useMemo(() => {
    let input = 0;
    let output = 0;
    let cached = 0;
    for (const t of turns) {
      input += t.inputTokens ?? 0;
      output += t.outputTokens ?? 0;
      cached += t.cachedTokens ?? 0;
    }
    const avgDecode = mean(traces.map((t) => t.decodeTps).filter(isFiniteNumber));
    const avgTtft = mean(traces.map((t) => t.ttftMs).filter(isFiniteNumber));
    return { input, output, cached, avgDecode, avgTtft };
  }, [turns, traces]);

  /**
   * Per-turn cold-tier reuse for this session. The API has always returned
   * these four fields on every trace row and nothing rendered them, so the one
   * question a cold tier exists to answer — "did resuming this session actually
   * reuse anything?" — had no answer anywhere in the UI.
   */
  const cold = useMemo(() => {
    let coldHits = 0;
    let coldMisses = 0;
    let coldBytesRestored = 0;
    let coldBytesWritten = 0;
    for (const t of traces) {
      coldHits += t.coldHits ?? 0;
      coldMisses += t.coldMisses ?? 0;
      coldBytesRestored += t.coldBytesRestored ?? 0;
      coldBytesWritten += t.coldBytesWritten ?? 0;
    }
    return { coldHits, coldMisses, coldBytesRestored, coldBytesWritten };
  }, [traces]);
  const coldLookups = cold.coldHits + cold.coldMisses;

  const hasTokenData = points.some((p) => p.input + p.output + p.cached > 0);
  const hasDecodeData = points.some((p) => p.decodeTps !== null);

  const copyResume = (): void => {
    if (session === undefined) return;
    const command = `mlx agent --session ${shQuote(session.path)}`;
    void navigator.clipboard.writeText(command).then(
      () => toast.success('Resume command copied', { description: command }),
      () => toast.error('Failed to copy to clipboard'),
    );
  };

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
      ) : detail.loading || session === undefined ? (
        /*
          The loaded header's own markup with the leaf text swapped for bars, not
          an approximation of it: same row, same `space-y-2` column, and each bar
          INSIDE the element carrying the font, so `1lh` measures one line box of
          that element and the header keeps its height when the session arrives.
          The pair of free-standing bars this replaces sat in a `space-y-3` column
          of their own and reflowed the whole page under them.

          The model badges are deliberately not reserved. That row is gated on
          `models.length > 0`, and `models` is derived from the METRICS request —
          a second fetch that is usually still in flight when this branch ends —
          so reserving the row here could not spare the page that row's own
          reflow, while a session that names no model would pay a fresh collapse
          for a row it never shows.
        */
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0 space-y-2">
            {/* The title's classes, but not its tag: a heading with no text in it
                announces an empty heading, and `h1` may not contain the bar's
                `div`. The classes are what decide the box. */}
            <div className="text-2xl font-semibold tracking-tight break-words">
              <Skeleton className="h-[1lh] w-96 max-w-full" />
            </div>
            <div className="text-muted-foreground flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
              <Skeleton className="h-[1lh] w-72 max-w-full" />
            </div>
          </div>
          {/* The copy button is a fixed-size control (`h-9`), so it can be stood
              in for exactly rather than left to appear from nowhere. */}
          <Skeleton className="h-9 w-44" />
        </div>
      ) : (
        <>
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="min-w-0 space-y-2">
              <h1 className="text-2xl font-semibold tracking-tight break-words">
                {session.name ?? session.firstMessage ?? session.id}
              </h1>
              <div className="text-muted-foreground flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
                <span className="font-mono text-xs break-all" title={session.cwd}>
                  {session.cwd}
                </span>
                <span aria-hidden>·</span>
                <span>Created {formatDateTime(session.created)}</span>
              </div>
              {models.length > 0 && (
                <div className="flex flex-wrap items-center gap-1">
                  {models.map((model) => (
                    <Badge key={model} variant="secondary" className="font-normal">
                      {model}
                    </Badge>
                  ))}
                </div>
              )}
            </div>
            <Button variant="outline" onClick={copyResume}>
              <Copy className="size-4" />
              Copy resume command
            </Button>
          </div>

          {!metrics.loading && !metrics.error && turns.length > 0 && (
            <div className="flex flex-wrap gap-2">
              <MetricChip label="Turns" value={formatNumber(turns.length)} />
              <MetricChip label="Input" value={formatCount(totals.input)} unit="tokens" />
              <MetricChip label="Output" value={formatCount(totals.output)} unit="tokens" />
              <MetricChip label="Cached" value={formatCount(totals.cached)} unit="tokens" />
              {totals.avgDecode !== null && (
                <MetricChip label="Decode" value={totals.avgDecode.toFixed(1)} unit="tok/s avg" />
              )}
              {totals.avgTtft !== null && (
                <MetricChip label="TTFT" value={Math.round(totals.avgTtft).toString()} unit="ms avg" />
              )}
              {coldLookups > 0 && (
                <>
                  <MetricChip label="Cold hit rate" value={formatRate(cold.coldHits, coldLookups)} />
                  <MetricChip
                    label="Cold reuse"
                    value={formatBytes(cold.coldBytesRestored)}
                    unit={`restored · ${formatBytes(cold.coldBytesWritten)} written`}
                  />
                </>
              )}
            </div>
          )}

          {metrics.error ? (
            <Card>
              <CardContent className="text-muted-foreground flex items-center gap-2 py-4 text-sm">
                <FileWarning className="size-4 shrink-0" aria-hidden />
                Metrics unavailable: {metrics.error.message}
              </CardContent>
            </Card>
          ) : (
            turns.length > 0 &&
            (hasTokenData || hasDecodeData) && (
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
                  <ChartCard title="Decode throughput per turn" subtitle="Tokens per second (turns with a trace)">
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
            )
          )}

          <Card>
            <CardHeader>
              <CardTitle>Transcript</CardTitle>
            </CardHeader>
            <CardContent>
              {detail.data?.transcriptError !== undefined && (
                <div className="text-muted-foreground bg-muted/50 mb-4 flex items-start gap-2 rounded-md border p-3 text-xs">
                  <FileWarning className="mt-0.5 size-4 shrink-0" aria-hidden />
                  Transcript could not be read: {detail.data.transcriptError}
                </div>
              )}
              {detail.data === undefined ? null : detail.data.transcript.length === 0 ? (
                <div className="text-muted-foreground flex flex-col items-center gap-2 py-10 text-sm">
                  <Inbox className="size-6" aria-hidden />
                  No messages in this session.
                </div>
              ) : (
                <Transcript entries={detail.data.transcript} />
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
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

function ChartCard({ title, subtitle, children }: { title: string; subtitle: string; children: ReactNode }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
        <p className="text-muted-foreground text-sm">{subtitle}</p>
      </CardHeader>
      <CardContent>
        <div className="h-56 w-full">{children}</div>
      </CardContent>
    </Card>
  );
}
