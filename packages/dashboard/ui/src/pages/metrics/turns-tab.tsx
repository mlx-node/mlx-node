import { useEffect, useMemo, useState, type ReactElement } from 'react';
import { useSearchParams } from 'react-router-dom';

import { DataTable, type DataColumn } from '../../components/arc-x/data-table/data-table';
import { ModelBadge } from '../../components/arc-x/model-badge/model-badge';
import { Panel } from '../../components/arc-x/panel/panel';
import { SplitBar } from '../../components/arc-x/split-bar/split-bar';
import { Badge } from '../../components/arc/badge/badge';
import { Button } from '../../components/arc/button/button';
import SegmentedControl, { type Segment } from '../../components/arc/segmented-control/segmented-control';
import { getJson } from '../../lib/api';
import { errMessage } from '../../lib/errors';
import {
  formatClock,
  formatCount,
  formatCountNoun,
  formatDay,
  formatDuration,
  formatNumber,
  formatRate,
  formatTps,
} from '../../lib/format';
import {
  isOutcomeFilter,
  outcomeText,
  phaseSegments,
  RECORD_HINT,
  turnPhases,
  type OutcomeFilter,
} from '../../lib/metrics-view';
import type { MetricsTurns, TurnRow } from '../../lib/types';
import { LIVE_REFETCH_FAST_MS, useJson } from '../../lib/use-api';
import { useMetricsScope, useReportBusy } from './shared';

const OUTCOME_OPTIONS: Segment[] = [
  { value: 'all', label: 'All' },
  { value: 'ok', label: 'Completed' },
  { value: 'error', label: 'Failed' },
  { value: 'cancelled', label: 'Cancelled' },
];

const OUTCOME_TONE = { ok: 'neutral', error: 'danger', cancelled: 'warning' } as const;

interface Older {
  /** Range key, model and outcome: a live data event moves from/to inside one key and keeps the loaded pages. */
  key: string;
  rows: TurnRow[];
  next: MetricsTurns['next'];
  /** The first page already folded into `rows`. */
  merged: MetricsTurns | undefined;
}

function foldFirstPage(older: Older, page: MetricsTurns): Older {
  const held = new Set(older.rows.map((row) => row.traceId));
  const fresh: TurnRow[] = [];
  let joined = false;
  for (const row of page.rows) {
    if (held.has(row.traceId)) {
      joined = true;
      break;
    }
    fresh.push(row);
  }
  // A full first page that shares no row with the held ones leaves turns between them that no cursor reaches.
  if (!joined && page.next !== null) return { key: older.key, rows: [], next: null, merged: undefined };
  return { ...older, rows: [...fresh, ...older.rows], merged: page };
}

export function TurnsTab(): ReactElement {
  const scope = useMetricsScope();
  const [params] = useSearchParams();
  const raw = params.get('outcome');
  const outcome: OutcomeFilter = raw !== null && isOutcomeFilter(raw) ? raw : 'all';
  const base = `/metrics/turns?${scope.q}${outcome === 'all' ? '' : `&outcome=${outcome}`}&limit=200`;
  const pagingKey = `${scope.rangeKey}|${scope.model ?? ''}|${outcome}`;
  const first = useJson<MetricsTurns>(base, { live: { minIntervalMs: LIVE_REFETCH_FAST_MS } });
  const [older, setOlder] = useState<Older>({ key: pagingKey, rows: [], next: null, merged: undefined });
  const [loadingMore, setLoadingMore] = useState(false);
  const [moreError, setMoreError] = useState<string | null>(null);
  useReportBusy(first, { loading: loadingMore, refreshing: false });
  if (older.key !== pagingKey) {
    setOlder({ key: pagingKey, rows: [], next: null, merged: undefined });
    setMoreError(null);
  } else if (older.rows.length > 0 && first.data !== undefined && older.merged !== first.data) {
    setOlder(foldFirstPage(older, first.data));
  }

  const rows = useMemo(
    () => (older.rows.length === 0 ? (first.data?.rows ?? []) : older.rows),
    [first.data, older.rows],
  );
  const next = older.rows.length > 0 ? older.next : (first.data?.next ?? null);
  const total = first.data?.total ?? 0;

  const { reportTurns } = scope;
  useEffect(() => {
    reportTurns(rows);
  }, [rows, reportTurns]);

  const loadMore = async (): Promise<void> => {
    if (next === null) return;
    const shown = rows;
    const shownFrom = first.data;
    const heldBefore = older.rows.length;
    setLoadingMore(true);
    setMoreError(null);
    try {
      const page = await getJson<MetricsTurns>(
        `${base}&beforeTs=${String(next.beforeTs)}&beforeTraceId=${encodeURIComponent(next.beforeTraceId)}`,
      );
      setOlder((current) => {
        // A fold that dropped the held rows also dropped the cursor this page was asked with.
        if (current.key !== pagingKey || (heldBefore > 0 && current.rows.length === 0)) return current;
        const held = current.rows.length === 0 ? shown : current.rows;
        const seen = new Set(held.map((row) => row.traceId));
        return {
          key: pagingKey,
          rows: [...held, ...page.rows.filter((row) => !seen.has(row.traceId))],
          next: page.next,
          merged: current.rows.length === 0 ? shownFrom : current.merged,
        };
      });
    } catch (err: unknown) {
      setMoreError(errMessage(err));
    } finally {
      setLoadingMore(false);
    }
  };

  const columns: DataColumn<TurnRow>[] = [
    {
      id: 'when',
      header: 'When',
      width: 116,
      cell: (row) => (
        <span className="whitespace-nowrap tabular-nums" title={new Date(row.ts).toISOString()}>
          {`${formatDay(row.ts)} ${formatClock(row.ts)}`}
        </span>
      ),
    },
    {
      id: 'model',
      header: 'Model',
      flex: true,
      minWidth: 140,
      truncate: true,
      cell: (row) => <ModelBadge name={row.model} size="sm" fit="width" detail={row.alias} />,
    },
    {
      id: 'result',
      header: 'Result',
      width: 76,
      cell: (row) => (
        <Badge size="sm" tone={OUTCOME_TONE[row.outcome]}>
          {outcomeText(row.outcome)}
        </Badge>
      ),
    },
    {
      id: 'prompt',
      header: 'Prompt',
      align: 'end',
      width: 92,
      cell: (row) => (
        <span className="grid justify-items-end tabular-nums">
          <span>{formatCount(row.promptTokens)}</span>
          {row.cachedTokens === null || row.promptTokens === null ? null : (
            <span className="text-caption text-muted">{`${formatRate(row.cachedTokens, row.promptTokens)} cached`}</span>
          )}
        </span>
      ),
    },
    { id: 'output', header: 'Output', align: 'end', width: 64, cell: (row) => formatCount(row.outputTokens) },
    {
      id: 'ttft',
      header: 'First token',
      align: 'end',
      width: 88,
      minTableWidth: 640,
      cell: (row) => formatDuration(row.ttftMs),
    },
    {
      id: 'prefill',
      header: 'Prefill tok/s',
      align: 'end',
      width: 100,
      minTableWidth: 760,
      cell: (row) => formatCount(row.prefillTps),
    },
    {
      id: 'decode',
      header: 'Decode tok/s',
      align: 'end',
      width: 100,
      minTableWidth: 660,
      cell: (row) => formatTps(row.decodeTps),
    },
    {
      id: 'total',
      header: 'Total',
      align: 'end',
      width: 66,
      cell: (row) =>
        row.durationMs === null && row.preDispatchMs === null
          ? '—'
          : formatDuration((row.preDispatchMs ?? 0) + (row.durationMs ?? 0)),
    },
    {
      id: 'context',
      header: 'Context',
      align: 'end',
      width: 72,
      minTableWidth: 760,
      cell: (row) => (row.ctxFillPct === null ? '—' : `${String(Math.round(row.ctxFillPct))}%`),
    },
    {
      id: 'via',
      header: 'Via',
      truncate: true,
      minTableWidth: 1000,
      cell: (row) => {
        const via = row.endpoint === null ? '—' : row.stream === true ? `${row.endpoint} · stream` : row.endpoint;
        return <span title={via}>{via}</span>;
      },
    },
    {
      id: 'phases',
      header: 'Phases',
      width: 96,
      minTableWidth: 520,
      cell: (row) => {
        const phases = turnPhases(row);
        return (
          <SplitBar
            size="dense"
            legend={false}
            label={`Phases of the turn at ${formatClock(row.ts)}`}
            segments={phaseSegments(phases)}
            total={phases.totalMs}
            formatValue={formatDuration}
          />
        );
      },
    },
  ];

  return (
    <div className="grid gap-3 pt-3">
      <Panel
        title="Turns"
        padding="none"
        actions={
          <SegmentedControl
            label="Outcome"
            options={OUTCOME_OPTIONS}
            value={outcome}
            onValueChange={(value) => {
              if (isOutcomeFilter(value)) scope.setParams({ outcome: value === 'all' ? null : value });
            }}
          />
        }
      >
        <DataTable
          caption="Turns"
          density="compact"
          virtualizeAbove={150}
          columns={columns}
          rows={rows}
          rowKey={(row) => row.traceId}
          rowAttrs={(row) => ({ 'data-model': row.model })}
          rowTo={(row) => scope.urlWith({ turn: row.traceId })}
          loading={first.loading}
          error={
            first.error === undefined || first.data !== undefined
              ? undefined
              : { message: first.error.message, retry: first.reload }
          }
          empty={{ title: 'No turns in this range', description: RECORD_HINT }}
          footer={
            first.data === undefined ? undefined : (
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-body text-secondary tabular-nums">
                  {`Showing ${formatNumber(rows.length)} of ${formatCountNoun(total, 'turn', 'turns')}`}
                </span>
                {moreError === null ? null : <span className="text-body text-danger">{moreError}</span>}
                {next === null ? null : (
                  <Button variant="secondary" size="sm" loading={loadingMore} onClick={() => void loadMore()}>
                    Load 200 more
                  </Button>
                )}
              </div>
            )
          }
        />
      </Panel>
    </div>
  );
}
