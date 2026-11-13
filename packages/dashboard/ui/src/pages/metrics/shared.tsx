import {
  createContext,
  useContext,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type ReactElement,
  type ReactNode,
} from 'react';

import type { ChartState } from '../../components/arc-x/chart-card/chart-card';
import type { StatDelta } from '../../components/arc-x/stat-tile/stat-tile';
import { Alert } from '../../components/arc/alert/alert';
import { Button } from '../../components/arc/button/button';
import { formatClock, formatNumber } from '../../lib/format';
import { useHeld } from '../../lib/held';
import { modelColor } from '../../lib/model-colors';
import type { AlignedRange, RangeKey } from '../../lib/range';
import { LOW_SAMPLE_N, LOW_SAMPLE_TAIL_N, type Quantiles, type TurnRow } from '../../lib/types';
import { LIVE_REFETCH_FAST_MS, useJson, type AsyncState } from '../../lib/use-api';

export interface MetricsScope {
  rangeKey: RangeKey;
  range: AlignedRange;
  model: string | null;
  /** `from=…&to=…` plus `&model=…` when a model is picked. */
  q: string;
  /** Writes params with replace navigation; null deletes. */
  setParams: (updates: Readonly<Record<string, string | null>>) => void;
  /** A `?…` link to the current page with these params changed. */
  urlWith: (updates: Readonly<Record<string, string | null>>) => string;
  /** Rows on screen, so the drawer opens without a lookup. */
  reportTurns: (rows: readonly TurnRow[]) => void;
  /** One query of the page (by `id`) is loading or refreshing; the page is busy while any is. */
  reportBusy: (id: string, busy: boolean) => void;
}

/** A loading tile keeps its delta line's box: the same slots loading and loaded. */
export const PENDING_DELTA: StatDelta = { current: null, previous: null, goodWhen: 'neutral', versus: '' };

export const MetricsScopeContext = createContext<MetricsScope | null>(null);

export function useMetricsScope(): MetricsScope {
  const scope = useContext(MetricsScopeContext);
  if (scope === null) throw new Error('useMetricsScope needs MetricsScopeContext');
  return scope;
}

export function scopeQuery(range: AlignedRange, model: string | null): string {
  const base = `from=${String(range.from)}&to=${String(range.to)}`;
  return model === null ? base : `${base}&model=${encodeURIComponent(model)}`;
}

export interface Query<T> {
  state: AsyncState<T>;
  data: T | undefined;
  held: boolean;
  /** The window `data` answers. A held answer keeps its own window: bucketed rows are never re-binned into another. */
  range: AlignedRange | undefined;
}

/** Tells the Metrics page (when there is one) that these queries are loading or refreshing. */
export function useReportBusy(...states: ReadonlyArray<Pick<AsyncState<unknown>, 'loading' | 'refreshing'>>): void {
  const scope = useContext(MetricsScopeContext);
  const id = useId();
  const busy = states.some((state) => state.loading || state.refreshing);
  const report = scope?.reportBusy;
  useEffect(() => {
    if (report === undefined) return undefined;
    report(id, busy);
    return () => report(id, false);
  }, [report, id, busy]);
}

/** A chart / tile query: live, read through `useHeld`. `range` = the window `path` asks for; `fast` is the 1 s throttle. */
export function useHeldQuery<T>(path: string, options: { range?: AlignedRange; fast?: boolean } = {}): Query<T> {
  const state = useJson<T>(path, { live: options.fast === true ? { minIntervalMs: LIVE_REFETCH_FAST_MS } : true });
  const held = useHeld(state);
  useReportBusy(state);
  const answered = useRef<{ data: T | undefined; range: AlignedRange | undefined }>({
    data: undefined,
    range: undefined,
  });
  const { data } = state;
  const asked = options.range;
  useLayoutEffect(() => {
    if (data !== undefined) answered.current = { data, range: asked };
  }, [data, asked]);
  const range = held.held && answered.current.data === held.data ? answered.current.range : asked;
  return { state, data: held.data, held: held.held, range };
}

/** loading → error → empty → held → ready. */
export function regionState(query: Query<unknown>, empty: boolean): ChartState {
  if (query.data === undefined) return query.state.error !== undefined ? 'error' : 'loading';
  if (empty) return 'empty';
  return query.held ? 'held' : 'ready';
}

export function regionError(query: Query<unknown>): { message: string; retry: () => void } | undefined {
  return query.state.error === undefined
    ? undefined
    : { message: query.state.error.message, retry: query.state.reload };
}

/** Tile props shared by every tile fed by one query. */
export function tileState(query: Query<unknown>): { loading: boolean; held: boolean; error?: string } {
  if (query.data === undefined && query.state.error !== undefined)
    return { loading: false, held: false, error: 'Could not load' };
  return { loading: query.data === undefined, held: query.held };
}

/** `Decode speed · qwen3` when a model filter is set. */
export function withModel(text: string, model: string | null): string {
  return model === null ? text : `${text} · ${model}`;
}

/** Shared column scale: the largest p95 over rows with n ≥ 20, else the largest p50. */
export function quantileMax(rows: readonly Quantiles[]): number {
  let tail = 0;
  let median = 0;
  for (const q of rows) {
    if (q.n >= LOW_SAMPLE_TAIL_N && q.p95 !== null) tail = Math.max(tail, q.p95);
    if (q.p50 !== null) median = Math.max(median, q.p50);
  }
  return tail > 0 ? tail : median;
}

/** A median cell: `n` in the title, a muted `n=3` chip below LOW_SAMPLE_N. */
export function MedianCell({ text, n }: { text: string; n: number }): ReactElement {
  return (
    <span className="whitespace-nowrap tabular-nums" title={`n=${formatNumber(n)}`}>
      {text}
      {n > 0 && n < LOW_SAMPLE_N ? <span className="text-muted">{` n=${String(n)}`}</span> : null}
    </span>
  );
}

/** One `Could not refresh` alert for every query of a region that failed while its data is still on screen. */
export function RefreshAlert({ queries }: { queries: ReadonlyArray<Query<unknown>> }): ReactElement | null {
  const failed = queries.find((query) => query.state.error !== undefined && query.data !== undefined);
  const [receivedAt, setReceivedAt] = useState<number | null>(null);
  const stamp = queries.map((query) => query.state.data);
  useEffect(() => {
    if (stamp.some((data) => data !== undefined)) setReceivedAt(Date.now());
  }, stamp);
  if (failed === undefined) return null;
  return (
    <Alert tone="warning" title="Could not refresh">
      <span className="grid justify-items-start gap-2">
        <span>{`Showing data from ${formatClock(receivedAt)}.`}</span>
        <Button variant="secondary" size="sm" onClick={failed.state.reload}>
          Try again
        </Button>
      </span>
    </Alert>
  );
}

/** Caption under a chart or table: 13 px secondary text. */
export function Caption({ children }: { children: ReactNode }): ReactElement {
  return <p className="text-caption text-secondary">{children}</p>;
}

/** A model's color as a bar fill: the third encoding (`other`) draws solid. */
export function modelPaint(name: string): { color: string; pattern: 'solid' | 'alt' } {
  const color = modelColor(name);
  return { color: color.color, pattern: color.pattern === 'alt' ? 'alt' : 'solid' };
}
