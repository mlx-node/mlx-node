import { useCallback, useEffect, useMemo, useState, type ReactElement } from 'react';
import { useSearchParams } from 'react-router-dom';

import { getJson } from '../lib/api';
import { errMessage } from '../lib/errors';
import {
  formatBytes,
  formatClock,
  formatCount,
  formatCountNoun,
  formatDayLong,
  formatDuration,
  formatNumber,
  formatRate,
  formatTps,
} from '../lib/format';
import { outcomeText, stopSentence, turnPhases, turnTimeline } from '../lib/metrics-view';
import { PREFILL_UNDER_TEXT, prefillSizeText, SETUP_HOLDS_TEXT } from '../lib/prefill';
import type { MetricsTurns, TurnRow } from '../lib/types';
import { KeyValue, type KeyValueItem } from './arc-x/key-value/key-value';
import { ModelBadge } from './arc-x/model-badge/model-badge';
import { Placeholder } from './arc-x/placeholder/placeholder';
import { SpanTimeline } from './arc-x/span-timeline/span-timeline';
import { Meter } from './arc-x/split-bar/split-bar';
import { Badge } from './arc/badge/badge';
import { Drawer, DrawerContent, useDrawerItem } from './arc/drawer/drawer';
import { EmptyState } from './arc/empty-state/empty-state';

export interface TurnDrawerProps {
  /** Row already on screen; null while it is fetched by id. */
  row: TurnRow | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  state?: 'ready' | 'loading' | 'missing';
  /** Set when the lookup failed: shown instead of the turn. */
  error?: string;
  /** Trace id whose table row takes the focus back when the drawer closes. */
  returnFocusTo?: string | null;
}

const OUTCOME_TONE = { ok: 'neutral', error: 'danger', cancelled: 'warning' } as const;

function count(value: number | null): string | null {
  return value === null ? null : formatNumber(value);
}

function tps(value: number | null): string | null {
  return value === null ? null : `${formatTps(value)} tok/s`;
}

/** The row of a table that links to `traceId`, still on screen. */
function rowOpener(traceId: string | null | undefined): HTMLElement | null {
  if (traceId === null || traceId === undefined) return null;
  for (const link of document.querySelectorAll<HTMLAnchorElement>('a[data-row-link]')) {
    const href = link.getAttribute('href') ?? '';
    const query = href.indexOf('?');
    if (query < 0 || new URLSearchParams(href.slice(query + 1)).get('turn') !== traceId) continue;
    return link.closest('tr') ?? link;
  }
  return null;
}

function viaText(row: TurnRow): string | null {
  if (row.endpoint === null) return null;
  return row.stream === true ? `${row.endpoint} · stream` : row.endpoint;
}

function tokenItems(row: TurnRow): KeyValueItem[] {
  const prompt = row.promptTokens;
  const cached = row.cachedTokens;
  const fresh = prompt === null ? null : Math.max(0, prompt - (cached ?? 0));
  return [
    { key: 'prompt', label: 'Prompt', value: count(prompt), numeric: true },
    {
      key: 'cached',
      label: 'Cached',
      value: count(cached),
      hint: cached === null ? undefined : formatRate(cached, prompt),
      numeric: true,
    },
    { key: 'new', label: 'New', value: count(fresh), numeric: true },
    { key: 'output', label: 'Output', value: count(row.outputTokens), numeric: true },
    { key: 'reasoning', label: 'Reasoning', value: count(row.reasoningTokens), numeric: true },
  ];
}

function speedItems(row: TurnRow | null): KeyValueItem[] {
  const measured = row !== null && row.prefillMs !== null && row.prefillTokens !== null;
  const size = measured ? prefillSizeText(row.prefillTokens ?? 0, row.prefillMs ?? 0) : null;
  const rated = measured && row.prefillTps !== null;
  let prefillHint: string | undefined;
  if (row !== null) {
    if (size === null) prefillHint = 'not measured for this turn';
    else prefillHint = rated ? size : `${size} · ${PREFILL_UNDER_TEXT}`;
  }
  return [
    { key: 'first', label: 'First token', value: row === null ? null : formatDuration(row.ttftMs), numeric: true },
    {
      key: 'setup',
      label: 'Setup',
      value: measured ? formatDuration(row.setupMs) : null,
      hint: measured ? SETUP_HOLDS_TEXT : undefined,
      wrapHint: true,
      numeric: true,
    },
    {
      key: 'prefill',
      label: 'Prefill',
      value: rated ? `${formatCount(row.prefillTps)} tok/s` : null,
      hint: prefillHint,
      wrapHint: true,
      numeric: true,
    },
    { key: 'decode', label: 'Decode', value: row === null ? null : tps(row.decodeTps), numeric: true },
  ];
}

function detailItems(row: TurnRow): KeyValueItem[] {
  const items: KeyValueItem[] = [];
  if (row.mtpCycles !== null) {
    const accepted = row.mtpMeanAccepted === null ? '—' : row.mtpMeanAccepted.toFixed(2);
    items.push({
      key: 'mtp',
      label: 'Speculation',
      value: `${formatCountNoun(row.mtpCycles, 'cycle', 'cycles')} · ${accepted} tokens per cycle`,
    });
  }
  if (row.coldHits !== null || row.coldMisses !== null) {
    items.push({
      key: 'cold',
      label: 'Cold cache',
      value: `${formatCountNoun(row.coldHits ?? 0, 'hit', 'hits')} · ${formatCountNoun(row.coldMisses ?? 0, 'miss', 'misses')}`,
    });
  }
  const request = [
    row.dispatch,
    row.toolCount === null ? null : `${formatCountNoun(row.toolCount, 'tool', 'tools')} offered`,
    row.toolCallCount === null ? null : formatCountNoun(row.toolCallCount, 'tool call', 'tool calls'),
    row.imageCount === null ? null : formatCountNoun(row.imageCount, 'image', 'images'),
  ].filter((part): part is string => part !== null);
  items.push({ key: 'request', label: 'Request', value: request.length > 0 ? request.join(' · ') : null });
  const overlap =
    row.concurrency === null && row.concurrencyPeak === null
      ? null
      : `${formatNumber(row.concurrency)} ${row.concurrency === 1 ? 'turn' : 'turns'} at start · peak ${formatNumber(row.concurrencyPeak)}`;
  items.push({ key: 'overlap', label: 'Overlap', value: overlap });
  items.push({
    key: 'memory',
    label: 'Memory',
    value: row.memActiveBytes === null ? null : `${formatBytes(row.memActiveBytes)} active at the end`,
  });
  items.push({ key: 'trace', label: 'Trace id', value: row.traceId, mono: true, copy: row.traceId });
  items.push({ key: 'pid', label: 'Process', value: row.pid === null ? null : `pid ${String(row.pid)}`, mono: true });
  if (row.alias !== null) items.push({ key: 'alias', label: 'Alias', value: row.alias, mono: true });
  return items;
}

function TurnBody({ row, loading }: { row: TurnRow | null; loading: boolean }): ReactElement {
  const timeline = useMemo(() => (row === null ? null : turnTimeline(row)), [row]);
  const phases = row === null ? null : turnPhases(row);
  const stop = row === null ? null : stopSentence(row);
  const via = row === null ? null : viaText(row);
  return (
    <div className="grid gap-5">
      {row === null ? (
        <Placeholder width="60%" />
      ) : (
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <ModelBadge name={row.model} />
          <Badge size="sm" tone={OUTCOME_TONE[row.outcome]}>
            {outcomeText(row.outcome)}
          </Badge>
          {via === null ? null : <span className="text-body text-secondary">{via}</span>}
        </div>
      )}
      <section className="grid gap-2" aria-label="Timeline">
        <div className="flex items-baseline justify-between gap-3">
          <h3 className="text-strong font-medium">Timeline</h3>
          <span className="text-body text-secondary tabular-nums">
            {phases === null ? null : formatDuration(phases.totalMs)}
          </span>
        </div>
        {timeline === null ? (
          <Placeholder height={56} radius="control" />
        ) : (
          <SpanTimeline
            label="Turn timeline"
            rows={timeline.rows}
            spans={timeline.spans}
            markers={timeline.markers}
            domainMs={timeline.domainMs}
            compact
            rowHeight={28}
            formatDuration={formatDuration}
          />
        )}
        {stop === null ? null : <p className="text-body">{stop}</p>}
        {row?.outcome === 'error' ? (
          <p className="text-body text-secondary">Error text is not recorded. See the server log around this time.</p>
        ) : null}
      </section>
      <section className="grid gap-2" aria-label="Tokens">
        <h3 className="text-strong font-medium">Tokens</h3>
        <KeyValue items={row === null ? tokenItems(EMPTY_ROW) : tokenItems(row)} loading={loading} ariaLabel="Tokens" />
      </section>
      <section className="grid gap-2" aria-label="Speed">
        <h3 className="text-strong font-medium">Speed</h3>
        <KeyValue items={speedItems(row)} loading={loading} ariaLabel="Speed" />
      </section>
      <section className="grid gap-2" aria-label="Context">
        <h3 className="text-strong font-medium">Context</h3>
        {row === null || row.ctxFillPct === null ? (
          <p className="text-body text-secondary">{loading ? '' : 'The context window was not recorded.'}</p>
        ) : (
          <div className="grid gap-1">
            <Meter label="Context window in use" value={row.ctxFillPct} max={100} />
            <span className="text-body text-secondary tabular-nums">
              {`${String(Math.round(row.ctxFillPct))}% of ${formatNumber(row.ctxWindow)} tokens`}
            </span>
          </div>
        )}
      </section>
      <KeyValue
        layout="rows"
        items={row === null ? detailItems(EMPTY_ROW) : detailItems(row)}
        loading={loading}
        ariaLabel="Turn details"
      />
    </div>
  );
}

const EMPTY_ROW: TurnRow = {
  traceId: '',
  ts: 0,
  model: '',
  alias: null,
  modelType: null,
  outcome: 'ok',
  errorKind: null,
  failPhase: null,
  finishReason: null,
  endpoint: null,
  stream: null,
  dispatch: null,
  promptTokens: null,
  cachedTokens: null,
  outputTokens: null,
  reasoningTokens: null,
  ctxWindow: null,
  ctxFillPct: null,
  preDispatchMs: null,
  queueMs: null,
  loadWaitMs: null,
  resolveMs: null,
  ttftMs: null,
  prefillMs: null,
  prefillTokens: null,
  setupMs: null,
  decodeMs: null,
  durationMs: null,
  prefillTps: null,
  decodeTps: null,
  mtpCycles: null,
  mtpMeanAccepted: null,
  coldHits: null,
  coldMisses: null,
  toolCount: null,
  toolCallCount: null,
  imageCount: null,
  concurrency: null,
  concurrencyPeak: null,
  memActiveBytes: null,
  pid: null,
};

/** One turn in a 480 px drawer on the right: identity, a phase waterfall, tokens, speed and the rest of its record. */
export function TurnDrawer({
  row,
  open,
  onOpenChange,
  state = 'ready',
  error,
  returnFocusTo,
}: TurnDrawerProps): ReactElement {
  const kept = useDrawerItem(row) ?? null;
  const shown = open ? row : kept;
  const description = shown === null ? undefined : `${formatDayLong(shown.ts)} · ${formatClock(shown.ts, true)}`;
  return (
    <Drawer open={open} onOpenChange={onOpenChange}>
      <DrawerContent
        side="right"
        title="Turn"
        description={description}
        onCloseAutoFocus={(event) => {
          const opener = rowOpener(returnFocusTo);
          if (opener === null) return;
          event.preventDefault();
          opener.focus();
        }}
      >
        {error !== undefined ? (
          <EmptyState title="Could not load the turn" description={error} />
        ) : state === 'missing' ? (
          <EmptyState title="Turn not found" description="It may be older than 30 days." />
        ) : (
          <TurnBody row={shown} loading={state === 'loading'} />
        )}
      </DrawerContent>
    </Drawer>
  );
}

interface Lookup {
  id: string;
  row: TurnRow | null;
  error?: string;
}

/** Reads ?turn=, finds the row in `known` or fetches it, renders the drawer. Opening and closing replace the URL. */
export function useTurnDrawer(known: readonly TurnRow[]): { open(traceId: string): void; element: ReactElement } {
  const [params, setParams] = useSearchParams();
  const id = params.get('turn');
  const found = id === null ? undefined : known.find((row) => row.traceId === id);
  const [lookup, setLookup] = useState<Lookup | null>(null);
  const [last, setLast] = useState<TurnRow | null>(null);
  if (found !== undefined && found !== last) setLast(found);
  const held = found === undefined && last !== null && last.traceId === id ? last : null;
  const [opened, setOpened] = useState<string | null>(id);
  if (id !== null && id !== opened) setOpened(id);
  const needsFetch = id !== null && found === undefined && held === null;

  useEffect(() => {
    if (!needsFetch || id === null) return undefined;
    let active = true;
    getJson<MetricsTurns>(`/metrics/turns?traceId=${encodeURIComponent(id)}`)
      .then((result) => {
        if (active) setLookup({ id, row: result.rows[0] ?? null });
      })
      .catch((err: unknown) => {
        if (active) setLookup({ id, row: null, error: errMessage(err) });
      });
    return () => {
      active = false;
    };
  }, [id, needsFetch]);

  const setTurn = useCallback(
    (traceId: string | null) => {
      setParams(
        (current) => {
          const next = new URLSearchParams(current);
          if (traceId === null) next.delete('turn');
          else next.set('turn', traceId);
          return next;
        },
        { replace: true },
      );
    },
    [setParams],
  );

  const fetched = lookup !== null && lookup.id === id ? lookup : null;
  const row = found ?? held ?? fetched?.row ?? null;
  const state: TurnDrawerProps['state'] =
    found !== undefined || held !== null
      ? 'ready'
      : fetched === null
        ? 'loading'
        : fetched.row === null
          ? 'missing'
          : 'ready';
  const element = (
    <TurnDrawer
      row={row}
      open={id !== null}
      state={state}
      error={held === null ? fetched?.error : undefined}
      returnFocusTo={opened}
      onOpenChange={(next) => {
        if (!next) setTurn(null);
      }}
    />
  );
  return { open: setTurn, element };
}
