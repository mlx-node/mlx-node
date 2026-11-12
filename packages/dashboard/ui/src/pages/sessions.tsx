import { RefreshCw } from 'lucide-react';
import { type ReactElement, useEffect, useMemo, useRef, useState } from 'react';
import { useSearchParams } from 'react-router-dom';

import { CompactField } from '../components/arc-x/compact-field/compact-field';
import { DataTable, type DataColumn } from '../components/arc-x/data-table/data-table';
import { IconButton } from '../components/arc-x/icon-button/icon-button';
import { LiveDot } from '../components/arc-x/live-dot/live-dot';
import { ModelBadge } from '../components/arc-x/model-badge/model-badge';
import { Panel } from '../components/arc-x/panel/panel';
import { PathText } from '../components/arc-x/path-text/path-text';
import { SplitBar } from '../components/arc-x/split-bar/split-bar';
import { StatGroup, StatTile } from '../components/arc-x/stat-tile/stat-tile';
import { Badge } from '../components/arc/badge/badge';
import { Button } from '../components/arc/button/button';
import { SearchField } from '../components/arc/search-field/search-field';
import SegmentedControl, { type Segment } from '../components/arc/segmented-control/segmented-control';
import { Select } from '../components/arc/select/select';
import { SessionActions, sessionUrl } from '../components/session-dialogs';
import { CHAT } from '../lib/chat-strings';
import { formatCount, formatNumber, formatRate, formatRelativeTime } from '../lib/format';
import { useHeld } from '../lib/held';
import { useHomeDir } from '../lib/home-dir';
import { truncateMiddle } from '../lib/labels';
import { tildePath } from '../lib/paths';
import { AGENT_MESSAGES_TIP, isoMs, LIVE_HINT, sessionTitle, sessionTitleTip } from '../lib/sessions';
import type { SessionListResponse, SessionRow, SessionTokens } from '../lib/types';
import { useJson } from '../lib/use-api';
import { type AppCommand, useCommands } from '../shell/commands';
import { usePageSearch } from '../shell/hotkeys';
import { Page, PageHeader } from '../shell/page';

/** The server's cap; asked for explicitly so the footnote below is the only truncation. */
export const SESSIONS_LIST_LIMIT = 500;

/** The request path for a set of filters, and therefore the `useJson` cache key. */
export function sessionsPath({ query, cwd }: { query: string; cwd: string }): string {
  const params = new URLSearchParams();
  if (query !== '') params.set('q', query);
  if (cwd !== 'all') params.set('cwd', cwd);
  params.set('limit', String(SESSIONS_LIST_LIMIT));
  return `/sessions?${params.toString()}`;
}

const SEARCH_DEBOUNCE_MS = 250;
const ALL_DIRECTORIES = 'all';

/** The list without the sessions the runtime just deleted, so their rows leave at once. */
export function withoutSessions(body: SessionListResponse, deleted: readonly string[]): SessionListResponse {
  const sessions = body.sessions.filter((row) => !deleted.includes(row.id));
  return { ...body, sessions, total: Math.max(0, body.total - (body.sessions.length - sessions.length)) };
}

/** The list with the title the runtime just saved. */
export function withTitle(body: SessionListResponse, id: string, title: string): SessionListResponse {
  return { ...body, sessions: body.sessions.map((row) => (row.id === id ? { ...row, title } : row)) };
}

/** Disjoint parts of a session's tokens: cache reads, fresh prompt, output. */
function tokenParts(tokens: SessionTokens): { cached: number; fresh: number; output: number } {
  const cached = Math.min(tokens.cached, tokens.input);
  return { cached, fresh: tokens.input - cached, output: tokens.output };
}

function tokenTotal(row: SessionRow): number | null {
  return row.tokens === null ? null : row.tokens.input + row.tokens.output;
}

interface TokenSums {
  input: number;
  output: number;
  cached: number;
  withUsage: number;
  withoutUsage: number;
}

function sumTokens(rows: readonly SessionRow[]): TokenSums {
  const sums: TokenSums = { input: 0, output: 0, cached: 0, withUsage: 0, withoutUsage: 0 };
  for (const row of rows) {
    if (row.tokens === null) {
      sums.withoutUsage += 1;
      continue;
    }
    sums.withUsage += 1;
    sums.input += row.tokens.input;
    sums.output += row.tokens.output;
    sums.cached += Math.min(row.tokens.cached, row.tokens.input);
  }
  return sums;
}

function TokenCell({ row, max }: { row: SessionRow; max: number }): ReactElement {
  const total = tokenTotal(row);
  if (row.tokens === null || total === null || total <= 0) return <span className="text-secondary">—</span>;
  const parts = tokenParts(row.tokens);
  return (
    <span className="inline-flex items-center justify-end gap-2">
      <span className="inline-block min-w-[6ch] text-end tabular-nums">{formatCount(total)}</span>
      <span className="inline-block w-[56px]">
        <SplitBar
          label="Tokens"
          size="dense"
          legend={false}
          total={max}
          formatValue={formatCount}
          segments={[
            { key: 'cached', label: 'Cached', value: parts.cached, color: 'var(--data-cached)' },
            { key: 'input', label: 'New input', value: parts.fresh, color: 'var(--data-input)' },
            { key: 'output', label: 'Output', value: parts.output, color: 'var(--data-output)' },
          ]}
        />
      </span>
    </span>
  );
}

function SessionCell({ row }: { row: SessionRow }): ReactElement {
  const title = sessionTitle(row);
  return (
    <span className="flex min-w-0 items-center gap-1.5">
      <span className="min-w-0 truncate font-medium text-foreground" title={sessionTitleTip(row)}>
        {title}
      </span>
      {row.live ? (
        <Badge tone="success" size="sm" title={LIVE_HINT}>
          <LiveDot tone="success" pulse label="Live" size={6} />
          Live
        </Badge>
      ) : null}
      {row.origin === 'chat' || row.origin === 'work' ? (
        <Badge size="sm" outline>
          {row.origin === 'work' ? CHAT.modeWork : CHAT.modeChat}
        </Badge>
      ) : null}
      {row.kind === 'fork' ? (
        <Badge size="sm" outline>
          fork
        </Badge>
      ) : null}
      {row.subagentCount > 0 ? (
        <Badge size="sm" title="Subagent sessions">
          {`+${formatNumber(row.subagentCount)}`}
        </Badge>
      ) : null}
    </span>
  );
}

export default function Sessions(): ReactElement {
  const [params, setParams] = useSearchParams();
  const urlQuery = params.get('q') ?? '';
  const cwd = params.get('cwd') ?? ALL_DIRECTORIES;
  const liveOnly = params.get('live') === '1';
  const [query, setQuery] = useState(urlQuery);
  const searchRef = useRef<HTMLInputElement | null>(null);
  usePageSearch(searchRef);

  const update = (changes: Record<string, string | null>): void => {
    setParams(
      (current) => {
        const next = new URLSearchParams(current);
        for (const [key, value] of Object.entries(changes)) {
          if (value === null) next.delete(key);
          else next.set(key, value);
        }
        return next;
      },
      { replace: true },
    );
  };
  const updateRef = useRef(update);
  updateRef.current = update;

  // Back from a session restores the URL; the field follows it.
  const [seenUrlQuery, setSeenUrlQuery] = useState(urlQuery);
  if (seenUrlQuery !== urlQuery) {
    setSeenUrlQuery(urlQuery);
    if (query.trim() !== urlQuery) setQuery(urlQuery);
  }
  useEffect(() => {
    const trimmed = query.trim();
    if (trimmed === urlQuery) return;
    const handle = setTimeout(() => updateRef.current({ q: trimmed === '' ? null : trimmed }), SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(handle);
  }, [query, urlQuery]);

  const path = useMemo(() => sessionsPath({ query: urlQuery, cwd }), [urlQuery, cwd]);
  const sessions = useJson<SessionListResponse>(path);
  const held = useHeld(sessions);

  // A delete can empty the selected directory; the Select would then show a blank trigger while the request still
  // filters by it.
  const servedCwds = sessions.data?.cwds;
  const home = useHomeDir();
  useEffect(() => {
    if (cwd !== ALL_DIRECTORIES && servedCwds !== undefined && !servedCwds.includes(cwd)) {
      updateRef.current({ cwd: null });
    }
  }, [cwd, servedCwds]);

  const listed = sessions.data?.sessions ?? [];
  const rows = liveOnly ? listed.filter((row) => row.live) : listed;
  const heldRows = held.data?.sessions ?? [];
  const heldVisible = liveOnly ? heldRows.filter((row) => row.live) : heldRows;
  const liveCount = heldRows.filter((row) => row.live).length;
  const sums = sumTokens(heldVisible);
  const total = held.data?.total;
  const capped = held.data !== undefined && held.data.total > held.data.sessions.length;
  const maxTokens = rows.reduce((max, row) => Math.max(max, tokenTotal(row) ?? 0), 0);
  const filtersActive = urlQuery !== '' || cwd !== ALL_DIRECTORIES || liveOnly;
  const tilesLoading = held.data === undefined && sessions.error === undefined;
  const tilesError = held.data === undefined && sessions.error !== undefined ? 'Failed to load sessions' : undefined;

  const clearFilters = (): void => {
    setQuery('');
    update({ q: null, cwd: null, live: null });
  };
  const reload = sessions.reload;
  useEffect(() => {
    window.addEventListener('focus', reload);
    return () => window.removeEventListener('focus', reload);
  }, [reload]);

  const commands = useMemo((): AppCommand[] => {
    const list: AppCommand[] = [
      liveOnly
        ? {
            id: 'sessions:show-all',
            label: 'Show all sessions',
            group: 'Sessions',
            run: () => updateRef.current({ live: null }),
          }
        : {
            id: 'sessions:show-live',
            label: 'Show live sessions only',
            group: 'Sessions',
            run: () => updateRef.current({ live: '1' }),
          },
    ];
    if (filtersActive) {
      list.push({
        id: 'sessions:clear-filters',
        label: 'Clear filters',
        group: 'Sessions',
        run: () => {
          setQuery('');
          updateRef.current({ q: null, cwd: null, live: null });
        },
      });
    }
    return list;
  }, [liveOnly, filtersActive]);
  useCommands(commands);

  const directoryOptions = [
    { value: ALL_DIRECTORIES, label: 'All directories' },
    ...(servedCwds ?? []).map((dir) => ({ value: dir, label: truncateMiddle(tildePath(dir, home), 40) })),
  ];
  const showOptions: Segment[] = [
    { value: 'all', label: 'All' },
    {
      value: 'live',
      label: 'Live',
      accessory:
        liveCount > 0 ? (
          <Badge size="sm" tone="success">
            {formatCount(liveCount)}
          </Badge>
        ) : undefined,
    },
  ];

  const columns: DataColumn<SessionRow>[] = [
    {
      id: 'session',
      header: 'Session',
      cell: (row) => <SessionCell row={row} />,
      sortValue: (row) => sessionTitle(row).toLowerCase(),
      placeholderWidth: '70%',
    },
    {
      id: 'cwd',
      header: 'Directory',
      cell: (row) => <PathText path={row.cwd} maxChars={28} className="block truncate text-caption text-secondary" />,
      sortValue: (row) => row.cwd,
      width: 212,
      minTableWidth: 880,
    },
    {
      id: 'model',
      header: 'Model',
      cell: (row) =>
        row.model !== null ? (
          <ModelBadge name={row.model} size="sm" maxChars={20} />
        ) : (
          <span className="text-secondary">—</span>
        ),
      sortValue: (row) => row.model,
      width: 160,
      minTableWidth: 700,
    },
    {
      id: 'branch',
      header: 'Branch',
      cell: (row) => <span className="font-mono text-caption text-secondary">{row.branch ?? '—'}</span>,
      sortValue: (row) => row.branch,
      width: 110,
      minTableWidth: 1040,
      truncate: true,
    },
    {
      id: 'last',
      header: 'Last active',
      cell: (row) => (
        <span className="text-secondary" title={row.lastActiveAt}>
          {formatRelativeTime(isoMs(row.lastActiveAt))}
        </span>
      ),
      sortValue: (row) => isoMs(row.lastActiveAt),
      align: 'end',
      width: 96,
    },
    {
      id: 'messages',
      header: 'Messages',
      headerTitle: AGENT_MESSAGES_TIP,
      cell: (row) => <span className="tabular-nums">{formatNumber(row.messageCount)}</span>,
      sortValue: (row) => row.messageCount,
      align: 'end',
      width: 84,
      minTableWidth: 980,
    },
    {
      id: 'tokens',
      header: 'Tokens',
      cell: (row) => <TokenCell row={row} max={maxTokens} />,
      sortValue: tokenTotal,
      align: 'end',
      width: 148,
      minTableWidth: 560,
    },
    {
      id: 'actions',
      header: '',
      cell: (row) => (
        <SessionActions
          session={row}
          compact
          onRenamed={(title) => sessions.patch((body) => withTitle(body, row.id, title))}
          onDeleted={(deleted) => sessions.patch((body) => withoutSessions(body, deleted))}
          onConflict={reload}
        />
      ),
      align: 'end',
      width: 120,
    },
  ];

  const tokenHint =
    sums.withUsage === 0 ? (
      'no usage recorded'
    ) : (
      <span className="grid">
        <span>{`${formatCount(sums.input)} in · ${formatCount(sums.output)} out`}</span>
        {sums.withoutUsage > 0 ? (
          <span>{`${formatCount(sums.withoutUsage)} session${sums.withoutUsage === 1 ? '' : 's'} without usage`}</span>
        ) : null}
        {capped ? <span>listed sessions only</span> : null}
      </span>
    );

  return (
    <Page busy={sessions.refreshing}>
      <PageHeader
        title="Sessions"
        description="mlx agent sessions on this Mac."
        sticky
        controls={
          <>
            <CompactField className="w-[220px]">
              <SearchField
                ref={searchRef}
                label="Search sessions"
                placeholder="Search title or directory…"
                value={query}
                onValueChange={setQuery}
              />
            </CompactField>
            <CompactField className="w-[200px]">
              <Select
                label="Filter by directory"
                options={directoryOptions}
                value={cwd}
                onValueChange={(value) => update({ cwd: value === ALL_DIRECTORIES ? null : value })}
              />
            </CompactField>
            <SegmentedControl
              label="Show"
              options={showOptions}
              value={liveOnly ? 'live' : 'all'}
              onValueChange={(value) => update({ live: value === 'live' ? '1' : null })}
            />
          </>
        }
        actions={
          <IconButton
            label="Refresh"
            variant="secondary"
            icon={<RefreshCw size={16} aria-hidden="true" />}
            loading={sessions.refreshing}
            onClick={reload}
          />
        }
      />

      <StatGroup label={filtersActive ? 'Matching sessions' : 'Sessions summary'}>
        <StatTile
          label="Sessions"
          value={total === undefined ? null : formatCount(total)}
          hint={
            capped && held.data !== undefined ? `newest ${formatNumber(held.data.sessions.length)} listed` : undefined
          }
          held={held.held}
          loading={tilesLoading}
          error={tilesError}
        />
        <StatTile
          label="Live now"
          value={held.data === undefined ? null : formatCount(liveCount)}
          onSelect={() => update({ live: liveOnly ? null : '1' })}
          selected={liveOnly}
          held={held.held}
          loading={tilesLoading}
          error={tilesError}
        />
        <StatTile
          label="Tokens"
          value={held.data === undefined ? null : sums.withUsage === 0 ? '—' : formatCount(sums.input + sums.output)}
          hint={held.data === undefined ? undefined : tokenHint}
          held={held.held}
          loading={tilesLoading}
          error={tilesError}
        />
        <StatTile
          label="Prompt from cache"
          value={held.data === undefined ? null : formatRate(sums.cached, sums.input)}
          hint={capped && sums.input > 0 ? 'listed sessions only' : undefined}
          footer={
            sums.input > 0 ? (
              <SplitBar
                label="Prompt tokens"
                size="dense"
                legend={false}
                formatValue={formatCount}
                segments={[
                  { key: 'cached', label: 'Cached', value: sums.cached, color: 'var(--data-cached)' },
                  { key: 'input', label: 'New', value: sums.input - sums.cached, color: 'var(--data-input)' },
                ]}
              />
            ) : undefined
          }
          held={held.held}
          loading={tilesLoading}
          error={tilesError}
        />
      </StatGroup>

      <Panel padding="none" ariaLabel="Sessions">
        <DataTable
          caption="Sessions"
          frame={false}
          columns={columns}
          rows={rows}
          rowKey={(row) => row.id}
          rowAttrs={(row) => ({ 'data-session-id': row.id })}
          rowTo={(row) => sessionUrl(row.id)}
          defaultSort={{ id: 'last', direction: 'desc' }}
          loading={sessions.data === undefined && sessions.error === undefined}
          loadingRows={6}
          error={
            sessions.error !== undefined && sessions.data === undefined
              ? { message: sessions.error.message, retry: reload }
              : undefined
          }
          empty={
            filtersActive
              ? {
                  title: 'No sessions match these filters.',
                  description: 'Clear the search or pick another directory.',
                  action: (
                    <Button variant="secondary" size="sm" onClick={clearFilters}>
                      Clear filters
                    </Button>
                  ),
                }
              : { title: 'No agent sessions yet.', description: 'Sessions appear here after you run mlx agent.' }
          }
          footer={
            sessions.data !== undefined && sessions.data.total > listed.length
              ? `Showing the newest ${formatNumber(listed.length)} of ${formatNumber(sessions.data.total)} matching sessions. Narrow with search or the directory filter to reach older ones.`
              : undefined
          }
        />
      </Panel>
    </Page>
  );
}
