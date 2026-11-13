import { Trash2 } from 'lucide-react';
import { type ReactElement, type RefObject, useMemo } from 'react';

import { CompactField } from '../../components/arc-x/compact-field/compact-field';
import { ConfirmPopover } from '../../components/arc-x/confirm-popover/confirm-popover';
import {
  ConversationList,
  type ConversationGroup,
  type ConversationItem,
} from '../../components/arc-x/conversation-list/conversation-list';
import { SwapCell } from '../../components/arc-x/swap-cell/swap-cell';
import { Alert } from '../../components/arc/alert/alert';
import { Button } from '../../components/arc/button/button';
import { SearchField } from '../../components/arc/search-field/search-field';
import { CHAT, fill } from '../../lib/chat-strings';
import { chatTitle } from '../../lib/chat-turns';
import type { ChatHostState, ChatMode, ConversationRow } from '../../lib/chat-types';
import { groupByDay, matchesQuery } from '../../lib/conversations';
import { deleteContent } from './conversation-actions';

export interface ListSource {
  rows: readonly ConversationRow[] | undefined;
  error: boolean;
}

export interface ListPaneProps {
  mode: ChatMode;
  chat: ListSource;
  work: ListSource;
  active: ChatHostState['active'];
  /** Work conversations not in Ask, from the host state. */
  raised: ChatHostState['permissionRaised'];
  currentId: string | null;
  query: string;
  onQueryChange: (query: string) => void;
  searchRef: RefObject<HTMLInputElement | null>;
  onOpen: (id: string) => void;
  onRename: (item: ConversationItem) => void;
  onDeleted: (id: string) => void;
  onRetry: () => void;
}

function itemOf(
  row: ConversationRow,
  active: ChatHostState['active'],
  raised: ChatHostState['permissionRaised'],
): ConversationItem {
  const live = active.find((entry) => entry.conversationId === row.id);
  return {
    permissionMode: row.mode === 'work' ? raised.find((entry) => entry.id === row.id)?.mode : undefined,
    id: row.id,
    title: chatTitle(row.title, row.mode),
    mode: row.mode,
    updatedAt: row.updatedAt,
    folder: row.mode === 'work' ? row.cwd : null,
    state: live === undefined ? 'idle' : live.waiting ? 'waiting' : live.busy ? 'running' : 'idle',
  };
}

function isBusy(id: string, active: ChatHostState['active']): boolean {
  return active.some((entry) => entry.conversationId === id && (entry.busy || entry.waiting));
}

/** The conversation list column: search over both workspaces, then the workspace's own list by day. */
export function ListPane(props: ListPaneProps): ReactElement {
  const { mode, active, raised } = props;
  const now = Date.now();
  const own = mode === 'chat' ? props.chat : props.work;
  const searching = props.query.trim() !== '';
  const items = useMemo(() => {
    const pick = (source: ListSource, m: ChatMode): ConversationItem[] =>
      (source.rows ?? []).filter((row) => row.mode === m).map((row) => itemOf(row, active, raised));
    return { chat: pick(props.chat, 'chat'), work: pick(props.work, 'work') };
  }, [props.chat, props.work, active, raised]);

  let groups: ConversationGroup[];
  if (searching) {
    const hits = [...items.chat, ...items.work]
      .filter((item) => matchesQuery({ title: item.title, folder: item.folder }, props.query))
      .sort((a, b) => b.updatedAt - a.updatedAt);
    groups = [{ label: null, items: hits }];
  } else {
    groups = groupByDay(items[mode], now);
  }
  const count = groups.reduce((sum, group) => sum + group.items.length, 0);
  const [emptyTitle, emptyText] = mode === 'work' ? CHAT.listEmptyWork : CHAT.listEmptyChat;
  const loading = searching ? props.chat.rows === undefined && props.work.rows === undefined : own.rows === undefined;

  const list =
    own.error && !searching ? (
      <div className="grid gap-2 p-3">
        <Alert tone="warning" title={CHAT.listError}>
          <span>
            <Button type="button" variant="ghost" size="sm" onClick={props.onRetry}>
              {CHAT.tryAgain}
            </Button>
          </span>
        </Alert>
      </div>
    ) : (
      <ConversationList
        label={CHAT.conversations}
        groups={groups}
        currentId={props.currentId}
        hrefFor={(id) => `/chat?c=${encodeURIComponent(id)}`}
        onOpen={props.onOpen}
        onRename={props.onRename}
        renderDelete={(item) => {
          const busy = isBusy(item.id, active);
          return (
            <ConfirmPopover
              trigger={{ kind: 'icon', label: CHAT.remove, icon: <Trash2 size={14} strokeWidth={1.75} /> }}
              align="end"
              disabled={busy}
              disabledReason={busy ? CHAT.deleteBusy : undefined}
              {...deleteContent(
                { id: item.id, title: item.title, mode: item.mode, folder: item.folder ?? null },
                props.onDeleted,
              )}
            />
          );
        }}
        showMode={searching}
        now={now}
        loading={loading}
        empty={
          searching ? (
            <p className="m-0 text-body text-secondary">{fill(CHAT.listNoMatch, { query: props.query.trim() })}</p>
          ) : (
            <span className="grid gap-0.5 text-center">
              <span className="text-body text-foreground">{emptyTitle}</span>
              <span className="text-caption text-secondary">{emptyText}</span>
            </span>
          )
        }
      />
    );

  return (
    <div className="grid content-start gap-1">
      <div className="px-3 pt-2 pb-1">
        <CompactField>
          <SearchField
            ref={props.searchRef}
            label={CHAT.listSearch}
            placeholder={CHAT.listSearchPlaceholder}
            value={props.query}
            onValueChange={props.onQueryChange}
            onKeyDown={(event) => {
              if (event.key === 'Escape' && props.query !== '') {
                event.preventDefault();
                event.stopPropagation();
                props.onQueryChange('');
              }
            }}
          />
        </CompactField>
        {searching ? (
          <p className="m-0 pt-1 text-caption text-secondary" aria-live="polite">
            {count === 1 ? CHAT.listCountOne : fill(CHAT.listCount, { n: count })}
          </p>
        ) : null}
      </div>
      <SwapCell active={searching ? 'search' : mode} items={{ [searching ? 'search' : mode]: list }} />
    </div>
  );
}
