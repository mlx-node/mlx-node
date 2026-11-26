import { Trash2 } from 'lucide-react';
import { type ReactElement, type RefObject, useEffect, useMemo, useState } from 'react';

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
import {
  groupByDay,
  groupByFolder,
  groupLive,
  matchesQuery,
  NO_FOLDER_KEY,
  placeKey,
  placeName,
  shownRows,
  spokenFolder,
} from '../../lib/conversations';
import { useHomeDir } from '../../lib/home-dir';
import { scratchShown, tildePath } from '../../lib/paths';
import { deleteContent } from './conversation-actions';
import type { FolderPick } from './drafts';
import { nextFolded, readFolded, revealed, setShownAll, shownAll, toReveal, writeFolded } from './work/folds';

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
  /** A group's "+": the New Work screen with that folder, or with no folder. */
  onNewIn: (pick: FolderPick) => void;
  /** The host's scratch root of Work without a folder; null while the host state is not known. */
  scratchRoot: string | null;
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
    ...(row.mode === 'work' && row.noFolder === true ? { noFolder: true } : {}),
    state:
      live === undefined
        ? 'idle'
        : live.waiting
          ? 'waiting'
          : live.busy || live.background || live.tasks > 0
            ? 'running'
            : 'idle',
    asks: live?.asks ?? 0,
    fork: row.forkOf !== undefined,
  };
}

/** Why a conversation cannot be deleted now, or null. */
function deleteBlock(id: string, active: ChatHostState['active']): string | null {
  const live = active.find((entry) => entry.conversationId === id);
  if (live === undefined) return null;
  if (live.busy || live.waiting) return CHAT.deleteBusy;
  if (live.background) return live.helpers ? CHAT.deleteBusyHelper : CHAT.deleteBusyRun;
  return live.tasks > 0 ? CHAT.deleteBusyTask : null;
}

/** The conversation list column: search over both workspaces, else Chat by day and Work by folder. */
export function ListPane(props: ListPaneProps): ReactElement {
  const { mode, active, raised, currentId, onNewIn } = props;
  const now = Date.now();
  const home = useHomeDir();
  const root = scratchShown(props.scratchRoot, home);
  const own = mode === 'chat' ? props.chat : props.work;
  const searching = props.query.trim() !== '';
  const items = useMemo(() => {
    const pick = (source: ListSource, m: ChatMode): ConversationItem[] =>
      (source.rows ?? []).filter((row) => row.mode === m).map((row) => itemOf(row, active, raised));
    return { chat: pick(props.chat, 'chat'), work: pick(props.work, 'work') };
  }, [props.chat, props.work, active, raised]);

  const [folded, setFolded] = useState(readFolded);
  const [allKeys, setAllKeys] = useState(shownAll);
  const keysNow = useMemo(() => new Set(items.work.map(placeKey)), [items.work]);
  const setFold = (key: string, open: boolean): void => {
    const next = nextFolded(folded, key, open, keysNow);
    setFolded(next);
    writeFolded(next);
  };
  const setAll = (key: string, on: boolean): void => {
    const next = new Set(allKeys);
    if (on) next.add(key);
    else next.delete(key);
    setShownAll(next);
    setAllKeys(next);
  };
  // Going to a conversation in a folded group opens that group; the window's first look does not.
  useEffect(() => {
    const id = toReveal(currentId);
    if (id === null) return;
    const item = items.work.find((entry) => entry.id === id);
    if (item === undefined) return;
    revealed();
    const key = placeKey(item);
    if (!folded.has(key)) return;
    const next = nextFolded(folded, key, true, keysNow);
    setFolded(next);
    writeFolded(next);
  }, [currentId, items.work, folded, keysNow]);

  let groups: ConversationGroup[];
  if (searching) {
    const hits = [...items.chat, ...items.work]
      .filter((item) => matchesQuery({ title: item.title, folderName: placeName(item) }, props.query))
      .sort((a, b) => b.updatedAt - a.updatedAt);
    groups = [{ key: 'search', label: null, items: hits }];
  } else if (mode === 'chat') {
    groups = groupByDay(items.chat, now);
  } else {
    groups = groupByFolder(items.work).map((group): ConversationGroup => {
      const { key } = group;
      const all = allKeys.has(key);
      const { shown, hidden } = shownRows(group.items, all, currentId);
      const long = hidden > 0 || (all && shownRows(group.items, false, currentId).hidden > 0);
      const name = group.name ?? CHAT.folderNone;
      const none = key === NO_FOLDER_KEY;
      const open = !folded.has(key);
      return {
        key,
        label: null,
        items: shown,
        fold: {
          open,
          onToggle: () => setFold(key, !open),
          count: group.items.length,
          live: groupLive(group.items),
          name,
          hint: group.hint,
          title: none ? fill(CHAT.list.noneTitle, { root }) : tildePath(key, home),
          action: none
            ? { label: CHAT.list.newNone, onPress: () => onNewIn({ kind: 'none' }) }
            : {
                label: fill(CHAT.list.newIn, { folder: spokenFolder({ name, hint: group.hint }) }),
                onPress: () => onNewIn({ kind: 'folder', path: key }),
              },
        },
        more: long ? { hidden, onMore: () => setAll(key, true), onFewer: () => setAll(key, false) } : undefined,
      };
    });
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
          const block = deleteBlock(item.id, active);
          return (
            <ConfirmPopover
              trigger={{ kind: 'icon', label: CHAT.remove, icon: <Trash2 size={14} strokeWidth={1.75} /> }}
              align="end"
              disabled={block !== null}
              disabledReason={block ?? undefined}
              {...deleteContent(
                {
                  id: item.id,
                  title: item.title,
                  mode: item.mode,
                  folder: item.folder ?? null,
                  noFolder: item.noFolder === true,
                },
                props.onDeleted,
              )}
            />
          );
        }}
        showMode={searching}
        showFolder={searching}
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
