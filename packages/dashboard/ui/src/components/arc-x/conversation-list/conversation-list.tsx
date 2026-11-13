import { Pencil, ShieldCheck, ShieldOff } from 'lucide-react';
import {
  useId,
  useLayoutEffect,
  useRef,
  type KeyboardEvent,
  type MouseEvent,
  type ReactElement,
  type ReactNode,
} from 'react';

import { CHAT } from '../../../lib/chat-strings';
import { Badge } from '../../arc/badge/badge';
import { IconButton } from '../icon-button/icon-button';
import { LiveDot } from '../live-dot/live-dot';
import { PathText } from '../path-text/path-text';
import { Placeholder } from '../placeholder/placeholder';
import styles from './conversation-list.module.css';

export interface ConversationItem {
  id: string;
  title: string;
  mode: 'chat' | 'work';
  updatedAt: number;
  /** Work: the folder, shown on a second line. */
  folder?: string | null;
  state: 'idle' | 'running' | 'waiting';
  /** Work: the permission mode, when it is not Ask. */
  permissionMode?: 'auto' | 'bypass';
}

export interface ConversationGroup {
  label: string | null;
  items: readonly ConversationItem[];
}

export interface ConversationListProps {
  label: string;
  groups: readonly ConversationGroup[];
  currentId: string | null;
  hrefFor: (id: string) => string;
  onOpen: (id: string) => void;
  onRename: (item: ConversationItem) => void;
  /** The delete control of a row: a ConfirmPopover with an icon trigger. */
  renderDelete: (item: ConversationItem) => ReactNode;
  /** Show the mode tag on every row (search results). */
  showMode?: boolean;
  /** For relative times. */
  now: number;
  loading?: boolean;
  /** Shown when there is no item. */
  empty?: ReactNode;
}

const ROW = 'data-conversation-row';
const ACTIONS = 'data-conversation-actions';

/** `just now`, `2m ago`, `3h ago`, `5d ago`, else the date; seen from `now`. */
export function relativeTime(ms: number, now: number): string {
  const seconds = Math.round((now - ms) / 1000);
  if (seconds < 45) return 'just now';
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  if (days < 30) return `${days}d ago`;
  return new Date(ms).toLocaleDateString();
}

/** The conversations of a workspace, grouped by day. One tab stop; arrows move, → reaches a row's actions. */
export function ConversationList(props: ConversationListProps): ReactElement {
  const rootRef = useRef<HTMLElement | null>(null);
  const seen = useRef<Set<string> | null>(null);
  const baseId = useId();
  const all = props.groups.flatMap((group) => group.items);

  const links = (): HTMLAnchorElement[] =>
    rootRef.current === null ? [] : [...rootRef.current.querySelectorAll<HTMLAnchorElement>(`a[${ROW}]`)];

  useLayoutEffect(() => {
    const items = links();
    const current = items.findIndex((link) => link.getAttribute(ROW) === props.currentId);
    const focused = items.indexOf(document.activeElement as HTMLAnchorElement);
    const stop = focused !== -1 ? focused : current !== -1 ? current : 0;
    items.forEach((link, index) => {
      link.tabIndex = index === stop ? 0 : -1;
    });
    rootRef.current?.querySelectorAll<HTMLElement>(`[${ACTIONS}] button`).forEach((button) => {
      button.tabIndex = -1;
    });
  });

  const fresh = (id: string): boolean => {
    if (seen.current === null) return false;
    return !seen.current.has(id);
  };
  useLayoutEffect(() => {
    if (props.loading === true) return;
    seen.current = new Set(all.map((item) => item.id));
  });

  const focusLink = (link: HTMLAnchorElement | undefined): void => {
    if (link === undefined) return;
    for (const other of links()) other.tabIndex = other === link ? 0 : -1;
    link.focus();
    link.scrollIntoView?.({ block: 'nearest' });
  };

  const onKeyDown = (event: KeyboardEvent<HTMLElement>): void => {
    const target = event.target as HTMLElement;
    const items = links();
    const row = target.closest('li');
    const link = row?.querySelector<HTMLAnchorElement>(`a[${ROW}]`) ?? undefined;
    const item = all.find((entry) => entry.id === link?.getAttribute(ROW));
    if (target.matches(`a[${ROW}]`)) {
      const index = items.indexOf(target as HTMLAnchorElement);
      let next = -1;
      if (event.key === 'ArrowDown') next = Math.min(items.length - 1, index + 1);
      else if (event.key === 'ArrowUp') next = Math.max(0, index - 1);
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = items.length - 1;
      if (next !== -1) {
        event.preventDefault();
        focusLink(items[next]);
        return;
      }
      if (event.key === 'ArrowRight') {
        const first = row?.querySelector<HTMLElement>(`[${ACTIONS}] button`);
        if (first !== null && first !== undefined) {
          event.preventDefault();
          first.focus();
        }
        return;
      }
    } else if (target.closest(`[${ACTIONS}]`) !== null) {
      const buttons = [...(row?.querySelectorAll<HTMLElement>(`[${ACTIONS}] button`) ?? [])];
      const at = buttons.indexOf(target);
      if (event.key === 'ArrowLeft') {
        event.preventDefault();
        if (at > 0) buttons[at - 1]?.focus();
        else focusLink(link);
        return;
      }
      if (event.key === 'ArrowRight' && at < buttons.length - 1) {
        event.preventDefault();
        buttons[at + 1]?.focus();
        return;
      }
    }
    if (item === undefined) return;
    if (event.key === 'F2') {
      event.preventDefault();
      props.onRename(item);
    } else if (event.key === 'Backspace' && event.metaKey) {
      event.preventDefault();
      const buttons = row?.querySelectorAll<HTMLElement>(`[${ACTIONS}] button`);
      buttons?.[buttons.length - 1]?.click();
    }
  };

  if (props.loading === true) {
    return (
      <nav className={styles.root} aria-label={props.label} aria-busy="true">
        <div role="status" className={styles.loading}>
          <span className={styles.srOnly}>{CHAT.listLoading}</span>
          {[0, 1, 2, 3, 4, 5].map((index) => (
            <Placeholder key={index} height={36} radius="control" />
          ))}
        </div>
      </nav>
    );
  }

  return (
    <nav ref={rootRef} className={styles.root} aria-label={props.label} onKeyDown={onKeyDown}>
      {all.length === 0 && props.empty !== undefined ? <div className={styles.empty}>{props.empty}</div> : null}
      {props.groups.map((group, groupIndex) => {
        if (group.items.length === 0) return null;
        const labelId = `${baseId}-g${groupIndex}`;
        return (
          <div key={group.label ?? `group-${groupIndex}`} className={styles.group}>
            {group.label !== null ? (
              <p id={labelId} className={styles.groupLabel}>
                {group.label}
              </p>
            ) : null}
            <ul
              className={styles.list}
              aria-labelledby={group.label !== null ? labelId : undefined}
              aria-label={group.label === null ? props.label : undefined}
            >
              {group.items.map((item) => {
                const current = item.id === props.currentId;
                const open = (event: MouseEvent<HTMLAnchorElement>): void => {
                  if (event.metaKey || event.ctrlKey || event.shiftKey || event.button !== 0) return;
                  event.preventDefault();
                  props.onOpen(item.id);
                };
                const hasFolder = item.mode === 'work' && item.folder !== undefined && item.folder !== null;
                const loud = item.permissionMode === 'bypass';
                const twoLine = hasFolder || loud;
                return (
                  <li
                    key={item.id}
                    className={styles.row}
                    data-current={current ? '' : undefined}
                    data-state={item.state}
                    data-two-line={twoLine ? '' : undefined}
                    data-enter={fresh(item.id) ? '' : undefined}
                  >
                    <a
                      className={styles.link}
                      href={props.hrefFor(item.id)}
                      aria-current={current ? 'page' : undefined}
                      onClick={open}
                      {...{ [ROW]: item.id }}
                    >
                      <span className={styles.dot}>
                        {item.state === 'running' ? (
                          <LiveDot tone="success" pulse size={6} label={CHAT.running} />
                        ) : item.state === 'waiting' ? (
                          <LiveDot tone="warning" pulse size={6} label={CHAT.waiting} />
                        ) : null}
                      </span>
                      <span className={styles.text}>
                        <span className={styles.titleLine}>
                          <span className={styles.title}>{item.title}</span>
                          {item.permissionMode === 'auto' ? (
                            <span className={styles.permission} data-permission-mode="auto">
                              <ShieldCheck size={12} strokeWidth={1.75} aria-hidden="true" />
                              <span className={styles.srOnly}>{CHAT.permission.modes.auto}</span>
                            </span>
                          ) : loud ? (
                            <span className={styles.permission} data-permission-mode="bypass">
                              <ShieldOff size={12} strokeWidth={1.75} aria-hidden="true" />
                            </span>
                          ) : null}
                          {props.showMode === true ? (
                            <Badge size="sm" outline className={styles.mode}>
                              {item.mode === 'work' ? CHAT.modeWork : CHAT.modeChat}
                            </Badge>
                          ) : null}
                        </span>
                        {twoLine ? (
                          <span className={styles.folder} data-loud={loud ? '' : undefined}>
                            {loud ? (
                              <span className={styles.permissionWord}>
                                {CHAT.permission.modes.bypass}
                                {hasFolder ? ' · ' : null}
                              </span>
                            ) : null}
                            {hasFolder ? <PathText path={item.folder ?? ''} keepEnd mono={false} /> : null}
                          </span>
                        ) : null}
                      </span>
                    </a>
                    <span className={styles.time}>{relativeTime(item.updatedAt, props.now)}</span>
                    <span className={styles.actions} {...{ [ACTIONS]: '' }}>
                      <IconButton
                        label={CHAT.rename}
                        icon={<Pencil size={14} strokeWidth={1.75} />}
                        onClick={() => props.onRename(item)}
                      />
                      {props.renderDelete(item)}
                    </span>
                  </li>
                );
              })}
            </ul>
          </div>
        );
      })}
    </nav>
  );
}
