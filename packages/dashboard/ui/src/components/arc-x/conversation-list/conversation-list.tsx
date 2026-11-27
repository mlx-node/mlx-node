import { ChevronRight, Pencil, Plus, ShieldCheck, ShieldOff } from 'lucide-react';
import { motion } from 'motion/react';
import {
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type KeyboardEvent,
  type MouseEvent,
  type ReactElement,
  type ReactNode,
} from 'react';

import { CHAT, fill } from '../../../lib/chat-strings';
import { Badge } from '../../arc/badge/badge';
import { IconButton } from '../icon-button/icon-button';
import { LiveDot } from '../live-dot/live-dot';
import { MiddleText } from '../middle-text/middle-text';
import { motionX } from '../motion/motion';
import { useStaticMotion } from '../motion/use-static-motion';
import { PathText } from '../path-text/path-text';
import { Placeholder } from '../placeholder/placeholder';
import styles from './conversation-list.module.css';

export interface ConversationItem {
  id: string;
  title: string;
  mode: 'chat' | 'work';
  updatedAt: number;
  /** Work: the folder. A second line with `showFolder`. */
  folder?: string | null;
  /** Work: started without a folder. */
  noFolder?: boolean;
  state: 'idle' | 'running' | 'waiting';
  /** Work: requests that wait for the person; a count in the time slot from 1. */
  asks?: number;
  /** Work: the permission mode, when it is not Ask. */
  permissionMode?: 'auto' | 'bypass';
  /** A fork of another conversation: a "Fork" tag after the title, read in the row's name. */
  fork?: boolean;
}

/** A group with a header that folds it. */
export interface GroupFold {
  open: boolean;
  onToggle: () => void;
  /** Rows in the group, shown or not. */
  count: number;
  /** A row that waits for the person, else one that runs: a dot on the folded header. */
  live: 'waiting' | 'running' | null;
  name: string;
  /** Muted words after the name. */
  hint: string | null;
  /** The header's hover text. */
  title?: string;
  /** "+": a new conversation in the group's place. */
  action?: { label: string; onPress: () => void };
}

/** The row under a long group's shown rows. */
export interface GroupMore {
  /** Rows not shown. 0: every row shows, and the row reads "Show fewer". */
  hidden: number;
  onMore: () => void;
  onFewer: () => void;
}

export interface ConversationGroup {
  /** Stable among the groups: a day label, a folder path. */
  key: string;
  label: string | null;
  items: readonly ConversationItem[];
  fold?: GroupFold;
  more?: GroupMore;
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
  /** A Work row's second line names its folder (search results). */
  showFolder?: boolean;
  /** For relative times. */
  now: number;
  loading?: boolean;
  /** Shown when there is no item. */
  empty?: ReactNode;
}

const ROW = 'data-conversation-row';
const ACTIONS = 'data-conversation-actions';
/** Every item of the arrow order: headers, row links, "Show n more" buttons. */
const STOP = 'data-list-stop';
const HEADER = 'data-group-header';
const HEAD_ACTION = 'data-group-action';
const GROUP = 'data-group';

const INSTANT = { duration: 0 } as const;

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

function untabActions(root: HTMLElement): void {
  for (const button of root.querySelectorAll<HTMLElement>(`[${ACTIONS}] button, [${HEAD_ACTION}] button`)) {
    if (button.tabIndex !== -1) button.tabIndex = -1;
  }
}

function headerName(fold: GroupFold, items: readonly ConversationItem[]): string {
  const name = fold.hint === null ? fold.name : `${fold.name} ${fold.hint}`;
  const count = fold.count === 1 ? CHAT.listCountOne : fill(CHAT.listCount, { n: fold.count });
  if (fold.open || fold.live === null) return `${name}, ${count}`;
  const n = items.filter((item) => item.state === fold.live).length;
  return `${name}, ${count}, ${fill(fold.live === 'waiting' ? CHAT.badgeWaiting : CHAT.badgeRunning, { n })}`;
}

/**
 * A folded group's rows leave the DOM once the fold ends; while it runs they are inert. The body stays, empty and
 * hidden, so the header's `aria-controls` always names it.
 */
function FoldBody({ id, open, children }: { id: string; open: boolean; children: ReactNode }): ReactElement {
  const staticMotion = useStaticMotion();
  const [atRest, setAtRest] = useState(!open);
  if (open && atRest) setAtRest(false);
  if (!open && staticMotion && !atRest) setAtRest(true);
  const gone = !open && atRest;
  return (
    <motion.div
      id={id}
      className={styles.body}
      hidden={gone}
      inert={!open}
      initial={false}
      animate={{ height: open ? 'auto' : 0 }}
      transition={staticMotion ? INSTANT : motionX.spring.expand}
      onAnimationComplete={() => {
        if (!open) setAtRest(true);
      }}
    >
      {gone ? null : children}
    </motion.div>
  );
}

/** The conversations of a workspace, in groups. One tab stop; arrows move over headers and rows. */
export function ConversationList(props: ConversationListProps): ReactElement {
  const rootRef = useRef<HTMLElement | null>(null);
  const seen = useRef<Set<string> | null>(null);
  /** Rows a "Show n more" reveals rise in with no animation. */
  const quiet = useRef(false);
  const reveal = useRef<{ key: string; before: Set<string> } | null>(null);
  const baseId = useId();
  const all = props.groups.flatMap((group) => group.items);

  const stops = (): HTMLElement[] =>
    rootRef.current === null
      ? []
      : [...rootRef.current.querySelectorAll<HTMLElement>(`[${STOP}]`)].filter(
          (stop) => stop.closest('[inert]') === null,
        );

  useLayoutEffect(() => {
    const root = rootRef.current;
    if (root === null) return;
    const every = [...root.querySelectorAll<HTMLElement>(`[${STOP}]`)];
    const items = every.filter((stop) => stop.closest('[inert]') === null);
    const current = items.findIndex((stop) => stop.getAttribute(ROW) === props.currentId);
    const focused = items.indexOf(document.activeElement as HTMLElement);
    const stop = focused !== -1 ? focused : current !== -1 ? current : 0;
    for (const other of every) other.tabIndex = -1;
    const chosen = items[stop];
    if (chosen !== undefined) chosen.tabIndex = 0;
    untabActions(root);
  });
  // Radix's Popover.Trigger remounts the delete button one commit after it mounts (its anchor moves), past the effect
  // above: a row that mounts alone (a group that opens) would keep a tab stop of its own.
  useEffect(() => {
    const root = rootRef.current;
    if (root === null) return;
    const observer = new MutationObserver(() => untabActions(root));
    observer.observe(root, { subtree: true, childList: true });
    return () => observer.disconnect();
  }, [props.loading]);

  const fresh = (id: string): boolean => {
    if (seen.current === null || quiet.current) return false;
    return !seen.current.has(id);
  };
  useLayoutEffect(() => {
    if (props.loading === true) return;
    seen.current = new Set(all.map((item) => item.id));
    quiet.current = false;
  });

  const focusStop = (stop: HTMLElement | null | undefined): void => {
    if (stop === null || stop === undefined) return;
    for (const other of stops()) other.tabIndex = other === stop ? 0 : -1;
    stop.focus();
    stop.scrollIntoView?.({ block: 'nearest' });
  };

  useLayoutEffect(() => {
    const wanted = reveal.current;
    if (wanted === null) return;
    reveal.current = null;
    const group = [...(rootRef.current?.querySelectorAll<HTMLElement>(`[${GROUP}]`) ?? [])].find(
      (node) => node.getAttribute(GROUP) === wanted.key,
    );
    const links = [...(group?.querySelectorAll<HTMLAnchorElement>(`a[${ROW}]`) ?? [])];
    focusStop(links.find((link) => !wanted.before.has(link.getAttribute(ROW) ?? '')));
  });

  const groupOf = (node: HTMLElement): ConversationGroup | undefined => {
    const key = node.closest(`[${GROUP}]`)?.getAttribute(GROUP);
    return key === null || key === undefined ? undefined : props.groups.find((group) => group.key === key);
  };
  const headerOf = (node: HTMLElement): HTMLElement | null =>
    node.closest(`[${GROUP}]`)?.querySelector<HTMLElement>(`[${HEADER}]`) ?? null;

  const onKeyDown = (event: KeyboardEvent<HTMLElement>): void => {
    const target = event.target as HTMLElement;
    const row = target.closest('li');
    const link = row?.querySelector<HTMLAnchorElement>(`a[${ROW}]`) ?? undefined;
    const item = all.find((entry) => entry.id === link?.getAttribute(ROW));
    if (target.hasAttribute(STOP)) {
      const items = stops();
      const index = items.indexOf(target);
      let next = -1;
      if (event.key === 'ArrowDown') next = Math.min(items.length - 1, index + 1);
      else if (event.key === 'ArrowUp') next = Math.max(0, index - 1);
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = items.length - 1;
      if (next !== -1) {
        event.preventDefault();
        focusStop(items[next]);
        return;
      }
      if (target.hasAttribute(HEADER)) {
        const fold = groupOf(target)?.fold;
        if (fold === undefined) return;
        if (event.key === 'ArrowRight') {
          event.preventDefault();
          if (!fold.open) fold.onToggle();
          else target.closest(`[${GROUP}]`)?.querySelector<HTMLElement>(`[${HEAD_ACTION}] button`)?.focus();
        } else if (event.key === 'ArrowLeft') {
          event.preventDefault();
          if (fold.open) fold.onToggle();
        }
        return;
      }
      if (event.key === 'ArrowLeft') {
        const header = headerOf(target);
        if (header !== null) {
          event.preventDefault();
          focusStop(header);
          return;
        }
      }
      if (event.key === 'ArrowRight' && link === target) {
        const first = row?.querySelector<HTMLElement>(`[${ACTIONS}] button`);
        if (first !== null && first !== undefined) {
          event.preventDefault();
          first.focus();
        }
        return;
      }
    } else if (target.closest(`[${HEAD_ACTION}]`) !== null) {
      if (event.key === 'ArrowLeft') {
        event.preventDefault();
        focusStop(headerOf(target));
      }
      return;
    } else if (target.closest(`[${ACTIONS}]`) !== null) {
      const buttons = [...(row?.querySelectorAll<HTMLElement>(`[${ACTIONS}] button`) ?? [])];
      const at = buttons.indexOf(target);
      if (event.key === 'ArrowLeft') {
        event.preventDefault();
        if (at > 0) buttons[at - 1]?.focus();
        else focusStop(link);
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

  const renderRow = (item: ConversationItem): ReactElement => {
    const current = item.id === props.currentId;
    const open = (event: MouseEvent<HTMLAnchorElement>): void => {
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.button !== 0) return;
      event.preventDefault();
      props.onOpen(item.id);
    };
    const placeLine = props.showFolder === true && item.mode === 'work';
    const noPlace = item.noFolder === true || item.folder === undefined || item.folder === null || item.folder === '';
    const loud = item.permissionMode === 'bypass';
    const twoLine = placeLine || loud;
    const asks = item.asks ?? 0;
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
          {...{ [ROW]: item.id, [STOP]: '' }}
        >
          <span className={styles.dot}>
            {item.state === 'running' ? (
              <LiveDot tone="success" pulse size={6} label={CHAT.running} />
            ) : item.state === 'waiting' ? (
              <LiveDot tone="warning" pulse size={6} label={CHAT.waiting} />
            ) : null}
            {asks > 0 ? (
              <span className={styles.srOnly}>
                {asks === 1 ? CHAT.needs.listBadgeOne : fill(CHAT.needs.listBadge, { n: asks })}
              </span>
            ) : null}
          </span>
          <span className={styles.text}>
            <span className={styles.titleLine}>
              <span className={styles.title}>{item.title}</span>
              {item.fork === true ? (
                <Badge size="sm" outline className={styles.mode} data-fork="">
                  {CHAT.history.forkTag}
                </Badge>
              ) : null}
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
                    {placeLine ? ' · ' : null}
                  </span>
                ) : null}
                {!placeLine ? null : noPlace ? (
                  <span>{CHAT.folderNone}</span>
                ) : (
                  <PathText path={item.folder ?? ''} keepEnd mono={false} />
                )}
              </span>
            ) : null}
          </span>
        </a>
        <span className={styles.time}>
          {asks > 0 ? (
            <Badge size="sm" tone="warning" pill aria-hidden="true" data-asks={asks}>
              {asks}
            </Badge>
          ) : (
            relativeTime(item.updatedAt, props.now)
          )}
        </span>
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
  };

  return (
    <nav ref={rootRef} className={styles.root} aria-label={props.label} onKeyDown={onKeyDown}>
      {all.length === 0 && props.empty !== undefined ? <div className={styles.empty}>{props.empty}</div> : null}
      {props.groups.map((group, groupIndex) => {
        if (group.items.length === 0) return null;
        const labelId = `${baseId}-g${groupIndex}`;
        const bodyId = `${baseId}-l${groupIndex}`;
        const fold = group.fold;
        const more = group.more;
        const list = (
          <ul
            className={styles.list}
            aria-labelledby={group.label !== null || fold !== undefined ? labelId : undefined}
            aria-label={group.label === null && fold === undefined ? props.label : undefined}
          >
            {group.items.map(renderRow)}
            {more !== undefined ? (
              <li className={styles.moreRow}>
                <button
                  type="button"
                  className={styles.more}
                  aria-label={
                    more.hidden > 0 ? fill(CHAT.list.moreIn, { n: more.hidden, folder: fold?.name ?? '' }) : undefined
                  }
                  onClick={() => {
                    if (more.hidden === 0) {
                      more.onFewer();
                      return;
                    }
                    quiet.current = true;
                    reveal.current = { key: group.key, before: new Set(group.items.map((item) => item.id)) };
                    more.onMore();
                  }}
                  {...{ [STOP]: '' }}
                >
                  {more.hidden > 0 ? fill(CHAT.list.more, { n: more.hidden }) : CHAT.list.fewer}
                </button>
              </li>
            ) : null}
          </ul>
        );
        const holdsCurrent = group.items.some((item) => item.id === props.currentId);
        return (
          <div key={group.key} className={styles.group} {...{ [GROUP]: group.key }}>
            {fold !== undefined ? (
              <div
                className={styles.head}
                data-open={fold.open ? '' : undefined}
                data-current={holdsCurrent ? '' : undefined}
                data-action={fold.action !== undefined ? '' : undefined}
              >
                <h3 className={styles.heading}>
                  <button
                    id={labelId}
                    type="button"
                    className={styles.header}
                    aria-expanded={fold.open}
                    aria-controls={bodyId}
                    aria-label={headerName(fold, group.items)}
                    title={fold.title}
                    onClick={fold.onToggle}
                    {...{ [STOP]: '', [HEADER]: '' }}
                  >
                    <ChevronRight className={styles.chevron} size={14} strokeWidth={1.75} aria-hidden="true" />
                    <span className={styles.headText}>
                      <MiddleText text={fold.name} className={styles.headName} title={fold.title} />
                      {fold.hint !== null ? <span className={styles.hint}>{fold.hint}</span> : null}
                    </span>
                    <span className={styles.headEnd} aria-hidden="true">
                      {fold.open ? null : (
                        <>
                          {fold.live !== null ? (
                            <LiveDot
                              tone={fold.live === 'waiting' ? 'warning' : 'success'}
                              pulse
                              size={6}
                              label={fold.live === 'waiting' ? CHAT.waiting : CHAT.running}
                            />
                          ) : null}
                          <span className={styles.count}>{fold.count}</span>
                        </>
                      )}
                    </span>
                  </button>
                </h3>
                {fold.action !== undefined ? (
                  <span className={styles.headAction} {...{ [HEAD_ACTION]: '' }}>
                    <IconButton
                      label={fold.action.label}
                      icon={<Plus size={14} strokeWidth={1.75} />}
                      onClick={fold.action.onPress}
                    />
                  </span>
                ) : null}
              </div>
            ) : group.label !== null ? (
              <p id={labelId} className={styles.groupLabel}>
                {group.label}
              </p>
            ) : null}
            {fold !== undefined ? (
              <FoldBody id={bodyId} open={fold.open}>
                {list}
              </FoldBody>
            ) : (
              list
            )}
          </div>
        );
      })}
    </nav>
  );
}
