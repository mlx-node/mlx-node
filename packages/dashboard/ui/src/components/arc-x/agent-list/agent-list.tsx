import { AnimatePresence, motion } from 'motion/react';
import {
  useLayoutEffect,
  useRef,
  type FocusEvent,
  type KeyboardEvent,
  type ReactElement,
  type ReactNode,
  type RefObject,
} from 'react';

import { agentKeyOf } from '../../../lib/agent-keys';
import { cn } from '../../../lib/cn';
import { motionTokens } from '../../arc/motion-tokens';
import { ScrollArea } from '../../arc/scroll-area/scroll-area';
import { Disclosure } from '../disclosure/disclosure';
import { useStaticMotion } from '../motion/use-static-motion';
import styles from './agent-list.module.css';

export interface AgentListGroup {
  key: string;
  title: ReactNode;
  /** `open`: a heading, always open (Needs attention, Running). `fold`: closed until opened (Done). */
  mode: 'open' | 'fold';
  /** Muted words after the title (`2 not started yet`). */
  note?: string;
  rows: ReadonlyArray<{ id: string; node: ReactNode }>;
  /** `fold`: controlled open state. */
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}

export interface AgentListProps {
  /** The list's name for a screen reader. */
  label: string;
  groups: readonly AgentListGroup[];
  /** px the list scrolls inside. Default 320. */
  maxHeight?: number;
  /** The person scrolled the list. */
  onScrolled?: () => void;
  className?: string;
}

const FOCUS = '[data-agent-focus]';
const LIST = '[data-agent-list]';
const INSTANT = { duration: 0 } as const;

/**
 * The rows of an agent list that its tab stop moves over: its own (a nested list's rows are that list's), and none
 * in a fold that is closed or closing (it keeps its rows mounted, hidden once it comes to rest).
 */
export function agentStops(list: HTMLElement): HTMLElement[] {
  return ownRows(list).filter((item) => {
    const hidden = item.closest('[hidden], [data-agent-closed]');
    return hidden === null || !list.contains(hidden);
  });
}

function ownRows(list: HTMLElement): HTMLElement[] {
  return [...list.querySelectorAll<HTMLElement>(FOCUS)].filter((item) => item.closest(LIST) === list);
}

export interface AgentRoving<T extends HTMLElement> {
  ref: RefObject<T | null>;
  onFocus: (event: FocusEvent<T>) => void;
  onKeyDown: (event: KeyboardEvent<T>) => void;
}

/**
 * One tab stop for the rows of an agent list, ↑ ↓ Home End moving between them: for the element marked
 * `data-agent-list`. After each render the stop stays on its row, or on the last row when fewer are left.
 */
export function useAgentRoving<T extends HTMLElement>(): AgentRoving<T> {
  const ref = useRef<T | null>(null);
  const current = useRef(0);

  const items = (): HTMLElement[] => (ref.current === null ? [] : agentStops(ref.current));
  // A hidden row gets -1 too: it must not come back from its fold as a second tab stop.
  const settle = (list: HTMLElement[]): void => {
    if (list.length === 0 || ref.current === null) return;
    current.current = Math.min(current.current, list.length - 1);
    const stop = list[current.current];
    for (const item of ownRows(ref.current)) item.tabIndex = item === stop ? 0 : -1;
  };

  useLayoutEffect(() => settle(items()));

  const onFocus = (event: FocusEvent<T>): void => {
    const list = items();
    const at = list.indexOf(event.target as HTMLElement);
    if (at === -1) return;
    current.current = at;
    settle(list);
  };

  const onKeyDown = (event: KeyboardEvent<T>): void => {
    if (event.defaultPrevented) return;
    const action = agentKeyOf(event, 'row', { pause: false });
    if (action !== 'up' && action !== 'down' && action !== 'first' && action !== 'last') return;
    const list = items();
    const at = list.indexOf(document.activeElement as HTMLElement);
    if (at === -1) return;
    const next =
      action === 'down'
        ? Math.min(list.length - 1, at + 1)
        : action === 'up'
          ? Math.max(0, at - 1)
          : action === 'first'
            ? 0
            : list.length - 1;
    event.preventDefault();
    current.current = next;
    settle(list);
    list[next]!.focus();
  };

  return { ref, onFocus, onKeyDown };
}

/**
 * A run's agents by need, inside a fixed height: one tab stop for the rows (↑ ↓ Home End move between them), each
 * group a heading or a fold. A row that enters rises in; one that leaves its group fades out.
 */
export function AgentList(props: AgentListProps): ReactElement {
  const staticMotion = useStaticMotion();
  const roving = useAgentRoving<HTMLDivElement>();

  const rise = staticMotion ? INSTANT : motionTokens.spring.smooth;
  const rows = (group: AgentListGroup): ReactNode => (
    <ul className={styles.rows} data-agent-closed={group.mode === 'fold' && group.open === false ? '' : undefined}>
      <AnimatePresence initial={false}>
        {group.rows.map((row) => (
          <motion.li
            key={row.id}
            className={styles.item}
            initial={staticMotion ? false : { opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={staticMotion ? undefined : { opacity: 0, transition: { duration: motionTokens.duration.exit } }}
            transition={rise}
          >
            {row.node}
          </motion.li>
        ))}
      </AnimatePresence>
    </ul>
  );

  return (
    <ScrollArea
      className={cn(styles.root, props.className)}
      maxHeight={props.maxHeight ?? 320}
      label={props.label}
      onScroll={props.onScrolled}
    >
      <div
        ref={roving.ref}
        className={styles.groups}
        data-agent-list=""
        onFocus={roving.onFocus}
        onKeyDown={roving.onKeyDown}
      >
        {props.groups.map((group) =>
          group.mode === 'fold' ? (
            <Disclosure
              key={group.key}
              title={group.title}
              meta={group.note}
              open={group.open}
              onOpenChange={group.onOpenChange}
              className={styles.fold}
            >
              {rows(group)}
            </Disclosure>
          ) : (
            <section key={group.key} className={styles.group} data-agent-group={group.key}>
              <h4 className={styles.heading}>
                {group.title}
                {group.note !== undefined ? <span className={styles.note}>{` · ${group.note}`}</span> : null}
              </h4>
              {rows(group)}
            </section>
          ),
        )}
      </div>
    </ScrollArea>
  );
}
