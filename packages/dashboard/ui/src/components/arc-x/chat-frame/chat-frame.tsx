import { ArrowDown, PanelLeft } from 'lucide-react';
import { useLayoutEffect, useRef, useState, type ReactElement, type ReactNode, type RefObject } from 'react';

import { AGENT_VIEW_DRAWER_WIDTH, agentViewPlace, type AgentViewPlace } from '../../../lib/agent-view';
import { CHAT } from '../../../lib/chat-strings';
import { Drawer, DrawerContent, useDrawerItem } from '../../arc/drawer/drawer';
import { ScrollArea } from '../../arc/scroll-area/scroll-area';
import { IconButton } from '../icon-button/icon-button';
import styles from './chat-frame.module.css';

export interface ChatFrameProps {
  /** Bar, left zone: the mode switch, then the New button. */
  barStart: ReactNode;
  /** Bar, right zone: the h1, then the conversation actions. */
  barEnd: ReactNode;
  /** The conversation list. Drawn in the docked column, or in the drawer: one instance. */
  list: ReactNode;
  /** The drawer title and the History button name. */
  listLabel: string;
  /** Drawer state; only read while collapsed. */
  listOpen: boolean;
  onListOpenChange: (open: boolean) => void;
  /** Where focus goes when the drawer closes, given what had focus when it opened. Default: that element. */
  listCloseFocus?: (opener: Element | null) => HTMLElement | null;
  /** The messages. */
  children: ReactNode;
  /** Sits 12 px above the dock, centered in the column. */
  jump?: ReactNode;
  /** The composer. */
  dock: ReactNode;
  /** The activity rail: a column right of the thread when `docked`, else a drawer from the right. */
  rail?: ChatRail;
}

export interface ChatRail {
  content: ReactNode;
  /** An agent's view: in the rail's column it takes the rail's place, else a drawer from the right. */
  view?: ChatAgentView;
  /** The column and drawer name. */
  label: string;
  /** The window is wide enough for the column. */
  docked: boolean;
  /** Docked: the column shows. Else: the drawer is open. */
  open: boolean;
  /** Only the drawer calls it (Escape, a click outside, its close button). */
  onOpenChange: (open: boolean) => void;
  /** The drawer: where focus goes once it closed, else back to what opened it. */
  closeFocus?: () => HTMLElement | null;
}

export interface ChatAgentView {
  content: ReactNode;
  /** The drawer's title: the agent's label. */
  title: string;
  place: AgentViewPlace;
  /** The drawer closed (Escape, a click outside, its close button). */
  onClose: () => void;
  /** The drawer: where focus goes as it opens; null leaves it to the drawer. */
  openFocus?: () => HTMLElement | null;
  /** The drawer: where focus goes once it closed; null leaves it where the page put it. */
  closeFocus?: () => HTMLElement | null;
}

/** The list's own roving focus scrolls it; its scroller is not a tab stop of its own. */
function untabbable(node: HTMLDivElement | null): void {
  node?.setAttribute('tabindex', '-1');
}

/** The width from which the conversation list is docked beside the thread. */
export const CHAT_DOCK_WIDTH = 840;

/** The width from which the activity rail is docked too: the list, the message column and the rail. */
export const CHAT_RAIL_DOCK_WIDTH = AGENT_VIEW_DRAWER_WIDTH;

/**
 * Whether the list (840 px) and the activity rail (1296 px) fit beside the thread, and where an agent's view stands;
 * a measured width of 0 (no layout) counts as docked.
 */
export function useChatLayout(ref: RefObject<Element | null>): {
  docked: boolean;
  railDocked: boolean;
  viewPlace: AgentViewPlace;
} {
  const width = useElementWidth(ref);
  return {
    docked: width === 0 || width >= CHAT_DOCK_WIDTH,
    railDocked: width === 0 || width >= CHAT_RAIL_DOCK_WIDTH,
    viewPlace: agentViewPlace(width),
  };
}

/** True when the element is at least 840 px wide; a measured width of 0 (no layout) counts as docked. */
export function useChatDocked(ref: RefObject<Element | null>): boolean {
  const width = useElementWidth(ref);
  return width === 0 || width >= CHAT_DOCK_WIDTH;
}

function useElementWidth(ref: RefObject<Element | null>): number {
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    const element = ref.current;
    if (element === null) return;
    const read = (next: number): void => {
      const rounded = Math.round(next);
      setWidth((current) => (current === rounded ? current : rounded));
    };
    read(element.getBoundingClientRect().width);
    if (typeof ResizeObserver !== 'function') return;
    let frame = 0;
    let pending = 0;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[entries.length - 1];
      pending = entry !== undefined ? entry.contentRect.width : element.getBoundingClientRect().width;
      if (typeof requestAnimationFrame !== 'function') {
        read(pending);
        return;
      }
      if (frame !== 0) return;
      frame = requestAnimationFrame(() => {
        frame = 0;
        read(pending);
      });
    });
    observer.observe(element);
    return () => {
      observer.disconnect();
      if (frame !== 0) cancelAnimationFrame(frame);
    };
  }, [ref]);
  return width;
}

/**
 * The Chat page frame inside the shell's one scroller: a sticky bar, a sticky conversation list (a drawer below
 * 840 px), the message column and a sticky composer dock.
 */
export function ChatFrame(props: ChatFrameProps): ReactElement {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const docked = useChatDocked(rootRef);
  const open = !docked && props.listOpen;
  const opener = useRef<Element | null>(null);
  useLayoutEffect(() => {
    if (open) opener.current = document.activeElement;
  }, [open]);
  const listCloseFocus = props.listCloseFocus;
  const onCloseAutoFocus = (event: Event): void => {
    event.preventDefault();
    const before = opener.current;
    opener.current = null;
    const target =
      listCloseFocus !== undefined ? listCloseFocus(before) : before instanceof HTMLElement ? before : null;
    target?.focus({ preventScroll: true });
  };
  const rail = props.rail;
  const railColumn = rail !== undefined && docked && rail.docked;
  const view = rail?.view;
  const viewColumn = railColumn && view !== undefined && view.place === 'column';
  const viewDrawer = view !== undefined && !viewColumn ? view : undefined;
  // The view stays drawn while its drawer slides out.
  const drawnView = useDrawerItem(viewDrawer) ?? undefined;
  const railOpen = viewColumn || rail?.open === true;
  return (
    <div ref={rootRef} className={styles.root}>
      <div
        className={styles.grid}
        data-docked={docked ? '' : undefined}
        data-list-open={open ? '' : undefined}
        data-rail={railColumn ? (railOpen ? 'open' : 'closed') : undefined}
        data-rail-view={viewColumn ? '' : undefined}
      >
        <header className={styles.bar}>
          <div className={styles.barStart}>
            {docked ? null : (
              <IconButton
                label={props.listLabel}
                icon={<PanelLeft size={16} strokeWidth={1.75} />}
                aria-expanded={open}
                onClick={() => props.onListOpenChange(true)}
              />
            )}
            {props.barStart}
          </div>
          <div className={styles.barEnd}>{props.barEnd}</div>
        </header>
        {docked ? (
          <div className={styles.list}>
            <ScrollArea className={styles.listScroll} viewportRef={untabbable}>
              {props.list}
            </ScrollArea>
          </div>
        ) : (
          <Drawer open={open} onOpenChange={props.onListOpenChange}>
            <DrawerContent
              side="left"
              title={props.listLabel}
              className={styles.drawer}
              onCloseAutoFocus={onCloseAutoFocus}
            >
              {props.list}
            </DrawerContent>
          </Drawer>
        )}
        <div className={styles.thread}>
          <section className={styles.column} aria-label="Messages">
            {props.children}
          </section>
          <div className={styles.dock}>
            <div className={styles.dockColumn}>
              {props.jump !== undefined && props.jump !== null ? <div className={styles.jump}>{props.jump}</div> : null}
              {props.dock}
            </div>
          </div>
        </div>
        {rail === undefined ? null : railColumn ? (
          <aside className={styles.rail} aria-label={viewColumn ? view.title : rail.label} hidden={!railOpen}>
            {viewColumn ? (
              <div className={styles.view}>{view.content}</div>
            ) : rail.open ? (
              <ScrollArea className={styles.railScroll}>{rail.content}</ScrollArea>
            ) : null}
          </aside>
        ) : (
          <Drawer open={rail.open} onOpenChange={rail.onOpenChange}>
            <DrawerContent
              side="right"
              title={rail.label}
              className={styles.drawer}
              onCloseAutoFocus={(event) => {
                const target = rail.closeFocus?.() ?? null;
                if (target === null) return;
                event.preventDefault();
                target.focus({ preventScroll: true });
              }}
            >
              {rail.content}
            </DrawerContent>
          </Drawer>
        )}
        {rail === undefined ? null : (
          <Drawer
            open={viewDrawer !== undefined}
            onOpenChange={(open) => {
              if (!open) viewDrawer?.onClose();
            }}
          >
            {drawnView !== undefined ? (
              <DrawerContent
                side="right"
                title={drawnView.title}
                className={drawnView.place === 'drawer' ? styles.viewDrawer : styles.viewFull}
                onOpenAutoFocus={(event) => {
                  const target = drawnView.openFocus?.() ?? null;
                  if (target === null) return;
                  event.preventDefault();
                  target.focus({ preventScroll: true });
                }}
                onCloseAutoFocus={(event) => {
                  event.preventDefault();
                  drawnView.closeFocus?.()?.focus({ preventScroll: true });
                }}
              >
                {drawnView.content}
              </DrawerContent>
            ) : null}
          </Drawer>
        )}
      </div>
    </div>
  );
}

export interface JumpButtonProps {
  /** The button itself is passed so the page can tell whether it had focus. */
  onJump: (button: HTMLButtonElement) => void;
  /** New content arrived since the user scrolled away: a dot and the longer name. */
  fresh?: boolean;
}

/** The "Jump to latest" pill; it pops in when mounted. */
export function JumpButton({ onJump, fresh = false }: JumpButtonProps): ReactElement {
  return (
    <button
      type="button"
      className={styles.jumpButton}
      aria-label={fresh ? CHAT.jumpNew : CHAT.jump}
      data-fresh={fresh ? '' : undefined}
      onClick={(event) => onJump(event.currentTarget)}
    >
      <ArrowDown size={14} strokeWidth={1.75} aria-hidden="true" />
      <span>{CHAT.jump}</span>
      {fresh ? <span className={styles.jumpDot} aria-hidden="true" /> : null}
    </button>
  );
}
