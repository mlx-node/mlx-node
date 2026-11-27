import { Check, ChevronRight, CircleDashed, CircleHelp, Square, X } from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import { useRef, type KeyboardEvent, type ReactElement, type ReactNode } from 'react';
import { Link } from 'react-router-dom';

import { agentKeyOf } from '../../../lib/agent-keys';
import { cn } from '../../../lib/cn';
import { formatClock } from '../../../lib/format';
import type { RowMark, RowWords } from '../../../lib/helper-words';
import { motionTokens } from '../../arc/motion-tokens';
import { pressControl, useFocusFallback } from '../agent-controls/agent-controls';
import { agentStops } from '../agent-list/agent-list';
import { LiveDot } from '../live-dot/live-dot';
import { MiddleText } from '../middle-text/middle-text';
import { useStaticMotion } from '../motion/use-static-motion';
import styles from './agent-row.module.css';

export interface AgentRowProps {
  /** The child session of its newest attempt. */
  id: string;
  label: string;
  words: RowWords;
  /** The kind before the second line's facts: `Subagent · explore`. None for a workflow agent. */
  kind?: string;
  /** Where the row opens; none: it does not open. */
  to?: string;
  /** Work: the row opens its Agent view (in place of `to`); `opener` is the label, where focus comes back to. */
  onOpen?: (opener: HTMLElement) => void;
  /** Its Agent view is open now: the row reads in accent ink. */
  open?: boolean;
  /** The open link's name: `Open {label}`. */
  openLabel?: string;
  /** Its dot pulses while it runs. A block keeps its rows still: its head pulses. */
  pulse?: boolean;
  /** Facts the caller adds to the second line (its permission counts). */
  more?: readonly string[];
  /** The start time, at the end of the line (a row of its own in the thread). */
  ts?: number;
  /** Its buttons ({@link AgentControls}), at the end of the first line; X and M open them from the row. */
  controls?: ReactNode;
  /** Under the row: its permission lists. */
  children?: ReactNode;
  className?: string;
}

const ICONS: Partial<Record<RowMark, ReactNode>> = {
  asking: <CircleHelp size={14} strokeWidth={1.75} />,
  stopping: <CircleDashed size={14} strokeWidth={1.75} />,
  done: <Check size={14} strokeWidth={1.75} />,
  failed: <X size={14} strokeWidth={1.75} />,
  stopped: <Square size={12} strokeWidth={1.75} />,
  unfinished: <CircleDashed size={14} strokeWidth={1.75} />,
};

/** The state glyph of an agent: a live dot while it runs, else the mark of how it ended. */
export function AgentGlyph({
  mark,
  tone,
  pulse,
}: {
  mark: RowMark;
  tone: RowWords['tone'];
  pulse: boolean;
}): ReactNode {
  if (mark === 'running' || mark === 'quiet')
    return <LiveDot tone={mark === 'quiet' ? 'warning' : tone} pulse={pulse && mark === 'running'} label="" />;
  return ICONS[mark];
}

const INSTANT = { duration: 0 } as const;

/** The tab stop of the list `leaving` sits in, read once it left: where focus goes when a row leaves its list. */
function listStop(leaving: HTMLElement): () => HTMLElement | null {
  const list = leaving.closest<HTMLElement>('[data-agent-list]');
  return () => {
    if (list === null) return null;
    const stops = agentStops(list);
    return stops.find((stop) => stop.getAttribute('tabindex') === '0') ?? stops[0] ?? null;
  };
}

/** The row's buttons: when they go while they hold focus (its agent stops), the row's label takes it. */
function Controls({ children, row }: { children: ReactNode; row: { current: HTMLElement | null } }): ReactElement {
  const ref = useRef<HTMLSpanElement>(null);
  useFocusFallback(ref, (leaving) => {
    const stop = listStop(leaving);
    const node = row.current;
    return () => (node?.isConnected === true ? node.querySelector<HTMLElement>('[data-agent-focus]') : stop());
  });
  return (
    <span ref={ref} className={styles.controls}>
      {children}
    </span>
  );
}

/** The row's own keys: X opens its Stop confirm, M its Message, Space opens its link as Enter does. */
function onRowKey(event: KeyboardEvent<HTMLDivElement>): void {
  const row = event.currentTarget;
  const target = event.target;
  if (event.defaultPrevented || !(target instanceof HTMLElement) || !row.contains(target)) return;
  if (event.key === ' ' && target.tagName === 'A' && target.hasAttribute('data-agent-focus')) {
    event.preventDefault();
    target.click();
    return;
  }
  const action = agentKeyOf(event, 'row', { pause: false });
  if (action !== 'stop' && action !== 'message') return;
  if (pressControl(row, action)) event.preventDefault();
}

/**
 * One agent in one line and a second muted line: its state glyph with its word, its label (cut in the middle, whole
 * as its title), three facts at most; its counts and attempts under it, a failed row's reason in danger ink.
 */
export function AgentRow(props: AgentRowProps): ReactElement {
  const { words } = props;
  const staticMotion = useStaticMotion();
  const sub = [...(props.kind === undefined ? [] : [props.kind]), ...words.sub, ...(props.more ?? [])];
  const label = <MiddleText text={props.label} className={styles.labelText} />;
  const timed = props.ts !== undefined && props.ts > 0;
  const rowRef = useRef<HTMLDivElement>(null);
  // The row leaves its list while it holds focus (it ended and moved to a closed Done): the list's tab stop takes it.
  useFocusFallback(rowRef, listStop);
  return (
    <div
      ref={rowRef}
      className={cn(styles.row, props.className)}
      data-agent-row=""
      data-agent-id={props.id}
      data-mark={words.mark}
      data-tone={words.tone}
      data-open={props.open === true ? '' : undefined}
      onKeyDown={onRowKey}
    >
      <div className={styles.line}>
        <span className={styles.glyph} aria-hidden="true">
          <AnimatePresence initial={false} mode="popLayout">
            <motion.span
              key={words.mark}
              className={styles.glyphFace}
              initial={staticMotion ? false : { opacity: 0, scale: 0.6 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={staticMotion ? undefined : { opacity: 0, scale: 0.6 }}
              transition={staticMotion ? INSTANT : motionTokens.spring.snappy}
            >
              <AgentGlyph mark={words.mark} tone={words.tone} pulse={props.pulse === true} />
            </motion.span>
          </AnimatePresence>
        </span>
        {props.onOpen !== undefined ? (
          <button
            type="button"
            className={cn(styles.label, styles.open)}
            title={props.label}
            aria-label={props.openLabel ?? props.label}
            aria-expanded={props.open === true}
            data-agent-focus=""
            onClick={(event) => props.onOpen?.(event.currentTarget)}
          >
            {label}
          </button>
        ) : props.to !== undefined ? (
          <Link
            className={cn('link', styles.label)}
            to={props.to}
            title={props.label}
            aria-label={props.openLabel ?? props.label}
            data-agent-focus=""
          >
            {label}
          </Link>
        ) : (
          <span className={styles.label} title={props.label} tabIndex={-1} data-agent-focus="">
            {label}
          </span>
        )}
        <span className={styles.word} data-agent-word="">
          {words.word}
        </span>
        {words.facts.length > 0 ? (
          <span className={styles.facts} data-agent-facts="">
            {words.facts.join(' · ')}
          </span>
        ) : null}
        {timed && props.ts !== undefined ? (
          <time className={styles.time} dateTime={new Date(props.ts).toISOString()}>
            {formatClock(props.ts)}
          </time>
        ) : null}
        {props.controls !== undefined && props.controls !== null ? (
          <Controls row={rowRef}>{props.controls}</Controls>
        ) : null}
        {props.onOpen !== undefined ? (
          <span
            className={styles.chevron}
            aria-hidden="true"
            onClick={() => {
              const label = rowRef.current?.querySelector<HTMLElement>('[data-agent-focus]');
              if (label != null) props.onOpen?.(label);
            }}
          >
            <ChevronRight size={14} strokeWidth={1.75} />
          </span>
        ) : null}
      </div>
      {sub.length > 0 ? (
        <div className={styles.sub} data-helper-sub="">
          {sub.join(' · ')}
        </div>
      ) : null}
      {words.reason !== null && words.reason !== '' ? <p className={styles.reason}>{words.reason}</p> : null}
      {props.children}
    </div>
  );
}
