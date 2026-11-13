import { ChevronRight } from 'lucide-react';
import { motion } from 'motion/react';
import {
  Children,
  createContext,
  useContext,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
  type ReactElement,
  type ReactNode,
} from 'react';

import { cn } from '../../../lib/cn';
import { formatClock, formatDuration } from '../../../lib/format';
import { CodePanel } from '../code-panel/code-panel';
import { LiveLabel } from '../live-label/live-label';
import { MorphLoader } from '../morph-loader/morph-loader';
import type { LoaderStatus } from '../morph-loader/shapes';
import { motionX } from '../motion/motion';
import { MotionGate } from '../motion/use-entrance';
import { useStaticMotion } from '../motion/use-static-motion';
import { SwapText } from '../swap-text/swap-text';
import styles from './tool-call.module.css';

export type ToolStatus = 'pending' | 'running' | 'done' | 'failed' | 'other';

export interface ToolCallProps {
  name: string;
  /** Plain-words summary; not rendered when empty or equal to name. */
  title?: string;
  status: ToolStatus;
  /** Shown instead of the glyph's default word for 'other' ("not finished"). */
  statusText?: string;
  durationMs?: number | null;
  ts?: number;
  input?: string;
  output?: string;
  truncated?: boolean;
  /** Default: status === 'failed'. */
  defaultOpen?: boolean;
  /** Only read when status === 'running': the session is live and the body is fresh. The ring turns and the title shimmers. Default false. */
  live?: boolean;
  /** 14 px, before the name. */
  icon?: ReactNode;
  /** 'code' (default): the name in the code font; 'word': in the body font. */
  nameStyle?: 'code' | 'word';
  /** First part of the meta, before the status word and the duration. */
  detail?: string;
  /** Replaces the Input / Output fields; the row opens when it is set. */
  body?: ReactNode;
}

export interface ToolCallGroupProps {
  /** ToolCall elements. Children 9+ sit in a closed MotionGate: same words and glyphs, no shimmer. */
  children: ReactNode;
  count: number;
  failed: number;
  running: number;
  /** Default true. */
  defaultOpen?: boolean;
  /** With running > 0: the label "Using {count} tools" shimmers. Default false. */
  live?: boolean;
}

const SHIMMER_CAP = 8;
const MAX_REVEAL_STEP = 7;
const INSTANT = { duration: 0 } as const;
const ROVING = 'data-tool-roving';

const InGroup = createContext(false);

const LOADER_STATUS: Record<Exclude<ToolStatus, 'other'>, LoaderStatus> = {
  pending: 'pending',
  running: 'loading',
  done: 'success',
  failed: 'error',
};

function statusWord(status: ToolStatus, statusText: string | undefined): string {
  if (status === 'other') return statusText !== undefined && statusText.length > 0 ? statusText : 'other';
  return status;
}

/** A height that opens on `spring` and clips while it closes; children unmount when the close is at rest. */
function Collapse({
  open,
  spring,
  id,
  className,
  children,
}: {
  open: boolean;
  spring: typeof motionX.spring.glide;
  id?: string;
  className?: string;
  children: ReactNode;
}): ReactElement | null {
  const staticMotion = useStaticMotion();
  const [mounted, setMounted] = useState(open);
  const [entered, setEntered] = useState(false);
  if (open && !mounted) {
    setMounted(true);
    setEntered(true);
  }
  if (!open && mounted && staticMotion) setMounted(false);
  if (!mounted) return null;
  return (
    <motion.div
      id={id}
      className={cn(styles.collapse, className)}
      initial={open && entered && !staticMotion ? { height: 0 } : false}
      animate={{ height: open ? 'auto' : 0 }}
      transition={staticMotion ? INSTANT : spring}
      onAnimationComplete={() => {
        if (!open) setMounted(false);
      }}
    >
      <div className={styles.collapseInner} data-enter={entered && open && !staticMotion ? '' : undefined}>
        {children}
      </div>
    </motion.div>
  );
}

/** One tool call: name, summary, time and a status glyph that morphs into a check or a cross; opens to its input and output. */
export function ToolCall(props: ToolCallProps): ReactElement {
  const inGroup = useContext(InGroup);
  const bodyId = useId();
  const hasBody = props.body !== undefined || props.input !== undefined || props.output !== undefined;
  const [open, setOpen] = useState(props.defaultOpen ?? props.status === 'failed');
  const liveRunning = props.live === true && props.status === 'running';
  const title = props.title !== undefined && props.title.length > 0 && props.title !== props.name ? props.title : null;
  const duration =
    props.durationMs !== undefined && props.durationMs !== null ? formatDuration(props.durationMs) : null;
  const word = statusWord(props.status, props.statusText);
  const shownWord = props.status === 'done' || liveRunning ? null : word;
  const detail = props.detail !== undefined && props.detail !== '' ? props.detail : null;
  const meta = [detail, shownWord, duration].filter((part): part is string => part !== null).join(' · ');
  const clock = props.ts !== undefined && props.ts > 0 ? formatClock(props.ts, true) : undefined;
  const accessible = [props.name, title, detail, word, duration]
    .filter((part): part is string => part !== null)
    .join(', ');

  const content = (
    <>
      {props.icon !== undefined ? (
        <span className={styles.icon} aria-hidden="true">
          {props.icon}
        </span>
      ) : null}
      <span className={styles.name} data-style={props.nameStyle === 'word' ? 'word' : undefined}>
        {props.nameStyle === 'word' ? <SwapText text={props.name} /> : props.name}
      </span>
      <span className={styles.title}>
        {title !== null ? <LiveLabel active={liveRunning}>{title}</LiveLabel> : null}
      </span>
      <span className={styles.meta}>{meta.length > 0 ? <SwapText text={meta} /> : null}</span>
      <span className={styles.status} data-status={props.status}>
        {props.status === 'other' ? (
          <span className={styles.dash} aria-hidden="true" />
        ) : (
          <MorphLoader size={14} status={LOADER_STATUS[props.status]} spinning={liveRunning} />
        )}
        <span className={styles.srOnly}>{word}</span>
      </span>
      <span className={styles.chevron} aria-hidden="true">
        {hasBody ? <ChevronRight size={14} strokeWidth={1.75} /> : null}
      </span>
    </>
  );

  return (
    <div
      className={styles.row}
      data-status={props.status}
      data-state={open && hasBody ? 'open' : 'closed'}
      data-live={liveRunning ? '' : undefined}
      data-bare={inGroup ? undefined : ''}
      data-icon={props.icon !== undefined ? '' : undefined}
    >
      {hasBody ? (
        <div className={styles.head}>
          <button
            type="button"
            className={styles.rowButton}
            aria-expanded={open}
            aria-controls={bodyId}
            aria-label={accessible}
            title={clock}
            tabIndex={inGroup ? -1 : 0}
            {...{ [ROVING]: '' }}
            onClick={() => setOpen((value) => !value)}
          />
          {content}
        </div>
      ) : (
        <div className={styles.head} title={clock}>
          {content}
        </div>
      )}
      {hasBody ? (
        <Collapse open={open} spring={motionX.spring.glide} id={bodyId}>
          <div className={styles.body}>
            {props.body !== undefined ? props.body : null}
            {props.body === undefined && props.input !== undefined ? (
              <div className={styles.field}>
                <span className={styles.fieldLabel}>Input</span>
                <CodePanel code={props.input} label="Input" wrap />
              </div>
            ) : null}
            {props.body === undefined && props.output !== undefined ? (
              <div className={styles.field}>
                <span className={styles.fieldLabel}>Output</span>
                <CodePanel
                  code={props.output}
                  label="Output"
                  maxLines={12}
                  wrap
                  className={props.status === 'failed' ? styles.failedOutput : undefined}
                />
              </div>
            ) : null}
            {props.truncated === true ? <p className={styles.note}>Long fields are truncated for display.</p> : null}
          </div>
        </Collapse>
      ) : null}
    </div>
  );
}

/** A captioned part of a row's body ("Command", "Output"), as the row's own input and output are shown. */
export function ToolCallField({ label, children }: { label: string; children: ReactNode }): ReactElement {
  return (
    <div className={styles.field}>
      <span className={styles.fieldLabel}>{label}</span>
      {children}
    </div>
  );
}

/** Consecutive tool calls in one card. One tab stop: arrows move between the header and the rows. */
export function ToolCallGroup(props: ToolCallGroupProps): ReactElement {
  const [open, setOpen] = useState(props.defaultOpen ?? true);
  const [toggled, setToggled] = useState(false);
  const [active, setActive] = useState(0);
  const focusNext = useRef(false);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const listId = useId();
  const label = props.running > 0 ? `Using ${props.count} tools` : `Used ${props.count} tools`;

  const items = (): HTMLElement[] => {
    const root = rootRef.current;
    if (root === null) return [];
    return [...root.querySelectorAll<HTMLElement>(`[${ROVING}]`)].filter(
      (item) => item.closest('[data-tool-group]') === root,
    );
  };

  useLayoutEffect(() => {
    const all = items();
    if (all.length === 0) return;
    const index = Math.min(active, all.length - 1);
    all.forEach((item, i) => {
      item.tabIndex = i === index ? 0 : -1;
    });
    if (focusNext.current) {
      focusNext.current = false;
      all[index]?.focus();
    }
  });

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    const all = items();
    const current = all.indexOf(document.activeElement as HTMLElement);
    if (current === -1) return;
    let next = current;
    if (event.key === 'ArrowDown') next = Math.min(all.length - 1, current + 1);
    else if (event.key === 'ArrowUp') next = Math.max(0, current - 1);
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = all.length - 1;
    else return;
    event.preventDefault();
    focusNext.current = true;
    setActive(next);
  };

  const onFocus = (): void => {
    const index = items().indexOf(document.activeElement as HTMLElement);
    if (index !== -1 && index !== active) setActive(index);
  };

  const rows = Children.toArray(props.children);

  return (
    <div
      ref={rootRef}
      className={styles.group}
      role="group"
      aria-label={`${props.count} tool calls, ${props.failed} failed`}
      data-state={open ? 'open' : 'closed'}
      data-tool-group=""
      onKeyDown={onKeyDown}
      onFocus={onFocus}
    >
      <div className={styles.groupHeader}>
        <button
          type="button"
          className={styles.groupToggle}
          aria-expanded={open}
          aria-controls={listId}
          {...{ [ROVING]: '' }}
          onClick={() => {
            setOpen((value) => !value);
            setToggled(true);
          }}
        >
          <LiveLabel active={props.live === true && props.running > 0} className={styles.groupLabel}>
            {label}
          </LiveLabel>
          {props.failed > 0 ? (
            <>
              {' '}
              <span className={styles.failedCount}>{`${props.failed} failed`}</span>
            </>
          ) : null}
          <ChevronRight className={styles.groupChevron} size={14} strokeWidth={1.75} aria-hidden="true" />
        </button>
      </div>
      <Collapse open={open} spring={motionX.spring.glide} id={listId} className={styles.listClip}>
        <InGroup.Provider value={true}>
          <div className={styles.list}>
            {rows.map((row, index) => (
              <div
                key={index}
                className={styles.item}
                data-reveal={toggled ? '' : undefined}
                style={{ '--i': Math.min(index, MAX_REVEAL_STEP) } as CSSProperties}
              >
                <MotionGate open={index < SHIMMER_CAP}>{row}</MotionGate>
              </div>
            ))}
          </div>
        </InGroup.Provider>
      </Collapse>
    </div>
  );
}
