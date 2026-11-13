import { AnimatePresence, motion } from 'motion/react';
import {
  forwardRef,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactElement,
  type ReactNode,
  type Ref,
} from 'react';

import { motionX } from '../motion/motion';
import { useStaticMotion } from '../motion/use-static-motion';
import styles from './interaction-card.module.css';

export interface InteractionCardProps {
  icon?: ReactNode;
  title: string;
  /** Right of the title: "in ~/code/app", "1 of 3". */
  meta?: ReactNode;
  /** The whole of what `meta` may shorten, read with the card's description: "in /Users/me/code/app". */
  metaText?: string;
  /** Ids of the elements that hold what the card asks about (its exact text), read when focus enters the card. */
  describedBy?: readonly string[];
  /** The agent's own words, plain text, one line. */
  subtitle?: string;
  /** The subtitle's hover text when the subtitle is a part of it. Default `subtitle`. */
  subtitleTitle?: string;
  /** A code well, a Markdown well, a text field. */
  children?: ReactNode;
  /** Warning lines above the actions. They block nothing. */
  warnings?: readonly string[];
  /** Buttons in tab order. */
  actions: ReactNode;
  /** Small lines under the actions. */
  notes?: readonly string[];
  busy?: boolean;
  warningIcon?: ReactNode;
  /** A neutral first line of the warning list: why this request asks although a rule could have allowed it. */
  reason?: { icon: ReactNode; text: string };
}

const INSTANT = { duration: 0 } as const;

/** A request of the agent drawn in the thread: what it asks, the exact thing it will do, and the answers it offers. */
export const InteractionCard = forwardRef<HTMLElement, InteractionCardProps>(
  function InteractionCard(props, ref): ReactElement {
    const staticMotion = useStaticMotion();
    const titleId = useId();
    const metaId = useId();
    const warningsId = useId();
    const hasMeta = props.metaText !== undefined && props.metaText !== '';
    const hasWarnings = (props.warnings !== undefined && props.warnings.length > 0) || props.reason !== undefined;
    const described = [
      ...(props.describedBy ?? []),
      ...(hasWarnings ? [warningsId] : []),
      ...(hasMeta ? [metaId] : []),
    ].join(' ');
    return (
      <motion.div
        className={styles.clip}
        initial={staticMotion ? false : { height: 0 }}
        animate={{ height: 'auto' }}
        exit={staticMotion ? { height: 0, transition: INSTANT } : { height: 0 }}
        transition={staticMotion ? INSTANT : motionX.spring.expand}
      >
        <section
          ref={ref}
          className={styles.root}
          role="group"
          aria-labelledby={titleId}
          aria-describedby={described === '' ? undefined : described}
          data-interaction-card=""
          aria-busy={props.busy === true ? true : undefined}
        >
          <div className={styles.top}>
            <div className={styles.head}>
              {props.icon !== undefined ? (
                <span className={styles.icon} aria-hidden="true">
                  {props.icon}
                </span>
              ) : null}
              <h3 id={titleId} className={styles.title}>
                {props.title}
              </h3>
              {props.meta !== undefined ? (
                <span className={styles.meta} aria-hidden={hasMeta ? true : undefined}>
                  {props.meta}
                </span>
              ) : null}
              {hasMeta ? (
                <span id={metaId} className={styles.srOnly}>
                  {props.metaText}
                </span>
              ) : null}
            </div>
            {props.subtitle !== undefined && props.subtitle !== '' ? (
              <p className={styles.subtitle} title={props.subtitleTitle ?? props.subtitle}>
                {props.subtitle}
              </p>
            ) : null}
          </div>
          {props.children}
          {hasWarnings ? (
            <ul id={warningsId} className={styles.warnings}>
              {props.reason !== undefined ? (
                <li className={styles.warning} data-tone="neutral">
                  <span className={styles.warningIcon} aria-hidden="true">
                    {props.reason.icon}
                  </span>
                  <span>{props.reason.text}</span>
                </li>
              ) : null}
              {(props.warnings ?? []).map((line) => (
                <li key={line} className={styles.warning}>
                  {props.warningIcon !== undefined ? (
                    <span className={styles.warningIcon} aria-hidden="true">
                      {props.warningIcon}
                    </span>
                  ) : null}
                  <span>{line}</span>
                </li>
              ))}
            </ul>
          ) : null}
          <div className={styles.foot}>
            <div className={styles.actions}>{props.actions}</div>
            {props.notes !== undefined && props.notes.length > 0 ? (
              <div className={styles.notes}>
                {props.notes.map((note) => (
                  <p key={note} className={styles.note}>
                    {note}
                  </p>
                ))}
              </div>
            ) : null}
          </div>
        </section>
      </motion.div>
    );
  },
);

/** Lets a card that leaves clip shut instead of vanishing. Key each card by its request id. */
export function InteractionPresence({ children }: { children: ReactNode }): ReactElement {
  return <AnimatePresence initial={false}>{children}</AnimatePresence>;
}

/** Lines a well shows before it scrolls (15rem of code lines). */
export const EXACT_VISIBLE_LINES = 12;
/** A run of this many spaces or tabs, or of two or more empty lines, is shown as a marked count. */
const SPACE_RUN = 16;

function lineCount(text: string): number {
  let lines = 1;
  for (let i = 0; i < text.length; i += 1) if (text[i] === '\n') lines += 1;
  return lines;
}

function blank(line: string): boolean {
  for (const c of line) if (c !== ' ' && c !== '\t') return false;
  return true;
}

export interface GapWords {
  /** "{n} spaces" */
  spaces: string;
  /** "{n} empty lines" */
  emptyLines: string;
}

function fill(template: string, n: number): string {
  return template.split('{n}').join(String(n));
}

/** One line, with long runs of spaces or tabs shown as a marked count instead of empty room. */
function lineParts(line: string, key: string, words: GapWords, out: ReactNode[]): void {
  let from = 0;
  let i = 0;
  while (i < line.length) {
    if (line[i] !== ' ' && line[i] !== '\t') {
      i += 1;
      continue;
    }
    let end = i;
    while (end < line.length && (line[end] === ' ' || line[end] === '\t')) end += 1;
    if (end - i >= SPACE_RUN) {
      out.push(line.slice(from, i));
      out.push(
        <span key={`${key}s${i}`} className={styles.gap}>
          {fill(words.spaces, end - i)}
        </span>,
      );
      from = end;
    }
    i = end;
  }
  out.push(line.slice(from));
}

/**
 * The text of an exact well. Padding that would push the rest out of sight shows as a marked count: two or more empty
 * lines in a row, or a long run of spaces. Every other character is shown as it is.
 */
export function exactParts(text: string, words: GapWords): ReactNode[] {
  const lines = text.split('\n');
  const out: ReactNode[] = [];
  let k = 0;
  while (k < lines.length) {
    let run = k;
    while (run < lines.length && blank(lines[run]!) && run < lines.length - 1) run += 1;
    if (run - k >= 2) {
      out.push(
        <span key={`e${k}`} className={styles.gap} data-gap-lines="">
          {fill(words.emptyLines, run - k)}
        </span>,
      );
      out.push('\n');
      k = run;
      continue;
    }
    lineParts(lines[k]!, `l${k}`, words, out);
    if (k < lines.length - 1) out.push('\n');
    k += 1;
  }
  return out;
}

export interface ExactTextProps {
  text: string;
  label: string;
  id?: string;
  /** The line under a well that does not fit, from its number of lines. */
  overflowNote: (lines: number) => string;
  gapWords: GapWords;
}

/**
 * A scrolling well for the exact text a request runs or changes: never cut, wrapped, plain text. When it does not fit,
 * a line under it says how long it is, and the end fades until it was scrolled to.
 */
export function ExactText(props: ExactTextProps): ReactElement {
  const ref = useRef<HTMLPreElement | null>(null);
  const lines = lineCount(props.text);
  const [measured, setMeasured] = useState(false);
  const [atEnd, setAtEnd] = useState(false);
  const noteId = useId();
  useLayoutEffect(() => {
    const node = ref.current;
    if (node === null) return;
    const check = (): void => {
      setMeasured(node.scrollHeight > node.clientHeight + 1);
      setAtEnd(node.scrollTop + node.clientHeight >= node.scrollHeight - 1);
    };
    check();
    if (typeof ResizeObserver !== 'function') return;
    const observer = new ResizeObserver(check);
    observer.observe(node);
    return () => observer.disconnect();
  }, [props.text]);
  const parts = exactParts(props.text, props.gapWords);
  let shownLines = 1;
  for (const part of parts) if (part === '\n') shownLines += 1;
  const overflow = measured || shownLines > EXACT_VISIBLE_LINES;
  return (
    <div className={styles.exactWrap} data-overflow={overflow ? '' : undefined} data-end={atEnd ? '' : undefined}>
      <pre
        ref={ref}
        id={props.id}
        className={styles.exact}
        role="region"
        aria-label={props.label}
        aria-describedby={overflow ? noteId : undefined}
        tabIndex={0}
        onScroll={(event) => {
          const node = event.currentTarget;
          setAtEnd(node.scrollTop + node.clientHeight >= node.scrollHeight - 1);
        }}
      >
        <code>{parts}</code>
      </pre>
      {overflow ? (
        <p id={noteId} className={styles.exactNote}>
          {props.overflowNote(lines)}
        </p>
      ) : null}
    </div>
  );
}

/** A scrolling well around rich content (a plan). */
export function Well({ children, label, id }: { children: ReactNode; label: string; id?: string }): ReactElement {
  return (
    <div id={id} className={styles.well} role="region" aria-label={label} tabIndex={0}>
      {children}
    </div>
  );
}

/** A labelled multi-line text field inside a card (plan feedback). Escape calls `onEscape`. */
export function CardTextField(props: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  fieldRef?: Ref<HTMLTextAreaElement>;
  maxLength?: number;
  /** "{n} of {max} characters", shown under the field when `maxLength` is set. */
  countText?: string;
  onEscape?: () => void;
}): ReactElement {
  const countId = useId();
  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>): void => {
    if (event.key !== 'Escape' || props.onEscape === undefined) return;
    event.preventDefault();
    event.stopPropagation();
    props.onEscape();
  };
  const field = (
    <label className={styles.fieldLabel}>
      {props.label}
      <textarea
        ref={props.fieldRef}
        rows={3}
        value={props.value}
        maxLength={props.maxLength}
        aria-describedby={props.countText !== undefined ? countId : undefined}
        onChange={(event) => props.onChange(event.target.value)}
        onKeyDown={onKeyDown}
        className={styles.field}
      />
    </label>
  );
  if (props.countText === undefined) return field;
  return (
    <div className={styles.fieldBox}>
      {field}
      <span id={countId} className={styles.count}>
        {props.countText}
      </span>
    </div>
  );
}
