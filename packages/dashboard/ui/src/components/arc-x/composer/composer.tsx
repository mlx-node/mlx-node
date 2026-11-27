import { ArrowBigUp, CircleAlert, CornerDownLeft, Info, TriangleAlert } from 'lucide-react';
import { motion } from 'motion/react';
import {
  useCallback,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ClipboardEvent,
  type DragEvent,
  type FormEvent,
  type KeyboardEvent,
  type ReactElement,
  type ReactNode,
  type Ref,
} from 'react';

import { slashToken } from '../../../lib/chat-input';
import { CHAT, fill } from '../../../lib/chat-strings';
import { formatNumber } from '../../../lib/format';
import { hasNonWhitespace } from '../../../lib/text-scan';
import { Button } from '../../arc/button/button';
import { CommandMenu, commandOrder, typedInFull, type CommandRow } from '../command-menu/command-menu';
import { ContextPopover } from '../context-panel/context-popover';
import { LiveLabel } from '../live-label/live-label';
import { motionX } from '../motion/motion';
import { useStaticMotion } from '../motion/use-static-motion';
import styles from './composer.module.css';
import { ContextRing, type ContextUse } from './context-ring';
import { SendButton, type ComposerPhase } from './send-button';

export { SendButton, sendFace, type ComposerPhase, type Queueing, type SendButtonProps } from './send-button';
export { contextText, type ContextUse } from './context-ring';

export interface ComposerNotice {
  tone: 'neutral' | 'warning' | 'danger';
  text: string;
  /** The whole sentence when `text` is the short one: the strip's hover title. */
  title?: string;
  /** The text shimmers (work in progress). */
  working?: boolean;
  action?: { label: string; onSelect: () => void };
  secondary?: { label: string; onSelect: () => void };
}

export interface ComposerProps {
  value: string;
  onValueChange: (value: string) => void;
  placeholder: string;
  phase: ComposerPhase;
  onSend: () => void;
  onStop: () => void;
  /**
   * While an answer starts or runs, or an op runs: Enter and the round button queue the text (the button reads Queue
   * and, for an answer, Stop moves to a small square before it). Without it the box only sends while idle.
   */
  onQueue?: () => void;
  /** ⌘Enter (Ctrl+Enter) while an answer starts or runs: the text goes into it. Without it ⌘Enter queues. */
  onSendNow?: () => void;
  /** Why Send (or Queue) is off although there is text. Undefined: Send is on when `value` is not blank. */
  blocked?: string;
  /** Why Queue alone is off (the list is full); ⌘Enter still goes to `onSendNow`, which needs no place in the list. */
  queueOff?: string;
  /** Words in the reason slot that leave Send on (a command's words, "Enter to run"); the ring gives way to them. */
  hint?: string;
  /**
   * The `/` command list: open while the text starts with `/` and the caret is in that first word; it never takes
   * focus. ↑ ↓ move (wrapping), Enter or Tab fills `/name `, Enter on a name typed in full runs it, Esc closes the
   * list only. `later`: a muted last line, more commands come once the conversation is loaded.
   */
  commands?: { rows: readonly CommandRow[]; later: boolean };
  notice?: ComposerNotice;
  /** The queued messages, under the notice strip and above the attachments. */
  queue?: ReactNode;
  /** Alt+↑ in an empty box: open the last queued message for edit. */
  onEditLast?: () => void;
  /** Set while a queued message is open for edit: Esc in the box closes that edit, and only the next Esc stops. */
  onCancelEdit?: () => void;
  /** Model picker and chips. */
  tools: ReactNode;
  /** How full the context window is: a ring at the end of the tools row while no reason or counter shows. */
  context?: ContextUse;
  /**
   * The Context panel: the ring becomes its button and the panel opens upward from it; while the ring is not drawn the
   * panel hangs from the end of the tools row.
   */
  contextPanel?: { open: boolean; onOpenChange: (open: boolean) => void; content: ReactNode };
  /** Default 100_000. */
  maxLength?: number;
  /** Accessible name of the textarea. Default 'Message'. */
  label?: string;
  textareaRef?: Ref<HTMLTextAreaElement>;
  /** The attached files, between the notice and the text field. */
  attachments?: ReactNode;
  /** Files pasted (with no plain text beside them) or dropped on the box. Without it the box ignores files. */
  onFiles?: (files: File[]) => void;
}

const MAX_LENGTH = 100_000;
const INSTANT = { duration: 0 } as const;

/** Send is on: idle, text that is not blank, nothing blocks it and it is not too long. */
export function composerCanSend(
  value: string,
  phase: ComposerPhase,
  blocked: string | undefined,
  max: number,
): boolean {
  return phase === 'idle' && blocked === undefined && hasNonWhitespace(value) && value.length <= max;
}

/** Queue is on: an answer starts or runs or an op runs, text that is not blank, nothing blocks it, not too long. */
export function composerCanQueue(
  value: string,
  phase: ComposerPhase,
  blocked: string | undefined,
  max: number,
): boolean {
  return (
    (phase === 'starting' || phase === 'running' || phase === 'op') &&
    blocked === undefined &&
    hasNonWhitespace(value) &&
    value.length <= max
  );
}

function setRef<T>(ref: Ref<T> | undefined, value: T | null): void {
  if (ref === undefined || ref === null) return;
  if (typeof ref === 'function') ref(value);
  else (ref as { current: T | null }).current = value;
}

/**
 * A tool part marked `data-fit-whole` wraps below the row's line when the chips at their floors leave no room for it.
 * Such a part is not drawn at all, so its room goes back to the chips.
 */
function placeWholeParts(fit: HTMLElement): void {
  const parts = fit.querySelectorAll<HTMLElement>('[data-fit-whole]');
  if (parts.length === 0) return;
  for (const part of parts) part.removeAttribute('data-dropped');
  const line = fit.firstElementChild?.getBoundingClientRect().bottom ?? 0;
  for (const part of parts) {
    const box = part.getBoundingClientRect();
    if (box.height > 0 && box.top >= line) part.setAttribute('data-dropped', '');
  }
}

function NoticeStrip({ notice }: { notice: ComposerNotice }): ReactElement {
  const staticMotion = useStaticMotion();
  const Icon = notice.tone === 'danger' ? CircleAlert : notice.tone === 'warning' ? TriangleAlert : Info;
  return (
    <motion.div
      className={styles.noticeClip}
      initial={staticMotion ? false : { height: 0 }}
      animate={{ height: 'auto' }}
      transition={staticMotion ? INSTANT : motionX.spring.expand}
    >
      <div
        className={styles.notice}
        data-tone={notice.tone}
        role={notice.tone === 'danger' ? 'alert' : 'status'}
        title={notice.title}
      >
        <Icon className={styles.noticeIcon} size={14} strokeWidth={1.75} aria-hidden="true" />
        <LiveLabel active={notice.working === true} className={styles.noticeText}>
          {notice.text}
        </LiveLabel>
        {notice.action !== undefined ? (
          <Button type="button" variant="ghost" size="sm" onClick={notice.action.onSelect}>
            {notice.action.label}
          </Button>
        ) : null}
        {notice.secondary !== undefined ? (
          <Button type="button" variant="ghost" size="sm" onClick={notice.secondary.onSelect}>
            {notice.secondary.label}
          </Button>
        ) : null}
      </div>
    </motion.div>
  );
}

/** The message box: a growing text field, an optional notice strip, the tools row and the Send / Stop button. */
export function Composer(props: ComposerProps): ReactElement {
  const max = props.maxLength ?? MAX_LENGTH;
  const label = props.label ?? CHAT.composerLabel;
  const area = useRef<HTMLTextAreaElement | null>(null);
  const fit = useRef<HTMLDivElement | null>(null);
  const fitWatch = useRef<ResizeObserver | null>(null);
  const ring = useRef<HTMLElement | null>(null);
  const end = useRef<HTMLDivElement | null>(null);
  const tooLong = props.value.length > max;
  const canSend = composerCanSend(props.value, props.phase, props.blocked, max);
  const queueable = composerCanQueue(props.value, props.phase, props.blocked, max);
  const canQueue = props.onQueue !== undefined && queueable && props.queueOff === undefined;
  const canSendNow = props.onSendNow !== undefined && queueable;
  const blank = !hasNonWhitespace(props.value);
  const onQueue = props.onQueue;
  const queueing =
    onQueue !== undefined && !blank && (props.phase === 'starting' || props.phase === 'running' || props.phase === 'op')
      ? { enabled: canQueue, onQueue }
      : undefined;
  const reason = props.blocked ?? (tooLong ? CHAT.tooLong : props.queueOff);
  const counter =
    props.value.length >= max * 0.9 && !tooLong
      ? fill(CHAT.counter, { length: formatNumber(props.value.length), max: formatNumber(max) })
      : null;
  const sideId = useId();
  const reasonShown = reason !== undefined && !blank;
  const hintShown = !reasonShown && counter === null && props.hint !== undefined && !blank;
  const side = reasonShown ? 'reason' : counter !== null ? 'count' : hintShown ? 'hint' : 'none';

  const listId = useId();
  const optionId = useCallback((index: number) => `${listId}-${index}`, [listId]);
  const [focused, setFocused] = useState(false);
  const [caret, setCaret] = useState<number | null>(null);
  const [closedAt, setClosedAt] = useState<string | null>(null);
  const [picked, setPicked] = useState({ typed: '', index: 0 });
  const nextCaret = useRef<number | null>(null);
  const commandRows = props.commands?.rows;
  const token = commandRows === undefined ? null : slashToken(props.value);
  const typed = token?.typed ?? '';
  const shownRows = useMemo(
    () => (commandRows === undefined || token === null ? [] : commandOrder(commandRows, typed)),
    [commandRows, token, typed],
  );
  const at = caret ?? props.value.length;
  const tokenKey = token === null ? null : `${token.start}:${typed}`;
  const menuOpen =
    token !== null && focused && at > token.start && at <= token.end && closedAt !== tokenKey && shownRows.length > 0;
  const active = picked.typed === typed ? Math.min(picked.index, shownRows.length - 1) : 0;

  const fillRow = (index: number): void => {
    const row = shownRows[index];
    if (token === null || row === undefined) return;
    const rest = props.value.slice(token.end);
    const tail = rest.startsWith(' ') ? rest : ` ${rest}`;
    const text = `${props.value.slice(0, token.start)}/${row.name}${tail}`;
    const want = token.start + row.name.length + 2;
    // The same text does not change the value, so the caret effect would not run and the request would stay pending.
    if (text === props.value) {
      area.current?.setSelectionRange(want, want);
      setCaret(want);
      return;
    }
    nextCaret.current = want;
    props.onValueChange(text);
  };

  const attach = useCallback(
    (node: HTMLTextAreaElement | null) => {
      area.current = node;
      setRef(props.textareaRef, node);
    },
    [props.textareaRef],
  );

  useLayoutEffect(() => {
    const box = fit.current;
    if (box === null || typeof ResizeObserver === 'undefined') return;
    let frame = 0;
    const watch = new ResizeObserver(() => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => placeWholeParts(box));
    });
    watch.observe(box);
    fitWatch.current = watch;
    return () => {
      cancelAnimationFrame(frame);
      watch.disconnect();
      fitWatch.current = null;
    };
  }, []);

  useLayoutEffect(() => {
    const box = fit.current;
    if (box === null) return;
    for (const chip of box.firstElementChild?.children ?? []) fitWatch.current?.observe(chip);
    placeWholeParts(box);
  });

  useLayoutEffect(() => {
    const node = area.current;
    const want = nextCaret.current;
    if (node === null || want === null) return;
    nextCaret.current = null;
    node.setSelectionRange(want, want);
    setCaret(want);
  }, [props.value]);

  useLayoutEffect(() => {
    const node = area.current;
    if (node === null) return;
    node.style.height = 'auto';
    const height = node.scrollHeight;
    if (height > 0) node.style.height = `${height}px`;
    const cap = Number.parseFloat(getComputedStyle(node).maxHeight);
    if (Number.isFinite(cap) && height > cap) node.setAttribute('data-at-max', '');
    else node.removeAttribute('data-at-max');
  }, [props.value]);

  const submit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    if (canSend) props.onSend();
    else if (canQueue) props.onQueue?.();
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>): void => {
    const native = event.nativeEvent;
    if (native.isComposing || native.keyCode === 229) return;
    if (menuOpen && !event.altKey && !event.metaKey && !event.ctrlKey) {
      const count = shownRows.length;
      if ((event.key === 'ArrowDown' || event.key === 'ArrowUp') && !event.shiftKey) {
        event.preventDefault();
        setPicked({ typed, index: (active + (event.key === 'ArrowDown' ? 1 : count - 1)) % count });
        return;
      }
      if (event.key === 'Escape') {
        event.preventDefault();
        setClosedAt(tokenKey);
        return;
      }
      if (event.key === 'Tab' && !event.shiftKey) {
        event.preventDefault();
        fillRow(active);
        return;
      }
      if (event.key === 'Enter' && !event.shiftKey && !typedInFull(shownRows[active], typed)) {
        event.preventDefault();
        fillRow(active);
        return;
      }
    }
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      if (canSend) props.onSend();
      else if (canSendNow && (event.metaKey || event.ctrlKey)) props.onSendNow?.();
      else if (canQueue) props.onQueue?.();
      return;
    }
    if (event.key === 'ArrowUp' && event.altKey && props.value === '' && props.onEditLast !== undefined) {
      event.preventDefault();
      props.onEditLast();
      return;
    }
    if (event.key === 'Escape' && !event.repeat && props.onCancelEdit !== undefined) {
      event.preventDefault();
      props.onCancelEdit();
      return;
    }
    if (event.key === 'Escape' && !event.repeat && (props.phase === 'starting' || props.phase === 'running')) {
      event.preventDefault();
      props.onStop();
    }
  };

  const ignoreFiles = (event: DragEvent<HTMLTextAreaElement>): void => {
    if (event.dataTransfer.types.includes('Files')) event.preventDefault();
  };

  const onFiles = props.onFiles;
  const [dragging, setDragging] = useState(false);

  const onPaste = (event: ClipboardEvent<HTMLTextAreaElement>): void => {
    if (onFiles === undefined) return;
    const data = event.clipboardData;
    if (data.types.includes('text/plain')) return;
    const files: File[] = [];
    for (const item of Array.from(data.items)) {
      if (item.kind !== 'file') continue;
      const file = item.getAsFile();
      if (file !== null) files.push(file);
    }
    if (files.length === 0) return;
    event.preventDefault();
    onFiles(files);
  };

  const onDragOver = (event: DragEvent<HTMLFormElement>): void => {
    if (onFiles === undefined || !event.dataTransfer.types.includes('Files')) return;
    event.preventDefault();
    setDragging(true);
  };

  const onDragLeave = (event: DragEvent<HTMLFormElement>): void => {
    if (event.currentTarget.contains(event.relatedTarget as Node | null)) return;
    setDragging(false);
  };

  const onDrop = (event: DragEvent<HTMLFormElement>): void => {
    if (onFiles === undefined || !event.dataTransfer.types.includes('Files')) return;
    event.preventDefault();
    setDragging(false);
    const files = Array.from(event.dataTransfer.files);
    if (files.length > 0) onFiles(files);
  };

  return (
    <form
      className={styles.root}
      aria-label={label}
      data-blocked={props.blocked !== undefined ? '' : undefined}
      data-dragging={dragging ? '' : undefined}
      onSubmit={submit}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      {menuOpen && props.commands !== undefined ? (
        <CommandMenu
          id={listId}
          rows={shownRows}
          active={active}
          optionId={optionId}
          later={props.commands.later}
          onPick={(index) => fillRow(index)}
          onHover={(index) => setPicked({ typed, index })}
        />
      ) : null}
      {props.notice !== undefined ? <NoticeStrip notice={props.notice} /> : null}
      {props.queue}
      {props.attachments}
      <div className={styles.field}>
        <textarea
          ref={attach}
          className={styles.input}
          rows={1}
          value={props.value}
          placeholder={props.placeholder}
          aria-label={label}
          aria-describedby={reasonShown || hintShown ? sideId : undefined}
          aria-autocomplete={props.commands === undefined ? undefined : 'list'}
          aria-expanded={props.commands === undefined ? undefined : menuOpen}
          aria-controls={menuOpen ? listId : undefined}
          aria-activedescendant={menuOpen ? optionId(active) : undefined}
          spellCheck
          onChange={(event) => {
            setCaret(event.target.selectionStart);
            props.onValueChange(event.target.value);
          }}
          onSelect={(event) => setCaret(event.currentTarget.selectionStart)}
          onFocus={(event) => {
            setFocused(true);
            setCaret(event.currentTarget.selectionStart);
          }}
          onBlur={() => setFocused(false)}
          onKeyDown={onKeyDown}
          onPaste={onPaste}
          onDragOver={ignoreFiles}
          onDrop={ignoreFiles}
        />
        {props.value === '' ? (
          <span className={styles.newLine}>
            <kbd className={styles.keys}>
              <ArrowBigUp size={12} strokeWidth={1.75} aria-hidden="true" />
              <span className={styles.srOnly}>{CHAT.keyNames.shift}</span>{' '}
              <CornerDownLeft size={12} strokeWidth={1.75} aria-hidden="true" />
              <span className={styles.srOnly}>{CHAT.keyNames.enter}</span>
            </kbd>{' '}
            <span>{CHAT.newLine}</span>
          </span>
        ) : null}
      </div>
      <div className={styles.tools}>
        <div ref={fit} className={styles.fit} data-side={side}>
          <div className={styles.toolsStart}>{props.tools}</div>
          {side === 'reason' ? (
            <span id={sideId} className={styles.reason}>
              {reason}
            </span>
          ) : side === 'hint' ? (
            <span id={sideId} className={styles.reason} data-hint="">
              {props.hint}
            </span>
          ) : side === 'count' ? (
            <span className={styles.count}>{counter}</span>
          ) : null}
        </div>
        <div ref={end} className={styles.end}>
          {props.context !== undefined && side === 'none' ? (
            <ContextRing
              ref={ring}
              context={props.context}
              onOpen={
                props.contextPanel === undefined
                  ? undefined
                  : () => props.contextPanel?.onOpenChange(!props.contextPanel.open)
              }
              expanded={props.contextPanel?.open === true}
            />
          ) : null}
          {props.contextPanel !== undefined ? (
            <ContextPopover
              open={props.contextPanel.open}
              onOpenChange={props.contextPanel.onOpenChange}
              ring={ring}
              fallback={end}
              focusFallback={() => area.current?.focus()}
            >
              {props.contextPanel.content}
            </ContextPopover>
          ) : null}
          <SendButton
            phase={props.phase}
            canSend={canSend}
            reason={blank ? undefined : reason}
            queueing={queueing}
            onSend={props.onSend}
            onStop={props.onStop}
          />
        </div>
      </div>
    </form>
  );
}
