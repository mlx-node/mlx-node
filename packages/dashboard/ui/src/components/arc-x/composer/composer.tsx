import { CircleAlert, Info, TriangleAlert } from 'lucide-react';
import { motion } from 'motion/react';
import {
  useCallback,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type DragEvent,
  type FormEvent,
  type KeyboardEvent,
  type ReactElement,
  type ReactNode,
  type Ref,
} from 'react';

import { CHAT, fill } from '../../../lib/chat-strings';
import { formatNumber } from '../../../lib/format';
import { hasNonWhitespace } from '../../../lib/text-scan';
import { Button } from '../../arc/button/button';
import { LiveLabel } from '../live-label/live-label';
import { motionX } from '../motion/motion';
import { useStaticMotion } from '../motion/use-static-motion';
import styles from './composer.module.css';
import { SendButton, type ComposerPhase } from './send-button';

export { SendButton, sendFace, type ComposerPhase, type SendButtonProps } from './send-button';

export interface ComposerNotice {
  tone: 'neutral' | 'warning' | 'danger';
  text: string;
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
  /** Why Send is off although there is text. Undefined: Send is on when `value` is not blank. */
  blocked?: string;
  notice?: ComposerNotice;
  /** Model picker and chips. */
  tools: ReactNode;
  /** Right-hand text of the tools row when there is no `blocked` text. */
  hint?: string;
  /** The hint shows only while the text field has focus. */
  hintOnFocus?: boolean;
  /** Default 100_000. */
  maxLength?: number;
  /** Accessible name of the textarea. Default 'Message'. */
  label?: string;
  textareaRef?: Ref<HTMLTextAreaElement>;
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

function setRef<T>(ref: Ref<T> | undefined, value: T | null): void {
  if (ref === undefined || ref === null) return;
  if (typeof ref === 'function') ref(value);
  else (ref as { current: T | null }).current = value;
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
      <div className={styles.notice} data-tone={notice.tone} role={notice.tone === 'danger' ? 'alert' : 'status'}>
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
  const tooLong = props.value.length > max;
  const canSend = composerCanSend(props.value, props.phase, props.blocked, max);
  const blank = !hasNonWhitespace(props.value);
  const reason = props.blocked ?? (tooLong ? CHAT.tooLong : undefined);
  const counter =
    props.value.length >= max * 0.9 && !tooLong
      ? fill(CHAT.counter, { length: formatNumber(props.value.length), max: formatNumber(max) })
      : null;
  const [focused, setFocused] = useState(false);
  const hint = props.hintOnFocus === true && !focused ? undefined : props.hint;
  const side = reason !== undefined && !blank ? reason : (counter ?? hint ?? '');
  const sideId = useId();
  const reasonShown = reason !== undefined && !blank;

  const attach = useCallback(
    (node: HTMLTextAreaElement | null) => {
      area.current = node;
      setRef(props.textareaRef, node);
    },
    [props.textareaRef],
  );

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
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>): void => {
    const native = event.nativeEvent;
    if (native.isComposing || native.keyCode === 229) return;
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      if (canSend) props.onSend();
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

  return (
    <form
      className={styles.root}
      aria-label={label}
      data-blocked={props.blocked !== undefined ? '' : undefined}
      onSubmit={submit}
    >
      {props.notice !== undefined ? <NoticeStrip notice={props.notice} /> : null}
      <textarea
        ref={attach}
        className={styles.input}
        rows={1}
        value={props.value}
        placeholder={props.placeholder}
        aria-label={label}
        aria-describedby={reasonShown ? sideId : undefined}
        spellCheck
        onChange={(event) => props.onValueChange(event.target.value)}
        onKeyDown={onKeyDown}
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
        onDragOver={ignoreFiles}
        onDrop={ignoreFiles}
      />
      <div className={styles.tools}>
        <div className={styles.toolsStart}>{props.tools}</div>
        <div className={styles.toolsEnd}>
          {side !== '' ? (
            <span id={sideId} className={styles.side} data-kind={reasonShown ? 'reason' : 'hint'}>
              {side}
            </span>
          ) : null}
          <SendButton
            phase={props.phase}
            canSend={canSend}
            reason={blank ? undefined : reason}
            onSend={props.onSend}
            onStop={props.onStop}
          />
        </div>
      </div>
    </form>
  );
}
