import { ChevronRight } from 'lucide-react';
import { motion } from 'motion/react';
import { useId, useLayoutEffect, useRef, useState, type ReactElement } from 'react';

import { cn } from '../../../lib/cn';
import { formatDuration } from '../../../lib/format';
import { LiveLabel } from '../live-label/live-label';
import { motionX } from '../motion/motion';
import { useRichMotion } from '../motion/use-entrance';
import { useStaticMotion } from '../motion/use-static-motion';
import { SwapText } from '../swap-text/swap-text';
import styles from './reasoning-block.module.css';

export interface ReasoningBlockProps {
  text: string;
  durationMs?: number | null;
  /** Default 'Thinking'. */
  label?: string;
  defaultOpen?: boolean;
  /** The thinking has no end yet and the body is fresh. Default false. */
  live?: boolean;
  /** With live: when the thinking started and when the body arrived; the header shows `for m:ss` = asOfMs − startTs. */
  startTs?: number;
  asOfMs?: number;
  /** With live: length of `text` already shown before this render; the rest fades in. Default text.length. */
  freshFrom?: number;
  className?: string;
}

const INSTANT = { duration: 0 } as const;
const TAIL_PX = 40;
/** The live tail shows the last two lines: it draws only the end of the text, cut at a line start. */
const TAIL_CHARS = 1_200;
const TAIL_STEP = 4_000;

/** Where the drawn tail starts: a line start near the end, else a fixed step so the wrap does not move each paint. */
function tailStart(text: string): number {
  if (text.length <= TAIL_CHARS) return 0;
  const line = text.lastIndexOf('\n', text.length - TAIL_CHARS);
  if (line !== -1 && text.length - line <= TAIL_STEP) return line + 1;
  return Math.max(0, Math.floor((text.length - TAIL_CHARS) / TAIL_STEP) * TAIL_STEP);
}

/** `0:07`, `12:30`; minutes are not capped. */
export function elapsedClock(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
}

/** The text with the part from `freshFrom` on in a span that fades in. */
function FreshText({ text, freshFrom }: { text: string; freshFrom: number | undefined }): ReactElement {
  const cut = freshFrom === undefined ? text.length : Math.min(Math.max(freshFrom, 0), text.length);
  if (cut >= text.length) return <>{text}</>;
  return (
    <>
      {text.slice(0, cut)}
      <span key={cut} className={styles.fresh} data-fresh="">
        {text.slice(cut)}
      </span>
    </>
  );
}

/** Recorded reasoning: a dashed card that opens to the text. While live: shimmering label, time as of the body, a feathered live tail. */
export function ReasoningBlock(props: ReasoningBlockProps): ReactElement {
  const staticMotion = useStaticMotion();
  const rich = useRichMotion();
  const bodyId = useId();
  const [open, setOpen] = useState(props.defaultOpen === true);
  const [bodyMounted, setBodyMounted] = useState(open);
  if (open && !bodyMounted) setBodyMounted(true);
  if (!open && bodyMounted && staticMotion) setBodyMounted(false);
  const live = props.live === true;
  const label = props.label ?? 'Thinking';
  const liveTime =
    live && props.startTs !== undefined && props.asOfMs !== undefined && props.startTs > 0
      ? elapsedClock(props.asOfMs - props.startTs)
      : null;
  const doneTime =
    !live && props.durationMs !== undefined && props.durationMs !== null ? formatDuration(props.durationMs) : null;
  const time = liveTime ?? doneTime;
  const tail = live && rich && !open;
  const fresh = rich && live ? props.freshFrom : undefined;
  const tailRef = useRef<HTMLDivElement | null>(null);
  const tailFrom = tail ? tailStart(props.text) : 0;
  const tailHeight = useRef<number | null>(null);

  useLayoutEffect(() => {
    const node = tailRef.current;
    if (node === null) {
      tailHeight.current = null;
      return;
    }
    const height = node.getBoundingClientRect().height;
    const previous = tailHeight.current;
    tailHeight.current = height;
    if (previous === null || height <= previous) return;
    node.style.transition = 'none';
    node.style.transform = `translateY(${height - previous}px)`;
    void node.offsetHeight;
    node.style.transition = '';
    node.style.transform = '';
  }, [props.text, tail]);

  return (
    <div
      className={cn(styles.root, props.className)}
      data-state={open ? 'open' : 'closed'}
      data-live={live ? '' : undefined}
    >
      <button
        type="button"
        className={styles.header}
        aria-expanded={open}
        aria-controls={bodyId}
        onClick={() => setOpen((v) => !v)}
      >
        <ChevronRight className={styles.chevron} size={14} strokeWidth={1.75} aria-hidden="true" />
        <span className={styles.label}>
          <LiveLabel active={live}>{label}</LiveLabel>
        </span>
        <span className={styles.meta}>
          {time !== null ? (
            <>
              {' for '}
              {liveTime !== null ? <SwapText text={liveTime} granularity="char" reserve="4ch" /> : time}
            </>
          ) : null}
        </span>
      </button>
      <motion.div
        className={styles.tailClip}
        aria-hidden="true"
        initial={false}
        animate={{ height: tail ? TAIL_PX : 0 }}
        transition={staticMotion ? INSTANT : motionX.spring.expand}
      >
        {tail ? (
          <div className={styles.tail}>
            <div ref={tailRef} className={styles.tailText}>
              <FreshText
                text={props.text.slice(tailFrom)}
                freshFrom={fresh === undefined ? undefined : fresh - tailFrom}
              />
            </div>
          </div>
        ) : null}
      </motion.div>
      <motion.div
        id={bodyId}
        className={styles.bodyClip}
        initial={false}
        animate={{ height: open ? 'auto' : 0 }}
        transition={staticMotion ? INSTANT : motionX.spring.expand}
        hidden={!open && !bodyMounted}
        onAnimationComplete={() => {
          if (!open) setBodyMounted(false);
        }}
      >
        {bodyMounted ? (
          <div className={styles.body} data-enter={staticMotion || !open ? undefined : ''}>
            <FreshText text={props.text} freshFrom={fresh} />
          </div>
        ) : null}
      </motion.div>
    </div>
  );
}
