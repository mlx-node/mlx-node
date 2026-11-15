import { useLayoutEffect, useRef, useState, type CSSProperties, type ReactElement } from 'react';

import { cn } from '../../../lib/cn';
import { NO_VALUE } from '../../../lib/format';
import { AnimatedCounter } from '../../arc/animated-counter/animated-counter';
import { useRichMotion } from '../motion/use-entrance';
import { useStaticMotion } from '../motion/use-static-motion';
import { SwapText } from '../swap-text/swap-text';
import styles from './number-value.module.css';
import { parseNumeric } from './parse-numeric';

export interface NumberValueProps {
  /** A formatted value from lib/format.ts, or null when nothing was measured. */
  text: string | null;
  /** Shown after the value by the caller's layout; part of the shape key, never animated. */
  unit?: string;
  /** Count up from zero the first time it is on screen. Default true. False: appear at the value, roll on change. */
  countUp?: boolean;
  /** min-width, e.g. '6ch', for a place whose neighbours must not move. */
  reserve?: string;
  className?: string;
}

interface Track {
  shape: string | null;
  text: string | null;
  /** Generation of the wheel wrapper; a new shape mounts a new one. */
  key: number;
  /** The text the current wrapper replaced; drawn leaving by ::before. */
  prev: string | null;
  /** The current wrapper is the first number after nothing was measured: it may count up. */
  first: boolean;
}

/** AnimatedCounter's own `useInView` amount: the share of the number that must be on screen for its count-up. */
const IN_VIEW_AMOUNT = 0.6;

const CLIPPING = ['hidden', 'clip', 'scroll', 'auto', 'overlay'];

/** The share of an element's box inside the viewport and inside every ancestor that clips it; 0 for an empty box. */
export function shownShare(element: Element): number {
  const box = element.getBoundingClientRect();
  const area = box.width * box.height;
  if (area <= 0) return 0;
  let left = Math.max(box.left, 0);
  let top = Math.max(box.top, 0);
  let right = Math.min(box.right, window.innerWidth);
  let bottom = Math.min(box.bottom, window.innerHeight);
  for (let node = element.parentElement; node !== null; node = node.parentElement) {
    const style = getComputedStyle(node);
    if (!CLIPPING.includes(style.overflowX) && !CLIPPING.includes(style.overflowY)) continue;
    const clip = node.getBoundingClientRect();
    left = Math.max(left, clip.left);
    top = Math.max(top, clip.top);
    right = Math.min(right, clip.right);
    bottom = Math.min(bottom, clip.bottom);
  }
  return (Math.max(0, right - left) * Math.max(0, bottom - top)) / area;
}

function isMeasured(text: string | null): text is string {
  return text !== null && text !== NO_VALUE;
}

function nextTrack(track: Track, shape: string | null, text: string | null): Track {
  if (shape === track.shape) return track.text === text ? track : { ...track, text };
  if (shape === null) return { shape: null, text, key: track.key, prev: null, first: !isMeasured(text) };
  if (!isMeasured(track.text)) return { shape, text, key: track.key + 1, prev: null, first: true };
  return { shape, text, key: track.key + 1, prev: track.text, first: false };
}

/**
 * A measure that counts up once from zero and rolls its digits when it changes. It counts up only when it is on screen
 * as its first value shows: off screen it appears at its value, so no zeros wait for a scroll. `—`, null and compound
 * text never get wheels; the visually hidden text is the final value from the first frame.
 */
export function NumberValue({ text, unit, countUp = true, reserve, className }: NumberValueProps): ReactElement {
  const staticMotion = useStaticMotion();
  const rich = useRichMotion();
  const measured = isMeasured(text);
  const parsed = measured ? parseNumeric(text) : null;
  const shape = parsed === null ? null : `${parsed.prefix}|${String(parsed.decimals)}|${parsed.suffix}|${unit ?? ''}`;

  const [track, setTrack] = useState<Track>(() => ({ shape, text, key: 0, prev: null, first: true }));
  const next = nextTrack(track, shape, text);
  if (next !== track) setTrack(next);

  const wheels = useRef<HTMLSpanElement | null>(null);
  /** The wheels with this key appear at their value: they were off screen when it first showed. */
  const [still, setStill] = useState<number | null>(null);
  const counting = next.first && countUp && still !== next.key;
  // Wheels can first mount without a new key, when motion turns on after the value showed.
  const drawn = measured && !staticMotion && rich && parsed !== null;
  useLayoutEffect(() => {
    if (!counting || !drawn || wheels.current === null) return;
    if (shownShare(wheels.current) < IN_VIEW_AMOUNT) setStill(next.key);
  }, [counting, drawn, next.key]);

  const style: CSSProperties | undefined = reserve === undefined ? undefined : { minWidth: reserve };
  const rootClass = cn(styles.root, className);

  if (!measured) {
    return (
      <span className={rootClass} style={style}>
        {NO_VALUE}
      </span>
    );
  }
  if (staticMotion) {
    return (
      <span className={rootClass} style={style}>
        {text}
      </span>
    );
  }
  if (!rich || parsed === null) {
    return (
      <span className={rootClass} style={style}>
        <SwapText text={text} granularity="char" reserve={reserve} />
      </span>
    );
  }
  return (
    <span className={rootClass} style={style}>
      <span
        key={next.key}
        ref={wheels}
        className={styles.wheels}
        data-prev={next.prev ?? undefined}
        onAnimationEnd={(event) => {
          if (event.target !== event.currentTarget || event.pseudoElement !== '::before') return;
          const key = next.key;
          setTrack((current) => (current.key === key && current.prev !== null ? { ...current, prev: null } : current));
        }}
      >
        <AnimatedCounter
          key={counting ? 'count' : 'still'}
          value={parsed.value}
          decimals={parsed.decimals}
          prefix={parsed.prefix}
          suffix={parsed.suffix}
          animateOnView={counting}
        />
      </span>
    </span>
  );
}
