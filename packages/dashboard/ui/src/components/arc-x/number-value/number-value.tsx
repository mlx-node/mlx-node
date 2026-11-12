import { useState, type CSSProperties, type ReactElement } from 'react';

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
 * A measure that counts up once from zero when first on screen and rolls its digits when it changes. `—`, null and
 * compound text never get wheels; the visually hidden text is the final value from the first frame.
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
        className={styles.wheels}
        data-prev={next.prev ?? undefined}
        onAnimationEnd={(event) => {
          if (event.target !== event.currentTarget || event.pseudoElement !== '::before') return;
          const key = next.key;
          setTrack((current) => (current.key === key && current.prev !== null ? { ...current, prev: null } : current));
        }}
      >
        <AnimatedCounter
          value={parsed.value}
          decimals={parsed.decimals}
          prefix={parsed.prefix}
          suffix={parsed.suffix}
          animateOnView={next.first ? countUp : false}
        />
      </span>
    </span>
  );
}
