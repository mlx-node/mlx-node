import type { ReactElement } from 'react';

import { cn } from '../../../lib/cn';
import styles from './middle-text.module.css';

/** Characters kept whole at the end of a cut name. */
export const MIDDLE_TAIL = 8;
/** A name this short is never cut in the middle. */
export const MIDDLE_MIN = 12;

/**
 * `text` in one line: whole when it fits, else cut in the middle by the width it gets; the last characters stay. The
 * head and tail are hidden from screen readers, which read the name whole: two flex items are read with a space between.
 */
export function MiddleText({ text, className }: { text: string; className?: string }): ReactElement {
  if (text.length <= MIDDLE_MIN) {
    return (
      <span className={cn(styles.single, className)} title={text}>
        {text}
      </span>
    );
  }
  return (
    <span className={cn(styles.root, className)} title={text}>
      <span className={styles.head} aria-hidden="true">
        {text.slice(0, text.length - MIDDLE_TAIL)}
      </span>
      <span className={styles.tail} aria-hidden="true">
        {text.slice(text.length - MIDDLE_TAIL)}
      </span>
      <span className={styles.spoken}>{text}</span>
    </span>
  );
}
