import type { ReactElement, ReactNode } from 'react';

import { formatClock } from '../../../lib/format';
import styles from './thread-mark.module.css';

export interface ThreadMarkProps {
  /** `data-thread-mark`: "mode", "settings". */
  kind: string;
  /** 12 px. */
  icon: ReactNode;
  /** "Mode: Auto". */
  label: string;
  /** One short sentence; it never wraps. */
  fact: string;
  /** The second line, centered under the rule. */
  detail?: ReactNode;
  tone?: 'neutral' | 'warning' | 'danger';
  ts?: number;
  /** Extra data attributes on the root, e.g. `{ 'data-mode': 'auto' }`. */
  data?: Record<`data-${string}`, string>;
}

/**
 * A moment of the thread's history ("from here on ..."): a hairline across the thread with the label, one fact and the
 * time centered in it, and an optional line under it. Read as plain text; no tab stop.
 */
export function ThreadMark(props: ThreadMarkProps): ReactElement {
  const timed = props.ts !== undefined && props.ts > 0;
  const iso = timed ? new Date(props.ts!).toISOString() : undefined;
  return (
    <div
      className={styles.mark}
      role="note"
      data-thread-mark={props.kind}
      data-tone={props.tone ?? 'neutral'}
      {...props.data}
    >
      <span className={styles.line}>
        <span className={styles.icon} aria-hidden="true">
          {props.icon}
        </span>
        <span className={styles.label}>{props.label}</span>
        <span className={styles.srOnly}>. </span>
        <span className={styles.fact} title={props.fact}>
          {props.fact}
        </span>
        {timed ? (
          <>
            <span className={styles.srOnly}> </span>
            <time className={styles.time} dateTime={iso} title={iso}>
              {formatClock(props.ts)}
            </time>
          </>
        ) : null}
      </span>
      {props.detail !== undefined ? <p className={styles.detail}>{props.detail}</p> : null}
    </div>
  );
}
