import type { ReactElement, ReactNode } from 'react';

import { cn } from '../../../lib/cn';
import { formatClock } from '../../../lib/format';
import { LiveLabel } from '../live-label/live-label';
import styles from './message.module.css';

export interface MessageProps {
  role: 'user' | 'assistant';
  /** "You", "You · interjection", or a ModelBadge. Not drawn by the chat variant. */
  label?: ReactNode;
  ts?: number;
  children: ReactNode;
  /** assistant only: the message has no end yet and the body is fresh. Default false. */
  streaming?: boolean;
  /** 'transcript' (default): label line and a 72ch body. 'chat': a right bubble for the user, full-width prose. */
  variant?: 'transcript' | 'chat';
  /** The live word while streaming. Default "Writing". */
  status?: string;
  /** Under the body: actions that show on hover or focus. */
  footer?: ReactNode;
  /** It arrived live (not from history): it rises in once. */
  enter?: boolean;
  className?: string;
}

/** One message of a transcript: a label line, then the body. A user message sits on a muted surface; an answer is plain prose. */
export function Message(props: MessageProps): ReactElement {
  const streaming = props.role === 'assistant' && props.streaming === true;
  const chat = props.variant === 'chat';
  const timed = props.ts !== undefined && props.ts > 0;
  const clock = timed ? formatClock(props.ts) : null;
  const roleWord = props.role === 'user' ? 'You' : 'Assistant';
  const status = props.status ?? 'Writing';
  const header = chat ? (
    streaming ? (
      <div className={styles.header}>
        <LiveLabel active className={styles.writing}>
          {status}
        </LiveLabel>
      </div>
    ) : null
  ) : (
    <div className={styles.header}>
      <span className={styles.label}>{props.label}</span>
      {streaming ? (
        <LiveLabel active className={styles.writing}>
          {status}
        </LiveLabel>
      ) : null}
      {clock !== null && props.ts !== undefined ? (
        <time
          className={styles.time}
          dateTime={new Date(props.ts).toISOString()}
          title={new Date(props.ts).toISOString()}
        >
          {clock}
        </time>
      ) : null}
    </div>
  );
  return (
    <article
      className={cn(styles.root, props.className)}
      data-role={props.role}
      data-variant={chat ? 'chat' : undefined}
      data-live={streaming ? '' : undefined}
      data-enter={props.enter === true ? '' : undefined}
      aria-label={clock !== null ? `${roleWord}, ${clock}` : roleWord}
      aria-busy={chat && streaming ? true : undefined}
      title={chat && clock !== null && props.ts !== undefined ? new Date(props.ts).toISOString() : undefined}
    >
      {header}
      <div className={styles.body}>{props.children}</div>
      {props.footer !== undefined ? <div className={styles.footer}>{props.footer}</div> : null}
    </article>
  );
}
