import { GitFork, ScrollText } from 'lucide-react';
import type { ReactElement } from 'react';

import { CHAT, fill } from '../lib/chat-strings';
import { formatClock } from '../lib/format';
import type { TranscriptEntry } from '../lib/types';
import { visibleText } from '../lib/work-tools';
import { ButtonLink } from './arc-x/button-link/button-link';
import { EventRow } from './arc-x/event-row/event-row';
import styles from './history-mark.module.css';

/**
 * The first line of a fork's thread: "Fork of <title> · Open", the link going to the conversation it was copied from;
 * "Fork of a deleted conversation" with no link once that one is gone (`title: null`).
 */
export function ForkOrigin(props: { title: string | null; to: string }): ReactElement {
  return (
    <p className={styles.origin} data-fork-origin="">
      <span className={styles.text}>
        <GitFork size={12} strokeWidth={1.75} aria-hidden="true" />
        {props.title === null ? (
          <span className={styles.title}>{CHAT.history.forkOfGone}</span>
        ) : (
          <>
            <span className={styles.title}>{fill(CHAT.history.forkOf, { title: props.title })}</span>
            <span aria-hidden="true">·</span>
            <ButtonLink to={props.to} size="sm" variant="ghost">
              {CHAT.history.forkOpen}
            </ButtonLink>
          </>
        )}
      </span>
    </p>
  );
}

/**
 * A recap the user asked for: "Recap" and its time, then the model's words as plain text with hidden characters shown
 * as their codes. The same in the thread and on the Sessions page.
 */
export function RecapRow({ entry }: { entry: Extract<TranscriptEntry, { kind: 'recap' }> }): ReactElement {
  const name = entry.ts > 0 ? fill(CHAT.history.recapName, { time: formatClock(entry.ts) }) : CHAT.history.recap;
  return (
    <article aria-label={name} data-recap-row="">
      <EventRow
        icon={<ScrollText size={14} strokeWidth={1.75} />}
        label={CHAT.history.recap}
        text=""
        detail={visibleText(entry.text)}
        ts={entry.ts}
      />
    </article>
  );
}
