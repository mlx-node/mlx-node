import { useEffect, useState, type ReactElement } from 'react';

import { CHAT, fill } from '../../../lib/chat-strings';
import { formatElapsed, type LoadVariant } from '../../../lib/chat-turns';
import { formatBytes } from '../../../lib/format';
import { LiveLabel } from '../live-label/live-label';
import { ModelBadge } from '../model-badge/model-badge';
import { MorphLoader } from '../morph-loader/morph-loader';
import { SwapText } from '../swap-text/swap-text';
import styles from './load-card.module.css';

export interface LoadCardProps {
  /** The model the message waits for. */
  modelId: string;
  variant: LoadVariant;
  /** `other-first`: the model the engine loads first. */
  other?: string | null;
  /** Epoch ms the clock counts from. */
  since: number;
  /** Size on disk; null leaves the size out. */
  sizeBytes: number | null;
  /** The clock. Default `Date.now`. */
  now?: () => number;
}

/** A message that waits for its model: what the engine does, for how long, and what Stop does. No bar: no measure follows a load. */
export function LoadCard(props: LoadCardProps): ReactElement {
  const clock = props.now ?? Date.now;
  const [at, setAt] = useState(clock);
  useEffect(() => {
    const timer = setInterval(() => setAt(clock()), 1000);
    return () => clearInterval(timer);
  }, [clock]);
  const title = props.variant === 'loading' ? CHAT.load.title : CHAT.load.titleQueued;
  const line =
    props.variant === 'loading'
      ? CHAT.load.wait
      : props.variant === 'other-first'
        ? fill(CHAT.load.otherFirst, { other: props.other ?? '' })
        : props.variant === 'in-memory'
          ? CHAT.load.inMemory
          : CHAT.load.busyFirst;
  return (
    <div className={styles.root} data-load-card="" data-variant={props.variant}>
      <div className={styles.head}>
        <MorphLoader size={14} status="loading" />
        <LiveLabel active className={styles.title}>
          {title}
        </LiveLabel>
        <span className={styles.clock} aria-hidden="true">
          <SwapText text={formatElapsed(at - props.since)} granularity="char" />
        </span>
      </div>
      <div className={styles.model}>
        <span className={styles.name}>
          <ModelBadge name={props.modelId} fit="width" />
        </span>
        {props.sizeBytes !== null ? (
          <span className={styles.size}>{fill(CHAT.load.size, { size: formatBytes(props.sizeBytes) })}</span>
        ) : null}
      </div>
      <p className={styles.line}>{line}</p>
      <p className={styles.line}>{props.variant === 'in-memory' ? CHAT.load.stopNoteInMemory : CHAT.load.stopNote}</p>
    </div>
  );
}
