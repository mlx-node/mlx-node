import { useEffect, useState, type ReactElement } from 'react';

import { CHAT, fill } from '../../../lib/chat-strings';
import { formatElapsed, type LoadVariant } from '../../../lib/chat-turns';
import { formatBytes, formatNumber } from '../../../lib/format';
import { LiveLabel } from '../live-label/live-label';
import { ModelBadge } from '../model-badge/model-badge';
import { Meter } from '../split-bar/split-bar';
import { SwapText } from '../swap-text/swap-text';
import { WaitOrb } from '../wait-orb/wait-orb';
import styles from './load-card.module.css';

/** What the engine reports about the load: `kernels` with `steps` known is the GPU kernel warm-up. */
export interface LoadProgressView {
  phase: 'weights' | 'kernels';
  stepsDone: number | null;
  steps: number | null;
}

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
  /** `loading`: the load's progress; null or `weights` shows the plain card. */
  progress?: LoadProgressView | null;
  /** The clock. Default `Date.now`. */
  now?: () => number;
}

/** The kernel steps of a `loading` card, or null when the engine reports none. */
export function kernelSteps(
  variant: LoadVariant,
  progress: LoadProgressView | null | undefined,
): { done: number; steps: number } | null {
  if (variant !== 'loading' || progress == null || progress.phase !== 'kernels') return null;
  const { stepsDone, steps } = progress;
  if (steps === null || stepsDone === null || steps <= 0) return null;
  return { done: Math.min(Math.max(stepsDone, 0), steps), steps };
}

/**
 * A message that waits for its model: what the engine does, for how long, and what Stop does. A bar only while the
 * engine prepares GPU kernels: the weight read reports no measure.
 */
export function LoadCard(props: LoadCardProps): ReactElement {
  const clock = props.now ?? Date.now;
  const [at, setAt] = useState(clock);
  useEffect(() => {
    const timer = setInterval(() => setAt(clock()), 1000);
    return () => clearInterval(timer);
  }, [clock]);
  const kernels = kernelSteps(props.variant, props.progress);
  const title =
    props.variant !== 'loading' ? CHAT.load.titleQueued : kernels !== null ? CHAT.load.titleKernels : CHAT.load.title;
  const line =
    props.variant === 'loading'
      ? kernels !== null
        ? CHAT.load.kernelNote
        : CHAT.load.wait
      : props.variant === 'other-first'
        ? fill(CHAT.load.otherFirst, { other: props.other ?? '' })
        : props.variant === 'in-memory'
          ? CHAT.load.inMemory
          : CHAT.load.busyFirst;
  const stepsText =
    kernels === null
      ? null
      : fill(CHAT.load.kernelSteps, { done: formatNumber(kernels.done), steps: formatNumber(kernels.steps) });
  return (
    <div
      className={styles.root}
      data-load-card=""
      data-variant={props.variant}
      data-phase={kernels === null ? undefined : 'kernels'}
    >
      <div className={styles.head}>
        <WaitOrb state={props.variant === 'loading' ? 'load' : 'hold'} />
        <LiveLabel active className={styles.title}>
          {title}
        </LiveLabel>
        {stepsText === null ? null : (
          <span className={styles.clock}>
            <SwapText text={stepsText} granularity="char" />
          </span>
        )}
        <span className={styles.clock} aria-hidden="true">
          <SwapText text={formatElapsed(at - props.since)} granularity="char" />
        </span>
      </div>
      {kernels === null ? null : (
        <Meter
          label={CHAT.load.titleKernels}
          value={kernels.done}
          max={kernels.steps}
          formatValue={() => stepsText ?? ''}
          size="dense"
          warnAt={2}
          dangerAt={2}
          busy
        />
      )}
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
