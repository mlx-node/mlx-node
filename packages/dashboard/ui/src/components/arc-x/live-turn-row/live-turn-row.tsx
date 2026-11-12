import { AnimatePresence, motion } from 'motion/react';
import { Children, isValidElement, type ReactElement, type ReactNode } from 'react';

import { formatCount, formatCountNoun, formatNumber, formatPercent, NO_VALUE } from '../../../lib/format';
import { Badge } from '../../arc/badge/badge';
import { LiveLabel } from '../live-label/live-label';
import { ModelBadge } from '../model-badge/model-badge';
import { motionX } from '../motion/motion';
import { MotionGate, useRichMotion } from '../motion/use-entrance';
import { NumberValue } from '../number-value/number-value';
import { Meter } from '../split-bar/split-bar';
import { SwapText } from '../swap-text/swap-text';
import styles from './live-turn-row.module.css';

export type LivePhase = 'queued' | 'prefill' | 'decode' | null;

export interface LiveTurnRowProps {
  /** Callers pass `turn.modelId ?? turn.model`: the checkpoint, else the alias the client used. */
  model: string;
  /** null = not streaming, no progress known. */
  phase: LivePhase;
  /** now − startedAt, from the snapshot clock. */
  elapsedMs: number;
  prefillDone: number | null;
  prefillTotal: number | null;
  generatedTokens: number | null;
  promptTokens: number | null;
  /** Callers pass `turn.ctxWindow`. It is never looked up by model name: an alias matches no `models[].name`. */
  ctxWindow: number | null;
  /** The name the client asked for, when it differs from `model`: 11 px muted text after the badge. */
  alias?: string;
  /** "start" | "continue" | "continue_tool" -> small muted word after the model. */
  dispatchLabel?: string;
  /** 44 px | 36 px; compact hides the context meter below 480 px. */
  density?: 'default' | 'compact';
  /** False when the snapshot is stale (the process stopped answering): no shimmer, no shine. Default true. */
  live?: boolean;
}

const RICH_ROWS = 8;

const PHASE_WORD: Record<'queued' | 'prefill' | 'decode' | 'running', string> = {
  queued: 'Queued',
  prefill: 'Prefill',
  decode: 'Decode',
  running: 'Running',
};

/** `m:ss`, or `h:mm:ss` from one hour. */
export function formatElapsed(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = String(total % 60).padStart(2, '0');
  return hours > 0
    ? `${String(hours)}:${String(minutes).padStart(2, '0')}:${seconds}`
    : `${String(minutes)}:${seconds}`;
}

function spokenElapsed(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  const secondText = `${String(seconds)} ${seconds === 1 ? 'second' : 'seconds'}`;
  if (minutes === 0) return secondText;
  return `${String(minutes)} ${minutes === 1 ? 'minute' : 'minutes'} ${secondText}`;
}

/** The middle cell's words, also used in the row's accessible name. */
export function progressText(props: LiveTurnRowProps): string {
  switch (props.phase) {
    case 'prefill':
      return props.prefillDone === null || props.prefillTotal === null
        ? 'prefill progress not reported'
        : `${formatCount(props.prefillDone)} of ${formatCount(props.prefillTotal)} tokens`;
    case 'decode':
      return `${formatCountNoun(props.generatedTokens, 'token', 'tokens')} out`;
    case 'queued':
      return 'waiting for a slot';
    default:
      return 'no progress for non-streamed turns';
  }
}

function contextShare(props: LiveTurnRowProps): number | null {
  if (props.promptTokens === null || props.ctxWindow === null || props.ctxWindow <= 0) return null;
  return (props.promptTokens + (props.generatedTokens ?? 0)) / props.ctxWindow;
}

/** One turn that is running now: model, phase word, progress, elapsed time, context fill. */
export function LiveTurnRow(props: LiveTurnRowProps): ReactElement {
  const rich = useRichMotion();
  const live = props.live !== false;
  const phaseKey = props.phase ?? 'running';
  const word = PHASE_WORD[phaseKey];
  const working = live && (props.phase === 'prefill' || props.phase === 'decode');
  const share = contextShare(props);
  const progress = progressText(props);
  const name = [
    props.model,
    word.toLowerCase(),
    progress,
    spokenElapsed(props.elapsedMs),
    `context ${share === null ? 'unknown' : formatPercent(share)}`,
  ].join(', ');

  let middle: ReactNode;
  if (props.phase === 'prefill' && props.prefillDone !== null && props.prefillTotal !== null) {
    middle = (
      <>
        <Meter
          label={`${props.model} prefill`}
          value={props.prefillDone}
          max={props.prefillTotal}
          size="dense"
          warnAt={2}
          dangerAt={2}
          busy={live && rich}
          className={styles.meter}
        />
        <span className={styles.progressText} data-full="">
          <SwapText text={progress} granularity="char" />
        </span>
        <span className={styles.progressText} data-short="" title={progress}>
          <SwapText
            text={`${formatCount(props.prefillDone)} / ${formatCount(props.prefillTotal)}`}
            granularity="char"
          />
        </span>
      </>
    );
  } else if (props.phase === 'decode') {
    middle = (
      <span className={styles.progressText}>
        <NumberValue text={formatNumber(props.generatedTokens)} countUp={false} />{' '}
        {props.generatedTokens === 1 ? 'token' : 'tokens'} out
      </span>
    );
  } else {
    middle = <span className={styles.progressMuted}>{progress}</span>;
  }

  const content = (
    <>
      <span className={styles.model}>
        <ModelBadge name={props.model} size="sm" className={styles.modelName} />
        {props.alias !== undefined ? (
          <span className={styles.alias} title={props.alias}>
            {props.alias}
          </span>
        ) : null}
        {props.dispatchLabel !== undefined ? (
          <span className={styles.dispatch} title={props.dispatchLabel}>
            {props.dispatchLabel}
          </span>
        ) : null}
      </span>
      <span className={styles.phase}>
        <Badge
          tone={props.phase === 'prefill' || props.phase === 'decode' ? 'info' : 'neutral'}
          outline={props.phase === 'queued'}
          size="sm"
        >
          <span className={styles.key} data-phase={phaseKey} aria-hidden="true" />
          <LiveLabel active={working}>{word}</LiveLabel>
        </Badge>
      </span>
      <span className={styles.progress}>{middle}</span>
      <span className={styles.elapsed}>
        <SwapText text={formatElapsed(props.elapsedMs)} reserve="5ch" granularity="char" />
      </span>
      <span className={styles.context}>
        {share === null ? (
          <span className={styles.progressMuted}>ctx {NO_VALUE}</span>
        ) : (
          <>
            <Meter label={`${props.model} context`} value={share} max={1} size="dense" className={styles.ctxMeter} />
            <span className={styles.ctxText}>{formatPercent(share)}</span>
          </>
        )}
      </span>
    </>
  );

  const shared = {
    className: styles.root,
    'aria-label': name,
    'data-density': props.density ?? 'default',
    'data-phase': phaseKey,
  };
  if (rich) {
    return (
      <motion.li
        {...shared}
        initial={{ height: 0, opacity: 0 }}
        animate={{ height: 'auto', opacity: 1 }}
        exit={{
          opacity: 0,
          height: 0,
          transition: {
            opacity: { duration: motionX.swap.out.duration },
            height: { ...motionX.spring.expand, delay: motionX.swap.out.duration },
          },
        }}
        transition={motionX.spring.expand}
      >
        <span className={styles.inner}>{content}</span>
      </motion.li>
    );
  }
  return (
    <li {...shared}>
      <span className={styles.inner}>{content}</span>
    </li>
  );
}

/** <ul aria-label>. Children 9+ sit in a closed MotionGate: the same words and numbers without shimmer or wheels. */
export function LiveTurnList(props: { label: string; children: ReactNode }): ReactElement {
  const items = Children.toArray(props.children);
  return (
    <ul className={styles.list} aria-label={props.label}>
      <AnimatePresence initial={false}>
        {items.map((child, index) => (
          <MotionGate key={isValidElement(child) && child.key !== null ? child.key : index} open={index < RICH_ROWS}>
            {child}
          </MotionGate>
        ))}
      </AnimatePresence>
    </ul>
  );
}
