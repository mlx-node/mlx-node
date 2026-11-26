import { AnimatePresence, motion } from 'motion/react';
import { Children, isValidElement, type ReactElement, type ReactNode } from 'react';

import { fill } from '../../../lib/chat-strings';
import { formatCount, formatCountNoun, formatNumber, formatPercent, formatTps, NO_VALUE } from '../../../lib/format';
import {
  BANDS,
  ENGINE_STATE,
  IDLE_LANE,
  LANES,
  METRIC,
  PREFILL_PROGRESS,
  WAIT_REASON,
} from '../../../lib/metric-strings';
import { PREFILL_MIN_TOKENS } from '../../../lib/prefill';
import { Badge } from '../../arc/badge/badge';
import { withUnit } from '../chart/value-format';
import { LiveLabel } from '../live-label/live-label';
import { ModelBadge } from '../model-badge/model-badge';
import { motionX } from '../motion/motion';
import { MotionGate, useRichMotion } from '../motion/use-entrance';
import { NumberValue } from '../number-value/number-value';
import { Meter } from '../split-bar/split-bar';
import { SwapText } from '../swap-text/swap-text';
import styles from './live-turn-row.module.css';

/** `idle`: the session's last call ended and its next has not come yet (`lib/live-lanes.ts`). */
export type LivePhase = 'queued' | 'slot' | 'prepare' | 'prefill' | 'decode' | 'idle' | null;

export interface LiveTurnRowProps {
  /** Callers pass `turn.modelId ?? turn.model`: the checkpoint, else the alias the client used. */
  model: string;
  /** null = not streaming, no progress known. */
  phase: LivePhase;
  /** now − startedAt, from the snapshot clock; for an idle lane, the time since its last call ended. */
  elapsedMs: number;
  prefillDone: number | null;
  prefillTotal: number | null;
  generatedTokens: number | null;
  promptTokens: number | null;
  /** Prefix tokens reused from the cache: the `prepare` phase restores them. */
  cachedTokens: number | null;
  /** Callers pass `turn.ctxWindow`. It is never looked up by model name: an alias matches no `models[].name`. */
  ctxWindow: number | null;
  /** ms the lane waited for the busy model (native `queued`), shown after "model busy". */
  waitMs?: number | null;
  /** ms the lane waited for a free lane (native `slot`), shown after "no free lane". */
  slotMs?: number | null;
  /** ms of this lane's prefill so far: the time of a one-shot prefill and the lane's prefill speed. */
  prefillElapsedMs?: number | null;
  /** ms since this lane's first token: the lane's decode speed. */
  decodeElapsedMs?: number | null;
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

/** A lane's decode speed needs this much decode time behind it. */
const DECODE_SPEED_MIN_MS = 250;

/** Both native waits read "Waiting"; the row text says what for. */
const PHASE_WORD: Record<NonNullable<LivePhase> | 'running', string> = {
  queued: METRIC.waiting.label,
  slot: METRIC.waiting.label,
  prepare: METRIC.preparing.label,
  prefill: BANDS.prefill,
  decode: BANDS.decode,
  idle: METRIC.idle.label,
  running: METRIC.running.label,
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

function known(value: number | null | undefined): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function waitText(reason: string, ms: number | null | undefined): string {
  return known(ms) ? `${reason} · ${formatElapsed(ms)}` : reason;
}

type ProgressFields = Pick<
  LiveTurnRowProps,
  | 'phase'
  | 'prefillDone'
  | 'prefillTotal'
  | 'generatedTokens'
  | 'cachedTokens'
  | 'waitMs'
  | 'slotMs'
  | 'prefillElapsedMs'
>;

/** The middle cell's words, also used in the row's accessible name. A prefill that counted nothing yet has no "0 of". */
export function progressText(props: ProgressFields): string {
  switch (props.phase) {
    case 'prefill':
      if (props.prefillDone === null || props.prefillTotal === null) return 'prefill progress not reported';
      if (props.prefillDone === 0) {
        return known(props.prefillElapsedMs)
          ? fill(PREFILL_PROGRESS.tokensElapsed, {
              total: formatCount(props.prefillTotal),
              elapsed: formatElapsed(props.prefillElapsedMs),
            })
          : formatCountNoun(props.prefillTotal, 'token', 'tokens', formatCount);
      }
      return fill(PREFILL_PROGRESS.tokensOf, {
        done: formatCount(props.prefillDone),
        total: formatCount(props.prefillTotal),
      });
    case 'decode':
      return `${formatCountNoun(props.generatedTokens, 'token', 'tokens')} out`;
    case 'queued':
      return waitText(WAIT_REASON.modelBusy, props.waitMs);
    case 'slot':
      return waitText(WAIT_REASON.noFreeLane, props.slotMs);
    case 'prepare':
      return props.cachedTokens !== null && props.cachedTokens > 0
        ? `restoring ${formatCountNoun(props.cachedTokens, 'cached token', 'cached tokens')}`
        : 'reading the prompt';
    case 'idle':
      return props.generatedTokens === null
        ? IDLE_LANE.lastCallEnded
        : fill(IDLE_LANE.lastCallOut, { tokens: formatCountNoun(props.generatedTokens, 'token', 'tokens') });
    default:
      return 'no progress for non-streamed turns';
  }
}

export type LaneSpeedFields = Pick<
  LiveTurnRowProps,
  'phase' | 'prefillDone' | 'prefillTotal' | 'generatedTokens' | 'prefillElapsedMs' | 'decodeElapsedMs'
>;

/**
 * This lane's speed, or null. Prefill: tokens counted ÷ prefill time, only for a prefill of `PREFILL_MIN_TOKENS`+
 * tokens that has counted some. Decode: tokens after the first ÷ decode time, from 2 tokens and 250 ms.
 */
export function laneSpeed(lane: LaneSpeedFields): { kind: 'prefill' | 'decode'; tps: number } | null {
  if (lane.phase === 'prefill') {
    const done = lane.prefillDone;
    const total = lane.prefillTotal;
    const ms = lane.prefillElapsedMs;
    if (!known(done) || !known(total) || !known(ms)) return null;
    if (total < PREFILL_MIN_TOKENS || done <= 0 || ms <= 0) return null;
    return { kind: 'prefill', tps: (done * 1000) / ms };
  }
  if (lane.phase === 'decode') {
    const tokens = lane.generatedTokens;
    const ms = lane.decodeElapsedMs;
    if (!known(tokens) || !known(ms) || tokens <= 1 || ms < DECODE_SPEED_MIN_MS) return null;
    return { kind: 'decode', tps: ((tokens - 1) * 1000) / ms };
  }
  return null;
}

function speedWord(kind: 'prefill' | 'decode'): { label: string; unit: string } {
  return kind === 'prefill' ? METRIC.prefillSpeed : METRIC.decodeSpeed;
}

/** "1,180 tok/s", or null when the lane has no speed under the rules of `laneSpeed`. */
export function laneSpeedText(lane: LaneSpeedFields): string | null {
  const speed = laneSpeed(lane);
  return speed === null ? null : withUnit(formatTps(speed.tps), speedWord(speed.kind).unit);
}

/** The Speed cell of a lane: its prefill or decode speed, or "—". */
export function LaneSpeed(props: LaneSpeedFields): ReactElement {
  const speed = laneSpeed(props);
  if (speed === null) return <span className={styles.progressMuted}>{NO_VALUE}</span>;
  const word = speedWord(speed.kind);
  return (
    <span className={styles.speedCell} title={word.label}>
      <SwapText text={withUnit(formatTps(speed.tps), word.unit)} granularity="char" />
    </span>
  );
}

function contextShare(props: Pick<LiveTurnRowProps, 'promptTokens' | 'generatedTokens' | 'ctxWindow'>): number | null {
  if (props.promptTokens === null || props.ctxWindow === null || props.ctxWindow <= 0) return null;
  return (props.promptTokens + (props.generatedTokens ?? 0)) / props.ctxWindow;
}

type ProgressProps = Pick<
  LiveTurnRowProps,
  | 'model'
  | 'phase'
  | 'prefillDone'
  | 'prefillTotal'
  | 'generatedTokens'
  | 'promptTokens'
  | 'cachedTokens'
  | 'ctxWindow'
  | 'live'
  | 'waitMs'
  | 'slotMs'
  | 'prefillElapsedMs'
  | 'decodeElapsedMs'
> & {
  /** Append the lane's speed after the words while the cell is wide enough (rows with no Speed column of their own). */
  withSpeed?: boolean;
};

/** The phase word in a badge, with its phase-colored key; the word shimmers while the turn works. */
export function PhaseBadge({ phase, live }: { phase: LivePhase; live?: boolean }): ReactElement {
  const phaseKey = phase ?? 'running';
  const waiting = phase === 'queued' || phase === 'slot';
  const busy = phase === 'prepare' || phase === 'prefill' || phase === 'decode';
  const working = live !== false && busy;
  return (
    <Badge tone={busy ? 'info' : 'neutral'} outline={waiting} size="sm">
      <span className={styles.key} data-phase={waiting ? 'waiting' : phaseKey} aria-hidden="true" />
      <LiveLabel active={working}>{PHASE_WORD[phaseKey]}</LiveLabel>
    </Badge>
  );
}

export type EngineStateKey = keyof typeof ENGINE_STATE;

export interface LaneCounts {
  decode: number;
  prefill: number;
  other: number;
  waiting: number;
}

/** "1 lane"; "3 lanes · 2 decode · 1 prefill"; null with no lane. */
export function laneCountText(lanes: LaneCounts): string | null {
  const parts: Array<[number, string]> = [
    [lanes.decode, LANES.decoding],
    [lanes.prefill, LANES.prefilling],
    [lanes.other, LANES.other],
    [lanes.waiting, LANES.waiting],
  ];
  const total = parts.reduce((sum, [count]) => sum + (count > 0 ? count : 0), 0);
  if (total <= 0) return null;
  if (total === 1) return LANES.one;
  return [
    fill(LANES.many, { n: formatNumber(total) }),
    ...parts.filter(([count]) => count > 0).map(([count, word]) => fill(word, { n: formatNumber(count) })),
  ].join(' · ');
}

const BUSY_STATES: ReadonlySet<EngineStateKey> = new Set(['loading', 'decode', 'prefill', 'prefillDecode', 'working']);

export interface EngineStateBadgeProps {
  state: EngineStateKey;
  /** Lanes per phase in the newest snapshot; their count follows the badge. */
  lanes?: LaneCounts;
  /** False when the snapshot is stale: no shimmer. Default true. */
  live?: boolean;
}

/** The engine's state word in a badge with its band-colored key, then the lane counts. */
export function EngineStateBadge(props: EngineStateBadgeProps): ReactElement {
  const busy = BUSY_STATES.has(props.state);
  const count = props.lanes === undefined ? null : laneCountText(props.lanes);
  return (
    <span className={styles.engineState}>
      <Badge tone={busy ? 'info' : 'neutral'} outline={props.state === 'waiting' || props.state === 'queued'} size="sm">
        <span className={styles.key} data-state={props.state} aria-hidden="true" />
        <LiveLabel active={props.live !== false && busy}>{ENGINE_STATE[props.state]}</LiveLabel>
      </Badge>
      {count !== null ? <span className={styles.laneCount}>{count}</span> : null}
    </span>
  );
}

/**
 * Prefill: a bar and "done of total tokens" ("done / total" when narrow); before the first count a busy bar, the
 * total and the time so far. Decode: tokens out. Else a muted word.
 */
export function TurnProgress(props: ProgressProps): ReactElement {
  const rich = useRichMotion();
  const live = props.live !== false;
  const progress = progressText(props);
  const speed = props.withSpeed === true ? laneSpeedText(props) : null;
  let middle: ReactNode;
  if (props.phase === 'prefill' && props.prefillDone === 0 && props.prefillTotal !== null) {
    middle = (
      <>
        <span
          className={styles.busyBar}
          role="progressbar"
          aria-label={`${props.model} prefill`}
          aria-valuetext={progress}
          data-busy={live && rich ? '' : undefined}
        />
        <span className={styles.progressText}>
          <SwapText text={progress} granularity="char" />
        </span>
      </>
    );
  } else if (props.phase === 'prefill' && props.prefillDone !== null && props.prefillTotal !== null) {
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
  return (
    <span className={styles.progress}>
      {middle}
      {speed !== null ? (
        <span className={styles.speed}>
          <SwapText text={speed} granularity="char" />
        </span>
      ) : null}
    </span>
  );
}

/** Prompt plus output against the turn's window: a bar and a percent, or "ctx —". */
export function ContextFill(props: ProgressProps): ReactElement {
  const share = contextShare(props);
  return (
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
  );
}

/** One turn that is running now: model, phase word, progress and the lane's speed, elapsed time, context fill. */
export function LiveTurnRow(props: LiveTurnRowProps): ReactElement {
  const rich = useRichMotion();
  const phaseKey = props.phase ?? 'running';
  const word = PHASE_WORD[phaseKey];
  const share = contextShare(props);
  const progress = progressText(props);
  const speed = laneSpeedText(props);
  const name = [
    props.model,
    word.toLowerCase(),
    progress,
    ...(speed === null ? [] : [speed]),
    spokenElapsed(props.elapsedMs),
    `context ${share === null ? 'unknown' : formatPercent(share)}`,
  ].join(', ');

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
        <PhaseBadge phase={props.phase} live={props.live} />
      </span>
      <TurnProgress {...props} withSpeed />
      <span className={styles.elapsed}>
        <SwapText text={formatElapsed(props.elapsedMs)} reserve="5ch" granularity="char" />
      </span>
      <ContextFill {...props} />
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
