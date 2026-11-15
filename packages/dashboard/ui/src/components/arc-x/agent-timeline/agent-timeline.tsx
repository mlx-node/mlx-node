import { CircleCheck, CircleX, CornerDownRight, Pause, Play, RotateCcw, RotateCw } from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import {
  Fragment,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
  type PointerEvent,
  type ReactElement,
  type ReactNode,
  type RefObject,
} from 'react';
import { createPortal } from 'react-dom';
import { Link } from 'react-router-dom';

import { cn } from '../../../lib/cn';
import { boxInView, clipBoxes } from '../../../lib/float-box';
import { formatCountNoun, formatDuration as defaultDuration, formatNumber } from '../../../lib/format';
import SegmentedControl from '../../arc/segmented-control/segmented-control';
import { ChartTable } from '../chart/chart-table';
import { IconButton } from '../icon-button/icon-button';
import { LiveDot } from '../live-dot/live-dot';
import { ModelLogo } from '../model-badge/model-logos';
import { motionX } from '../motion/motion';
import { useEntrance, useRichMotion } from '../motion/use-entrance';
import { useStaticMotion } from '../motion/use-static-motion';
import { Placeholder } from '../placeholder/placeholder';
import { SwapText } from '../swap-text/swap-text';
import {
  activeLanes,
  axisTicks,
  cardPlace,
  drawnSpans,
  isPending,
  liveAxisEnd,
  liveNow,
  moveFocus,
  replayRate,
  scrubStep,
  secondsAxis,
  selectionPath,
  tabStop,
  tokensAt,
  type AgentClock,
  type AgentHandoff,
  type AgentLane,
  type AgentOutcome,
  type AgentSpan,
  type AgentTimelineMode,
  type AgentTokenReport,
  type AxisUnit,
  type FocusKey,
} from './agent-timeline-lib';
import styles from './agent-timeline.module.css';
import { browserClock, useReplayClock } from './use-replay-clock';

export type {
  AgentClock,
  AgentHandoff,
  AgentLane,
  AgentOutcome,
  AgentSpan,
  AgentTimelineMode,
  AgentTokenReport,
} from './agent-timeline-lib';

export interface AgentTimelineProps {
  /** Accessible name of the section: "Turn 12 timeline". */
  label: string;
  title: string;
  lanes: readonly AgentLane[];
  spans: readonly AgentSpan[];
  handoffs?: readonly AgentHandoff[];
  reports?: readonly AgentTokenReport[];
  /** Length of the record in ms; values under 1 are read as 1. Live: the offset of the last recorded event. */
  durationMs: number;
  mode: AgentTimelineMode;
  /** How the record ended; shown when settled and in stale mode. */
  outcome: AgentOutcome;
  /**
   * Where the turn starts on the axis (default 0). Above 0 the axis begins with background work before a turn that a
   * background event started: a rule marks the start, and the status line counts the turn alone.
   */
  turnStartMs?: number;
  /** 'card' (default): own border, radius and fill. 'flush': none, for use inside a Panel. */
  surface?: 'card' | 'flush';
  formatDuration?: (ms: number) => string;
  /** The unit of the axis for an axis of `spanMs`. Default: seconds printed with formatDuration. */
  axisUnit?: (spanMs: number) => AxisUnit;
  /** Test seam. Default: Date.now and requestAnimationFrame. */
  clock?: AgentClock;
  className?: string;
}

type Phase = 'settled' | 'playing' | 'paused';

const KIND_WORD: Record<AgentSpan['kind'], string> = {
  generate: 'Generating',
  tool: 'Tool call',
  wait: 'Waiting',
  run: 'Ran',
  idle: 'Idle',
};
const ENTRANCE_STEPS = 4;
const TOKENS_TITLE = 'Prompt and output tokens of the turns that finished by this point';

function minutesClock(ms: number): string {
  const total = Math.floor(ms / 1000);
  const seconds = total % 60;
  return `${String(Math.floor(total / 60))}:${seconds < 10 ? '0' : ''}${String(seconds)}`;
}

/** Recorded time under the playhead: one decimal under a minute, then m:ss. */
export function replayTime(ms: number): string {
  if (ms <= 0) return '0 s';
  if (ms < 60_000) return `${(Math.floor(ms / 100) / 10).toFixed(1)} s`;
  return minutesClock(ms);
}

/** Running time of a live turn: whole seconds, then m:ss. */
function liveTime(ms: number): string {
  if (ms < 60_000) return `${String(Math.floor(Math.max(0, ms) / 1000))} s`;
  return minutesClock(ms);
}

/** The status text moves at most 10 times a second under a minute and once a second after it. */
function quantum(ms: number, live: boolean): number {
  if (live || ms >= 60_000) return Math.floor(ms / 1000) * 10;
  return Math.floor(ms / 100);
}

function countBelow(sorted: readonly number[], value: number): number {
  let low = 0;
  let high = sorted.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (sorted[middle]! < value) low = middle + 1;
    else high = middle;
  }
  return low;
}

function capital(text: string): string {
  return text === '' ? text : text[0]!.toUpperCase() + text.slice(1);
}

/** Text in a box held at the width of the widest `reserve` (drawn invisibly from attributes, so never read). */
function Sized({ text, reserve = [] }: { text: ReactNode; reserve?: readonly string[] }): ReactElement {
  return (
    <span className={styles.sized}>
      {[...new Set(reserve)].map((value) => (
        <span key={value} className={styles.reserve} data-reserve={value} />
      ))}
      <span className={styles.sizedText}>{text}</span>
    </span>
  );
}

interface StatusPart {
  key: string;
  node: ReactNode;
}

interface CardSpot {
  id: string;
  x: number;
  y: number;
  height: number;
  /** The card came from another span: it morphs there instead of following at once. */
  morph: boolean;
}

interface CardLayerProps {
  spanId: string | undefined;
  card: ReactElement | null;
  rich: boolean;
  /** Changes whenever the timeline's own layout may have moved the span. */
  layout: string;
  buttons: RefObject<Map<string, HTMLButtonElement>>;
  labels: RefObject<HTMLDivElement | null>;
}

/**
 * The detail card in a window-sized layer on the body, so no ancestor of the timeline clips it. It follows its span on
 * every scroll and resize and is removed while the span is out of view.
 */
function CardLayer(props: CardLayerProps): ReactElement | null {
  const { spanId, buttons, labels, layout, rich, card } = props;
  const measureRef = useRef<HTMLDivElement | null>(null);
  const [spot, setSpot] = useState<CardSpot | null>(null);
  const [moved, setMoved] = useState(0);

  useLayoutEffect(() => {
    const button = spanId === undefined ? undefined : buttons.current.get(spanId);
    const measure = measureRef.current;
    let next: CardSpot | null = null;
    if (spanId !== undefined && button !== undefined && measure !== null) {
      const rect = button.getBoundingClientRect();
      const { view, limits } = clipBoxes(button);
      const labelsRight = labels.current?.getBoundingClientRect().right;
      if (labelsRight !== undefined) view.left = Math.max(view.left, labelsRight);
      const shown = boxInView(rect, view);
      if (shown !== null) {
        const size = { width: measure.offsetWidth, height: measure.offsetHeight };
        const at = cardPlace({ left: shown.left, top: rect.top, bottom: rect.bottom }, size, limits);
        next = { id: spanId, x: at.x, y: at.y, height: size.height, morph: spot !== null && spot.id !== spanId };
      }
    }
    const changed =
      next === null
        ? spot !== null
        : spot === null || spot.id !== next.id || spot.x !== next.x || spot.y !== next.y || spot.height !== next.height;
    if (changed) setSpot(next);
  }, [spanId, buttons, labels, layout, moved, spot]);

  useEffect(() => {
    if (spanId === undefined) return;
    const move = (): void => setMoved((count) => count + 1);
    window.addEventListener('scroll', move, { capture: true, passive: true });
    window.addEventListener('resize', move);
    return () => {
      window.removeEventListener('scroll', move, { capture: true });
      window.removeEventListener('resize', move);
    };
  }, [spanId]);

  if (typeof document === 'undefined') return null;
  const follow = rich && spot?.morph === true ? motionX.spring.morph : { duration: 0 };
  return createPortal(
    <div className={styles.cardLayer} aria-hidden="true">
      <div ref={measureRef} className={cn(styles.card, styles.measure)}>
        {card}
      </div>
      <AnimatePresence>
        {card !== null && spot !== null ? (
          <motion.div
            key="card"
            className={styles.card}
            data-pop={rich ? '' : undefined}
            initial={{ x: spot.x, y: spot.y, height: spot.height }}
            animate={{ x: spot.x, y: spot.y, height: spot.height }}
            exit={{ opacity: 0, transition: rich ? motionX.swap.out : { duration: 0 } }}
            transition={{ x: follow, y: follow, height: rich ? motionX.spring.morph : { duration: 0 } }}
          >
            {card}
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>,
    document.body,
  );
}

/** One agent turn as lanes of spans, with handoffs, a scrubbable replay of recorded times and a live follow mode. */
export function AgentTimeline(props: AgentTimelineProps): ReactElement {
  const { lanes, spans, mode, outcome } = props;
  const handoffs = useMemo(() => props.handoffs ?? [], [props.handoffs]);
  const reports = useMemo(() => props.reports ?? [], [props.reports]);
  const clock = props.clock ?? browserClock;
  const format = props.formatDuration ?? ((ms: number) => defaultDuration(ms));
  const staticMotion = useStaticMotion();
  const rich = useRichMotion();
  const duration = Math.max(1, props.durationMs);
  const total = format(duration);
  const turnStart = Math.min(Math.max(0, props.turnStartMs ?? 0), duration);
  /** The turn alone, without the background stretch before it. */
  const turnTotal = format(duration - turnStart);
  const idleText = turnStart > 0 ? `after ${format(turnStart)} idle` : null;
  /** Recorded time under the playhead; at the end it is the total, so the two sides never disagree. */
  const timeText = (ms: number): string => (ms >= duration ? total : replayTime(ms));
  const mainLane = lanes[0]?.id ?? '';
  const live = mode.kind === 'live';
  const stale = mode.kind === 'stale';
  const recorded = mode.kind === 'recorded';
  const summaryId = useId();
  const laneIdBase = useId();

  const rootRef = useRef<HTMLElement | null>(null);
  const frameRef = useRef<HTMLDivElement | null>(null);
  const plotRef = useRef<HTMLDivElement | null>(null);
  const trackRef = useRef<HTMLDivElement | null>(null);
  const scrollerRef = useRef<HTMLDivElement | null>(null);
  const knobRef = useRef<HTMLDivElement | null>(null);
  const labelsRef = useRef<HTMLDivElement | null>(null);
  const buttons = useRef(new Map<string, HTMLButtonElement>());
  const entrance = useEntrance(rootRef);

  const liveElapsed = (): number => (mode.kind === 'live' ? liveNow(clock.now() - mode.originMs, staticMotion) : 0);

  const [phase, setPhase] = useState<Phase>('settled');
  const [rate, setRate] = useState(() => replayRate(duration));
  const [viewT, setViewT] = useState(() => (mode.kind === 'live' ? liveElapsed() : duration));
  const [liveAxis, setLiveAxis] = useState(() => liveAxisEnd(Math.max(liveElapsed(), duration)));
  const [arrived, setArrived] = useState<ReadonlySet<string>>(() => new Set());
  /** Spans with a glyph whose start a playing replay crossed: their glyph pops when it first shows. */
  const [arrivedSpans, setArrivedSpans] = useState<ReadonlySet<string>>(() => new Set());
  const [announce, setAnnounce] = useState('');
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [hoverId, setHoverId] = useState<string | null>(null);
  const [focusId, setFocusId] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [layoutTick, setLayoutTick] = useState(0);
  const [seenKind, setSeenKind] = useState(mode.kind);
  const [seenDuration, setSeenDuration] = useState(duration);

  const pos = useRef(mode.kind === 'live' ? viewT : duration);
  const viewRef = useRef({ count: -1, quantum: -1, t: viewT });

  if (seenDuration !== duration) {
    setSeenDuration(duration);
    if (recorded && phase === 'settled') {
      pos.current = duration;
      setViewT(duration);
    }
    if (recorded) setRate(replayRate(duration));
  }
  if (seenKind !== mode.kind) {
    setSeenKind(mode.kind);
    if (mode.kind === 'recorded') {
      setPhase('settled');
      pos.current = duration;
      setViewT(duration);
      setRate(replayRate(duration));
      if (seenKind === 'live' || seenKind === 'stale') {
        setAnnounce(
          outcome.detail === undefined
            ? `Turn finished: ${outcome.word}, ${turnTotal}`
            : `${outcome.word}: ${outcome.detail}`,
        );
      }
    } else if (mode.kind === 'stale') {
      if (seenKind === 'live') setAnnounce('No update for 10 seconds');
      else setLiveAxis(liveAxisEnd(Math.max(mode.atMs, duration)));
    } else {
      const now = liveElapsed();
      pos.current = now;
      setViewT(now);
      setLiveAxis(liveAxisEnd(Math.max(now, duration)));
    }
  }

  const settled = recorded && phase === 'settled';
  const playing = recorded && phase === 'playing';
  // A turn that goes stale keeps the axis its live clock reached.
  const axisMs = recorded
    ? duration
    : stale
      ? Math.max(liveAxis, liveAxisEnd(Math.max(mode.atMs, duration)))
      : liveAxis;
  const t = settled ? duration : stale ? mode.atMs : viewT;
  const axisUnitOf = props.axisUnit ?? secondsAxis(format);
  const unit = axisUnitOf(axisMs);
  /** An offset in words: the axis unit of an axis under a minute, then m:ss as on the status line. */
  const wordUnit = axisUnitOf(Math.min(axisMs, 59_999));
  const offset = (ms: number): string => (ms >= 60_000 ? minutesClock(ms) : wordUnit.format(Math.max(0, ms)));

  const laneIndex = useMemo(() => new Map(lanes.map((lane, index) => [lane.id, index])), [lanes]);
  const placed = useMemo(
    () =>
      spans
        .filter((span) => laneIndex.has(span.laneId))
        .sort((a, b) => laneIndex.get(a.laneId)! - laneIndex.get(b.laneId)! || a.startMs - b.startMs),
    [spans, laneIndex],
  );
  const { drawn, hidden } = useMemo(() => drawnSpans(placed), [placed]);
  const eventTimes = useMemo(() => {
    const times = [
      ...drawn.map((span) => span.startMs),
      ...handoffs.map((handoff) => handoff.atMs),
      ...reports.map((report) => report.atMs),
    ];
    return times.sort((a, b) => a - b);
  }, [drawn, handoffs, reports]);
  const sortedHandoffs = useMemo(() => [...handoffs].sort((a, b) => a.atMs - b.atMs), [handoffs]);

  const writeNow = (ms: number): void => {
    trackRef.current?.style.setProperty('--now', String(ms / 1000));
  };

  const syncView = (ms: number, followsClock: boolean): void => {
    const count = countBelow(eventTimes, ms);
    const step = quantum(ms, followsClock);
    const seen = viewRef.current;
    if (count === seen.count && step === seen.quantum) return;
    if (rich && count !== seen.count && ms > seen.t) {
      const crossed = sortedHandoffs.filter((handoff) => handoff.atMs >= seen.t && handoff.atMs < ms);
      if (crossed.length > 0) setArrived((current) => new Set([...current, ...crossed.map((handoff) => handoff.id)]));
      if (!followsClock) {
        const glyphs = drawn.filter(
          (span) =>
            span.startMs >= seen.t && span.startMs < ms && (span.status === 'failed' || span.retry !== undefined),
        );
        if (glyphs.length > 0) setArrivedSpans((current) => new Set([...current, ...glyphs.map((span) => span.id)]));
      }
    }
    viewRef.current = { count, quantum: step, t: ms };
    setViewT(ms);
  };

  const jumpTo = (ms: number, next: Phase): void => {
    pos.current = ms;
    viewRef.current = { count: countBelow(eventTimes, ms), quantum: quantum(ms, false), t: ms };
    setArrived(new Set());
    setArrivedSpans(new Set());
    setViewT(ms);
    setPhase(next);
    writeNow(next === 'settled' ? duration : ms);
  };

  useReplayClock(playing || live, clock, (dt) => {
    if (mode.kind === 'live') {
      const now = liveNow(clock.now() - mode.originMs, staticMotion);
      if (now === pos.current) return;
      pos.current = now;
      writeNow(now);
      const axis = liveAxisEnd(Math.max(now, duration));
      if (axis !== liveAxis) setLiveAxis(axis);
      syncView(now, true);
      return;
    }
    let next = pos.current + dt * rate;
    if (next >= duration) {
      next = duration;
      pos.current = next;
      writeNow(next);
      viewRef.current = { count: eventTimes.length, quantum: quantum(next, false), t: next };
      setViewT(next);
      setArrivedSpans(new Set());
      setPhase('settled');
      setAnnounce('Replay finished');
      return;
    }
    pos.current = next;
    writeNow(next);
    syncView(next, false);
  });

  useLayoutEffect(() => {
    if (settled) writeNow(duration);
    else if (stale) writeNow(mode.atMs);
    else writeNow(pos.current);
  });

  const scrubTo = (ms: number, end = false): void => {
    const value = Math.max(0, Math.min(duration, ms));
    if (end || (value >= duration && phase === 'settled')) {
      jumpTo(duration, 'settled');
      return;
    }
    jumpTo(value, 'paused');
  };

  const play = (): void => {
    if (phase === 'playing') {
      setPhase('paused');
      setViewT(pos.current);
      setAnnounce(`Paused at ${timeText(pos.current)}`);
      return;
    }
    if (phase === 'settled' || pos.current >= duration) {
      jumpTo(0, 'playing');
      return;
    }
    setPhase('playing');
  };

  const shown = drawn.filter((span) => !isPending(span, t, settled));
  const stop = tabStop(activeId, drawn, t, settled);
  const path = selectionPath(selectedId, drawn, handoffs, mainLane);
  const laneOrder = lanes.map((lane) => lane.id);

  const focusSpan = (id: string | null): void => {
    if (id === null) return;
    setActiveId(id);
    buttons.current.get(id)?.focus();
  };

  useLayoutEffect(() => {
    const focused = document.activeElement;
    if (!(focused instanceof HTMLElement) || !plotRef.current?.contains(focused)) return;
    if (focused.dataset.pending !== undefined) plotRef.current.focus();
  });

  const onPlotKey = (event: KeyboardEvent<HTMLDivElement>): void => {
    const target = event.target as HTMLElement;
    if (event.key === 'Escape') {
      setSelectedId(null);
      event.preventDefault();
      return;
    }
    if (target === plotRef.current) {
      if ((event.key === 'ArrowRight' || event.key === 'ArrowDown') && shown.length > 0) {
        focusSpan(shown[0]!.id);
        event.preventDefault();
      }
      return;
    }
    const id = target.dataset.span;
    if (id === undefined) return;
    const keys: readonly string[] = ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'];
    if (!keys.includes(event.key)) return;
    event.preventDefault();
    focusSpan(moveFocus(id, event.key as FocusKey, shown, laneOrder, event.metaKey || event.ctrlKey));
  };

  const onKnobKey = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (!recorded) return;
    const step = scrubStep(duration);
    const big = event.shiftKey ? 10 : 1;
    const here = settled ? duration : pos.current;
    switch (event.key) {
      case 'ArrowRight':
      case 'ArrowUp':
        scrubTo(here + step * big);
        break;
      case 'ArrowLeft':
      case 'ArrowDown':
        scrubTo(here - step * big);
        break;
      case 'PageUp':
        scrubTo(here + step * 10);
        break;
      case 'PageDown':
        scrubTo(here - step * 10);
        break;
      case 'Home':
        scrubTo(0);
        break;
      case 'End':
        scrubTo(duration, true);
        break;
      default:
        return;
    }
    event.preventDefault();
  };

  const scrubFromPointer = (event: PointerEvent<HTMLDivElement>): void => {
    const rect = event.currentTarget.getBoundingClientRect();
    if (rect.width <= 0) return;
    scrubTo(((event.clientX - rect.left) / rect.width) * duration);
  };

  // Card: hovered, else focused, else selected; never a pending span.
  const cardId = [hoverId, focusId, selectedId].find(
    (id) => id !== null && drawn.some((span) => span.id === id && !isPending(span, t, settled)),
  );
  const cardSpan = cardId === undefined ? undefined : drawn.find((span) => span.id === cardId);

  useEffect(() => {
    const frame = frameRef.current;
    if (frame === null || typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(() => setLayoutTick((tick) => tick + 1));
    observer.observe(frame);
    return () => observer.disconnect();
  }, []);

  // Words.
  const spanEnd = (span: AgentSpan): number =>
    span.endMs ?? (live ? t : stale ? mode.atMs : Math.max(span.startMs, duration));
  const staleRunning = (span: AgentSpan): boolean => stale && span.status === 'running';
  const laneOf = (span: AgentSpan): AgentLane | undefined => lanes[laneIndex.get(span.laneId) ?? -1];
  const asOf = mode.kind === 'stale' ? mode.asOfText : '';
  const spanName = (span: AgentSpan): string => {
    const end = spanEnd(span);
    const parts = [
      laneOf(span)?.name ?? '',
      span.label,
      format(end - span.startMs),
      `from ${offset(span.startMs)} to ${span.endMs === null ? 'now' : offset(end)}`,
    ];
    if (span.status === 'failed') parts.push('failed');
    if (span.retry !== undefined) parts.push(span.retry);
    if (span.status === 'running') parts.push(stale ? `running as of ${asOf}` : 'running');
    if (span.status === 'unfinished') parts.push('not finished');
    if (span.open === 'start' || span.open === 'both') parts.push('start not recorded');
    if (span.open === 'end' || span.open === 'both') parts.push('end not recorded');
    if (span.cut === 'start' || span.cut === 'both') parts.push('began before this turn');
    if (span.cut === 'end' || span.cut === 'both') parts.push('went on after this turn');
    return parts.join(', ');
  };

  const tokens = tokensAt(reports, mainLane, t, turnStart);
  const subagentsText = (count: number, figure: number, withCount: boolean, before = false): string =>
    (withCount
      ? `${formatNumber(figure)} in ${formatCountNoun(count, 'subagent', 'subagents')}`
      : `${formatNumber(figure)} in subagents`) + (before ? ' before the turn' : '');
  const swap = (text: string, char = false): ReactNode =>
    rich ? <SwapText text={text} granularity={char ? 'char' : 'text'} /> : text;

  const statusParts: StatusPart[] = [];
  let word: ReactNode;
  if (live) {
    word = <LiveDot tone="success" pulse label="Live" showLabel />;
  } else if (stale) {
    word = swap('In progress');
  } else if (settled) {
    const icon =
      outcome.tone === 'success' ? (
        <CircleCheck size={14} aria-hidden="true" />
      ) : outcome.tone === 'danger' ? (
        <CircleX size={14} aria-hidden="true" />
      ) : null;
    word = (
      <span className={styles.word} data-tone={outcome.tone}>
        {icon}
        {swap(outcome.word)}
      </span>
    );
  } else {
    word = swap(playing ? 'Replaying' : t > 0 ? 'Paused' : 'Ready');
  }
  statusParts.push({ key: 'word', node: word });
  if (live) statusParts.push({ key: 'time', node: <Sized text={liveTime(t - turnStart)} /> });
  else if (stale) statusParts.push({ key: 'time', node: `as of ${mode.asOfText}` });
  else if (settled) statusParts.push({ key: 'time', node: outcome.detail ?? turnTotal });
  else {
    const left = timeText(t);
    statusParts.push({
      key: 'time',
      node: (
        <Sized
          text={
            <>
              {t >= 60_000 ? swap(left, true) : left}
              {` of ${total}`}
            </>
          }
          reserve={[
            `${total} of ${total}`,
            `${replayTime(Math.min(duration - 1, 59_900))} of ${total}`,
            ...(duration > 60_000 ? [`${replayTime(duration - 1)} of ${total}`] : []),
          ]}
        />
      ),
    });
  }
  if (idleText !== null && (live || stale || settled)) statusParts.push({ key: 'idle', node: idleText });
  if (tokens.main !== null && !live && !stale && outcome.detail === undefined) {
    statusParts.push({
      key: 'main',
      node: <span title={TOKENS_TITLE}>{swap(`${formatNumber(tokens.main)} tokens`, true)}</span>,
    });
  }
  if (tokens.sub !== null) {
    statusParts.push({
      key: 'sub',
      node: <span title={TOKENS_TITLE}>{swap(subagentsText(tokens.subLanes, tokens.sub, settled), true)}</span>,
    });
  }
  if (tokens.before !== null) {
    statusParts.push({
      key: 'before',
      node: (
        <span title={TOKENS_TITLE}>{swap(subagentsText(tokens.beforeLanes, tokens.before, settled, true), true)}</span>
      ),
    });
  }

  const finalTokens = tokensAt(reports, mainLane, duration, turnStart);
  const summaryTime = outcome.detail !== undefined ? capital(outcome.detail) : turnTotal;
  const summaryParts = [`${outcome.word}.`, `${summaryTime}${idleText !== null ? `, ${idleText}` : ''}.`];
  const tokenWords: string[] = [];
  if (finalTokens.main !== null && outcome.detail === undefined)
    tokenWords.push(`${formatNumber(finalTokens.main)} tokens`);
  if (finalTokens.sub !== null) tokenWords.push(subagentsText(finalTokens.subLanes, finalTokens.sub, true));
  if (finalTokens.before !== null)
    tokenWords.push(subagentsText(finalTokens.beforeLanes, finalTokens.before, true, true));
  if (tokenWords.length > 0) summaryParts.push(`${tokenWords.join(', ')}.`);

  const ticks = axisTicks(axisMs, unit.unitMs);
  const plotName =
    drawn.length === 0 ? 'Steps, none recorded' : stop === null ? `Steps, none reached at ${replayTime(t)}` : 'Steps';
  const anyOpen = drawn.some((span) => span.open !== undefined || span.status === 'unfinished' || staleRunning(span));
  const legend: Array<{ key: string; label: string; mark: ReactNode }> = [];
  const swatch = (kind: string): ReactNode => <span className={styles.swatch} data-swatch={kind} aria-hidden="true" />;
  if (placed.some((span) => span.kind === 'generate'))
    legend.push({ key: 'generate', label: 'Generating', mark: swatch('generate') });
  if (placed.some((span) => span.kind === 'tool'))
    legend.push({ key: 'tool', label: 'Tool call', mark: swatch('tool') });
  if (placed.some((span) => span.kind === 'wait')) legend.push({ key: 'wait', label: 'Waiting', mark: swatch('wait') });
  if (placed.some((span) => span.kind === 'idle')) legend.push({ key: 'idle', label: 'Idle', mark: swatch('idle') });
  if (handoffs.length > 0) {
    legend.push({ key: 'handoff', label: 'Handoff', mark: <CornerDownRight size={13} aria-hidden="true" /> });
  }
  if (placed.some((span) => span.status === 'failed') || handoffs.some((handoff) => handoff.failed === true)) {
    legend.push({
      key: 'failed',
      label: 'Failed',
      mark: <CircleX size={13} aria-hidden="true" className={styles.danger} />,
    });
  }
  if (placed.some((span) => span.retry !== undefined)) {
    legend.push({ key: 'retry', label: 'Retried', mark: <RotateCw size={13} aria-hidden="true" /> });
  }
  if (placed.some((span) => span.kind === 'run'))
    legend.push({ key: 'run', label: 'No step record', mark: swatch('run') });
  if (anyOpen) legend.push({ key: 'open', label: 'Time not recorded', mark: swatch('open') });

  const lastOfLane = new Set<string>();
  {
    const last = new Map<string, AgentSpan>();
    for (const span of drawn) last.set(span.laneId, span);
    for (const span of last.values()) lastOfLane.add(span.id);
  }

  const sliderNow = playing ? Math.floor(t / 1000) * 1000 : t;
  const playLabel = settled ? 'Replay' : playing ? 'Pause' : 'Play';
  const disabledTip = 'Replay is available when the turn has ended.';

  const cardBody = (span: AgentSpan): ReactElement => {
    const lane = laneOf(span);
    const end = spanEnd(span);
    const meta = [
      format(end - span.startMs),
      `${offset(span.startMs)} → ${span.endMs === null ? (stale ? `as of ${asOf}` : 'now') : offset(end)}`,
      ...(span.facts ?? []),
    ];
    const handoffsInside =
      span.ownsHandoffs === true
        ? handoffs.filter((handoff) => handoff.atMs >= span.startMs && handoff.atMs <= end).length
        : 0;
    return (
      <div className={styles.cardBody}>
        <span className={styles.cardHead}>
          <span className={styles.dot} style={{ '--c': lane?.color } as CSSProperties} />
          <span className={styles.cardLane}>{lane?.name}</span>
          {lane?.model !== undefined ? <span className={styles.cardModel}>{lane.model}</span> : null}
        </span>
        <span className={styles.cardLabel}>{span.label}</span>
        <span className={styles.cardMeta}>{meta.join(' · ')}</span>
        {span.status === 'failed' ? <span className={styles.danger}>Failed</span> : null}
        {span.retry !== undefined ? <span className={styles.cardNote}>{span.retry}</span> : null}
        {span.note !== undefined ? <span className={styles.cardNote}>{span.note}</span> : null}
        {handoffsInside > 0 ? (
          <span className={styles.cardNote}>
            {formatCountNoun(handoffsInside, 'handoff', 'handoffs')} during this span
          </span>
        ) : null}
      </div>
    );
  };
  const card = cardSpan === undefined ? null : cardBody(cardSpan);

  const laneText = (lane: AgentLane | undefined): string =>
    lane === undefined ? '' : lane.model !== undefined ? `${lane.name} · ${lane.model}` : lane.name;
  const spanRows = placed.map((span) => {
    const end = span.endMs;
    const status = [
      span.status === 'unfinished' ? 'not finished' : span.status,
      ...(span.retry !== undefined ? [span.retry.toLowerCase()] : []),
      ...(span.open === 'start' || span.open === 'both' ? ['start not recorded'] : []),
      ...(span.cut === 'start' || span.cut === 'both' ? ['began before this turn'] : []),
      ...(span.cut === 'end' || span.cut === 'both' ? ['went on after this turn'] : []),
    ];
    return {
      key: span.id,
      current: span.id === selectedId,
      cells: [
        laneText(laneOf(span)),
        KIND_WORD[span.kind],
        span.kind === 'run' && span.note !== undefined ? span.note : span.label,
        offset(span.startMs),
        end === null ? 'running' : span.open === 'end' || span.open === 'both' ? 'not recorded' : offset(end),
        end === null ? '—' : format(end - span.startMs),
        status.join(', '),
      ],
    };
  });
  const handoffRows = handoffs.map((handoff) => ({
    key: handoff.id,
    cells: [
      lanes[laneIndex.get(handoff.laneId) ?? -1]?.name ?? handoff.laneId,
      handoff.kind === 'spawn' ? 'spawned' : 'returned',
      offset(handoff.atMs),
      handoff.failed === true ? 'failed' : 'ok',
    ],
  }));

  const lastLane = lanes.length - 1;

  return (
    <section
      ref={rootRef}
      className={cn(styles.root, props.className)}
      aria-label={props.label}
      aria-describedby={recorded ? summaryId : undefined}
      data-surface={props.surface ?? 'card'}
      data-mode={mode.kind}
      data-playing={playing ? '' : undefined}
      data-settled={settled ? '' : undefined}
      data-entrance={entrance}
      data-rich={rich ? '' : undefined}
    >
      <div ref={frameRef} className={styles.frame} style={{ '--lanes': Math.max(1, lanes.length) } as CSSProperties}>
        <header className={styles.header}>
          <h3 className={styles.title} title={props.title}>
            {props.title}
          </h3>
          <p className={styles.status} aria-hidden={recorded ? 'true' : undefined}>
            {statusParts.map((part, index) => (
              <Fragment key={part.key}>
                {index > 0 ? ' ' : null}
                <span className={styles.part}>
                  {part.node}
                  {index < statusParts.length - 1 ? (
                    <span className={styles.sep} aria-hidden="true">
                      {' ·'}
                    </span>
                  ) : null}
                </span>
              </Fragment>
            ))}
            {live ? (
              <span
                className={styles.srOnly}
              >{`, ${formatCountNoun(activeLanes(drawn, t), 'agent', 'agents')} working`}</span>
            ) : null}
          </p>
          {recorded ? (
            <p id={summaryId} className={styles.srOnly}>
              {summaryParts.join(' ')}
            </p>
          ) : null}
        </header>

        <div
          ref={scrollerRef}
          className={styles.scroller}
          onScroll={(event) => {
            const next = event.currentTarget.scrollLeft > 0;
            if (next !== scrolled) setScrolled(next);
          }}
        >
          <div className={styles.body}>
            <div ref={labelsRef} className={styles.labels} data-scrolled={scrolled ? '' : undefined}>
              {lanes.map((lane, index) => (
                <div
                  key={lane.id}
                  className={styles.label}
                  style={{ '--i': index, '--c': lane.color } as CSSProperties}
                >
                  <span className={styles.dot} aria-hidden="true" />
                  <span className={styles.nameLine}>
                    {lane.to !== undefined ? (
                      <Link
                        id={`${laneIdBase}-${String(index)}`}
                        className={styles.name}
                        to={lane.to}
                        title={lane.name}
                      >
                        {lane.name}
                      </Link>
                    ) : (
                      <span id={`${laneIdBase}-${String(index)}`} className={styles.name} title={lane.name}>
                        {lane.name}
                      </span>
                    )}
                    {lane.outcome !== undefined ? (
                      <span className={styles.outcome} data-tone={lane.outcome.tone}>
                        {lane.outcome.text}
                      </span>
                    ) : null}
                  </span>
                  {lane.model !== undefined || lane.extra !== undefined ? (
                    <span className={styles.model}>
                      {lane.model !== undefined ? <ModelLogo model={lane.model} className={styles.mark} /> : null}
                      <span className={styles.modelText}>
                        {[lane.model, lane.extra].filter((part) => part !== undefined).join(' · ')}
                      </span>
                    </span>
                  ) : null}
                </div>
              ))}
            </div>

            <div
              ref={trackRef}
              className={styles.track}
              style={{ '--axis': Math.max(1, axisMs) / 1000 } as CSSProperties}
            >
              <div
                ref={plotRef}
                className={styles.plot}
                role="group"
                aria-label={plotName}
                tabIndex={stop === null ? 0 : -1}
                data-path={path !== null ? '' : undefined}
                onKeyDown={onPlotKey}
                onClick={(event) => {
                  if ((event.target as HTMLElement).closest('[data-span]') === null) setSelectedId(null);
                }}
              >
                {ticks.map((tick) => (
                  <span
                    key={tick}
                    className={styles.grid}
                    style={{ '--t': tick / 1000 } as CSSProperties}
                    aria-hidden="true"
                  />
                ))}
                {turnStart > 0 ? (
                  <span
                    className={styles.turnStart}
                    data-turn-start=""
                    style={{ '--t': turnStart / 1000 } as CSSProperties}
                    aria-hidden="true"
                  />
                ) : null}
                <div className={styles.links} aria-hidden="true">
                  {handoffs.map((handoff) => {
                    const index = laneIndex.get(handoff.laneId);
                    if (index === undefined || index === 0) return null;
                    return (
                      <span
                        key={handoff.id}
                        className={styles.link}
                        data-kind={handoff.kind}
                        data-pending={!settled && !(handoff.atMs < t) ? '' : undefined}
                        data-arrive={arrived.has(handoff.id) ? '' : undefined}
                        data-hot={path?.handoffs.has(handoff.id) === true ? '' : undefined}
                        style={
                          { '--at': handoff.atMs / 1000, '--to': index, '--c': lanes[index]!.color } as CSSProperties
                        }
                      />
                    );
                  })}
                </div>
                {lanes.map((lane, index) => (
                  <div
                    key={lane.id}
                    className={styles.lane}
                    role="group"
                    aria-labelledby={`${laneIdBase}-${String(index)}`}
                    data-entrance-last={index === lastLane ? '' : undefined}
                    style={
                      {
                        '--i': index,
                        '--c': lane.color,
                        '--step': Math.min(index, ENTRANCE_STEPS - 1),
                      } as CSSProperties
                    }
                  >
                    {drawn
                      .filter((span) => span.laneId === lane.id)
                      .map((span) => {
                        const pending = isPending(span, t, settled);
                        const status = staleRunning(span) ? 'unfinished' : span.status;
                        const open = staleRunning(span) ? (span.open === 'start' ? 'both' : 'end') : span.open;
                        const length = span.endMs === null ? 1e9 : Math.max(0, span.endMs - span.startMs) / 1000;
                        const cutEnd = span.cut === 'end' || span.cut === 'both';
                        return (
                          <button
                            key={span.id}
                            ref={(node) => {
                              if (node === null) buttons.current.delete(span.id);
                              else buttons.current.set(span.id, node);
                            }}
                            type="button"
                            className={styles.span}
                            data-span={span.id}
                            data-kind={span.kind}
                            data-status={status}
                            data-open={open}
                            data-cut={span.cut}
                            data-retry={span.retry !== undefined ? '' : undefined}
                            data-pending={pending ? '' : undefined}
                            data-arrive={arrivedSpans.has(span.id) ? '' : undefined}
                            data-tail={lastOfLane.has(span.id) && status === 'unfinished' && !cutEnd ? '' : undefined}
                            data-dim={path !== null && !path.spans.has(span.id) ? '' : undefined}
                            aria-pressed={selectedId === span.id}
                            aria-label={spanName(span)}
                            tabIndex={!pending && span.id === stop ? 0 : -1}
                            style={{ '--s': span.startMs / 1000, '--d': length } as CSSProperties}
                            onPointerEnter={() => setHoverId(span.id)}
                            onPointerLeave={() => setHoverId((current) => (current === span.id ? null : current))}
                            onFocus={() => {
                              setActiveId(span.id);
                              setFocusId(span.id);
                            }}
                            onBlur={() => setFocusId((current) => (current === span.id ? null : current))}
                            onClick={() => {
                              setActiveId(span.id);
                              setSelectedId((current) => (current === span.id ? null : span.id));
                            }}
                          >
                            {span.status === 'failed' ? (
                              <span className={styles.glyph} aria-hidden="true">
                                <CircleX size={10} />
                              </span>
                            ) : span.retry !== undefined ? (
                              <span className={styles.glyph} aria-hidden="true">
                                <RotateCw size={10} />
                              </span>
                            ) : null}
                          </button>
                        );
                      })}
                  </div>
                ))}
                <div className={styles.ports} aria-hidden="true">
                  {handoffs.map((handoff) => {
                    const index = laneIndex.get(handoff.laneId);
                    if (index === undefined || index === 0) return null;
                    return (
                      <span
                        key={handoff.id}
                        className={styles.port}
                        data-failed={handoff.failed === true ? '' : undefined}
                        data-pending={!settled && !(handoff.atMs < t) ? '' : undefined}
                        data-arrive={arrived.has(handoff.id) ? '' : undefined}
                        data-hot={path?.handoffs.has(handoff.id) === true ? '' : undefined}
                        style={{ '--at': handoff.atMs / 1000, '--c': lanes[index]!.color } as CSSProperties}
                      />
                    );
                  })}
                </div>
                <span className={styles.playhead} aria-hidden="true" />
              </div>
              <div
                className={styles.axis}
                data-scrub={recorded ? '' : undefined}
                onPointerDown={(event) => {
                  if (!recorded) return;
                  // Keeps the compatibility mousedown from moving focus off the knob.
                  event.preventDefault();
                  event.currentTarget.setPointerCapture?.(event.pointerId);
                  setDragging(true);
                  scrubFromPointer(event);
                  knobRef.current?.focus();
                }}
                onPointerMove={(event) => {
                  if (dragging) scrubFromPointer(event);
                }}
                onPointerUp={() => setDragging(false)}
                onPointerCancel={() => setDragging(false)}
              >
                {ticks.map((tick, index) => (
                  <span
                    key={index}
                    className={styles.tick}
                    data-end={tick >= axisMs ? '' : undefined}
                    style={{ '--t': tick / 1000 } as CSSProperties}
                    aria-hidden="true"
                  >
                    {swap(tick === 0 ? '0' : unit.format(tick))}
                  </span>
                ))}
                <div
                  ref={knobRef}
                  className={styles.knob}
                  role="slider"
                  aria-label="Turn time"
                  aria-orientation="horizontal"
                  aria-valuemin={0}
                  aria-valuemax={duration / 1000}
                  aria-valuenow={Number((sliderNow / 1000).toFixed(3))}
                  aria-valuetext={`${timeText(sliderNow)} of ${total}`}
                  aria-disabled={recorded ? undefined : 'true'}
                  tabIndex={recorded ? 0 : -1}
                  data-active={dragging ? '' : undefined}
                  onKeyDown={onKnobKey}
                />
              </div>
            </div>
          </div>
        </div>
        {hidden > 0 ? <p className={styles.more}>{`+${String(hidden)} short spans not drawn`}</p> : null}

        <footer className={styles.footer}>
          <div className={styles.controls}>
            <IconButton
              label={playLabel}
              variant="secondary"
              size="sm"
              disabled={!recorded}
              tooltip={recorded ? true : disabledTip}
              icon={
                <span className={styles.playGlyphs} data-on={playing ? 'pause' : 'play'} aria-hidden="true">
                  <Play size={14} className={styles.playGlyph} data-glyph="play" />
                  <Pause size={14} className={styles.playGlyph} data-glyph="pause" />
                </span>
              }
              onClick={play}
            />
            <IconButton
              label="Restart"
              variant="ghost"
              size="sm"
              disabled={!recorded}
              tooltip={recorded ? true : disabledTip}
              icon={<RotateCcw size={14} aria-hidden="true" />}
              onClick={() => jumpTo(0, 'playing')}
            />
            {recorded ? (
              <SegmentedControl
                label="Playback speed"
                options={[
                  { value: String(replayRate(duration)), label: `${String(replayRate(duration))}×` },
                  { value: String(replayRate(duration) * 2), label: `${String(replayRate(duration) * 2)}×` },
                ]}
                value={String(rate)}
                onValueChange={(value) => setRate(Number(value))}
              />
            ) : null}
          </div>
          {legend.length > 0 ? (
            <ul className={styles.legend} aria-label="Legend">
              {legend.map((entry) => (
                <li key={entry.key} className={styles.legendItem}>
                  {entry.mark}
                  {entry.label}
                </li>
              ))}
            </ul>
          ) : null}
        </footer>

        <CardLayer
          spanId={cardSpan?.id}
          card={card}
          rich={rich}
          layout={`${String(layoutTick)}:${String(viewT)}`}
          buttons={buttons}
          labels={labelsRef}
        />
        <p className={styles.srOnly} aria-live="polite">
          {announce}
        </p>
        <div className={styles.tables}>
          <ChartTable
            caption={props.label}
            table={{ head: ['Lane', 'Kind', 'Detail', 'Start', 'End', 'Duration', 'Status'], rows: spanRows }}
          />
          {handoffRows.length > 0 ? (
            <ChartTable
              caption={`${props.label}: handoffs`}
              table={{ head: ['Subagent', 'Event', 'At', 'Status'], rows: handoffRows }}
            />
          ) : null}
        </div>
      </div>
    </section>
  );
}

export interface AgentTimelineLoadingProps {
  label: string;
  title: string;
  /** Lanes the turn is expected to have. */
  lanes: number;
  surface?: 'card' | 'flush';
}

/** The timeline's frame at its final height while its body loads. */
export function AgentTimelineLoading(props: AgentTimelineLoadingProps): ReactElement {
  const lanes = Math.max(1, props.lanes);
  return (
    <section
      className={styles.root}
      aria-label={props.label}
      aria-busy="true"
      data-surface={props.surface ?? 'card'}
      data-mode="loading"
      data-entrance="done"
    >
      <div className={styles.frame} style={{ '--lanes': lanes } as CSSProperties}>
        <header className={styles.header}>
          <h3 className={styles.title} title={props.title}>
            {props.title}
          </h3>
          <div className={styles.status}>
            <Placeholder width="16rem" />
          </div>
        </header>
        <div className={styles.scroller}>
          <div className={styles.body}>
            <div className={styles.labels}>
              {Array.from({ length: lanes }, (_, index) => (
                <div key={index} className={styles.label} style={{ '--i': index } as CSSProperties}>
                  <span />
                  <Placeholder width="70%" />
                </div>
              ))}
            </div>
            <div className={styles.track}>
              <div className={styles.plot} data-loading="">
                {Array.from({ length: lanes }, (_, index) => (
                  <div
                    key={index}
                    className={styles.lane}
                    data-placeholder-lane=""
                    style={{ '--i': index } as CSSProperties}
                  >
                    <span className={styles.bone}>
                      <Placeholder width="100%" height="100%" radius="text" step={index} />
                    </span>
                  </div>
                ))}
              </div>
              <div className={styles.axis} />
            </div>
          </div>
        </div>
        <footer className={styles.footer}>
          <div className={styles.controls}>
            <IconButton
              label="Replay"
              variant="secondary"
              size="sm"
              disabled
              icon={<Play size={14} aria-hidden="true" />}
            />
            <IconButton
              label="Restart"
              variant="ghost"
              size="sm"
              disabled
              icon={<RotateCcw size={14} aria-hidden="true" />}
            />
          </div>
        </footer>
      </div>
    </section>
  );
}
