import { motion } from 'motion/react';
import {
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
  type ReactElement,
  type ReactNode,
} from 'react';

import { cn } from '../../../lib/cn';
import { formatTps, NO_VALUE } from '../../../lib/format';
import { LIVE_GAP_MS } from '../../../lib/live-history';
import { Tooltip } from '../../arc/tooltip/tooltip';
import {
  formatLanes,
  stackAxisMax,
  stackLaneMs,
  stackPeak,
  stackPointAt,
  stackSteps,
  type StackBand,
  type StackPoint,
} from '../activity-strip/phase-stack';
import { ChartFrame, plotWidthOf, type ScrubSpec } from '../chart/chart-frame';
import { ChartLegend } from '../chart/chart-legend';
import { ChartTooltip, type TooltipRow } from '../chart/chart-tooltip';
import { areaPath, linePath, stepStackPaths, type XY } from '../chart/paths';
import { linearScale, niceTicks } from '../chart/scale';
import { useChartSize } from '../chart/use-chart-size';
import { withUnit } from '../chart/value-format';
import { motionX } from '../motion/motion';
import { useRichMotion } from '../motion/use-entrance';
import { NumberValue } from '../number-value/number-value';
import styles from './live-stream.module.css';

export interface LiveStreamPoint {
  t: number;
  value: number | null;
}

/** One fixed value above the plot. It keeps its slot whatever the value is. */
export interface LiveStreamReadout {
  key: string;
  label: string;
  /** Formatted; null -> "—". */
  value: string | null;
  unit?: string;
  /** The definition, in a tooltip on the label. */
  tip?: string;
  /** No lane is in this phase now, so the value is the last window and not this instant: drawn in the secondary color. */
  quiet?: boolean;
}

/** Lanes by phase under the line: stacked steps on the same x. */
export interface LiveStreamStack {
  /** Name of the stack, before its legend. */
  label: string;
  /** Unit of the band values in the hidden table. */
  unit: string;
  /** Plot height of the stack, default 40. */
  height?: number;
  /** Bottom to top. */
  bands: readonly StackBand[];
  /** Ascending t0, on the same clock as `points`. */
  points: readonly StackPoint[];
}

export interface LiveStreamProps {
  /** Name of the line: the figure, the scrub, the tooltip and the table column. */
  label: string;
  /** Unit of the line. */
  unit: string;
  /** Ascending t, ≤ 600. */
  points: readonly LiveStreamPoint[];
  /** Right edge of the window. The newest snapshot time, not Date.now(). */
  nowMs: number;
  /** Default 600_000. */
  windowMs?: number;
  /** False = the source stopped (no running process). */
  live: boolean;
  /** Fixed values above the plot, left to right. */
  readouts?: readonly LiveStreamReadout[];
  /** Before the readouts: the engine state badge. */
  state?: ReactNode;
  stack?: LiveStreamStack;
  /** Default formatTps. */
  formatValue?: (value: number) => string;
  /** Y tick text of the line; default formatValue. */
  formatTick?: (value: number) => string;
  /** Height of both plots and the gap between them, default 160. */
  height?: number;
  /** Default var(--series-1). */
  color?: string;
  /** Default LIVE_GAP_MS. */
  gapMs?: number;
  className?: string;
}

const TABLE_STEP_MS = 10_000;
const TABLE_ROWS = 60;
const MAX_SLIDE_MS = 1000;
const STACK_HEIGHT = 40;
/** Room between the line plot and the stack, so the line's 0 tick and the stack's top tick do not touch. */
const STACK_GAP_PX = 14;
const MIN_LINE_HEIGHT = 24;

function agoText(ms: number): string {
  const seconds = Math.max(0, Math.round(ms / 1000));
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  if (minutes === 0) return `${String(rest)} ${rest === 1 ? 'second' : 'seconds'} ago`;
  const minuteText = `${String(minutes)} ${minutes === 1 ? 'minute' : 'minutes'}`;
  return rest === 0 ? `${minuteText} ago` : `${minuteText} ${String(rest)} ${rest === 1 ? 'second' : 'seconds'} ago`;
}

function clockAgo(ms: number): string {
  const seconds = Math.max(0, Math.round(ms / 1000));
  return `−${String(Math.floor(seconds / 60))}:${String(seconds % 60).padStart(2, '0')}`;
}

/** Relative x ticks: 6 for 10 minutes ("−10m … −2m, now"); seconds below 2 minutes. */
function relativeTicks(windowMs: number): Array<{ ago: number; text: string }> {
  const ticks: Array<{ ago: number; text: string }> = [];
  for (let i = 5; i >= 0; i -= 1) {
    const ago = (windowMs * i) / 5;
    if (i === 0) ticks.push({ ago, text: 'now' });
    else if (windowMs >= 120_000) ticks.push({ ago, text: `−${String(Math.round(ago / 60_000))}m` });
    else ticks.push({ ago, text: `−${String(Math.round(ago / 1000))}s` });
  }
  return ticks;
}

interface TableRow {
  ago: number;
  /** undefined: no line sample in the step. */
  value: number | null | undefined;
  /** Mean lanes per band over the covered part of the step; null when no interval covers it. */
  bands: Record<string, number> | null;
}

/** One row per 10 s step that holds a sample, newest first, at most 60: the last line value and the mean lanes. */
function tableRows(
  points: readonly LiveStreamPoint[],
  stack: readonly StackPoint[],
  keys: readonly string[],
  now: number,
): TableRow[] {
  const newest = new Map<number, { t: number; value: number | null | undefined }>();
  const stepOf = (t: number): number => Math.floor(Math.max(0, now - t) / TABLE_STEP_MS);
  for (let i = points.length - 1; i >= 0; i -= 1) {
    const step = stepOf(points[i].t);
    if (step < TABLE_ROWS && !newest.has(step)) newest.set(step, { t: points[i].t, value: points[i].value });
  }
  for (const point of stack) {
    const step = stepOf(point.t1);
    if (step >= TABLE_ROWS) continue;
    const seen = newest.get(step);
    if (seen === undefined) newest.set(step, { t: point.t1, value: undefined });
    else if (point.t1 > seen.t) newest.set(step, { ...seen, t: point.t1 });
  }
  return [...newest.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([step, row]) => {
      const to = now - step * TABLE_STEP_MS;
      const sums = stackLaneMs(stack, keys, to - TABLE_STEP_MS, to);
      let bands: Record<string, number> | null = null;
      if (sums.coveredMs > 0) {
        bands = {};
        for (const key of keys) bands[key] = sums.laneMs[key] / sums.coveredMs;
      }
      return { ago: now - row.t, value: row.value, bands };
    });
}

/** Points in [now − window, now] as plot coordinates, null between two samples further apart than `gapMs`. */
function plotPoints(
  points: readonly LiveStreamPoint[],
  now: number,
  windowMs: number,
  gapMs: number,
  x: (t: number) => number,
  y: (value: number) => number,
): (XY | null)[] {
  const out: (XY | null)[] = [];
  let previous: LiveStreamPoint | null = null;
  for (const point of points) {
    if (point.t < now - windowMs - gapMs || point.t > now) continue;
    if (previous !== null && point.t - previous.t > gapMs) out.push(null);
    out.push(point.value === null ? null : { x: x(point.t), y: y(point.value) });
    previous = point;
  }
  return out;
}

/** The pill's label: its clipped wrapper follows the text width on `expand`; the text itself swaps at once. */
function PillLabel({ text }: { text: string }): ReactElement {
  const rich = useRichMotion();
  const textRef = useRef<HTMLSpanElement | null>(null);
  const [width, setWidth] = useState<number | null>(null);
  useLayoutEffect(() => {
    const next = textRef.current?.offsetWidth ?? 0;
    if (next > 0) setWidth(next);
  }, [text]);
  const inner = (
    <span ref={textRef} className={styles.pillText}>
      {text}
    </span>
  );
  if (!rich || width === null) return <span className={styles.pillLabel}>{inner}</span>;
  return (
    <motion.span className={styles.pillLabel} initial={false} animate={{ width }} transition={motionX.spring.expand}>
      {inner}
    </motion.span>
  );
}

function Readout({ readout }: { readout: LiveStreamReadout }): ReactElement {
  const quiet = readout.quiet === true;
  return (
    <div role="group" aria-label={readout.label} className={styles.readout} data-quiet={quiet ? '' : undefined}>
      <span className={styles.readoutLabel}>
        {readout.tip !== undefined ? (
          <Tooltip content={readout.tip}>
            <span className={styles.tip} tabIndex={0}>
              {readout.label}
            </span>
          </Tooltip>
        ) : (
          readout.label
        )}
      </span>
      <span className={cn(styles.readoutLine, quiet && styles.quiet)} data-stat-value="">
        <NumberValue text={readout.value} countUp={false} className={styles.readoutValue} />
        {readout.value !== null && readout.unit !== undefined ? (
          <>
            {' '}
            <span className={styles.unit}>{readout.unit}</span>
          </>
        ) : null}
      </span>
    </div>
  );
}

interface Frozen {
  now: number;
  points: readonly LiveStreamPoint[];
  stack: readonly StackPoint[];
  readouts: readonly LiveStreamReadout[];
  state: ReactNode;
}

/** The live panel: fixed readouts, a line that slides left as time passes and, under it, lanes by phase. */
export function LiveStream(props: LiveStreamProps): ReactElement {
  const figureRef = useRef<HTMLElement | null>(null);
  const plotRef = useRef<HTMLElement | null>(null);
  const width = useChartSize(figureRef);
  const rich = useRichMotion();
  const windowMs = props.windowMs ?? 600_000;
  const gapMs = props.gapMs ?? LIVE_GAP_MS;
  const format = props.formatValue ?? ((value: number) => formatTps(value));
  const [paused, setPaused] = useState<Frozen | null>(null);
  const now = paused?.now ?? props.nowMs;
  const points = paused?.points ?? props.points;
  const stack = props.stack;
  const stackPoints = paused?.stack ?? stack?.points ?? [];
  const readouts = paused?.readouts ?? props.readouts ?? [];
  const stateNode = paused === null ? props.state : paused.state;
  const fresh = paused === null ? 0 : props.points.filter((point) => point.t > paused.now).length;
  const state = !props.live ? 'stopped' : paused !== null ? 'paused' : 'live';
  const lineColor = props.color ?? 'var(--series-1)';

  const height = props.height ?? 160;
  const stackHeight = stack === undefined ? 0 : (stack.height ?? STACK_HEIGHT);
  const lineHeight = stack === undefined ? height : Math.max(MIN_LINE_HEIGHT, height - stackHeight - STACK_GAP_PX);
  const plotHeight = stack === undefined ? height : lineHeight + STACK_GAP_PX + stackHeight;

  const plotWidth = plotWidthOf(width, 6);
  const inWindow = points.filter((point) => point.t >= now - windowMs && point.t <= now);
  let peak = 0;
  for (const point of inWindow) if (point.value !== null) peak = Math.max(peak, point.value);
  const axis = niceTicks(0, peak > 0 ? peak : 1);
  const y = linearScale([axis.min, axis.max], [lineHeight, 0]);
  const x = (t: number): number => plotWidth * (1 - (now - t) / windowMs);
  const drawn = plotPoints(points, now, windowMs, gapMs, x, y);
  const path = linePath(drawn, 'monotone');
  const area = areaPath(drawn, lineHeight, 'monotone');
  let end: XY | null = null;
  for (let i = drawn.length - 1; i >= 0 && end === null; i -= 1) end = drawn[i] ?? null;

  const bands = stack?.bands ?? [];
  const keys = bands.map((band) => band.key);
  const stackTop = stackAxisMax(stackPeak(stackPoints, keys, now - windowMs, now));
  const stackY = linearScale([0, stackTop], [plotHeight, plotHeight - stackHeight]);
  const stackPaths =
    stack === undefined ? [] : stepStackPaths(stackSteps(stackPoints, keys, now - windowMs - gapMs, now, x), stackY);
  const stackInWindow = stackPoints.some((point) => point.t1 > now - windowMs && point.t0 < now);
  const enough = inWindow.length >= 2 || (stack !== undefined && stackInWindow);

  // One axis for both plots: the stack's two ticks first so their indices stay put while the line's ticks change.
  const lineTick = props.formatTick ?? ((value: number) => format(value));
  const ticks: Array<{ px: number; text: string }> = [
    ...(stack === undefined
      ? []
      : [
          { px: stackY(stackTop), text: String(stackTop) },
          { px: stackY(0), text: '0' },
        ]),
    ...axis.ticks.map((tick) => ({ px: y(tick), text: lineTick(tick) })),
  ];

  const togglePause = (): void => {
    setPaused((current) =>
      current === null
        ? {
            now: props.nowMs,
            points: props.points,
            stack: props.stack?.points ?? [],
            readouts: props.readouts ?? [],
            state: props.state,
          }
        : null,
    );
  };

  const slideRef = useRef<HTMLDivElement | null>(null);
  const dotRef = useRef<SVGCircleElement | null>(null);
  const last = useRef<{ now: number; endY: number | null }>({ now, endY: end?.y ?? null });
  useLayoutEffect(() => {
    const previous = last.current;
    last.current = { now, endY: end?.y ?? null };
    const group = slideRef.current;
    if (!rich || group === null || previous.now === now || paused !== null) return;
    const elapsed = Math.min(Math.max(0, now - previous.now), MAX_SLIDE_MS);
    if (elapsed <= 0 || now - previous.now > MAX_SLIDE_MS * 4) return;
    const dx = (plotWidth * (now - previous.now)) / windowMs;
    group.style.transition = 'none';
    group.style.transform = `translateX(${String(dx)}px)`;
    const dot = dotRef.current;
    const dy = previous.endY !== null && end !== null ? previous.endY - end.y : 0;
    if (dot !== null) {
      dot.style.transition = 'none';
      dot.style.transform = `translateY(${String(dy)}px)`;
    }
    void group.getBoundingClientRect();
    group.style.transition = `transform ${String(elapsed)}ms linear`;
    group.style.transform = 'translateX(0)';
    if (dot !== null) {
      dot.style.transition = `transform ${String(elapsed)}ms linear`;
      dot.style.transform = 'translateY(0)';
    }
  });

  const lineAt = new Map<number, number | null>();
  for (const point of inWindow) lineAt.set(point.t, point.value);
  const times = new Set<number>(lineAt.keys());
  if (stack !== undefined)
    for (const point of stackPoints) if (point.t1 >= now - windowMs && point.t1 <= now) times.add(point.t1);
  const positions = [...times].sort((a, b) => a - b);
  const lineText = (t: number): string | null => {
    const value = lineAt.get(t);
    return value === undefined || value === null ? null : withUnit(format(value), props.unit);
  };
  const bandsAt = (t: number): StackPoint | null => (stack === undefined ? null : stackPointAt(stackPoints, t));

  const scrub: ScrubSpec = {
    count: positions.length,
    anchor: (index) => ({ x: x(positions[index] ?? now), y: null }),
    nearest: (px) => {
      let best: number | null = null;
      let bestDistance = Infinity;
      positions.forEach((t, index) => {
        const distance = Math.abs(x(t) - px);
        if (distance < bestDistance) {
          bestDistance = distance;
          best = index;
        }
      });
      return best;
    },
    reading: (index) => {
      const t = positions[index];
      if (t === undefined) return 'no data';
      const parts = [lineText(t) ?? 'no data'];
      const at = bandsAt(t);
      if (at !== null)
        parts.push(bands.map((band) => `${band.label} ${formatLanes(at.values[band.key] ?? 0)}`).join(', '));
      parts.push(agoText(now - t));
      return parts.join('; ');
    },
    tooltip: (index) => {
      const t = positions[index];
      if (t === undefined) return null;
      const at = bandsAt(t);
      const text = lineText(t);
      if (text === null && at === null) return null;
      const rows: TooltipRow[] = [
        { key: 'line', label: props.label, value: text ?? NO_VALUE, kind: 'line', color: lineColor },
        ...bands.map((band) => ({
          key: band.key,
          label: band.label,
          value: at === null ? NO_VALUE : formatLanes(at.values[band.key] ?? 0),
          kind: 'square' as const,
          color: band.color,
        })),
      ];
      return <ChartTooltip title={clockAgo(now - t)} rows={rows} />;
    },
    crosshair: true,
    rest: positions.length > 0 ? positions.length - 1 : null,
    emptyReading: 'Collecting live samples',
    onKey: (event: KeyboardEvent<HTMLDivElement>) => {
      if (event.key !== ' ') return false;
      togglePause();
      return true;
    },
  };

  const rows = tableRows(points, stack === undefined ? [] : stackPoints, keys, now);
  const head = [
    'Time',
    `${props.label} (${props.unit})`,
    ...(stack === undefined ? [] : bands.map((band) => `${band.label} (${stack.unit})`)),
  ];
  const xLabels = relativeTicks(windowMs).map((tick) => ({ key: tick.text, x: x(now - tick.ago), text: tick.text }));
  const empty = !enough
    ? !props.live && points.length === 0 && stackPoints.length === 0
      ? 'Not running'
      : 'Collecting live samples…'
    : null;
  const pillLabel = paused === null ? 'Live' : fresh > 0 ? `Paused, ${String(fresh)} new` : 'Paused';
  const colorStyle = { '--c': lineColor } as CSSProperties;

  return (
    <figure
      ref={(node) => {
        figureRef.current = node;
      }}
      role="group"
      aria-label={props.label}
      className={cn(styles.root, props.className)}
      data-state={state}
      style={colorStyle}
    >
      <div className={styles.head}>
        <div className={styles.lead}>
          {stateNode !== undefined && stateNode !== null ? <div className={styles.state}>{stateNode}</div> : null}
          {readouts.map((readout) => (
            <Readout key={readout.key} readout={readout} />
          ))}
        </div>
        {props.live ? (
          <button type="button" className={styles.pill} aria-pressed={paused !== null} onClick={togglePause}>
            <span className={styles.pillDot} aria-hidden="true" />
            <PillLabel text={pillLabel} />
          </button>
        ) : null}
      </div>
      <ChartFrame
        key={enough ? 'line' : 'empty'}
        figureRef={plotRef}
        bare
        label={props.label}
        unit={props.unit}
        height={plotHeight}
        plotWidth={plotWidth}
        rightPad={6}
        yAxis={{
          ticks: ticks.map((_, index) => index),
          y: (index) => ticks[index]?.px ?? 0,
          format: (index) => ticks[index]?.text ?? '',
        }}
        xLabels={xLabels}
        table={{
          head,
          rows: rows.map((row) => ({
            key: String(row.ago),
            cells: [
              agoText(row.ago),
              row.value === null || row.value === undefined ? 'no data' : format(row.value),
              ...(stack === undefined
                ? []
                : keys.map((key) => (row.bands === null ? 'no data' : formatLanes(row.bands[key] ?? 0)))),
            ],
          })),
        }}
        scrub={empty === null ? scrub : null}
        empty={empty}
        marks={() => (
          <>
            <div className={styles.clip}>
              <div ref={slideRef} className={styles.slide}>
                <svg className={styles.marks} width={plotWidth} height={plotHeight} aria-hidden="true">
                  <g className={styles.trace} data-entrance-last={end === null ? '' : undefined}>
                    <path className={styles.area} d={area} />
                    <path className={styles.line} d={path.d} />
                    {path.lone.map((point) => (
                      <circle key={String(point.x)} className={styles.lone} cx={point.x} cy={point.y} r={2} />
                    ))}
                    {stack !== undefined ? (
                      <g data-stack="">
                        {bands.map((band, index) =>
                          stackPaths[index] === undefined || stackPaths[index] === '' ? null : (
                            <path
                              key={band.key}
                              className={styles.band}
                              d={stackPaths[index]}
                              style={{ '--band': band.color } as CSSProperties}
                              data-band={band.key}
                            />
                          ),
                        )}
                      </g>
                    ) : null}
                  </g>
                </svg>
              </div>
            </div>
            {end !== null ? (
              <svg className={styles.endLayer} width={plotWidth + 6} height={plotHeight} aria-hidden="true">
                <circle ref={dotRef} className={styles.dot} cx={end.x} cy={end.y} r={4} data-entrance-last="" />
              </svg>
            ) : null}
          </>
        )}
      />
      {stack !== undefined ? (
        <div className={styles.legendRow}>
          <span className={styles.legendTitle}>{stack.label}</span>
          <ChartLegend
            label={stack.label}
            items={bands.map((band) => ({ key: band.key, label: band.label, color: band.color, kind: 'square' }))}
          />
        </div>
      ) : null}
    </figure>
  );
}
