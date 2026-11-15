import { motion } from 'motion/react';
import { useLayoutEffect, useRef, useState, type CSSProperties, type KeyboardEvent, type ReactElement } from 'react';

import { cn } from '../../../lib/cn';
import { formatTps } from '../../../lib/format';
import { LIVE_GAP_MS } from '../../../lib/live-history';
import { ChartFrame, plotWidthOf, type ScrubSpec } from '../chart/chart-frame';
import { ChartTooltip } from '../chart/chart-tooltip';
import { areaPath, linePath, type XY } from '../chart/paths';
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

export interface LiveStreamProps {
  /** "Output rate" */
  label: string;
  /** "tok/s" */
  unit: string;
  /** Ascending t, ≤ 600. */
  points: readonly LiveStreamPoint[];
  /** Right edge of the window. The newest snapshot time, not Date.now(). */
  nowMs: number;
  /** Default 600_000. */
  windowMs?: number;
  /** False = the source stopped (no running process). */
  live: boolean;
  /** Text under the value when live is false or the value is 0: "Idle", "Not running". */
  status?: string;
  /** Default formatTps. */
  formatValue?: (value: number) => string;
  /** Y tick text; default formatValue. */
  formatTick?: (value: number) => string;
  /** Plot height, default 160. */
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

/** Last value in each 10 s step, newest first, at most 60 rows. */
function tableRows(points: readonly LiveStreamPoint[], now: number): Array<{ ago: number; value: number | null }> {
  const rows: Array<{ ago: number; value: number | null }> = [];
  let bucket = -1;
  for (let i = points.length - 1; i >= 0 && rows.length < TABLE_ROWS; i -= 1) {
    const point = points[i];
    const step = Math.floor(Math.max(0, now - point.t) / TABLE_STEP_MS);
    if (step === bucket) continue;
    bucket = step;
    rows.push({ ago: now - point.t, value: point.value });
  }
  return rows;
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

/** One live measure: a big current value and a line that slides left as time passes. */
export function LiveStream(props: LiveStreamProps): ReactElement {
  const figureRef = useRef<HTMLElement | null>(null);
  const plotRef = useRef<HTMLElement | null>(null);
  const width = useChartSize(figureRef);
  const rich = useRichMotion();
  const windowMs = props.windowMs ?? 600_000;
  const gapMs = props.gapMs ?? LIVE_GAP_MS;
  const height = props.height ?? 160;
  const format = props.formatValue ?? ((value: number) => formatTps(value));
  const [paused, setPaused] = useState<{ now: number; points: readonly LiveStreamPoint[] } | null>(null);
  const now = paused?.now ?? props.nowMs;
  const points = paused?.points ?? props.points;
  const fresh = paused === null ? 0 : props.points.filter((point) => point.t > paused.now).length;
  const state = !props.live ? 'stopped' : paused !== null ? 'paused' : 'live';

  let newest: LiveStreamPoint | null = null;
  for (let i = points.length - 1; i >= 0 && newest === null; i -= 1) if (points[i].value !== null) newest = points[i];
  const valueText = props.live && newest !== null && newest.value !== null ? format(newest.value) : null;
  const showStatus = props.status !== undefined && (!props.live || newest?.value === 0);

  const plotWidth = plotWidthOf(width, 6);
  const inWindow = points.filter((point) => point.t >= now - windowMs && point.t <= now);
  let peak = 0;
  for (const point of inWindow) if (point.value !== null) peak = Math.max(peak, point.value);
  const axis = niceTicks(0, peak > 0 ? peak : 1);
  const y = linearScale([axis.min, axis.max], [height, 0]);
  const x = (t: number): number => plotWidth * (1 - (now - t) / windowMs);
  const drawn = plotPoints(points, now, windowMs, gapMs, x, y);
  const path = linePath(drawn);
  const area = areaPath(drawn, height);
  let end: XY | null = null;
  for (let i = drawn.length - 1; i >= 0 && end === null; i -= 1) end = drawn[i] ?? null;
  const enough = inWindow.length >= 2;

  const togglePause = (): void => {
    setPaused((current) => (current === null ? { now: props.nowMs, points: props.points } : null));
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

  const scrubPoints = inWindow.filter((point) => point.value !== null);
  const scrub: ScrubSpec = {
    count: scrubPoints.length,
    anchor: (index) => ({ x: x(scrubPoints[index]?.t ?? now), y: null }),
    nearest: (px) => {
      let best: number | null = null;
      let bestDistance = Infinity;
      scrubPoints.forEach((point, index) => {
        const distance = Math.abs(x(point.t) - px);
        if (distance < bestDistance) {
          bestDistance = distance;
          best = index;
        }
      });
      return best;
    },
    reading: (index) => {
      const point = scrubPoints[index];
      if (point === undefined || point.value === null) return 'no data';
      return `${withUnit(format(point.value), props.unit)}, ${agoText(now - point.t)}`;
    },
    tooltip: (index) => {
      const point = scrubPoints[index];
      if (point === undefined || point.value === null) return null;
      return (
        <ChartTooltip
          title={clockAgo(now - point.t)}
          rows={[{ key: 'value', label: props.label, value: withUnit(format(point.value), props.unit) }]}
        />
      );
    },
    crosshair: true,
    rest: scrubPoints.length > 0 ? scrubPoints.length - 1 : null,
    emptyReading: 'Collecting live samples',
    onKey: (event: KeyboardEvent<HTMLDivElement>) => {
      if (event.key !== ' ') return false;
      togglePause();
      return true;
    },
  };

  const rows = tableRows(points, now);
  const xLabels = relativeTicks(windowMs).map((tick) => ({ key: tick.text, x: x(now - tick.ago), text: tick.text }));
  const empty = !enough
    ? !props.live && points.length === 0
      ? (props.status ?? 'Not running')
      : 'Collecting live samples…'
    : null;
  const pillLabel = paused === null ? 'Live' : fresh > 0 ? `Paused, ${String(fresh)} new` : 'Paused';
  const colorStyle = { '--c': props.color ?? 'var(--series-1)' } as CSSProperties;

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
        <div className={styles.reading}>
          <span className={styles.valueLine} data-stat-value="">
            <NumberValue text={valueText} className={styles.value} />
            {valueText !== null ? (
              <>
                {' '}
                <span className={styles.unit}>{props.unit}</span>
              </>
            ) : null}
          </span>
          {showStatus ? <span className={styles.status}>{props.status}</span> : null}
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
        height={height}
        plotWidth={plotWidth}
        rightPad={6}
        yAxis={{ ticks: axis.ticks, y, format: props.formatTick ?? ((value) => format(value)) }}
        xLabels={xLabels}
        table={{
          head: ['Time', `${props.label} (${props.unit})`],
          rows: rows.map((row) => ({
            key: String(row.ago),
            cells: [agoText(row.ago), row.value === null ? 'no data' : format(row.value)],
          })),
        }}
        scrub={empty === null ? scrub : null}
        empty={empty}
        marks={() => (
          <>
            <div className={styles.clip}>
              <div ref={slideRef} className={styles.slide}>
                <svg className={styles.marks} width={plotWidth} height={height} aria-hidden="true">
                  <g className={styles.trace} data-entrance-last={end === null ? '' : undefined}>
                    <path className={styles.area} d={area} />
                    <path className={styles.line} d={path.d} />
                    {path.lone.map((point) => (
                      <circle key={String(point.x)} className={styles.lone} cx={point.x} cy={point.y} r={2} />
                    ))}
                  </g>
                </svg>
              </div>
            </div>
            {end !== null ? (
              <svg className={styles.endLayer} width={plotWidth + 6} height={height} aria-hidden="true">
                <circle ref={dotRef} className={styles.dot} cx={end.x} cy={end.y} r={4} data-entrance-last="" />
              </svg>
            ) : null}
          </>
        )}
      />
    </figure>
  );
}
