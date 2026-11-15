import { useRef, useState, type CSSProperties, type ReactElement } from 'react';

import { ChartFrame, plotWidthOf, type ScrubSpec } from '../chart/chart-frame';
import { ChartLegend, type LegendItem } from '../chart/chart-legend';
import type { ChartTableSpec } from '../chart/chart-table';
import { ChartTooltip, type TooltipRow } from '../chart/chart-tooltip';
import { linearScale, niceTicks } from '../chart/scale';
import type { ChartSeries } from '../chart/types';
import { useArrivals } from '../chart/use-arrivals';
import { useChartSize } from '../chart/use-chart-size';
import { defaultChartFormat, withUnit } from '../chart/value-format';
import { useRichMotion } from '../motion/use-entrance';
import styles from './scatter-chart.module.css';

export interface ScatterPoint {
  /** traceId */
  key: string;
  x: number;
  y: number;
  /** ChartSeries.key */
  series: string;
  /** Second encoding: hollow marker. */
  hollow?: boolean;
  detail?: ReadonlyArray<{ label: string; value: string }>;
}

export interface ScatterChartProps {
  label: string;
  /** "New prompt tokens" */
  xLabel: string;
  /** "tokens" */
  xUnit: string;
  /** "ms" | "s" */
  yUnit: string;
  /** Names the y value in the tooltip and the hidden table. Default "Value". */
  yLabel?: string;
  /** ≤ 1000 drawn (newest first wins). */
  points: readonly ScatterPoint[];
  series: readonly ChartSeries[];
  /** Legend entry for hollow markers. */
  hollowLabel?: string;
  /** Default 240. */
  height?: number;
  formatX?: (value: number) => string;
  formatY?: (value: number) => string;
  hiddenSeries?: readonly string[];
  onHiddenSeriesChange?: (hidden: string[]) => void;
  onSelect?: (point: ScatterPoint) => void;
  emptyLabel?: string;
}

const MAX_POINTS = 1000;
const WAVE_POINTS = 300;
const TABLE_ROWS = 200;
const HIT_PX = 12;

function overlap(previous: readonly string[], next: readonly string[]): number {
  if (next.length === 0 || previous.length === 0) return 0;
  const seen = new Set(previous);
  let shared = 0;
  for (const key of next) if (seen.has(key)) shared += 1;
  return shared / next.length;
}

const DIAMOND_HALF = 3.5;

/** The alt marker is drawn as geometry: a `transform` attribute would be shifted by the CSS transform-origin the pop animation needs. */
export function diamondPath(cx: number, cy: number): string {
  const top = `M${String(cx)} ${String(cy - DIAMOND_HALF)}`;
  const right = `L${String(cx + DIAMOND_HALF)} ${String(cy)}`;
  const bottom = `L${String(cx)} ${String(cy + DIAMOND_HALF)}`;
  const left = `L${String(cx - DIAMOND_HALF)} ${String(cy)}`;
  return `${top}${right}${bottom}${left}Z`;
}

/** Raw points on two linear axes: each turn one mark, keyed by trace id. */
export function ScatterChart(props: ScatterChartProps): ReactElement {
  const [ownHidden, setOwnHidden] = useState<readonly string[]>([]);
  const hidden = props.hiddenSeries ?? ownHidden;
  const toggle = (key: string): void => {
    const next = hidden.includes(key) ? hidden.filter((item) => item !== key) : [...hidden, key];
    props.onHiddenSeriesChange?.(next);
    if (props.hiddenSeries === undefined) setOwnHidden(next);
  };
  const keys = props.points.slice(0, MAX_POINTS).map((point) => point.key);
  const [question, setQuestion] = useState<{ keys: readonly string[]; generation: number }>({ keys, generation: 0 });
  if (question.keys !== keys && question.keys.join('\u0000') !== keys.join('\u0000')) {
    const fresh = overlap(question.keys, keys) < 0.5 && keys.length > 0;
    setQuestion({ keys, generation: fresh ? question.generation + 1 : question.generation });
  }
  return <ScatterPlot key={question.generation} {...props} hidden={hidden} onToggle={toggle} />;
}

interface ScatterPlotProps extends ScatterChartProps {
  hidden: readonly string[];
  onToggle: (key: string) => void;
}

function ScatterPlot(props: ScatterPlotProps): ReactElement {
  const figureRef = useRef<HTMLElement | null>(null);
  const width = useChartSize(figureRef);
  const rich = useRichMotion();
  const height = props.height ?? 240;
  const formatX = props.formatX ?? defaultChartFormat;
  const formatY = props.formatY ?? defaultChartFormat;
  const plotWidth = plotWidthOf(width, 8);
  const seriesByKey = new Map(props.series.map((item) => [item.key, item]));
  const points = props.points.slice(0, MAX_POINTS);
  const visible = points.filter((point) => !props.hidden.includes(point.series) && seriesByKey.has(point.series));
  const byX = [...visible].sort((a, b) => a.x - b.x || a.y - b.y);
  const rank = new Map(byX.map((point, index) => [point.key, index]));

  let maxX = 0;
  let maxY = 0;
  for (const point of visible) {
    maxX = Math.max(maxX, point.x);
    maxY = Math.max(maxY, point.y);
  }
  const xAxis = niceTicks(0, maxX > 0 ? maxX : 1);
  const yAxis = niceTicks(0, maxY > 0 ? maxY : 1);
  const x = linearScale([xAxis.min, xAxis.max], [0, plotWidth]);
  const y = linearScale([yAxis.min, yAxis.max], [height, 0]);
  const arrivals = useArrivals(
    visible.map((point) => point.key),
    rich,
  );
  const wave = visible.length <= WAVE_POINTS;

  const describe = (point: ScatterPoint): string => {
    const series = seriesByKey.get(point.series);
    const detail = (point.detail ?? []).map((item) => `${item.label} ${item.value}`);
    return [
      `${series?.label ?? point.series}: ${withUnit(formatX(point.x), props.xUnit)}, ${withUnit(formatY(point.y), props.yUnit)}`,
      ...detail,
    ].join(', ');
  };

  const scrub: ScrubSpec = {
    count: byX.length,
    anchor: (index) => {
      const point = byX[index];
      return point === undefined ? { x: 0, y: null } : { x: x(point.x), y: y(point.y) };
    },
    nearest: (px, py) => {
      let best: number | null = null;
      let bestDistance = HIT_PX * HIT_PX;
      byX.forEach((point, index) => {
        const dx = x(point.x) - px;
        const dy = y(point.y) - py;
        const distance = dx * dx + dy * dy;
        if (distance <= bestDistance) {
          bestDistance = distance;
          best = index;
        }
      });
      return best;
    },
    reading: (index) => {
      const point = byX[index];
      return point === undefined ? '' : describe(point);
    },
    tooltip: (index) => {
      const point = byX[index];
      if (point === undefined) return null;
      const series = seriesByKey.get(point.series);
      const rows: TooltipRow[] = [
        { key: 'x', label: props.xLabel, value: withUnit(formatX(point.x), props.xUnit) },
        { key: 'y', label: props.yLabel ?? 'Value', value: withUnit(formatY(point.y), props.yUnit) },
        ...(point.detail ?? []).map((item) => ({ key: item.label, label: item.label, value: item.value })),
      ];
      return <ChartTooltip title={series?.label ?? point.series} rows={rows} />;
    },
    onActivate:
      props.onSelect === undefined
        ? undefined
        : (index) => {
            const point = byX[index];
            if (point !== undefined) props.onSelect?.(point);
          },
    rest: null,
    emptyReading: `${String(visible.length)} ${visible.length === 1 ? 'point' : 'points'}`,
  };

  const tableRows = points.slice(0, TABLE_ROWS).map((point) => ({
    key: point.key,
    cells: [seriesByKey.get(point.series)?.label ?? point.series, `${formatX(point.x)}`, `${formatY(point.y)}`],
  }));
  if (points.length > TABLE_ROWS) {
    tableRows.push({
      key: '__more',
      cells: [`+${String(points.length - TABLE_ROWS)} more points. Open the Turns tab for the full list.`, '', ''],
    });
  }
  const table: ChartTableSpec = {
    head: ['Series', `${props.xLabel} (${props.xUnit})`, `${props.yLabel ?? 'Value'} (${props.yUnit})`],
    rows: tableRows,
  };

  const legendItems: LegendItem[] = props.series.map((item) => ({
    key: item.key,
    label: item.label,
    color: item.color,
    pattern: item.pattern,
    kind: 'square',
    hidden: props.hidden.includes(item.key),
    toggle: props.series.length >= 2,
  }));
  if (props.hollowLabel !== undefined) legendItems.push({ key: '__hollow', label: props.hollowLabel, kind: 'hollow' });
  const legend =
    legendItems.length >= 2 ? <ChartLegend label={props.label} items={legendItems} onToggle={props.onToggle} /> : null;

  const xLabels = xAxis.ticks.map((tick) => ({ key: String(tick), x: x(tick), text: formatX(tick) }));
  const empty = visible.length === 0 ? (props.emptyLabel ?? 'No turns in this range') : null;
  const lastKey = byX[byX.length - 1]?.key;

  return (
    <ChartFrame
      figureRef={figureRef}
      label={props.label}
      unit={props.yUnit}
      height={height}
      plotWidth={plotWidth}
      rightPad={8}
      yAxis={{ ticks: yAxis.ticks, y, format: formatY }}
      xLabels={xLabels}
      xTitle={`${props.xLabel} · ${props.xUnit}`}
      legend={legend}
      table={table}
      scrub={empty === null ? scrub : null}
      empty={empty}
      style={{ '--n': Math.max(1, Math.min(visible.length, WAVE_POINTS)) } as CSSProperties}
      marks={({ active }) => {
        const activePoint = active === null ? undefined : byX[active];
        return (
          <svg
            className={styles.points}
            width={plotWidth + 8}
            height={height}
            aria-hidden="true"
            data-wave={wave ? '' : undefined}
            data-entrance-last={wave ? undefined : ''}
          >
            {visible.map((point) => {
              const series = seriesByKey.get(point.series);
              const cx = x(point.x);
              const cy = y(point.y);
              const style = {
                '--c': series?.color ?? 'var(--series-1)',
                '--i': rank.get(point.key) ?? 0,
                transformOrigin: `${String(cx)}px ${String(cy)}px`,
              } as CSSProperties;
              const shared = {
                className: styles.point,
                style,
                'data-hollow': point.hollow === true ? '' : undefined,
                'data-arrive': arrivals.arriving.has(point.key) ? '' : undefined,
                'data-entrance-last': wave && point.key === lastKey ? '' : undefined,
                onAnimationEnd: () => arrivals.settle(point.key),
              };
              return series?.pattern === 'alt' ? (
                <path key={point.key} {...shared} d={diamondPath(cx, cy)} />
              ) : (
                <circle key={point.key} {...shared} cx={cx} cy={cy} r={3} />
              );
            })}
            {activePoint !== undefined ? (
              <circle
                className={styles.ring}
                style={{ '--c': seriesByKey.get(activePoint.series)?.color ?? 'var(--series-1)' } as CSSProperties}
                cx={x(activePoint.x)}
                cy={y(activePoint.y)}
                r={6}
              />
            ) : null}
          </svg>
        );
      }}
    />
  );
}
