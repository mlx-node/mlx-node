import { useRef, useState, type CSSProperties, type ReactElement } from 'react';

import { cn } from '../../../lib/cn';
import { LOW_SAMPLE_N } from '../../../lib/types';
import { ChartFrame, plotWidthOf, type ScrubSpec } from '../chart/chart-frame';
import { ChartLegend, type LegendItem } from '../chart/chart-legend';
import { tableCell, type ChartTableSpec } from '../chart/chart-table';
import { ChartTooltip, type TooltipRow } from '../chart/chart-tooltip';
import { resample } from '../chart/morph';
import { MorphPath } from '../chart/morph-path';
import { areaPath, bandPath, linePath, sampleXs, zipPoints, type XY } from '../chart/paths';
import { linearScale, maxOf, minOf, niceTicks, sparseLabels } from '../chart/scale';
import type { ChartPoint, ChartSeries } from '../chart/types';
import { useArrivals } from '../chart/use-arrivals';
import { useChartSize } from '../chart/use-chart-size';
import { useReshape } from '../chart/use-reshape';
import { defaultChartFormat, labelWidthPx, withUnit } from '../chart/value-format';
import { motionX } from '../motion/motion';
import { useRichMotion } from '../motion/use-entrance';
import styles from './trend-chart.module.css';

export interface TrendChartProps {
  /** Accessible name, e.g. "Decode speed over time". */
  label: string;
  /** "tok/s" */
  unit: string;
  /** ≤ 200 points. */
  data: readonly ChartPoint[];
  /** ≤ 8 drawn. */
  series: readonly ChartSeries[];
  /** Default 200. */
  height?: number;
  formatValue?: (value: number) => string;
  formatTick?: (value: number) => string;
  /** Y axis start. Default 0. 'auto' only for measures where 0 is meaningless; the subtitle must then say "axis does not start at 0". */
  yMin?: 0 | 'auto';
  curve?: 'linear' | 'monotone';
  /** Default LOW_SAMPLE_N. */
  lowSampleBelow?: number;
  hiddenSeries?: readonly string[];
  onHiddenSeriesChange?: (hidden: string[]) => void;
  /** Band behind the line, e.g. p05–p95 for a single series. */
  band?: { lowKey: string; highKey: string };
  /** Marks one x position (a ChartPoint.key): a ring on its points and aria-current="true" on its hidden-table row. */
  selectedKey?: string | null;
  /**
   * A comparison line under the marks (the previous period), aligned to `data` by index: 1.5 px dashed muted, no
   * markers, a legend key, a muted tooltip row and a second table column; not part of aria-valuetext.
   */
  ghost?: { label: string; values: readonly (number | null)[] };
  emptyLabel?: string;
  className?: string;
}

const MAX_SERIES = 8;
const MAX_POINTS = 200;
const DIRECT_LABEL_PAD = 72;
const END_PAD = 8;
const LABEL_GAP = 13;

function valueOf(point: ChartPoint | undefined, key: string): number | null {
  const value = point?.values[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/**
 * Spreads direct labels so none overlap: sorted by y, each at least LABEL_GAP below the one above; a stack that runs
 * past the bottom is pushed back up (two lines that both end at 0 stay one gap apart).
 */
export function spreadLabels(items: Array<{ key: string; y: number }>, height: number): Map<string, number> {
  const sorted = [...items].sort((a, b) => a.y - b.y);
  const ys: number[] = [];
  let last = -Infinity;
  for (const item of sorted) {
    last = Math.max(item.y, last + LABEL_GAP);
    ys.push(last);
  }
  let below = height;
  for (let i = ys.length - 1; i >= 0; i -= 1) {
    ys[i] = Math.min(ys[i], below);
    below = ys[i] - LABEL_GAP;
  }
  const placed = new Map<string, number>();
  sorted.forEach((item, index) => placed.set(item.key, Math.max(0, ys[index])));
  return placed;
}

/**
 * A line chart for one measure over time: gaps for null, low samples hollow, entrance once, reshape on new data.
 * A new column count reshapes too: lines are resampled onto the same x span, so the plot is never remounted.
 */
export function TrendChart(props: TrendChartProps): ReactElement {
  const [ownHidden, setOwnHidden] = useState<readonly string[]>([]);
  const hidden = props.hiddenSeries ?? ownHidden;
  const toggle = (key: string): void => {
    const next = hidden.includes(key) ? hidden.filter((item) => item !== key) : [...hidden, key];
    if (props.onHiddenSeriesChange !== undefined) props.onHiddenSeriesChange(next);
    if (props.hiddenSeries === undefined) setOwnHidden(next);
  };
  const data = props.data.length > MAX_POINTS ? props.data.slice(props.data.length - MAX_POINTS) : props.data;
  return <TrendPlot {...props} data={data} hidden={hidden} onToggle={toggle} />;
}

interface TrendPlotProps extends TrendChartProps {
  hidden: readonly string[];
  onToggle: (key: string) => void;
}

function TrendPlot(props: TrendPlotProps): ReactElement {
  const figureRef = useRef<HTMLElement | null>(null);
  const width = useChartSize(figureRef);
  const rich = useRichMotion();
  const height = props.height ?? 200;
  const format = props.formatValue ?? defaultChartFormat;
  const formatTick = props.formatTick ?? format;
  const lowBelow = props.lowSampleBelow ?? LOW_SAMPLE_N;
  const curve = props.curve ?? 'linear';
  const data = props.data;
  const count = data.length;
  const series = props.series.slice(0, MAX_SERIES);
  const visible = series.filter((item) => !props.hidden.includes(item.key));
  const single = series.length === 1;
  const ghost = single ? props.ghost : undefined;
  const direct = series.length >= 2 && series.length <= 4;
  const rightPad = direct ? DIRECT_LABEL_PAD : END_PAD;
  const plotWidth = plotWidthOf(width, rightPad);

  const values: (number | null)[] = [];
  for (const item of visible) for (const point of data) values.push(valueOf(point, item.key));
  if (props.band !== undefined && single) {
    for (const point of data) values.push(valueOf(point, props.band.lowKey), valueOf(point, props.band.highKey));
  }
  if (ghost !== undefined) values.push(...ghost.values);
  const high = maxOf(values);
  const low = props.yMin === 'auto' ? minOf(values) : 0;
  const axis = niceTicks(Math.min(low ?? 0, high ?? 0), high ?? 1);
  const y = linearScale([axis.min, axis.max], [height, 0]);
  const xs = data.map((_, index) => (count <= 1 ? plotWidth / 2 : (index / (count - 1)) * plotWidth));
  const step = count <= 1 ? plotWidth : plotWidth / (count - 1);

  const hasData = series.some((item) => data.some((point) => valueOf(point, item.key) !== null));
  const empty = count === 0 || !hasData ? (props.emptyLabel ?? 'No data in this range') : null;

  const signature = [
    visible.map((item) => item.key).join(','),
    String(count),
    ...visible.map((item) =>
      data
        .map((point) => {
          const value = valueOf(point, item.key);
          return value === null ? '' : format(value);
        })
        .join(','),
    ),
  ].join('|');
  const progress = useReshape(signature);
  const arrivals = useArrivals(
    visible.map((item) => item.key),
    rich,
  );

  const unitText = (value: number | null, samples?: number): string => {
    if (value === null) return 'no data';
    const text = withUnit(format(value), props.unit);
    return samples !== undefined && samples < lowBelow ? `${text}, n=${String(samples)}` : text;
  };

  const reading = (index: number): string => {
    const point = data[index];
    const parts = visible.map((item) => {
      const text = unitText(valueOf(point, item.key), point?.samples?.[item.key]);
      return series.length > 1 ? `${item.label} ${text}` : text;
    });
    return `${point?.label ?? ''}: ${parts.join(', ')}`;
  };

  const tooltip = (index: number): ReactElement => {
    const point = data[index];
    const rows: TooltipRow[] = visible.map((item) => ({
      key: item.key,
      label: item.label,
      value: unitText(valueOf(point, item.key), point?.samples?.[item.key]),
      kind: 'line',
      color: item.color,
      pattern: item.pattern,
    }));
    if (ghost !== undefined) {
      const value = ghost.values[index] ?? null;
      rows.push({ key: '__ghost', label: ghost.label, value: unitText(value), kind: 'ghost', muted: true });
    }
    return <ChartTooltip title={point?.label ?? ''} rows={rows} />;
  };

  const scrub: ScrubSpec = {
    count,
    anchor: (index) => ({ x: xs[index] ?? 0, y: null }),
    nearest: (x) => (count === 0 ? null : Math.min(count - 1, Math.max(0, Math.round(x / step)))),
    reading,
    tooltip,
    crosshair: true,
    rest: count > 0 ? count - 1 : null,
  };

  const table: ChartTableSpec = {
    head: [
      'Period',
      ...series.map((item) => `${item.label} (${props.unit})`),
      ...(ghost !== undefined ? [`${ghost.label} (${props.unit})`] : []),
    ],
    rows: data.map((point, index) => ({
      key: point.key,
      current: props.selectedKey === point.key,
      cells: [
        point.label,
        ...series.map((item) => tableCell(valueOf(point, item.key), format, point.samples?.[item.key], lowBelow)),
        ...(ghost !== undefined ? [tableCell(ghost.values[index] ?? null, format)] : []),
      ],
    })),
  };

  const legendItems: LegendItem[] = series.map((item) => ({
    key: item.key,
    label: item.label,
    color: item.color,
    pattern: item.pattern,
    kind: 'line',
    hidden: props.hidden.includes(item.key),
    toggle: series.length >= 2,
  }));
  if (ghost !== undefined) legendItems.push({ key: '__ghost', label: ghost.label, kind: 'ghost' });
  const legend =
    series.length >= 2 || ghost !== undefined ? (
      <ChartLegend label={props.label} items={legendItems} onToggle={props.onToggle} />
    ) : null;

  const labelled = data
    .map((point, index) => ({ point, index }))
    .filter((entry) => entry.point.axisLabel !== undefined && entry.point.axisLabel.length > 0);
  const widest = labelled.reduce((max, entry) => Math.max(max, labelWidthPx(entry.point.axisLabel ?? '')), 0);
  const keep = new Set(sparseLabels(labelled.length, plotWidth * (labelled.length / Math.max(1, count)), widest));
  const xLabels = labelled
    .filter((_, position) => keep.has(position))
    .map((entry) => ({ key: entry.point.key, x: xs[entry.index] ?? 0, text: entry.point.axisLabel ?? '' }));

  const sampleCount = motionX.morphSamples;
  const sx = sampleXs(sampleCount, xs[0] ?? 0, xs[count - 1] ?? 0);

  return (
    <ChartFrame
      figureRef={figureRef}
      label={props.label}
      unit={props.unit}
      height={height}
      plotWidth={plotWidth}
      rightPad={rightPad}
      yAxis={{ ticks: axis.ticks, y, format: formatTick }}
      xLabels={xLabels}
      legend={legend}
      table={table}
      scrub={empty === null ? scrub : null}
      empty={empty}
      className={cn(styles.root, props.className)}
      marks={({ active }) => {
        const lines = visible.map((item, order) => {
          const ys = data.map((point) => {
            const value = valueOf(point, item.key);
            return value === null ? null : y(value);
          });
          const points = zipPoints(xs, ys);
          const path = linePath(points, curve);
          const samples = resample(ys, sampleCount);
          const style = { '--c': item.color } as CSSProperties;
          return { item, order, ys, points, path, samples, style };
        });
        const ends = lines
          .map((line) => {
            for (let i = line.points.length - 1; i >= 0; i -= 1) {
              const point = line.points[i];
              if (point !== null) return { key: line.item.key, point, style: line.style };
            }
            return null;
          })
          .filter((end): end is { key: string; point: XY; style: CSSProperties } => end !== null);
        const bandD =
          props.band !== undefined && single
            ? bandPath(
                xs,
                data.map((point) => {
                  const value = valueOf(point, props.band?.lowKey ?? '');
                  return value === null ? null : y(value);
                }),
                data.map((point) => {
                  const value = valueOf(point, props.band?.highKey ?? '');
                  return value === null ? null : y(value);
                }),
              )
            : '';
        const ghostD =
          ghost !== undefined
            ? linePath(
                zipPoints(
                  xs,
                  ghost.values.slice(0, count).map((value) => (value === null ? null : y(value))),
                ),
              )
            : null;
        const selectedIndex =
          props.selectedKey === undefined || props.selectedKey === null
            ? -1
            : data.findIndex((point) => point.key === props.selectedKey);
        const directLabels = direct
          ? spreadLabels(
              ends.map((end) => ({ key: end.key, y: end.point.y })),
              height,
            )
          : null;
        return (
          <>
            <svg className={styles.under} width={plotWidth + rightPad} height={height} aria-hidden="true">
              {bandD.length > 0 ? (
                <path
                  className={styles.band}
                  d={bandD}
                  style={{ '--c': series[0]?.color ?? 'var(--series-1)' } as CSSProperties}
                />
              ) : null}
              {ghostD !== null ? (
                <>
                  <path className={styles.ghost} d={ghostD.d} />
                  {ghostD.lone.map((point) => (
                    <circle key={`${point.x}`} className={styles.ghostDot} cx={point.x} cy={point.y} r={1.5} />
                  ))}
                </>
              ) : null}
            </svg>
            <svg
              className={styles.lines}
              width={plotWidth + rightPad}
              height={height}
              aria-hidden="true"
              data-entrance-last={ends.length === 0 ? '' : undefined}
            >
              {lines.length === 1 ? (
                rich ? (
                  <MorphPath
                    className={styles.area}
                    style={lines[0].style}
                    exact={areaPath(lines[0].points, height)}
                    samples={lines[0].samples}
                    toPath={(mixed) => areaPath(zipPoints(sx, mixed), height)}
                    signature={signature}
                    progress={progress}
                  />
                ) : (
                  <path className={styles.area} d={areaPath(lines[0].points, height)} style={lines[0].style} />
                )
              ) : null}
              {lines.map((line) => {
                const arriving = arrivals.arriving.has(line.item.key);
                return (
                  <g
                    key={line.item.key}
                    style={line.style}
                    data-pattern={line.item.pattern}
                    data-arrive={arriving ? '' : undefined}
                    className={styles.series}
                    onAnimationEnd={(event) => {
                      if (event.target === event.currentTarget) arrivals.settle(line.item.key);
                    }}
                  >
                    {rich ? (
                      <MorphPath
                        className={styles.line}
                        exact={line.path.d}
                        samples={line.samples}
                        toPath={(mixed) => linePath(zipPoints(sx, mixed), curve).d}
                        signature={signature}
                        progress={progress}
                      />
                    ) : (
                      <path className={styles.line} d={line.path.d} />
                    )}
                    {line.path.lone.map((point) => (
                      <circle key={`${point.x}`} className={styles.lone} cx={point.x} cy={point.y} r={2.5} />
                    ))}
                  </g>
                );
              })}
            </svg>
            <svg className={styles.markers} width={plotWidth + rightPad} height={height} aria-hidden="true">
              {lines.map((line) =>
                data.map((point, index) => {
                  const samples = point.samples?.[line.item.key];
                  const at = line.points[index];
                  if (at === null || at === undefined || samples === undefined || samples >= lowBelow) return null;
                  return (
                    <circle
                      key={`${line.item.key}-low-${String(index)}`}
                      className={styles.hollow}
                      style={line.style}
                      cx={at.x}
                      cy={at.y}
                      r={4}
                    />
                  );
                }),
              )}
              {selectedIndex >= 0
                ? lines.map((line) => {
                    const at = line.points[selectedIndex];
                    if (at === null || at === undefined) return null;
                    return (
                      <circle
                        key={`${line.item.key}-selected`}
                        className={styles.selected}
                        style={line.style}
                        cx={at.x}
                        cy={at.y}
                        r={6}
                      />
                    );
                  })
                : null}
              {ends.map((end, index) => (
                <circle
                  key={`${end.key}-end`}
                  className={styles.end}
                  style={end.style}
                  cx={end.point.x}
                  cy={end.point.y}
                  r={4}
                  data-entrance-last={index === ends.length - 1 ? '' : undefined}
                />
              ))}
              {active !== null
                ? lines.map((line) => {
                    const at = line.points[active];
                    if (at === null || at === undefined) return null;
                    return (
                      <circle
                        key={`${line.item.key}-active`}
                        className={styles.hover}
                        style={line.style}
                        cx={at.x}
                        cy={at.y}
                        r={4}
                      />
                    );
                  })
                : null}
            </svg>
            {directLabels !== null ? (
              <div className={styles.direct} aria-hidden="true">
                {lines.map((line) => {
                  const at = directLabels.get(line.item.key);
                  if (at === undefined) return null;
                  return (
                    <span
                      key={line.item.key}
                      className={styles.directLabel}
                      style={{ ...line.style, left: plotWidth + 10, top: at }}
                    >
                      <span className={styles.directText}>{line.item.label}</span>
                    </span>
                  );
                })}
              </div>
            ) : null}
          </>
        );
      }}
    />
  );
}
