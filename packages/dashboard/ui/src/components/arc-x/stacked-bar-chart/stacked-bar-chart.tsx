import { useEffect, useId, useRef, useState, type CSSProperties, type ReactElement } from 'react';

import { cn } from '../../../lib/cn';
import { LOW_SAMPLE_N } from '../../../lib/types';
import { ChartFrame, plotWidthOf, type ScrubSpec } from '../chart/chart-frame';
import { ChartLegend, type LegendItem } from '../chart/chart-legend';
import { tableCell, type ChartTableSpec } from '../chart/chart-table';
import { ChartTooltip, type TooltipRow } from '../chart/chart-tooltip';
import type { Rect } from '../chart/morph';
import { barPath, svgId } from '../chart/paths';
import { ReshapeBar } from '../chart/reshape-bar';
import { bandIndex, bandLayout, linearScale, niceTicks, sparseLabels } from '../chart/scale';
import type { ChartPoint, ChartSeries } from '../chart/types';
import { useChartSize } from '../chart/use-chart-size';
import { useReshape } from '../chart/use-reshape';
import { defaultChartFormat, labelWidthPx, withUnit } from '../chart/value-format';
import { useRichMotion } from '../motion/use-entrance';
import styles from './stacked-bar-chart.module.css';

export interface StackedBarChartProps {
  label: string;
  /** "tokens" */
  unit: string;
  /** ≤ 90 columns. */
  data: readonly ChartPoint[];
  /** Stack order = array order, bottom first. Every series is drawn: a fold series ('Other') sums into the totals. */
  series: readonly ChartSeries[];
  /** Default 'stacked'. */
  mode?: 'stacked' | 'grouped';
  /** Default 200. */
  height?: number;
  formatValue?: (value: number) => string;
  formatTick?: (value: number) => string;
  lowSampleBelow?: number;
  /** Click / Enter on a column. */
  onSelect?: (point: ChartPoint) => void;
  /** Marks one column: a 2 px accent tick under it and aria-current="true" on its hidden-table row. */
  selectedKey?: string | null;
  emptyLabel?: string;
  className?: string;
}

const MAX_COLUMNS = 90;
const BAR_RADIUS = 4;
const STACK_GAP = 2;
const GROUP_GAP = 8;
const MAX_BAR = 24;

/**
 * Grouped mode: `slots` bars side by side per column. Both gaps shrink with the column, so 30 days of 2 series in a
 * narrow card keep bars about 3 px wide instead of 1 px needles.
 */
export function groupedLayout(
  count: number,
  plotWidth: number,
  series: number,
): { layout: ReturnType<typeof bandLayout>; barWidth: number; innerGap: number } {
  const slots = Math.max(1, series);
  const groupGap = count > 0 ? Math.min(GROUP_GAP, (plotWidth / count) * 0.25) : GROUP_GAP;
  const layout = bandLayout(count, plotWidth, slots * (MAX_BAR + STACK_GAP), groupGap);
  const innerGap = Math.min(STACK_GAP, layout.band * 0.1);
  return { layout, barWidth: Math.max(1, (layout.band - innerGap * (slots - 1)) / slots), innerGap };
}

interface Bar {
  key: string;
  series: ChartSeries;
  seriesIndex: number;
  rect: Rect;
  d: string;
}

function valueOf(point: ChartPoint | undefined, key: string): number | null {
  const value = point?.values[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** Bars per bucket, stacked (bottom first) or grouped; null is an empty slot with a dash on the baseline. */
export function StackedBarChart(props: StackedBarChartProps): ReactElement {
  const [hidden, setHidden] = useState<readonly string[]>([]);
  const toggle = (key: string): void =>
    setHidden((current) => (current.includes(key) ? current.filter((item) => item !== key) : [...current, key]));
  const data = props.data.length > MAX_COLUMNS ? props.data.slice(props.data.length - MAX_COLUMNS) : props.data;
  return (
    <BarPlot
      key={`${String(data.length)}-${props.mode ?? 'stacked'}`}
      {...props}
      data={data}
      hidden={hidden}
      onToggle={toggle}
    />
  );
}

interface BarPlotProps extends StackedBarChartProps {
  hidden: readonly string[];
  onToggle: (key: string) => void;
}

function BarPlot(props: BarPlotProps): ReactElement {
  const figureRef = useRef<HTMLElement | null>(null);
  const width = useChartSize(figureRef);
  const rich = useRichMotion();
  const uid = svgId(useId());
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const height = props.height ?? 200;
  const format = props.formatValue ?? defaultChartFormat;
  const formatTick = props.formatTick ?? format;
  const lowBelow = props.lowSampleBelow ?? LOW_SAMPLE_N;
  const mode = props.mode ?? 'stacked';
  const data = props.data;
  const count = data.length;
  const series = props.series;
  const visible = series.filter((item) => !props.hidden.includes(item.key));
  const plotWidth = plotWidthOf(width, 4);

  let high = 0;
  for (const point of data) {
    if (mode === 'stacked') {
      let sum = 0;
      for (const item of visible) sum += Math.max(0, valueOf(point, item.key) ?? 0);
      high = Math.max(high, sum);
    } else {
      for (const item of visible) high = Math.max(high, valueOf(point, item.key) ?? 0);
    }
  }
  const axis = niceTicks(0, high > 0 ? high : 1);
  const y = linearScale([axis.min, axis.max], [height, 0]);
  const grouped = groupedLayout(count, plotWidth, visible.length);
  const layout = mode === 'stacked' ? bandLayout(count, plotWidth, MAX_BAR, STACK_GAP) : grouped.layout;

  const columns = data.map((point, index) => {
    const x = index * layout.step + layout.offset;
    const bars: Bar[] = [];
    if (mode === 'stacked') {
      const parts = visible
        .map((item) => ({ item, value: valueOf(point, item.key) }))
        .filter((part): part is { item: ChartSeries; value: number } => part.value !== null && part.value > 0);
      let sum = 0;
      parts.forEach((part, order) => {
        const bottom = y(sum) - (order > 0 ? STACK_GAP / 2 : 0);
        sum += part.value;
        const top = y(sum) + (order < parts.length - 1 ? STACK_GAP / 2 : 0);
        if (bottom - top <= 0) return;
        const rect = { x, y: top, width: layout.band, height: bottom - top };
        const radius = order === parts.length - 1 ? BAR_RADIUS : 0;
        bars.push({
          key: part.item.key,
          series: part.item,
          seriesIndex: series.indexOf(part.item),
          rect,
          d: barPath(x, top, layout.band, bottom, radius),
        });
      });
    } else {
      const { barWidth, innerGap } = grouped;
      visible.forEach((item, slot) => {
        const value = valueOf(point, item.key);
        if (value === null || value <= 0) return;
        const left = x + slot * (barWidth + innerGap);
        const top = y(value);
        bars.push({
          key: item.key,
          series: item,
          seriesIndex: series.indexOf(item),
          rect: { x: left, y: top, width: barWidth, height: height - top },
          d: barPath(left, top, barWidth, height, BAR_RADIUS),
        });
      });
    }
    const allNull = visible.every((item) => valueOf(point, item.key) === null);
    const top = bars.reduce((min, bar) => Math.min(min, bar.rect.y), height);
    return { point, index, x, bars, allNull, top };
  });

  const hasData = data.some((point) => series.some((item) => valueOf(point, item.key) !== null));
  const empty = count === 0 || !hasData ? (props.emptyLabel ?? 'No data in this range') : null;
  const signature = [
    visible.map((item) => item.key).join(','),
    String(count),
    String(Math.round(axis.max * 1000)),
    ...data.map((point) =>
      visible
        .map((item) => {
          const value = valueOf(point, item.key);
          return value === null ? '' : format(value);
        })
        .join(','),
    ),
  ].join('|');
  const progress = useReshape(signature);

  const unitText = (value: number | null, samples?: number): string => {
    if (value === null) return 'no data';
    const text = withUnit(format(value), props.unit);
    return samples !== undefined && samples < lowBelow ? `${text}, n=${String(samples)}` : text;
  };
  const totalOf = (point: ChartPoint | undefined): number | null => {
    let sum: number | null = null;
    for (const item of visible) {
      const value = valueOf(point, item.key);
      if (value !== null) sum = (sum ?? 0) + value;
    }
    return sum;
  };
  const withTotal = mode === 'stacked' && series.length >= 2;

  const scrub: ScrubSpec = {
    count,
    anchor: (index) => ({
      x: (columns[index]?.x ?? 0) + layout.band / 2,
      y: columns[index]?.bars.length ? (columns[index]?.top ?? null) : null,
    }),
    nearest: (x) => (count === 0 ? null : bandIndex(layout, count, x)),
    reading: (index) => {
      const point = data[index];
      const parts = visible.map((item) => {
        const text = unitText(valueOf(point, item.key), point?.samples?.[item.key]);
        return series.length > 1 ? `${item.label} ${text}` : text;
      });
      if (withTotal) parts.push(`total ${unitText(totalOf(point))}`);
      return `${point?.label ?? ''}: ${parts.join(', ')}`;
    },
    tooltip: (index) => {
      const point = data[index];
      const rows: TooltipRow[] = [...visible].reverse().map((item) => ({
        key: item.key,
        label: item.label,
        value: unitText(valueOf(point, item.key), point?.samples?.[item.key]),
        kind: 'square',
        color: item.color,
        pattern: item.pattern,
      }));
      if (withTotal) rows.push({ key: '__total', label: 'Total', value: unitText(totalOf(point)) });
      return <ChartTooltip title={point?.label ?? ''} rows={rows} />;
    },
    onActivate:
      props.onSelect === undefined
        ? undefined
        : (index) => {
            const point = data[index];
            if (point !== undefined) props.onSelect?.(point);
          },
    rest: count > 0 ? count - 1 : null,
  };

  const table: ChartTableSpec = {
    head: ['Period', ...series.map((item) => `${item.label} (${props.unit})`), ...(withTotal ? ['Total'] : [])],
    rows: data.map((point) => ({
      key: point.key,
      current: props.selectedKey === point.key,
      cells: [
        point.label,
        ...series.map((item) => tableCell(valueOf(point, item.key), format, point.samples?.[item.key], lowBelow)),
        ...(withTotal ? [tableCell(totalOf(point), format)] : []),
      ],
    })),
  };

  const legend =
    series.length >= 2 ? (
      <ChartLegend
        label={props.label}
        onToggle={props.onToggle}
        items={series.map((item): LegendItem => ({
          key: item.key,
          label: item.label,
          color: item.color,
          pattern: item.pattern,
          kind: 'square',
          hidden: props.hidden.includes(item.key),
          toggle: true,
        }))}
      />
    ) : null;

  const labelled = columns.filter((column) => (column.point.axisLabel ?? '').length > 0);
  const widest = labelled.reduce((max, column) => Math.max(max, labelWidthPx(column.point.axisLabel ?? '')), 0);
  const keep = new Set(sparseLabels(labelled.length, plotWidth * (labelled.length / Math.max(1, count)), widest));
  const xLabels = labelled
    .filter((_, position) => keep.has(position))
    .map((column) => ({ key: column.point.key, x: column.x + layout.band / 2, text: column.point.axisLabel ?? '' }));

  const columnStyle = (index: number, x: number): CSSProperties =>
    ({
      '--i': index,
      '--n': Math.max(1, count),
      transformOrigin: `${String(x)}px ${String(height)}px`,
    }) as CSSProperties;

  return (
    <ChartFrame
      figureRef={figureRef}
      label={props.label}
      unit={props.unit}
      height={height}
      plotWidth={plotWidth}
      rightPad={4}
      yAxis={{ ticks: axis.ticks, y, format: formatTick }}
      xLabels={xLabels}
      legend={legend}
      table={table}
      scrub={empty === null ? scrub : null}
      empty={empty}
      className={cn(styles.root, props.className)}
      marks={({ active }) => (
        <svg className={styles.bars} width={plotWidth + 4} height={height + 6} aria-hidden="true">
          <defs>
            {series.map((item, index) =>
              item.pattern === 'alt' ? (
                <pattern
                  key={item.key}
                  id={`${uid}-h${String(index)}`}
                  width={5}
                  height={5}
                  patternUnits="userSpaceOnUse"
                  patternTransform="rotate(45)"
                  style={{ '--c': item.color } as CSSProperties}
                >
                  <rect className={styles.hatchBase} width={5} height={5} />
                  <rect className={styles.hatchLine} width={1} height={5} />
                </pattern>
              ) : null,
            )}
          </defs>
          {columns.map((column) => (
            <g
              key={column.index}
              className={styles.column}
              style={columnStyle(column.index, column.x)}
              data-active={active === column.index ? '' : undefined}
              data-dim={active !== null && active !== column.index ? '' : undefined}
              data-entrance-last={column.index === count - 1 ? '' : undefined}
            >
              {column.bars.map((bar) => {
                const alt = bar.series.pattern === 'alt';
                const style = {
                  '--c': bar.series.color,
                  ...(alt ? { fill: `url("#${uid}-h${String(bar.seriesIndex)}")` } : {}),
                } as CSSProperties;
                const born = mounted ? { x: bar.rect.x, y: height, width: bar.rect.width, height: 0 } : undefined;
                return rich ? (
                  <ReshapeBar
                    key={bar.key}
                    d={bar.d}
                    rect={bar.rect}
                    bornFrom={born}
                    signature={signature}
                    progress={progress}
                    className={alt ? cn(styles.bar, styles.alt) : styles.bar}
                    style={style}
                  />
                ) : (
                  <path
                    key={bar.key}
                    d={bar.d}
                    className={alt ? cn(styles.bar, styles.alt) : styles.bar}
                    style={style}
                  />
                );
              })}
              {column.allNull ? (
                <rect className={styles.dash} x={column.x + layout.band / 2 - 3} y={height - 2} width={6} height={2} />
              ) : null}
              {props.selectedKey === column.point.key ? (
                <rect className={styles.selected} x={column.x} y={height + 3} width={layout.band} height={2} />
              ) : null}
            </g>
          ))}
        </svg>
      )}
    />
  );
}
