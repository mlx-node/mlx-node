import { useEffect, useId, useRef, useState, type CSSProperties, type ReactElement } from 'react';

import { cn } from '../../../lib/cn';
import { formatPercent } from '../../../lib/format';
import { ChartFrame, plotWidthOf, type ScrubSpec } from '../chart/chart-frame';
import type { ChartTableSpec } from '../chart/chart-table';
import { ChartTooltip, type TooltipRow } from '../chart/chart-tooltip';
import { barPath, svgId } from '../chart/paths';
import { ReshapeBar } from '../chart/reshape-bar';
import { bandIndex, bandLayout, linearScale, niceTicks, sparseLabels } from '../chart/scale';
import { useChartSize } from '../chart/use-chart-size';
import { useReshape } from '../chart/use-reshape';
import { defaultChartFormat, labelWidthPx, withUnit } from '../chart/value-format';
import { useRichMotion } from '../motion/use-entrance';
import styles from './histogram.module.css';

/** to = Infinity -> the overflow bin. */
export interface HistogramBin {
  from: number;
  to: number;
  count: number;
}

export interface HistogramProps {
  label: string;
  /** Unit of the x axis ("ms"); '' when `formatBound` prints its own unit (durations that switch ms / s / min). */
  unit: string;
  /** { one: 'turn', other: 'turns' } */
  countLabel: { one: string; other: string };
  /** ≤ 60, contiguous, precomputed by the API. */
  bins: readonly HistogramBin[];
  /** Vertical reference rules, e.g. p50 / p95. */
  markers?: ReadonlyArray<{ value: number; label: string }>;
  /** Default 160. */
  height?: number;
  formatBound?: (value: number) => string;
  /** Default var(--series-1). */
  color?: string;
  emptyLabel?: string;
}

const MAX_BINS = 60;
const BAR_RADIUS = 4;
const MARKER_LABEL_PX = 44;

/** Counts per fixed bin; bars touch with a 2 px gap; the open last bin is hatched and labelled "≥ x". */
export function Histogram(props: HistogramProps): ReactElement {
  const bins = props.bins.slice(0, MAX_BINS);
  return <HistogramPlot key={String(bins.length)} {...props} bins={bins} />;
}

function HistogramPlot(props: HistogramProps): ReactElement {
  const figureRef = useRef<HTMLElement | null>(null);
  const width = useChartSize(figureRef);
  const rich = useRichMotion();
  const uid = svgId(useId());
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const height = props.height ?? 160;
  const formatBound = props.formatBound ?? defaultChartFormat;
  const color = props.color ?? 'var(--series-1)';
  const bins = props.bins;
  const count = bins.length;
  const plotWidth = plotWidthOf(width, 4);
  const total = bins.reduce((sum, bin) => sum + Math.max(0, bin.count), 0);
  const high = bins.reduce((max, bin) => Math.max(max, bin.count), 0);
  const axis = niceTicks(0, high > 0 ? high : 1);
  const y = linearScale([axis.min, axis.max], [height, 0]);
  const layout = bandLayout(count, plotWidth, Number.POSITIVE_INFINITY, 2);
  const countText = (n: number): string =>
    `${defaultChartFormat(n)} ${n === 1 ? props.countLabel.one : props.countLabel.other}`;
  const rangeText = (bin: HistogramBin): string =>
    Number.isFinite(bin.to)
      ? withUnit(`${formatBound(bin.from)}–${formatBound(bin.to)}`, props.unit)
      : withUnit(`≥ ${formatBound(bin.from)}`, props.unit);
  const share = (bin: HistogramBin): string => (total > 0 ? formatPercent(bin.count / total) : '—');

  const xOfValue = (value: number): number | null => {
    for (let i = 0; i < count; i += 1) {
      const bin = bins[i];
      if (value < bin.from) continue;
      const left = i * layout.step;
      if (!Number.isFinite(bin.to)) return left + layout.step / 2;
      if (value < bin.to || i === count - 1) {
        const span = bin.to - bin.from;
        return left + (span > 0 ? Math.min(1, (value - bin.from) / span) : 0.5) * layout.step;
      }
    }
    return count > 0 && value < bins[0].from ? 0 : null;
  };
  const markers = (props.markers ?? [])
    .map((marker) => ({ ...marker, x: xOfValue(marker.value) }))
    .filter((marker): marker is { value: number; label: string; x: number } => marker.x !== null);
  let lastLabel = -Infinity;
  const labelled = new Set<number>();
  markers.forEach((marker, index) => {
    if (marker.x - lastLabel >= MARKER_LABEL_PX) {
      labelled.add(index);
      lastLabel = marker.x;
    }
  });

  const empty = count === 0 || total === 0 ? (props.emptyLabel ?? 'No data in this range') : null;
  const signature = `${String(count)}|${String(axis.max)}|${bins.map((bin) => String(bin.count)).join(',')}`;
  const progress = useReshape(signature);

  const markersIn = (index: number): string[] => {
    const left = index * layout.step;
    return markers
      .filter((marker) => marker.x >= left && marker.x < left + layout.step)
      .map((marker) => `${marker.label} ${withUnit(formatBound(marker.value), props.unit)}`);
  };

  const scrub: ScrubSpec = {
    count,
    anchor: (index) => ({ x: index * layout.step + layout.step / 2, y: y(bins[index]?.count ?? 0) }),
    nearest: (x) => (count === 0 ? null : bandIndex(layout, count, x)),
    reading: (index) => {
      const bin = bins[index];
      return bin === undefined ? '' : `${rangeText(bin)}: ${countText(bin.count)}, ${share(bin)}`;
    },
    tooltip: (index) => {
      const bin = bins[index];
      if (bin === undefined) return null;
      const rows: TooltipRow[] = [
        { key: 'count', label: props.countLabel.other, value: defaultChartFormat(bin.count), kind: 'square', color },
        { key: 'share', label: 'Share', value: share(bin) },
      ];
      const inside = markersIn(index);
      return (
        <ChartTooltip title={rangeText(bin)} rows={rows} note={inside.length > 0 ? inside.join(' · ') : undefined} />
      );
    },
    rest: null,
    emptyReading: `${countText(total)} in ${String(count)} bins`,
  };

  const table: ChartTableSpec = {
    head: [props.unit === '' ? 'Range' : `Range (${props.unit})`, 'Count', 'Share'],
    rows: bins.map((bin, index) => ({
      key: String(index),
      cells: [rangeText(bin), defaultChartFormat(bin.count), share(bin)],
    })),
  };

  const edges = bins.map((bin, index) => ({
    key: String(index),
    x: index * layout.step,
    text: Number.isFinite(bin.to) || index === 0 ? formatBound(bin.from) : `≥ ${formatBound(bin.from)}`,
  }));
  const widest = edges.reduce((max, edge) => Math.max(max, labelWidthPx(edge.text)), 0);
  const keep = new Set(sparseLabels(count, plotWidth, widest));
  const xLabels = edges.filter((_, index) => keep.has(index));

  const barStyle = { '--c': color } as CSSProperties;
  const lastIsMarker = markers.length > 0;

  return (
    <ChartFrame
      figureRef={figureRef}
      label={props.label}
      unit={props.countLabel.other}
      height={height}
      plotWidth={plotWidth}
      rightPad={4}
      yAxis={{ ticks: axis.ticks, y, format: (value) => defaultChartFormat(value) }}
      xLabels={xLabels}
      xTitle={props.unit === '' ? undefined : props.unit}
      table={table}
      scrub={empty === null ? scrub : null}
      empty={empty}
      className={styles.root}
      style={{ '--n': Math.max(1, count) } as CSSProperties}
      marks={({ active }) => (
        <>
          <svg className={styles.bars} width={plotWidth + 4} height={height} aria-hidden="true" style={barStyle}>
            <defs>
              <pattern id={`${uid}-h`} width={5} height={5} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                <rect className={styles.hatchBase} width={5} height={5} />
                <rect className={styles.hatchLine} width={1} height={5} />
              </pattern>
            </defs>
            {bins.map((bin, index) => {
              const x = index * layout.step + layout.offset;
              const top = y(Math.max(0, bin.count));
              const overflow = !Number.isFinite(bin.to);
              const rect = { x, y: top, width: layout.band, height: height - top };
              const d = barPath(x, top, layout.band, height, BAR_RADIUS);
              const style = overflow ? ({ fill: `url("#${uid}-h")` } as CSSProperties) : undefined;
              const className = overflow ? cn(styles.bar, styles.overflow) : styles.bar;
              return (
                <g
                  key={index}
                  className={styles.column}
                  style={{ '--i': index, transformOrigin: `${String(x)}px ${String(height)}px` } as CSSProperties}
                  data-dim={active !== null && active !== index ? '' : undefined}
                  data-entrance-last={!lastIsMarker && index === count - 1 ? '' : undefined}
                >
                  {bin.count > 0 ? (
                    rich ? (
                      <ReshapeBar
                        d={d}
                        rect={rect}
                        bornFrom={mounted ? { x, y: height, width: layout.band, height: 0 } : undefined}
                        signature={signature}
                        progress={progress}
                        className={className}
                        style={style}
                      />
                    ) : (
                      <path d={d} className={className} style={style} />
                    )
                  ) : null}
                </g>
              );
            })}
          </svg>
          {markers.length > 0 ? (
            <div className={styles.markers} aria-hidden="true" data-entrance-last="">
              {markers.map((marker, index) => (
                <span
                  key={`${marker.label}-${String(index)}`}
                  className={styles.marker}
                  style={{ transform: `translateX(${String(marker.x)}px)`, height }}
                >
                  {labelled.has(index) ? <span className={styles.markerLabel}>{marker.label}</span> : null}
                </span>
              ))}
            </div>
          ) : null}
        </>
      )}
    />
  );
}
