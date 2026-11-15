import { useRef, type CSSProperties, type ReactElement } from 'react';

import { resample } from '../chart/morph';
import { MorphPath } from '../chart/morph-path';
import { linePath, sampleXs, zipPoints } from '../chart/paths';
import { useReshape } from '../chart/use-reshape';
import { motionX } from '../motion/motion';
import { useEntrance, useRichMotion } from '../motion/use-entrance';
import { downsampleMean } from './downsample';
import styles from './mini-spark.module.css';

export interface MiniSparkProps {
  /** ≤ 60 points; more are bucket means, nulls kept. */
  data: readonly (number | null)[];
  /** sr-only: "Decode speed, last 24 hours". */
  label: string;
  /** Default var(--text-muted); the last point in var(--series-1). */
  color?: string;
  /** Default 64. */
  width?: number;
  /** Default 24. */
  height?: number;
}

const MAX_POINTS = 60;
const PAD = 3;

/** A sparkline beside a value: no axis, gaps for null, an end dot; traces once per mount, then reshapes to any new data. */
export function MiniSpark(props: MiniSparkProps): ReactElement {
  const ref = useRef<SVGSVGElement | null>(null);
  const entrance = useEntrance(ref);
  const rich = useRichMotion();
  const width = props.width ?? 64;
  const height = props.height ?? 24;
  const data = downsampleMean(props.data, MAX_POINTS);
  let low = Infinity;
  let high = -Infinity;
  for (const value of data) {
    if (value === null || !Number.isFinite(value)) continue;
    low = Math.min(low, value);
    high = Math.max(high, value);
  }
  const flat = !Number.isFinite(low) || high === low;
  const yOf = (value: number): number =>
    flat ? height / 2 : PAD + (1 - (value - low) / (high - low)) * (height - PAD * 2);
  const xs = data.map((_, i) => (data.length <= 1 ? width / 2 : PAD + (i / (data.length - 1)) * (width - PAD * 2)));
  const ys = data.map((value) => (value === null || !Number.isFinite(value) ? null : yOf(value)));
  const points = zipPoints(xs, ys);
  const path = linePath(points);
  let end: { x: number; y: number } | null = null;
  for (let i = points.length - 1; i >= 0 && end === null; i -= 1) end = points[i] ?? null;
  const signature = `${String(data.length)}|${data.map((value) => (value === null ? '' : value.toPrecision(4))).join(',')}`;
  const progress = useReshape(signature);
  const sx = sampleXs(motionX.morphSamples, xs[0] ?? 0, xs[xs.length - 1] ?? 0);
  const style = {
    '--spark-line': props.color ?? 'var(--text-muted)',
    '--spark-end': props.color ?? 'var(--series-1)',
  } as CSSProperties;

  return (
    <svg
      ref={ref}
      className={styles.root}
      width={width}
      height={height}
      viewBox={`0 0 ${String(width)} ${String(height)}`}
      role="img"
      aria-label={props.label}
      data-entrance={entrance}
      style={style}
    >
      <line className={styles.base} x1={0} x2={width} y1={height - 0.5} y2={height - 0.5} />
      <g className={styles.trace} data-entrance-last={end === null ? '' : undefined}>
        {rich ? (
          <MorphPath
            className={styles.line}
            exact={path.d}
            samples={resample(ys, motionX.morphSamples)}
            toPath={(mixed) => linePath(zipPoints(sx, mixed)).d}
            signature={signature}
            progress={progress}
          />
        ) : (
          <path className={styles.line} d={path.d} />
        )}
        {path.lone.map((point) => (
          <circle key={String(point.x)} className={styles.lone} cx={point.x} cy={point.y} r={1.25} />
        ))}
      </g>
      {end !== null ? <circle className={styles.end} cx={end.x} cy={end.y} r={2.5} data-entrance-last="" /> : null}
    </svg>
  );
}
