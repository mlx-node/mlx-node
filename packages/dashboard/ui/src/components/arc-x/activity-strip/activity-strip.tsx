import type { CSSProperties, ReactElement } from 'react';

import { cn } from '../../../lib/cn';
import { stepStackPaths } from '../chart/paths';
import styles from './activity-strip.module.css';
import { stackAxisMax, stackPeak, stackSteps, type StackBand, type StackPoint } from './phase-stack';

export interface ActivityStripProps {
  /** Ascending t0; intervals in ms on the snapshot clock. */
  points: readonly StackPoint[];
  /** Bottom to top, usually `PHASE_BANDS`. */
  bands: readonly StackBand[];
  /** Right edge of the strip: the newest snapshot time, not Date.now(). */
  nowMs: number;
  /** Default 120_000. */
  windowMs?: number;
  /** Default 96. */
  width?: number;
  /** Default 24. */
  height?: number;
  /** The accessible name, with the band shares in words. */
  label: string;
  className?: string;
}

/** A small stacked step picture of lanes by phase over true time: no axis, empty where the engine was idle. */
export function ActivityStrip(props: ActivityStripProps): ReactElement {
  const width = props.width ?? 96;
  const height = props.height ?? 24;
  const windowMs = props.windowMs ?? 120_000;
  const from = props.nowMs - windowMs;
  const keys = props.bands.map((band) => band.key);
  const top = stackAxisMax(stackPeak(props.points, keys, from, props.nowMs));
  const x = (t: number): number => width * (1 - (props.nowMs - t) / windowMs);
  const y = (value: number): number => height - (value / top) * height;
  const paths = stepStackPaths(stackSteps(props.points, keys, from, props.nowMs, x), y);
  const size = { width, height } as CSSProperties;

  return (
    <svg
      className={cn(styles.root, props.className)}
      style={size}
      width={width}
      height={height}
      viewBox={`0 0 ${String(width)} ${String(height)}`}
      role="img"
      aria-label={props.label}
    >
      <line className={styles.base} x1={0} x2={width} y1={height - 0.5} y2={height - 0.5} />
      {props.bands.map((band, index) =>
        paths[index] === undefined || paths[index] === '' ? null : (
          <path
            key={band.key}
            className={styles.band}
            d={paths[index]}
            style={{ '--c': band.color } as CSSProperties}
            data-band={band.key}
          />
        ),
      )}
    </svg>
  );
}
