import { useRef, type CSSProperties, type ReactElement } from 'react';

import { LOW_SAMPLE_N, LOW_SAMPLE_TAIL_N, type Quantiles } from '../../../lib/types';
import { Tooltip } from '../../arc/tooltip/tooltip';
import { defaultChartFormat, withUnit } from '../chart/value-format';
import { useEntrance } from '../motion/use-entrance';
import styles from './quantile-bar.module.css';

export interface QuantileBarProps {
  /** "Decode speed, qwen3.6-27b" */
  label: string;
  /** { n, p05, p50, p95 } from the API. */
  quantiles: Quantiles;
  /** Shared scale for every row of the column. */
  min: number;
  max: number;
  unit: string;
  formatValue?: (value: number) => string;
  /** Default var(--series-1). */
  color?: string;
  /** Default 112. */
  width?: number;
}

function finite(value: number | null): value is number {
  return value !== null && Number.isFinite(value);
}

/** The words of a QuantileBar, for its aria-label and tooltip. */
export function quantileText(props: QuantileBarProps): { name: string; lines: string[] } {
  const format = props.formatValue ?? defaultChartFormat;
  const { n, p05, p50, p95 } = props.quantiles;
  const turns = `${String(n)} ${n === 1 ? 'turn' : 'turns'}`;
  if (!finite(p50)) return { name: `${props.label}: no data`, lines: [props.label, 'no data', turns] };
  const median = `median ${withUnit(format(p50), props.unit)}`;
  const tails = n >= LOW_SAMPLE_TAIL_N && finite(p05) && finite(p95) ? `p05 ${format(p05)}, p95 ${format(p95)}` : null;
  const count = n < LOW_SAMPLE_N ? `${turns}, low sample` : turns;
  const parts = [median, ...(tails === null ? [] : [tails]), count];
  return {
    name: `${props.label}: ${parts.join(', ')}`,
    lines: [props.label, median, [tails, count].filter((part) => part !== null).join(' · ')],
  };
}

/** A p05–p95 band with a median tick, small enough for a table cell; a ring when the sample is low. */
export function QuantileBar(props: QuantileBarProps): ReactElement {
  const ref = useRef<HTMLSpanElement | null>(null);
  const entrance = useEntrance(ref);
  const width = props.width ?? 112;
  const { n, p05, p50, p95 } = props.quantiles;
  const span = props.max - props.min;
  const at = (value: number): number =>
    span <= 0 ? width / 2 : Math.min(width, Math.max(0, ((value - props.min) / span) * width));
  const band = n >= LOW_SAMPLE_TAIL_N && finite(p05) && finite(p95) && finite(p50);
  const text = quantileText(props);
  const style = {
    width,
    '--c': props.color ?? 'var(--series-1)',
    '--mid': `${String(finite(p50) ? at(p50) : 0)}px`,
    '--lo': `${String(band && finite(p05) ? at(p05) : 0)}px`,
    '--hi': `${String(band && finite(p95) ? width - at(p95) : 0)}px`,
    '--mid-rest': `${String(width - (finite(p50) ? at(p50) : 0))}px`,
  } as CSSProperties;
  return (
    <Tooltip
      content={
        <span className={styles.tip}>
          {text.lines.map((line) => (
            <span key={line}>{line}</span>
          ))}
        </span>
      }
    >
      <span ref={ref} className={styles.root} role="img" aria-label={text.name} style={style} data-entrance={entrance}>
        <span className={styles.track} />
        {band ? <span className={styles.band} data-kind="band" /> : null}
        {finite(p50) ? (
          n < LOW_SAMPLE_N ? (
            <span className={styles.ring} data-kind="ring" data-entrance-last="" />
          ) : (
            <span className={styles.median} data-kind="median" data-entrance-last="" />
          )
        ) : null}
      </span>
    </Tooltip>
  );
}
