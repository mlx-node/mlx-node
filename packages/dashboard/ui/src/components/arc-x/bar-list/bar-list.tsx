import { motion } from 'motion/react';
import { useRef, useState, type CSSProperties, type ReactElement, type ReactNode } from 'react';
import { Link } from 'react-router-dom';

import { cn } from '../../../lib/cn';
import { NO_VALUE } from '../../../lib/format';
import { LOW_SAMPLE_N } from '../../../lib/types';
import { Button } from '../../arc/button/button';
import { useArrivals } from '../chart/use-arrivals';
import { defaultChartFormat } from '../chart/value-format';
import { motionX } from '../motion/motion';
import { MotionGate, useEntrance, useRichMotion } from '../motion/use-entrance';
import { SwapText } from '../swap-text/swap-text';
import styles from './bar-list.module.css';

export interface BarListRow {
  key: string;
  /** Usually <ModelBadge/>. */
  label: ReactNode;
  /** Plain text of the label, for the table view and sorting. */
  text: string;
  value: number | null;
  samples?: number;
  /** Default var(--series-1). */
  color?: string;
  pattern?: 'solid' | 'alt';
  to?: string;
  /** Right of the value, muted: "412 turns". */
  hint?: string;
}

export interface BarListProps {
  label: string;
  /** '' when `formatValue` prints its own unit. */
  unit: string;
  rows: readonly BarListRow[];
  /** Scale end. Default = max value. Pass a fixed number to compare two lists fairly. */
  max?: number;
  formatValue?: (value: number) => string;
  /** Default desc, nulls last. */
  sort?: 'desc' | 'asc' | 'none';
  /** Default 8, then a "Show all n" ghost button. */
  limit?: number;
  lowSampleBelow?: number;
  emptyLabel?: string;
}

const RICH_ROWS = 40;

function sorted(rows: readonly BarListRow[], sort: 'desc' | 'asc' | 'none'): BarListRow[] {
  if (sort === 'none') return [...rows];
  const sign = sort === 'desc' ? -1 : 1;
  return [...rows].sort((a, b) => {
    if (a.value === null && b.value === null) return a.text < b.text ? -1 : a.text > b.text ? 1 : 0;
    if (a.value === null) return 1;
    if (b.value === null) return -1;
    if (a.value !== b.value) return sign * (a.value - b.value);
    return a.text < b.text ? -1 : a.text > b.text ? 1 : 0;
  });
}

interface RowProps {
  row: BarListRow;
  rank: number;
  share: number | null;
  format: (value: number) => string;
  unit: string;
  low: boolean;
  last: boolean;
  arriving: boolean;
  onSettled: () => void;
}

function Row({ row, rank, share, format, unit, low, last, arriving, onSettled }: RowProps): ReactElement {
  const rich = useRichMotion();
  const rest = share === null ? 100 : (1 - Math.min(1, Math.max(0, share))) * 100;
  const style = {
    '--c': row.color ?? 'var(--series-1)',
    '--rest': `${String(rest)}%`,
    '--step': Math.min(rank, motionX.reveal.maxSteps - 1),
  } as CSSProperties;
  const content = (
    <>
      {share !== null ? (
        <>
          <span
            className={styles.fill}
            data-pattern={row.pattern === 'alt' || low ? 'hatch' : undefined}
            data-entrance-last={last ? '' : undefined}
            aria-hidden="true"
          />
          <span className={styles.cap} aria-hidden="true" />
        </>
      ) : null}
      <span className={styles.label} title={row.text}>
        {row.to !== undefined ? (
          <Link className={cn(styles.link, styles.labelText)} to={row.to}>
            {row.label}
          </Link>
        ) : (
          <span className={styles.labelText}>{row.label}</span>
        )}
      </span>
      <span className={styles.reading}>
        <span className={styles.value}>
          <SwapText text={row.value === null ? NO_VALUE : format(row.value)} />
        </span>
        {row.value !== null && unit !== '' ? <span className={styles.unit}>{unit}</span> : null}
        {low && row.samples !== undefined ? <span className={styles.low}>n={row.samples}</span> : null}
        {row.hint !== undefined ? <span className={styles.hint}>{row.hint}</span> : null}
      </span>
    </>
  );
  const shared = {
    className: styles.row,
    style,
    'data-link': row.to !== undefined ? '' : undefined,
    'data-arrive': arriving ? '' : undefined,
    'data-entrance-last': last && share === null ? '' : undefined,
    onAnimationEnd: (event: { target: EventTarget; currentTarget: EventTarget }) => {
      if (event.target === event.currentTarget) onSettled();
    },
  };
  if (rich) {
    return (
      <motion.li layout="position" transition={motionX.spring.travel} {...shared}>
        {content}
      </motion.li>
    );
  }
  return <li {...shared}>{content}</li>;
}

/** Ranked rows with a tinted bar behind each: rows reveal once, fills grow, re-ranks travel, values swap. */
export function BarList(props: BarListProps): ReactElement {
  const ref = useRef<HTMLDivElement | null>(null);
  const entrance = useEntrance(ref);
  const rich = useRichMotion();
  const [expanded, setExpanded] = useState(false);
  const format = props.formatValue ?? defaultChartFormat;
  const lowBelow = props.lowSampleBelow ?? LOW_SAMPLE_N;
  const limit = props.limit ?? 8;
  const rows = sorted(props.rows, props.sort ?? 'desc');
  const shown = expanded ? rows : rows.slice(0, limit);
  let max = props.max;
  if (max === undefined) {
    max = 0;
    for (const row of rows) if (row.value !== null && Number.isFinite(row.value)) max = Math.max(max, row.value);
  }
  const arrivals = useArrivals(
    shown.map((row) => row.key),
    rich && entrance === 'done',
  );
  const lastAnimated = Math.min(shown.length, motionX.reveal.maxSteps) - 1;

  return (
    <div ref={ref} className={styles.root} data-entrance={entrance}>
      {rows.length === 0 ? (
        <p className={styles.empty}>{props.emptyLabel ?? 'Nothing to list'}</p>
      ) : (
        <ol className={styles.list} aria-label={props.unit === '' ? props.label : `${props.label} (${props.unit})`}>
          {shown.map((row, index) => {
            const share = row.value === null || max === undefined || max <= 0 ? null : row.value / max;
            return (
              <MotionGate key={row.key} open={index < RICH_ROWS}>
                <Row
                  row={row}
                  rank={index}
                  share={share}
                  format={format}
                  unit={props.unit}
                  low={row.samples !== undefined && row.samples < lowBelow}
                  last={index === lastAnimated}
                  arriving={arrivals.arriving.has(row.key)}
                  onSettled={() => arrivals.settle(row.key)}
                />
              </MotionGate>
            );
          })}
        </ol>
      )}
      {rows.length > limit ? (
        <Button variant="ghost" size="sm" onClick={() => setExpanded((value) => !value)}>
          {expanded ? 'Show fewer' : `Show all ${String(rows.length)}`}
        </Button>
      ) : null}
    </div>
  );
}
