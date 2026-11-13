import { Fragment, useRef, useState, type CSSProperties, type ReactElement } from 'react';

import { cn } from '../../../lib/cn';
import { formatNumber, formatPercent } from '../../../lib/format';
import { useEntrance } from '../motion/use-entrance';
import { SwapText } from '../swap-text/swap-text';
import { cellCounts } from './cell-counts';
import styles from './cell-grid.module.css';

export interface CellSegment {
  key: string;
  label: string;
  /** In capacity units (blocks). */
  value: number;
  /** CSS color expression. */
  color: string;
  pattern?: 'solid' | 'hatch';
}

export interface CellGridProps {
  /** "KV block pool, qwen3.6-27b" */
  label: string;
  /** 0 or less -> empty state. */
  capacity: number;
  /** Drawn in order from the first cell; the rest is free. */
  segments: readonly CellSegment[];
  /** Units inside the free part that are promised. Drawn as outlined cells at the END of the grid. */
  reserved?: { value: number; label: string };
  /** A level drawn as a dashed line after that many units. */
  threshold?: { value: number; label: string };
  /** { one: 'block', other: 'blocks' } */
  unit: { one: string; other: string };
  /** Default "Free". */
  freeLabel?: string;
  /** Default 20. */
  columns?: number;
  /** Default 10. */
  rows?: number;
  /** Default formatNumber. */
  formatValue?: (value: number) => string;
  emptyLabel?: string;
  className?: string;
}

const FREE = 'free';
const RESERVED = 'reserved';

interface CellLayout {
  kinds: string[];
  counts: number[];
  reservedCells: number;
  thresholdCell: number | null;
}

function layoutCells(props: CellGridProps, cells: number): CellLayout {
  const counts = cellCounts(
    props.capacity,
    props.segments.map((segment) => segment.value),
    cells,
  );
  const kinds: string[] = [];
  props.segments.forEach((segment, index) => {
    for (let i = 0; i < counts[index]; i += 1) kinds.push(segment.key);
  });
  const free = cells - kinds.length;
  const reservedCells =
    props.reserved === undefined || props.capacity <= 0
      ? 0
      : Math.min(free, Math.max(0, Math.round((props.reserved.value / props.capacity) * cells)));
  for (let i = 0; i < free - reservedCells; i += 1) kinds.push(FREE);
  for (let i = 0; i < reservedCells; i += 1) kinds.push(RESERVED);
  const threshold = props.threshold;
  const thresholdCell =
    threshold === undefined || threshold.value <= 0 || threshold.value >= props.capacity
      ? null
      : Math.min(cells - 1, Math.floor((threshold.value / props.capacity) * cells));
  return { kinds, counts, reservedCells, thresholdCell };
}

interface WaveState {
  kinds: readonly string[];
  /** Order of each changed cell in the last data change (grid order); -1 = unchanged. */
  ranks: readonly number[];
}

/** A capacity as a fixed grid of cells filled by category; cells take their kind in a wave. */
export function CellGrid(props: CellGridProps): ReactElement {
  const ref = useRef<HTMLDivElement | null>(null);
  const entrance = useEntrance(ref);
  const [dim, setDim] = useState<string | null>(null);
  const columns = props.columns ?? 20;
  const rows = props.rows ?? 10;
  const cells = columns * rows;
  const format = props.formatValue ?? ((value: number) => formatNumber(value));
  const empty = props.capacity <= 0;
  const layout = layoutCells(props, cells);
  const [wave, setWave] = useState<WaveState>({ kinds: layout.kinds, ranks: layout.kinds.map(() => -1) });
  if (wave.kinds.join(',') !== layout.kinds.join(',')) {
    let rank = 0;
    const ranks = layout.kinds.map((kind, index) => (wave.kinds[index] === kind ? -1 : rank++));
    setWave({ kinds: layout.kinds, ranks });
  }
  const ranks = wave.kinds.join(',') === layout.kinds.join(',') ? wave.ranks : layout.kinds.map(() => -1);

  const freeUnits = Math.max(
    0,
    props.capacity -
      props.segments.reduce((sum, segment) => sum + Math.max(0, segment.value), 0) -
      (props.reserved?.value ?? 0),
  );
  const unitWord = (value: number): string => (value === 1 ? props.unit.one : props.unit.other);
  const legend = [
    ...props.segments.map((segment) => ({
      key: segment.key,
      label: segment.label,
      value: segment.value,
      color: segment.color,
      pattern: segment.pattern,
    })),
    ...(props.reserved !== undefined
      ? [
          {
            key: RESERVED,
            label: props.reserved.label,
            value: props.reserved.value,
            color: undefined,
            pattern: undefined,
          },
        ]
      : []),
    { key: FREE, label: props.freeLabel ?? 'Free', value: freeUnits, color: undefined, pattern: undefined },
  ];
  const description = [
    props.label,
    ...legend.map(
      (row) =>
        `${row.label} ${format(row.value)} ${unitWord(row.value)}, ${formatPercent(props.capacity > 0 ? row.value / props.capacity : null)}`,
    ),
  ].join('; ');
  const perCell = props.capacity / cells;
  const perCellText = perCell < 10 ? String(Math.round(perCell * 10) / 10) : formatNumber(Math.round(perCell));
  const colorOf = new Map(props.segments.map((segment) => [segment.key, segment]));
  const thresholdRow = layout.thresholdCell === null ? null : Math.floor(layout.thresholdCell / columns);

  return (
    <div
      ref={ref}
      className={cn(styles.root, props.className)}
      data-entrance={entrance}
      data-dim={dim ?? undefined}
      style={{ '--cols': columns, '--rows': rows, '--cells': cells } as CSSProperties}
    >
      {empty ? (
        <div className={styles.emptyBox} role="img" aria-label={`${props.label}: ${props.emptyLabel ?? 'no capacity'}`}>
          {props.emptyLabel ?? 'Nothing to show'}
        </div>
      ) : (
        <div className={styles.grid} role="img" aria-label={description}>
          {layout.kinds.map((kind, index) => {
            const segment = colorOf.get(kind);
            const style = {
              '--i': index,
              '--j': Math.max(0, ranks[index] ?? 0),
              ...(segment !== undefined ? { '--c': segment.color } : {}),
            } as CSSProperties;
            const late = kind === FREE && layout.thresholdCell !== null && index > layout.thresholdCell;
            return (
              <Fragment key={index}>
                <span
                  className={styles.cell}
                  style={style}
                  data-kind={kind}
                  data-pattern={segment?.pattern}
                  data-late={late ? '' : undefined}
                  data-dimmed={dim !== null && dim !== kind ? '' : undefined}
                  aria-hidden="true"
                />
                {thresholdRow !== null && index === (thresholdRow + 1) * columns - 1 ? (
                  <span className={styles.threshold} aria-hidden="true">
                    <span
                      className={styles.notch}
                      style={{ left: `${String(((((layout.thresholdCell ?? 0) % columns) + 1) / columns) * 100)}%` }}
                    />
                  </span>
                ) : null}
              </Fragment>
            );
          })}
        </div>
      )}
      <div className={styles.caption}>
        <span>
          Each cell is {perCellText} {perCell === 1 ? props.unit.one : props.unit.other}
        </span>
        {layout.thresholdCell !== null && props.threshold !== undefined ? (
          <span className={styles.thresholdLabel}>
            <span className={styles.thresholdKey} aria-hidden="true" />
            {props.threshold.label}
          </span>
        ) : null}
      </div>
      <ul className={styles.legend} aria-label={`${props.label} legend`}>
        {legend.map((row, index) => (
          <li
            key={row.key}
            className={styles.row}
            tabIndex={0}
            data-free={row.key === FREE ? '' : undefined}
            data-entrance-last={index === legend.length - 1 ? '' : undefined}
            style={
              {
                '--step': Math.min(index, 7),
                ...(row.color !== undefined ? { '--c': row.color } : {}),
              } as CSSProperties
            }
            onPointerEnter={() => setDim(row.key)}
            onPointerLeave={() => setDim(null)}
            onFocus={() => setDim(row.key)}
            onBlur={() => setDim(null)}
          >
            <span
              className={styles.key}
              data-kind={row.key === FREE || row.key === RESERVED ? row.key : 'segment'}
              data-pattern={row.pattern}
              aria-hidden="true"
            />
            <span className={styles.label}>{row.label}</span>
            <span className={styles.value} data-zero={row.value === 0 ? '' : undefined}>
              <SwapText text={format(row.value)} />
            </span>
            <span className={styles.share}>
              <SwapText text={formatPercent(props.capacity > 0 ? row.value / props.capacity : null)} />
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
