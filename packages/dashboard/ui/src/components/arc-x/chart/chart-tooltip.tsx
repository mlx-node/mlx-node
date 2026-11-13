import type { ReactElement, ReactNode } from 'react';

import { LegendKey, type LegendItem } from './chart-legend';
import styles from './chart.module.css';

export interface TooltipRow {
  key: string;
  label: string;
  value: string;
  kind?: LegendItem['kind'];
  color?: string;
  pattern?: 'solid' | 'alt';
  /** A comparison row (the previous period): muted. */
  muted?: boolean;
}

export interface ChartTooltipProps {
  title: string;
  rows: readonly TooltipRow[];
  note?: ReactNode;
}

/** The chart tooltip: the floating-panel look, one row per series. */
export function ChartTooltip({ title, rows, note }: ChartTooltipProps): ReactElement {
  return (
    <div className={styles.tooltip}>
      <span className={styles.tooltipTitle}>{title}</span>
      {rows.map((row) => (
        <span key={row.key} className={styles.tooltipRow} data-muted={row.muted === true ? '' : undefined}>
          {row.kind !== undefined ? <LegendKey kind={row.kind} color={row.color} pattern={row.pattern} /> : null}
          <span className={styles.tooltipLabel}>{row.label}</span>
          <span className={styles.tooltipValue}>{row.value}</span>
        </span>
      ))}
      {note !== undefined && note !== null ? <span className={styles.tooltipNote}>{note}</span> : null}
    </div>
  );
}
