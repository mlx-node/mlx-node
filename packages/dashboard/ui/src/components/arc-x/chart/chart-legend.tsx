import type { CSSProperties, ReactElement } from 'react';

import styles from './chart.module.css';

export interface LegendItem {
  key: string;
  label: string;
  /** CSS color expression. */
  color?: string;
  pattern?: 'solid' | 'alt';
  /** line: 10 × 2 px key; square: 8 × 8 px; ghost: dashed muted line; hollow: a ring (low sample). */
  kind: 'line' | 'square' | 'ghost' | 'hollow';
  hidden?: boolean;
  /** A toggle button (aria-pressed = shown); otherwise a plain key. */
  toggle?: boolean;
}

export interface ChartLegendProps {
  items: readonly LegendItem[];
  onToggle?: (key: string) => void;
  label: string;
}

export function LegendKey({ kind, color, pattern }: Pick<LegendItem, 'kind' | 'color' | 'pattern'>): ReactElement {
  const style = color === undefined ? undefined : ({ '--key-color': color } as CSSProperties);
  return <span className={styles.key} data-kind={kind} data-pattern={pattern} style={style} aria-hidden="true" />;
}

/** The legend above a plot: toggles for series (aria-pressed), plain keys for the rest. */
export function ChartLegend({ items, onToggle, label }: ChartLegendProps): ReactElement {
  return (
    <div className={styles.legend} role="group" aria-label={`${label} legend`}>
      {items.map((item) => {
        const content = (
          <>
            <LegendKey kind={item.kind} color={item.color} pattern={item.pattern} />
            {item.label}
          </>
        );
        if (item.toggle === true && onToggle !== undefined) {
          return (
            <button
              key={item.key}
              type="button"
              className={styles.legendItem}
              aria-pressed={item.hidden !== true}
              data-hidden={item.hidden === true ? '' : undefined}
              onClick={() => onToggle(item.key)}
            >
              {content}
            </button>
          );
        }
        return (
          <span key={item.key} className={styles.legendItem}>
            {content}
          </span>
        );
      })}
    </div>
  );
}
