import type { ReactElement } from 'react';

import styles from './chart.module.css';

export interface ChartTableRow {
  key: string;
  /** First cell is the category (a row header). */
  cells: readonly string[];
  current?: boolean;
}

export interface ChartTableSpec {
  head: readonly string[];
  rows: readonly ChartTableRow[];
}

/** The visually hidden table every chart ships: the data in text, for assistive tech and tests. */
export function ChartTable({ caption, table }: { caption: string; table: ChartTableSpec }): ReactElement {
  return (
    <table className={styles.srOnly}>
      <caption>{caption}</caption>
      <thead>
        <tr>
          {table.head.map((cell, index) => (
            <th key={index} scope="col">
              {cell}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {table.rows.map((row) => (
          <tr key={row.key} aria-current={row.current === true ? 'true' : undefined}>
            {row.cells.map((cell, index) =>
              index === 0 ? (
                <th key={index} scope="row">
                  {cell}
                </th>
              ) : (
                <td key={index}>{cell}</td>
              ),
            )}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** "no data" for null, else the formatted value with " (n=3)" when the sample is low. */
export function tableCell(
  value: number | null | undefined,
  format: (value: number) => string,
  samples?: number,
  lowBelow?: number,
): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return 'no data';
  const low = samples !== undefined && lowBelow !== undefined && samples < lowBelow;
  return low ? `${format(value)} (n=${String(samples)})` : format(value);
}
