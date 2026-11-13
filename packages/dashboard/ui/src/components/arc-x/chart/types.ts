export interface ChartSeries {
  key: string;
  label: string;
  /** CSS color expression, e.g. 'var(--model-3)'. Never a resolved hex from JS. */
  color: string;
  /** alt = dashed line / hatched bar (models 9–16). */
  pattern?: 'solid' | 'alt';
}

export interface ChartPoint {
  /** Stable id, e.g. bucket start ms as string. */
  key: string;
  /** Tooltip + table: "Mon, Nov 3". */
  label: string;
  /** X axis: "Nov 3"; omit to keep the axis quiet. */
  axisLabel?: string;
  values: Record<string, number | null>;
  /** Sample count behind each value. */
  samples?: Record<string, number>;
}
