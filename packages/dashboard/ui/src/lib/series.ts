/** Fixed series of charts that do not plot models. Two-series charts use palette slots 1 and 2; a reference is grey and dashed. */

import type { ChartSeries } from '../components/arc-x/chart/types';

export const SERIES_A = 'var(--model-1)';
export const SERIES_B = 'var(--model-2)';
export const SERIES_REF: Pick<ChartSeries, 'color' | 'pattern'> = { color: 'var(--model-other)', pattern: 'alt' };

/** Stack order, bottom first: cached, input, output, reasoning. */
export const TOKEN_SERIES: ChartSeries[] = [
  { key: 'cached', label: 'Cached', color: 'var(--data-cached)' },
  { key: 'input', label: 'Input', color: 'var(--data-input)' },
  { key: 'answer', label: 'Output (answer)', color: 'var(--data-output)' },
  { key: 'reasoning', label: 'Reasoning', color: 'var(--data-reasoning)' },
];

export const OUTCOME_SERIES: ChartSeries[] = [
  { key: 'error', label: 'Error', color: 'var(--model-8)', pattern: 'solid' },
  { key: 'cancelled', label: 'Cancelled', color: 'var(--model-other)', pattern: 'alt' },
];

export const LOOKUP_SERIES: ChartSeries[] = [
  { key: 'hits', label: 'Hits', color: 'var(--hit)' },
  { key: 'misses', label: 'Misses', color: 'var(--miss)' },
];
