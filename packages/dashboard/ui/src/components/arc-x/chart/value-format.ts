const COMPACT = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 });
const PRECISE = [
  new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 }),
  new Intl.NumberFormat('en-US', { maximumFractionDigits: 1 }),
  new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }),
] as const;

/** The default value and tick text of an own chart: 2 decimals below 10, 1 below 100, compact from 10,000. */
export function defaultChartFormat(value: number): string {
  const size = Math.abs(value);
  if (size >= 10_000) return COMPACT.format(value);
  return PRECISE[size < 10 ? 0 : size < 100 ? 1 : 2].format(value);
}

/** Approximate rendered width of an 11 px tabular label, for sparse x labels. */
export function labelWidthPx(text: string): number {
  return text.length * 6.4 + 14;
}

/**
 * A formatted value with its unit: '' adds nothing, '%' follows the number with no space, any other unit after a
 * space. The formatter of a chart prints the bare number; a formatter that prints its own unit goes with unit ''.
 */
export function withUnit(text: string, unit: string | undefined): string {
  if (unit === undefined || unit === '') return text;
  return unit === '%' ? `${text}%` : `${text} ${unit}`;
}
