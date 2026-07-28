/**
 * Shared recharts primitives for the Metrics and Cache pages, matching the
 * conventions C9 established in `pages/session-detail.tsx`: recessive
 * ink-token axes, a popover-styled tooltip, and the `--viz-*` chart palette
 * (never shadcn `--chart-*`, which fail the dataviz contrast/CVD gates).
 */

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { AlertCircle } from 'lucide-react';
import type { ComponentType, CSSProperties, ReactNode } from 'react';

/** Recessive axis tick styling (muted ink, never a series colour). */
export const AXIS_TICK = { fill: 'var(--color-muted-foreground)', fontSize: 12 } as const;

/** Tooltip surface matching the popover token, so it reads on either theme. */
export const TOOLTIP_CONTENT_STYLE = {
  background: 'var(--color-popover)',
  border: '1px solid var(--color-border)',
  borderRadius: 8,
  fontSize: 12,
  color: 'var(--color-popover-foreground)',
} as const;

/**
 * Fixed-order categorical palette (CSS vars from `index.css`). Assigned to
 * entities in a stable order and never cycled; a 9th entity takes
 * `OTHER_SERIES_COLOR`. Slots 1–3 are the token-category hues, 4–8 the next
 * validated steps of the same reference palette (both modes gate-clean).
 */
export const SERIES_COLORS = [
  'var(--viz-series-1)',
  'var(--viz-series-2)',
  'var(--viz-series-3)',
  'var(--viz-series-4)',
  'var(--viz-series-5)',
  'var(--viz-series-6)',
  'var(--viz-series-7)',
  'var(--viz-series-8)',
] as const;

/** Muted fill for entities past the palette's eight slots (folded to "Other"). */
export const OTHER_SERIES_COLOR = 'var(--color-muted-foreground)';

/**
 * Map an ordered, de-duplicated list of entity keys (e.g. model names) to fixed
 * palette slots. The order is the caller's responsibility (a stable key such as
 * usage rank) so a colour follows the entity rather than its position in any one
 * chart.
 */
export function buildSeriesColorMap(orderedKeys: string[]): Map<string, string> {
  const map = new Map<string, string>();
  orderedKeys.forEach((key, index) => {
    map.set(key, index < SERIES_COLORS.length ? SERIES_COLORS[index] : OTHER_SERIES_COLOR);
  });
  return map;
}

/**
 * Longest a category (model) label may run before it is shortened.
 *
 * This is the ONLY bound on how wide a `<YAxis width="auto">` can grow, so it is
 * also the guard that keeps a 60-character model id from eating the plot area.
 */
export const CATEGORY_LABEL_MAX_CHARS = 26;

/**
 * Shorten `text` to `maxChars` by dropping its MIDDLE and keeping both ends.
 *
 * Head truncation is wrong for model ids specifically: the family and size sit
 * at the front while the quantisation recipe sits at the back, so
 * `Gemma-4-31B-IT-UD-Q4_K_XL-mlx` and `Gemma-4-31B-IT-UD-Q4_K_M-mlx` are
 * identical for their first 21 characters. A tail-clipping shortener renders
 * both as the same string on a chart whose entire purpose is to compare them.
 */
export function truncateMiddle(text: string, maxChars: number): string {
  if (maxChars < 2) return text.slice(0, Math.max(0, maxChars));
  if (text.length <= maxChars) return text;
  const keep = maxChars - 1;
  const head = Math.ceil(keep / 2);
  return `${text.slice(0, head)}…${text.slice(text.length - (keep - head))}`;
}

/**
 * Map each key to a display label that is short enough for an axis AND distinct
 * from every other label in the same chart.
 *
 * Distinctness is the point. Shortening alone can map two different models onto
 * one string, and two rows reading the same name on a comparison chart is worse
 * than a long label: the reader cannot tell which model is which, and nothing in
 * the rendering says so. When shortening collides, the later key keeps a
 * numbered suffix so the rows stay tellable apart; the full id is still in the
 * tick's `title`/tooltip.
 */
export function categoryLabels(keys: Iterable<string>, maxChars = CATEGORY_LABEL_MAX_CHARS): Map<string, string> {
  const unique = [...new Set(keys)];
  const shortened = unique.map((key) => truncateMiddle(key, maxChars));
  if (new Set(shortened).size === shortened.length) {
    return new Map(unique.map((key, index) => [key, shortened[index]]));
  }
  // Leave room for the ` (2)` discriminator before it is appended, so a
  // disambiguated label is no wider than an ordinary one.
  const bases = unique.map((key) => truncateMiddle(key, Math.max(2, maxChars - 4)));
  const labels = new Map<string, string>();
  const taken = new Set<string>();
  unique.forEach((key, index) => {
    const base = bases[index];
    let label = base;
    for (let nth = 2; taken.has(label); nth++) label = `${base} (${nth})`;
    taken.add(label);
    labels.set(key, label);
  });
  return labels;
}

/** Vertical room one category row needs before its tick labels start colliding. */
export const CATEGORY_ROW_PX = 30;

/** Room under the rows for the value axis plus the chart's own top margin. */
const CATEGORY_AXIS_PX = 44;

/** Floor, so a one-bar or empty chart still matches the cards beside it. */
const CATEGORY_MIN_PLOT_PX = 224;

/**
 * Plot height for a horizontal bar chart with `rowCount` categories.
 *
 * A fixed-height plot is what silently drops category labels: recharts measures
 * each tick and hides the ones that would overlap, so an 11-bar chart in a 224px
 * box renders 11 bars and 6 names. Forcing every tick to draw (`interval={0}`)
 * only helps if the rows have the room, hence a height that grows with the data.
 */
export function categoryChartHeight(rowCount: number): number {
  return Math.max(CATEGORY_MIN_PLOT_PX, rowCount * CATEGORY_ROW_PX + CATEGORY_AXIS_PX);
}

interface ChartCardProps {
  title: string;
  subtitle: string;
  children: ReactNode;
  /** Plot height utility class; defaults to `h-56`. Ignored when `heightPx` is set. */
  heightClass?: string;
  /** Exact plot height in px, for charts that size themselves from their data. */
  heightPx?: number;
}

/** Titled card wrapping a fixed-height plot area, mirroring C9's `ChartCard`. */
export function ChartCard({ title, subtitle, children, heightClass = 'h-56', heightPx }: ChartCardProps) {
  // A px height has to be inline: the row count is data, and Tailwind only ships
  // the height utilities that appear literally in the source.
  const style: CSSProperties | undefined = heightPx === undefined ? undefined : { height: heightPx };
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
        <p className="text-muted-foreground text-sm">{subtitle}</p>
      </CardHeader>
      <CardContent>
        <div className={heightPx === undefined ? `${heightClass} w-full` : 'w-full'} style={style}>
          {children}
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * Empty-state body for a plot area: a glyph, a headline, and a hint naming what
 * produces the data. Sized to fill a `ChartCard`'s plot area so cards keep an
 * even height whether populated or empty.
 */
export function ChartEmpty({
  icon: Icon,
  message,
  hint,
}: {
  icon: ComponentType<{ className?: string }>;
  message: string;
  hint: string;
}) {
  return (
    <div className="text-muted-foreground flex h-full flex-col items-center justify-center gap-2 px-4 text-center text-sm">
      <Icon className="size-6" aria-hidden />
      <p>{message}</p>
      <p className="text-xs">{hint}</p>
    </div>
  );
}

/** Inline error body for a plot area. */
export function ChartError({ message }: { message: string }) {
  return (
    <div className="text-destructive flex h-full items-center justify-center gap-2 px-4 text-center text-sm">
      <AlertCircle className="size-4 shrink-0" aria-hidden />
      {message}
    </div>
  );
}

/** Loading skeleton filling a plot area. */
export function ChartSkeleton() {
  return <Skeleton className="h-full w-full" />;
}

/**
 * Resolves a plot area to one of four states in priority order: error, loading,
 * empty, then the chart itself. `children` (the chart) is only rendered when
 * there is non-empty data to draw.
 */
export function ChartBody({
  loading,
  error,
  isEmpty,
  empty,
  children,
}: {
  loading: boolean;
  error: Error | undefined;
  isEmpty: boolean;
  empty: ReactNode;
  children: ReactNode;
}) {
  if (error !== undefined) return <ChartError message={error.message} />;
  if (loading) return <ChartSkeleton />;
  if (isEmpty) return <>{empty}</>;
  return <>{children}</>;
}
