/**
 * Shared recharts primitives for the Metrics and Cache pages, matching the
 * conventions C9 established in `pages/session-detail.tsx`: recessive
 * ink-token axes, a popover-styled tooltip, and the `--viz-*` chart palette
 * (never shadcn `--chart-*`, which fail the dataviz contrast/CVD gates).
 */

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { AlertCircle } from 'lucide-react';
import type { ComponentType, ReactNode } from 'react';

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

interface ChartCardProps {
  title: string;
  subtitle: string;
  children: ReactNode;
  /** Plot height utility class; defaults to `h-56`. */
  heightClass?: string;
}

/** Titled card wrapping a fixed-height plot area, mirroring C9's `ChartCard`. */
export function ChartCard({ title, subtitle, children, heightClass = 'h-56' }: ChartCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
        <p className="text-muted-foreground text-sm">{subtitle}</p>
      </CardHeader>
      <CardContent>
        <div className={`${heightClass} w-full`}>{children}</div>
      </CardContent>
    </Card>
  );
}

/**
 * Empty-state body for a plot area: a glyph, a headline, and a hint naming what
 * produces the data. Sized to fill a `ChartCard`'s plot area so cards keep an
 * even height whether populated or empty.
 */
export function ChartEmpty({ icon: Icon, message, hint }: { icon: ComponentType<{ className?: string }>; message: string; hint: string }) {
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
