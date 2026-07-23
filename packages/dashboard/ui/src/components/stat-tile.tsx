import { Card } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import type { LucideIcon } from 'lucide-react';
import type { ReactNode } from 'react';

export interface StatTileProps {
  /** Sentence-case metric name, no trailing colon. */
  label: string;
  /** Headline value — proportional figures (auto-compact via `formatCount`). */
  value: ReactNode;
  /** Optional secondary line under the value (muted). */
  sub?: ReactNode;
  /** Optional metric glyph, shown top-right. */
  icon?: LucideIcon;
  /** Optional element below the tile body (e.g. a usage meter). */
  footer?: ReactNode;
  className?: string;
}

/**
 * KPI tile: a muted label, one semibold headline value, an optional sub-line and
 * footer. Text stays in ink tokens; the value uses the font's default
 * proportional figures (tabular-nums is reserved for aligned columns).
 */
export function StatTile({ label, value, sub, icon: Icon, footer, className }: StatTileProps) {
  return (
    <Card className={cn('hover:shadow-soft-lg gap-0 py-5 transition-shadow', className)}>
      <div className="flex items-start justify-between gap-3 px-5">
        <div className="min-w-0 space-y-1.5">
          <p className="text-muted-foreground text-sm font-medium">{label}</p>
          <div className="text-3xl leading-none font-semibold tracking-tight">{value}</div>
          {sub !== undefined && <div className="text-muted-foreground text-sm">{sub}</div>}
        </div>
        {Icon !== undefined && (
          <span className="bg-brand-gradient text-primary-foreground shadow-soft flex size-9 shrink-0 items-center justify-center rounded-lg">
            <Icon className="size-5" aria-hidden />
          </span>
        )}
      </div>
      {footer !== undefined && <div className="mt-4 px-5">{footer}</div>}
    </Card>
  );
}
