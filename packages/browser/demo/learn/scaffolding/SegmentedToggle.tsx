import { cn } from '@/lib/utils';
import * as React from 'react';

/**
 * Shared segmented toggle — a compact "pick one of N" control used across the
 * course's interactive widgets: scale switches (Linear / Log), mode pickers
 * (Plain FFN / SwiGLU), preset choosers, on/off switches, and so on.
 *
 * Why this exists: the same hand-rolled idiom (a bordered `inline-flex`
 * wrapper around two-or-more `<button>`s, with `bg-primary/15 text-primary`
 * for the active one) had been copy-pasted into ~13 widgets. Every copy put a
 * fixed `rounded` (4px) chip inside the theme's `rounded-md` (10px) container,
 * so the selected chip's near-square corners never nested inside the rounder
 * track — it read as visually "off" / cramped. Centralizing it fixes the
 * geometry once: a concentric `rounded-sm` (8px) chip on a filled `rounded-lg`
 * (12px) track with `p-1` padding (12 − 4 = 8, so the chip's corners sit
 * parallel to the track's), plus a subtle raised shadow on the active chip.
 *
 * Behavior is a faithful match for the buttons it replaces: each option is a
 * real `<button>` with `aria-pressed`, optional per-option and group-level
 * `disabled`, and an `onChange(value)` callback. Option values may be strings,
 * numbers, or booleans.
 */
export type SegmentedOption<T extends string | number | boolean> = {
  value: T;
  label: React.ReactNode;
  /** Accessible label, used when `label` is non-textual (an icon or symbol). */
  ariaLabel?: string;
  disabled?: boolean;
};

export function SegmentedToggle<T extends string | number | boolean>({
  value,
  onChange,
  options,
  ariaLabel,
  disabled = false,
  wrap = false,
  className,
}: {
  value: T;
  onChange: (value: T) => void;
  options: ReadonlyArray<SegmentedOption<T>>;
  /** Labels the whole group for screen readers (renders `role="group"`). */
  ariaLabel?: string;
  /** Disable every option at once (e.g. while a job is running). */
  disabled?: boolean;
  /** Let options wrap onto multiple rows (for 3–4+ wide options). */
  wrap?: boolean;
  className?: string;
}) {
  return (
    <div
      // Only become a labeled ARIA group when a name is supplied. Without a
      // label, role="group" would be an *anonymous* group (a screen-reader
      // smell) — and the un-labeled call sites were plain button rows before
      // this component existed, so this preserves their original semantics.
      role={ariaLabel ? 'group' : undefined}
      aria-label={ariaLabel}
      className={cn(
        'inline-flex items-center gap-0.5 rounded-lg border border-border bg-muted/40 p-1',
        wrap && 'flex-wrap',
        className,
      )}
    >
      {options.map((opt) => {
        const active = opt.value === value;
        const isDisabled = disabled || opt.disabled;
        return (
          <button
            key={String(opt.value)}
            type="button"
            aria-pressed={active}
            aria-label={opt.ariaLabel}
            disabled={isDisabled}
            onClick={() => onChange(opt.value)}
            className={cn(
              'rounded-sm px-2.5 py-1 text-xs font-medium transition-colors',
              'disabled:pointer-events-none disabled:opacity-50',
              active
                ? 'bg-primary/15 text-primary shadow-sm'
                : 'text-muted-foreground hover:bg-foreground/5 hover:text-foreground',
            )}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
