import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react';

import { cn } from '../../../lib/cn';
import { Tooltip } from '../../arc/tooltip/tooltip';
import { SwapText } from '../swap-text/swap-text';
import styles from './tool-chip.module.css';

export interface ToolChipProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'children'> {
  icon: ReactNode;
  label: string;
  /** 'off' = dashed outline, secondary text. Default 'on'. */
  state?: 'on' | 'off';
  /** A 6 px dot after the label. The words for it go into `aria-label`. */
  dot?: 'warning' | 'danger';
  /** Icon only; the label becomes the tooltip. */
  compact?: boolean;
  /** 'active': a quiet fill for a setting that is raised. 'danger': the danger border, tint and ink. */
  tone?: 'active' | 'danger';
  /** The label stays where a narrow row drops the labels of its chips. */
  keepLabel?: boolean;
}

/** A 28 px pill that names a tool of the composer and opens its menu. */
export const ToolChip = forwardRef<HTMLButtonElement, ToolChipProps>(function ToolChip(
  { icon, label, state = 'on', dot, compact = false, tone, keepLabel = false, className, type = 'button', ...rest },
  ref,
) {
  const chip = (
    <button
      ref={ref}
      type={type}
      className={cn(styles.root, className)}
      data-chip={state}
      data-compact={compact ? '' : undefined}
      data-chip-tone={tone}
      data-keep-label={keepLabel ? '' : undefined}
      aria-label={rest['aria-label'] ?? label}
      {...rest}
    >
      <span className={styles.icon} aria-hidden="true">
        {icon}
      </span>
      {compact ? null : (
        <span className={styles.label} data-chip-label="">
          <SwapText text={label} />
        </span>
      )}
      {dot !== undefined ? <span className={styles.dot} data-tone={dot} aria-hidden="true" /> : null}
    </button>
  );
  return compact ? <Tooltip content={label}>{chip}</Tooltip> : chip;
});
