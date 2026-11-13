import type { ReactElement } from 'react';

import { cn } from '../../../lib/cn';
import { LiveLabel } from '../live-label/live-label';
import styles from './live-dot.module.css';

export interface LiveDotProps {
  tone: 'success' | 'warning' | 'danger' | 'accent' | 'neutral';
  pulse?: boolean;
  /** Always present for assistive tech; visible only with showLabel. */
  label: string;
  showLabel?: boolean;
  /** px. Default 8. */
  size?: 6 | 8;
  /** With showLabel: the label shimmers while this is true (a state that is work in progress). */
  shimmer?: boolean;
  className?: string;
}

/** A status dot that is always paired with its words. */
export function LiveDot({
  tone,
  pulse = false,
  label,
  showLabel = false,
  size = 8,
  shimmer = false,
  className,
}: LiveDotProps): ReactElement {
  return (
    <span
      className={cn(styles.root, className)}
      data-tone={tone}
      data-size={size}
      role={showLabel ? undefined : 'img'}
      aria-label={showLabel ? undefined : label}
    >
      <span className={styles.dot} data-pulse={pulse ? '' : undefined} aria-hidden="true" />
      {showLabel ? (
        <LiveLabel active={shimmer} className={styles.label}>
          {label}
        </LiveLabel>
      ) : null}
    </span>
  );
}
