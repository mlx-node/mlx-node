import type { ReactElement } from 'react';

import { TextShimmer } from '../../arc/text-shimmer/text-shimmer';
import { useRichMotion } from '../motion/use-entrance';
import { SwapText } from '../swap-text/swap-text';
import styles from './live-label.module.css';

export interface LiveLabelProps {
  /** The label only; icons, counts and timers go beside it. */
  children: string;
  /** True only while the latest data reports this work as running. */
  active: boolean;
  /** Default 'span'. */
  as?: 'span' | 'div' | 'p';
  className?: string;
}

/** The one place that mounts Arc TextShimmer: a label of work that is running now. */
export function LiveLabel({ children, active, as = 'span', className }: LiveLabelProps): ReactElement {
  const rich = useRichMotion();
  if (rich) {
    return (
      <TextShimmer
        as={as}
        className={className === undefined ? styles.shimmer : `${styles.shimmer} ${className}`}
        active={active}
      >
        {children}
      </TextShimmer>
    );
  }
  const Tag = as;
  return (
    <Tag className={className}>
      <SwapText text={children} />
    </Tag>
  );
}
