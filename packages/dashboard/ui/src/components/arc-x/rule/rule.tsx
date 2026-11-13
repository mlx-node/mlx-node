import type { ReactElement } from 'react';

import { cn } from '../../../lib/cn';
import styles from './rule.module.css';

export interface RuleProps {
  className?: string;
  /** 'strong' on raised and floating layers, where --border-subtle is about the surface's own lightness. */
  tone?: 'subtle' | 'strong';
}

/** A 1 px hairline; the parent's gap spaces it. */
export function Rule({ className, tone = 'subtle' }: RuleProps): ReactElement {
  return <hr className={cn(styles.root, className)} data-tone={tone} />;
}
