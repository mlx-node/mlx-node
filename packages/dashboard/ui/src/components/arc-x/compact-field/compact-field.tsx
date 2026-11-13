import type { ReactElement, ReactNode } from 'react';

import { cn } from '../../../lib/cn';
import styles from './compact-field.module.css';

export interface CompactFieldProps {
  /** One Arc Select, SearchField or Input; its first label is hidden visually and keeps naming the control. */
  children: ReactNode;
  className?: string;
}

/** A field for toolbars and page-header controls: the label stays the accessible name but takes no space. */
export function CompactField({ children, className }: CompactFieldProps): ReactElement {
  return <div className={cn(styles.compact, className)}>{children}</div>;
}
