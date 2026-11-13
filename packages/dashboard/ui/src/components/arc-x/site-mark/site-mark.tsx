import { Globe } from 'lucide-react';
import type { ReactElement } from 'react';

import { cn } from '../../../lib/cn';
import styles from './site-mark.module.css';

export interface SiteMarkProps {
  host: string;
  /** px. Default 16. */
  size?: 16 | 20;
  /** Overrides the host's first letter (provider tiles). */
  letter?: string;
  className?: string;
}

function isLetter(ch: string): boolean {
  return ch.toLowerCase() !== ch.toUpperCase();
}

/** A local letter tile for a site: never a fetched favicon. A host that starts with a digit or a symbol gets a globe. */
export function SiteMark({ host, size = 16, letter, className }: SiteMarkProps): ReactElement {
  const first = (letter ?? host).charAt(0);
  return (
    <span className={cn(styles.root, className)} data-size={size} aria-hidden="true">
      {first !== '' && isLetter(first) ? first.toUpperCase() : <Globe className={styles.globe} strokeWidth={1.75} />}
    </span>
  );
}
