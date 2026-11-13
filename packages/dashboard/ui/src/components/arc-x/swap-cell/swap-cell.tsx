import { type ReactElement, type ReactNode, useState } from 'react';

import { cn } from '../../../lib/cn';
import styles from './swap-cell.module.css';

export interface SwapCellProps {
  /** The key of the item shown. A new key swaps: the new item fades in with a small rise. */
  active: string;
  /** Every item is laid out in the one cell, so the cell keeps the height of the tallest; the others are hidden and inert. */
  items: Readonly<Record<string, ReactNode>>;
  className?: string;
}

/** One grid cell whose content swaps in place without changing its size. The first content of a mount appears without motion. */
export function SwapCell({ active, items, className }: SwapCellProps): ReactElement {
  const [seen, setSeen] = useState({ key: active, swapped: false });
  if (seen.key !== active) setSeen({ key: active, swapped: true });
  return (
    <div className={cn(styles.root, className)}>
      {Object.entries(items).map(([key, item]) => {
        const shown = key === active;
        return (
          <div
            key={key}
            className={styles.item}
            data-enter={shown && seen.swapped ? '' : undefined}
            data-hidden={shown ? undefined : ''}
            aria-hidden={shown ? undefined : true}
            inert={!shown}
          >
            {item}
          </div>
        );
      })}
    </div>
  );
}
