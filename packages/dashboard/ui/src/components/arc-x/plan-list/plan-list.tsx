import { Check } from 'lucide-react';
import type { ReactElement } from 'react';

import { cn } from '../../../lib/cn';
import { formatClock } from '../../../lib/format';
import styles from './plan-list.module.css';

export interface PlanItem {
  content: string;
  status: string;
  priority?: string;
}

export interface PlanListProps {
  items: readonly PlanItem[];
  ts?: number;
  className?: string;
}

function Glyph({ status }: { status: string }): ReactElement {
  if (status === 'completed') {
    return (
      <span className={styles.glyph} data-status={status} role="img" aria-label={status}>
        <Check size={14} strokeWidth={2} aria-hidden="true" />
      </span>
    );
  }
  return (
    <span className={styles.glyph} data-status={status} role="img" aria-label={status}>
      <svg viewBox="0 0 14 14" width={14} height={14} aria-hidden="true">
        <circle cx={7} cy={7} r={5.25} />
        {status === 'in_progress' ? <circle className={styles.dot} cx={7} cy={7} r={2} /> : null}
      </svg>
    </span>
  );
}

/** An agent's plan: one line per item with its state. */
export function PlanList({ items, ts, className }: PlanListProps): ReactElement {
  const timed = ts !== undefined && ts > 0;
  return (
    <section className={cn(styles.root, className)} aria-label="Plan">
      <div className={styles.header}>
        <span className={styles.heading}>Plan</span>
        {timed ? (
          <time className={styles.time} dateTime={new Date(ts).toISOString()} title={new Date(ts).toISOString()}>
            {formatClock(ts)}
          </time>
        ) : null}
      </div>
      <ul className={styles.list}>
        {items.map((item, index) => (
          <li key={index} className={styles.item} data-status={item.status}>
            <Glyph status={item.status} />
            <span className={styles.text}>{item.content}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
