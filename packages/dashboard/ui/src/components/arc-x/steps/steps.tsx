import { Check } from 'lucide-react';
import type { ReactElement } from 'react';

import { cn } from '../../../lib/cn';
import styles from './steps.module.css';

export interface StepsProps {
  label: string;
  steps: readonly string[];
  /** 0-based. */
  current: number;
  className?: string;
}

/** Numbered steps joined by hairlines; done steps show a check. Narrow: only the current label is shown. */
export function Steps({ label, steps, current, className }: StepsProps): ReactElement {
  return (
    <div className={cn(styles.root, className)}>
      <ol className={styles.list} aria-label={label}>
        {steps.map((step, index) => {
          const complete = index < current;
          const isCurrent = index === current;
          return (
            <li
              key={step}
              className={styles.step}
              aria-current={isCurrent ? 'step' : undefined}
              data-complete={complete ? '' : undefined}
              data-current={isCurrent ? '' : undefined}
            >
              {index > 0 ? <span className={styles.line} aria-hidden="true" /> : null}
              <span className={styles.dot} aria-hidden="true">
                {complete ? (
                  <Check key="check" className={styles.check} size={11} strokeWidth={2.5} />
                ) : (
                  <span key="digit" className={styles.digit}>
                    {index + 1}
                  </span>
                )}
              </span>
              <span className={styles.label}>{step}</span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
