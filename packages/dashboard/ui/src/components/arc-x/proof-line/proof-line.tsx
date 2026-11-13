import { Check, Minus } from 'lucide-react';
import type { CSSProperties, ReactElement } from 'react';

import { Badge } from '../../arc/badge/badge';
import { motionX } from '../motion/motion';
import styles from './proof-line.module.css';

export interface ProofLineProps {
  /** 'installed' draws a check, 'removed' a dash. */
  kind: 'installed' | 'removed';
  text: string;
  /** Names shown as code chips after the text. */
  chips?: readonly string[];
}

/** A one-line record of a finished change: it opens its height, then its chips arrive one after another. */
export function ProofLine({ kind, text, chips = [] }: ProofLineProps): ReactElement {
  return (
    <div className={styles.root} data-kind={kind}>
      <div className={styles.clip}>
        <p className={styles.line}>
          {kind === 'installed' ? (
            <Check className={styles.icon} size={14} strokeWidth={2} aria-hidden="true" />
          ) : (
            <Minus className={styles.icon} size={14} strokeWidth={2} aria-hidden="true" />
          )}
          <span>{text}</span>
          {chips.map((name, index) => (
            <span
              key={name}
              className={styles.chip}
              style={{ '--step': Math.min(index, motionX.reveal.maxSteps - 1) } as CSSProperties}
            >
              <Badge size="sm" outline>
                <code className={styles.code}>{name}</code>
              </Badge>
            </span>
          ))}
        </p>
      </div>
    </div>
  );
}
