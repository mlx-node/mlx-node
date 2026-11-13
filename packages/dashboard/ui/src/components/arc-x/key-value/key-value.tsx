import type { ReactElement, ReactNode } from 'react';

import { cn } from '../../../lib/cn';
import { NO_VALUE } from '../../../lib/format';
import { CopyButton } from '../../arc/copy-button/copy-button';
import { Placeholder } from '../placeholder/placeholder';
import { SwapText } from '../swap-text/swap-text';
import styles from './key-value.module.css';

export interface KeyValueItem {
  key: string;
  label: string;
  /** Hover text of the label. */
  labelTitle?: string;
  /** null -> "—" */
  value: ReactNode | null;
  /** Muted text after the value. */
  hint?: string;
  /** The hint wraps onto more lines instead of ending in an ellipsis. */
  wrapHint?: boolean;
  /** Paths, ids. */
  mono?: boolean;
  /** Shows a copy button for this text. */
  copy?: string;
  /** tabular-nums, right-aligned in 'rows' layout. */
  numeric?: boolean;
}

export interface KeyValueProps {
  items: readonly KeyValueItem[];
  /** grid (default): label above value in auto-fit columns; rows: label left, value right, hairlines. */
  layout?: 'grid' | 'rows';
  loading?: boolean;
  ariaLabel?: string;
  className?: string;
}

function valueOf(value: ReactNode | null): ReactNode {
  if (value === null || value === undefined) return NO_VALUE;
  if (typeof value === 'string') return <SwapText text={value} />;
  return value;
}

/** A definition list of labelled values; values hand over from their loading boxes one after another. */
export function KeyValue({
  items,
  layout = 'grid',
  loading = false,
  ariaLabel,
  className,
}: KeyValueProps): ReactElement {
  return (
    <dl className={cn(styles.root, className)} data-layout={layout} aria-label={ariaLabel}>
      {items.map((item, index) => (
        <div key={item.key} className={styles.row}>
          <dt className={styles.label} title={item.labelTitle}>
            {item.label}
          </dt>
          <dd
            className={styles.value}
            data-mono={item.mono === true ? '' : undefined}
            data-numeric={item.numeric === true ? '' : undefined}
            data-copy={item.copy !== undefined ? '' : undefined}
          >
            <Placeholder loading={loading} width="60%" step={index} className={styles.slot}>
              <span className={styles.text}>{valueOf(item.value)}</span>
              {item.hint !== undefined ? (
                <span
                  className={styles.hint}
                  data-wrap={item.wrapHint === true ? '' : undefined}
                  title={item.wrapHint === true ? undefined : item.hint}
                >
                  {item.hint}
                </span>
              ) : null}
              {item.copy !== undefined ? (
                <span className={styles.copy}>
                  <CopyButton value={item.copy} label={`Copy ${item.label}`} iconOnly variant="plain" />
                </span>
              ) : null}
            </Placeholder>
          </dd>
        </div>
      ))}
    </dl>
  );
}
