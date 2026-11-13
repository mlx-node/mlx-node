import { Check } from 'lucide-react';
import type { ReactElement, ReactNode } from 'react';

import { cn } from '../../../lib/cn';
import { Placeholder } from '../placeholder/placeholder';
import styles from './choice-card.module.css';

export interface ChoiceCardProps {
  name: string;
  value: string;
  checked: boolean;
  disabled?: boolean;
  onChange(value: string): void;
  ariaLabel: string;
  describedBy?: string;
  media?: ReactNode;
  badge?: ReactNode;
  title: string;
  /** Rendered in a <p>; the caller passes the id used by describedBy through descriptionId. */
  description: ReactNode;
  descriptionId?: string;
  meta?: ReactNode;
  warning?: ReactNode;
  loading?: boolean;
  className?: string;
}

/** A radio as a card: media, badge, title, description, meta and a warning; the whole card is the hit target. */
export function ChoiceCard(props: ChoiceCardProps): ReactElement {
  if (props.loading === true) {
    return (
      <div className={cn(styles.root, props.className)} aria-hidden="true" data-loading="">
        <div className={styles.top}>
          <Placeholder width={28} height={28} radius="pill" />
          <span className={styles.mark} />
        </div>
        <div className={styles.badge} />
        <Placeholder width="60%" />
        <div className={styles.description}>
          <Placeholder />
          <Placeholder width="80%" />
        </div>
        <Placeholder width="40%" />
      </div>
    );
  }
  const disabled = props.disabled === true;
  return (
    <label
      className={cn(styles.root, props.className)}
      data-selected={props.checked ? '' : undefined}
      data-blocked={disabled ? '' : undefined}
    >
      <input
        className={styles.input}
        type="radio"
        name={props.name}
        value={props.value}
        checked={props.checked}
        disabled={disabled}
        aria-label={props.ariaLabel}
        aria-describedby={props.describedBy}
        onChange={() => props.onChange(props.value)}
      />
      <div className={styles.top}>
        <span className={styles.media}>{props.media}</span>
        <span className={styles.mark} aria-hidden="true">
          {props.checked ? <Check className={styles.check} size={11} strokeWidth={2.5} /> : null}
        </span>
      </div>
      <div className={styles.badge}>{props.badge}</div>
      <span className={styles.title}>{props.title}</span>
      <p id={props.descriptionId} className={styles.description}>
        {props.description}
      </p>
      {props.meta !== undefined && props.meta !== null ? <span className={styles.meta}>{props.meta}</span> : null}
      {props.warning !== undefined && props.warning !== null ? (
        <span className={styles.warning}>{props.warning}</span>
      ) : null}
    </label>
  );
}
