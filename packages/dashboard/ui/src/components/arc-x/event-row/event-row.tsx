import type { ReactElement, ReactNode } from 'react';
import { Link } from 'react-router-dom';

import { cn } from '../../../lib/cn';
import { formatClock } from '../../../lib/format';
import styles from './event-row.module.css';

export interface EventRowProps {
  icon?: ReactNode;
  /** "Subagent · explore". */
  label: string;
  text: string;
  /** Router path; omitted = plain text. */
  to?: string;
  /** Nullish and empty entries are skipped, joined with " · ". */
  facts?: ReadonlyArray<string | null | undefined>;
  tone?: 'neutral' | 'danger';
  error?: string;
  ts?: number;
  className?: string;
  /** Added to the label's own class (an ink of its own). */
  labelClassName?: string;
}

export interface RuleLabelProps {
  children: ReactNode;
  tone?: 'neutral' | 'danger';
  /** Shown centered under the rule (an error text). */
  detail?: string;
  className?: string;
}

/** One event of a transcript in a single line: icon, label, text or link, facts, time; an error under it. */
export function EventRow(props: EventRowProps): ReactElement {
  const facts = (props.facts ?? []).filter(
    (fact): fact is string => fact !== null && fact !== undefined && fact !== '',
  );
  const timed = props.ts !== undefined && props.ts > 0;
  return (
    <div className={cn(styles.row, props.className)} data-tone={props.tone ?? 'neutral'}>
      <div className={styles.line}>
        {props.icon !== undefined ? (
          <span className={styles.icon} aria-hidden="true">
            {props.icon}
          </span>
        ) : null}
        <span className={cn(styles.label, props.labelClassName)}>{props.label}</span>
        {props.to !== undefined ? (
          <Link className={cn('link', styles.text)} to={props.to} title={props.text}>
            {props.text}
          </Link>
        ) : (
          <span className={styles.text} title={props.text}>
            {props.text}
          </span>
        )}
        {facts.length > 0 ? <span className={styles.facts}>{facts.join(' · ')}</span> : null}
        {timed && props.ts !== undefined ? (
          <time
            className={styles.time}
            dateTime={new Date(props.ts).toISOString()}
            title={new Date(props.ts).toISOString()}
          >
            {formatClock(props.ts)}
          </time>
        ) : null}
      </div>
      {props.error !== undefined && props.error !== '' ? <p className={styles.error}>{props.error}</p> : null}
    </div>
  );
}

/** A hairline across the transcript with a short centered label (a model switch, a turn end). */
export function RuleLabel(props: RuleLabelProps): ReactElement {
  return (
    <div className={cn(styles.ruleRoot, props.className)} data-tone={props.tone ?? 'neutral'}>
      <div className={styles.rule} role="separator">
        <span className={styles.ruleText}>{props.children}</span>
      </div>
      {props.detail !== undefined && props.detail !== '' ? <p className={styles.detail}>{props.detail}</p> : null}
    </div>
  );
}
