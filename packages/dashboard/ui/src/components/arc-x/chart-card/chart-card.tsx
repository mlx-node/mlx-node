import { useId, type ReactElement, type ReactNode } from 'react';

import { cn } from '../../../lib/cn';
import { Button } from '../../arc/button/button';
import { ChartHeldContext } from '../chart/chart-frame';
import { Placeholder } from '../placeholder/placeholder';
import styles from './chart-card.module.css';

/** 'held': the last answer while a new question loads. */
export type ChartState = 'ready' | 'held' | 'loading' | 'empty' | 'error';

export interface ChartCardProps {
  title: string;
  /** States aggregation + unit: "Median per hour · tok/s". Required: a chart without a stated unit is not allowed. */
  subtitle: string;
  /** Segmented metric switch, menu. */
  actions?: ReactNode;
  state: ChartState;
  /** Plot box height in px, same in every state. Default 200. */
  height?: number;
  empty?: { title: string; hint?: string };
  error?: { message: string; retry?: () => void };
  /** Scope and sample note under the plot: "412 turns · this cache only". */
  footnote?: ReactNode;
  /** The chart; rendered only when state is 'ready' or 'held'. */
  children: ReactNode;
  className?: string;
}

/** Room the chart frame takes around its plot box: the unit row above, the x labels below. */
export const CHART_CHROME_PX = 44;

/** The card around one chart: a fixed-height body, so switching state never moves the page. */
export function ChartCard(props: ChartCardProps): ReactElement {
  const titleId = useId();
  const height = props.height ?? 200;
  const box = height + CHART_CHROME_PX;
  const held = props.state === 'held';
  const busy = props.state === 'loading' || held;

  let body: ReactNode;
  if (props.state === 'empty') {
    body = (
      <div className={styles.message} style={{ height: box }}>
        <span className={styles.messageTitle}>{props.empty?.title ?? 'No data'}</span>
        {props.empty?.hint !== undefined ? <span className={styles.messageHint}>{props.empty.hint}</span> : null}
      </div>
    );
  } else if (props.state === 'error') {
    body = (
      <div className={styles.message} style={{ height: box }} role="alert">
        <span className={styles.messageTitle}>{props.error?.message ?? 'Could not load this chart'}</span>
        {props.error?.retry !== undefined ? (
          <Button variant="secondary" size="sm" onClick={props.error.retry}>
            Try again
          </Button>
        ) : null}
      </div>
    );
  } else {
    body = (
      <Placeholder loading={props.state === 'loading'} height={box} radius="control">
        <ChartHeldContext.Provider value={held}>
          <div className={styles.chart}>{props.children}</div>
        </ChartHeldContext.Provider>
      </Placeholder>
    );
  }

  return (
    <section
      className={cn(styles.root, props.className)}
      aria-labelledby={titleId}
      aria-busy={busy ? true : undefined}
      data-state={props.state}
      data-held={held ? '' : undefined}
    >
      <header className={styles.header}>
        <div className={styles.heading}>
          <h3 id={titleId} className={styles.title}>
            {props.title}
          </h3>
          <p className={styles.subtitle}>{props.subtitle}</p>
        </div>
        {props.actions !== undefined ? <div className={styles.actions}>{props.actions}</div> : null}
      </header>
      <div className={styles.body} style={{ minHeight: box }}>
        {body}
      </div>
      {props.footnote !== undefined && props.footnote !== null ? (
        <div className={styles.footnote}>{props.footnote}</div>
      ) : null}
    </section>
  );
}
