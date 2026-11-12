import { useId, type ReactElement, type ReactNode } from 'react';

import { cn } from '../../../lib/cn';
import styles from './panel.module.css';

export interface PanelProps {
  /** h2/h3 14 px/500; omitted -> bare surface. */
  title?: string;
  /** Inline after the title, inside the heading: a live state word. */
  titleSuffix?: ReactNode;
  /** 13 px secondary text under the title. */
  description?: string;
  /** Right of the header. */
  actions?: ReactNode;
  /** 16 px | 0 (tables bring their own). Default 'default'. */
  padding?: 'default' | 'none';
  /** Default 2. */
  headingLevel?: 2 | 3;
  /** Accessible name when there is no title. */
  ariaLabel?: string;
  busy?: boolean;
  children: ReactNode;
  className?: string;
}

/** The card surface: border, surface radius, no shadow, clipped. Labelled by its title. */
export function Panel(props: PanelProps): ReactElement {
  const titleId = useId();
  const hasTitle = props.title !== undefined && props.title.length > 0;
  const Heading = props.headingLevel === 3 ? 'h3' : 'h2';
  const hasHeader = hasTitle || props.actions !== undefined;
  return (
    <section
      className={cn(styles.root, props.className)}
      data-padding={props.padding ?? 'default'}
      aria-labelledby={hasTitle ? titleId : undefined}
      aria-label={hasTitle ? undefined : props.ariaLabel}
      aria-busy={props.busy === true ? true : undefined}
    >
      {hasHeader ? (
        <header className={styles.header}>
          <div className={styles.heading}>
            {hasTitle ? (
              <Heading id={titleId} className={styles.title}>
                {props.title}
                {props.titleSuffix}
              </Heading>
            ) : null}
            {props.description !== undefined ? <p className={styles.description}>{props.description}</p> : null}
          </div>
          {props.actions !== undefined ? <div className={styles.actions}>{props.actions}</div> : null}
        </header>
      ) : null}
      {props.children}
    </section>
  );
}
