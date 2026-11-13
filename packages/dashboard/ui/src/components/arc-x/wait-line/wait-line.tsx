import type { ReactElement } from 'react';

import { LiveLabel } from '../live-label/live-label';
import { WaitOrb, type WaitOrbState } from '../wait-orb/wait-orb';
import styles from './wait-line.module.css';

export interface WaitLineProps {
  /** Which wait this is. */
  orb: WaitOrbState;
  /** The status. Real text: the orb beside it is decoration. */
  text: string;
  /** It arrived live (not from history): it rises in once. */
  enter?: boolean;
}

/** The pending row of a turn with no answer yet: an orb and the status, one line high unless the text wraps. */
export function WaitLine({ orb, text, enter }: WaitLineProps): ReactElement {
  return (
    <article
      className={styles.root}
      aria-label="Assistant"
      aria-busy="true"
      data-wait-line={orb}
      data-enter={enter === true ? '' : undefined}
    >
      <WaitOrb state={orb} />
      <LiveLabel active className={styles.text}>
        {text}
      </LiveLabel>
    </article>
  );
}
