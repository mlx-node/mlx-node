import { AnimatePresence, motion } from 'motion/react';
import type { ReactElement, ReactNode } from 'react';

import { motionTokens } from '../../arc/motion-tokens';
import { useStaticMotion } from '../motion/use-static-motion';
import styles from './composer-tray.module.css';

export interface ComposerTrayProps {
  /** The composer card. */
  children: ReactNode;
  /** The strip's chips, or null while it is not shown. Each chip names its part with `data-tray-part`. */
  tray: ReactNode | null;
  /** The strip's accessible name. */
  label: string;
}

const INSTANT = { duration: 0 } as const;

/** A strip under the composer card for the chips of a new conversation; it opens and closes with the card. */
export function ComposerTray({ children, tray, label }: ComposerTrayProps): ReactElement {
  const staticMotion = useStaticMotion();
  return (
    <div className={styles.root}>
      <div className={styles.card}>{children}</div>
      <div className={styles.pocket}>
        <AnimatePresence initial={false}>
          {tray !== null ? (
            <motion.div
              key="tray"
              className={styles.clip}
              initial={staticMotion ? { opacity: 0 } : { height: 0 }}
              animate={staticMotion ? { opacity: 1 } : { height: 'auto' }}
              exit={staticMotion ? { opacity: 0 } : { height: 0 }}
              transition={staticMotion ? INSTANT : motionTokens.spring.smooth}
            >
              <div className={styles.tray} role="group" aria-label={label}>
                {tray}
              </div>
            </motion.div>
          ) : null}
        </AnimatePresence>
      </div>
    </div>
  );
}
