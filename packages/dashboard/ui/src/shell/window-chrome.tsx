import type { ReactElement } from 'react';

import styles from './app-shell.module.css';

/** The 40 px drag strip of a full-window page with no shell (`/welcome`); the first 78 px belong to the traffic lights. */
export function WindowDragStrip(): ReactElement {
  return <div className={styles.dragStrip} aria-hidden="true" />;
}
