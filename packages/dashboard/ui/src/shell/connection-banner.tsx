import type { ReactElement } from 'react';

import { Alert } from '../components/arc/alert/alert';
import styles from './app-shell.module.css';

/** At the top of `<main>` while the RPC port is down; pages keep their last data under it. */
export function ConnectionBanner({ connected }: { connected: boolean }): ReactElement | null {
  if (connected) return null;
  return (
    <div className={styles.banner}>
      <Alert tone="warning" title="Reconnecting to the control panel service…" />
    </div>
  );
}
