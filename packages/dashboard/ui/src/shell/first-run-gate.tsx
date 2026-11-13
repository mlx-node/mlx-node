import { type ReactElement, type ReactNode, useState } from 'react';
import { Navigate, useLocation } from 'react-router-dom';

import { hasDismissedOnboarding } from '../lib/onboarding';
import type { ModelsResponse } from '../lib/types';
import { useJson } from '../lib/use-api';
import styles from './app-shell.module.css';

/**
 * Entry at `/` with an empty model library (not dismissed for that library) goes to `/welcome`. Decided once, on
 * entry: a user already in the workspace who deletes the last model stays where they are.
 */
export function FirstRunGate({ children }: { children: ReactNode }): ReactElement {
  const { pathname } = useLocation();
  const [checking, setChecking] = useState(pathname === '/');
  const models = useJson<ModelsResponse>('/models');
  if (checking) {
    if (models.loading || models.refreshing) {
      return (
        <div className={styles.boot} role="status">
          <span className={styles.bootSpinner} aria-hidden="true" />
          Opening your workspace…
        </div>
      );
    }
    if (
      models.error === undefined &&
      models.data !== undefined &&
      models.data.dir.length > 0 &&
      models.data.models.length === 0 &&
      !hasDismissedOnboarding(models.data.dir)
    ) {
      return <Navigate to="/welcome" replace />;
    }
    setChecking(false);
  }
  return <>{children}</>;
}
