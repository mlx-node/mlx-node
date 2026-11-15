import { MotionConfig } from 'motion/react';
import type { ReactElement } from 'react';
import { BrowserRouter, Route, Routes } from 'react-router-dom';

import { ArcProvider } from './components/arc/lib/arc-provider';
import { ToastStack, ToastStackProvider } from './components/arc/toast-stack/toast-stack';
import { ToastBridge } from './lib/toast';
import Models from './pages/models';
import { AppShell } from './shell/app-shell';
import { ArcLink } from './shell/arc-link';
import { FirstRunGate } from './shell/first-run-gate';
import { APP_ROUTES } from './shell/routes';
import toastStyles from './shell/toasts.module.css';

export default function App(): ReactElement {
  return (
    <MotionConfig reducedMotion="user">
      <ToastStackProvider>
        <ToastBridge />
        <BrowserRouter>
          <ArcProvider link={ArcLink}>
            <Routes>
              <Route path="welcome" element={<Models onboarding />} />
              <Route
                element={
                  <FirstRunGate>
                    <AppShell />
                  </FirstRunGate>
                }
              >
                {APP_ROUTES.map((route) =>
                  route.path === '/' ? (
                    <Route key={route.id} index element={<route.element />} />
                  ) : (
                    <Route key={route.id} path={route.path} element={<route.element />} />
                  ),
                )}
              </Route>
            </Routes>
          </ArcProvider>
        </BrowserRouter>
        <ToastStack position="bottom-right" label="Notifications" className={toastStyles.viewport} />
      </ToastStackProvider>
    </MotionConfig>
  );
}
