// routes/zh.tsx — layout route for the Chinese subtree (/zh/*).
//
// The ONLY job of this layout is to mount <LocaleProvider locale="zh"> around
// the child routes — every component below reads the locale from context
// (useLocale/useUiStrings/useLocaleNavigate), never from window.location. The
// English routes mount no provider (the context default is 'en').

import { createFileRoute, Outlet } from '@tanstack/react-router';

import { LocaleProvider } from '../lib/i18n-react';

export const Route = createFileRoute('/zh')({
  component: () => (
    <LocaleProvider locale="zh">
      <Outlet />
    </LocaleProvider>
  ),
});
