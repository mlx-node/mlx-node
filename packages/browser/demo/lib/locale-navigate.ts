// locale-navigate.ts — locale-aware navigation for the shared page components.
//
// The shared pages (learn/pages/*) render under BOTH the unprefixed English
// routes and the /zh subtree, so their navigation handlers cannot use the
// hardcoded typed `to: '/chapters/$chapterId'` patterns — the target path must
// be locale-prefixed at runtime via localePath(). TanStack Router's typed `to`
// only accepts literal route patterns, so the computed string is cast at the
// `to` value only (runtime navigation by concrete path is fully supported; the
// cast is purely a typing concession and does not weaken any other types).

import { useNavigate } from '@tanstack/react-router';
import * as React from 'react';

import { localePath } from './i18n';
import { useLocale } from './i18n-react';

/**
 * Returns `go(path)`: navigates to the locale-prefixed form of an app-internal
 * path ('/chapters/attention' → '/zh/chapters/attention' under /zh), always
 * preserving the current search params (the `search: (prev) => prev` pattern
 * used by every route in this app).
 */
export function useLocaleNavigate(): (path: string) => void {
  const navigate = useNavigate();
  const locale = useLocale();
  return React.useCallback(
    (path: string) => {
      void navigate({ to: localePath(locale, path) as never, search: (prev: unknown) => prev as never });
    },
    [navigate, locale],
  );
}
