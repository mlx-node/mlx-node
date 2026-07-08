// i18n-react.tsx — the React side of the locale core: context + provider +
// hook. Kept separate so lib/i18n.ts stays React-free (importable by pure
// modules like seo-metadata.ts).
//
// The provider is mounted once per locale subtree: the English routes never
// mount one (the context default is 'en'), the /zh route layout mounts
// <LocaleProvider locale="zh">, and the prerender entry wraps each page in the
// locale it is rendering. Components read the locale with useLocale() and must
// never re-derive it from window.location.

import * as React from 'react';
import { DEFAULT_LOCALE, localePath, type Locale } from './i18n';

const LocaleContext = React.createContext<Locale>(DEFAULT_LOCALE);

export function LocaleProvider({ locale, children }: { locale: Locale; children: React.ReactNode }) {
  return <LocaleContext.Provider value={locale}>{children}</LocaleContext.Provider>;
}

export function useLocale(): Locale {
  return React.useContext(LocaleContext);
}

/** Locale-prefixed href for the current subtree: useHref('/chapters') → '/zh/chapters' under /zh. */
export function useHref(path: string): string {
  return localePath(useLocale(), path);
}
