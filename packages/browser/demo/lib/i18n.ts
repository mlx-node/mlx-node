// i18n.ts — React-free locale core. The single source of truth for which
// locales exist and how they map to URLs.
//
// Importable by BOTH the prerender SSR entry (scripts/prerender.entry.tsx, run
// under node) AND client code — same rule as seo-metadata.ts: no React, no
// worker, no browser-only imports (the localStorage helpers below are
// SSR-guarded).
//
// URL scheme: English is unprefixed (`/chapters/attention`), every other
// locale gets a path prefix (`/zh/chapters/attention`). English stays at the
// bare URLs so existing links/SEO never move.

export type Locale = 'en' | 'zh';

export const LOCALES: readonly Locale[] = ['en', 'zh'] as const;

export const DEFAULT_LOCALE: Locale = 'en';

/** Value for the <html lang> attribute and JSON-LD inLanguage. */
export function htmlLang(locale: Locale): string {
  return locale === 'zh' ? 'zh-CN' : 'en';
}

/** Human label for the language switcher. */
export const LOCALE_LABELS: Record<Locale, string> = {
  en: 'English',
  zh: '中文',
};

/**
 * Prefix an app-internal path ('/chapters/attention', '/') for a locale.
 * English returns the path unchanged; zh returns '/zh/chapters/attention'
 * ('/' → '/zh'). `path` must start with '/'.
 */
export function localePath(locale: Locale, path: string): string {
  if (locale === DEFAULT_LOCALE) return path;
  return path === '/' ? `/${locale}` : `/${locale}${path}`;
}

/**
 * Split a pathname into its locale and the unprefixed app path.
 * '/zh/chapters/rope' → { locale: 'zh', path: '/chapters/rope' };
 * '/chapters/rope'    → { locale: 'en', path: '/chapters/rope' }.
 * Only exact segment matches count ('/zhang' is NOT zh).
 */
export function stripLocalePrefix(pathname: string): { locale: Locale; path: string } {
  for (const locale of LOCALES) {
    if (locale === DEFAULT_LOCALE) continue;
    if (pathname === `/${locale}`) return { locale, path: '/' };
    if (pathname.startsWith(`/${locale}/`)) return { locale, path: pathname.slice(locale.length + 1) };
  }
  return { locale: DEFAULT_LOCALE, path: pathname };
}

// ── Preference persistence (same mlx:* namespace + defensive style as
//    QuickCheck's localStorage handling) ─────────────────────────────────────

export const LOCALE_STORAGE_KEY = 'mlx:preferences:locale';

export function readStoredLocale(): Locale | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(LOCALE_STORAGE_KEY);
    return raw !== null && (LOCALES as readonly string[]).includes(raw) ? (raw as Locale) : null;
  } catch {
    return null; // privacy mode / storage disabled
  }
}

export function storeLocale(locale: Locale): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(LOCALE_STORAGE_KEY, locale);
  } catch {
    // best-effort only
  }
}
