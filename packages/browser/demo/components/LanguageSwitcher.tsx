// LanguageSwitcher — the EN | 中文 toggle mounted in the LessonLayout header,
// the ChapterIndex top bar, and the landing screen.
//
// On switch it maps the CURRENT pathname to its counterpart in the other
// locale (stripLocalePrefix + localePath — '/zh/chapters/rope' ↔
// '/chapters/rope'), persists the choice (read back by future sessions; there
// is deliberately NO auto-redirect by browser language), and navigates,
// preserving search params. The active locale comes from React context
// (useLocale), never from window.location.

import { useNavigate, useRouterState } from '@tanstack/react-router';
import * as React from 'react';

import { LOCALES, LOCALE_LABELS, localePath, storeLocale, stripLocalePrefix, type Locale } from '../lib/i18n';
import { useLocale } from '../lib/i18n-react';
import { UI_STRINGS } from '../learn/i18n/ui';

export function LanguageSwitcher({ className }: { className?: string }) {
  const locale = useLocale();
  const navigate = useNavigate();
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  const switchTo = (target: Locale) => {
    if (target === locale) return;
    const { path } = stripLocalePrefix(pathname);
    storeLocale(target);
    void navigate({ to: localePath(target, path) as never, search: (prev: unknown) => prev as never });
  };

  return (
    <div
      role="group"
      aria-label={UI_STRINGS[locale].languageSwitcher.ariaLabel}
      className={['inline-flex items-center gap-1', className ?? ''].join(' ')}
    >
      {LOCALES.map((l) => {
        const active = l === locale;
        return (
          <button
            key={l}
            type="button"
            lang={l === 'zh' ? 'zh-CN' : 'en'}
            aria-pressed={active}
            onClick={() => switchTo(l)}
            className={[
              // 44px tap target on touch/small screens, compact (h-8, matching the
              // header's other controls) from sm up.
              'inline-flex min-h-11 items-center justify-center rounded-md border px-3 py-1 text-xs font-medium transition-colors outline-none sm:min-h-8',
              'focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50',
              active
                ? 'border-primary/60 bg-primary/10 text-foreground'
                : 'border-border bg-background text-muted-foreground hover:bg-accent hover:text-accent-foreground',
            ].join(' ')}
          >
            {LOCALE_LABELS[l]}
          </button>
        );
      })}
    </div>
  );
}
