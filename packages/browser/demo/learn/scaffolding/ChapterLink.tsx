import { Link } from '@tanstack/react-router';
import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * Locale-aware cross-chapter link for use INSIDE chapter bodies and widgets.
 * Chapter bodies render under both /chapters/* and /zh/chapters/* (the /zh
 * tree falls back to the English body until its translation lands), so a
 * hardcoded <Link to="/chapters/$chapterId"> would silently eject a /zh reader
 * back to the English route. This wrapper picks the route for the current
 * locale; props otherwise mirror the plain Link usage it replaces
 * (search is preserved via `(prev) => prev`, same as every body link).
 */
export function ChapterLink({
  chapterId,
  className,
  children,
}: {
  chapterId: string;
  className?: string;
  children: React.ReactNode;
}) {
  const locale = useLocale();
  return locale === 'zh' ? (
    <Link to="/zh/chapters/$chapterId" params={{ chapterId }} search={(prev) => prev} className={className}>
      {children}
    </Link>
  ) : (
    <Link to="/chapters/$chapterId" params={{ chapterId }} search={(prev) => prev} className={className}>
      {children}
    </Link>
  );
}
