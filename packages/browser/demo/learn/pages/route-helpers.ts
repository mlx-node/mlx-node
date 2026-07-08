// route-helpers.ts — shared beforeLoad validation for the chapter routes.
//
// The English layout (routes/chapters.$chapterId.tsx) and the /zh mirror
// (routes/zh.chapters.$chapterId.tsx) validate the same way; the only
// difference is the locale, which picks the display-string overlay for the
// { chapter } put into route context (so titles downstream are localized) and
// the locale-prefixed redirect target on a miss.

import { redirect } from '@tanstack/react-router';

import { localePath, type Locale } from '../../lib/i18n';
import type { ChapterMeta, SectionMeta } from '../chapters';
import { localizedChapter } from '../i18n/localized';

/** Chapter-layout beforeLoad: validate chapterId, return locale-resolved context. */
export function resolveChapterContext(locale: Locale, chapterId: string): { chapter: ChapterMeta } {
  const chapter = localizedChapter(locale, chapterId);
  if (!chapter) {
    throw redirect({ to: localePath(locale, '/chapters') as never });
  }
  return { chapter };
}

/** Section-route beforeLoad: validate sectionId against the (already locale-resolved) chapter. */
export function resolveSectionContext(
  locale: Locale,
  chapter: ChapterMeta,
  sectionId: string,
): { section: SectionMeta } {
  const section = chapter.sections?.find((s) => s.id === sectionId);
  if (!section) {
    throw redirect({ to: localePath(locale, `/chapters/${chapter.id}`) as never });
  }
  return { section };
}
