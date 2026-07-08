// learn/pages/SectionPage.tsx — the shared sub-chapter page, extracted from
// the former routes/chapters.$chapterId.$sectionId.tsx route component so the
// English route and the /zh mirror render the SAME logic.
//
// A "go deeper" deep-dive served as its own route: independently prerendered,
// SEO-indexed, deep-linkable, and shown nested under its parent chapter in the
// sidebar. Pure reading — no "Try it now" panel and NO model auto-load.
//
// `chapter` and `section` arrive already locale-resolved from route context
// (the zh layout's beforeLoad applies the display-string overlay), so the
// breadcrumb/sidebar titles are localized for free.

import { useNavigate } from '@tanstack/react-router';

import { useLocale } from '../../lib/i18n-react';
import { useLocaleNavigate } from '../../lib/locale-navigate';
import { useFreeChat } from '../../providers/free-chat';
import type { ChapterMeta, SectionMeta } from '../chapters';
import { ZH_SECTION_BODIES } from '../i18n/bodies.zh';
import { TranslationPendingNote } from '../i18n/TranslationPendingNote';
import { useUiStrings } from '../i18n/ui-react';
import { LessonLayout } from '../LessonLayout';
import { SECTION_BODIES, sectionBodyKey } from '../section-bodies';

export function SectionPage({ chapter, section }: { chapter: ChapterMeta; section: SectionMeta }) {
  const locale = useLocale();
  const ui = useUiStrings();
  const navigate = useNavigate();
  const go = useLocaleNavigate();
  // Keep the worker provider mounted/consistent with the chapter routes, even
  // though this reading-only page never touches the model.
  useFreeChat();

  // Body resolution mirrors ChapterPage: Chinese override under /zh, falling
  // back to the shared English registry with a pending-translation note.
  const key = sectionBodyKey(chapter.id, section.id);
  const EnglishBody = SECTION_BODIES[key];
  const ZhBody = locale === 'zh' ? ZH_SECTION_BODIES[key] : undefined;
  const Body = ZhBody ?? EnglishBody;
  const showTranslationPending = locale === 'zh' && ZhBody === undefined && Body !== undefined;

  return (
    <LessonLayout
      current={chapter}
      currentSectionId={section.id}
      sectionTitle={section.title}
      // Pure reading — no live demo column, no model load.
      tryItPanel={null}
      onOpenChapter={(chapterId) => {
        go(`/chapters/${chapterId}`);
      }}
      onOpenSection={(chapterId, sectionId) => {
        go(`/chapters/${chapterId}/${sectionId}`);
      }}
      onBackToIndex={() => {
        go('/chapters');
      }}
      onOpenFreeChat={() => {
        // Chat is deliberately NOT locale-prefixed — there is no /zh/chat route.
        void navigate({ to: '/chat', search: (prev) => prev });
      }}
    >
      {Body ? (
        <>
          {showTranslationPending ? <TranslationPendingNote /> : null}
          <Body />
        </>
      ) : (
        <div className="text-muted-foreground">{ui.sectionPage.notAuthored}</div>
      )}
    </LessonLayout>
  );
}
