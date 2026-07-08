// routes/zh.chapters.$chapterId.index.tsx — the Chinese chapter page
// (/zh/chapters/:chapterId). Same shared <ChapterPage /> as the English route;
// `chapter` comes from the zh layout's context already locale-resolved, and
// the body falls back to English (+ <TranslationPendingNote />) until the
// chapter is registered in learn/i18n/bodies.zh.ts.

import { createFileRoute } from '@tanstack/react-router';

import { ChapterPage } from '../learn/pages/ChapterPage';

function ZhChapterRouteComponent() {
  const { chapter } = Route.useRouteContext();
  return <ChapterPage chapter={chapter} />;
}

export const Route = createFileRoute('/zh/chapters/$chapterId/')({
  component: ZhChapterRouteComponent,
});
