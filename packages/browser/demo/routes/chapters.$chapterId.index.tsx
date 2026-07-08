// routes/chapters.$chapterId.index.tsx — the chapter page (/chapters/:chapterId).
//
// Thin wrapper: the full page logic (model auto-load effects, demo-panel
// wiring, LessonLayout) lives in the shared, locale-agnostic
// learn/pages/ChapterPage.tsx — also rendered by the /zh mirror. This route
// just reads the { chapter } the parent layout resolved into context.

import { createFileRoute } from '@tanstack/react-router';

import { ChapterPage } from '../learn/pages/ChapterPage';

function ChapterRouteComponent() {
  const { chapter } = Route.useRouteContext();
  return <ChapterPage chapter={chapter} />;
}

export const Route = createFileRoute('/chapters/$chapterId/')({
  component: ChapterRouteComponent,
});
