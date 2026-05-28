// routes/chapters.$chapterId.tsx — Single chapter route (/chapters/:chapterId).
// Phase 2.A skeleton: beforeLoad validates chapterId and redirects to /chapters
// on miss. Placeholder component only — real LessonLayout wired in Phase 2.C.

import { createFileRoute, redirect } from '@tanstack/react-router';

import { findChapter } from '../learn/chapters';

export const Route = createFileRoute('/chapters/$chapterId')({
  beforeLoad: ({ params }) => {
    const chapter = findChapter(params.chapterId);
    if (!chapter) {
      throw redirect({ to: '/chapters' });
    }
    return { chapter };
  },
  component: function ChapterRoute() {
    const { chapter } = Route.useRouteContext();
    return <div>Chapter {chapter.id} route</div>;
  },
});
