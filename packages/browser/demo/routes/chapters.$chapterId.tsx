// routes/chapters.$chapterId.tsx — layout route for a chapter and its
// sub-chapters (/chapters/:chapterId/*).
//
// beforeLoad validates chapterId once (shared helper, also used by the /zh
// mirror) and puts the resolved { chapter } into route context. Both children
// inherit that context (no re-resolve):
//   - chapters.$chapterId.index.tsx  → the chapter page (/chapters/:chapterId)
//   - chapters.$chapterId.$sectionId.tsx → a sub-chapter (/chapters/:chapterId/:sectionId)
// The layout itself renders only <Outlet />.

import { createFileRoute, Outlet } from '@tanstack/react-router';

import { resolveChapterContext } from '../learn/pages/route-helpers';

export const Route = createFileRoute('/chapters/$chapterId')({
  beforeLoad: ({ params }) => resolveChapterContext('en', params.chapterId),
  component: () => <Outlet />,
});
