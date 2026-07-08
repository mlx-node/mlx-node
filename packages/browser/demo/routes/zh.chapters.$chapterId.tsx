// routes/zh.chapters.$chapterId.tsx — Chinese layout route for a chapter and
// its sub-chapters (/zh/chapters/:chapterId/*). Mirror of
// routes/chapters.$chapterId.tsx via the shared beforeLoad helper — with
// locale 'zh', so the { chapter } put into route context carries the LOCALIZED
// display strings (title/blurb/section titles) for everything downstream
// (breadcrumb, sidebar, SubChapterNav).

import { createFileRoute, Outlet } from '@tanstack/react-router';

import { resolveChapterContext } from '../learn/pages/route-helpers';

export const Route = createFileRoute('/zh/chapters/$chapterId')({
  beforeLoad: ({ params }) => resolveChapterContext('zh', params.chapterId),
  component: () => <Outlet />,
});
