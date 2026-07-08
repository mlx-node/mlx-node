// routes/zh.chapters.$chapterId.$sectionId.tsx — a Chinese sub-chapter page
// (/zh/chapters/:chapterId/:sectionId). Mirror of
// routes/chapters.$chapterId.$sectionId.tsx via the shared helpers; the
// { chapter } from the zh layout context is already locale-resolved, so the
// section validated out of it carries the Chinese title/blurb.

import { createFileRoute } from '@tanstack/react-router';

import type { ChapterMeta } from '../learn/chapters';
import { SectionPage } from '../learn/pages/SectionPage';
import { resolveSectionContext } from '../learn/pages/route-helpers';

function ZhSectionRouteComponent() {
  const { chapter, section } = Route.useRouteContext();
  return <SectionPage chapter={chapter} section={section} />;
}

export const Route = createFileRoute('/zh/chapters/$chapterId/$sectionId')({
  beforeLoad: ({ context, params }) => {
    const { chapter } = context as { chapter: ChapterMeta };
    return resolveSectionContext('zh', chapter, params.sectionId);
  },
  component: ZhSectionRouteComponent,
});
