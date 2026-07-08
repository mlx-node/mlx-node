// routes/chapters.$chapterId.$sectionId.tsx — a sub-chapter page
// (/chapters/:chapterId/:sectionId).
//
// Thin wrapper: the page logic lives in the shared, locale-agnostic
// learn/pages/SectionPage.tsx — also rendered by the /zh mirror. beforeLoad
// reads the { chapter } the parent layout already resolved (inherited route
// context) and validates the sectionId via the shared helper (redirecting to
// the chapter page on a miss).

import { createFileRoute } from '@tanstack/react-router';

import type { ChapterMeta } from '../learn/chapters';
import { SectionPage } from '../learn/pages/SectionPage';
import { resolveSectionContext } from '../learn/pages/route-helpers';

function SectionRouteComponent() {
  const { chapter, section } = Route.useRouteContext();
  return <SectionPage chapter={chapter} section={section} />;
}

export const Route = createFileRoute('/chapters/$chapterId/$sectionId')({
  beforeLoad: ({ context, params }) => {
    const { chapter } = context as { chapter: ChapterMeta };
    return resolveSectionContext('en', chapter, params.sectionId);
  },
  component: SectionRouteComponent,
});
