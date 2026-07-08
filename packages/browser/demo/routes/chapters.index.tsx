// routes/chapters.index.tsx — Chapters index route (/chapters).
//
// Thin wrapper: the page logic (model pre-warm effect, <ChapterIndex> wiring)
// lives in the shared, locale-agnostic learn/pages/ChaptersHubPage.tsx — also
// rendered by the /zh mirror.

import { createFileRoute } from '@tanstack/react-router';

import { ChaptersHubPage } from '../learn/pages/ChaptersHubPage';

export const Route = createFileRoute('/chapters/')({
  component: ChaptersHubPage,
});
