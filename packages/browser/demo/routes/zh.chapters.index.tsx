// routes/zh.chapters.index.tsx — Chinese chapters index route (/zh/chapters).
// Same shared <ChaptersHubPage /> as the English /chapters route.

import { createFileRoute } from '@tanstack/react-router';

import { ChaptersHubPage } from '../learn/pages/ChaptersHubPage';

export const Route = createFileRoute('/zh/chapters/')({
  component: ChaptersHubPage,
});
