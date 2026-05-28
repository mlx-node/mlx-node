// routes/chapters.index.tsx — Chapters index route (/chapters).
//
// Phase 2.C: renders <ChapterIndex /> once the model worker is ready. While
// the model is still loading the same <Loading /> overlay used elsewhere is
// shown inline, matching the previous query-param SPA behavior. Navigation
// targets are driven through the TanStack Router useNavigate hook.

import { createFileRoute, useNavigate } from '@tanstack/react-router';
import { useEffect } from 'react';

import { Loading } from '../components/loading/Loading';
import { ChapterIndex } from '../learn/ChapterIndex';
import { triggerLocalPicker } from '../lib/local-model-picker';
import { useModelLoader } from '../providers/model-loader';

function ChaptersIndexRouteComponent() {
  const navigate = useNavigate();
  const { status, loadingText, loadingProgress, hostedModelAvailable, kickoffLoad } =
    useModelLoader();

  // Auto-kickoff the model load when the user lands directly on /chapters via
  // bookmark / hard reload. The Landing route also kicks off on "Start
  // learning", but a deep-link skips that path entirely. We avoid auto-load
  // when no hosted model is available — that flow requires the user to pick
  // a local model directory. kickoffLoad is idempotent at the App level.
  useEffect(() => {
    if (status === 'ready') return;
    if (hostedModelAvailable === false) return;
    kickoffLoad();
  }, [status, hostedModelAvailable, kickoffLoad]);

  if (status !== 'ready') {
    return <Loading status={loadingText || null} progress={loadingProgress} />;
  }

  return (
    <ChapterIndex
      onOpenChapter={(chapterId) => {
        void navigate({ to: '/chapters/$chapterId', params: { chapterId } });
      }}
      onBackToLanding={() => {
        void navigate({ to: '/' });
      }}
      onOpenFreeChat={() => {
        if (hostedModelAvailable === false) {
          triggerLocalPicker();
          return;
        }
        kickoffLoad();
        void navigate({ to: '/chat' });
      }}
    />
  );
}

export const Route = createFileRoute('/chapters/')({
  component: ChaptersIndexRouteComponent,
});
