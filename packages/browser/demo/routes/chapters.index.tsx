// routes/chapters.index.tsx — Chapters index route (/chapters).
//
// The chapter list is pure, model-free content and ALWAYS renders — it is
// never gated behind the model. Landing here pre-warms the model: when a HOSTED
// model is available the route auto-starts the fetch on entry (gesture-free), so
// it is ready by the time the reader opens a chapter. When no hosted model
// exists the load needs the local-file picker (a user gesture), so we leave the
// explicit affordances in place instead.
//
// The model-dependent piece here is the <ForwardPassFlow> hero demo inside
// <ChapterIndex>; it auto-runs once the model is ready, and otherwise surfaces
// its own "load the model" affordance. Navigation targets are driven through
// the TanStack Router useNavigate hook.

import { createFileRoute, useNavigate } from '@tanstack/react-router';
import { useEffect } from 'react';

import { ChapterIndex } from '../learn/ChapterIndex';
import { triggerLocalPicker } from '../lib/local-model-picker';
import { useFreeChat } from '../providers/free-chat';
import { useModelLoader } from '../providers/model-loader';

function ChaptersIndexRouteComponent() {
  const navigate = useNavigate();
  const { mlxWorkerRef, inspectorAbortRef } = useFreeChat();
  const { status, hostedModelAvailable, kickoffLoad } = useModelLoader();

  // Pre-warm on entry: auto-start the hosted-model fetch when landing on the
  // index, so the model is loading/ready before the reader opens a chapter.
  // Guarded to status 'idle' (never auto-retries a failed load or disturbs an
  // in-flight/ready model) and to hostedModelAvailable === true (the no-hosted
  // path needs triggerLocalPicker, which browsers only allow from a real user
  // gesture). kickoffLoad is idempotent, so a re-run is harmless.
  useEffect(() => {
    if (status !== 'idle') return;
    if (hostedModelAvailable === true) kickoffLoad();
  }, [status, hostedModelAvailable, kickoffLoad]);

  return (
    <ChapterIndex
      workerRef={mlxWorkerRef}
      abortRef={inspectorAbortRef}
      modelReady={status === 'ready'}
      onLoadModel={() => {
        if (hostedModelAvailable === false) {
          triggerLocalPicker();
          return;
        }
        kickoffLoad();
      }}
      onOpenChapter={(chapterId) => {
        void navigate({ to: '/chapters/$chapterId', params: { chapterId }, search: (prev) => prev });
      }}
      onBackToLanding={() => {
        void navigate({ to: '/', search: (prev) => prev });
      }}
      onOpenFreeChat={() => {
        // Just open the chat surface — do NOT kick off a load here. The chat
        // overlay (<ChatLayerOverlay>) is the single consent gate for chat: it
        // surfaces its own "Load the model to chat" CTA (and the local-model
        // picker when no hosted model is available) when the model isn't ready.
        void navigate({ to: '/chat', search: (prev) => prev });
      }}
    />
  );
}

export const Route = createFileRoute('/chapters/')({
  component: ChaptersIndexRouteComponent,
});
