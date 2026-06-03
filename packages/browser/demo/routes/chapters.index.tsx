// routes/chapters.index.tsx — Chapters index route (/chapters).
//
// The chapter list is pure, model-free content and ALWAYS renders — it is
// never gated behind the model. Opening this route does NOT auto-download the
// ~1.6 GB model: loading happens only via explicit user action (the consent
// layer on a chapter's live panel, or the global header Load button).
//
// The one model-dependent piece here is the <ForwardPassFlow> hero demo inside
// <ChapterIndex>; it only auto-runs once the model is ready, and otherwise
// surfaces its own "load the model" affordance. Navigation targets are driven
// through the TanStack Router useNavigate hook.

import { createFileRoute, useNavigate } from '@tanstack/react-router';

import { ChapterIndex } from '../learn/ChapterIndex';
import { triggerLocalPicker } from '../lib/local-model-picker';
import { useFreeChat } from '../providers/free-chat';
import { useModelLoader } from '../providers/model-loader';

function ChaptersIndexRouteComponent() {
  const navigate = useNavigate();
  const { mlxWorkerRef, inspectorAbortRef } = useFreeChat();
  const { status, hostedModelAvailable, kickoffLoad } = useModelLoader();

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
