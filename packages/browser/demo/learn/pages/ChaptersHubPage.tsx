// learn/pages/ChaptersHubPage.tsx — the shared chapters-hub page, extracted
// from the former routes/chapters.index.tsx route component so the English
// route and the /zh mirror render the SAME logic.
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
// its own "load the model" affordance.

import { useNavigate } from '@tanstack/react-router';
import { useEffect } from 'react';

import { canAutoLoadModel } from '../../lib/device-capability';
import { triggerLocalPicker } from '../../lib/local-model-picker';
import { useLocaleNavigate } from '../../lib/locale-navigate';
import { useFreeChat } from '../../providers/free-chat';
import { useModelLoader } from '../../providers/model-loader';
import { ChapterIndex } from '../ChapterIndex';

export function ChaptersHubPage() {
  const navigate = useNavigate();
  const go = useLocaleNavigate();
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
    // Never auto-start the load on a device that would OOM/crash (iOS Safari,
    // low-RAM, no WebGPU). ModelConsentLayer shows a "run on desktop" message
    // there instead. Desktops are unaffected (canAutoLoadModel() === true).
    if (!canAutoLoadModel()) return;
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
        go(`/chapters/${chapterId}`);
      }}
      onBackToLanding={() => {
        go('/');
      }}
      onOpenFreeChat={() => {
        // Just open the chat surface — do NOT kick off a load here. The chat
        // overlay (<ChatLayerOverlay>) is the single consent gate for chat: it
        // surfaces its own "Load the model to chat" CTA (and the local-model
        // picker when no hosted model is available) when the model isn't ready.
        // Chat is deliberately NOT locale-prefixed — there is no /zh/chat route.
        void navigate({ to: '/chat', search: (prev) => prev });
      }}
    />
  );
}
