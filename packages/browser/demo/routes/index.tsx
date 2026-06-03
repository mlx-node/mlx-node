// routes/index.tsx — Landing route (/).
//
// Phase 2.C: hosts the <Landing /> screen. Wires navigation handlers to the
// router and triggers model load via useModelLoader() when the user enters
// the learning flow or asks to load the hosted model.

import { createFileRoute, useNavigate } from '@tanstack/react-router';

import { Landing } from '../components/landing/Landing';
import { triggerLocalPicker } from '../lib/local-model-picker';
import { useModelLoader } from '../providers/model-loader';

function LandingRouteComponent() {
  const navigate = useNavigate();
  const { hostedModelAvailable, errorBanner, kickoffLoad, status } = useModelLoader();

  // SEO <head> (title/canonical/OG/JSON-LD) is synced centrally by the root
  // route's head manager (routes/__root.tsx) on pathname change — including "/".

  return (
    <Landing
      onLoad={() => {
        if (hostedModelAvailable === false) {
          triggerLocalPicker();
          return;
        }
        // If the model is already loaded (e.g. the user reloaded the
        // Landing page after visiting a chapter, or returned via the
        // browser back button), kickoffLoad() is an internal no-op and
        // the button would appear broken. Navigate straight to the chat
        // surface instead — that's the obvious "use the loaded model"
        // next step and matches the new "Open Chat →" label.
        if (status === 'ready') {
          void navigate({ to: '/chat', search: (prev) => prev });
          return;
        }
        kickoffLoad();
      }}
      onLocalModel={triggerLocalPicker}
      onStartLearning={() => {
        // Entering the course must NOT download the model — that would re-couple
        // model loading to navigation, which this refactor exists to remove. The
        // ~1.6 GB load is gated behind an explicit click on a chapter's live-demo
        // consent layer (or the global header "Load model" button / the chat
        // overlay). So this only navigates into the chapter index.
        void navigate({ to: '/chapters', search: (prev) => prev });
      }}
      errorBanner={errorBanner}
      hostedModelAvailable={hostedModelAvailable}
      modelReady={status === 'ready'}
    />
  );
}

export const Route = createFileRoute('/')({
  component: LandingRouteComponent,
});
