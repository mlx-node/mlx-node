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

  return (
    <Landing
      onLoad={() => {
        if (hostedModelAvailable === false) {
          triggerLocalPicker();
          return;
        }
        kickoffLoad();
      }}
      onLocalModel={triggerLocalPicker}
      onStartLearning={() => {
        // Kick off the model load alongside the screen transition. The
        // chapter routes also auto-kickoff on direct URL landings; this
        // handler covers the explicit "Start learning" click. Skipped when
        // no hosted model is available — those users go through the local
        // model picker, which we deliberately do NOT auto-open from a
        // learning click. kickoffLoad is idempotent so the chapters route
        // calling it again post-navigate is a safe no-op.
        if (status !== 'ready' && hostedModelAvailable !== false) {
          kickoffLoad();
        }
        void navigate({ to: '/chapters', search: (prev) => prev });
      }}
      errorBanner={errorBanner}
      hostedModelAvailable={hostedModelAvailable}
    />
  );
}

export const Route = createFileRoute('/')({
  component: LandingRouteComponent,
});
