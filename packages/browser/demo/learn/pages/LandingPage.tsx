// learn/pages/LandingPage.tsx — the shared landing page, extracted from the
// former routes/index.tsx route component so the English route and the /zh
// mirror render the SAME logic. Hosts the <Landing /> screen, wires navigation
// handlers to the router, and triggers model load via useModelLoader() when
// the user enters the learning flow or asks to load the hosted model.

import { useNavigate } from '@tanstack/react-router';
import * as React from 'react';

import { Landing } from '../../components/landing/Landing';
import { detectModelBlockReason } from '../../lib/device-capability';
import { storeLocale } from '../../lib/i18n';
import { useLocale } from '../../lib/i18n-react';
import { triggerLocalPicker } from '../../lib/local-model-picker';
import { useLocaleNavigate } from '../../lib/locale-navigate';
import { useModelLoader } from '../../providers/model-loader';

export function LandingPage() {
  const navigate = useNavigate();
  const go = useLocaleNavigate();
  const locale = useLocale();
  const { hostedModelAvailable, errorBanner, kickoffLoad, status } = useModelLoader();

  // Persist the current locale so the standalone /jspace route — which has no
  // LocaleProvider and reads locale ONLY from localStorage — inherits it. Covers
  // DIRECT visitors to /zh (or "/") who never touched the language switcher:
  // without this, /zh → "Open J-Space" lands on the English gallery in a fresh
  // profile (codex #8). Symmetric: "/" persists 'en', "/zh" persists 'zh'.
  React.useEffect(() => {
    storeLocale(locale);
  }, [locale]);

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
          // Chat is deliberately NOT locale-prefixed — there is no /zh/chat route.
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
        go('/chapters');
      }}
      onOpenJSpace={() => {
        // J-Space is a standalone tool with a single route — deliberately NOT
        // locale-prefixed (there is no /zh/jspace), like /chat.
        void navigate({ to: '/jspace', search: (prev) => prev });
      }}
      errorBanner={errorBanner}
      hostedModelAvailable={hostedModelAvailable}
      modelReady={status === 'ready'}
      modelBlockReason={detectModelBlockReason()}
    />
  );
}
