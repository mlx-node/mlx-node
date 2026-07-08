// routes/index.tsx — Landing route (/).
//
// Thin wrapper: the page logic (model-load CTA wiring, navigation handlers)
// lives in the shared, locale-agnostic learn/pages/LandingPage.tsx — also
// rendered by the /zh mirror.

import { createFileRoute } from '@tanstack/react-router';

import { LandingPage } from '../learn/pages/LandingPage';

export const Route = createFileRoute('/')({
  component: LandingPage,
});
