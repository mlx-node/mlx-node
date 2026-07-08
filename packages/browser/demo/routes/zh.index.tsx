// routes/zh.index.tsx — Chinese landing route (/zh).
//
// Same shared <LandingPage /> as the English "/" route; the zh layout's
// <LocaleProvider> swaps the dictionary strings and locale-prefixes the
// navigation targets.

import { createFileRoute } from '@tanstack/react-router';

import { LandingPage } from '../learn/pages/LandingPage';

export const Route = createFileRoute('/zh/')({
  component: LandingPage,
});
