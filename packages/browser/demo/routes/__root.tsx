// __root.tsx — Root route for the TanStack Router tree.
//
// validateSearch accepts all legacy query-param aliases used by the existing
// query-param SPA (app.tsx) so deep-links continue to work after Phase 2.C
// migration. Unknown keys are stripped (Zod strips by default on .parse()).
// Normalization to canonical names happens in Phase 2.B.
//
// IMPORTANT: Zod strips unknown keys silently — any new legacy query param
// consumed by app.tsx MUST be added to searchSchema before Phase 2.C mounts
// <RouterProvider />, or that param will be lost from the URL after navigation.
//
// Phase 2.C: the root component renders <Outlet /> for child routes plus the
// always-mounted <ChatLayerOverlay />. Visibility of the chat overlay is
// driven by the current pathname (=== '/chat'); the JSX stays mounted so the
// imperative chat-DOM `useEffect` in app.tsx keeps writing to live refs.
//
// Phase 2.D: beforeLoad handles legacy ?screen=… URLs by redirecting to the
// equivalent TanStack Router path, preserving model-config search params.

import { createRootRoute, Outlet, redirect, useRouterState } from '@tanstack/react-router';
import { z } from 'zod';

import { ChatLayerOverlay } from '../components/ChatLayerOverlay';
import { findChapter } from '../learn/chapters';

export const searchSchema = z.object({
  // Model URL — legacy aliases: model_url, modelUrl, model
  model_url: z.string().optional(),
  modelUrl: z.string().optional(),
  model: z.string().optional(),

  // Model display label — legacy aliases: model_label, modelLabel
  model_label: z.string().optional(),
  modelLabel: z.string().optional(),

  // Max output tokens — legacy aliases: max_new_tokens, maxOutputTokens
  max_new_tokens: z.coerce.number().int().positive().optional(),
  maxOutputTokens: z.coerce.number().int().positive().optional(),

  // Sampling temperature — legacy aliases: temperature, temp
  temperature: z.coerce.number().min(0).max(2).optional(),
  temp: z.coerce.number().min(0).max(2).optional(),

  // App-preview / tools toggle — legacy aliases: tools, app_preview.
  // Coerces ?tools=1 / ?tools=true → true, ?tools=0 / ?tools=false → false.
  tools: z.coerce.number().int().min(0).max(1).transform(v => v === 1).optional(),
  app_preview: z.coerce.number().int().min(0).max(1).transform(v => v === 1).optional(),
});

export type RootSearch = z.infer<typeof searchSchema>;

function RootComponent() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const isChatRoute = pathname === '/chat';
  return (
    <>
      <Outlet />
      <ChatLayerOverlay visible={isChatRoute} />
    </>
  );
}

export const Route = createRootRoute({
  validateSearch: (search) => searchSchema.parse(search),
  beforeLoad: ({ location }) => {
    const raw = new URLSearchParams(location.search);
    const screen = raw.get('screen');
    const chapterId = raw.get('chapterId');

    // Only redirect when legacy params are present.
    if (!screen && !chapterId) return;

    // Preserve model-config params by running them through the schema.
    // Unknown keys (screen, chapterId) are stripped automatically.
    const preservedSearch = searchSchema.parse(Object.fromEntries(raw.entries()));

    switch (screen) {
      case 'chapter': {
        const chapter = chapterId ? findChapter(chapterId) : null;
        const to = chapter ? `/chapters/${chapter.id}` : '/chapters';
        throw redirect({ to, search: preservedSearch, replace: true });
      }
      case 'chapter_index':
        throw redirect({ to: '/chapters', search: preservedSearch, replace: true });
      case 'chat':
        throw redirect({ to: '/chat', search: preservedSearch, replace: true });
      case 'landing':
      case 'loading':
        // loading is transient — fall back to landing (/)
        throw redirect({ to: '/', search: preservedSearch, replace: true });
      default:
        // Unknown screen value — if chapterId is present without screen=chapter,
        // still strip it by redirecting to landing.
        if (chapterId) {
          throw redirect({ to: '/', search: preservedSearch, replace: true });
        }
    }
  },
  component: RootComponent,
});
