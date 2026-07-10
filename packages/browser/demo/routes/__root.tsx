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
import { useEffect } from 'react';
import { z } from 'zod';

import { ChatLayerOverlay } from '../components/ChatLayerOverlay';
import { findChapter } from '../learn/chapters';
import { localizedChapter, localizedChapters } from '../learn/i18n/localized';
import { stripLocalePrefix } from '../lib/i18n';
import { applySeoHead } from '../lib/seo-head';
import {
  chapterJsonLd,
  courseJsonLd,
  getChapterSeo,
  getChaptersHubSeo,
  getChatSeo,
  getJspaceSeo,
  getLandingSeo,
  getSectionSeo,
  sectionJsonLd,
} from '../lib/seo-metadata';

export const searchSchema = z.object({
  // Model URL — legacy aliases: model_url, modelUrl, model
  model_url: z.string().optional().catch(undefined),
  modelUrl: z.string().optional().catch(undefined),
  model: z.string().optional().catch(undefined),

  // Model display label — legacy aliases: model_label, modelLabel
  model_label: z.string().optional().catch(undefined),
  modelLabel: z.string().optional().catch(undefined),

  // Max output tokens — legacy aliases: max_new_tokens, maxOutputTokens
  max_new_tokens: z.coerce.number().int().positive().optional().catch(undefined),
  maxOutputTokens: z.coerce.number().int().positive().optional().catch(undefined),

  // Sampling temperature — legacy aliases: temperature, temp
  temperature: z.coerce.number().min(0).max(2).optional().catch(undefined),
  temp: z.coerce.number().min(0).max(2).optional().catch(undefined),

  // App-preview / tools toggle — legacy aliases: tools, app_preview.
  // Coerces ?tools=1 / ?tools=true → true, ?tools=0 / ?tools=false → false.
  tools: z.coerce
    .number()
    .int()
    .min(0)
    .max(1)
    .transform((v) => v === 1)
    .optional()
    .catch(undefined),
  app_preview: z.coerce
    .number()
    .int()
    .min(0)
    .max(1)
    .transform((v) => v === 1)
    .optional()
    .catch(undefined),
});

export type RootSearch = z.infer<typeof searchSchema>;

function RootComponent() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const isChatRoute = pathname === '/chat';

  // Centralized per-route <head> sync. The prerendered pages already carry the
  // correct head on DIRECT entry (what crawlers/unfurlers fetch); this keeps the
  // <html lang>, title, canonical, hreflang alternates, description, OG/Twitter,
  // and JSON-LD consistent with the active route after CLIENT-SIDE navigation —
  // so a JS-executing crawler or an in-session transition never shows one
  // route's title against another route's canonical/structured data. One place
  // covers every route (no per-route omission). Locale is derived from the URL
  // prefix (/zh/… vs unprefixed) and meta strings come pre-localized from the
  // overlay registry — the SEO builders never re-localize.
  useEffect(() => {
    const { locale, path } = stripLocalePrefix(pathname);
    if (path === '/') {
      applySeoHead(getLandingSeo(locale), courseJsonLd(localizedChapters(locale), locale));
      return;
    }
    if (path === '/chapters' || path === '/chapters/') {
      applySeoHead(getChaptersHubSeo(locale), courseJsonLd(localizedChapters(locale), locale));
      return;
    }
    // Section match is MORE specific than the chapter match — test it first so a
    // /chapters/<id>/<section> URL never matches the chapter branch.
    const sectionMatch = path.match(/^\/chapters\/([^/]+)\/([^/]+)\/?$/);
    if (sectionMatch) {
      const chapter = localizedChapter(locale, decodeURIComponent(sectionMatch[1]));
      const sectionId = decodeURIComponent(sectionMatch[2]);
      const section = chapter?.sections?.find((s) => s.id === sectionId);
      if (chapter && section) {
        applySeoHead(getSectionSeo(chapter, section, locale), sectionJsonLd(chapter, section, locale));
        return;
      }
    }
    const chapterMatch = path.match(/^\/chapters\/([^/]+)\/?$/);
    if (chapterMatch) {
      const chapter = localizedChapter(locale, decodeURIComponent(chapterMatch[1]));
      if (chapter) {
        applySeoHead(getChapterSeo(chapter, locale), chapterJsonLd(chapter, locale));
        return;
      }
    }
    // /chat exists in English only — /zh/chat (or any unknown /zh subpath)
    // falls through to the locale-level landing identity below.
    if (path === '/chat' && locale === 'en') {
      applySeoHead(getChatSeo(), null);
      return;
    }
    // /jspace is English-only (the Jacobian-lens app) — same fall-through rules
    // as /chat: any /zh/jspace lands on the locale-level landing identity below.
    if (path === '/jspace' && locale === 'en') {
      applySeoHead(getJspaceSeo(), null);
      return;
    }
    // Unknown path — fall back to the site-level landing identity (same locale).
    applySeoHead(getLandingSeo(locale), courseJsonLd(localizedChapters(locale), locale));
  }, [pathname]);

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
