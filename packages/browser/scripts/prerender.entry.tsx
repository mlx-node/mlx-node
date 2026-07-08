// scripts/prerender.entry.tsx — SSR prerender entry.
//
// Bundled by esbuild (scripts/prerender.mjs) into dist/.prerender/entry.cjs and
// executed by a clean node subprocess. For BOTH locales (en at the unprefixed
// URLs, zh under /zh) it server-renders each chapter/sub-chapter BODY component
// (model/worker-free, validated in Phase 1) to HTML, clones the freshly-built
// dist/client/index.html shell, injects the prose into <div id="root"> and
// swaps in a correct per-page <head> (title/description/canonical/hreflang/
// OG/Twitter/JSON-LD). zh pages additionally get <html lang="zh-CN"> and, for
// chapters not yet translated in learn/i18n/bodies.zh.ts, the English body
// prefixed with <TranslationPendingNote />. It also writes a static /chapters
// hub per locale, rewrites the landing <head> (en in place, zh at
// /zh/index.html), and emits a dual-locale sitemap.xml + robots.txt.
//
// The SPA boots over these files with createRoot (NOT hydrate), so the
// prerendered prose is cleanly cleared+replaced — no hydration mismatch.

import * as React from 'react';
import { renderToString } from 'react-dom/server';
import {
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter,
  RouterContextProvider,
} from '@tanstack/react-router';

// Imports use relative paths (not the "@" alias) so the file type-resolves
// standalone under the linter, which does not pick up the demo-only tsconfig
// path mappings. esbuild still aliases "@mlx-node/lm/tools" transitively.
import { CHAPTERS } from '../demo/learn/chapters';
import type { ChapterMeta } from '../demo/learn/chapters';
import {
  chapterJsonLd,
  courseJsonLd,
  getChapterSeo,
  getChaptersHubSeo,
  getLandingSeo,
  getSectionSeo,
  ogLocale,
  sectionJsonLd,
  SITE_NAME,
  SITE_ORIGIN,
  hreflangAlternates,
  type PageSeo,
} from '../demo/lib/seo-metadata';
import { DEFAULT_LOCALE, htmlLang, localePath, LOCALES, type Locale } from '../demo/lib/i18n';
import { LocaleProvider } from '../demo/lib/i18n-react';
import { localizedChapters } from '../demo/learn/i18n/localized';
import { ZH_CHAPTER_BODIES, ZH_SECTION_BODIES } from '../demo/learn/i18n/bodies.zh';
import { TranslationPendingNote } from '../demo/learn/i18n/TranslationPendingNote';

// Single shared chapter-body registry — same map the chapter routes consume
// (replaces the import block this file used to mirror by hand).
import { CHAPTER_BODIES } from '../demo/learn/bodies';
// Single shared sub-chapter registry — same map the section route consumes.
import { SECTION_BODIES, sectionBodyKey } from '../demo/learn/section-bodies';

// The real SPA layout shell — rendered statically around each body so the
// prerendered HTML matches the SPA's chrome (header + sidebar + centered,
// padded, scrollable reading column) instead of dumping the bare body
// full-width. It is SSR-safe (only a Button + icons + the chapter registry).
import { LessonLayout } from '../demo/learn/LessonLayout';

import fs from 'node:fs';
import path from 'node:path';

// Chapters whose interactive content lives inline in the body — they have no
// right-hand "Try it now" column. Mirror of demo/learn/pages/ChapterPage.tsx
// (its demoPanel ternary yields null for these four).
const PANEL_LESS = new Set(['overview', 'post-training', 'architecture', 'scaling']);
const noop = (): void => {};
// Static stand-in for the live demo panel so panel chapters keep their
// 3-column grid (hence the same reading-column width) in the pre-JS HTML; the
// SPA swaps in the real consent layer / demo on boot via createRoot.
function tryItPlaceholder(locale: Locale): React.ReactNode {
  return (
    <div className="rounded-lg border border-border bg-card/40 p-6 text-sm text-muted-foreground">
      {locale === 'zh' ? '互动演示加载中……' : 'Loading the interactive demo…'}
    </div>
  );
}

/** HTML-attribute / text escape. */
function esc(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/** XML text/attribute escape for the sitemap. */
function escXml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

/** Serialize a JSON-LD payload, guarding against `</script>` breakout. */
function jsonLdScript(payload: object | object[]): string {
  const json = JSON.stringify(payload).replace(/</g, '\\u003c');
  // The id lets the client-side head manager (demo/lib/seo-head.ts) update THIS
  // same node on SPA navigation instead of appending a second, conflicting block.
  return `<script type="application/ld+json" id="seo-jsonld">${json}</script>`;
}

// Static head fragment shared by every page: the font preconnects, the Google
// Fonts stylesheet, and the favicon. Copied verbatim from the built shell head.
const STATIC_HEAD_LINKS = [
  '<link rel="preconnect" href="https://fonts.googleapis.com" />',
  '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />',
  '<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet" />',
  '<link rel="icon" href="/capybara.png" type="image/png" />',
].join('\n    ');

function buildHead(seo: PageSeo, jsonLdObjects: object[], assetTags: string): string {
  // Reciprocal hreflang alternates (en / zh-CN / x-default) — the SAME set on
  // every page in both locales; [] on single-locale pages (none prerendered).
  const alternateLinks = seo.alternates.map(
    (a) => `<link rel="alternate" hreflang="${esc(a.hreflang)}" href="${esc(a.href)}" />`,
  );
  const lines = [
    '<meta charset="UTF-8" />',
    '<meta name="viewport" content="width=device-width, initial-scale=1.0" />',
    `<title>${esc(seo.title)}</title>`,
    `<meta name="description" content="${esc(seo.description)}" />`,
    `<link rel="canonical" href="${esc(seo.canonical)}" />`,
    ...alternateLinks,
    `<meta property="og:type" content="${seo.ogType}" />`,
    `<meta property="og:site_name" content="${esc(SITE_NAME)}" />`,
    `<meta property="og:locale" content="${ogLocale(seo.locale)}" />`,
    `<meta property="og:url" content="${esc(seo.canonical)}" />`,
    `<meta property="og:title" content="${esc(seo.title)}" />`,
    `<meta property="og:description" content="${esc(seo.description)}" />`,
    `<meta property="og:image" content="${esc(seo.image)}" />`,
    '<meta property="og:image:width" content="1200" />',
    '<meta property="og:image:height" content="630" />',
    '<meta name="twitter:card" content="summary_large_image" />',
    `<meta name="twitter:title" content="${esc(seo.title)}" />`,
    `<meta name="twitter:description" content="${esc(seo.description)}" />`,
    `<meta name="twitter:image" content="${esc(seo.image)}" />`,
    '<meta name="theme-color" content="#08070D" />',
    STATIC_HEAD_LINKS,
    jsonLdScript(jsonLdObjects.length === 1 ? jsonLdObjects[0] : jsonLdObjects),
    assetTags,
  ];
  return lines.join('\n    ');
}

/**
 * Extract the Vite-injected /assets/* tags from the built shell. Reads the
 * <head>, collects every <script ...>...</script> and <link ...> whose
 * src/href contains "/assets/". Hashes change every build, so these are read at
 * runtime and re-emitted verbatim — never hardcoded.
 */
function extractAssetTags(shellHtml: string): string {
  const headMatch = shellHtml.match(/<head[^>]*>([\s\S]*?)<\/head>/i);
  const head = headMatch ? headMatch[1] : shellHtml;

  const tags: string[] = [];
  // <script ...>...</script> tags referencing /assets/.
  const scriptRe = /<script\b[^>]*>[\s\S]*?<\/script>/gi;
  for (const m of head.matchAll(scriptRe)) {
    if (/(?:src)\s*=\s*["'][^"']*\/assets\//i.test(m[0])) tags.push(m[0].trim());
  }
  // <link ...> tags referencing /assets/ (self-closing or not).
  const linkRe = /<link\b[^>]*>/gi;
  for (const m of head.matchAll(linkRe)) {
    if (/href\s*=\s*["'][^"']*\/assets\//i.test(m[0])) tags.push(m[0].trim());
  }
  return tags.join('\n    ');
}

const HEAD_RE = /<head[^>]*>[\s\S]*?<\/head>/i;
const ROOT_PLACEHOLDER = '<div id="root"></div>';
// The shell is authored as `<html lang="en">` (demo/index.html); Vite copies
// the attribute through verbatim. Matched leniently on surrounding attributes
// but exactly on the lang value so a shape change fails LOUD, not silent.
const HTML_LANG_RE = /(<html\b[^>]*\blang=")en(")/i;

/** Swap the shell <head>; throws if no <head> is found (never silently keep the stale head). */
function swapHead(shellHtml: string, headInner: string): string {
  if (!HEAD_RE.test(shellHtml)) {
    throw new Error('prerender: no <head>…</head> found in the built shell — refusing to emit a page with the stale head.');
  }
  return shellHtml.replace(HEAD_RE, `<head>\n    ${headInner}\n  </head>`);
}

/**
 * Rewrite the shell's <html lang="en"> for non-English output. Throws if the
 * attribute is missing (a zh page must never ship lang="en").
 */
function setHtmlLang(html: string, locale: Locale): string {
  if (locale === DEFAULT_LOCALE) return html;
  if (!HTML_LANG_RE.test(html)) {
    throw new Error(
      'prerender: <html ... lang="en"> not found in the built shell — refusing to emit a non-English page with the wrong lang.',
    );
  }
  return html.replace(HTML_LANG_RE, `$1${htmlLang(locale)}$2`);
}

/**
 * Inject prerendered prose into #root. Throws if the exact placeholder is absent
 * (e.g. Vite changed the shell's quoting/whitespace/attributes) instead of
 * SILENTLY writing a page with an EMPTY #root — the crawlable-prose guarantee
 * must fail loud and break the build, not ship blank pages.
 */
function injectRoot(html: string, rootInner: string): string {
  if (!html.includes(ROOT_PLACEHOLDER)) {
    throw new Error(
      `prerender: root placeholder ${ROOT_PLACEHOLDER} not found in the built shell — its shape changed; refusing to write a page with no prose.`,
    );
  }
  return html.replace(ROOT_PLACEHOLDER, `<div id="root">${rootInner}</div>`);
}

/** Replace the shell's <head>, inject prose into #root, fix <html lang>, return the page. */
function composePage(shellHtml: string, headInner: string, rootInner: string, locale: Locale): string {
  return setHtmlLang(injectRoot(swapHead(shellHtml, headInner), rootInner), locale);
}

function buildChaptersHubInner(chapters: ChapterMeta[], locale: Locale): string {
  const hub = getChaptersHubSeo(locale);
  const items = chapters
    .map(
      (c) =>
        `<li><a href="${localePath(locale, `/chapters/${c.id}`)}"><strong>${c.number}. ${esc(c.title)}</strong> — ${esc(c.blurb)}</a></li>`,
    )
    .join('');
  const heading = locale === 'zh' ? '章节' : 'Chapters';
  return `<div class="mx-auto max-w-3xl px-8 py-10"><h1>${esc(heading)}</h1><p>${esc(hub.description)}</p><ul>${items}</ul></div>`;
}

function buildSitemap(chapters: ChapterMeta[]): string {
  const lastmod = new Date().toISOString().slice(0, 10);
  // Bare (unprefixed) app paths — each becomes one <url> PER locale, every
  // entry carrying the full reciprocal hreflang alternate set.
  const pages = [
    { path: '/', priority: '1.0' },
    { path: '/chapters', priority: '0.7' },
    ...chapters.flatMap((c) => [
      { path: `/chapters/${c.id}`, priority: '0.8' },
      // Sub-chapters: their own indexable URLs, one notch below the chapter.
      ...(c.sections ?? []).map((s) => ({ path: `/chapters/${c.id}/${s.id}`, priority: '0.75' })),
    ]),
  ];
  const entries = pages.flatMap(({ path: p, priority }) => {
    const altXml = hreflangAlternates(p)
      .map((a) => `    <xhtml:link rel="alternate" hreflang="${escXml(a.hreflang)}" href="${escXml(a.href)}"/>`)
      .join('\n');
    return LOCALES.map((locale) => {
      const loc = SITE_ORIGIN + localePath(locale, p);
      // xhtml:link alternates go AFTER priority: the sitemap XSD only admits
      // foreign-namespace elements at the end of <url>; Google accepts both.
      return `  <url>\n    <loc>${escXml(loc)}</loc>\n    <lastmod>${lastmod}</lastmod>\n    <changefreq>monthly</changefreq>\n    <priority>${priority}</priority>\n${altXml}\n  </url>`;
    });
  });
  return `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:xhtml="http://www.w3.org/1999/xhtml">\n${entries.join('\n')}\n</urlset>\n`;
}

function buildRobots(): string {
  const aiCrawlers = [
    'GPTBot',
    'ChatGPT-User',
    'OAI-SearchBot',
    'ClaudeBot',
    'anthropic-ai',
    'Claude-Web',
    'PerplexityBot',
    'Google-Extended',
    'CCBot',
    'Applebot-Extended',
    'Bytespider',
  ];
  const lines = ['User-agent: *', 'Allow: /', ''];
  for (const ua of aiCrawlers) {
    lines.push(`User-agent: ${ua}`, 'Allow: /', '');
  }
  lines.push(`Sitemap: ${SITE_ORIGIN}/sitemap.xml`, '');
  return lines.join('\n');
}

void (async function main() {
  const distClient = process.env.PRERENDER_DIST_CLIENT;
  if (!distClient) {
    throw new Error('PRERENDER_DIST_CLIENT env var is required (absolute path to dist/client).');
  }
  const shellPath = path.join(distClient, 'index.html');
  if (!fs.existsSync(shellPath)) {
    throw new Error(`Built shell not found at ${shellPath} — run vite build first.`);
  }
  const shellHtml = fs.readFileSync(shellPath, 'utf8');
  const assetTags = extractAssetTags(shellHtml);
  if (!assetTags) {
    throw new Error('No /assets/* tags found in the built shell — the SPA would not boot.');
  }

  // Fail loudly if any registered chapter lacks a body component.
  for (const c of CHAPTERS) {
    if (!CHAPTER_BODIES[c.id]) {
      throw new Error(`No body component registered for chapter id "${c.id}".`);
    }
    // …or if any declared sub-chapter lacks a body in the shared registry.
    for (const s of c.sections ?? []) {
      if (!SECTION_BODIES[sectionBodyKey(c.id, s.id)]) {
        throw new Error(`No body component registered for sub-chapter "${c.id}/${s.id}".`);
      }
    }
  }

  /** Output file for a bare app path in a locale: en unprefixed, zh under /zh/. */
  function outFileFor(locale: Locale, barePath: string): string {
    const rel = localePath(locale, barePath).replace(/^\//, '');
    const outDir = rel === '' ? distClient! : path.join(distClient!, rel);
    fs.mkdirSync(outDir, { recursive: true });
    return path.join(outDir, 'index.html');
  }

  // Minimal router: provides the <Link> context only. Its Outlet is unused —
  // bodies are rendered as children of RouterContextProvider. Both the en and
  // /zh route shapes are registered so any <Link> the layout renders (in either
  // locale subtree) resolves to a real href in the prerendered HTML.
  const rootRoute = createRootRoute();
  const routeTree = rootRoute.addChildren([
    createRoute({ getParentRoute: () => rootRoute, path: '/' }),
    createRoute({ getParentRoute: () => rootRoute, path: '/chapters' }),
    createRoute({ getParentRoute: () => rootRoute, path: '/chapters/$chapterId' }),
    // Registered so the in-chapter "go deeper" <Link to="/chapters/$chapterId/$sectionId">
    // resolves to a real href in the prerendered HTML (crawlable <a>).
    createRoute({ getParentRoute: () => rootRoute, path: '/chapters/$chapterId/$sectionId' }),
    createRoute({ getParentRoute: () => rootRoute, path: '/zh' }),
    createRoute({ getParentRoute: () => rootRoute, path: '/zh/chapters' }),
    createRoute({ getParentRoute: () => rootRoute, path: '/zh/chapters/$chapterId' }),
    createRoute({ getParentRoute: () => rootRoute, path: '/zh/chapters/$chapterId/$sectionId' }),
  ]);
  const router = createRouter({ routeTree, history: createMemoryHistory({ initialEntries: ['/'] }) });
  await router.load();

  let chapterFilesWritten = 0;
  let sectionFilesWritten = 0;
  let totalBytes = 0;
  let sampleTitle = '';

  for (const locale of LOCALES) {
    // Display strings localized through the overlay (titles/blurbs fall back to
    // English per-field while the zh overlay fills in) — the SEO/JSON-LD helpers
    // consume this ALREADY-LOCALIZED meta as-is.
    const chapters = localizedChapters(locale);

    // 1) Per-chapter pages.
    for (const c of chapters) {
      const EnglishBody = CHAPTER_BODIES[c.id];
      const ZhBody = locale === 'zh' ? ZH_CHAPTER_BODIES[c.id] : undefined;
      // zh chapters without a translated body render the English body under a
      // visible TranslationPendingNote — still fully prerendered.
      const body =
        locale === 'zh' ? (
          ZhBody ? (
            <ZhBody />
          ) : (
            <>
              <TranslationPendingNote />
              <EnglishBody />
            </>
          )
        ) : (
          <EnglishBody />
        );
      // Render the body inside the REAL LessonLayout (header + sidebar + centered,
      // padded, scrollable reading column) so the pre-JS HTML matches the SPA
      // instead of dumping the bare body full-width. Handlers are no-ops: the SPA
      // boots with createRoot and REPLACES this subtree, so they back only the
      // brief pre-JS view + crawlers. Panel chapters get a placeholder so their
      // 3-column reading-column width matches pre-JS↔post-JS.
      const rootHtml = renderToString(
        <RouterContextProvider router={router}>
          <LocaleProvider locale={locale}>
            <LessonLayout
              current={c}
              wideBody={c.id === 'architecture'}
              tryItPanel={PANEL_LESS.has(c.id) ? null : tryItPlaceholder(locale)}
              onOpenChapter={noop}
              onBackToIndex={noop}
              onOpenFreeChat={noop}
            >
              {body}
            </LessonLayout>
          </LocaleProvider>
        </RouterContextProvider>,
      );
      if (rootHtml.trim().length < 50) {
        throw new Error(
          `prerender: chapter "${c.id}" (${locale}) rendered an empty/too-small body (${rootHtml.length} chars) — refusing to write.`,
        );
      }

      const seo = getChapterSeo(c, locale);
      const head = buildHead(seo, chapterJsonLd(c, locale), assetTags);
      const page = composePage(shellHtml, head, rootHtml, locale);

      fs.writeFileSync(outFileFor(locale, `/chapters/${c.id}`), page, 'utf8');

      chapterFilesWritten += 1;
      totalBytes += Buffer.byteLength(page, 'utf8');
      if (c.id === 'attention' && locale === DEFAULT_LOCALE) sampleTitle = seo.title;
    }

    // 1b) Per-sub-chapter pages. Reading-only deep-dives: rendered panel-less in
    // the SAME LessonLayout chrome (header breadcrumb + nested sidebar) so the
    // pre-JS HTML matches the SPA, with the section body from the shared registry.
    for (const c of chapters) {
      for (const s of c.sections ?? []) {
        const EnglishSectionBody = SECTION_BODIES[sectionBodyKey(c.id, s.id)];
        const ZhSectionBody = locale === 'zh' ? ZH_SECTION_BODIES[sectionBodyKey(c.id, s.id)] : undefined;
        const body =
          locale === 'zh' ? (
            ZhSectionBody ? (
              <ZhSectionBody />
            ) : (
              <>
                <TranslationPendingNote />
                <EnglishSectionBody />
              </>
            )
          ) : (
            <EnglishSectionBody />
          );
        const rootHtml = renderToString(
          <RouterContextProvider router={router}>
            <LocaleProvider locale={locale}>
              <LessonLayout
                current={c}
                currentSectionId={s.id}
                sectionTitle={s.title}
                tryItPanel={null}
                onOpenChapter={noop}
                onOpenSection={noop}
                onBackToIndex={noop}
                onOpenFreeChat={noop}
              >
                {body}
              </LessonLayout>
            </LocaleProvider>
          </RouterContextProvider>,
        );
        if (rootHtml.trim().length < 50) {
          throw new Error(
            `prerender: sub-chapter "${c.id}/${s.id}" (${locale}) rendered an empty/too-small body (${rootHtml.length} chars) — refusing to write.`,
          );
        }

        const seo = getSectionSeo(c, s, locale);
        const head = buildHead(seo, sectionJsonLd(c, s, locale), assetTags);
        const page = composePage(shellHtml, head, rootHtml, locale);

        fs.writeFileSync(outFileFor(locale, `/chapters/${c.id}/${s.id}`), page, 'utf8');

        sectionFilesWritten += 1;
        totalBytes += Buffer.byteLength(page, 'utf8');
      }
    }

    // 2) /chapters hub (pure-string inner, no React) — localized titles + hrefs.
    {
      const seo = getChaptersHubSeo(locale);
      const head = buildHead(seo, [courseJsonLd(chapters, locale)], assetTags);
      const page = composePage(shellHtml, head, buildChaptersHubInner(chapters, locale), locale);
      fs.writeFileSync(outFileFor(locale, '/chapters'), page, 'utf8');
    }

    // 3) Landing — head-swap only, #root stays empty (Landing has animation +
    // model hooks; the SPA mounts it client-side, locale-aware). en rewrites
    // the shell in place; zh writes /zh/index.html with lang="zh-CN".
    {
      const seo = getLandingSeo(locale);
      const head = buildHead(seo, [courseJsonLd(chapters, locale)], assetTags);
      const page = setHtmlLang(swapHead(shellHtml, head), locale);
      const outFile = locale === DEFAULT_LOCALE ? shellPath : outFileFor(locale, '/');
      fs.writeFileSync(outFile, page, 'utf8');
    }
  }

  // 4) sitemap.xml + robots.txt (sitemap covers both locales; robots unchanged).
  fs.writeFileSync(path.join(distClient, 'sitemap.xml'), buildSitemap(CHAPTERS), 'utf8');
  fs.writeFileSync(path.join(distClient, 'robots.txt'), buildRobots(), 'utf8');

  console.log(
    `[prerender] wrote ${chapterFilesWritten} chapter files + ${sectionFilesWritten} sub-chapter files across ${LOCALES.length} locales (${totalBytes} bytes) + hubs + landings + sitemap.xml + robots.txt`,
  );
  console.log(`[prerender] sample chapter title (attention): ${sampleTitle}`);
})();
