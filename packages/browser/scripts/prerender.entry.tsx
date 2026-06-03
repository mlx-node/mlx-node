// scripts/prerender.entry.tsx — SSR prerender entry.
//
// Bundled by esbuild (scripts/prerender.mjs) into dist/.prerender/entry.cjs and
// executed by a clean node subprocess. For each of the 16 chapters it
// server-renders the chapter BODY component (model/worker-free, validated in
// Phase 1) to HTML, clones the freshly-built dist/client/index.html shell,
// injects the prose into <div id="root"> and swaps in a correct per-chapter
// <head> (title/description/canonical/OG/Twitter/JSON-LD). It also writes a
// static /chapters hub, rewrites the landing <head>, and emits sitemap.xml +
// robots.txt.
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
  SITE_NAME,
  SITE_ORIGIN,
  type PageSeo,
} from '../demo/lib/seo-metadata';

// The real SPA layout shell — rendered statically around each body so the
// prerendered HTML matches the SPA's chrome (header + sidebar + centered,
// padded, scrollable reading column) instead of dumping the bare body
// full-width. It is SSR-safe (only a Button + icons + the chapter registry).
import { LessonLayout } from '../demo/learn/LessonLayout';

// Mirror of the id->body import block in demo/routes/chapters.$chapterId.tsx.
import { OverviewChapterBody } from '../demo/learn/chapters/00-overview';
import { TokenizationChapterBody } from '../demo/learn/chapters/01-tokenization';
import { EmbeddingsChapterBody } from '../demo/learn/chapters/02-embeddings';
import { AttentionChapterBody } from '../demo/learn/chapters/03-attention';
import { MultiheadGqaChapterBody } from '../demo/learn/chapters/04-multihead-gqa';
import { RopeChapterBody } from '../demo/learn/chapters/05-rope';
import { RmsNormChapterBody } from '../demo/learn/chapters/06-rms-norm';
import { MlpChapterBody } from '../demo/learn/chapters/07-mlp';
import { FullBlockChapterBody } from '../demo/learn/chapters/08-full-block';
import { LmHeadChapterBody } from '../demo/learn/chapters/09-lm-head';
import { SamplingChapterBody } from '../demo/learn/chapters/09-sampling';
import { KvCacheChapterBody } from '../demo/learn/chapters/10-kv-cache';
import { TrainingChapterBody } from '../demo/learn/chapters/12-training';
import { ScalingChapterBody } from '../demo/learn/chapters/13-scaling';
import { PostTrainingChapterBody } from '../demo/learn/chapters/15-post-training';
import { ArchitectureChapterBody } from '../demo/learn/chapters/14-architecture';

import fs from 'node:fs';
import path from 'node:path';

const BODIES: Record<string, React.ComponentType> = {
  overview: OverviewChapterBody,
  tokenization: TokenizationChapterBody,
  embeddings: EmbeddingsChapterBody,
  attention: AttentionChapterBody,
  'multi-head-gqa': MultiheadGqaChapterBody,
  rope: RopeChapterBody,
  rmsnorm: RmsNormChapterBody,
  mlp: MlpChapterBody,
  'full-block': FullBlockChapterBody,
  'lm-head': LmHeadChapterBody,
  sampling: SamplingChapterBody,
  'kv-cache': KvCacheChapterBody,
  training: TrainingChapterBody,
  scaling: ScalingChapterBody,
  'post-training': PostTrainingChapterBody,
  architecture: ArchitectureChapterBody,
};

// Chapters whose interactive content lives inline in the body — they have no
// right-hand "Try it now" column. Mirror of demo/routes/chapters.$chapterId.tsx
// (the route renders no demoPanel for these three).
const PANEL_LESS = new Set(['overview', 'post-training', 'architecture']);
const noop = (): void => {};
// Static stand-in for the live demo panel so panel chapters keep their
// 3-column grid (hence the same reading-column width) in the pre-JS HTML; the
// SPA swaps in the real consent layer / demo on boot via createRoot.
const TRY_IT_PLACEHOLDER = (
  <div className="rounded-lg border border-border bg-card/40 p-6 text-sm text-muted-foreground">
    Loading the interactive demo…
  </div>
);

/** HTML-attribute / text escape. */
function esc(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
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
  const lines = [
    '<meta charset="UTF-8" />',
    '<meta name="viewport" content="width=device-width, initial-scale=1.0" />',
    `<title>${esc(seo.title)}</title>`,
    `<meta name="description" content="${esc(seo.description)}" />`,
    `<link rel="canonical" href="${esc(seo.canonical)}" />`,
    `<meta property="og:type" content="${seo.ogType}" />`,
    `<meta property="og:site_name" content="${esc(SITE_NAME)}" />`,
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

/** Swap the shell <head>; throws if no <head> is found (never silently keep the stale head). */
function swapHead(shellHtml: string, headInner: string): string {
  if (!HEAD_RE.test(shellHtml)) {
    throw new Error('prerender: no <head>…</head> found in the built shell — refusing to emit a page with the stale head.');
  }
  return shellHtml.replace(HEAD_RE, `<head>\n    ${headInner}\n  </head>`);
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

/** Replace the shell's <head> and inject prose into #root, return the page. */
function composePage(shellHtml: string, headInner: string, rootInner: string): string {
  return injectRoot(swapHead(shellHtml, headInner), rootInner);
}

function buildChaptersHubInner(chapters: ChapterMeta[]): string {
  const hub = getChaptersHubSeo();
  const items = chapters
    .map(
      (c) =>
        `<li><a href="/chapters/${c.id}"><strong>${c.number}. ${esc(c.title)}</strong> — ${esc(c.blurb)}</a></li>`,
    )
    .join('');
  return `<div class="mx-auto max-w-3xl px-8 py-10"><h1>Chapters</h1><p>${esc(hub.description)}</p><ul>${items}</ul></div>`;
}

function buildSitemap(chapters: ChapterMeta[]): string {
  const lastmod = new Date().toISOString().slice(0, 10);
  const url = (loc: string, priority: string, changefreq = 'monthly') =>
    `  <url>\n    <loc>${loc}</loc>\n    <lastmod>${lastmod}</lastmod>\n    <changefreq>${changefreq}</changefreq>\n    <priority>${priority}</priority>\n  </url>`;
  const entries = [
    url(`${SITE_ORIGIN}/`, '1.0'),
    url(`${SITE_ORIGIN}/chapters`, '0.7'),
    ...chapters.map((c) => url(`${SITE_ORIGIN}/chapters/${c.id}`, '0.8')),
  ];
  return `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${entries.join('\n')}\n</urlset>\n`;
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
    if (!BODIES[c.id]) {
      throw new Error(`No body component registered for chapter id "${c.id}".`);
    }
  }

  // Minimal router: provides the <Link> context only. Its Outlet is unused —
  // bodies are rendered as children of RouterContextProvider.
  const rootRoute = createRootRoute();
  const indexRoute = createRoute({ getParentRoute: () => rootRoute, path: '/' });
  const chaptersIndexRoute = createRoute({ getParentRoute: () => rootRoute, path: '/chapters' });
  const chapterRoute = createRoute({ getParentRoute: () => rootRoute, path: '/chapters/$chapterId' });
  const routeTree = rootRoute.addChildren([indexRoute, chaptersIndexRoute, chapterRoute]);
  const router = createRouter({ routeTree, history: createMemoryHistory({ initialEntries: ['/'] }) });
  await router.load();

  let chapterFilesWritten = 0;
  let totalBytes = 0;
  let sampleTitle = '';

  // 1) Per-chapter pages.
  for (const c of CHAPTERS) {
    const Body = BODIES[c.id];
    // Render the body inside the REAL LessonLayout (header + sidebar + centered,
    // padded, scrollable reading column) so the pre-JS HTML matches the SPA
    // instead of dumping the bare body full-width. Handlers are no-ops: the SPA
    // boots with createRoot and REPLACES this subtree, so they back only the
    // brief pre-JS view + crawlers. Panel chapters get a placeholder so their
    // 3-column reading-column width matches pre-JS↔post-JS.
    const rootHtml = renderToString(
      <RouterContextProvider router={router}>
        <LessonLayout
          current={c}
          wideBody={c.id === 'architecture'}
          tryItPanel={PANEL_LESS.has(c.id) ? null : TRY_IT_PLACEHOLDER}
          onOpenChapter={noop}
          onBackToIndex={noop}
          onOpenFreeChat={noop}
        >
          <Body />
        </LessonLayout>
      </RouterContextProvider>,
    );
    if (rootHtml.trim().length < 50) {
      throw new Error(
        `prerender: chapter "${c.id}" rendered an empty/too-small body (${rootHtml.length} chars) — refusing to write.`,
      );
    }

    const seo = getChapterSeo(c);
    const head = buildHead(seo, chapterJsonLd(c), assetTags);
    const page = composePage(shellHtml, head, rootHtml);

    const outDir = path.join(distClient, 'chapters', c.id);
    fs.mkdirSync(outDir, { recursive: true });
    const outFile = path.join(outDir, 'index.html');
    fs.writeFileSync(outFile, page, 'utf8');

    chapterFilesWritten += 1;
    totalBytes += Buffer.byteLength(page, 'utf8');
    if (c.id === 'attention') sampleTitle = seo.title;
  }

  // 2) /chapters hub (pure-string inner, no React).
  {
    const seo = getChaptersHubSeo();
    const head = buildHead(seo, [courseJsonLd(CHAPTERS)], assetTags);
    const page = composePage(shellHtml, head, buildChaptersHubInner(CHAPTERS));
    const outFile = path.join(distClient, 'chapters', 'index.html');
    fs.mkdirSync(path.dirname(outFile), { recursive: true });
    fs.writeFileSync(outFile, page, 'utf8');
  }

  // 3) Landing — rewrite the shell head in place, leave #root empty.
  {
    const seo = getLandingSeo();
    const head = buildHead(seo, [courseJsonLd(CHAPTERS)], assetTags);
    // Only swap the head; #root stays empty (Landing has animation + model hooks).
    const page = swapHead(shellHtml, head);
    fs.writeFileSync(shellPath, page, 'utf8');
  }

  // 4) sitemap.xml + robots.txt.
  fs.writeFileSync(path.join(distClient, 'sitemap.xml'), buildSitemap(CHAPTERS), 'utf8');
  fs.writeFileSync(path.join(distClient, 'robots.txt'), buildRobots(), 'utf8');

  console.log(
    `[prerender] wrote ${chapterFilesWritten} chapter files (${totalBytes} bytes) + hub + landing + sitemap.xml + robots.txt`,
  );
  console.log(`[prerender] sample chapter title (attention): ${sampleTitle}`);
})();
