// seo-metadata.ts — React-free, pure SEO data + helpers.
//
// Importable by BOTH the prerender SSR entry (scripts/prerender.entry.tsx, run
// under node) AND client route components. MUST NOT import React, the worker,
// or anything browser-only — only the (type-only) ChapterMeta shape and the
// equally React-free locale core (lib/i18n.ts).
//
// Locale rules:
// - Every builder takes a trailing `locale` param defaulting to 'en', so
//   pre-i18n call sites keep compiling with unchanged behavior.
// - Chapter/section titles + blurbs must arrive ALREADY localized (callers
//   pass `localizedChapter()` / `localizedSection()` output) — these helpers
//   never re-localize meta; they only localize the site-level strings
//   (landing/hub) and the URL/inLanguage plumbing.
// - Title convention (one convention, applied everywhere): SITE_NAME stays
//   "How LLMs Work" verbatim in zh titles too — `分词 · How LLMs Work`.

import type { ChapterMeta, SectionMeta } from '../learn/chapters';
import { DEFAULT_LOCALE, htmlLang, localePath, type Locale } from './i18n';

export const SITE_ORIGIN = 'https://mlx.void.app';
export const SITE_NAME = 'How LLMs Work';
export const OG_IMAGE = SITE_ORIGIN + '/og-image.png';

export const LANDING_TITLE = 'How LLMs Work — Run a Real Language Model in Your Browser';
export const LANDING_DESCRIPTION =
  'An interactive course on how large language models actually work. Step through 16 chapters — tokenization, embeddings, attention, RoPE, sampling, KV cache, training — while a real 0.8-billion-parameter model (Qwen3.5) runs live in your browser with WebGPU. No install, no API keys, nothing leaves your machine.';

export const LANDING_TITLE_ZH = '大语言模型是如何工作的 — 在浏览器里运行一个真实的语言模型';
export const LANDING_DESCRIPTION_ZH =
  '一门讲解大语言模型究竟如何工作的交互式课程。16 个章节带你逐步理解分词、嵌入、注意力、RoPE、采样、KV 缓存与训练——同时,一个真实的 8 亿参数模型(Qwen3.5)通过 WebGPU 在你的浏览器里本地运行。无需安装、无需 API key,数据不会离开你的设备。';

const LANDING_TITLES: Record<Locale, string> = { en: LANDING_TITLE, zh: LANDING_TITLE_ZH };
const LANDING_DESCRIPTIONS: Record<Locale, string> = { en: LANDING_DESCRIPTION, zh: LANDING_DESCRIPTION_ZH };

const CHAPTERS_HUB_TITLES: Record<Locale, string> = {
  en: `Chapters · ${SITE_NAME}`,
  zh: `章节 · ${SITE_NAME}`,
};
const CHAPTERS_HUB_DESCRIPTIONS: Record<Locale, string> = {
  en: 'All 16 chapters of How LLMs Work — from tokenization and embeddings to attention, training, and the full model architecture, each taught with a real LLM running live in your browser.',
  zh: 'How LLMs Work 全部 16 个章节——从分词、嵌入到注意力、训练,再到完整的模型架构,每一章都配有一个在你浏览器里实时运行的真实大语言模型。',
};

/** Breadcrumb label for the /chapters hub. */
const CHAPTERS_CRUMB: Record<Locale, string> = { en: 'Chapters', zh: '章节' };

/** Open Graph locale tag value (og:locale). */
export function ogLocale(locale: Locale): string {
  return locale === 'zh' ? 'zh_CN' : 'en_US';
}

export type HreflangAlternate = { hreflang: string; href: string };

/**
 * Reciprocal hreflang alternates for a bare (unprefixed) app path.
 * Every bilingual page carries the SAME set in both locales:
 * en → unprefixed URL, zh-CN → /zh URL, x-default → the English URL.
 * Single-locale pages (e.g. /chat) must NOT use this — they carry no alternates.
 */
export function hreflangAlternates(path: string): HreflangAlternate[] {
  return [
    { hreflang: 'en', href: SITE_ORIGIN + path },
    { hreflang: 'zh-CN', href: SITE_ORIGIN + localePath('zh', path) },
    { hreflang: 'x-default', href: SITE_ORIGIN + path },
  ];
}

export type PageSeo = {
  title: string;
  description: string;
  /** Absolute canonical URL of the SAME-locale page. */
  canonical: string;
  ogType: 'website' | 'article';
  image: string;
  /** Locale this page is rendered in (drives <html lang> + og:locale). */
  locale: Locale;
  /**
   * hreflang alternates for this page (en + zh-CN + x-default), or [] for
   * pages that exist in a single locale only (e.g. /chat).
   */
  alternates: HreflangAlternate[];
};

/** Absolute same-locale URL for a bare app path. */
function localeUrl(locale: Locale, path: string): string {
  return SITE_ORIGIN + localePath(locale, path);
}

export function getChapterSeo(chapter: ChapterMeta, locale: Locale = DEFAULT_LOCALE): PageSeo {
  const path = `/chapters/${chapter.id}`;
  return {
    title: `${chapter.title} · ${SITE_NAME}`,
    description: chapter.blurb,
    canonical: localeUrl(locale, path),
    ogType: 'article',
    image: OG_IMAGE,
    locale,
    alternates: hreflangAlternates(path),
  };
}

export function getSectionSeo(chapter: ChapterMeta, section: SectionMeta, locale: Locale = DEFAULT_LOCALE): PageSeo {
  const path = `/chapters/${chapter.id}/${section.id}`;
  return {
    title: `${section.title} · ${chapter.title} · ${SITE_NAME}`,
    description: section.blurb,
    canonical: localeUrl(locale, path),
    ogType: 'article',
    image: OG_IMAGE,
    locale,
    alternates: hreflangAlternates(path),
  };
}

export function getLandingSeo(locale: Locale = DEFAULT_LOCALE): PageSeo {
  return {
    title: LANDING_TITLES[locale],
    description: LANDING_DESCRIPTIONS[locale],
    canonical: localeUrl(locale, '/'),
    ogType: 'website',
    image: OG_IMAGE,
    locale,
    alternates: hreflangAlternates('/'),
  };
}

export function getChatSeo(): PageSeo {
  return {
    title: `Free chat · ${SITE_NAME}`,
    description:
      'Chat with a 0.8-billion-parameter language model (Qwen3.5) running entirely in your browser via WebGPU — part of the How LLMs Work course. No server, no API keys.',
    canonical: `${SITE_ORIGIN}/chat`,
    ogType: 'website',
    image: OG_IMAGE,
    // /chat is English-only (there is no /zh/chat) — no hreflang alternates.
    locale: 'en',
    alternates: [],
  };
}

export function getJspaceSeo(): PageSeo {
  return {
    title: `J-Space · ${SITE_NAME}`,
    description:
      'Type any prompt and watch a language model make up its mind, layer by layer. A Jacobian lens that runs entirely in your browser on WebGPU.',
    canonical: `${SITE_ORIGIN}/jspace`,
    ogType: 'website',
    image: OG_IMAGE,
    // /jspace is English-only (there is no /zh/jspace) — no hreflang alternates.
    locale: 'en',
    alternates: [],
  };
}

export function getChaptersHubSeo(locale: Locale = DEFAULT_LOCALE): PageSeo {
  return {
    title: CHAPTERS_HUB_TITLES[locale],
    description: CHAPTERS_HUB_DESCRIPTIONS[locale],
    canonical: localeUrl(locale, '/chapters'),
    ogType: 'website',
    image: OG_IMAGE,
    locale,
    alternates: hreflangAlternates('/chapters'),
  };
}

export function courseJsonLd(chapters: ChapterMeta[], locale: Locale = DEFAULT_LOCALE): object {
  return {
    '@context': 'https://schema.org',
    '@type': 'Course',
    name: SITE_NAME,
    description: LANDING_DESCRIPTIONS[locale],
    url: localeUrl(locale, '/'),
    inLanguage: htmlLang(locale),
    isAccessibleForFree: true,
    provider: {
      '@type': 'Organization',
      name: SITE_NAME,
      url: SITE_ORIGIN + '/',
    },
    hasPart: chapters.map((c, i) => ({
      '@type': 'LearningResource',
      name: c.title,
      url: localeUrl(locale, `/chapters/${c.id}`),
      position: i + 1,
    })),
  };
}

export function chapterJsonLd(chapter: ChapterMeta, locale: Locale = DEFAULT_LOCALE): object[] {
  const url = localeUrl(locale, `/chapters/${chapter.id}`);
  return [
    {
      '@context': 'https://schema.org',
      '@type': ['LearningResource', 'Article'],
      name: chapter.title,
      headline: chapter.title,
      description: chapter.blurb,
      url,
      image: OG_IMAGE,
      learningResourceType: 'lesson',
      educationalLevel: 'beginner',
      inLanguage: htmlLang(locale),
      isAccessibleForFree: true,
      position: chapter.number,
      isPartOf: {
        '@type': 'Course',
        name: SITE_NAME,
        url: localeUrl(locale, '/'),
      },
      author: {
        '@type': 'Organization',
        name: SITE_NAME,
      },
      publisher: {
        '@type': 'Organization',
        name: SITE_NAME,
      },
    },
    {
      '@context': 'https://schema.org',
      '@type': 'BreadcrumbList',
      itemListElement: [
        { '@type': 'ListItem', position: 1, name: SITE_NAME, item: localeUrl(locale, '/') },
        { '@type': 'ListItem', position: 2, name: CHAPTERS_CRUMB[locale], item: localeUrl(locale, '/chapters') },
        { '@type': 'ListItem', position: 3, name: chapter.title, item: url },
      ],
    },
  ];
}

export function sectionJsonLd(chapter: ChapterMeta, section: SectionMeta, locale: Locale = DEFAULT_LOCALE): object[] {
  const url = localeUrl(locale, `/chapters/${chapter.id}/${section.id}`);
  return [
    {
      '@context': 'https://schema.org',
      '@type': ['LearningResource', 'Article'],
      name: section.title,
      headline: `${section.title} — ${chapter.title}`,
      description: section.blurb,
      url,
      image: OG_IMAGE,
      learningResourceType: 'supplemental lesson',
      educationalLevel: 'intermediate',
      inLanguage: htmlLang(locale),
      isAccessibleForFree: true,
      // The parent chapter this deep-dive belongs to.
      isPartOf: {
        '@type': ['LearningResource', 'Article'],
        name: chapter.title,
        url: localeUrl(locale, `/chapters/${chapter.id}`),
      },
      author: {
        '@type': 'Organization',
        name: SITE_NAME,
      },
      publisher: {
        '@type': 'Organization',
        name: SITE_NAME,
      },
    },
    {
      '@context': 'https://schema.org',
      '@type': 'BreadcrumbList',
      itemListElement: [
        { '@type': 'ListItem', position: 1, name: SITE_NAME, item: localeUrl(locale, '/') },
        { '@type': 'ListItem', position: 2, name: CHAPTERS_CRUMB[locale], item: localeUrl(locale, '/chapters') },
        {
          '@type': 'ListItem',
          position: 3,
          name: chapter.title,
          item: localeUrl(locale, `/chapters/${chapter.id}`),
        },
        { '@type': 'ListItem', position: 4, name: section.title, item: url },
      ],
    },
  ];
}
