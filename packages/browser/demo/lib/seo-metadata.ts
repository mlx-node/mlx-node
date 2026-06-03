// seo-metadata.ts — React-free, pure SEO data + helpers.
//
// Importable by BOTH the prerender SSR entry (scripts/prerender.entry.tsx, run
// under node) AND client route components. MUST NOT import React, the worker,
// or anything browser-only — only the (type-only) ChapterMeta shape.

import type { ChapterMeta } from '../learn/chapters';

export const SITE_ORIGIN = 'https://mlx.void.app';
export const SITE_NAME = 'How LLMs Work';
export const OG_IMAGE = SITE_ORIGIN + '/og-image.png';

export const LANDING_TITLE = 'How LLMs Work — Run a Real Language Model in Your Browser';
export const LANDING_DESCRIPTION =
  'An interactive course on how large language models actually work. Step through 16 chapters — tokenization, embeddings, attention, RoPE, sampling, KV cache, training — while a real 0.8-billion-parameter model (Qwen3.5) runs live in your browser with WebGPU. No install, no API keys, nothing leaves your machine.';

export type PageSeo = {
  title: string;
  description: string;
  canonical: string;
  ogType: 'website' | 'article';
  image: string;
};

export function getChapterSeo(chapter: ChapterMeta): PageSeo {
  return {
    title: `${chapter.title} · ${SITE_NAME}`,
    description: chapter.blurb,
    canonical: `${SITE_ORIGIN}/chapters/${chapter.id}`,
    ogType: 'article',
    image: OG_IMAGE,
  };
}

export function getLandingSeo(): PageSeo {
  return {
    title: LANDING_TITLE,
    description: LANDING_DESCRIPTION,
    canonical: `${SITE_ORIGIN}/`,
    ogType: 'website',
    image: OG_IMAGE,
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
  };
}

export function getChaptersHubSeo(): PageSeo {
  return {
    title: `Chapters · ${SITE_NAME}`,
    description:
      'All 16 chapters of How LLMs Work — from tokenization and embeddings to attention, training, and the full model architecture, each taught with a real LLM running live in your browser.',
    canonical: `${SITE_ORIGIN}/chapters`,
    ogType: 'website',
    image: OG_IMAGE,
  };
}

export function courseJsonLd(chapters: ChapterMeta[]): object {
  return {
    '@context': 'https://schema.org',
    '@type': 'Course',
    name: SITE_NAME,
    description: LANDING_DESCRIPTION,
    url: SITE_ORIGIN + '/',
    inLanguage: 'en',
    isAccessibleForFree: true,
    provider: {
      '@type': 'Organization',
      name: SITE_NAME,
      url: SITE_ORIGIN + '/',
    },
    hasPart: chapters.map((c, i) => ({
      '@type': 'LearningResource',
      name: c.title,
      url: `${SITE_ORIGIN}/chapters/${c.id}`,
      position: i + 1,
    })),
  };
}

export function chapterJsonLd(chapter: ChapterMeta): object[] {
  return [
    {
      '@context': 'https://schema.org',
      '@type': ['LearningResource', 'Article'],
      name: chapter.title,
      headline: chapter.title,
      description: chapter.blurb,
      url: `${SITE_ORIGIN}/chapters/${chapter.id}`,
      image: OG_IMAGE,
      learningResourceType: 'lesson',
      educationalLevel: 'beginner',
      inLanguage: 'en',
      isAccessibleForFree: true,
      position: chapter.number,
      isPartOf: {
        '@type': 'Course',
        name: SITE_NAME,
        url: SITE_ORIGIN + '/',
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
        { '@type': 'ListItem', position: 1, name: SITE_NAME, item: SITE_ORIGIN + '/' },
        { '@type': 'ListItem', position: 2, name: 'Chapters', item: SITE_ORIGIN + '/chapters' },
        { '@type': 'ListItem', position: 3, name: chapter.title, item: `${SITE_ORIGIN}/chapters/${chapter.id}` },
      ],
    },
  ];
}
