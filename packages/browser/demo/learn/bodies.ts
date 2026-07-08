// bodies.ts — the canonical chapterId → English body-component map.
//
// Single source of truth shared by the chapter routes (English and /zh, which
// falls back to these when a chapter has no translation in
// learn/i18n/bodies.zh.ts yet) and the prerender entry
// (scripts/prerender.entry.tsx), which previously mirrored this import block.
// Demo panels are NOT here — they take props and stay with the route.

import type * as React from 'react';

import { OverviewChapterBody } from './chapters/00-overview';
import { TokenizationChapterBody } from './chapters/01-tokenization';
import { EmbeddingsChapterBody } from './chapters/02-embeddings';
import { AttentionChapterBody } from './chapters/03-attention';
import { MultiheadGqaChapterBody } from './chapters/04-multihead-gqa';
import { RopeChapterBody } from './chapters/05-rope';
import { RmsNormChapterBody } from './chapters/06-rms-norm';
import { MlpChapterBody } from './chapters/07-mlp';
import { FullBlockChapterBody } from './chapters/08-full-block';
import { LmHeadChapterBody } from './chapters/09-lm-head';
import { SamplingChapterBody } from './chapters/09-sampling';
import { KvCacheChapterBody } from './chapters/10-kv-cache';
import { TrainingChapterBody } from './chapters/12-training';
import { ScalingChapterBody } from './chapters/13-scaling';
import { ArchitectureChapterBody } from './chapters/14-architecture';
import { PostTrainingChapterBody } from './chapters/15-post-training';

export const CHAPTER_BODIES: Record<string, React.ComponentType> = {
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
  architecture: ArchitectureChapterBody,
  'post-training': PostTrainingChapterBody,
};
