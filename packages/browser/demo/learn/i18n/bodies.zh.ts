// bodies.zh.ts — registry of TRANSLATED chapter/section body components.
//
// Translation workflow: translate learn/chapters/NN-x.tsx into
// learn/chapters/zh/NN-x.tsx (same body export name, demos omitted — they stay
// English and render from the English module; widgets shared via
// '../../widgets/*'; cross-chapter links via the locale-aware ChapterLink),
// then register the body component here. Any chapter/section missing from
// these maps falls back to the English body, rendered under /zh with a
// <TranslationPendingNote />.

import type * as React from 'react';

import { OverviewChapterBody } from '../chapters/zh/00-overview';
import { TokenizationChapterBody } from '../chapters/zh/01-tokenization';
import { EmbeddingsChapterBody } from '../chapters/zh/02-embeddings';
import { AttentionChapterBody } from '../chapters/zh/03-attention';
import { MultiheadGqaChapterBody } from '../chapters/zh/04-multihead-gqa';
import { RopeChapterBody } from '../chapters/zh/05-rope';
import { RmsNormChapterBody } from '../chapters/zh/06-rms-norm';
import { MlpChapterBody } from '../chapters/zh/07-mlp';
import { FullBlockChapterBody } from '../chapters/zh/08-full-block';
import { LmHeadChapterBody } from '../chapters/zh/09-lm-head';
import { SamplingChapterBody } from '../chapters/zh/09-sampling';
import { KvCacheChapterBody } from '../chapters/zh/10-kv-cache';
import { TrainingChapterBody } from '../chapters/zh/12-training';
import { ScalingChapterBody } from '../chapters/zh/13-scaling';
import { ArchitectureChapterBody } from '../chapters/zh/14-architecture';
import { PostTrainingChapterBody } from '../chapters/zh/15-post-training';
import {
  ArchitectureEvaluationSection,
  ArchitectureLocalEdgeSection,
} from '../sections/zh/architecture';
import {
  AttentionFlashAttentionSection,
  AttentionMemoryWallSection,
  AttentionRooflineSection,
} from '../sections/zh/attention';
import {
  KvCacheBatchingSection,
  KvCacheNumberFormatsSection,
  KvCachePrefixCachingSection,
  KvCacheQuantizationSection,
} from '../sections/zh/kv-cache';
import { LmHeadLogitLensSection } from '../sections/zh/lm-head';
import { PostTrainingDistillationSection } from '../sections/zh/post-training';
import {
  SamplingMtpSection,
  SamplingReasoningSection,
  SamplingToolCallingSection,
} from '../sections/zh/sampling';
import { ScalingLawsSection } from '../sections/zh/scaling';
import { TrainingDataCurationSection } from '../sections/zh/training';

/** Keyed by chapter id, e.g. 'overview'. */
export const ZH_CHAPTER_BODIES: Partial<Record<string, React.ComponentType>> = {
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

/** Keyed by `${chapterId}/${sectionId}`, e.g. 'attention/flash-attention'. */
export const ZH_SECTION_BODIES: Partial<Record<string, React.ComponentType>> = {
  'attention/memory-wall': AttentionMemoryWallSection,
  'attention/flash-attention': AttentionFlashAttentionSection,
  'attention/roofline': AttentionRooflineSection,
  'sampling/multi-token-prediction': SamplingMtpSection,
  'sampling/reasoning': SamplingReasoningSection,
  'sampling/tool-calling': SamplingToolCallingSection,
  'kv-cache/batching': KvCacheBatchingSection,
  'kv-cache/prefix-caching': KvCachePrefixCachingSection,
  'kv-cache/quantization': KvCacheQuantizationSection,
  'kv-cache/number-formats': KvCacheNumberFormatsSection,
  'lm-head/logit-lens': LmHeadLogitLensSection,
  'training/data-curation': TrainingDataCurationSection,
  'scaling/scaling-laws': ScalingLawsSection,
  'post-training/distillation': PostTrainingDistillationSection,
  'architecture/local-edge': ArchitectureLocalEdgeSection,
  'architecture/evaluation': ArchitectureEvaluationSection,
};
