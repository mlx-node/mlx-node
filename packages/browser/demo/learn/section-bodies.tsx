// section-bodies.tsx — the single registry mapping a sub-chapter to its body
// component, keyed `${chapterId}/${sectionId}`.
//
// ONE source of truth, imported by BOTH the section route
// (routes/chapters.$chapterId.$sectionId.tsx) and the prerenderer
// (scripts/prerender.entry.tsx). This avoids the chapter pattern's duplication
// (a route-level id→component ternary AND a separate prerender BODIES map that
// must be kept in sync). Section bodies are pure, model-free reading prose, so
// the same component renders in the SPA and to static crawlable HTML.

import type * as React from 'react';

import {
  ArchitectureEvaluationSection,
  ArchitectureLocalEdgeSection,
} from './sections/architecture';
import {
  AttentionFlashAttentionSection,
  AttentionMemoryWallSection,
  AttentionRooflineSection,
} from './sections/attention';
import {
  KvCacheBatchingSection,
  KvCacheNumberFormatsSection,
  KvCachePrefixCachingSection,
  KvCacheQuantizationSection,
} from './sections/kv-cache';
import { LmHeadJacobianLensSection, LmHeadLogitLensSection } from './sections/lm-head';
import { PostTrainingDistillationSection } from './sections/post-training';
import {
  SamplingMtpSection,
  SamplingReasoningSection,
  SamplingToolCallingSection,
} from './sections/sampling';
import { ScalingLawsSection } from './sections/scaling';
import { TrainingDataCurationSection } from './sections/training';

export const SECTION_BODIES: Record<string, React.ComponentType> = {
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
  'lm-head/jacobian-lens': LmHeadJacobianLensSection,
  'training/data-curation': TrainingDataCurationSection,
  'scaling/scaling-laws': ScalingLawsSection,
  'post-training/distillation': PostTrainingDistillationSection,
  'architecture/local-edge': ArchitectureLocalEdgeSection,
  'architecture/evaluation': ArchitectureEvaluationSection,
};

/** Registry key for a sub-chapter. */
export function sectionBodyKey(chapterId: string, sectionId: string): string {
  return `${chapterId}/${sectionId}`;
}
