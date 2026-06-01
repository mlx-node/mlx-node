// routes/chapters.$chapterId.tsx — Single chapter route (/chapters/:chapterId).
//
// Phase 2.A: beforeLoad validates chapterId and redirects to /chapters on miss.
// Phase 2.C: the component renders the full <LessonLayout /> from the legacy
// app, including the chapter body + "Try it now" demo. The chapter is resolved
// inside beforeLoad and surfaced through `Route.useRouteContext()`.

import { createFileRoute, redirect, useNavigate } from '@tanstack/react-router';
import { useEffect } from 'react';

import { Loading } from '../components/loading/Loading';
import { findChapter } from '../learn/chapters';
import { OverviewChapterBody } from '../learn/chapters/00-overview';
import { TokenizationChapterBody, TokenizerDemo } from '../learn/chapters/01-tokenization';
import { EmbeddingsChapterBody, EmbeddingsDemo } from '../learn/chapters/02-embeddings';
import { AttentionChapterBody, AttentionDemo } from '../learn/chapters/03-attention';
import { MultiheadGqaChapterBody, MultiheadGqaDemo } from '../learn/chapters/04-multihead-gqa';
import { RopeChapterBody, RopeDemo } from '../learn/chapters/05-rope';
import { RmsNormChapterBody, RmsNormDemo } from '../learn/chapters/06-rms-norm';
import { MlpChapterBody, MlpDemo } from '../learn/chapters/07-mlp';
import { FullBlockChapterBody, FullBlockDemo } from '../learn/chapters/08-full-block';
import { LmHeadChapterBody, LmHeadDemo } from '../learn/chapters/09-lm-head';
import { SamplingChapterBody, SamplingDemo } from '../learn/chapters/09-sampling';
import { KvCacheChapterBody, KvCacheDemo } from '../learn/chapters/10-kv-cache';
import { TrainingChapterBody, TrainingDemo } from '../learn/chapters/12-training';
import { ScalingChapterBody, ScalingDemo } from '../learn/chapters/13-scaling';
import { ArchitectureChapterBody } from '../learn/chapters/14-architecture';
import { PostTrainingChapterBody } from '../learn/chapters/15-post-training';
import { LessonLayout } from '../learn/LessonLayout';
import { triggerLocalPicker } from '../lib/local-model-picker';
import { useFreeChat } from '../providers/free-chat';
import { useModelLoader } from '../providers/model-loader';

// Chapters that are pure prose + self-contained widgets — no live model/worker
// demo (their tryItPanel is null, like the architecture chapter). They must stay
// readable while the model is still downloading or unavailable, so the
// route-level model gate below is skipped for them.
const PROSE_ONLY_CHAPTERS = new Set(['overview', 'post-training', 'architecture']);

// Chapters whose live demo needs the WebGPU device + WASM runtime, but NOT the
// big 1.6 GB model. The Training playground trains a tiny from-scratch
// transformer on the MLX autograd/optimizer stack, so it gates on the device
// being up (deviceReady) rather than the full model, and triggers a
// device-only init so the model is never downloaded for this chapter alone.
const DEVICE_ONLY_CHAPTERS = new Set(['training']);

function ChapterRouteComponent() {
  const { chapter } = Route.useRouteContext();
  const navigate = useNavigate();
  const { mlxWorkerRef, inspectorAbortRef } = useFreeChat();
  const { status, loadingText, loadingProgress, hostedModelAvailable, deviceReady, kickoffLoad, kickoffDeviceOnly } =
    useModelLoader();

  const isDeviceOnlyChapter = DEVICE_ONLY_CHAPTERS.has(chapter.id);

  // Auto-kickoff on direct URL landings (bookmark, hard reload of
  // /chapters/<id>). DEVICE_ONLY chapters (Training) bring up just the WebGPU
  // device — no model download. Every other live chapter triggers the full
  // model load. Both kickoffs are idempotent at the App level.
  //
  // Note: device-only kickoff is intentionally NOT gated on `hostedModelAvailable`
  // — it never fetches the model, so it works even when no hosted model exists
  // (and even on hosts where the user would otherwise pick a local directory).
  useEffect(() => {
    if (status === 'ready') return;
    if (isDeviceOnlyChapter) {
      if (deviceReady) return;
      kickoffDeviceOnly();
      return;
    }
    if (hostedModelAvailable === false) return;
    kickoffLoad();
  }, [status, isDeviceOnlyChapter, deviceReady, hostedModelAvailable, kickoffLoad, kickoffDeviceOnly]);

  // Gate the live demo behind the resource it needs. DEVICE_ONLY chapters
  // render once the WebGPU device is up (deviceReady); other live chapters wait
  // for the full model ('ready'). Prose-only chapters render immediately so
  // onboarding (Ch.1) isn't blocked behind a 1.6 GB download.
  if (isDeviceOnlyChapter) {
    if (!deviceReady && status !== 'ready') {
      return <Loading status={loadingText || 'Initializing WebGPU device...'} progress={loadingProgress} />;
    }
  } else if (status !== 'ready' && !PROSE_ONLY_CHAPTERS.has(chapter.id)) {
    return <Loading status={loadingText || null} progress={loadingProgress} />;
  }

  return (
    <LessonLayout
      current={chapter}
      // Only the architecture poster needs the full ~1280px width; the other
      // panel-less chapters (overview, post-training) are pure reading and
      // center at a comfortable article width.
      wideBody={chapter.id === 'architecture'}
      onOpenChapter={(chapterId) => {
        void navigate({ to: '/chapters/$chapterId', params: { chapterId }, search: (prev) => prev });
      }}
      onBackToIndex={() => {
        void navigate({ to: '/chapters', search: (prev) => prev });
      }}
      onOpenFreeChat={() => {
        if (hostedModelAvailable === false) {
          triggerLocalPicker();
          return;
        }
        kickoffLoad();
        void navigate({ to: '/chat', search: (prev) => prev });
      }}
      tryItPanel={
        chapter.id === 'attention' ? (
          <AttentionDemo workerRef={mlxWorkerRef} abortRef={inspectorAbortRef} />
        ) : chapter.id === 'multi-head-gqa' ? (
          <MultiheadGqaDemo workerRef={mlxWorkerRef} abortRef={inspectorAbortRef} />
        ) : chapter.id === 'tokenization' ? (
          <TokenizerDemo workerRef={mlxWorkerRef} abortRef={inspectorAbortRef} />
        ) : chapter.id === 'embeddings' ? (
          <EmbeddingsDemo workerRef={mlxWorkerRef} abortRef={inspectorAbortRef} />
        ) : chapter.id === 'rope' ? (
          <RopeDemo workerRef={mlxWorkerRef} abortRef={inspectorAbortRef} />
        ) : chapter.id === 'rmsnorm' ? (
          <RmsNormDemo workerRef={mlxWorkerRef} abortRef={inspectorAbortRef} />
        ) : chapter.id === 'mlp' ? (
          <MlpDemo workerRef={mlxWorkerRef} abortRef={inspectorAbortRef} />
        ) : chapter.id === 'full-block' ? (
          <FullBlockDemo workerRef={mlxWorkerRef} abortRef={inspectorAbortRef} />
        ) : chapter.id === 'lm-head' ? (
          <LmHeadDemo workerRef={mlxWorkerRef} abortRef={inspectorAbortRef} />
        ) : chapter.id === 'sampling' ? (
          <SamplingDemo workerRef={mlxWorkerRef} abortRef={inspectorAbortRef} />
        ) : chapter.id === 'kv-cache' ? (
          <KvCacheDemo workerRef={mlxWorkerRef} abortRef={inspectorAbortRef} />
        ) : chapter.id === 'training' ? (
          <TrainingDemo workerRef={mlxWorkerRef} abortRef={inspectorAbortRef} />
        ) : chapter.id === 'scaling' ? (
          <ScalingDemo workerRef={mlxWorkerRef} abortRef={inspectorAbortRef} />
        ) : null
        /* 'architecture' renders its interactive poster inline in the body and
           intentionally has no "Try it now" panel (the column is dropped). */
      }
    >
      {chapter.id === 'attention' ? (
        <AttentionChapterBody />
      ) : chapter.id === 'multi-head-gqa' ? (
        <MultiheadGqaChapterBody />
      ) : chapter.id === 'tokenization' ? (
        <TokenizationChapterBody />
      ) : chapter.id === 'embeddings' ? (
        <EmbeddingsChapterBody />
      ) : chapter.id === 'rope' ? (
        <RopeChapterBody />
      ) : chapter.id === 'rmsnorm' ? (
        <RmsNormChapterBody />
      ) : chapter.id === 'mlp' ? (
        <MlpChapterBody />
      ) : chapter.id === 'full-block' ? (
        <FullBlockChapterBody />
      ) : chapter.id === 'lm-head' ? (
        <LmHeadChapterBody />
      ) : chapter.id === 'sampling' ? (
        <SamplingChapterBody />
      ) : chapter.id === 'kv-cache' ? (
        <KvCacheChapterBody />
      ) : chapter.id === 'training' ? (
        <TrainingChapterBody />
      ) : chapter.id === 'scaling' ? (
        <ScalingChapterBody />
      ) : chapter.id === 'architecture' ? (
        <ArchitectureChapterBody />
      ) : chapter.id === 'overview' ? (
        <OverviewChapterBody />
      ) : chapter.id === 'post-training' ? (
        <PostTrainingChapterBody />
      ) : (
        <div className="text-muted-foreground">This chapter is not yet authored.</div>
      )}
    </LessonLayout>
  );
}

export const Route = createFileRoute('/chapters/$chapterId')({
  beforeLoad: ({ params }) => {
    const chapter = findChapter(params.chapterId);
    if (!chapter) {
      throw redirect({ to: '/chapters' });
    }
    return { chapter };
  },
  component: ChapterRouteComponent,
});
