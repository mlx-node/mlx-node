// routes/chapters.$chapterId.tsx — Single chapter route (/chapters/:chapterId).
//
// Phase 2.A: beforeLoad validates chapterId and redirects to /chapters on miss.
// Phase 2.C: the component renders the full <LessonLayout /> from the legacy
// app, including the chapter body + "Try it now" demo. The chapter is resolved
// inside beforeLoad and surfaced through `Route.useRouteContext()`.

import { createFileRoute, redirect, useNavigate } from '@tanstack/react-router';
import { useEffect } from 'react';

import { PanelLoading } from '../components/loading/PanelLoading';
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

// The chapter body always renders immediately (LessonLayout's reading column is
// model-free). Only the right-hand "Try it now" demo panel is gated, classified
// by the resource it needs:
//
//   model  — needs the full ~1.6 GB model ('ready'); these warm the model in the
//            background while the learner reads.
//   device — needs only the WebGPU device + WASM runtime, NOT the model
//            (Training trains a tiny from-scratch transformer); gates on
//            `deviceReady` and triggers a device-only init.
//   none   — pure JS, needs nothing; the demo renders immediately and we do NOT
//            kick off any download (one exception below: `overview`).
//
// Chapters absent from all three sets have no demo panel at all (their inline
// content/poster lives in the body) and pass `null` for `tryItPanel`.

// model-panel chapters — gate the demo on the full model and background-warm it.
const MODEL_PANEL_CHAPTERS = new Set([
  'tokenization',
  'embeddings',
  'attention',
  'multi-head-gqa',
  'rmsnorm',
  'mlp',
  'full-block',
  'lm-head',
  'sampling',
]);

// device-panel chapters — gate on the WebGPU device only (no model download).
const DEVICE_ONLY_CHAPTERS = new Set(['training']);

function ChapterRouteComponent() {
  const { chapter } = Route.useRouteContext();
  const navigate = useNavigate();
  const { mlxWorkerRef, inspectorAbortRef } = useFreeChat();
  const { status, loadingText, loadingProgress, hostedModelAvailable, deviceReady, kickoffLoad, kickoffDeviceOnly } =
    useModelLoader();

  const isModelPanelChapter = MODEL_PANEL_CHAPTERS.has(chapter.id);
  const isDeviceOnlyChapter = DEVICE_ONLY_CHAPTERS.has(chapter.id);

  // Whether the demo panel's resource is up. `none`-resource panels (rope,
  // kv-cache, scaling) are pure JS and always ready. model panels wait for the
  // full model; device panels wait for the WebGPU device (a full 'ready'
  // implies the device is up too).
  const panelReady = isModelPanelChapter
    ? status === 'ready'
    : isDeviceOnlyChapter
      ? deviceReady || status === 'ready'
      : true;

  // Auto-kickoff (background warm-up) on direct URL landings (bookmark, hard
  // reload of /chapters/<id>) so the panel's resource is loading while the
  // learner reads. DEVICE_ONLY chapters (Training) bring up just the WebGPU
  // device — no model download. model-panel chapters trigger the full model
  // load; we also preload for `overview` (Ch.1) as a deliberate onboarding
  // cache-warm — the learner will need the model in Ch.2. Pure-JS `none`
  // chapters (rope, kv-cache, scaling, post-training, architecture) trigger NO
  // download. All kickoffs are idempotent at the App level.
  //
  // Note: device-only kickoff is intentionally NOT gated on `hostedModelAvailable`
  // — it never fetches the model, so it works even when no hosted model exists
  // (and even on hosts where the user would otherwise pick a local directory).
  const shouldPreloadModel = isModelPanelChapter || chapter.id === 'overview';
  useEffect(() => {
    if (status === 'ready') return;
    if (isDeviceOnlyChapter) {
      if (deviceReady) return;
      kickoffDeviceOnly();
      return;
    }
    if (!shouldPreloadModel) return;
    if (hostedModelAvailable === false) return;
    kickoffLoad();
  }, [
    status,
    isDeviceOnlyChapter,
    shouldPreloadModel,
    deviceReady,
    hostedModelAvailable,
    kickoffLoad,
    kickoffDeviceOnly,
  ]);

  // The chapter's live demo, or null for panel-less chapters (overview,
  // post-training, architecture — their inline content/poster lives in the
  // body). `none`-resource demos (rope, kv-cache, scaling) render here
  // immediately; model/device demos render here once `panelReady`.
  const demoPanel =
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
    ) : null;
  // 'architecture', 'overview', 'post-training' render inline in the body and
  // intentionally have no "Try it now" panel (the column is dropped).

  const hasDemoPanel = demoPanel !== null;

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
        // Gate ONLY the demo panel: render the real demo once its resource is
        // ready; otherwise show a compact in-column loading affordance so the
        // 3-col layout (and the reading body) stays put while it warms.
        hasDemoPanel && !panelReady ? (
          <PanelLoading
            status={loadingText || null}
            progress={loadingProgress}
            hostedUnavailable={isModelPanelChapter && hostedModelAvailable === false}
            onLoadLocal={triggerLocalPicker}
          />
        ) : (
          demoPanel
        )
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
