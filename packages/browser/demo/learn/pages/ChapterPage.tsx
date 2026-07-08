// learn/pages/ChapterPage.tsx — the shared chapter page (model auto-load,
// demo-panel wiring, LessonLayout), extracted from the former
// routes/chapters.$chapterId.index.tsx route component so the English route
// and the /zh mirror render the SAME logic. Locale-agnostic: the locale flows
// in through React context (LocaleProvider in the /zh layout route), the
// already-locale-resolved `chapter` comes in as a prop from route context, and
// navigation targets are locale-prefixed via useLocaleNavigate().
//
// Auto-load on entry: opening a chapter whose demo needs a resource starts the
// load automatically (see the effect below), along the gesture-free paths only.

import { useNavigate } from '@tanstack/react-router';
import { useEffect } from 'react';

import { ModelConsentLayer } from '../../components/loading/ModelConsentLayer';
import { triggerLocalPicker } from '../../lib/local-model-picker';
import { useLocale } from '../../lib/i18n-react';
import { useLocaleNavigate } from '../../lib/locale-navigate';
import { useFreeChat } from '../../providers/free-chat';
import { useModelLoader } from '../../providers/model-loader';
import { CHAPTER_BODIES } from '../bodies';
import type { ChapterMeta } from '../chapters';
import { TokenizerDemo } from '../chapters/01-tokenization';
import { EmbeddingsDemo } from '../chapters/02-embeddings';
import { AttentionDemo } from '../chapters/03-attention';
import { MultiheadGqaDemo } from '../chapters/04-multihead-gqa';
import { RopeDemo } from '../chapters/05-rope';
import { RmsNormDemo } from '../chapters/06-rms-norm';
import { MlpDemo } from '../chapters/07-mlp';
import { FullBlockDemo } from '../chapters/08-full-block';
import { LmHeadDemo } from '../chapters/09-lm-head';
import { SamplingDemo } from '../chapters/09-sampling';
import { KvCacheDemo } from '../chapters/10-kv-cache';
import { TrainingDemo } from '../chapters/12-training';
import { ZH_CHAPTER_BODIES } from '../i18n/bodies.zh';
import { TranslationPendingNote } from '../i18n/TranslationPendingNote';
import { useUiStrings } from '../i18n/ui-react';
import { LessonLayout } from '../LessonLayout';

// The chapter body always renders immediately (LessonLayout's reading column is
// model-free). Only the right-hand "Try it now" demo panel is gated, classified
// by the resource it needs:
//
//   model  — needs the full ~1.6 GB model ('ready'); wrapped in a
//            <ModelConsentLayer mode="model">. On entry the route auto-starts a
//            HOSTED fetch (see the auto-load effect below); when no hosted model
//            is available the layer's CTA stays as the local-picker fallback.
//   device — needs only the WebGPU device + WASM runtime, NOT the model
//            (Training trains a tiny from-scratch transformer); wrapped in a
//            <ModelConsentLayer mode="device">. On entry the route auto-brings
//            up WebGPU (no download); the CTA remains as a manual/retry fallback.
//   none   — pure JS, needs nothing; the demo renders immediately with no
//            consent gate and triggers NO load.
//
// Chapters absent from all three sets have no demo panel at all (their inline
// content/poster lives in the body) and pass `null` for `tryItPanel`.

// model-panel chapters — gate the demo on the full model behind a consent CTA.
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

export function ChapterPage({ chapter }: { chapter: ChapterMeta }) {
  const locale = useLocale();
  const ui = useUiStrings();
  const navigate = useNavigate();
  const go = useLocaleNavigate();
  const { mlxWorkerRef, inspectorAbortRef } = useFreeChat();
  const { status, hostedModelAvailable, kickoffLoad, kickoffDeviceOnly, clearLoadError } = useModelLoader();

  const isModelPanelChapter = MODEL_PANEL_CHAPTERS.has(chapter.id);
  const isDeviceOnlyChapter = DEVICE_ONLY_CHAPTERS.has(chapter.id);

  // Auto-load on chapter entry: opening a chapter whose demo needs a resource
  // now starts the load automatically instead of waiting for a click — but only
  // along the paths that need NO user gesture:
  //   - device-only chapters (Training): bring the WebGPU device up. No model
  //     download; kickoffDeviceOnly is idempotent.
  //   - model-panel chapters: only when a HOSTED model is available, so the
  //     fetch is a gesture-free download. When no hosted model exists the load
  //     needs the local-file picker (triggerLocalPicker), which browsers only
  //     allow from a real user gesture — so we leave that chapter's consent CTA
  //     in place rather than failing to open a dialog.
  // Guarded to status 'idle': this never auto-retries a failed load (which could
  // loop on an unsupported device) and never disturbs an in-flight/ready model.
  // The <ModelConsentLayer> still renders the loading/error/ready states; its
  // CTA stays as the manual fallback (notably the local-picker path above).
  useEffect(() => {
    if (status !== 'idle') return;
    if (isDeviceOnlyChapter) {
      kickoffDeviceOnly();
    } else if (isModelPanelChapter && hostedModelAvailable === true) {
      kickoffLoad();
    }
  }, [
    chapter.id,
    status,
    isModelPanelChapter,
    isDeviceOnlyChapter,
    hostedModelAvailable,
    kickoffLoad,
    kickoffDeviceOnly,
  ]);

  // errorBanner is a single GLOBAL loader state. A failed load on a prior
  // chapter would otherwise carry its error message into THIS freshly-opened
  // chapter's consent layer (and the chat overlay) even though the user has
  // taken no action here. Clear a stale error on chapter change so the user
  // sees the neutral consent prompt. clearLoadError is guarded to the error
  // state only — it never aborts an in-flight load or drops a ready model — so
  // this does NOT start (or restart) any download.
  useEffect(() => {
    clearLoadError();
  }, [chapter.id, clearLoadError]);

  // The chapter's SEO <head> (title/canonical/OG/JSON-LD) is synced centrally by
  // the root route's head manager (routes/__root.tsx) on pathname change.

  // The chapter's live demo, or null for panel-less chapters (overview,
  // post-training, architecture, scaling — their inline content/poster lives in
  // the body). `none`-resource demos (rope, kv-cache) render here immediately;
  // model/device demos are gated by <ModelConsentLayer> at the wire-in below, so
  // they only mount once their resource is ready. Demos take props, so they stay
  // an inline ternary here (unlike the bodies, which come from CHAPTER_BODIES).
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
    ) : null;
  // 'scaling', 'architecture', 'overview', 'post-training' render inline in the
  // body and intentionally have no "Try it now" panel (the column is dropped).

  const hasDemoPanel = demoPanel !== null;

  // Body resolution: the canonical English map, with a per-chapter Chinese
  // override under /zh. A /zh chapter with no translation yet falls back to
  // the English body, flagged by <TranslationPendingNote /> on top.
  const EnglishBody = CHAPTER_BODIES[chapter.id];
  const ZhBody = locale === 'zh' ? ZH_CHAPTER_BODIES[chapter.id] : undefined;
  const Body = ZhBody ?? EnglishBody;
  const showTranslationPending = locale === 'zh' && ZhBody === undefined && Body !== undefined;

  return (
    <LessonLayout
      current={chapter}
      // Only the architecture poster needs the full ~1280px width; the other
      // panel-less chapters (overview, post-training) are pure reading and
      // center at a comfortable article width.
      wideBody={chapter.id === 'architecture'}
      modelReady={status === 'ready'}
      onLoadModel={() => {
        if (hostedModelAvailable === false) {
          triggerLocalPicker();
          return;
        }
        kickoffLoad();
      }}
      onOpenChapter={(chapterId) => {
        go(`/chapters/${chapterId}`);
      }}
      onOpenSection={(chapterId, sectionId) => {
        go(`/chapters/${chapterId}/${sectionId}`);
      }}
      onBackToIndex={() => {
        go('/chapters');
      }}
      onOpenFreeChat={() => {
        // Just open the chat surface — do NOT kick off a load here. The chat
        // overlay (<ChatLayerOverlay>) is the single consent gate for chat: it
        // surfaces its own "Load the model to chat" CTA (and the local-model
        // picker when no hosted model is available) when the model isn't ready.
        // This keeps the ~1.6 GB download behind that explicit chat consent.
        // Chat is deliberately NOT locale-prefixed — there is no /zh/chat route;
        // the overlay is the same surface for both locales.
        void navigate({ to: '/chat', search: (prev) => prev });
      }}
      tryItPanel={
        // Gate ONLY the demo panel; the reading body always renders. model and
        // device demos are wrapped in <ModelConsentLayer>, which surfaces the
        // loading/error/ready states (and a manual CTA fallback) and mounts the
        // live demo only once its resource is ready. The route auto-starts the
        // load on entry (see the auto-load effect above). Pure-JS `none` demos
        // (rope, kv-cache) need nothing and render directly.
        hasDemoPanel && (isModelPanelChapter || isDeviceOnlyChapter) ? (
          <ModelConsentLayer mode={isDeviceOnlyChapter ? 'device' : 'model'}>{demoPanel}</ModelConsentLayer>
        ) : (
          demoPanel
        )
      }
    >
      {Body ? (
        <>
          {showTranslationPending ? <TranslationPendingNote /> : null}
          <Body />
        </>
      ) : (
        <div className="text-muted-foreground">{ui.chapterPage.notAuthored}</div>
      )}
    </LessonLayout>
  );
}
