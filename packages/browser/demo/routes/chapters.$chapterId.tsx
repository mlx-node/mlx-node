// routes/chapters.$chapterId.tsx — Single chapter route (/chapters/:chapterId).
//
// Phase 2.A: beforeLoad validates chapterId and redirects to /chapters on miss.
// Phase 2.C: the component renders the full <LessonLayout /> from the legacy
// app, including the chapter body + "Try it now" demo. The chapter is resolved
// inside beforeLoad and surfaced through `Route.useRouteContext()`.

import { createFileRoute, redirect, useNavigate } from '@tanstack/react-router';
import { useEffect } from 'react';

import { Loading } from '../components/loading/Loading';
import { LessonLayout } from '../learn/LessonLayout';
import { triggerLocalPicker } from '../lib/local-model-picker';
import { findChapter } from '../learn/chapters';
import { AttentionChapterBody, AttentionDemo } from '../learn/chapters/03-attention';
import { MultiheadGqaChapterBody, MultiheadGqaDemo } from '../learn/chapters/04-multihead-gqa';
import { TokenizationChapterBody, TokenizerDemo } from '../learn/chapters/01-tokenization';
import { EmbeddingsChapterBody, EmbeddingsDemo } from '../learn/chapters/02-embeddings';
import { RopeChapterBody, RopeDemo } from '../learn/chapters/05-rope';
import { RmsNormChapterBody, RmsNormDemo } from '../learn/chapters/06-rms-norm';
import { MlpChapterBody, MlpDemo } from '../learn/chapters/07-mlp';
import { FullBlockChapterBody, FullBlockDemo } from '../learn/chapters/08-full-block';
import { SamplingChapterBody, SamplingDemo } from '../learn/chapters/09-sampling';
import { KvCacheChapterBody, KvCacheDemo } from '../learn/chapters/10-kv-cache';
import { useFreeChat } from '../providers/free-chat';
import { useModelLoader } from '../providers/model-loader';

function ChapterRouteComponent() {
  const { chapter } = Route.useRouteContext();
  const navigate = useNavigate();
  const { mlxWorkerRef, inspectorAbortRef } = useFreeChat();
  const { status, loadingText, loadingProgress, hostedModelAvailable, kickoffLoad } =
    useModelLoader();

  // Auto-kickoff the model load on direct URL landings (bookmark, hard reload
  // of /chapters/<id>). Skipped when no hosted model is available since that
  // flow requires the user to pick a local model directory. kickoffLoad is
  // idempotent at the App level.
  useEffect(() => {
    if (status === 'ready') return;
    if (hostedModelAvailable === false) return;
    kickoffLoad();
  }, [status, hostedModelAvailable, kickoffLoad]);

  if (status !== 'ready') {
    return <Loading status={loadingText || null} progress={loadingProgress} />;
  }

  return (
    <LessonLayout
      current={chapter}
      onOpenChapter={(chapterId) => {
        void navigate({ to: '/chapters/$chapterId', params: { chapterId } });
      }}
      onBackToIndex={() => {
        void navigate({ to: '/chapters' });
      }}
      onOpenFreeChat={() => {
        if (hostedModelAvailable === false) {
          triggerLocalPicker();
          return;
        }
        kickoffLoad();
        void navigate({ to: '/chat' });
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
        ) : chapter.id === 'sampling' ? (
          <SamplingDemo workerRef={mlxWorkerRef} abortRef={inspectorAbortRef} />
        ) : chapter.id === 'kv-cache' ? (
          <KvCacheDemo workerRef={mlxWorkerRef} abortRef={inspectorAbortRef} />
        ) : null
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
      ) : chapter.id === 'sampling' ? (
        <SamplingChapterBody />
      ) : chapter.id === 'kv-cache' ? (
        <KvCacheChapterBody />
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
