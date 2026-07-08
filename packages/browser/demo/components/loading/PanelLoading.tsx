// PanelLoading — compact, in-column loading affordance for the "Try it now"
// panel. Unlike <Loading>, this does NOT use the full-screen `overlay-screen`
// wrapper: the chapter body renders immediately on the left and only the demo
// panel shows this while its resource (the model, or the WebGPU device) warms
// up in the background.

import { cn } from '@/lib/utils';
import { Loader2Icon } from 'lucide-react';

import { formatBytes, formatFileName } from '../../lib/display-helpers';
import { useLocale } from '../../lib/i18n-react';
import { Button } from '../ui/button';
import type { LoadingProgress } from './Loading';

// Per-locale chrome strings (the dynamic `status` text comes from the loader
// pipeline and is shown as-is; only this component's own copy localizes).
const COPY = {
  en: {
    needsModel: 'This live demo needs the model.',
    loadModel: 'Load a model',
    initializing: 'Initializing model…',
    progressAria: (pct: number) => `Loading progress ${pct}%`,
    readyOnLeft: 'The lesson is ready to read on the left — this panel will light up when the model finishes.',
  },
  zh: {
    needsModel: '这个在线演示需要加载模型。',
    loadModel: '加载模型',
    initializing: '正在初始化模型……',
    progressAria: (pct: number) => `加载进度 ${pct}%`,
    readyOnLeft: '左侧的课文已经可以阅读——模型就绪后，这个面板会自动亮起。',
  },
} as const;

export type PanelLoadingProps = {
  status: string | null;
  progress?: LoadingProgress | null;
  /**
   * When true, this demo needs the model but no hosted model is available, so
   * instead of a spinner we surface a "load a model" affordance wired to
   * `onLoadLocal`.
   */
  hostedUnavailable?: boolean;
  onLoadLocal?: () => void;
};

export function PanelLoading({ status, progress, hostedUnavailable, onLoadLocal }: PanelLoadingProps) {
  const copy = COPY[useLocale()];
  if (hostedUnavailable) {
    return (
      <div className="flex flex-col items-start gap-3 rounded-lg border border-border bg-card/40 p-4">
        <p className="text-sm text-muted-foreground">{copy.needsModel}</p>
        <Button variant="outline" size="sm" onClick={onLoadLocal} disabled={!onLoadLocal}>
          {copy.loadModel}
        </Button>
      </div>
    );
  }

  const pct = progress ? Math.round(progress.pct) : null;
  const fileName = formatFileName(progress?.file);
  const bytes =
    progress?.loadedBytes != null && progress.totalBytes != null
      ? `${formatBytes(progress.loadedBytes)} / ${formatBytes(progress.totalBytes)}`
      : null;
  const meta = [fileName, bytes, progress?.cacheSource].filter(Boolean).join(' · ');

  return (
    <div className="flex flex-col items-start gap-3 rounded-lg border border-border bg-card/40 p-4">
      <div className="flex items-center gap-2">
        <Loader2Icon className={cn('size-4 animate-spin text-primary')} aria-hidden="true" />
        <span className="loader-text">{status && status.trim().length > 0 ? status : copy.initializing}</span>
      </div>
      {progress && (
        <div className="loader-progress w-full" aria-label={copy.progressAria(pct ?? 0)}>
          <div className="loader-progress-track">
            <div className="loader-progress-fill" style={{ width: `${progress.pct}%` }} />
          </div>
          <div className="loader-progress-meta">
            <span>{pct}%</span>
            {meta && <span>{meta}</span>}
          </div>
        </div>
      )}
      <p className="text-xs leading-relaxed text-muted-foreground">{copy.readyOnLeft}</p>
    </div>
  );
}
