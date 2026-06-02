// PanelLoading — compact, in-column loading affordance for the "Try it now"
// panel. Unlike <Loading>, this does NOT use the full-screen `overlay-screen`
// wrapper: the chapter body renders immediately on the left and only the demo
// panel shows this while its resource (the model, or the WebGPU device) warms
// up in the background.

import { cn } from '@/lib/utils';
import { Loader2Icon } from 'lucide-react';

import { formatBytes, formatFileName, formatLoadingText } from '../../lib/display-helpers';
import { Button } from '../ui/button';
import type { LoadingProgress } from './Loading';

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
  if (hostedUnavailable) {
    return (
      <div className="flex flex-col items-start gap-3 rounded-lg border border-border bg-card/40 p-4">
        <p className="text-sm text-muted-foreground">This live demo needs the model.</p>
        <Button variant="outline" size="sm" onClick={onLoadLocal} disabled={!onLoadLocal}>
          Load a model
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
        <span className="loader-text">{formatLoadingText(status)}</span>
      </div>
      {progress && (
        <div className="loader-progress w-full" aria-label={`Loading progress ${pct}%`}>
          <div className="loader-progress-track">
            <div className="loader-progress-fill" style={{ width: `${progress.pct}%` }} />
          </div>
          <div className="loader-progress-meta">
            <span>{pct}%</span>
            {meta && <span>{meta}</span>}
          </div>
        </div>
      )}
      <p className="text-xs leading-relaxed text-muted-foreground">
        The lesson is ready to read on the left — this panel will light up when the model finishes.
      </p>
    </div>
  );
}
