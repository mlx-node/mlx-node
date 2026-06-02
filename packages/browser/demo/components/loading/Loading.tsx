import { formatBytes, formatFileName, formatLoadingText } from '../../lib/display-helpers';

export type LoadingProgress = {
  pct: number;
  loadedBytes?: number;
  totalBytes?: number;
  file?: string;
  cacheSource?: string;
};

export type LoadingProps = {
  status: string | null;
  progress?: LoadingProgress | null;
};

export function Loading({ status, progress }: LoadingProps) {
  const pct = progress ? Math.round(progress.pct) : null;
  const fileName = formatFileName(progress?.file);
  const bytes =
    progress?.loadedBytes != null && progress.totalBytes != null
      ? `${formatBytes(progress.loadedBytes)} / ${formatBytes(progress.totalBytes)}`
      : null;
  const meta = [fileName, bytes, progress?.cacheSource].filter(Boolean).join(' · ');

  return (
    <div className="overlay-screen" style={{ background: 'var(--bg)' }}>
      <div className="loading-stack">
        <div className="loader-ring" aria-hidden="true" />
        <div>
          <div className="loader-text">{formatLoadingText(status)}</div>
          {progress && (
            <div className="loader-progress" aria-label={`Model loading progress ${pct}%`}>
              <div className="loader-progress-track">
                <div className="loader-progress-fill" style={{ width: `${progress.pct}%` }} />
              </div>
              <div className="loader-progress-meta">
                <span>{pct}%</span>
                {meta && <span>{meta}</span>}
              </div>
            </div>
          )}
          <div className="loader-sub">Model weights are cached for future visits.</div>
        </div>
      </div>
    </div>
  );
}
