import { subscribeSSE } from '@/lib/api';
import { formatBytes } from '@/lib/format';
import { cn } from '@/lib/utils';
import { AlertCircle, CheckCircle2, Loader2 } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

/** Mirrors `DownloadEvent` from `packages/dashboard/src/download.ts`. */
type DownloadEvent =
  | { type: 'start'; id: string; repo: string; totalBytes: number; fileCount: number }
  | {
      type: 'progress';
      id: string;
      file: string;
      receivedBytes: number;
      totalBytes: number;
      fileIndex: number;
      fileCount: number;
    }
  | { type: 'done'; id: string; outputDir: string }
  | { type: 'error'; id: string; message: string };

type Phase = 'connecting' | 'running' | 'done' | 'error';

interface ProgressState {
  phase: Phase;
  fileCount: number;
  fileIndex: number;
  currentFile: string;
  /** Aggregate bytes across all files (summed per file index). */
  receivedBytes: number;
  /** Job total from the `start` event; 0 until it arrives. */
  totalBytes: number;
  message?: string;
}

const INITIAL: ProgressState = {
  phase: 'connecting',
  fileCount: 0,
  fileIndex: 0,
  currentFile: '',
  receivedBytes: 0,
  totalBytes: 0,
};

export interface DownloadProgressProps {
  /** Download job id from `POST /api/downloads`. */
  id: string;
  onDone?: () => void;
  onError?: (message: string) => void;
}

/**
 * Live progress for one catalog download, driven by the
 * `GET /api/downloads/:id/events` SSE stream. Aggregate byte progress is summed
 * from the per-file `progress` frames (the stream carries per-file bytes plus a
 * job total on `start`), so the bar reflects whole-job completion.
 */
export function DownloadProgress({ id, onDone, onError }: DownloadProgressProps) {
  const [state, setState] = useState<ProgressState>(INITIAL);
  const perFile = useRef<Map<number, number>>(new Map());
  const doneCb = useRef(onDone);
  const errorCb = useRef(onError);
  doneCb.current = onDone;
  errorCb.current = onError;

  useEffect(() => {
    perFile.current = new Map();
    setState(INITIAL);
    const sub = subscribeSSE<DownloadEvent>(
      `/downloads/${id}/events`,
      (event) => {
        setState((prev) => {
          switch (event.type) {
            case 'start':
              return { ...prev, phase: 'running', fileCount: event.fileCount, totalBytes: event.totalBytes };
            case 'progress': {
              perFile.current.set(event.fileIndex, event.receivedBytes);
              let received = 0;
              for (const value of perFile.current.values()) received += value;
              const terminal = prev.phase === 'done' || prev.phase === 'error';
              return {
                ...prev,
                phase: terminal ? prev.phase : 'running',
                fileCount: event.fileCount > 0 ? event.fileCount : prev.fileCount,
                fileIndex: event.fileIndex,
                currentFile: event.file,
                receivedBytes: received,
              };
            }
            case 'done':
              return {
                ...prev,
                phase: 'done',
                receivedBytes: prev.totalBytes > 0 ? prev.totalBytes : prev.receivedBytes,
              };
            case 'error':
              return { ...prev, phase: 'error', message: event.message };
          }
        });
        if (event.type === 'done') doneCb.current?.();
        if (event.type === 'error') errorCb.current?.(event.message);
      },
      undefined,
      ['start', 'progress', 'done', 'error'],
    );
    return () => sub.close();
  }, [id]);

  const { phase, totalBytes, receivedBytes, fileCount, fileIndex, currentFile, message } = state;
  const fraction =
    phase === 'done'
      ? 1
      : totalBytes > 0
        ? Math.min(1, receivedBytes / totalBytes)
        : fileCount > 0
          ? Math.min(1, fileIndex / fileCount)
          : 0;
  const pct = Math.round(fraction * 100);

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between gap-2 text-sm">
        <span className="flex min-w-0 items-center gap-1.5">
          {phase === 'done' ? (
            <CheckCircle2 className="text-foreground size-4 shrink-0" aria-hidden />
          ) : phase === 'error' ? (
            <AlertCircle className="text-destructive size-4 shrink-0" aria-hidden />
          ) : (
            <Loader2 className="text-muted-foreground size-4 shrink-0 animate-spin" aria-hidden />
          )}
          <span className={cn('truncate', phase === 'error' && 'text-destructive')}>
            {phase === 'done'
              ? 'Complete'
              : phase === 'error'
                ? (message ?? 'Download failed')
                : phase === 'connecting'
                  ? 'Starting…'
                  : currentFile || 'Downloading…'}
          </span>
        </span>
        <span className="text-muted-foreground shrink-0 tabular-nums">
          {phase === 'error'
            ? ''
            : totalBytes > 0
              ? `${formatBytes(receivedBytes)} / ${formatBytes(totalBytes)}`
              : `${pct}%`}
        </span>
      </div>
      <div className="bg-secondary h-2 w-full overflow-hidden rounded-full" role="progressbar" aria-valuenow={pct}>
        <div
          className={cn(
            'h-full rounded-full transition-[width] duration-300',
            phase === 'error' ? 'bg-destructive' : 'bg-primary',
          )}
          style={{ width: `${phase === 'error' ? 100 : pct}%` }}
        />
      </div>
      {phase === 'running' && fileCount > 0 && (
        <p className="text-muted-foreground text-xs tabular-nums">
          File {Math.min(fileIndex + 1, fileCount)} of {fileCount}
        </p>
      )}
    </div>
  );
}
