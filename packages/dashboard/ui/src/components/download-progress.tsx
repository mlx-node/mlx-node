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
      /** Bytes of the named file; `jobReceivedBytes` covers the whole job. */
      receivedBytes: number;
      jobReceivedBytes: number;
      /** Size of the named file; the job total travels on `start`. */
      totalBytes: number;
      fileIndex: number;
      fileCount: number;
    }
  | { type: 'done'; id: string; outputDir: string }
  | { type: 'error'; id: string; message: string }
  | { type: 'cancelled'; id: string };

type Phase = 'connecting' | 'running' | 'done' | 'error';

interface ProgressState {
  phase: Phase;
  fileCount: number;
  fileIndex: number;
  currentFile: string;
  /** Aggregate bytes across all files, as reported by the latest frame. */
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
  /**
   * The job stopped without finishing. Distinct from {@link onError}: a cancel is
   * not a failure and must not raise a red notice. Needed because a `cancelled`
   * frame does NOT imply this card asked for it — a second tab, or `shutdown()`
   * aborting the in-flight job, sends one to a card that clicked nothing.
   */
  onCancelled?: () => void;
}

/**
 * Live progress for one catalog download, driven by the
 * `GET /api/downloads/:id/events` SSE stream. The bar reads each `progress`
 * frame's `jobReceivedBytes` against the job total from `start`, so it reflects
 * whole-job completion.
 *
 * Deliberately NOT summed from the per-file frames this card witnessed: the
 * server replays only `start` plus the single latest event, so a card mounted
 * mid-download (a page reload, or navigating back to Models) has never seen the
 * finished files and would render the current file's bytes as the job's —
 * short by every completed shard.
 */
export function DownloadProgress({ id, onDone, onError, onCancelled }: DownloadProgressProps) {
  const [state, setState] = useState<ProgressState>(INITIAL);
  const doneCb = useRef(onDone);
  const errorCb = useRef(onError);
  const cancelledCb = useRef(onCancelled);
  doneCb.current = onDone;
  errorCb.current = onError;
  cancelledCb.current = onCancelled;

  useEffect(() => {
    setState(INITIAL);
    const sub = subscribeSSE<DownloadEvent>(
      `/downloads/${id}/events`,
      (event) => {
        setState((prev) => {
          switch (event.type) {
            case 'start':
              return { ...prev, phase: 'running', fileCount: event.fileCount, totalBytes: event.totalBytes };
            case 'progress': {
              const terminal = prev.phase === 'done' || prev.phase === 'error';
              return {
                ...prev,
                phase: terminal ? prev.phase : 'running',
                fileCount: event.fileCount > 0 ? event.fileCount : prev.fileCount,
                fileIndex: event.fileIndex,
                currentFile: event.file,
                receivedBytes: event.jobReceivedBytes,
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
            case 'cancelled':
              // Settle on a non-spinning "stopped" state rather than a false
              // "Complete". Usually invisible: the owner clears this card on the
              // DELETE response and it unmounts. It IS what a card sees when the
              // cancel came from somewhere else, which is why the frame must also
              // call back below rather than only render.
              return { ...prev, phase: 'error', message: 'Cancelled' };
          }
        });
        if (event.type === 'done') doneCb.current?.();
        if (event.type === 'error') errorCb.current?.(event.message);
        // A terminal frame the owner cannot have been waiting for. The server
        // answers a cancel while the job is still `running` — it lets `processJob`
        // emit this frame as it unwinds — so a cancel from a second tab, or from
        // `shutdown()`, lands on a card that never clicked Cancel. Without this
        // the card renders "Cancelled" beside a live Cancel button forever: the
        // page keeps the job in `active`, and nothing polls to correct it.
        if (event.type === 'cancelled') cancelledCb.current?.();
      },
      undefined,
      ['start', 'progress', 'done', 'error', 'cancelled'],
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
