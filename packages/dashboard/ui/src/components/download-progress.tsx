import { useEffect, useRef, useState, useSyncExternalStore, type CSSProperties, type ReactElement } from 'react';

import type { DownloadEvent } from '../../../src/download.js';
import { subscribeDownload } from '../lib/api';
import { cn } from '../lib/cn';
import { getConnectionGeneration, subscribeConnection } from '../lib/connection';
import { formatBytes } from '../lib/format';
import { LiveLabel } from './arc-x/live-label/live-label';
import { MorphLoader } from './arc-x/morph-loader/morph-loader';
import type { LoaderStatus } from './arc-x/morph-loader/shapes';
import { useStaticMotion } from './arc-x/motion/use-static-motion';
import { SwapText } from './arc-x/swap-text/swap-text';
import { Progress } from './arc/progress/progress';
import styles from './download-progress.module.css';

export type ProgressPhase = 'connecting' | 'running' | 'done' | 'error';

export interface RateState {
  /** performance.now() of the anchor frame. */
  at: number;
  bytes: number;
  bytesPerSec: number | null;
}

export interface ProgressState {
  phase: ProgressPhase;
  fileCount: number;
  fileIndex: number;
  currentFile: string;
  /** Aggregate bytes across all files, as reported by the latest frame. */
  receivedBytes: number;
  /** Job total from the `start` event; 0 until it arrives. */
  totalBytes: number;
  message?: string;
  /** null until a progress frame anchors it; reset by `start`. */
  rate: RateState | null;
  /** No start / progress event for STALL_MS while connecting or running. */
  stalled: boolean;
}

export const INITIAL_PROGRESS: ProgressState = {
  phase: 'connecting',
  fileCount: 0,
  fileIndex: 0,
  currentFile: '',
  receivedBytes: 0,
  totalBytes: 0,
  rate: null,
  stalled: false,
};

/** Without a start or progress event for this long the bar stops saying it is moving. */
export const STALL_MS = 5000;
const RATE_MIN_DT_MS = 1000;
const RATE_WEIGHT = 0.3;

export type ProgressEvent = DownloadEvent | { type: 'stalled' };

function nextRate(prev: RateState | null, bytes: number, now: number): RateState {
  if (prev === null) return { at: now, bytes, bytesPerSec: null };
  const dt = now - prev.at;
  if (dt < RATE_MIN_DT_MS) return prev;
  const inst = ((bytes - prev.bytes) / dt) * 1000;
  if (inst < 0) return { at: now, bytes, bytesPerSec: prev.bytesPerSec };
  const bytesPerSec = prev.bytesPerSec === null ? inst : RATE_WEIGHT * inst + (1 - RATE_WEIGHT) * prev.bytesPerSec;
  return { at: now, bytes, bytesPerSec };
}

/** The card's state machine. `now` is performance.now() of the event; only the transfer rate reads it. */
export function reduceProgress(
  prev: ProgressState,
  event: ProgressEvent,
  now: number = performance.now(),
): ProgressState {
  switch (event.type) {
    case 'start':
      return {
        ...prev,
        phase: 'running',
        fileCount: event.fileCount,
        totalBytes: event.totalBytes,
        rate: null,
        stalled: false,
      };
    case 'progress': {
      const terminal = prev.phase === 'done' || prev.phase === 'error';
      return {
        ...prev,
        phase: terminal ? prev.phase : 'running',
        fileCount: event.fileCount > 0 ? event.fileCount : prev.fileCount,
        fileIndex: event.fileIndex,
        currentFile: event.file,
        receivedBytes: event.jobReceivedBytes,
        rate: nextRate(prev.rate, event.jobReceivedBytes, now),
        stalled: false,
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
      // cancel came from somewhere else, which is why the event must also
      // call back below rather than only render.
      return { ...prev, phase: 'error', message: 'Cancelled' };
    case 'stalled':
      return prev.phase === 'connecting' || prev.phase === 'running' ? { ...prev, stalled: true } : prev;
  }
}

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
  className?: string;
}

function timeLeft(state: ProgressState): string | null {
  const rate = state.rate?.bytesPerSec ?? null;
  if (rate === null || rate <= 0 || state.totalBytes <= 0) return null;
  const seconds = Math.max(0, state.totalBytes - state.receivedBytes) / rate;
  if (seconds < 60) return 'under a minute left';
  return `about ${Math.round(seconds / 60)} min left`;
}

const LOADER: Record<ProgressPhase, LoaderStatus> = {
  connecting: 'loading',
  running: 'loading',
  done: 'success',
  error: 'error',
};

/**
 * Live progress for one catalog download, driven by a port subscription on the
 * job id. The bar reads each `progress` event's `jobReceivedBytes` against the
 * job total from `start`, so it reflects whole-job completion.
 *
 * Deliberately NOT summed from the per-file events this card witnessed: the host
 * replays only `start` plus the single latest event, so a card mounted
 * mid-download (a page reload, or navigating back to Models) has never seen the
 * finished files and would render the current file's bytes as the job's —
 * short by every completed shard.
 */
export function DownloadProgress({ id, onDone, onError, onCancelled, className }: DownloadProgressProps): ReactElement {
  const [state, setState] = useState<ProgressState>(INITIAL_PROGRESS);
  const connection = useSyncExternalStore(subscribeConnection, getConnectionGeneration, getConnectionGeneration);
  const doneCb = useRef(onDone);
  const errorCb = useRef(onError);
  const cancelledCb = useRef(onCancelled);
  doneCb.current = onDone;
  errorCb.current = onError;
  cancelledCb.current = onCancelled;

  useEffect(() => {
    setState(INITIAL_PROGRESS);
    let stallTimer: ReturnType<typeof setTimeout> | undefined;
    const armStall = (): void => {
      clearTimeout(stallTimer);
      stallTimer = setTimeout(() => setState((prev) => reduceProgress(prev, { type: 'stalled' })), STALL_MS);
    };
    armStall();
    const sub = subscribeDownload(id, (event) => {
      const now = performance.now();
      setState((prev) => reduceProgress(prev, event, now));
      if (event.type === 'start' || event.type === 'progress') armStall();
      else clearTimeout(stallTimer);
      if (event.type === 'done') doneCb.current?.();
      if (event.type === 'error') errorCb.current?.(event.message);
      // A terminal event the owner cannot have been waiting for. The host answers
      // a cancel while the job is still `running` — it lets `processJob` emit this
      // event as it unwinds — so a cancel from a second tab, or from `shutdown()`,
      // lands on a card that never clicked Cancel. Without this the card renders
      // "Cancelled" beside a live Cancel button forever: the page keeps the job in
      // `active`, and nothing polls to correct it.
      if (event.type === 'cancelled') cancelledCb.current?.();
    });
    return () => {
      clearTimeout(stallTimer);
      sub.close();
    };
    // `connection` rebinds the subscription after CONTROL PANEL restarts. The old
    // client's `close()` drops its listener immediately while keeping the port
    // only long enough to drain in-flight calls, so without this the card is
    // permanently deaf — frozen mid-progress against a healthy new runtime.
  }, [id, connection]);

  return <DownloadProgressView state={state} className={className} />;
}

/** The card for one state of {@link reduceProgress}: status line, bar, and file / rate line. */
export function DownloadProgressView({ state, className }: { state: ProgressState; className?: string }): ReactElement {
  const staticMotion = useStaticMotion();
  const { phase, totalBytes, receivedBytes, fileCount, fileIndex, currentFile, message, stalled } = state;
  const fraction =
    phase === 'done'
      ? 1
      : totalBytes > 0
        ? Math.min(1, receivedBytes / totalBytes)
        : fileCount > 0
          ? Math.min(1, fileIndex / fileCount)
          : 0;
  const pct = Math.round(fraction * 100);
  const moving = (phase === 'connecting' || phase === 'running') && !stalled;
  const shine = phase === 'running' && !stalled && !staticMotion;

  const status =
    phase === 'done'
      ? 'Complete'
      : phase === 'error'
        ? (message ?? 'Download failed')
        : phase === 'connecting'
          ? 'Starting…'
          : currentFile || 'Downloading…';
  const bytes =
    phase === 'error' || phase === 'connecting'
      ? null
      : totalBytes > 0
        ? `${formatBytes(receivedBytes)} / ${formatBytes(totalBytes)}`
        : `${pct}%`;
  const rate = state.rate?.bytesPerSec ?? null;
  const details =
    phase === 'running' && fileCount > 0
      ? [
          `File ${Math.min(fileIndex + 1, fileCount)} of ${fileCount}`,
          rate !== null ? `${formatBytes(rate)}/s` : null,
          timeLeft(state),
        ]
          .filter((part): part is string => part !== null)
          .join(' · ')
      : null;

  return (
    <div className={cn('grid min-w-0 gap-1.5', className)} role="group" aria-label="Download progress">
      <div className="flex min-w-0 items-center justify-between gap-2 text-body">
        <span className="flex min-w-0 items-center gap-1.5">
          <MorphLoader
            size={14}
            status={LOADER[phase]}
            spinning={moving}
            className={phase === 'connecting' || phase === 'running' ? 'text-muted' : undefined}
          />
          {phase === 'connecting' || phase === 'running' ? (
            <LiveLabel active={moving} className="min-w-0 truncate">
              {status}
            </LiveLabel>
          ) : (
            <span className={cn('min-w-0 truncate', phase === 'error' && 'text-danger')}>{status}</span>
          )}
        </span>
        <span className="shrink-0 text-secondary tabular-nums">
          {bytes !== null ? <SwapText text={bytes} granularity="char" /> : null}
        </span>
      </div>
      <div
        className={styles.bar}
        data-shine={shine ? '' : undefined}
        data-phase={phase}
        style={{ '--pct': `${pct}%` } as CSSProperties}
      >
        <Progress value={pct} label="Download progress" />
      </div>
      {details !== null ? (
        <p className="m-0 text-caption text-secondary tabular-nums" aria-hidden="true">
          <SwapText text={details} granularity="char" />
        </p>
      ) : null}
    </div>
  );
}
