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
  /** The latest frame hashes a file already in the folder (Update) instead of downloading it. */
  checking: boolean;
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
  checking: false,
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
        checking: event.phase === 'check',
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

export interface DownloadProgressProps extends DownloadProgressCallbacks {
  /** Download job id from `POST /api/downloads`. */
  id: string;
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

export interface DownloadProgressCallbacks {
  /** `keptDir`: where a later update put the files of the old folder that the app may not delete. */
  onDone?: (keptDir: string | undefined) => void;
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
 * Live progress for one catalog download, driven by a port subscription on the
 * job id; no subscription while `id` is undefined. The bar reads each `progress`
 * event's `jobReceivedBytes` against the job total from `start`, so it reflects
 * whole-job completion.
 *
 * Deliberately NOT summed from the per-file events this card witnessed: the host
 * replays only `start` plus the single latest event, so a card mounted
 * mid-download (a page reload, or navigating back to Models) has never seen the
 * finished files and would render the current file's bytes as the job's —
 * short by every completed shard.
 */
export function useDownloadProgress(id: string | undefined, callbacks: DownloadProgressCallbacks): ProgressState {
  const [state, setState] = useState<ProgressState>(INITIAL_PROGRESS);
  const connection = useSyncExternalStore(subscribeConnection, getConnectionGeneration, getConnectionGeneration);
  const doneCb = useRef(callbacks.onDone);
  const errorCb = useRef(callbacks.onError);
  const cancelledCb = useRef(callbacks.onCancelled);
  doneCb.current = callbacks.onDone;
  errorCb.current = callbacks.onError;
  cancelledCb.current = callbacks.onCancelled;

  useEffect(() => {
    setState(INITIAL_PROGRESS);
    if (id === undefined) return;
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
      if (event.type === 'done') doneCb.current?.(event.keptDir);
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

  return state;
}

export function DownloadProgress({ id, onDone, onError, onCancelled, className }: DownloadProgressProps): ReactElement {
  const state = useDownloadProgress(id, { onDone, onError, onCancelled });
  return <DownloadProgressView state={state} className={className} />;
}

/** The share of the job that is done, 0-100. */
export function progressPercent(state: ProgressState): number {
  const { phase, totalBytes, receivedBytes, fileCount, fileIndex } = state;
  const fraction =
    phase === 'done'
      ? 1
      : totalBytes > 0
        ? Math.min(1, receivedBytes / totalBytes)
        : fileCount > 0
          ? Math.min(1, fileIndex / fileCount)
          : 0;
  return Math.round(fraction * 100);
}

const DECIMAL_UNITS = ['B', 'kB', 'MB', 'GB', 'TB'] as const;

/**
 * Bytes in powers of 1000, the units of the catalog's sizes: a card that says `16.6 GB` counts its download up to
 * `16.6 GB`. {@link formatBytes} counts in powers of 1024 and would end at `15.4 GB`.
 */
export function decimalBytesParts(bytes: number): { value: string; unit: string } {
  if (!Number.isFinite(bytes) || bytes <= 0) return { value: '0', unit: 'B' };
  let exp = 0;
  let value = bytes;
  while (value >= 1000 && exp < DECIMAL_UNITS.length - 1) {
    value /= 1000;
    exp += 1;
  }
  return { value: value.toFixed(exp === 0 || value >= 100 ? 0 : 1), unit: DECIMAL_UNITS[exp]! };
}

function decimalBytes(bytes: number): string {
  const parts = decimalBytesParts(bytes);
  return `${parts.value} ${parts.unit}`;
}

/** `7.4 / 16.6 GB`, or `812 MB / 16.6 GB` when the units differ. */
export function bytesOfTotal(received: number, total: number): string {
  const done = decimalBytesParts(received);
  const all = decimalBytesParts(total);
  return done.unit === all.unit
    ? `${done.value} / ${all.value} ${all.unit}`
    : `${done.value} ${done.unit} / ${all.value} ${all.unit}`;
}

/** The one line a catalog card shows about a job: its step, or bytes, rate and time left. */
export function progressSummary(state: ProgressState, job: { committing: boolean; cancelling: boolean }): string {
  if (job.cancelling) return 'Cancelling…';
  if (job.committing || state.phase === 'done') return 'Finishing…';
  if (state.phase === 'error') return state.message ?? 'Download failed';
  if (state.phase === 'connecting' || state.totalBytes <= 0) return 'Starting…';
  const bytes = bytesOfTotal(state.receivedBytes, state.totalBytes);
  if (state.checking) return `Checking files · ${bytes}`;
  const rate = state.rate?.bytesPerSec ?? null;
  return [bytes, rate !== null ? `${decimalBytes(rate)}/s` : null, timeLeft(state)]
    .filter((part): part is string => part !== null)
    .join(' · ');
}

/** The file the job is on, as `{file} · File i of n`, or null before the first file. */
export function progressFileLine(state: ProgressState): string | null {
  if (state.currentFile === '' || state.fileCount <= 0) return null;
  return `${state.currentFile} · File ${Math.min(state.fileIndex + 1, state.fileCount)} of ${state.fileCount}`;
}

/** The bar alone, with the moving shine while bytes arrive. */
export function DownloadBar({ state, label }: { state: ProgressState; label: string }): ReactElement {
  const staticMotion = useStaticMotion();
  const pct = progressPercent(state);
  const shine = state.phase === 'running' && !state.stalled && !staticMotion;
  return (
    <div
      className={styles.bar}
      data-shine={shine ? '' : undefined}
      data-phase={state.phase}
      style={{ '--pct': `${pct}%` } as CSSProperties}
    >
      <Progress value={pct} label={label} />
    </div>
  );
}

/** The card for one state of {@link reduceProgress}: status line, bar, and file / rate line. */
export function DownloadProgressView({ state, className }: { state: ProgressState; className?: string }): ReactElement {
  const { phase, totalBytes, receivedBytes, fileCount, fileIndex, currentFile, checking, message, stalled } = state;
  const pct = progressPercent(state);
  const moving = (phase === 'connecting' || phase === 'running') && !stalled;

  const status =
    phase === 'done'
      ? 'Complete'
      : phase === 'error'
        ? (message ?? 'Download failed')
        : phase === 'connecting'
          ? 'Starting…'
          : checking
            ? `Checking ${currentFile}`
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
      <DownloadBar state={state} label="Download progress" />
      {details !== null ? (
        <p className="m-0 text-caption text-secondary tabular-nums" aria-hidden="true">
          <SwapText text={details} granularity="char" />
        </p>
      ) : null}
    </div>
  );
}
