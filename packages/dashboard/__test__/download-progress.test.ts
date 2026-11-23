/**
 * @vitest-environment happy-dom
 */

/**
 * Render guards for the download progress card.
 *
 * The card is the only consumer of the download event stream, and the rule that
 * matters — what the byte counter reads when a client attaches PART-WAY through
 * a sharded model — lives in its reducer, not in a helper. The runtime replay is
 * one `start` event plus the single latest one, so a card that derives the job
 * total by summing the per-file events it happened to witness renders a
 * mount-order-dependent number: correct for a card that watched from the first
 * byte, short by every already-finished file for one mounted after a reload.
 * These tests drive the component through a stubbed download subscription over
 * the REAL port, with exactly the events `DownloadManager` emits (see
 * `download.test.ts`), and assert on the rendered text — so that difference is a
 * red test rather than a bar that silently lies by gigabytes.
 *
 * `happy-dom` is scoped to this file by the docblock above.
 */

import {
  bytesOfTotal,
  DownloadProgress,
  INITIAL_PROGRESS,
  progressFileLine,
  progressPercent,
  progressSummary,
  reduceProgress,
  timeLeft,
  timeLeftText,
  type ProgressState,
} from '@/components/download-progress';
import { act, createElement } from 'react';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import type { DownloadEvent } from '../src/download.js';
import { renderPage, stubApi, type RenderedPage } from './render.js';

const MIB = 1024 * 1024;

/**
 * One card's live subscription, as the runtime sees it. `deliver` pushes an event
 * through the real port the SPA client is attached to, and `closed` records the
 * unsubscribe the card issues when it unmounts.
 */
interface FakeStream {
  jobId: string;
  closed: boolean;
  deliver(event: DownloadEvent): void;
}

let streams: FakeStream[] = [];
let mounted: RenderedPage | undefined;
let restoreApi: (() => void) | undefined;

/**
 * Let a port hop — and the render it causes — land. Deliberately macrotask-based:
 * an event crosses a real `MessageChannel`, so nothing here settles on the
 * microtask queue.
 */
async function settle(): Promise<void> {
  for (let i = 0; i < 5; i++) {
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }
}

afterEach(() => {
  mounted?.unmount();
  mounted = undefined;
  restoreApi?.();
  restoreApi = undefined;
  streams = [];
});

const startEvent: DownloadEvent = {
  type: 'start',
  id: 'job-1',
  repo: 'org/sharded-model',
  totalBytes: 7 * MIB,
  fileCount: 3,
};

/** The stream a card that watched from the first byte sees for files 1-2. */
const settled: DownloadEvent[] = [
  {
    type: 'progress',
    id: 'job-1',
    file: 'config.json',
    receivedBytes: MIB,
    jobReceivedBytes: MIB,
    totalBytes: MIB,
    fileIndex: 0,
    fileCount: 3,
  },
  {
    type: 'progress',
    id: 'job-1',
    file: 'model-00001-of-00002.safetensors',
    receivedBytes: 2 * MIB,
    jobReceivedBytes: 3 * MIB,
    totalBytes: 2 * MIB,
    fileIndex: 1,
    fileCount: 3,
  },
];

/**
 * The single latest event at the moment of the reload: the third file is 1 MiB
 * into its 4 MiB, so the JOB holds 4 MiB of 7 MiB. Taken from the manager
 * repro in `download.test.ts`.
 */
const latestEvent: DownloadEvent = {
  type: 'progress',
  id: 'job-1',
  file: 'model-00002-of-00002.safetensors',
  receivedBytes: MIB,
  jobReceivedBytes: 4 * MIB,
  totalBytes: 4 * MIB,
  fileIndex: 2,
  fileCount: 3,
};

/** Mount the card and return its open subscription, settled on the connecting state. */
async function mountCard(): Promise<FakeStream> {
  restoreApi = stubApi(
    {},
    {
      subscribe: (jobId, listener) => {
        const stream: FakeStream = {
          jobId,
          closed: false,
          deliver: (event) => listener(event),
        };
        streams.push(stream);
        return () => {
          stream.closed = true;
        };
      },
    },
  );
  mounted = await renderPage(createElement(DownloadProgress, { id: 'job-1' }), (text) => text.includes('Starting'));
  // The subscribe crosses the port, so the runtime has not been asked yet when
  // the connecting state first renders.
  await settle();
  expect(streams).toHaveLength(1);
  return streams[0]!;
}

/** Deliver events and let React flush the state they produce. */
async function deliver(source: FakeStream, ...events: DownloadEvent[]): Promise<void> {
  for (const event of events) source.deliver(event);
  await settle();
}

describe('DownloadProgress — aggregate bytes across a sharded job', () => {
  it('renders the whole-job total from the two events a mid-job subscriber is replayed', async () => {
    const source = await mountCard();
    // Exactly what `subscribe` replays on attach: the one-shot `start`, then the
    // single latest event. Files 1-2 are finished and their events are gone.
    await deliver(source, startEvent, latestEvent);

    expect(mounted!.text()).toContain('4.0 MB / 7.0 MB');
    expect(mounted!.text()).toContain('File 3 of 3');
    expect(mounted!.container.querySelector('[role="progressbar"]')!.getAttribute('aria-valuenow')).toBe('57');
  });

  it('reads the same total for a card that watched the job from its first byte', async () => {
    // The over-correction guard: the mid-job number above is only right if it
    // agrees with the full stream. Both cards are looking at one job.
    const source = await mountCard();
    await deliver(source, startEvent, ...settled, latestEvent);

    expect(mounted!.text()).toContain('4.0 MB / 7.0 MB');
  });

  it('never rewinds when a reconnect replays start plus the latest event again', async () => {
    // A subscription that drops re-subscribes, and the runtime replays on every
    // attach. The card is already at 4 MiB; the replay must not walk it back to
    // the current file's own 1 MiB.
    const source = await mountCard();
    await deliver(source, startEvent, ...settled, latestEvent);
    await deliver(source, startEvent, latestEvent);

    expect(mounted!.text()).toContain('4.0 MB / 7.0 MB');
  });

  it('settles at the job total on done and releases the subscription on unmount', async () => {
    const source = await mountCard();
    await deliver(source, startEvent, latestEvent, {
      type: 'done',
      id: 'job-1',
      outputDir: '/models/sharded-model',
    });

    expect(mounted!.text()).toContain('Complete');
    expect(mounted!.text()).toContain('7.0 MB / 7.0 MB');

    mounted!.unmount();
    mounted = undefined;
    // The unsubscribe crosses the port too.
    await settle();
    expect(source.closed).toBe(true);
  });
});

describe('DownloadProgress — rate, glyph and stall', () => {
  it('shows a rate after two progress frames a second apart, and the time left from 10 s on', async () => {
    let clock = 10_000;
    const now = vi.spyOn(performance, 'now').mockImplementation(() => clock);
    try {
      const source = await mountCard();
      await deliver(source, { ...startEvent, totalBytes: 700 * MIB }, { ...latestEvent, jobReceivedBytes: 2 * MIB });
      expect(mounted!.text()).not.toContain('/s');
      clock += 1000;
      await deliver(source, { ...latestEvent, jobReceivedBytes: 4 * MIB });
      expect(mounted!.text()).toContain('File 3 of 3 · 2.0 MB/s');
      expect(mounted!.text()).not.toContain('left');
      clock += 9000;
      await deliver(source, { ...latestEvent, jobReceivedBytes: 22 * MIB });
      expect(mounted!.text()).toContain('File 3 of 3 · 2.0 MB/s · about 6 min left');
    } finally {
      now.mockRestore();
    }
  });

  it("the glyph's data-status goes loading -> success on done", async () => {
    const source = await mountCard();
    const glyph = (): string | null | undefined =>
      mounted!.container.querySelector('svg[data-status]')?.getAttribute('data-status');
    expect(glyph()).toBe('loading');
    await deliver(source, startEvent, latestEvent);
    expect(glyph()).toBe('loading');
    await deliver(source, { type: 'done', id: 'job-1', outputDir: '/models/sharded-model' });
    expect(glyph()).toBe('success');
  });

  it('shows Checking <file> for a check frame', async () => {
    const source = await mountCard();
    await deliver(source, startEvent, { ...latestEvent, file: 'Clairic-Flash-UD-Q4_K_XL.gguf', phase: 'check' });
    expect(mounted!.text()).toContain('Checking Clairic-Flash-UD-Q4_K_XL.gguf');
    expect(mounted!.text()).toContain('4.0 MB / 7.0 MB');
    await deliver(source, { ...latestEvent, file: 'mmproj-BF16.gguf' });
    expect(mounted!.text()).toContain('mmproj-BF16.gguf');
    expect(mounted!.text()).not.toContain('Checking');
  });

  it("the glyph's data-status goes loading -> error on error", async () => {
    const source = await mountCard();
    await deliver(source, startEvent, latestEvent, { type: 'error', id: 'job-1', message: 'disk full' });
    expect(mounted!.container.querySelector('svg[data-status]')?.getAttribute('data-status')).toBe('error');
    expect(mounted!.text()).toContain('disk full');
    expect(mounted!.container.querySelector('[role="progressbar"]')!.getAttribute('aria-valuenow')).toBe('57');
  });
});

describe('reduceProgress', () => {
  const running = (): ProgressState => reduceProgress(INITIAL_PROGRESS, startEvent, 0);

  it('ignores a rewind on reconnect when computing the rate', () => {
    let state = reduceProgress(running(), { ...latestEvent, jobReceivedBytes: 2 * MIB }, 0);
    state = reduceProgress(state, { ...latestEvent, jobReceivedBytes: 4 * MIB }, 1000);
    expect(state.rate?.bytesPerSec).toBe(2 * MIB);
    state = reduceProgress(state, { ...latestEvent, jobReceivedBytes: MIB }, 2000);
    expect(state.rate).toEqual({ at: 2000, bytes: MIB, bytesPerSec: 2 * MIB, since: 0 });
    state = reduceProgress(state, { ...latestEvent, jobReceivedBytes: 2 * MIB }, 3000);
    expect(state.rate?.bytesPerSec).toBeCloseTo(0.3 * MIB + 0.7 * 2 * MIB);
  });

  it('keeps the anchor for frames less than a second apart, and a start resets the rate', () => {
    let state = reduceProgress(running(), { ...latestEvent, jobReceivedBytes: MIB }, 0);
    state = reduceProgress(state, { ...latestEvent, jobReceivedBytes: 3 * MIB }, 400);
    expect(state.rate).toEqual({ at: 0, bytes: MIB, bytesPerSec: null, since: 0 });
    state = reduceProgress(state, startEvent, 500);
    expect(state.rate).toBeNull();
  });

  it('stalled sets stalled and leaves the phase; the next progress or start clears it', () => {
    const stalled = reduceProgress(reduceProgress(running(), latestEvent, 0), { type: 'stalled' });
    expect(stalled).toMatchObject({ phase: 'running', stalled: true });
    expect(reduceProgress(stalled, latestEvent, 10).stalled).toBe(false);
    expect(reduceProgress(stalled, startEvent, 10).stalled).toBe(false);
    expect(reduceProgress(INITIAL_PROGRESS, { type: 'stalled' })).toMatchObject({ phase: 'connecting', stalled: true });
  });

  it('stalled in done or error changes nothing', () => {
    const done = reduceProgress(running(), { type: 'done', id: 'job-1', outputDir: '/m' }, 0);
    expect(reduceProgress(done, { type: 'stalled' })).toBe(done);
    const failed = reduceProgress(running(), { type: 'error', id: 'job-1', message: 'x' }, 0);
    expect(reduceProgress(failed, { type: 'stalled' })).toBe(failed);
  });
});

describe('time left', () => {
  const GB = 1e9;
  const start = reduceProgress(INITIAL_PROGRESS, { ...startEvent, totalBytes: 20.2 * GB, fileCount: 1 }, 0);
  const at = (bytes: number, phase?: 'check'): DownloadEvent => ({
    ...latestEvent,
    receivedBytes: bytes,
    jobReceivedBytes: bytes,
    totalBytes: 20.2 * GB,
    fileIndex: 0,
    fileCount: 1,
    ...(phase === undefined ? {} : { phase }),
  });
  /** Frames one second apart at `rate` bytes a second, from `from` ms for `seconds`. */
  const steady = (state: ProgressState, rate: number, from: number, seconds: number, phase?: 'check') => {
    let next = state;
    for (let i = 0; i <= seconds; i += 1) {
      next = reduceProgress(next, at(next.receivedBytes + (i === 0 ? 0 : rate), phase), from + i * 1000);
    }
    return next;
  };

  it.each<[number, string]>([
    [59, 'under a minute left'],
    [61, 'about 1 min left'],
    [3570, 'about 1 h left'],
    [5400, 'about 1 h 30 min left'],
    [35_880, 'about 10 h left'],
    [40_000, 'about 11 h left'],
    [82_800, 'about 23 h left'],
    [86_000, 'more than a day left'],
  ])('%i s reads "%s"', (seconds, text) => {
    expect(timeLeftText(seconds)).toBe(text);
  });

  it('says nothing in the first 10 s of the download, then the time left', () => {
    // Mutation: drop the warm-up → the first seconds read a time → red.
    const first = reduceProgress(reduceProgress(start, at(0), 0), at(79e3), 1000);
    expect(progressSummary(first, { committing: false, cancelling: false })).toBe('79.0 kB / 20.2 GB · 79.0 kB/s');
    expect(timeLeft(steady(start, 48e6, 0, 9))).toBeNull();
    expect(timeLeft(steady(start, 48e6, 0, 10))).toBe('about 7 min left');
    // A true 63.8 kB/s is more than a day for 20.2 GB, never a count of thousands of minutes.
    expect(timeLeft(steady(start, 63.8e3, 0, 12))).toBe('more than a day left');
  });

  it('says nothing while checking or stalled, and measures the download from its own first frame', () => {
    const checked = steady(start, 2e9, 0, 12, 'check');
    expect(timeLeft(checked)).toBeNull();
    expect(timeLeft(reduceProgress(steady(start, 48e6, 0, 12), { type: 'stalled' }))).toBeNull();
    // Mutation: keep the hashing rate across the phase change → a time from the check speed at once → red.
    const downloading = reduceProgress(checked, at(checked.receivedBytes), 13_000);
    expect(downloading.rate).toEqual({ at: 13_000, bytes: checked.receivedBytes, bytesPerSec: null, since: 13_000 });
    expect(timeLeft(steady(downloading, 48e6, 14_000, 8))).toBeNull();
    expect(timeLeft(steady(downloading, 48e6, 14_000, 10))).not.toBeNull();
  });
});

describe('progressSummary — the build line of a catalog card', () => {
  const GB = 1e9;
  const idle = { committing: false, cancelling: false };
  const started = reduceProgress(INITIAL_PROGRESS, {
    type: 'start',
    id: 'j',
    repo: 'org/model',
    totalBytes: 16.6 * GB,
    fileCount: 2,
  });
  const frame = (bytes: number, phase?: 'check') =>
    ({
      type: 'progress',
      id: 'j',
      file: 'model-00002-of-00002.gguf',
      receivedBytes: bytes,
      jobReceivedBytes: bytes,
      totalBytes: 16.6 * GB,
      fileIndex: 1,
      fileCount: 2,
      ...(phase === undefined ? {} : { phase }),
    }) as DownloadEvent;
  const running = reduceProgress(reduceProgress(started, frame(6.6 * GB), 0), frame(7.4 * GB), 10_000);

  it('names each step of a job', () => {
    expect(progressSummary(INITIAL_PROGRESS, idle)).toBe('Starting…');
    expect(progressSummary(running, idle)).toBe('7.4 / 16.6 GB · 80.0 MB/s · about 2 min left');
    expect(progressSummary(reduceProgress(started, frame(GB, 'check'), 0), idle)).toBe(
      'Checking files · 1.0 / 16.6 GB',
    );
    // Mutation: committing gives the running text → red.
    expect(progressSummary(running, { committing: true, cancelling: false })).toBe('Finishing…');
    expect(progressSummary(running, { committing: false, cancelling: true })).toBe('Cancelling…');
  });

  it('gives the bar its percent and its file line', () => {
    expect(progressPercent(running)).toBe(45);
    expect(progressFileLine(running)).toBe('model-00002-of-00002.gguf · File 2 of 2');
    expect(progressFileLine(started)).toBeNull();
    expect(bytesOfTotal(512e6, 2e9)).toBe('512 MB / 2.0 GB');
    // Powers of 1000, as the catalog states a size: 16,564,259,744 B is 16.6 GB, not 15.4.
    expect(bytesOfTotal(16_564_259_744, 16_564_259_744)).toBe('16.6 / 16.6 GB');
  });
});
