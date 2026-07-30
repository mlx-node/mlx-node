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

import { DownloadProgress } from '@/components/download-progress';
import { act, createElement } from 'react';
import { afterEach, describe, expect, it } from 'vite-plus/test';

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
