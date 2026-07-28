/**
 * @vitest-environment happy-dom
 */

/**
 * Render guards for the download progress card.
 *
 * The card is the only consumer of the SSE download stream, and the rule that
 * matters — what the byte counter reads when a client attaches PART-WAY through
 * a sharded model — lives in its reducer, not in a helper. The server replay is
 * one `start` frame plus the single latest event, so a card that derives the
 * job total by summing the per-file frames it happened to witness renders a
 * mount-order-dependent number: correct for a card that watched from the first
 * byte, short by every already-finished file for one mounted after a reload.
 * These tests drive the component through a stubbed `EventSource` with exactly
 * the frames `DownloadManager` emits (see `download.test.ts`) and assert on the
 * rendered text, so that difference is a red test rather than a bar that
 * silently lies by gigabytes.
 *
 * `happy-dom` is scoped to this file by the docblock above.
 */

import { DownloadProgress } from '@/components/download-progress';
import { act, createElement } from 'react';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { renderPage, type RenderedPage } from './render.js';

const MIB = 1024 * 1024;

/** Mirrors `DownloadEvent` from `packages/dashboard/src/download.ts`. */
type DownloadEvent =
  | { type: 'start'; id: string; repo: string; totalBytes: number; fileCount: number }
  | {
      type: 'progress';
      id: string;
      file: string;
      receivedBytes: number;
      jobReceivedBytes: number;
      totalBytes: number;
      fileIndex: number;
      fileCount: number;
    }
  | { type: 'done'; id: string; outputDir: string }
  | { type: 'error'; id: string; message: string }
  | { type: 'cancelled'; id: string };

/**
 * A stand-in for the browser `EventSource` the SPA client opens. Frames are
 * delivered by NAME (`event: progress`), which is how the server tags them and
 * therefore the only way the card's listeners ever fire.
 */
class FakeEventSource {
  static instances: FakeEventSource[] = [];

  readonly url: string;
  closed = false;
  onmessage: ((ev: MessageEvent<string>) => void) | null = null;
  onerror: ((ev: Event) => void) | null = null;
  private readonly listeners = new Map<string, Array<(ev: MessageEvent<string>) => void>>();

  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }

  addEventListener(name: string, fn: (ev: MessageEvent<string>) => void): void {
    const forName = this.listeners.get(name) ?? [];
    forName.push(fn);
    this.listeners.set(name, forName);
  }

  close(): void {
    this.closed = true;
  }

  /** Push one server frame at the card, as the SSE stream spells it. */
  deliver(event: DownloadEvent): void {
    const message = new MessageEvent<string>(event.type, { data: JSON.stringify(event) });
    for (const fn of this.listeners.get(event.type) ?? []) fn(message);
  }
}

/** Install {@link FakeEventSource} as the global; returns a disposer. */
function stubEventSource(): () => void {
  const previous = globalThis.EventSource;
  FakeEventSource.instances = [];
  globalThis.EventSource = FakeEventSource as unknown as typeof EventSource;
  return () => {
    globalThis.EventSource = previous;
  };
}

const startFrame: DownloadEvent = {
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
 * The single latest frame at the moment of the reload: the third file is 1 MiB
 * into its 4 MiB, so the JOB holds 4 MiB of 7 MiB. Taken from the manager
 * repro in `download.test.ts`.
 */
const latestFrame: DownloadEvent = {
  type: 'progress',
  id: 'job-1',
  file: 'model-00002-of-00002.safetensors',
  receivedBytes: MIB,
  jobReceivedBytes: 4 * MIB,
  totalBytes: 4 * MIB,
  fileIndex: 2,
  fileCount: 3,
};

let mounted: RenderedPage | undefined;
let restoreEventSource: (() => void) | undefined;

afterEach(() => {
  mounted?.unmount();
  mounted = undefined;
  restoreEventSource?.();
  restoreEventSource = undefined;
});

/** Mount the card and return its open stream, settled on the connecting state. */
async function mountCard(): Promise<FakeEventSource> {
  restoreEventSource = stubEventSource();
  mounted = await renderPage(createElement(DownloadProgress, { id: 'job-1' }), (text) => text.includes('Starting'));
  expect(FakeEventSource.instances).toHaveLength(1);
  return FakeEventSource.instances[0]!;
}

/** Deliver frames inside `act` so React flushes the state they produce. */
async function deliver(source: FakeEventSource, ...events: DownloadEvent[]): Promise<void> {
  await act(async () => {
    for (const event of events) source.deliver(event);
  });
}

describe('DownloadProgress — aggregate bytes across a sharded job', () => {
  it('renders the whole-job total from the two frames a mid-job subscriber is replayed', async () => {
    const source = await mountCard();
    // Exactly what `subscribe` replays on attach: the one-shot `start`, then the
    // single latest event. Files 1-2 are finished and their frames are gone.
    await deliver(source, startFrame, latestFrame);

    expect(mounted!.text()).toContain('4.0 MB / 7.0 MB');
    expect(mounted!.text()).toContain('File 3 of 3');
    expect(mounted!.container.querySelector('[role="progressbar"]')!.getAttribute('aria-valuenow')).toBe('57');
  });

  it('reads the same total for a card that watched the job from its first byte', async () => {
    // The over-correction guard: the mid-job number above is only right if it
    // agrees with the full stream. Both cards are looking at one job.
    const source = await mountCard();
    await deliver(source, startFrame, ...settled, latestFrame);

    expect(mounted!.text()).toContain('4.0 MB / 7.0 MB');
  });

  it('never rewinds when a reconnect replays start plus the latest frame again', async () => {
    // An EventSource that drops re-subscribes, and the server replays on every
    // attach. The card is already at 4 MiB; the replay must not walk it back to
    // the current file's own 1 MiB.
    const source = await mountCard();
    await deliver(source, startFrame, ...settled, latestFrame);
    await deliver(source, startFrame, latestFrame);

    expect(mounted!.text()).toContain('4.0 MB / 7.0 MB');
  });

  it('settles at the job total on done and closes the stream on unmount', async () => {
    const source = await mountCard();
    await deliver(source, startFrame, latestFrame, { type: 'done', id: 'job-1', outputDir: '/models/sharded-model' });

    expect(mounted!.text()).toContain('Complete');
    expect(mounted!.text()).toContain('7.0 MB / 7.0 MB');

    mounted!.unmount();
    mounted = undefined;
    expect(source.closed).toBe(true);
  });
});
