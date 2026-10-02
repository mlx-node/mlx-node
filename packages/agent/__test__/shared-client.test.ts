import { spawn, type ChildProcess } from 'node:child_process';
import { EventEmitter } from 'node:events';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { emptyUsage } from '../src/provider/events.js';
import { ensureSharedWorker, sharedStreamFactory } from '../src/provider/shared-client.js';
import {
  SHARED_PROTOCOL,
  sharedLocation,
  type SharedFrame,
  type SharedRequest,
} from '../src/provider/shared-protocol.js';
import { startSharedService } from '../src/provider/shared-service.js';
import type { StreamSimpleHost } from '../src/provider/stream-adapter.js';

vi.mock('node:child_process', () => ({ spawn: vi.fn() }));
vi.mock('../src/provider/shared-protocol.js', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../src/provider/shared-protocol.js')>()),
  sharedLocation: vi.fn(),
}));

afterEach(() => {
  vi.restoreAllMocks();
  vi.mocked(spawn).mockReset();
});

const endpoint = { protocol: SHARED_PROTOCOL, port: 19000, token: 'fixture', pid: 1 };
const model = { id: 'local', api: 'mlx', provider: 'mlx' } as SharedRequest['model'];
const host = {
  modelInfo: () => ({ name: 'local', path: '/fixture', modelType: 'qwen3' }),
} as unknown as StreamSimpleHost;
const done: SharedFrame = {
  event: {
    type: 'done',
    reason: 'stop',
    message: {
      role: 'assistant',
      content: [],
      api: 'mlx',
      provider: 'mlx',
      model: 'local',
      usage: emptyUsage(),
      stopReason: 'stop',
      timestamp: 0,
    },
  },
};
async function collect(connect: typeof ensureSharedWorker, signal?: AbortSignal) {
  const stream = sharedStreamFactory({ persistPagedCache: true, preserveEmbeddedGemmaDraft: false }, connect)(host)(
    model,
    { messages: [] },
    { signal },
  );
  const events = [];
  for await (const event of stream) events.push(event);
  return events;
}

describe('shared worker rotation', () => {
  it.each([401, 503])('rediscovers after a pre-admission HTTP %i rejection', async (status) => {
    const fetch = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(new Response('', { status }))
      .mockResolvedValueOnce(new Response(`${JSON.stringify(done)}\n`));
    const connect = vi.fn(async () => endpoint);
    expect(await collect(connect)).toEqual([done.event]);
    expect(connect).toHaveBeenCalledTimes(2);
    expect(fetch.mock.calls[0]![1]!.body).toBe(fetch.mock.calls[1]![1]!.body);
  });

  it('rediscovers after connection refusal before admission', async () => {
    vi.spyOn(globalThis, 'fetch')
      .mockRejectedValueOnce(new TypeError('fetch failed', { cause: { code: 'ECONNREFUSED' } }))
      .mockResolvedValueOnce(new Response(`${JSON.stringify(done)}\n`));
    const connect = vi.fn(async () => endpoint);
    expect(await collect(connect)).toEqual([done.event]);
    expect(connect).toHaveBeenCalledTimes(2);
  });

  it('does not treat queue saturation as retirement', async () => {
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('', { status: 429 }));
    const connect = vi.fn(async () => endpoint);
    expect(await collect(connect)).toMatchObject([
      { type: 'error', error: { errorMessage: 'Shared delegate inference failed (HTTP 429).' } },
    ]);
    expect(connect).toHaveBeenCalledOnce();
    expect(fetch).toHaveBeenCalledOnce();
  });

  it('does not replay a request whose acceptance is unknown', async () => {
    const fetch = vi
      .spyOn(globalThis, 'fetch')
      .mockRejectedValue(new TypeError('fetch failed', { cause: { code: 'ECONNRESET' } }));
    const connect = vi.fn(async () => endpoint);
    expect(await collect(connect)).toMatchObject([{ type: 'error', error: { errorMessage: 'fetch failed' } }]);
    expect(connect).toHaveBeenCalledOnce();
    expect(fetch).toHaveBeenCalledOnce();
  });

  it('bounds repeated retirement retries', async () => {
    const fetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => new Response('', { status: 503 }));
    const connect = vi.fn(async () => endpoint);
    expect(await collect(connect)).toMatchObject([
      { type: 'error', error: { errorMessage: 'Shared delegate inference failed (HTTP 503).' } },
    ]);
    expect(connect).toHaveBeenCalledTimes(3);
    expect(fetch).toHaveBeenCalledTimes(3);
  });

  it('stops retrying when the caller cancels during rediscovery', async () => {
    const controller = new AbortController();
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('', { status: 503 }));
    const connect = vi.fn(async (signal?: AbortSignal) => {
      if (connect.mock.calls.length === 2) controller.abort();
      signal?.throwIfAborted();
      return endpoint;
    });
    expect(await collect(connect, controller.signal)).toMatchObject([{ type: 'error', reason: 'aborted' }]);
    expect(connect).toHaveBeenCalledTimes(2);
    expect(fetch).toHaveBeenCalledOnce();
  });

  it('re-elects after a contender exits while the retiring worker holds the claim', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'mlx-shared-discovery-'));
    let release!: () => void;
    const cleanupGate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const original = await startSharedService({
      directory,
      port: 0,
      loadBackend: async () => ({
        busy: () => false,
        close: () => cleanupGate,
        async *stream() {
          yield done;
        },
      }),
    });
    const response = await fetch(`http://127.0.0.1:${original.endpoint.port}/stream`, {
      method: 'POST',
      headers: { authorization: `Bearer ${original.endpoint.token}` },
      body: JSON.stringify({
        clientId: 'fixture',
        profile: { discovered: { path: '/fixture', name: 'local' } },
        model,
        context: { messages: [] },
      }),
    });
    await response.text();
    const closing = original.close();
    const location = { directory };
    vi.mocked(sharedLocation).mockReturnValue(location);
    let lostElection!: () => void;
    const contenderExited = new Promise<void>((resolve) => {
      lostElection = resolve;
    });
    const loadBackend = vi.fn();
    const elected: Array<Awaited<ReturnType<typeof startSharedService>>> = [];
    const launches: Promise<void>[] = [];
    vi.mocked(spawn).mockImplementation(() => {
      const child = Object.assign(new EventEmitter(), {
        exitCode: null as number | null,
        signalCode: null,
        unref() {},
      });
      launches.push(
        startSharedService({ ...location, loadBackend }).then(
          (service) => {
            elected.push(service);
          },
          (error: NodeJS.ErrnoException) => {
            child.exitCode = error.code === 'EADDRINUSE' ? 0 : 1;
            child.emit('exit', child.exitCode, null);
            lostElection();
          },
        ),
      );
      return child as ChildProcess;
    });
    const controller = new AbortController();
    const pending = ensureSharedWorker(controller.signal);
    try {
      await contenderExited;
      expect(loadBackend).not.toHaveBeenCalled();
      release();
      await closing;
      const replacement = await pending;
      expect(replacement).toEqual(elected[0]!.endpoint);
      expect(replacement.token).not.toBe(original.endpoint.token);
      expect(spawn).toHaveBeenCalledTimes(2);
      expect(elected).toHaveLength(1);
      expect(loadBackend).not.toHaveBeenCalled();
    } finally {
      controller.abort();
      await pending.catch(() => {});
      release();
      await closing;
      await Promise.all(launches);
      for (const service of elected) await service.close();
      await rm(directory, { recursive: true, force: true });
    }
  });
});
