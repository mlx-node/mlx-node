import { mkdtemp, readFile, rm, stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { sharedStreamFactory } from '../src/provider/shared-client.js';
import { SHARED_PROTOCOL, type SharedRequest } from '../src/provider/shared-protocol.js';
import { startSharedService, type SharedBackend } from '../src/provider/shared-service.js';
import type { StreamSimpleHost } from '../src/provider/stream-adapter.js';

const request: SharedRequest = {
  profile: {
    discovered: { name: 'local', path: '/models/local', modelType: 'qwen3' },
    persistPagedCache: true,
    preserveEmbeddedGemmaDraft: false,
  },
  model: {
    id: 'local',
    api: 'mlx',
    provider: 'mlx',
    contextWindow: 32768,
    maxTokens: 4096,
    input: ['text'],
  } as SharedRequest['model'],
  context: { messages: [{ role: 'user', content: 'task', timestamp: 0 }] },
  options: {},
};
const cleanup: Array<() => Promise<unknown>> = [];
afterEach(async () => {
  for (const close of cleanup.splice(0).reverse()) await close();
});

async function fixture(stream: SharedBackend['stream'], idleMs?: number) {
  const directory = await mkdtemp(join(tmpdir(), 'mlx-shared-service-'));
  cleanup.push(() => rm(directory, { recursive: true, force: true }));
  const close = vi.fn(async () => {});
  const loadBackend = vi.fn(async () => ({ stream, busy: () => false, close }));
  const service = await startSharedService({ directory, port: 0, loadBackend, idleMs });
  cleanup.push(() => service.close());
  const url = `http://127.0.0.1:${service.endpoint.port}`;
  const headers = { authorization: `Bearer ${service.endpoint.token}` };
  return { ...service, directory, closeBackend: close, loadBackend, url, headers };
}

describe('shared delegate service transport', () => {
  it('elects one listener before loading the backend and publishes a private endpoint', async () => {
    const service = await fixture(async function* () {
      yield { error: 'fixture' };
    });
    expect(service.loadBackend).not.toHaveBeenCalled();
    const contender = vi.fn();
    await expect(
      startSharedService({ directory: service.directory, port: service.endpoint.port, loadBackend: contender }),
    ).rejects.toMatchObject({ code: 'EADDRINUSE' });
    expect(contender).not.toHaveBeenCalled();
    expect(JSON.parse(await readFile(join(service.directory, 'endpoint.json'), 'utf8'))).toEqual(service.endpoint);
    expect((await stat(join(service.directory, 'endpoint.json'))).mode & 0o777).toBe(0o600);
    const health = await fetch(`${service.url}/health`, { headers: service.headers });
    expect(await health.json()).toEqual({ protocol: SHARED_PROTOCOL, pid: process.pid });
    const denied = await fetch(`${service.url}/stream`, { method: 'POST', body: JSON.stringify(request) });
    expect(denied.status).toBe(401);
    await denied.body?.cancel();
    expect(service.loadBackend).not.toHaveBeenCalled();
  });

  it('reuses one backend for simultaneous requests and never mixes their outputs', async () => {
    let started = 0;
    let release!: () => void;
    const both = new Promise<void>((resolve) => {
      release = resolve;
    });
    const service = await fixture(async function* (body) {
      if (++started === 2) release();
      await both;
      yield { error: body.options.sessionId! };
    });
    const send = async (sessionId: string) => {
      const response = await fetch(`${service.url}/stream`, {
        method: 'POST',
        headers: service.headers,
        body: JSON.stringify({ ...request, options: { sessionId } }),
      });
      return response.text();
    };
    expect(await Promise.all([send('one'), send('two')])).toEqual(['{"error":"one"}\n', '{"error":"two"}\n']);
    expect(service.loadBackend).toHaveBeenCalledOnce();
  });

  it('keeps terminal message identity, settings, model limits, and metrics across the client', async () => {
    const message = {
      role: 'assistant',
      content: [{ type: 'text', text: 'answer' }],
      api: 'mlx',
      provider: 'mlx',
      model: 'local',
      usage: {},
      stopReason: 'stop',
      timestamp: 1,
    } as never;
    let received!: SharedRequest;
    const service = await fixture(async function* (body) {
      received = body;
      yield { event: { type: 'start', partial: message } };
      yield { event: { type: 'done', reason: 'stop', message } };
      yield { performance: { decodeTokensPerSecond: 20, prefillTokensPerSecond: 100 } as never, message };
      yield { model: { contextWindow: 2048, maxTokens: 1024, input: ['text', 'image'] } };
    });
    const performance = vi.fn();
    const host = { modelInfo: () => request.profile.discovered } as unknown as StreamSimpleHost;
    const model = { ...request.model };
    const stream = sharedStreamFactory(request.profile, async () => service.endpoint)(
      host,
      performance,
      () => 'root',
      undefined,
      undefined,
      () => '/session',
      () => 128,
    )(model, request.context, { sessionId: 'child', reasoning: 'high', temperature: 0.2, maxTokens: 456 });
    const events = [];
    for await (const event of stream) events.push(event);
    const terminal = events.at(-1)!;
    expect(terminal.type).toBe('done');
    if (terminal.type !== 'done') throw new Error('Expected completion');
    expect(performance.mock.calls[0]![0]).toBe(terminal.message);
    expect(received).toMatchObject({
      rootSessionId: 'root',
      rootSessionFile: '/session',
      thinkingBudget: 128,
      options: { sessionId: 'child', reasoning: 'high', temperature: 0.2, maxTokens: 456 },
    });
    expect(model).toMatchObject({ contextWindow: 2048, maxTokens: 1024, input: ['text', 'image'] });
  });

  it('reports a truncated stream as failure and never invokes the local model host', async () => {
    const service = await fixture(async function* () {
      yield { model: { contextWindow: 1, maxTokens: 1, input: ['text'] } };
    });
    const local = vi.fn();
    const host = { modelInfo: () => request.profile.discovered, runWithResident: local } as unknown as StreamSimpleHost;
    const stream = sharedStreamFactory(request.profile, async () => service.endpoint)(host)(
      request.model,
      request.context,
    );
    const events = [];
    for await (const event of stream) events.push(event);
    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({
      type: 'error',
      reason: 'error',
      error: { errorMessage: 'Shared inference connection ended before completion.' },
    });
    expect(local).not.toHaveBeenCalled();
  });

  it('cancels only the disconnected request and allows the other to finish', async () => {
    let cancelled!: () => void;
    const aborted = new Promise<void>((resolve) => {
      cancelled = resolve;
    });
    let started!: () => void;
    const ready = new Promise<void>((resolve) => {
      started = resolve;
    });
    const service = await fixture(async function* (body, signal) {
      if (body.options.sessionId === 'abort') {
        started();
        await new Promise<void>((resolve) =>
          signal.addEventListener(
            'abort',
            () => {
              cancelled();
              resolve();
            },
            { once: true },
          ),
        );
      } else yield { error: 'other finished' };
    });
    const controller = new AbortController();
    const pending = fetch(`${service.url}/stream`, {
      method: 'POST',
      headers: service.headers,
      signal: controller.signal,
      body: JSON.stringify({ ...request, options: { sessionId: 'abort' } }),
    });
    const rejected = expect(pending).rejects.toThrow();
    await ready;
    controller.abort();
    await rejected;
    await aborted;
    const response = await fetch(`${service.url}/stream`, {
      method: 'POST',
      headers: service.headers,
      body: JSON.stringify(request),
    });
    expect(await response.text()).toContain('other finished');
  });

  it('releases the service after its idle timeout', async () => {
    const service = await fixture(async function* () {
      yield { error: 'done' };
    }, 30);
    await (
      await fetch(`${service.url}/stream`, { method: 'POST', headers: service.headers, body: JSON.stringify(request) })
    ).text();
    await vi.waitFor(() => expect(service.closeBackend).toHaveBeenCalledOnce());
  });

  it('holds the election port until model cleanup finishes', async () => {
    const service = await fixture(async function* () {
      yield { error: 'done' };
    });
    await (
      await fetch(`${service.url}/stream`, { method: 'POST', headers: service.headers, body: JSON.stringify(request) })
    ).text();
    let release!: () => void;
    service.closeBackend.mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          release = resolve;
        }),
    );
    const closing = service.close();
    await vi.waitFor(() => expect(service.closeBackend).toHaveBeenCalledOnce());
    try {
      await expect(
        startSharedService({ directory: service.directory, port: service.endpoint.port, loadBackend: vi.fn() }),
      ).rejects.toMatchObject({ code: 'EADDRINUSE' });
    } finally {
      release();
      await closing;
    }
  });
});
