import { mkdtemp, readFile, rm, stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { sharedStreamFactory } from '../src/provider/shared-client.js';
import { SHARED_PROTOCOL, sharedCacheOwners, type SharedRequest } from '../src/provider/shared-protocol.js';
import { startSharedService, type SharedBackend } from '../src/provider/shared-service.js';
import type { StreamSimpleHost } from '../src/provider/stream-adapter.js';

const request: SharedRequest = {
  clientId: 'fixture',
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

  it('elects one worker per private directory even when contenders use different free ports', async () => {
    const service = await fixture(async function* () {});
    const loadBackend = vi.fn();
    await expect(startSharedService({ directory: service.directory, port: 0, loadBackend })).rejects.toMatchObject({
      code: 'EADDRINUSE',
    });
    expect(loadBackend).not.toHaveBeenCalled();
    const anotherUser = await fixture(async function* () {});
    expect(anotherUser.endpoint.port).not.toBe(service.endpoint.port);
    expect(JSON.parse(await readFile(join(service.directory, 'endpoint.json'), 'utf8'))).toEqual(service.endpoint);
  });

  it('does not replace a live owner whose health check times out', async () => {
    const service = await fixture(async function* () {});
    const fetch = vi.spyOn(globalThis, 'fetch').mockRejectedValue(new DOMException('expired', 'TimeoutError'));
    try {
      await expect(startSharedService({ directory: service.directory, loadBackend: vi.fn() })).rejects.toMatchObject({
        code: 'EADDRINUSE',
      });
      expect(JSON.parse(await readFile(join(service.directory, 'endpoint.json'), 'utf8'))).toEqual(service.endpoint);
      expect(service.loadBackend).not.toHaveBeenCalled();
    } finally {
      fetch.mockRestore();
    }
  });

  it('recovers an unreleased claim when its PID is alive but its listener is gone', async () => {
    const service = await fixture(async function* () {});
    await service.close();
    // Simulate a crash followed by PID reuse: the recorded PID is this still
    // live test process, but the claimed listener no longer exists.
    await rm(join(service.directory, 'claims', '0.json.released'));
    const replacement = await startSharedService({ directory: service.directory, loadBackend: vi.fn() });
    cleanup.push(() => replacement.close());
    expect(replacement.endpoint.token).not.toBe(service.endpoint.token);
  });

  it('keeps cache owners stable across turns and isolates callers resuming the same Pi session', async () => {
    const owners: ReturnType<typeof sharedCacheOwners>[] = [];
    const service = await fixture(async function* (body) {
      owners.push(sharedCacheOwners(body));
      yield { error: 'captured' };
    });
    const host = { modelInfo: () => request.profile.discovered } as unknown as StreamSimpleHost;
    const caller = () =>
      sharedStreamFactory(request.profile, async () => service.endpoint)(host, undefined, () => 'same-root');
    const first = caller();
    for (const stream of [first, first, caller()]) {
      for await (const _event of stream(request.model, request.context, { sessionId: 'same-child' })) {
        /* drain */
      }
    }
    expect(owners[0]).toEqual(owners[1]);
    expect(owners[2]!.owner).not.toBe(owners[0]!.owner);
    expect(owners[2]!.root).not.toBe(owners[0]!.root);
    expect(owners[0]!.owner).not.toBe(owners[0]!.root);
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
    const connect = vi.fn(async () => service.endpoint);
    const stream = sharedStreamFactory(request.profile, connect)(host)(request.model, request.context);
    const events = [];
    for await (const event of stream) events.push(event);
    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({
      type: 'error',
      reason: 'error',
      error: { errorMessage: 'Shared inference connection ended before completion.' },
    });
    expect(local).not.toHaveBeenCalled();
    expect(connect).toHaveBeenCalledOnce();
  });

  it('reconnects when a healthy worker retires before accepting the stream', async () => {
    const originalStream = vi.fn(async function* () {
      yield { error: 'primed' };
    });
    const service = await fixture(originalStream);
    await (
      await fetch(`${service.url}/stream`, { method: 'POST', headers: service.headers, body: JSON.stringify(request) })
    ).text();
    let release!: () => void;
    const cleanupGate = new Promise<void>((resolve) => {
      release = resolve;
    });
    service.closeBackend.mockImplementation(() => cleanupGate);
    let closing: Promise<void> | undefined;
    const replacementStream = vi.fn(async function* () {
      yield {
        event: {
          type: 'done',
          reason: 'stop',
          message: { content: [{ type: 'text', text: 'replacement finished' }] },
        },
      } as never;
    });
    const connect = vi.fn(async () => {
      if (!closing) {
        const health = await fetch(`${service.url}/health`, { headers: service.headers });
        expect(health.ok).toBe(true);
        await health.body?.cancel();
        // Deterministically place shutdown between discovery and POST /stream.
        closing = service.close();
        return service.endpoint;
      }
      release();
      await closing;
      const replacement = await startSharedService({
        directory: service.directory,
        port: service.endpoint.port,
        loadBackend: async () => ({ stream: replacementStream, busy: () => false, close: async () => {} }),
      });
      cleanup.push(() => replacement.close());
      return replacement.endpoint;
    });
    const local = vi.fn();
    const host = { modelInfo: () => request.profile.discovered, runWithResident: local } as unknown as StreamSimpleHost;
    try {
      const stream = sharedStreamFactory(request.profile, connect)(host)(request.model, request.context);
      const events = [];
      for await (const event of stream) events.push(event);
      expect(events).toEqual([
        { type: 'done', reason: 'stop', message: { content: [{ type: 'text', text: 'replacement finished' }] } },
      ]);
      expect(connect).toHaveBeenCalledTimes(2);
      expect(originalStream).toHaveBeenCalledOnce();
      expect(replacementStream).toHaveBeenCalledOnce();
      expect(local).not.toHaveBeenCalled();
    } finally {
      release();
      await closing;
    }
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

  it('does not cancel backend cleanup after a normally completed response', async () => {
    let signal: AbortSignal | undefined;
    const service = await fixture(async function* (_body, requestSignal) {
      signal = requestSignal;
      yield { model: { contextWindow: 2048, maxTokens: 1024, input: ['text'] } };
    });
    await (
      await fetch(`${service.url}/stream`, { method: 'POST', headers: service.headers, body: JSON.stringify(request) })
    ).text();
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(signal?.aborted).toBe(false);
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
      await expect(startSharedService({ directory: service.directory, loadBackend: vi.fn() })).rejects.toMatchObject({
        code: 'EADDRINUSE',
      });
      await expect(
        startSharedService({ directory: service.directory, port: service.endpoint.port, loadBackend: vi.fn() }),
      ).rejects.toMatchObject({ code: 'EADDRINUSE' });
    } finally {
      release();
      await closing;
    }
  });

  it('releases the listener when backend cleanup fails', async () => {
    const service = await fixture(async function* () {
      yield { error: 'primed' };
    });
    await (
      await fetch(`${service.url}/stream`, { method: 'POST', headers: service.headers, body: JSON.stringify(request) })
    ).text();
    service.closeBackend.mockRejectedValue(new Error('native owner release failed'));
    // The expected rejection is already asserted below; afterEach can safely
    // retry the idempotent close without treating it as a second failure.
    cleanup.pop();
    await expect(service.close()).rejects.toThrow('native owner release failed');
    const replacement = await startSharedService({
      directory: service.directory,
      port: service.endpoint.port,
      loadBackend: vi.fn(),
    });
    cleanup.push(() => replacement.close());
  });
});
