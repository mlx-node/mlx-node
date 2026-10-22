import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import type { RerankerModel } from '../../packages/rerank/src/types.js';
import { withCapabilityModels, type CapabilityAdapter } from '../../packages/server/src/host/capability-models.js';
import { RerankRegistry } from '../../packages/server/src/rerank-registry.js';
import { resolveRerankLimits } from '../../packages/server/src/rerank-service.js';
import { createServer } from '../../packages/server/src/server.js';
function model(): RerankerModel & { dispose: ReturnType<typeof vi.fn<RerankerModel['dispose']>> } {
  return {
    info: {
      family: 'jina_reranker',
      execution: 'listwise',
      scoreType: 'cosine',
      profile: 'jina-v3',
      contextWindow: 131072,
      quantized: false,
      weightFormats: ['dense'],
      policy: {
        maxBlockTokens: 32768,
        maxDocumentsPerBlock: 125,
        prefillChunkSize: 512,
        maxQueryTokens: 512,
        maxDocumentTokens: 2048,
        maxDocuments: 1024,
        maxInputTokens: 1000000,
      },
    },
    rerank: vi.fn<RerankerModel['rerank']>(async (request) => ({
      results: request.documents.map((_, index) => ({ index, relevanceScore: 1 - index * 0.1 })),
      usage: { inputTokens: 10, promptTokens: 50 },
      meta: {
        profile: 'jina-v3',
        blocks: 1,
        processedDocuments: request.documents.length,
        maxBlockTokens: 32768,
        maxDocumentsPerBlock: 125,
        truncatedQuery: false,
        truncatedDocumentIndices: [],
      },
    })),
    dispose: vi.fn(async () => {}),
  };
}
const running: Awaited<ReturnType<typeof createServer>>[] = [];
afterEach(async () => {
  await Promise.all(running.splice(0).map((s) => s.close()));
});
async function server(options: Parameters<typeof createServer>[0] = {}) {
  const s = await createServer({ port: 0, disableStore: true, ...options });
  running.push(s);
  return s;
}
function url(s: Awaited<ReturnType<typeof createServer>>) {
  const a = s.server.address();
  if (!a || typeof a === 'string') throw Error('not listening');
  return `http://127.0.0.1:${a.port}`;
}
async function post(s: Awaited<ReturnType<typeof createServer>>, body: unknown, path = '/v1/rerank') {
  return fetch(url(s) + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}
describe('rerank service', () => {
  it('rejects deadlines that Node would silently reduce to one millisecond', () => {
    expect(resolveRerankLimits({ timeoutMs: 2 ** 31 - 1 }).timeoutMs).toBe(2 ** 31 - 1);
    expect(() => resolveRerankLimits({ timeoutMs: 2 ** 31 })).toThrow('timer range');
  });
  it('serves registered models, keeps indices and advertises capability', async () => {
    const s = await server();
    const m = model();
    s.registry.rerank.register('rank', m);
    const response = await post(s, { model: 'rank', query: 'q', documents: ['a', { text: 'b' }] });
    expect(response.status).toBe(200);
    expect((await response.json()).results).toEqual([
      { index: 0, relevance_score: 1 },
      { index: 1, relevance_score: 0.9 },
    ]);
    const models = await (await fetch(url(s) + '/v1/models')).json();
    expect(models.data).toContainEqual(expect.objectContaining({ id: 'rank', task: 'rerank' }));
    expect((await post(s, { model: 'rank', messages: [] }, '/v1/messages')).status).toBe(400);
    expect((await post(s, { model: 'rank', input: 'hello', voice: 'v' }, '/v1/audio/speech')).status).toBe(400);
    expect(s.registry.rerank.pending).toBe(0);
  });
  it('validates before loading and rejects unknown model aliases', async () => {
    const resolve = vi.fn(async () => {});
    const s = await server({
      resolveModel: resolve,
      listModels: () => [{ id: 'chat', object: 'model', created: 0, owned_by: 'test' }],
    });
    expect((await post(s, { model: 'chat', query: 'q', documents: [] })).status).toBe(400);
    expect((await post(s, { model: 'unknown', query: 'q', documents: [] })).status).toBe(404);
    expect((await post(s, { model: 'chat', query: 'q', documents: [], top_n: -1 })).status).toBe(400);
    expect(resolve).not.toHaveBeenCalled();
  });
  it('authenticates, bounds body size, and rejects methods', async () => {
    const s = await server({ authToken: 'secret', rerank: { maxBodyBytes: 32 } });
    expect((await post(s, {})).status).toBe(401);
    const r = await fetch(url(s) + '/v1/rerank', {
      method: 'POST',
      headers: { Authorization: 'Bearer secret' },
      body: 'x'.repeat(100),
    });
    expect(r.status).toBe(413);
    expect((await fetch(url(s) + '/v1/rerank', { headers: { Authorization: 'Bearer secret' } })).status).toBe(405);
  });
  it('holds inference admission after deadline until the model settles', async () => {
    const s = await server({ rerank: { timeoutMs: 30 } });
    const m = model();
    let finish!: () => void;
    let entered!: () => void;
    const started = new Promise<void>((r) => {
      entered = r;
    });
    const stopped = new Promise<void>((r) => {
      finish = r;
    });
    m.rerank = async (request) => {
      entered();
      await stopped;
      request.signal?.throwIfAborted();
      throw Error('expected abort');
    };
    s.registry.rerank.register('rank', m);
    const response = post(s, { model: 'rank', query: 'q', documents: ['a'] });
    await started;
    await new Promise((r) => setTimeout(r, 50));
    expect(s.registry.rerank.pending).toBe(1);
    expect(m.dispose).not.toHaveBeenCalled();
    finish();
    expect((await response).status).toBe(504);
    expect(s.registry.rerank.pending).toBe(0);
  });
});
describe('rerank admission and residency', () => {
  it('shares queue capacity and disposal across aliases', async () => {
    const registry = new RerankRegistry(1);
    const m = model();
    registry.register('a', m);
    registry.register('b', m);
    const first = registry.acquire('a')!;
    const second = registry.acquire('b')!;
    expect(() => registry.acquire('a')).toThrow('queue');
    first.release();
    second.release();
    await registry.disposeAll();
    expect(m.dispose).toHaveBeenCalledTimes(1);
  });
  it('cancelled waiters cannot overtake predecessors', async () => {
    const registry = new RerankRegistry(3);
    registry.register('a', model());
    let finish!: () => void;
    const running = new Promise<void>((r) => {
      finish = r;
    });
    const events: string[] = [];
    const a = registry.acquire('a')!,
      b = registry.acquire('a')!,
      c = registry.acquire('a')!;
    const first = a.run(async () => {
      events.push('a');
      await running;
    });
    const abort = new AbortController();
    const second = b.run(async () => {
      events.push('b');
    }, abort.signal);
    const failed = expect(second).rejects.toThrow();
    abort.abort(new Error('cancel'));
    const third = c.run(async () => {
      events.push('c');
    });
    await failed;
    await Promise.resolve();
    expect(events).toEqual(['a']);
    finish();
    await Promise.all([first, third]);
    expect(events).toEqual(['a', 'c']);
    a.release();
    b.release();
    c.release();
  });
  it('single residency unloads every other capability in either direction', async () => {
    const resident = new Set<string>();
    const adapters: CapabilityAdapter[] = (['chat', 'asr', 'tts', 'rerank'] as const).map((kind) => ({
      kind,
      listModels: () => [{ id: kind, object: 'model', created: 0, owned_by: 'test' }],
      hasResident: () => resident.has(kind),
      resolveModel: async () => {
        resident.add(kind);
      },
      disposeAll: async () => {
        resident.delete(kind);
      },
    }));
    const single = withCapabilityModels(adapters, 'single');
    for (const kind of ['chat', 'rerank', 'asr', 'rerank', 'tts', 'chat']) {
      await single.resolveModel(kind);
      expect([...resident]).toEqual([kind]);
    }
    resident.clear();
    const multi = withCapabilityModels(adapters, 'capability');
    for (const kind of ['chat', 'asr', 'tts', 'rerank']) await multi.resolveModel(kind);
    expect(resident.size).toBe(4);
  });
});

it('bounds cold-load fan-in and releases every transferred admission', async () => {
  let release!: () => void;
  const loading = new Promise<void>((r) => {
    release = r;
  });
  let entered!: () => void;
  const started = new Promise<void>((r) => {
    entered = r;
  });
  const m = model();
  const s = await server({
    maxQueueDepthPerModel: 1,
    listModels: () => [{ id: 'rank', object: 'model', created: 0, owned_by: 'test', task: 'rerank' }],
    resolveModel: async () => {
      entered();
      await loading;
      s.registry.rerank.register('rank', m);
    },
  });
  const one = post(s, { model: 'rank', query: 'q', documents: ['a'] });
  await started;
  const two = post(s, { model: 'rank', query: 'q', documents: ['a'] });
  await new Promise((r) => setTimeout(r, 20));
  const three = await post(s, { model: 'rank', query: 'q', documents: ['a'] });
  expect(three.status).toBe(429);
  release();
  expect((await one).status).toBe(200);
  expect((await two).status).toBe(200);
  expect(s.registry.rerank.pending).toBe(0);
});

it('returns retryable unavailability when another cold load replaces the requested model', async () => {
  const { promise: entered, resolve: enter } = Promise.withResolvers<void>();
  const { promise: loading, resolve: release } = Promise.withResolvers<void>();
  const loaded = new Map<string, ReturnType<typeof model>>();
  const ranked: string[] = [];
  const s = await server({
    listModels: () => ['a', 'b'].map((id) => ({ id, object: 'model', created: 0, owned_by: 'test', task: 'rerank' })),
    resolveModel: async (name) => {
      if (s.registry.rerank.get(name)) return;
      await s.registry.rerank.disposeAll();
      if (name === 'a') {
        enter();
        await loading;
      }
      const next = model();
      const rerank = next.rerank.bind(next);
      next.rerank = async (request) => {
        ranked.push(name);
        return rerank(request);
      };
      loaded.set(name, next);
      s.registry.rerank.register(name, next);
    },
  });
  const first = post(s, { model: 'a', query: 'q', documents: ['a'] });
  await entered;
  const second = post(s, { model: 'b', query: 'q', documents: ['b'] });
  try {
    await vi.waitFor(() => expect(s.modelWork.waitingWriters).toBe(1));
  } finally {
    release();
  }
  const [a, b] = await Promise.all([first, second]);
  expect(a.status).toBe(503);
  expect(a.headers.get('Retry-After')).toBe('1');
  expect((await a.json()).error.code).toBe('model_unavailable');
  expect(b.status).toBe(200);
  expect(loaded.get('a')!.dispose).toHaveBeenCalledTimes(1);
  expect(ranked).toEqual(['b']);
  expect(s.registry.rerank.pending).toBe(0);
  expect(s.modelWork.requestLoadAdmissionCount).toBe(0);
  expect((await post(s, { model: 'a', query: 'q', documents: ['a'] })).status).toBe(200);
  expect((await post(s, { model: 'unknown', query: 'q', documents: [] })).status).toBe(404);
});

it('shutdown waits for cancelled native work before disposing the model', async () => {
  const s = await server();
  const m = model();
  let release!: () => void;
  const done = new Promise<void>((r) => {
    release = r;
  });
  let entered!: () => void;
  const started = new Promise<void>((r) => {
    entered = r;
  });
  let cancelled = false;
  m.rerank = async (request) => {
    request.signal!.addEventListener('abort', () => {
      cancelled = true;
    });
    entered();
    await done;
    request.signal!.throwIfAborted();
    throw Error('expected cancellation');
  };
  s.registry.rerank.register('rank', m);
  const request = post(s, { model: 'rank', query: 'q', documents: ['a'] });
  await started;
  const closed = s.close();
  await Promise.resolve();
  expect(cancelled).toBe(true);
  expect(m.dispose).not.toHaveBeenCalled();
  release();
  expect((await request).status).toBe(503);
  await closed;
  expect(m.dispose).toHaveBeenCalledTimes(1);
});

it('releases failed cold-load admissions and permits a successful retry', async () => {
  let attempt = 0;
  const m = model();
  const s = await server({
    maxQueueDepthPerModel: 1,
    listModels: () => [{ id: 'rank', object: 'model', created: 0, owned_by: 'test', task: 'rerank' }],
    resolveModel: async () => {
      if (++attempt === 1) throw Error('rerank_invalid_model: missing weights');
      s.registry.rerank.register('rank', m);
    },
  });
  const failed = await post(s, { model: 'rank', query: 'q', documents: ['a'] });
  expect(failed.status).toBe(500);
  expect((await failed.json()).error.code).toBe('invalid_model');
  expect(s.registry.rerank.list()).toEqual([]);
  expect((await post(s, { model: 'rank', query: 'q', documents: ['a'] })).status).toBe(200);
  expect(s.registry.rerank.pending).toBe(0);
});
it('keeps HTTP policy authoritative even with truncate and answers CORS preflight', async () => {
  const s = await server({ rerank: { maxDocuments: 1 } });
  const m = model();
  m.rerank = async () => {
    throw new Error('input limit must reject before inference');
  };
  s.registry.rerank.register('rank', m);
  expect((await post(s, { model: 'rank', query: 'q', documents: ['a', 'b'], truncate: true })).status).toBe(413);
  const response = await fetch(url(s) + '/v1/rerank', {
    method: 'OPTIONS',
    headers: { Origin: 'http://localhost:1234', 'Access-Control-Request-Method': 'POST' },
  });
  expect(response.status).toBe(204);
  expect(response.headers.get('Access-Control-Allow-Origin')).toBe('*');
});
it('client disconnect cancels work but keeps its lease until native completion', async () => {
  const { request } = await import('node:http');
  const s = await server();
  const m = model();
  let enter!: () => void;
  const entered = new Promise<void>((r) => {
    enter = r;
  });
  let abort!: () => void;
  const aborted = new Promise<void>((r) => {
    abort = r;
  });
  let finish!: () => void;
  const finished = new Promise<void>((r) => {
    finish = r;
  });
  m.rerank = async (r) => {
    enter();
    r.signal!.addEventListener('abort', abort, { once: true });
    await finished;
    r.signal!.throwIfAborted();
    throw Error('must abort');
  };
  s.registry.rerank.register('rank', m);
  const client = request(url(s) + '/v1/rerank', { method: 'POST' });
  client.on('error', () => {});
  client.end(JSON.stringify({ model: 'rank', query: 'q', documents: ['a'] }));
  await entered;
  client.destroy();
  await aborted;
  expect(s.registry.rerank.pending).toBe(1);
  expect(m.dispose).not.toHaveBeenCalled();
  finish();
  await s.close();
  expect(m.dispose).toHaveBeenCalledTimes(1);
  expect(s.registry.rerank.pending).toBe(0);
});
it('shutdown drains a pending load and disposes its late resident', async () => {
  let enter!: () => void;
  const entered = new Promise<void>((r) => {
    enter = r;
  });
  let finish!: () => void;
  const loading = new Promise<void>((r) => {
    finish = r;
  });
  const m = model();
  const s = await server({
    listModels: () => [{ id: 'rank', object: 'model', created: 0, owned_by: 'test', task: 'rerank' }],
    resolveModel: async () => {
      enter();
      await loading;
      s.registry.rerank.register('rank', m);
    },
  });
  const response = post(s, { model: 'rank', query: 'q', documents: ['a'] });
  await entered;
  let closed = false;
  const closing = s.close().then(() => {
    closed = true;
  });
  await Promise.resolve();
  expect(closed).toBe(false);
  finish();
  expect((await response).status).toBe(503);
  await closing;
  expect(m.dispose).toHaveBeenCalledTimes(1);
  expect(s.registry.rerank.list()).toEqual([]);
});
