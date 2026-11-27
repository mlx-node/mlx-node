import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { MessageChannel } from 'node:worker_threads';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import type { MainApiContext } from '../src/api/context.js';
import { dispatchMain } from '../src/api/dispatch.js';
import { catalogWithState } from '../src/catalog.js';
import type { InferenceEndpoint } from '../src/chat/agent/endpoint.js';
import { chatPaths } from '../src/chat/paths.js';
import { MemorySecretStore } from '../src/chat/providers/secrets.js';
import { ProvidersService } from '../src/chat/providers/service.js';
import { ChatService } from '../src/chat/service.js';
import type { RuntimeLive } from '../src/live.js';
import { createRpcClient } from '../src/rpc/client.js';
import { serveRuntimeOverPort, type RpcRuntime } from '../src/rpc/host.js';
import { bindEventTargetPort } from '../src/rpc/port.js';
import {
  SYSTEMONE_HISTORY_LIMIT,
  SYSTEMONE_HISTORY_MEDIA_LIMIT_BYTES,
  SystemOneService,
  sanitizeRequestForHistory,
  type SystemOneJobSnapshot,
} from '../src/systemone.js';
import { ggufHeader } from './helpers/configless-gguf-fixtures.js';

let rootDir: string;
let modelsDir: string;
let historyPath: string;
const servers: Server[] = [];

beforeEach(() => {
  rootDir = mkdtempSync(join(tmpdir(), 'dash-systemone-'));
  modelsDir = join(rootDir, 'models');
  historyPath = join(rootDir, 'desktop', 'systemone-history.json');
  mkdirSync(modelsDir, { recursive: true });
});

afterEach(async () => {
  for (const server of servers.splice(0)) {
    await new Promise<void>((resolve) => {
      server.closeAllConnections();
      server.close(() => resolve());
    });
  }
  rmSync(rootDir, { recursive: true, force: true });
});

function writeClairicModel(name: string): string {
  const dir = join(modelsDir, name);
  mkdirSync(dir, { recursive: true });
  writeFileSync(
    join(dir, 'config.json'),
    JSON.stringify({
      model_type: 'qwen3_5',
      vision_config: { depth: 24 },
      text_config: { max_position_embeddings: 65536 },
    }),
  );
  writeFileSync(join(dir, 'model.safetensors'), Buffer.alloc(32));
  writeFileSync(join(dir, 'clair_head_config.json'), '{}');
  writeFileSync(join(dir, 'clair_head.safetensors'), Buffer.alloc(16));
  return dir;
}

async function startFakeSidecar(
  handler: (req: {
    method: string;
    url: string;
    apiKey: string | undefined;
    body: string;
    signal: AbortSignal;
  }) => Promise<{
    status: number;
    body: unknown;
  }>,
): Promise<{ url: string; token: string }> {
  const token = 'test-secret-token';
  const server = createServer((req, res) => {
    const abort = new AbortController();
    const onClose = () => abort.abort();
    req.once('aborted', onClose);
    res.once('close', onClose);
    const chunks: Buffer[] = [];
    req.on('data', (chunk: Buffer) => chunks.push(chunk));
    req.on('end', () => {
      void handler({
        method: req.method ?? 'GET',
        url: req.url ?? '/',
        apiKey: typeof req.headers['x-api-key'] === 'string' ? req.headers['x-api-key'] : undefined,
        body: Buffer.concat(chunks).toString('utf8'),
        signal: abort.signal,
      }).then(
        (reply) => {
          if (abort.signal.aborted || res.destroyed) return;
          res.writeHead(reply.status, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(reply.body));
        },
        (err: unknown) => {
          if (abort.signal.aborted || res.destroyed) return;
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: { message: err instanceof Error ? err.message : String(err) } }));
        },
      );
    });
  });
  servers.push(server);
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const port = (server.address() as AddressInfo).port;
  return { url: `http://127.0.0.1:${port}`, token };
}

function liveWithModels(pid: number, residentNames: string[]): { readNow(): Promise<RuntimeLive> } {
  return {
    readNow: async (): Promise<RuntimeLive> => ({
      now: Date.now(),
      status: 'running',
      supervisor: null,
      processes: [
        {
          pid,
          startedAt: Date.now() - 1000,
          ts: Date.now(),
          intervalMs: 5000,
          host: 'desktop',
          version: '0.0.21',
          port: 8080,
          persistPagedCache: true,
          coldCapture: null,
          health: 'ok',
          mem: { active: 0, peak: 0, cache: 0, wiredLimit: 0, rss: 0 },
          work: { ms: { loading: 0, waiting: 0, prepare: 0, prefill: 0, decode: 0, running: 0 } },
          rates: { windowMs: 10_000, outputTps: null, prefillTps: null },
          loading: [],
          counters: { ok: 0, error: 0, cancelled: 0, outputTokens: 0, loads: 0, rejected: 0 },
          models: residentNames.map((name) => ({
            name,
            modelType: 'clairic',
            ctxWindow: 65536,
            paged: false,
            maxConcurrent: 1,
            loadedAt: Date.now() - 500,
            loadMs: 250,
            inFlight: 0,
            queued: 0,
            queueLimit: 16,
            pool: null,
            experts: null,
            images: null,
          })),
          turns: [],
          status: 'running',
          ageMs: 0,
        },
      ],
    }),
  };
}

describe('catalogWithState category propagation', () => {
  it('tags chat and systemone entries with their category', () => {
    const items = catalogWithState(modelsDir);
    const systemOne = items.filter((item) => item.category === 'systemone');
    expect(systemOne.map((item) => item.builds[0].hfRepo)).toEqual([
      'Brooooooklyn/Clairic-Flash-GGUF',
      'Brooooooklyn/Clairic-GGUF',
      'ggml-org/Clef-Flash-GGUF',
      'ggml-org/Clef-GGUF',
    ]);
    for (const item of items) {
      expect(item.category === 'chat' || item.category === 'systemone').toBe(true);
    }
  });
});

describe('System One models in the CLEF catalog folders', () => {
  it('lists the GGUF of each folder by its file name, as CLEF, with the folder in its path', async () => {
    const files = [
      ['clef-flash-gguf', 'Clef-Flash-Q8_0.gguf'],
      ['clef-gguf-q4_k_m', 'Clef-Q4_K_M.gguf'],
    ] as const;
    for (const [folder, file] of files) {
      mkdirSync(join(modelsDir, folder));
      writeFileSync(join(modelsDir, folder, file), ggufHeader('clef'));
    }
    const service = new SystemOneService({ modelsDir, readEndpoint: async () => null, historyPath });
    const { models } = await service.state();
    expect(models.map((model) => [model.id, model.modelType, model.variant, model.supportsImages, model.path])).toEqual(
      [
        ['Clef-Flash-Q8_0', 'clef', 'Q8_0', false, join(modelsDir, 'clef-flash-gguf', 'Clef-Flash-Q8_0.gguf')],
        ['Clef-Q4_K_M', 'clef', 'Q4_K_M', false, join(modelsDir, 'clef-gguf-q4_k_m', 'Clef-Q4_K_M.gguf')],
      ],
    );
  });
});

describe('ChatService decision model filtering', () => {
  it('filters clairic and clef models out of ChatHostState.models', async () => {
    const chatRoot = join(rootDir, 'chat');
    const paths = chatPaths(chatRoot);
    const providers = new ProvidersService({ paths, secrets: new MemorySecretStore(), workPath: () => '' });
    const endpoint: InferenceEndpoint = {
      version: 1,
      pid: process.pid,
      url: 'http://127.0.0.1:9999',
      token: 'tok',
      models: [
        {
          name: 'qwen3.8-27b',
          path: '/m/qwen',
          base: 'qwen3.8-27b',
          variant: 'MLX',
          format: 'mlx',
          modelType: 'qwen3_5',
          kind: 'chat',
        },
        {
          name: 'clairic-flash-4bit',
          path: '/m/clairic',
          base: 'clairic-flash-4bit',
          variant: '4-bit',
          format: 'mlx',
          modelType: 'clairic',
          kind: 'decision',
        },
        { name: 'clef-9b', path: '/m/clef', base: 'clef-9b', variant: 'MLX', format: 'mlx', modelType: 'clef' },
      ],
    };
    const chat = new ChatService({
      paths,
      homeDir: join(rootDir, 'home'),
      workAgentHome: join(rootDir, 'agent-home'),
      readEndpoint: async () => endpoint,
      modelSize: async () => 1024,
      resolveAgent: () => ({ launch: { failure: 'agent-missing' }, source: 'none', tag: null }),
      readSession: async () => null,
      providers,
      clientVersion: '0.0.0',
    });
    try {
      const state = await chat.state();
      expect(state.models.map((m) => m.id)).toEqual(['qwen3.8-27b']);
      expect(state.defaultModelId).toBe('qwen3.8-27b');
    } finally {
      await chat.close();
    }
  });
});

describe('SystemOneService', () => {
  it('reports state, runs extraction jobs, streams snapshots, persists history, and strips huge blobs', async () => {
    const clairicPath = writeClairicModel('clairic-flash-4bit');
    let seenApiKey: string | undefined;
    const sidecar = await startFakeSidecar(async (req) => {
      seenApiKey = req.apiKey;
      if (req.url === '/v1/models/unload') {
        return { status: 200, body: { unloaded: ['clairic-flash-4bit'] } };
      }
      return {
        status: 200,
        body: {
          model: 'clairic-flash-4bit',
          answers: { total: { value: 42.5, confidence: 0.99 } },
          truncated: false,
          usage: { input_tokens: 150, output_tokens: 12 },
        },
      };
    });

    const endpoint: InferenceEndpoint = {
      version: 1,
      pid: process.pid,
      url: sidecar.url,
      token: sidecar.token,
      models: [
        {
          name: 'clairic-flash-4bit',
          path: clairicPath,
          base: 'clairic-flash-4bit',
          variant: '4-bit',
          format: 'mlx',
          modelType: 'clairic',
          kind: 'decision',
        },
      ],
    };

    const service = new SystemOneService({
      modelsDir,
      readEndpoint: async () => endpoint,
      live: liveWithModels(process.pid, ['clairic-flash-4bit']),
      historyPath,
    });

    const initial = await service.state();
    expect(initial.sidecarRunning).toBe(true);
    expect(initial.activeJobId).toBeNull();
    expect(initial.models).toEqual([
      {
        id: 'clairic-flash-4bit',
        name: 'clairic-flash-4bit',
        modelType: 'clairic',
        format: 'mlx',
        variant: '4-bit',
        supportsImages: true,
        contextWindow: 65536,
        resident: true,
        path: clairicPath,
      },
    ]);

    const hugeBase64 = 'A'.repeat(SYSTEMONE_HISTORY_MEDIA_LIMIT_BYTES + 1024);
    const { jobId } = await service.run({
      model: 'clairic-flash-4bit',
      mode: 'extract',
      payload: {
        state: 'Invoice total: $42.50',
        images: [{ content_type: 'image/png', base64: hugeBase64 }],
        fields: { total: { type: 'number' } },
      },
    });

    const events: SystemOneJobSnapshot[] = [];
    const done = new Promise<SystemOneJobSnapshot>((resolve) => {
      const off = service.watch(jobId, (snap) => {
        events.push(snap);
        if (snap.status !== 'running') {
          off();
          resolve(snap);
        }
      });
    });

    const final = await done;
    expect(seenApiKey).toBe(sidecar.token);
    expect(final.status).toBe('done');
    expect(final.result).toMatchObject({ model: 'clairic-flash-4bit' });
    expect(service.job(jobId).status).toBe('done');

    await service.close();

    // Verify persisted history survives restart and huge base64 was stripped while text/fields survived.
    const rawSaved = JSON.parse(readFileSync(historyPath, 'utf8')) as Array<Record<string, unknown>>;
    expect(rawSaved).toHaveLength(1);
    expect(JSON.stringify(rawSaved)).not.toContain(hugeBase64);

    const reloaded = new SystemOneService({
      modelsDir,
      readEndpoint: async () => endpoint,
      historyPath,
    });
    const afterRestart = await reloaded.state();
    expect(afterRestart.history).toHaveLength(1);
    expect(afterRestart.history[0]).toMatchObject({
      id: jobId,
      model: 'clairic-flash-4bit',
      mode: 'extract',
      status: 'done',
      fieldCount: 1,
      inputTokens: 150,
      outputTokens: 12,
    });
    expect(afterRestart.history[0]!.request.state).toBe('Invoice total: $42.50');
    expect(afterRestart.history[0]!.request.fields).toEqual({ total: { type: 'number' } });

    const unloaded = await reloaded.unload();
    expect(unloaded).toEqual({ unloaded: ['clairic-flash-4bit'] });

    await reloaded.clearHistory();
    expect((await reloaded.state()).history).toEqual([]);
    await reloaded.close();
  });

  it('cancels an in-flight job and aborts the sidecar HTTP request immediately', async () => {
    let sidecarAborted = false;
    let startedResolve!: () => void;
    const started = new Promise<void>((resolve) => {
      startedResolve = resolve;
    });
    const sidecar = await startFakeSidecar(async (req) => {
      startedResolve();
      await new Promise<void>((resolve) => {
        if (req.signal.aborted) {
          sidecarAborted = true;
          resolve();
          return;
        }
        req.signal.addEventListener(
          'abort',
          () => {
            sidecarAborted = true;
            resolve();
          },
          { once: true },
        );
      });
      return { status: 200, body: {} };
    });

    const service = new SystemOneService({
      modelsDir,
      readEndpoint: async () => ({
        version: 1,
        pid: process.pid,
        url: sidecar.url,
        token: sidecar.token,
        models: [],
      }),
      historyPath,
    });

    const { jobId } = await service.run({
      model: 'clairic-flash-4bit',
      payload: { state: 'slow doc', fields: { summary: { type: 'text' } } },
    });

    const settled = new Promise<SystemOneJobSnapshot>((resolve) => {
      const off = service.watch(jobId, (snap) => {
        if (snap.status !== 'running') {
          off();
          resolve(snap);
        }
      });
    });

    await started;
    expect(service.cancel(jobId)).toEqual({ ok: true });
    const snap = await settled;
    expect(snap.status).toBe('cancelled');
    // Wait a tick for the TCP FIN/RST to reach the fake sidecar.
    await new Promise((resolve) => setTimeout(resolve, 25));
    expect(sidecarAborted).toBe(true);
    await service.close();
  });

  it('dispatches all /api/systemone/* routes and streams job updates over RPC MessagePort', async () => {
    const sidecar = await startFakeSidecar(async (req) => {
      if (req.url === '/v1/models/unload') return { status: 200, body: { unloaded: ['clairic-flash-4bit'] } };
      return {
        status: 200,
        body: {
          model: 'clairic-flash-4bit',
          answers: { ok: { value: true } },
          usage: { input_tokens: 10, output_tokens: 2 },
        },
      };
    });
    const endpoint: InferenceEndpoint = {
      version: 1,
      pid: process.pid,
      url: sidecar.url,
      token: sidecar.token,
      models: [
        {
          name: 'clairic-flash-4bit',
          path: join(modelsDir, 'clairic-flash-4bit'),
          base: 'clairic-flash-4bit',
          variant: '4-bit',
          format: 'mlx',
          modelType: 'clairic',
          kind: 'decision',
        },
      ],
    };
    const systemone = new SystemOneService({
      modelsDir,
      readEndpoint: async () => endpoint,
      historyPath,
    });
    const ctx = {
      modelsDir,
      tracesDir: '',
      runtimeDir: '',
      cacheRoot: undefined,
      prefixCacheRoot: undefined,
      systemone,
    } as unknown as MainApiContext;

    const stateRes = await dispatchMain(ctx, {
      method: 'GET',
      pathname: '/api/systemone/state',
      query: new URLSearchParams(),
    });
    expect(stateRes).toMatchObject({ ok: true, status: 200, body: { sidecarRunning: true } });

    const runRes = await dispatchMain(ctx, {
      method: 'POST',
      pathname: '/api/systemone/run',
      query: new URLSearchParams(),
      body: {
        model: 'clairic-flash-4bit',
        payload: { state: 'yes', fields: { ok: { type: 'boolean' } } },
      },
    });
    expect(runRes.ok).toBe(true);
    if (!runRes.ok) throw new Error('expected ok');
    expect(runRes.status).toBe(202);
    const { jobId } = runRes.body as { jobId: string };

    const { port1, port2 } = new MessageChannel();
    port1.unref();
    port2.unref();
    const rpcRuntime: RpcRuntime = {
      call: async (c) =>
        dispatchMain(ctx, { method: c.method, pathname: c.path, query: new URLSearchParams(), body: c.body }),
      subscribe: () => () => {},
      watchLive: () => () => {},
      watchSystemOne: (id, listener) => systemone.watch(id, listener),
    };
    const dispose = serveRuntimeOverPort(rpcRuntime, bindEventTargetPort(port2));
    const client = createRpcClient(bindEventTargetPort(port1));
    try {
      const terminal = await new Promise<SystemOneJobSnapshot>((resolve) => {
        const off = client.watchSystemOne(jobId, (snap) => {
          if (snap.status !== 'running') {
            off();
            resolve(snap);
          }
        });
      });
      expect(terminal.status).toBe('done');

      const jobRes = await client.call({ method: 'GET', path: `/api/systemone/jobs/${jobId}` });
      expect(jobRes).toMatchObject({ ok: true, status: 200, body: { id: jobId, status: 'done' } });

      const unloadRes = await client.call({ method: 'POST', path: '/api/systemone/unload' });
      expect(unloadRes).toMatchObject({ ok: true, status: 200, body: { unloaded: ['clairic-flash-4bit'] } });

      const clearRes = await client.call({ method: 'DELETE', path: '/api/systemone/history' });
      expect(clearRes).toMatchObject({ ok: true, status: 200, body: { ok: true } });
    } finally {
      client.close();
      dispose();
      await systemone.close();
    }
  });

  it('sanitizeRequestForHistory preserves small payloads and strips oversized media strings', () => {
    const small = 'data:image/png;base64,AAAA';
    const large = `data:image/png;base64,${'B'.repeat(SYSTEMONE_HISTORY_MEDIA_LIMIT_BYTES + 10)}`;
    const cleaned = sanitizeRequestForHistory({
      state: 'hello',
      images: [small, large],
      documents: [{ content_type: 'application/pdf', base64: 'C'.repeat(SYSTEMONE_HISTORY_MEDIA_LIMIT_BYTES + 10) }],
      fields: { x: { type: 'text' } },
    });
    expect(cleaned.state).toBe('hello');
    expect(cleaned.fields).toEqual({ x: { type: 'text' } });
    expect((cleaned.images as unknown[])[0]).toBe(small);
    expect(String((cleaned.images as unknown[])[1])).toContain('omitted');
    expect((cleaned.documents as Array<Record<string, unknown>>)[0]).toMatchObject({
      content_type: 'application/pdf',
      omitted: true,
    });
    expect(SYSTEMONE_HISTORY_LIMIT).toBe(30);
  });
});
