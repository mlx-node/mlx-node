import {
  Agent,
  createServer,
  request as httpRequest,
  type IncomingMessage,
  type Server,
  type ServerResponse,
} from 'node:http';
import type { AddressInfo } from 'node:net';

import type { ClairicResult, DecisionModel, ExtractionModel } from '@mlx-node/lm';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

vi.mock('@mlx-node/core', () => ({ __internal__: { clearCache: () => {} } }));

import { formatClefResult, validateClefRequest } from '../../packages/lm/src/clef-types.js';
import { DecisionQueueFullError } from '../../packages/server/src/decision-registry.js';
import { handleSystemOne } from '../../packages/server/src/endpoints/systemone.js';
import { createHandler } from '../../packages/server/src/handler.js';
import { makeSwapController } from '../../packages/server/src/host/swap.js';
import type { MetricsTraceInput } from '../../packages/server/src/metrics-trace.js';
import { ModelWorkCoordinator } from '../../packages/server/src/model-work-coordinator.js';
import { ModelRegistry } from '../../packages/server/src/registry.js';

const result = {
  answers: { flag: { type: 'noul' as const, noul: 0.8 } },
  usage: { input_tokens: 123, output_tokens: 0 as const },
};
const request = { model: 'flash', state: 'hello', questions: { flag: { type: 'noul' } } };
const servers: Server[] = [];
afterEach(async () => {
  await Promise.all(
    servers.splice(0).map(
      (s) =>
        new Promise<void>((resolve) => {
          s.closeAllConnections();
          s.close(() => resolve());
        }),
    ),
  );
});
async function start(registry: ModelRegistry) {
  const server = createServer(createHandler(registry, { store: null }));
  servers.push(server);
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  return `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
}
function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}
function model(): DecisionModel & { decideRaw: ReturnType<typeof vi.fn> } {
  return { kind: 'decision', decideRaw: vi.fn().mockResolvedValue(result) };
}

describe('CLEF System One', () => {
  it('serves raw question order, aliases, request IDs and both model-list contracts', async () => {
    const registry = new ModelRegistry();
    const native = model();
    registry.registerDecision('flash', native);
    registry.registerDecision('alias', native);
    expect(registry.getSessionRegistry('flash')).toBeUndefined();
    const base = await start(registry);
    const raw = '{"model":"alias","state":"x","questions":{"10":{"type":"noul"},"2":{"type":"noul"}}}';
    const response = await fetch(`${base}/v1/systemone`, { method: 'POST', body: raw });
    expect(response.status).toBe(200);
    expect(response.headers.get('x-typesafe-request-id')).toBeTruthy();
    expect(await response.json()).toEqual({ model: 'flash', ...result });
    expect(native.decideRaw).toHaveBeenCalledWith(raw, expect.objectContaining({ signal: expect.any(AbortSignal) }));
    const listing = await (await fetch(`${base}/v1/models`)).json();
    expect(listing.data.map((v: { id: string }) => v.id)).toEqual(['flash', 'alias']);
    expect(listing.models.map((v: { name: string }) => v.name)).toEqual(['flash', 'alias']);
    expect((await fetch(`${base}/v1/systemone`)).status).toBe(405);
  });
  it('rejects malformed and media requests before invoking native inference', async () => {
    const registry = new ModelRegistry();
    const native = model();
    registry.registerDecision('flash', native);
    const base = await start(registry);
    for (const body of [
      { ...request, images: ['ignored'] },
      { ...request, questions: {} },
      { ...request, questions: { x: { type: 'score', criteria: ['one'] } } },
      { ...request, model: 4 },
    ]) {
      expect((await fetch(`${base}/v1/systemone`, { method: 'POST', body: JSON.stringify(body) })).status).toBe(400);
    }
    expect(native.decideRaw).not.toHaveBeenCalled();
  });
  it('shares admission across aliases and re-registration and releases errors', async () => {
    const registry = new ModelRegistry({ maxQueueDepth: 1 });
    const native = model();
    registry.registerDecision('flash', native);
    registry.registerDecision('alias', native);
    const first = registry.decisions.acquire('flash')!;
    const second = registry.decisions.acquire('alias')!;
    expect(() => registry.decisions.acquire('flash')).toThrow(DecisionQueueFullError);
    const gate = deferred();
    const order: string[] = [];
    const a = first.run(async () => {
      order.push('a');
      await gate.promise;
      throw new Error('failure');
    });
    const b = second.run(async () => {
      order.push('b');
    });
    registry.unregister('flash');
    registry.unregister('alias');
    registry.registerDecision('flash', native);
    expect(() => registry.decisions.acquire('flash')).toThrow(DecisionQueueFullError);
    await Promise.resolve();
    expect(order).toEqual(['a']);
    gate.resolve();
    await expect(a).rejects.toThrow('failure');
    first.release();
    await b;
    second.release();
    expect(order).toEqual(['a', 'b']);
    const third = registry.decisions.acquire('flash')!;
    third.release();
  });
  it('transfers cold reservations into the same bounded decision budget', async () => {
    const registry = new ModelRegistry({ maxQueueDepth: 1 });
    const coordinator = new ModelWorkCoordinator(1);
    registry.setModelLoadAdmissionCoordinator(coordinator);
    const coldA = coordinator.beginRequestLoadAdmission('flash');
    const coldB = coordinator.beginRequestLoadAdmission('alias');
    const native = model();
    registry.registerDecision('flash', native);
    registry.registerDecision('alias', native);
    expect(() => registry.decisions.acquire('flash')).toThrow(DecisionQueueFullError);
    const a = registry.decisions.acquire('flash', coldA.transferToResident(registry.decisions.lane('flash')!))!;
    const b = registry.decisions.acquire('alias', coldB.transferToResident(registry.decisions.lane('alias')!))!;
    await a.run(async () => {});
    a.release();
    coldA.release();
    await b.run(async () => {});
    b.release();
    coldB.release();
    const fresh = registry.decisions.acquire('flash')!;
    fresh.release();
  });
  it('keeps model loads behind active decisions and detects replaced bindings', async () => {
    const registry = new ModelRegistry();
    registry.registerDecision('flash', model());
    const lease = registry.decisions.acquire('flash')!;
    const coordinator = new ModelWorkCoordinator(1);
    const gate = deferred();
    const started = deferred();
    const swap = vi.fn();
    const run = coordinator.withInference(() =>
      lease.run(async () => {
        started.resolve();
        await gate.promise;
      }),
    );
    await started.promise;
    const load = coordinator.withModelLoad(() => {
      swap();
      registry.registerDecision('flash', model());
    });
    await Promise.resolve();
    expect(swap).not.toHaveBeenCalled();
    gate.resolve();
    await run;
    lease.release();
    await load;
    expect(lease.current()).toBe(false);
  });
  it('an aborted waiter releases its read-queue slot instead of parking behind a writer', async () => {
    const coordinator = new ModelWorkCoordinator(1);
    const gate = deferred();
    const writerDone = deferred();
    const writer = coordinator.withModelLoad(async () => {
      await gate.promise;
      writerDone.resolve();
    });
    const abort = new AbortController();
    const ran = vi.fn();
    const parked = coordinator.withInference(ran, abort.signal);
    abort.abort();
    await expect(parked).rejects.toThrow();
    expect(ran).not.toHaveBeenCalled();
    // The aborted reader left the queue: a live reader still waits on the
    // writer, but proceeds the moment the writer finishes.
    const live = vi.fn();
    const liveParked = coordinator.withInference(live);
    gate.resolve();
    await Promise.all([writer, liveParked]);
    expect(live).toHaveBeenCalled();
    // Aborted queued writers dequeue too and never run their load.
    const writerGate = deferred();
    const held = coordinator.withModelLoad(() => writerGate.promise);
    const abort2 = new AbortController();
    const deadLoad = vi.fn();
    const dead = coordinator.withModelLoad(deadLoad, 'dead', abort2.signal);
    abort2.abort();
    await expect(dead).rejects.toThrow();
    writerGate.resolve();
    await held;
    expect(deadLoad).not.toHaveBeenCalled();
  });
  it('does not execute an aborted queued request', async () => {
    const registry = new ModelRegistry();
    registry.registerDecision('flash', model());
    const a = registry.decisions.acquire('flash')!;
    const b = registry.decisions.acquire('flash')!;
    const gate = deferred();
    const fn = vi.fn();
    const abort = new AbortController();
    const first = a.run(() => gate.promise);
    const second = b.run(fn, abort.signal);
    abort.abort();
    gate.resolve();
    await first;
    await expect(second).rejects.toThrow();
    a.release();
    b.release();
    expect(fn).not.toHaveBeenCalled();
  });
  it('loads decisions through the host without allocating chat sessions', async () => {
    const registry = new ModelRegistry();
    const native = model();
    const controller = makeSwapController(
      [
        {
          name: 'flash',
          path: '/flash',
          modelType: 'clef',
          preset: { sampling: {}, maxOutputTokens: 0 },
          contextWindow: 16384,
          supportsImages: false,
          base: 'flash',
          variant: 'MLX',
          format: 'mlx',
        },
      ],
      registry,
      vi.fn().mockResolvedValue(native),
    );
    await controller.resolveModel('flash');
    await controller.resolveModel('alias');
    expect(registry.decisions.get('alias')).toBe(native);
    expect(registry.listSessionRegistries()).toEqual([]);
  });
  it('uses Jev confidence and protects special object keys', () => {
    const body = JSON.parse(
      '{"state":"x","questions":{"__proto__":{"type":"choice","criteria":{"a":null,"b":null}},"score":{"type":"score","criteria":["low","mid","high"]},"n":{"type":"noul"}}}',
    );
    const formatted = formatClefResult(
      {
        input_tokens: 4,
        questions: [
          { id: '__proto__', type: 1, options: ['a', 'b'], probabilities: [0.8, 0.2] },
          { id: 'score', type: 2, options: ['0', '1', '2'], probabilities: [0.1, 0.1, 0.8] },
          { id: 'n', type: 0, options: ['true', 'false'], probabilities: [0.3, 0.7] },
        ],
      },
      body,
    );
    const choice = formatted.answers.__proto__;
    if (choice.type !== 'choice') throw new Error('wrong type');
    expect(choice.confidence).toBeCloseTo(0.6);
    const score = formatted.answers.score;
    if (score.type !== 'score') throw new Error('wrong type');
    expect(score.score).toBeCloseTo(1.7);
    expect(score.confidence).toBeCloseTo(0.55);
    expect(formatted.answers.n).toEqual({ type: 'noul', noul: 0.3 });
    expect(() => validateClefRequest({ ...request, questions: { x: { type: 'made-up' } } })).toThrow();
  });
});

const clairicResult: ClairicResult = {
  model: 'clairic-flash',
  answers: {
    invoice_total: { value: 42.5, valid: true, complete: true, confidence: 0.99, p_null: 0.01, source: 'input' },
  },
  truncated: false,
  usage: {
    state_tokens: 100,
    image_tokens: 0,
    schema_tokens: 24,
    input_tokens: 124,
    output_tokens: 8,
    passes: 1,
  },
};

function clairicModel(): ExtractionModel & { extractRaw: ReturnType<typeof vi.fn> } {
  return { kind: 'decision', extractRaw: vi.fn().mockResolvedValue(clairicResult) };
}

describe('Clairic System One', () => {
  it('routes fields payloads to extractRaw with untouched raw JSON and request ID header', async () => {
    const registry = new ModelRegistry();
    const native = clairicModel();
    registry.registerDecision('clairic-flash', native);
    const base = await start(registry);
    const raw =
      '{"model":"clairic-flash","state":"Total: $42.50","fields":{"invoice_total":{"type":"number","decimals":2}}}';

    const response = await fetch(`${base}/v1/systemone`, { method: 'POST', body: raw });
    expect(response.status).toBe(200);
    expect(response.headers.get('x-typesafe-request-id')).toBeTruthy();
    expect(await response.json()).toEqual(clairicResult);
    expect(native.extractRaw).toHaveBeenCalledWith(raw, expect.objectContaining({ signal: expect.any(AbortSignal) }));
  });

  it('rejects requests with both questions and fields, neither, or non-object fields', async () => {
    const registry = new ModelRegistry();
    const native = clairicModel();
    registry.registerDecision('clairic-flash', native);
    const base = await start(registry);

    for (const body of [
      { model: 'clairic-flash', state: 'x', questions: { q: { type: 'noul' } }, fields: { a: { type: 'text' } } },
      { model: 'clairic-flash', state: 'x' },
      { model: 'clairic-flash', state: 'x', fields: null },
      { model: 'clairic-flash', state: 'x', fields: ['not-an-object'] },
      { state: 'x', fields: { a: { type: 'text' } } },
    ]) {
      const response = await fetch(`${base}/v1/systemone`, { method: 'POST', body: JSON.stringify(body) });
      expect(response.status).toBe(400);
    }
    expect(native.extractRaw).not.toHaveBeenCalled();
  });

  it('rejects capability mismatches between CLEF and Clairic models with 400', async () => {
    const registry = new ModelRegistry();
    const clef = model();
    const clairic = clairicModel();
    registry.registerDecision('clef-only', clef);
    registry.registerDecision('clairic-only', clairic);
    const base = await start(registry);

    const fieldsToClef = await fetch(`${base}/v1/systemone`, {
      method: 'POST',
      body: JSON.stringify({ model: 'clef-only', state: 'x', fields: { a: { type: 'text' } } }),
    });
    expect(fieldsToClef.status).toBe(400);
    expect(await fieldsToClef.json()).toMatchObject({
      error: { message: expect.stringContaining('does not support extraction') },
    });

    const questionsToClairic = await fetch(`${base}/v1/systemone`, {
      method: 'POST',
      body: JSON.stringify({ model: 'clairic-only', state: 'x', questions: { q: { type: 'noul' } } }),
    });
    expect(questionsToClairic.status).toBe(400);
    expect(await questionsToClairic.json()).toMatchObject({
      error: { message: expect.stringContaining('does not support decisions') },
    });

    expect(clef.decideRaw).not.toHaveBeenCalled();
    expect(clairic.extractRaw).not.toHaveBeenCalled();
  });

  it('maps Invalid Clairic request errors to 400 and DecisionQueueFullError to 429', async () => {
    const registry = new ModelRegistry({ maxQueueDepth: 0 });
    const gate = deferred();
    const started = deferred();
    const native: ExtractionModel = {
      kind: 'decision',
      extractRaw: vi
        .fn()
        .mockRejectedValueOnce(new Error('Invalid Clairic request: unknown field type'))
        .mockImplementationOnce(async () => {
          started.resolve();
          await gate.promise;
          return clairicResult;
        }),
    };
    registry.registerDecision('clairic-flash', native);
    const base = await start(registry);

    const bad = await fetch(`${base}/v1/systemone`, {
      method: 'POST',
      body: JSON.stringify({ model: 'clairic-flash', state: 'x', fields: { a: { type: 'bogus' } } }),
    });
    expect(bad.status).toBe(400);
    expect(await bad.json()).toMatchObject({
      error: { message: expect.stringContaining('Invalid Clairic request:') },
    });

    const payload = JSON.stringify({ model: 'clairic-flash', state: 'x', fields: { a: { type: 'text' } } });
    const inFlight = fetch(`${base}/v1/systemone`, { method: 'POST', body: payload });
    await started.promise;

    const rejected = await fetch(`${base}/v1/systemone`, { method: 'POST', body: payload });
    expect(rejected.status).toBe(429);

    gate.resolve();
    expect((await inFlight).status).toBe(200);
  });

  it('accepts /v1/systemone payloads larger than 10 MiB up to 16 MiB and rejects > 16 MiB', async () => {
    const registry = new ModelRegistry();
    const native = clairicModel();
    registry.registerDecision('clairic-flash', native);
    const base = await start(registry);

    const elevenMiB = JSON.stringify({
      model: 'clairic-flash',
      state: 'a'.repeat(11 * 1024 * 1024),
      fields: { summary: { type: 'text' } },
    });
    const accepted = await fetch(`${base}/v1/systemone`, { method: 'POST', body: elevenMiB });
    expect(accepted.status).toBe(200);
    expect(native.extractRaw).toHaveBeenCalledTimes(1);

    const seventeenMiB = JSON.stringify({
      model: 'clairic-flash',
      state: 'a'.repeat(17 * 1024 * 1024),
      fields: { summary: { type: 'text' } },
    });
    const tooLarge = await fetch(`${base}/v1/systemone`, { method: 'POST', body: seventeenMiB });
    expect(tooLarge.status).toBe(400);
    expect(await tooLarge.json()).toMatchObject({
      error: { message: 'Request body too large' },
    });
    expect(native.extractRaw).toHaveBeenCalledTimes(1);
  });

  // MUTATION: the Anthropic body cap back to 10 MiB -> the 12 MiB bodies get 'Request body too large'.
  it('reads a 12 MiB body on /v1/messages and /v1/messages/count_tokens', async () => {
    const base = await start(new ModelRegistry());
    const twelveMiB = JSON.stringify({ data: 'x'.repeat(12 * 1024 * 1024) });
    for (const route of ['/v1/messages', '/v1/messages/count_tokens']) {
      const response = await fetch(`${base}${route}`, { method: 'POST', body: twelveMiB });
      expect(response.status).toBe(400);
      expect(await response.json()).toMatchObject({ error: { message: 'Missing required field: model' } });
    }
  });

  it('closes keep-alive connections after oversized bodies on /v1/responses, /v1/messages, and /v1/messages/count_tokens', async () => {
    const registry = new ModelRegistry();
    const base = await start(registry);
    const agent = new Agent({ keepAlive: true, maxSockets: 1 });
    try {
      for (const [route, mebibytes] of [
        ['/v1/responses', 11],
        ['/v1/messages', 17],
        ['/v1/messages/count_tokens', 17],
      ] as const) {
        const oversized = JSON.stringify({ data: 'x'.repeat(mebibytes * 1024 * 1024) });
        const rejected = await new Promise<{ status: number; connection: string | undefined; body: unknown }>(
          (resolve, reject) => {
            let settled = false;
            const req = httpRequest(
              `${base}${route}`,
              {
                method: 'POST',
                agent,
                headers: { 'Content-Length': Buffer.byteLength(oversized) },
              },
              (res) => {
                let text = '';
                res.on('data', (chunk) => {
                  text += chunk;
                });
                res.on('error', (err) => {
                  if (!settled) {
                    settled = true;
                    reject(err);
                  }
                });
                res.on('end', () => {
                  if (!settled) {
                    settled = true;
                    resolve({
                      status: res.statusCode ?? 0,
                      connection: res.headers.connection,
                      body: JSON.parse(text),
                    });
                  }
                });
              },
            );
            req.on('error', (err) => {
              if (!settled) {
                settled = true;
                reject(err);
              }
            });
            req.setTimeout(5000, () => req.destroy(new Error(`oversized request timed out on ${route}`)));
            req.end(oversized);
          },
        );
        expect(rejected.status).toBe(400);
        expect(rejected.connection).toBe('close');
        expect(rejected.body).toMatchObject({ error: { message: 'Request body too large' } });

        // Follow-up request through the same keep-alive agent must not hang on unread body bytes.
        const followup = await new Promise<number>((resolve, reject) => {
          const req = httpRequest(`${base}/health`, { method: 'GET', agent }, (res) => {
            res.resume();
            res.on('end', () => resolve(res.statusCode ?? 0));
          });
          req.on('error', reject);
          req.setTimeout(2000, () => req.destroy(new Error(`follow-up request hung after ${route}`)));
          req.end();
        });
        expect(followup).toBe(200);
      }
    } finally {
      agent.destroy();
    }
  });

  it('includes resident decision models in registry.listResidentModels()', () => {
    const registry = new ModelRegistry();
    const native = clairicModel();
    registry.registerAny('clairic-flash', native, {
      identity: { modelId: 'clairic-flash', modelType: 'clairic', contextWindow: 65536 },
    });
    registry.registerAny('clairic-alias', native);
    const resident = registry.listResidentModels();
    expect(resident).toHaveLength(1);
    expect(resident[0]!.kind).toBe('decision');
    expect(resident[0]!.names).toEqual(['clairic-flash', 'clairic-alias']);
    expect(resident[0]!.sessions.identity).toEqual({
      modelId: 'clairic-flash',
      modelType: 'clairic',
      contextWindow: 65536,
    });
    expect(resident[0]!.sessions.servedWindowTokens).toBe(65536);

    registry.unregister('clairic-flash');
    registry.unregister('clairic-alias');
    expect(registry.listResidentModels()).toEqual([]);
  });

  it('emits a metrics-trace record for completed /v1/systemone requests and tolerates sink errors', async () => {
    const registry = new ModelRegistry();
    const native = clairicModel();
    registry.registerAny('clairic-flash', native, {
      identity: { modelId: 'clairic-flash', modelType: 'clairic', contextWindow: 65536 },
    });
    const recorded: MetricsTraceInput[] = [];
    const server = createServer((req: IncomingMessage, res: ServerResponse) => {
      const chunks: Buffer[] = [];
      req.on('data', (chunk: Buffer) => chunks.push(chunk));
      req.on('end', () => {
        const raw = Buffer.concat(chunks).toString('utf8');
        void handleSystemOne(req, res, raw, registry, null, undefined, undefined, undefined, {
          record: (rec) => {
            recorded.push(rec);
            if (recorded.length === 2) throw new Error('disk full');
          },
        });
      });
    });
    servers.push(server);
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;

    const body = JSON.stringify({
      model: 'clairic-flash',
      state: 'Total: $42.50',
      images: ['data:image/png;base64,AAAA'],
      fields: { invoice_total: { type: 'number' } },
    });
    const first = await fetch(`${base}/v1/systemone`, { method: 'POST', body });
    expect(first.status).toBe(200);
    expect(recorded).toHaveLength(1);
    expect(recorded[0]).toMatchObject({
      model: 'clairic-flash',
      modelId: 'clairic-flash',
      modelType: 'clairic',
      outcome: 'ok',
      endpoint: 'systemone',
      stream: false,
      dispatch: 'start',
      finishReason: 'stop',
      promptTokens: 124,
      outputTokens: 8,
      ctxWindow: 65536,
      imageCount: 1,
      toolCount: 0,
    });

    // A throwing trace sink must not fail the response.
    const second = await fetch(`${base}/v1/systemone`, { method: 'POST', body });
    expect(second.status).toBe(200);
    expect(recorded).toHaveLength(2);
  });
});
