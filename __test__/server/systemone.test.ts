import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';

import type { DecisionModel } from '@mlx-node/lm';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { formatClefResult, validateClefRequest } from '../../packages/lm/src/clef-types.js';
import { DecisionQueueFullError } from '../../packages/server/src/decision-registry.js';
import { createHandler } from '../../packages/server/src/handler.js';
import { makeSwapController } from '../../packages/server/src/host/swap.js';
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
