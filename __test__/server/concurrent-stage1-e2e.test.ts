/**
 * Stage-1 server admission + native batching gate.
 *
 * Enable with a converted dense Qwen3 checkpoint:
 *
 *   QWEN3_STAGE1_MODEL_PATH=/abs/path/to/qwen3-mlx-bf16 \
 *     yarn test __test__/server/concurrent-stage1-e2e.test.ts
 */

import { existsSync, readFileSync, statSync } from 'node:fs';
import type { AddressInfo } from 'node:net';
import { join } from 'node:path';

import { loadModel, type ChatStreamEvent, type LoadableModel, type SessionCapableModel } from '@mlx-node/lm';
import { createServer, type ServerInstance } from '@mlx-node/server';
import { afterAll, beforeAll, describe, expect, it } from 'vite-plus/test';

const MODEL_ENV_NAME = 'QWEN3_STAGE1_MODEL_PATH';
const MODEL_ENV_PRESENT = Object.prototype.hasOwnProperty.call(process.env, MODEL_ENV_NAME);
const MODEL_PATH = process.env[MODEL_ENV_NAME];
const MODEL_NAME = 'stage1-qwen3';
const REQUEST_TIMEOUT_MS = 120_000;
const ACCOUNTING_TIMEOUT_MS = 20_000;
const TEST_TIMEOUT_MS = REQUEST_TIMEOUT_MS + ACCOUNTING_TIMEOUT_MS + 10_000;

type SchedulerStatsModel = SessionCapableModel & {
  schedulerStats(): Promise<{ maxBatchOccupancy: number }>;
};

const delay = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

async function waitUntil(predicate: () => boolean, label: string, timeoutMs = 5_000): Promise<void> {
  const deadline = performance.now() + timeoutMs;
  while (performance.now() < deadline) {
    if (predicate()) return;
    await delay(10);
  }
  throw new Error(`${label} (>${timeoutMs}ms)`);
}

async function withRequestTimeout<T>(label: string, run: (signal: AbortSignal) => Promise<T>): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(
    () => controller.abort(new Error(`${label} exceeded ${REQUEST_TIMEOUT_MS}ms`)),
    REQUEST_TIMEOUT_MS,
  );
  try {
    return await run(controller.signal);
  } finally {
    clearTimeout(timer);
  }
}

function isStage1Model(model: LoadableModel): model is LoadableModel & SchedulerStatsModel {
  const candidate = model as Partial<SchedulerStatsModel>;
  return (
    typeof candidate.chatStreamSessionStart === 'function' &&
    typeof candidate.hasBlockPagedCache === 'function' &&
    typeof candidate.maxConcurrentSequences === 'function' &&
    typeof candidate.schedulerStats === 'function'
  );
}

function instrumentStreams(target: SchedulerStatsModel): {
  model: SchedulerStatsModel;
  peakActive: () => number;
  releasedOwners: () => readonly string[];
} {
  const streamMethods = new Set<PropertyKey>([
    'chatStreamSessionStart',
    'chatStreamSessionContinue',
    'chatStreamSessionContinueTool',
  ]);
  let active = 0;
  let peak = 0;
  const releasedOwners: string[] = [];
  const model = new Proxy(target, {
    get(nativeModel, property) {
      const value = Reflect.get(nativeModel, property, nativeModel) as unknown;
      if (property === 'releaseCacheOwner' && typeof value === 'function') {
        return async (ownerId: string): Promise<void> => {
          await (value as (ownerId: string) => Promise<void>).call(nativeModel, ownerId);
          releasedOwners.push(ownerId);
        };
      }
      if (streamMethods.has(property) && typeof value === 'function') {
        return async function* (...args: unknown[]): AsyncGenerator<ChatStreamEvent> {
          active += 1;
          peak = Math.max(peak, active);
          try {
            const stream = (value as (...callArgs: unknown[]) => AsyncGenerator<ChatStreamEvent>).apply(
              nativeModel,
              args,
            );
            yield* stream;
          } finally {
            active -= 1;
          }
        };
      }
      return typeof value === 'function' ? value.bind(nativeModel) : value;
    },
  }) as SchedulerStatsModel;
  return { model, peakActive: () => peak, releasedOwners: () => releasedOwners };
}

async function postStream(baseUrl: string, input: string): Promise<string> {
  return withRequestTimeout('Stage-1 SSE request', async (signal) => {
    const response = await fetch(`${baseUrl}/v1/responses`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: MODEL_NAME,
        input,
        stream: true,
        temperature: 0,
        max_output_tokens: 64,
      }),
      signal,
    });
    expect(response.status).toBe(200);
    return await response.text();
  });
}

async function postMessage(
  baseUrl: string,
  input: string,
): Promise<{ content?: Array<{ type?: string; text?: string }> }> {
  return withRequestTimeout('Stage-1 Messages request', async (signal) => {
    const response = await fetch(`${baseUrl}/v1/messages`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: MODEL_NAME,
        messages: [{ role: 'user', content: input }],
        max_tokens: 16,
        temperature: 0,
      }),
      signal,
    });
    expect(response.status).toBe(200);
    return (await response.json()) as { content?: Array<{ type?: string; text?: string }> };
  });
}

async function postResponse(baseUrl: string, input: string): Promise<{ id?: string; output_text?: string }> {
  return withRequestTimeout('Stage-1 Responses request', async (signal) => {
    const response = await fetch(`${baseUrl}/v1/responses`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: MODEL_NAME,
        input,
        temperature: 0,
        max_output_tokens: 16,
      }),
      signal,
    });
    expect(response.status).toBe(200);
    return (await response.json()) as { id?: string; output_text?: string };
  });
}

const stage1Describe = MODEL_ENV_PRESENT ? describe.sequential : describe.skip;

stage1Describe('Stage-1 real-model server admission', () => {
  let instance: ServerInstance;
  let nativeModel: SchedulerStatsModel;
  let peakActive: () => number;
  let releasedOwners: () => readonly string[];
  let baseUrl: string;

  async function waitForAccountingDrain(label: string): Promise<void> {
    await waitUntil(
      () =>
        instance.registry.getSessionRegistry(MODEL_NAME)?.queueDepth === 0 &&
        instance.registry.getSessionRegistry(MODEL_NAME)?.preDispatchAdmitCount === 0 &&
        instance.registry.getSessionRegistry(MODEL_NAME)?.pendingDisposalCount === 0 &&
        instance.health().work.inFlight === 0,
      label,
      ACCOUNTING_TIMEOUT_MS,
    );
  }

  beforeAll(async () => {
    if (!MODEL_ENV_PRESENT) return;
    const modelPath = MODEL_PATH;
    if (modelPath == null || modelPath.trim() === '') {
      throw new Error(`${MODEL_ENV_NAME} is explicitly set but empty`);
    }
    if (!existsSync(modelPath) || !statSync(modelPath).isDirectory()) {
      throw new Error(`${MODEL_ENV_NAME} is not a directory: ${modelPath}`);
    }
    const configPath = join(modelPath, 'config.json');
    if (!existsSync(configPath)) throw new Error(`${MODEL_ENV_NAME} has no config.json: ${modelPath}`);
    const modelType = (JSON.parse(readFileSync(configPath, 'utf8')) as { model_type?: unknown }).model_type;
    if (modelType !== 'qwen3') {
      throw new Error(`${MODEL_ENV_NAME} must identify Qwen3; model_type=${String(modelType)}`);
    }

    instance = await createServer({
      port: 0,
      host: '127.0.0.1',
      disableStore: true,
      idleClearCacheMs: 3_600_000,
      maxQueueDepthPerModel: 1,
    });
    await instance.loadModel({
      name: MODEL_NAME,
      load: async () => {
        const loaded = await loadModel(modelPath);
        if (!isStage1Model(loaded)) throw new Error(`${MODEL_ENV_NAME} did not load a Stage-1 Qwen3 model`);
        nativeModel = loaded;
        const instrumented = instrumentStreams(loaded);
        peakActive = instrumented.peakActive;
        releasedOwners = instrumented.releasedOwners;
        return instrumented.model;
      },
    });
    const address = instance.server.address() as AddressInfo;
    baseUrl = `http://127.0.0.1:${address.port}`;
  }, 120_000);

  afterAll(async () => {
    if (instance !== undefined) await instance.close({ timeoutMs: 10_000 });
  }, 20_000);

  it(
    'admits two SSE turns together and executes a real N=2 decode batch',
    async () => {
      expect(nativeModel.hasBlockPagedCache?.()).toBe(true);
      const capacity = nativeModel.maxConcurrentSequences?.() ?? 0;
      expect(capacity).toBeGreaterThanOrEqual(2);
      expect(instance.registry.getSessionRegistry(MODEL_NAME)?.concurrentAdmissionLimit).toBe(capacity);

      const prompt = 'Continue this sequence with concise comma-separated integers: ' + '1, 2, 3, '.repeat(256);
      const [firstResult, secondResult] = await Promise.allSettled([
        postStream(baseUrl, `${prompt}\nFirst stream.`),
        postStream(baseUrl, `${prompt}\nSecond stream.`),
      ]);
      await waitForAccountingDrain('batched SSE request accounting did not settle');
      if (firstResult.status === 'rejected') throw firstResult.reason;
      if (secondResult.status === 'rejected') throw secondResult.reason;
      const first = firstResult.value;
      const second = secondResult.value;

      expect(first).toContain('response.created');
      expect(second).toContain('response.created');
      expect(first).toMatch(/response\.(completed|incomplete)/u);
      expect(second).toMatch(/response\.(completed|incomplete)/u);
      const stats = await nativeModel.schedulerStats();
      expect(peakActive(), `native stream peak; scheduler occupancy=${stats.maxBatchOccupancy}`).toBe(2);
      expect(stats.maxBatchOccupancy).toBeGreaterThanOrEqual(2);
    },
    TEST_TIMEOUT_MS,
  );

  it(
    'releases each stateless Messages cache owner after its response',
    async () => {
      const before = releasedOwners().length;
      const [firstResult, secondResult] = await Promise.allSettled([
        postMessage(baseUrl, 'Reply with one short word for red.'),
        postMessage(baseUrl, 'Reply with one short word for blue.'),
      ]);
      if (firstResult.status === 'rejected') {
        await waitForAccountingDrain('failed Messages request accounting did not settle');
        throw firstResult.reason;
      }
      if (secondResult.status === 'rejected') {
        await waitForAccountingDrain('failed Messages request accounting did not settle');
        throw secondResult.reason;
      }
      const first = firstResult.value;
      const second = secondResult.value;

      expect(first.content).toBeInstanceOf(Array);
      expect(second.content).toBeInstanceOf(Array);
      await waitUntil(
        () =>
          releasedOwners().length === before + 2 &&
          instance.registry.getSessionRegistry(MODEL_NAME)?.queueDepth === 0 &&
          instance.registry.getSessionRegistry(MODEL_NAME)?.preDispatchAdmitCount === 0 &&
          instance.health().work.inFlight === 0,
        'stateless owner release and request accounting did not settle',
        ACCOUNTING_TIMEOUT_MS,
      );
      const released = releasedOwners().slice(before);
      expect(released).toHaveLength(2);
      expect(released.every((owner) => owner.length > 0)).toBe(true);
      expect(new Set(released).size).toBe(2);
      expect(instance.registry.getSessionRegistry(MODEL_NAME)?.queueDepth).toBe(0);
      expect(instance.registry.getSessionRegistry(MODEL_NAME)?.preDispatchAdmitCount).toBe(0);
      expect(instance.health().work.inFlight).toBe(0);
    },
    TEST_TIMEOUT_MS,
  );

  it(
    'releases the prior Responses owner when a stateless chain replaces the warm slot',
    async () => {
      try {
        const first = await postResponse(baseUrl, 'Reply with one short word for circle.');
        // The first request's adopt evicts test 1's surviving warm session,
        // and that release is only recorded after the response completes.
        // Drain before snapshotting so `afterFirst` cannot miss it.
        await waitForAccountingDrain('first Responses request accounting did not settle');
        const afterFirst = releasedOwners().length;
        const second = await postResponse(baseUrl, 'Reply with one short word for square.');

        expect(first.id).toMatch(/^resp_/u);
        expect(second.id).toMatch(/^resp_/u);
        expect(second.id).not.toBe(first.id);
        await waitUntil(
          () =>
            releasedOwners().length === afterFirst + 1 &&
            instance.registry.getSessionRegistry(MODEL_NAME)?.queueDepth === 0 &&
            instance.registry.getSessionRegistry(MODEL_NAME)?.preDispatchAdmitCount === 0 &&
            instance.health().work.inFlight === 0,
          'Responses owner replacement and request accounting did not settle',
          ACCOUNTING_TIMEOUT_MS,
        );
        expect(releasedOwners().slice(afterFirst)).toHaveLength(1);
      } catch (error) {
        await waitForAccountingDrain('failed Responses request accounting did not settle');
        throw error;
      }
    },
    TEST_TIMEOUT_MS,
  );
});
