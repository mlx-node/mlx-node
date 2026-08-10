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

type SchedulerStatsModel = SessionCapableModel & {
  schedulerStats(): Promise<{ maxBatchOccupancy: number }>;
};

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
} {
  const streamMethods = new Set<PropertyKey>([
    'chatStreamSessionStart',
    'chatStreamSessionContinue',
    'chatStreamSessionContinueTool',
  ]);
  let active = 0;
  let peak = 0;
  const model = new Proxy(target, {
    get(nativeModel, property) {
      const value = Reflect.get(nativeModel, property, nativeModel) as unknown;
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
  return { model, peakActive: () => peak };
}

async function postStream(baseUrl: string, input: string): Promise<string> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new Error('Stage-1 SSE request exceeded 30s')), 30_000);
  try {
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
      signal: controller.signal,
    });
    expect(response.status).toBe(200);
    return await response.text();
  } finally {
    clearTimeout(timer);
  }
}

const stage1Describe = MODEL_ENV_PRESENT ? describe.sequential : describe.skip;

stage1Describe('Stage-1 real-model server admission', () => {
  let instance: ServerInstance;
  let nativeModel: SchedulerStatsModel;
  let peakActive: () => number;
  let baseUrl: string;

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
        return instrumented.model;
      },
    });
    const address = instance.server.address() as AddressInfo;
    baseUrl = `http://127.0.0.1:${address.port}`;
  }, 120_000);

  afterAll(async () => {
    if (instance !== undefined) await instance.close({ timeoutMs: 10_000 });
  }, 20_000);

  it('admits two SSE turns together and executes a real N=2 decode batch', async () => {
    expect(nativeModel.hasBlockPagedCache?.()).toBe(true);
    const capacity = nativeModel.maxConcurrentSequences?.() ?? 0;
    expect(capacity).toBeGreaterThanOrEqual(2);
    expect(instance.registry.getSessionRegistry(MODEL_NAME)?.concurrentAdmissionLimit).toBe(capacity);

    const prompt = 'Continue this sequence with concise comma-separated integers: ' + '1, 2, 3, '.repeat(256);
    const [first, second] = await Promise.all([
      postStream(baseUrl, `${prompt}\nFirst stream.`),
      postStream(baseUrl, `${prompt}\nSecond stream.`),
    ]);

    expect(first).toContain('response.created');
    expect(second).toContain('response.created');
    expect(first).toMatch(/response\.(completed|incomplete)/u);
    expect(second).toMatch(/response\.(completed|incomplete)/u);
    const stats = await nativeModel.schedulerStats();
    expect(peakActive(), `native stream peak; scheduler occupancy=${stats.maxBatchOccupancy}`).toBe(2);
    expect(stats.maxBatchOccupancy).toBeGreaterThanOrEqual(2);
    expect(instance.registry.getSessionRegistry(MODEL_NAME)?.queueDepth).toBe(0);
  }, 45_000);
});
