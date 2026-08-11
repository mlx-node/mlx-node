import { existsSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

import { ChatSession, loadModel, type ChatResult, type LoadableModel, type SessionCapableModel } from '@mlx-node/lm';
import { afterAll, beforeAll, describe, expect, it } from 'vite-plus/test';

const MODEL_ENV_NAME = 'MLX_TEST_GEMMA4_CONCURRENCY_PATH';
const MODEL_ENV_PRESENT = Object.prototype.hasOwnProperty.call(process.env, MODEL_ENV_NAME);
const MODEL_PATH = process.env[MODEL_ENV_NAME];

type ScheduledGemma = SessionCapableModel & {
  hasBlockPagedCache(): boolean;
  maxConcurrentSequences(): number;
  schedulerStats(): Promise<{
    maxBatchOccupancy: number;
    decodeBatchOccupancyHist: Array<{ occupancy: number; steps: number }>;
    fusedGreedyEpilogueSteps: number;
  }>;
};

function isScheduledGemma(model: LoadableModel): model is LoadableModel & ScheduledGemma {
  const candidate = model as Partial<ScheduledGemma>;
  return (
    typeof candidate.hasBlockPagedCache === 'function' &&
    typeof candidate.maxConcurrentSequences === 'function' &&
    typeof candidate.schedulerStats === 'function'
  );
}

const gemmaDescribe = MODEL_ENV_PRESENT ? describe.sequential : describe.skip;

gemmaDescribe('Gemma4 grouped hybrid KV continuous batching', () => {
  let model: ScheduledGemma;
  const sessions: ChatSession[] = [];
  const activeSends = new Set<{
    controller: AbortController;
    promise: Promise<ChatResult>;
  }>();

  beforeAll(async () => {
    const modelPath = MODEL_PATH;
    if (modelPath == null || modelPath.trim() === '') {
      throw new Error(`${MODEL_ENV_NAME} is explicitly set but empty`);
    }
    if (!existsSync(modelPath) || !statSync(modelPath).isDirectory()) {
      throw new Error(`${MODEL_ENV_NAME} is not a directory: ${modelPath}`);
    }
    const configPath = join(modelPath, 'config.json');
    if (!existsSync(configPath)) throw new Error(`${MODEL_ENV_NAME} has no config.json: ${modelPath}`);
    const raw = JSON.parse(readFileSync(configPath, 'utf8')) as {
      model_type?: unknown;
      text_config?: { model_type?: unknown };
    };
    const modelType = raw.text_config?.model_type ?? raw.model_type;
    if (modelType !== 'gemma4_text' && modelType !== 'gemma4') {
      throw new Error(`${MODEL_ENV_NAME} must identify Gemma4 text; model_type=${String(modelType)}`);
    }

    const loaded = await loadModel(modelPath);
    if (!isScheduledGemma(loaded)) throw new Error(`${MODEL_ENV_NAME} did not load the scheduled Gemma4 surface`);
    model = loaded;
  }, 900_000);

  afterAll(async () => {
    for (const { controller } of activeSends) controller.abort();
    await Promise.allSettled([...activeSends].map(({ promise }) => promise));
    await Promise.all(sessions.map(async (session) => session.dispose()));
  }, 60_000);

  function session(): ChatSession {
    const value = new ChatSession(model);
    sessions.push(value);
    return value;
  }

  const config = {
    // Two tokens are enough to force one genuine decode step after prefill;
    // keeping this small makes the real 9.6 GiB CI checkpoint practical.
    maxNewTokens: 2,
    temperature: 0,
    repetitionPenalty: 1,
    repetitionContextSize: 0,
    reportPerformance: true,
  } as const;

  function send(value: ChatSession, prompt: string): Promise<ChatResult> {
    const controller = new AbortController();
    let active!: { controller: AbortController; promise: Promise<ChatResult> };
    const promise = value.send(prompt, { config, signal: controller.signal }).finally(() => activeSends.delete(active));
    active = { controller, promise };
    activeSends.add(active);
    return promise;
  }

  it('matches serial replay across two concurrent starts and continuations with a real N=2 decode wave', async () => {
    expect(model.hasBlockPagedCache()).toBe(true);
    expect(model.maxConcurrentSequences()).toBeGreaterThanOrEqual(2);

    const serialA = session();
    const serialB = session();
    const batchedA = session();
    const batchedB = session();
    const sharedPrefix = 'Count upward using comma-separated integers. ' + '1, 2, 3, '.repeat(4);
    const promptA = `${sharedPrefix}\nEnd the answer after the next eight integers.`;
    const promptB = `${sharedPrefix}\nEnd the answer after the next twelve integers.`;

    const serialStartA = await send(serialA, promptA);
    const serialStartB = await send(serialB, promptB);
    const [batchedStartA, batchedStartB] = await Promise.all([send(batchedA, promptA), send(batchedB, promptB)]);
    expect(batchedStartA.rawText).toBe(serialStartA.rawText);
    expect(batchedStartB.rawText).toBe(serialStartB.rawText);

    const followA = 'Now repeat only the final four integers from your prior answer.';
    const followB = 'Now repeat only the final six integers from your prior answer.';
    const serialContinueA = await send(serialA, followA);
    const serialContinueB = await send(serialB, followB);
    const [batchedContinueA, batchedContinueB] = await Promise.all([send(batchedA, followA), send(batchedB, followB)]);
    expect(batchedContinueA.rawText).toBe(serialContinueA.rawText);
    expect(batchedContinueB.rawText).toBe(serialContinueB.rawText);

    const stats = await model.schedulerStats();
    expect(stats.maxBatchOccupancy).toBeGreaterThanOrEqual(2);
    expect(stats.decodeBatchOccupancyHist.some(({ occupancy, steps }) => occupancy >= 2 && steps > 0)).toBe(true);
    expect(stats.fusedGreedyEpilogueSteps).toBeGreaterThan(0);
  }, 300_000);
});
