import { Qwen35Model, MxArray } from '@mlx-node/core';
import { loadModel } from '@mlx-node/lm';
import { afterAll, beforeAll, describe, expect, it } from 'vite-plus/test';

import { createTempQwen35Model, type TempModel } from '../test-model-utils';
import { shape } from '../test-utils';

describe.sequential('Qwen3.5 Generation', () => {
  let tempModel: TempModel;

  beforeAll(async () => {
    tempModel = await createTempQwen35Model();
  });

  afterAll(() => {
    tempModel?.cleanup();
  });

  it('should generate tokens from prompt', async () => {
    const loaded = await loadModel(tempModel.modelPath);
    expect(loaded).toBeInstanceOf(Qwen35Model);
    const model = loaded as unknown as Qwen35Model;

    const prompt = MxArray.fromInt32(new Int32Array([1, 2, 3, 4, 5]), shape(1, 5));
    const result = await model.generate(prompt, {
      maxNewTokens: 5,
      temperature: 0.0, // greedy
    });

    expect(result.tokens.length).toBeGreaterThan(0);
    expect(result.tokens.length).toBeLessThanOrEqual(5);
    expect(result.numTokens).toBe(result.tokens.length);
    expect(['stop', 'length']).toContain(result.finishReason);
  });

  it('should respect maxNewTokens limit', async () => {
    const loaded = await loadModel(tempModel.modelPath);
    expect(loaded).toBeInstanceOf(Qwen35Model);
    const model = loaded as unknown as Qwen35Model;

    const prompt = MxArray.fromInt32(new Int32Array([1, 2, 3]), shape(1, 3));
    const result = await model.generate(prompt, {
      maxNewTokens: 3,
    });

    expect(result.tokens.length).toBeLessThanOrEqual(3);
  });
});
