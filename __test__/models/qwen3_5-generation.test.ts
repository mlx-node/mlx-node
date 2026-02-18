import { describe, it, expect } from 'vite-plus/test';
import { Qwen35Model, MxArray } from '@mlx-node/core';
import { getQwen35Config } from '@mlx-node/lm';
import { shape } from '../test-utils';

describe.sequential('Qwen3.5 Generation', () => {
  it('should generate tokens from prompt', () => {
    const config = getQwen35Config('qwen3.5-0.6b');
    const model = new Qwen35Model(config);

    const prompt = MxArray.fromInt32(new Int32Array([1, 2, 3, 4, 5]), shape(1, 5));
    const result = model.generate(prompt, {
      maxNewTokens: 5,
      temperature: 0.0, // greedy
    });

    expect(result.tokens.length).toBeGreaterThan(0);
    expect(result.tokens.length).toBeLessThanOrEqual(5);
    expect(result.numTokens).toBe(result.tokens.length);
    expect(['eos', 'length']).toContain(result.finishReason);
  });

  it('should respect maxNewTokens limit', () => {
    const config = getQwen35Config('qwen3.5-0.6b');
    const model = new Qwen35Model(config);

    const prompt = MxArray.fromInt32(new Int32Array([1, 2, 3]), shape(1, 3));
    const result = model.generate(prompt, {
      maxNewTokens: 3,
    });

    expect(result.tokens.length).toBeLessThanOrEqual(3);
  });
});
