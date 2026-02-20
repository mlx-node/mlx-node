import { describe, it, expect } from 'vite-plus/test';
import { Qwen35Model, MxArray } from '@mlx-node/core';
import type { Qwen35Config } from '@mlx-node/core';
import { shape } from '../test-utils';

// Tiny config for generation tests — the full 0.6B config is too slow
// due to sequential GatedDeltaNet recurrence (21 linear attention layers).
// This uses 4 layers (1 full cycle: 3 linear + 1 full attention) with
// small dimensions to keep generation under a few seconds.
const TINY_GEN_CONFIG: Qwen35Config = {
  vocabSize: 1000,
  hiddenSize: 128,
  numLayers: 4,
  numHeads: 4,
  numKvHeads: 2,
  intermediateSize: 256,
  rmsNormEps: 1e-6,
  headDim: 32,
  tieWordEmbeddings: true,
  attentionBias: false,
  maxPositionEmbeddings: 512,
  padTokenId: 0,
  eosTokenId: 1,
  bosTokenId: 0,
  linearNumValueHeads: 8,
  linearNumKeyHeads: 4,
  linearKeyHeadDim: 32,
  linearValueHeadDim: 16,
  linearConvKernelDim: 4,
  fullAttentionInterval: 4,
  partialRotaryFactor: 0.25,
  ropeTheta: 10000.0,
};

describe.sequential('Qwen3.5 Generation', () => {
  it('should generate tokens from prompt', async () => {
    const model = new Qwen35Model(TINY_GEN_CONFIG);

    const prompt = MxArray.fromInt32(new Int32Array([1, 2, 3, 4, 5]), shape(1, 5));
    const result = await model.generate(prompt, {
      maxNewTokens: 5,
      temperature: 0.0, // greedy
    });

    expect(result.tokens.length).toBeGreaterThan(0);
    expect(result.tokens.length).toBeLessThanOrEqual(5);
    expect(result.numTokens).toBe(result.tokens.length);
    expect(['eos', 'length']).toContain(result.finishReason);
  });

  it('should respect maxNewTokens limit', async () => {
    const model = new Qwen35Model(TINY_GEN_CONFIG);

    const prompt = MxArray.fromInt32(new Int32Array([1, 2, 3]), shape(1, 3));
    const result = await model.generate(prompt, {
      maxNewTokens: 3,
    });

    expect(result.tokens.length).toBeLessThanOrEqual(3);
  });
});
