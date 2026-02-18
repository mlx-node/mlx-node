import { describe, it, expect } from 'vite-plus/test';
import { Qwen35Model, MxArray } from '@mlx-node/core';
import type { Qwen35Config } from '@mlx-node/core';
import { shape } from '../test-utils';

describe.sequential('Qwen3.5 MoE', () => {
  const denseConfig: Qwen35Config = {
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
    fullAttentionInterval: 2,
    partialRotaryFactor: 0.25,
    ropeTheta: 10000.0,
  };

  it('should create dense model when numExperts is undefined', () => {
    const model = new Qwen35Model(denseConfig);
    expect(model).toBeDefined();

    const input = MxArray.fromInt32(new Int32Array([1, 2]), shape(1, 2));
    const logits = model.forward(input);
    expect(Number(logits.shape()[2])).toBe(1000);
  });

  it('should create MoE model with experts', () => {
    const moeConfig: Qwen35Config = {
      ...denseConfig,
      numExperts: 4,
      numExpertsPerTok: 2,
      decoderSparseStep: 1,
      sharedExpertIntermediateSize: 128,
      moeIntermediateSize: 64,
      normTopkProb: true,
    };

    const model = new Qwen35Model(moeConfig);
    expect(model).toBeDefined();

    const input = MxArray.fromInt32(new Int32Array([1, 2]), shape(1, 2));
    const logits = model.forward(input);
    expect(Number(logits.shape()[2])).toBe(1000);
  });

  it('should have more parameters with MoE than dense', () => {
    const dense = new Qwen35Model(denseConfig);
    const moe = new Qwen35Model({
      ...denseConfig,
      numExperts: 4,
      numExpertsPerTok: 2,
      decoderSparseStep: 1,
      sharedExpertIntermediateSize: 128,
      moeIntermediateSize: 64,
      normTopkProb: true,
    });

    expect(moe.numParameters()).toBeGreaterThan(dense.numParameters());
  });
});
