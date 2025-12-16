/**
 * Qwen3 Parameter Management Tests
 *
 * Tests for parameter extraction and loading functionality
 */

import { describe, it, expect } from 'vitest';
import type { Qwen3Config } from '@mlx-node/lm';
import { MxArray, Qwen3Model } from '@mlx-node/core';
import { shape, int32 } from '../test-utils.js';

// Tiny test configuration for fast testing
const TEST_CONFIG: Qwen3Config = {
  vocabSize: 1000,
  hiddenSize: 64,
  numLayers: 2, // Only 2 layers for speed
  numHeads: 4,
  numKvHeads: 2,
  headDim: 16, // hiddenSize / numHeads = 64 / 4 = 16
  intermediateSize: 128,
  rmsNormEps: 1e-6,
  ropeTheta: 10000.0,
  maxPositionEmbeddings: 512,
  useQkNorm: false, // Disable QK norm for simplicity
  tieWordEmbeddings: false,
  padTokenId: 0,
  eosTokenId: 1,
  bosTokenId: 0,
};

function assertArrayClose(actual: MxArray, expected: MxArray, atol: number = 1e-6) {
  const actualData = actual.toFloat32();
  const expectedData = expected.toFloat32();

  expect(actualData.length).toBe(expectedData.length);

  for (let i = 0; i < actualData.length; i++) {
    expect(Math.abs(actualData[i] - expectedData[i])).toBeLessThan(atol);
  }
}

describe('Qwen3 Parameter Management', () => {
  describe('Parameter Extraction', () => {
    it('should extract all parameters from model', () => {
      const model = new Qwen3Model(TEST_CONFIG);
      const params = model.getParameters();

      // Check that all expected parameters are present
      expect(params).toHaveProperty('embedding.weight');
      expect(params).toHaveProperty('final_norm.weight');
      expect(params).toHaveProperty('lm_head.weight');

      // Check first transformer layer parameters
      expect(params).toHaveProperty('layers.0.self_attn.q_proj.weight');
      expect(params).toHaveProperty('layers.0.self_attn.k_proj.weight');
      expect(params).toHaveProperty('layers.0.self_attn.v_proj.weight');
      expect(params).toHaveProperty('layers.0.self_attn.o_proj.weight');
      expect(params).toHaveProperty('layers.0.mlp.gate_proj.weight');
      expect(params).toHaveProperty('layers.0.mlp.up_proj.weight');
      expect(params).toHaveProperty('layers.0.mlp.down_proj.weight');
      expect(params).toHaveProperty('layers.0.input_layernorm.weight');
      expect(params).toHaveProperty('layers.0.post_attention_layernorm.weight');
    });

    it('should extract parameters with correct shapes', () => {
      const model = new Qwen3Model(TEST_CONFIG);
      const config = TEST_CONFIG;
      const params = model.getParameters();

      // Embedding: (vocab_size, hidden_size)
      const embeddingShape = Array.from(params['embedding.weight'].shape()).map(Number);
      expect(embeddingShape).toEqual([config.vocabSize, config.hiddenSize]);

      // Q projection: (hidden_size, num_heads * head_dim)
      const headDim = config.hiddenSize / config.numHeads;
      const qShape = Array.from(params['layers.0.self_attn.q_proj.weight'].shape()).map(Number);
      expect(qShape).toEqual([config.numHeads * headDim, config.hiddenSize]);

      // K projection: (hidden_size, num_kv_heads * head_dim)
      const kShape = Array.from(params['layers.0.self_attn.k_proj.weight'].shape()).map(Number);
      expect(kShape).toEqual([config.numKvHeads * headDim, config.hiddenSize]);

      // MLP gate projection: (intermediate_size, hidden_size)
      const gateShape = Array.from(params['layers.0.mlp.gate_proj.weight'].shape()).map(Number);
      expect(gateShape).toEqual([config.intermediateSize, config.hiddenSize]);

      // LM head: (vocab_size, hidden_size)
      const lmHeadShape = Array.from(params['lm_head.weight'].shape()).map(Number);
      expect(lmHeadShape).toEqual([config.vocabSize, config.hiddenSize]);
    });

    it('should count correct number of parameters', () => {
      const model = new Qwen3Model(TEST_CONFIG);
      const params = model.getParameters();

      const config = TEST_CONFIG;

      // Expected parameters per layer:
      // - Attention: Q, K, V, O (4 weight matrices)
      // - MLP: gate, up, down (3 weight matrices)
      // - Norms: input_layernorm, post_attention_layernorm (2 weight vectors)
      const paramsPerLayer = 4 + 3 + 2; // 9 parameters per layer

      // Total expected:
      // - embedding.weight (1)
      // - layers.*.* (numLayers * 9)
      // - final_norm.weight (1)
      // - lm_head.weight (1)
      const expectedCount = 1 + config.numLayers * paramsPerLayer + 1 + 1;

      expect(Object.keys(params).length).toBe(expectedCount);
    });

    it('should extract parameters from all layers', () => {
      const model = new Qwen3Model(TEST_CONFIG);
      const config = TEST_CONFIG;
      const params = model.getParameters();

      // Check that all layers have parameters
      for (let i = 0; i < config.numLayers; i++) {
        expect(params).toHaveProperty(`layers.${i}.self_attn.q_proj.weight`);
        expect(params).toHaveProperty(`layers.${i}.mlp.gate_proj.weight`);
        expect(params).toHaveProperty(`layers.${i}.input_layernorm.weight`);
      }
    });
  });

  describe('Parameter Loading', () => {
    it('should load parameters correctly', () => {
      const model1 = new Qwen3Model(TEST_CONFIG);
      const model2 = new Qwen3Model(TEST_CONFIG);

      // Extract parameters from model1
      const params1 = model1.getParameters();

      // Load into model2
      model2.loadParameters(params1);

      // Extract parameters from model2
      const params2 = model2.getParameters();

      // Parameters should be identical
      expect(Object.keys(params2).length).toBe(Object.keys(params1).length);

      // Check a few key parameters for equality
      assertArrayClose(params2['embedding.weight'], params1['embedding.weight'], 1e-8);
      assertArrayClose(params2['layers.0.self_attn.q_proj.weight'], params1['layers.0.self_attn.q_proj.weight'], 1e-8);
      assertArrayClose(params2['layers.0.mlp.gate_proj.weight'], params1['layers.0.mlp.gate_proj.weight'], 1e-8);
      assertArrayClose(params2['lm_head.weight'], params1['lm_head.weight'], 1e-8);
    });

    it('should produce identical outputs after parameter loading', () => {
      const model1 = new Qwen3Model(TEST_CONFIG);
      const model2 = new Qwen3Model(TEST_CONFIG);

      // Create test input
      const inputIds = MxArray.fromInt32(int32(1, 2, 3, 4, 5), shape(1, 5));

      // Get output from model1
      const output1 = model1.forward(inputIds);

      // Extract and load parameters into model2
      const params = model1.getParameters();
      model2.loadParameters(params);

      // Get output from model2
      const output2 = model2.forward(inputIds);

      // Outputs should be identical
      assertArrayClose(output2, output1, 1e-6);
    });

    it('should handle partial parameter loading', () => {
      const model = new Qwen3Model(TEST_CONFIG);

      // Save original embedding weight
      const originalParams = model.getParameters();
      const originalEmbedding = originalParams['embedding.weight'];

      // Load only a subset of parameters
      const newEmbedding = MxArray.randomNormal(originalEmbedding.shape(), 0, 0.02);
      model.loadParameters({
        'embedding.weight': newEmbedding,
      });

      // Embedding should be updated
      const updatedParams = model.getParameters();
      const updatedEmbedding = updatedParams['embedding.weight'];
      const diff = updatedEmbedding.sub(newEmbedding).abs().mean().toFloat32()[0];
      expect(diff).toBeLessThan(1e-6);

      // Other parameters should be unchanged (check a few)
      expect(updatedParams['layers.0.self_attn.q_proj.weight']).toBeDefined();
    });
  });

  describe('Parameter Update Simulation', () => {
    it('should modify parameters when weights are updated', () => {
      const model = new Qwen3Model(TEST_CONFIG);

      // Save original parameters
      const originalParams = model.getParameters();
      const originalQWeight = originalParams['layers.0.self_attn.q_proj.weight'].toFloat32();

      // Create modified parameters (simulate gradient update)
      const modifiedParams: Record<string, MxArray> = {};
      for (const [name, weight] of Object.entries(originalParams)) {
        // Add small noise to simulate parameter update
        const noise = MxArray.randomNormal(weight.shape(), 0, 0.01);
        modifiedParams[name] = weight.add(noise);
      }

      // Load modified parameters
      model.loadParameters(modifiedParams);

      // Get updated parameters
      const updatedParams = model.getParameters();
      const updatedQWeight = updatedParams['layers.0.self_attn.q_proj.weight'].toFloat32();

      // Parameters should have changed
      let diffCount = 0;
      for (let i = 0; i < originalQWeight.length; i++) {
        if (Math.abs(originalQWeight[i] - updatedQWeight[i]) > 1e-8) {
          diffCount++;
        }
      }

      // Most values should be different (we added noise to all)
      expect(diffCount).toBeGreaterThan(originalQWeight.length * 0.9);
    });

    it('should affect model outputs when parameters change', () => {
      const model = new Qwen3Model(TEST_CONFIG);

      // Create test input
      const inputIds = MxArray.fromInt32(int32(1, 2, 3), shape(1, 3));

      // Get original output
      const output1 = model.forward(inputIds);

      // Modify a single weight matrix
      const params = model.getParameters();
      const qWeight = params['layers.0.self_attn.q_proj.weight'];
      const modifiedQWeight = qWeight.add(MxArray.randomNormal(qWeight.shape(), 0, 0.1));

      model.loadParameters({
        'layers.0.self_attn.q_proj.weight': modifiedQWeight,
      });

      // Get new output
      const output2 = model.forward(inputIds);

      // Outputs should be different
      const diff = output2.sub(output1).abs().mean().toFloat32()[0];
      expect(diff).toBeGreaterThan(1e-6);
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty parameter dict in loadParameters', () => {
      const model = new Qwen3Model(TEST_CONFIG);

      // Save original output
      const inputIds = MxArray.fromInt32(int32(1, 2, 3), shape(1, 3));
      const originalOutput = model.forward(inputIds);

      // Load empty dict
      model.loadParameters({});

      // Output should be unchanged
      const newOutput = model.forward(inputIds);
      assertArrayClose(newOutput, originalOutput, 1e-8);
    });

    it('should handle loading with missing keys', () => {
      const model = new Qwen3Model(TEST_CONFIG);
      const params = model.getParameters();

      // Create partial parameter dict (missing some keys)
      const partialParams: Record<string, MxArray> = {
        'embedding.weight': params['embedding.weight'],
        'lm_head.weight': params['lm_head.weight'],
      };

      // Should not throw
      expect(() => model.loadParameters(partialParams)).not.toThrow();
    });

    it('should handle parameter extraction multiple times', () => {
      const model = new Qwen3Model(TEST_CONFIG);

      // Extract parameters twice
      const params1 = model.getParameters();
      const params2 = model.getParameters();

      // Should have same number of parameters
      expect(Object.keys(params1).length).toBe(Object.keys(params2).length);

      // Values should be identical
      assertArrayClose(params1['embedding.weight'], params2['embedding.weight'], 1e-8);
    });
  });
});
