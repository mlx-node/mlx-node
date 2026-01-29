/**
 * Qwen3 Parameter Validation Tests
 *
 * Tests for the parameter validation functionality that ensures all required
 * parameters are loaded with correct shapes after SafeTensors loading.
 */

import { describe, it, expect } from 'vite-plus/test';
import type { Qwen3Config } from '@mlx-node/lm';
import { MxArray, Qwen3Model } from '@mlx-node/core';
import { shape } from '../test-utils.js';

// Tiny test configuration without QK norm
const TEST_CONFIG: Qwen3Config = {
  vocabSize: 1000,
  hiddenSize: 64,
  numLayers: 2,
  numHeads: 4,
  numKvHeads: 2,
  headDim: 16,
  intermediateSize: 128,
  rmsNormEps: 1e-6,
  ropeTheta: 10000.0,
  maxPositionEmbeddings: 512,
  useQkNorm: false,
  tieWordEmbeddings: false,
  padTokenId: 0,
  eosTokenId: 1,
  bosTokenId: 0,
};

// Configuration with QK norm enabled
const TEST_CONFIG_WITH_QK_NORM: Qwen3Config = {
  ...TEST_CONFIG,
  useQkNorm: true,
};

// Configuration with tied embeddings
const TEST_CONFIG_TIED_EMBEDDINGS: Qwen3Config = {
  ...TEST_CONFIG,
  tieWordEmbeddings: true,
};

describe('Qwen3 Parameter Validation', () => {
  describe('validateParameters', () => {
    it('should validate complete parameters successfully', () => {
      const model = new Qwen3Model(TEST_CONFIG);
      const params = model.getParameters();

      // Should not throw - all parameters are present and correct
      expect(() => model.validateParameters(params)).not.toThrow();
    });

    it('should detect missing embedding parameter', () => {
      const model = new Qwen3Model(TEST_CONFIG);
      const params = model.getParameters();

      // Remove a required parameter
      delete (params as Record<string, MxArray>)['embedding.weight'];

      // Should throw with helpful error message
      expect(() => model.validateParameters(params)).toThrow(/embedding\.weight/);
      expect(() => model.validateParameters(params)).toThrow(/Missing/);
    });

    it('should detect missing final_norm parameter', () => {
      const model = new Qwen3Model(TEST_CONFIG);
      const params = model.getParameters();

      delete (params as Record<string, MxArray>)['final_norm.weight'];

      expect(() => model.validateParameters(params)).toThrow(/final_norm\.weight/);
    });

    it('should detect missing lm_head parameter when not tied', () => {
      const model = new Qwen3Model(TEST_CONFIG);
      const params = model.getParameters();

      delete (params as Record<string, MxArray>)['lm_head.weight'];

      expect(() => model.validateParameters(params)).toThrow(/lm_head\.weight/);
    });

    it('should not require lm_head when embeddings are tied', () => {
      const model = new Qwen3Model(TEST_CONFIG_TIED_EMBEDDINGS);
      const params = model.getParameters();

      // With tied embeddings, lm_head.weight should not be required
      // (and shouldn't be in params anyway)
      expect(params['lm_head.weight']).toBeUndefined();

      // Should validate successfully
      expect(() => model.validateParameters(params)).not.toThrow();
    });

    it('should detect missing layer parameters', () => {
      const model = new Qwen3Model(TEST_CONFIG);
      const params = model.getParameters();

      // Remove a layer parameter
      delete (params as Record<string, MxArray>)['layers.0.self_attn.q_proj.weight'];

      expect(() => model.validateParameters(params)).toThrow(/layers\.0\.self_attn\.q_proj\.weight/);
    });

    it('should detect missing MLP parameters', () => {
      const model = new Qwen3Model(TEST_CONFIG);
      const params = model.getParameters();

      delete (params as Record<string, MxArray>)['layers.1.mlp.gate_proj.weight'];

      expect(() => model.validateParameters(params)).toThrow(/layers\.1\.mlp\.gate_proj\.weight/);
    });

    it('should detect shape mismatch in embedding', () => {
      const model = new Qwen3Model(TEST_CONFIG);
      const params = model.getParameters();

      // Replace with wrong shape - should be [vocab_size, hidden_size] = [1000, 64]
      (params as Record<string, MxArray>)['embedding.weight'] = MxArray.zeros(shape(500, 64)); // Wrong vocab_size

      expect(() => model.validateParameters(params)).toThrow(/embedding\.weight/);
      expect(() => model.validateParameters(params)).toThrow(/Shape mismatch/);
    });

    it('should detect shape mismatch in attention weights', () => {
      const model = new Qwen3Model(TEST_CONFIG);
      const params = model.getParameters();

      // Q projection should be [num_heads * head_dim, hidden_size] = [64, 64]
      (params as Record<string, MxArray>)['layers.0.self_attn.q_proj.weight'] = MxArray.zeros(shape(32, 64)); // Wrong output dim

      expect(() => model.validateParameters(params)).toThrow(/layers\.0\.self_attn\.q_proj\.weight/);
      expect(() => model.validateParameters(params)).toThrow(/Shape mismatch/);
    });

    it('should detect shape mismatch in MLP weights', () => {
      const model = new Qwen3Model(TEST_CONFIG);
      const params = model.getParameters();

      // gate_proj should be [intermediate_size, hidden_size] = [128, 64]
      (params as Record<string, MxArray>)['layers.0.mlp.gate_proj.weight'] = MxArray.zeros(shape(64, 64)); // Wrong output dim

      expect(() => model.validateParameters(params)).toThrow(/layers\.0\.mlp\.gate_proj\.weight/);
      expect(() => model.validateParameters(params)).toThrow(/Shape mismatch/);
    });

    it('should detect shape mismatch in layer norm', () => {
      const model = new Qwen3Model(TEST_CONFIG);
      const params = model.getParameters();

      // LayerNorm should be [hidden_size] = [64]
      (params as Record<string, MxArray>)['layers.0.input_layernorm.weight'] = MxArray.zeros(shape(32)); // Wrong size

      expect(() => model.validateParameters(params)).toThrow(/layers\.0\.input_layernorm\.weight/);
      expect(() => model.validateParameters(params)).toThrow(/Shape mismatch/);
    });

    it('should report multiple missing parameters', () => {
      const model = new Qwen3Model(TEST_CONFIG);
      const params = model.getParameters();

      // Remove multiple parameters
      delete (params as Record<string, MxArray>)['embedding.weight'];
      delete (params as Record<string, MxArray>)['layers.0.self_attn.q_proj.weight'];
      delete (params as Record<string, MxArray>)['layers.1.mlp.down_proj.weight'];

      try {
        model.validateParameters(params);
        expect.fail('Should have thrown');
      } catch (e) {
        const message = (e as Error).message;
        expect(message).toContain('Missing');
        expect(message).toContain('embedding.weight');
        expect(message).toContain('layers.0.self_attn.q_proj.weight');
        expect(message).toContain('layers.1.mlp.down_proj.weight');
      }
    });
  });

  describe('QK Norm Validation', () => {
    it('should require QK norm parameters when use_qk_norm is true', () => {
      const model = new Qwen3Model(TEST_CONFIG_WITH_QK_NORM);
      const params = model.getParameters();

      // Should have QK norm parameters
      expect(params['layers.0.self_attn.q_norm.weight']).toBeDefined();
      expect(params['layers.0.self_attn.k_norm.weight']).toBeDefined();

      // Should validate successfully
      expect(() => model.validateParameters(params)).not.toThrow();
    });

    it('should detect missing QK norm parameters', () => {
      const model = new Qwen3Model(TEST_CONFIG_WITH_QK_NORM);
      const params = model.getParameters();

      // Remove QK norm parameter
      delete (params as Record<string, MxArray>)['layers.0.self_attn.q_norm.weight'];

      expect(() => model.validateParameters(params)).toThrow(/layers\.0\.self_attn\.q_norm\.weight/);
    });

    it('should detect QK norm shape mismatch', () => {
      const model = new Qwen3Model(TEST_CONFIG_WITH_QK_NORM);
      const params = model.getParameters();

      // QK norm should be [head_dim] = [16]
      (params as Record<string, MxArray>)['layers.0.self_attn.q_norm.weight'] = MxArray.zeros(shape(32)); // Wrong size

      expect(() => model.validateParameters(params)).toThrow(/layers\.0\.self_attn\.q_norm\.weight/);
      expect(() => model.validateParameters(params)).toThrow(/Shape mismatch/);
    });

    it('should not require QK norm when use_qk_norm is false', () => {
      const model = new Qwen3Model(TEST_CONFIG); // QK norm disabled
      const params = model.getParameters();

      // Should not have QK norm parameters
      expect(params['layers.0.self_attn.q_norm.weight']).toBeUndefined();
      expect(params['layers.0.self_attn.k_norm.weight']).toBeUndefined();

      // Should validate successfully without QK norm params
      expect(() => model.validateParameters(params)).not.toThrow();
    });
  });

  describe('loadParameters QK Norm Consistency', () => {
    it('should error when loading params without QK norm but config has use_qk_norm=true', () => {
      // Create model with use_qk_norm=true
      const model = new Qwen3Model(TEST_CONFIG_WITH_QK_NORM);

      // Get params from a model without QK norm (missing the QK norm weights)
      const modelWithoutQkNorm = new Qwen3Model(TEST_CONFIG);
      const paramsWithoutQkNorm = modelWithoutQkNorm.getParameters();

      // Should throw when loading params that don't have QK norm weights
      // when the model's config says use_qk_norm=true
      expect(() => model.loadParameters(paramsWithoutQkNorm)).toThrow(/use_qk_norm=true/);
      expect(() => model.loadParameters(paramsWithoutQkNorm)).toThrow(/q_norm\.weight not found/);
    });

    it('should succeed when loading params with QK norm and config has use_qk_norm=true', () => {
      const model = new Qwen3Model(TEST_CONFIG_WITH_QK_NORM);
      const params = model.getParameters();

      // Should succeed - params have QK norm weights
      expect(() => model.loadParameters(params)).not.toThrow();
    });

    it('should succeed when loading params without QK norm and config has use_qk_norm=false', () => {
      const model = new Qwen3Model(TEST_CONFIG);
      const params = model.getParameters();

      // Should succeed - both config and params agree: no QK norm
      expect(() => model.loadParameters(params)).not.toThrow();
    });
  });

  describe('Expected Parameter Count', () => {
    it('should validate correct parameter count without QK norm', () => {
      const model = new Qwen3Model(TEST_CONFIG);
      const params = model.getParameters();

      // Expected: 1 embedding + 2 layers * 9 params each + 1 final_norm + 1 lm_head = 21
      const expectedCount = 1 + TEST_CONFIG.numLayers * 9 + 1 + 1;
      expect(Object.keys(params).length).toBe(expectedCount);
    });

    it('should validate correct parameter count with QK norm', () => {
      const model = new Qwen3Model(TEST_CONFIG_WITH_QK_NORM);
      const params = model.getParameters();

      // Expected: 1 embedding + 2 layers * 11 params each + 1 final_norm + 1 lm_head = 25
      // (11 = 9 base + 2 QK norm)
      const expectedCount = 1 + TEST_CONFIG_WITH_QK_NORM.numLayers * 11 + 1 + 1;
      expect(Object.keys(params).length).toBe(expectedCount);
    });

    it('should validate correct parameter count with tied embeddings', () => {
      const model = new Qwen3Model(TEST_CONFIG_TIED_EMBEDDINGS);
      const params = model.getParameters();

      // Expected: 1 embedding + 2 layers * 9 params each + 1 final_norm = 20 (no lm_head)
      const expectedCount = 1 + TEST_CONFIG_TIED_EMBEDDINGS.numLayers * 9 + 1;
      expect(Object.keys(params).length).toBe(expectedCount);
    });
  });
});
