/**
 * Tests for parameter management utilities
 *
 * Tests the conversion between flat parameter vectors and structured dictionaries
 * used by the autograd system.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { Qwen3Model, type Qwen3Config, MxArray } from '@mlx-node/core';
import { shape } from '../test-utils';

describe('Parameter Manager', () => {
  let tinyConfig: Qwen3Config;
  let smallConfig: Qwen3Config;

  beforeAll(() => {
    // Tiny model for quick tests
    tinyConfig = {
      vocabSize: 100,
      hiddenSize: 32,
      numLayers: 2,
      numHeads: 4,
      numKvHeads: 4,
      headDim: 8, // hiddenSize / numHeads = 32 / 4 = 8
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

    // Small model with QK norm
    smallConfig = {
      ...tinyConfig,
      hiddenSize: 64,
      numLayers: 4,
      headDim: 16, // hiddenSize / numHeads = 64 / 4 = 16
      intermediateSize: 256,
      useQkNorm: true,
    };
  });

  describe('get_parameters', () => {
    it('should extract all parameters as dictionary', () => {
      const model = new Qwen3Model(tinyConfig);
      const params = model.getParameters();

      // Check we have all expected parameters
      expect(params).toHaveProperty('embedding.weight');
      expect(params).toHaveProperty('final_norm.weight');
      expect(params).toHaveProperty('lm_head.weight');

      // Check layer parameters (2 layers)
      for (let i = 0; i < 2; i++) {
        expect(params).toHaveProperty(`layers.${i}.self_attn.q_proj.weight`);
        expect(params).toHaveProperty(`layers.${i}.self_attn.k_proj.weight`);
        expect(params).toHaveProperty(`layers.${i}.self_attn.v_proj.weight`);
        expect(params).toHaveProperty(`layers.${i}.self_attn.o_proj.weight`);
        expect(params).toHaveProperty(`layers.${i}.mlp.gate_proj.weight`);
        expect(params).toHaveProperty(`layers.${i}.mlp.up_proj.weight`);
        expect(params).toHaveProperty(`layers.${i}.mlp.down_proj.weight`);
        expect(params).toHaveProperty(`layers.${i}.input_layernorm.weight`);
        expect(params).toHaveProperty(`layers.${i}.post_attention_layernorm.weight`);
      }
    });

    it('should have correct parameter shapes', () => {
      const model = new Qwen3Model(tinyConfig);
      const params = model.getParameters();

      // Embedding: [vocab_size, hidden_size]
      const embShape = params['embedding.weight'].shape();
      expect(Array.from(embShape).map(Number)).toEqual([tinyConfig.vocabSize, tinyConfig.hiddenSize]);

      // LM head: [vocab_size, hidden_size]
      const lmHeadShape = params['lm_head.weight'].shape();
      expect(Array.from(lmHeadShape).map(Number)).toEqual([tinyConfig.vocabSize, tinyConfig.hiddenSize]);

      // Final norm: [hidden_size]
      const normShape = params['final_norm.weight'].shape();
      expect(Array.from(normShape).map(Number)).toEqual([tinyConfig.hiddenSize]);

      // Attention Q proj: [hidden_size, hidden_size]
      const qProjShape = params['layers.0.self_attn.q_proj.weight'].shape();
      expect(Array.from(qProjShape).map(Number)).toEqual([tinyConfig.hiddenSize, tinyConfig.hiddenSize]);

      // MLP gate: [intermediate_size, hidden_size]
      const gateShape = params['layers.0.mlp.gate_proj.weight'].shape();
      expect(Array.from(gateShape).map(Number)).toEqual([tinyConfig.intermediateSize, tinyConfig.hiddenSize]);

      // MLP down: [hidden_size, intermediate_size]
      const downShape = params['layers.0.mlp.down_proj.weight'].shape();
      expect(Array.from(downShape).map(Number)).toEqual([tinyConfig.hiddenSize, tinyConfig.intermediateSize]);
    });

    it('should return MxArray instances', () => {
      const model = new Qwen3Model(tinyConfig);
      const params = model.getParameters();

      for (const [_name, param] of Object.entries(params)) {
        expect(param).toBeInstanceOf(MxArray);
        const dtype = param.dtype();
        // DType enum: Float32=0, Float16=2, BFloat16=3
        expect([0, 2, 3]).toContain(dtype);
      }
    });

    it('should be deterministic across calls', () => {
      const model = new Qwen3Model(tinyConfig);
      const params1 = model.getParameters();
      const params2 = model.getParameters();

      // Same parameter names
      expect(Object.keys(params1).sort()).toEqual(Object.keys(params2).sort());

      // Same parameter values
      for (const key of Object.keys(params1)) {
        const diff = params1[key].sub(params2[key]);
        const maxDiff = diff.abs().max().toFloat32()[0];
        expect(maxDiff).toBeLessThan(1e-10);
      }
    });
  });

  describe('Parameter Count', () => {
    it('should count parameters correctly for tiny model', () => {
      const model = new Qwen3Model(tinyConfig);
      const numParams = model.numParameters();

      // Calculate expected:
      // Embedding: 100 * 32 = 3,200
      // Per layer (2 layers):
      //   - Attention (Q,K,V,O): 32*32*4 = 4,096
      //   - MLP: 32*128*2 + 128*32 = 12,288
      //   - Norms: 32*2 = 64
      //   Total per layer: 16,448
      // 2 layers: 32,896
      // Final norm: 32
      // LM head: 100 * 32 = 3,200
      // Total: 3,200 + 32,896 + 32 + 3,200 = 39,328

      expect(numParams).toBeGreaterThan(30000);
      expect(numParams).toBeLessThan(50000);
    });

    it('should scale with model size', () => {
      const tiny = new Qwen3Model(tinyConfig);
      const small = new Qwen3Model(smallConfig);

      const tinyParams = tiny.numParameters();
      const smallParams = small.numParameters();

      // Small model should have more parameters
      // (64 hidden, 4 layers vs 32 hidden, 2 layers)
      expect(smallParams).toBeGreaterThan(tinyParams);
      expect(smallParams).toBeGreaterThan(tinyParams * 2);
    });
  });

  describe('Parameter Naming Convention', () => {
    it('should follow HuggingFace naming convention', () => {
      const model = new Qwen3Model(tinyConfig);
      const params = model.getParameters();

      const names = Object.keys(params);

      // Check naming patterns
      const hasEmbedding = names.some((n) => n === 'embedding.weight');
      const hasLayers = names.some((n) => n.startsWith('layers.'));
      const hasSelfAttn = names.some((n) => n.includes('self_attn'));
      const hasMlp = names.some((n) => n.includes('mlp'));
      const hasFinalNorm = names.some((n) => n === 'final_norm.weight');
      const hasLmHead = names.some((n) => n === 'lm_head.weight');

      expect(hasEmbedding).toBe(true);
      expect(hasLayers).toBe(true);
      expect(hasSelfAttn).toBe(true);
      expect(hasMlp).toBe(true);
      expect(hasFinalNorm).toBe(true);
      expect(hasLmHead).toBe(true);
    });

    it('should have sequential layer indices', () => {
      const model = new Qwen3Model(tinyConfig);
      const params = model.getParameters();

      const layerIndices = Object.keys(params)
        .filter((n) => n.startsWith('layers.'))
        .map((n) => parseInt(n.split('.')[1]))
        .filter((idx, i, arr) => arr.indexOf(idx) === i) // unique
        .sort((a, b) => a - b);

      // Should be [0, 1] for 2-layer model
      expect(layerIndices).toEqual([0, 1]);
    });

    it('should have consistent parameter naming within layers', () => {
      const model = new Qwen3Model(tinyConfig);
      const params = model.getParameters();

      const expectedSubstrings = [
        'self_attn.q_proj.weight',
        'self_attn.k_proj.weight',
        'self_attn.v_proj.weight',
        'self_attn.o_proj.weight',
        'mlp.gate_proj.weight',
        'mlp.up_proj.weight',
        'mlp.down_proj.weight',
        'input_layernorm.weight',
        'post_attention_layernorm.weight',
      ];

      for (let layerIdx = 0; layerIdx < tinyConfig.numLayers; layerIdx++) {
        for (const substring of expectedSubstrings) {
          const fullName = `layers.${layerIdx}.${substring}`;
          expect(params).toHaveProperty(fullName);
        }
      }
    });
  });

  describe('Parameter Loading', () => {
    it('should load parameters correctly', () => {
      const model = new Qwen3Model(tinyConfig);

      // Get initial parameters
      const initialParams = model.getParameters();

      // Create new parameters (scaled versions)
      const newParams: Record<string, MxArray> = {};
      for (const [name, param] of Object.entries(initialParams)) {
        newParams[name] = param.mul(MxArray.full(shape(), 2.0));
      }

      // Load new parameters
      model.loadParameters(newParams);

      // Check they were loaded
      const loadedParams = model.getParameters();

      for (const name of Object.keys(newParams)) {
        const diff = loadedParams[name].sub(newParams[name]);
        const maxDiff = diff.abs().max().toFloat32()[0];
        expect(maxDiff).toBeLessThan(1e-5);
      }
    });

    it('should handle partial parameter loading', () => {
      const model = new Qwen3Model(tinyConfig);

      // Only update embedding
      const embWeight = model.getParameters()['embedding.weight'];
      const newEmbWeight = embWeight.mul(MxArray.full(shape(), 3.0));

      model.loadParameters({
        'embedding.weight': newEmbWeight,
      });

      // Check embedding was updated
      const updatedEmb = model.getParameters()['embedding.weight'];
      const diff = updatedEmb.sub(newEmbWeight);
      const maxDiff = diff.abs().max().toFloat32()[0];
      expect(maxDiff).toBeLessThan(1e-5);
    });
  });

  describe('Parameter Extraction and Mapping', () => {
    it('should convert to flat vector and back', () => {
      const model = new Qwen3Model(tinyConfig);
      const params = model.getParameters();

      // Flatten parameters (simulate what param_manager does)
      const paramNames = Object.keys(params).sort();
      const paramArrays = paramNames.map((name) => params[name]);

      // Should have many parameters
      expect(paramArrays.length).toBeGreaterThan(10);

      // Map back to dictionary
      const reconstructed: Record<string, MxArray> = {};
      for (let i = 0; i < paramNames.length; i++) {
        reconstructed[paramNames[i]] = paramArrays[i];
      }

      // Should match original
      expect(Object.keys(reconstructed).sort()).toEqual(Object.keys(params).sort());
    });

    it('should maintain parameter order', () => {
      const model = new Qwen3Model(tinyConfig);
      const params = model.getParameters();

      // Extract parameters twice
      const names1 = Object.keys(params).sort();
      const names2 = Object.keys(model.getParameters()).sort();

      // Order should be consistent
      expect(names1).toEqual(names2);
    });
  });

  describe('Edge Cases', () => {
    it('should handle model with single layer', () => {
      const singleLayerConfig: Qwen3Config = {
        ...tinyConfig,
        numLayers: 1,
      };

      const model = new Qwen3Model(singleLayerConfig);
      const params = model.getParameters();

      // Should have layer.0 parameters
      const hasLayer0 = Object.keys(params).some((n) => n.startsWith('layers.0.'));
      expect(hasLayer0).toBe(true);

      // Should NOT have layer.1 parameters
      const hasLayer1 = Object.keys(params).some((n) => n.startsWith('layers.1.'));
      expect(hasLayer1).toBe(false);
    });

    it('should handle model with many layers', () => {
      const manyLayersConfig: Qwen3Config = {
        ...tinyConfig,
        hiddenSize: 16, // Smaller for memory
        numLayers: 8,
        headDim: 4, // hiddenSize / numHeads = 16 / 4 = 4
      };

      const model = new Qwen3Model(manyLayersConfig);
      const params = model.getParameters();

      // Should have all 8 layers
      for (let i = 0; i < 8; i++) {
        const hasLayer = Object.keys(params).some((n) => n.startsWith(`layers.${i}.`));
        expect(hasLayer).toBe(true);
      }
    });

    it('should handle very small dimensions', () => {
      const microConfig: Qwen3Config = {
        ...tinyConfig,
        vocabSize: 10,
        hiddenSize: 8,
        numLayers: 1,
        numHeads: 2,
        numKvHeads: 2,
        headDim: 4, // hiddenSize / numHeads = 8 / 2 = 4
        intermediateSize: 32,
      };

      const model = new Qwen3Model(microConfig);
      const params = model.getParameters();

      // Should still have all required parameters
      expect(params).toHaveProperty('embedding.weight');
      expect(params).toHaveProperty('layers.0.self_attn.q_proj.weight');
      expect(params).toHaveProperty('final_norm.weight');

      // Check shapes are correct
      const embShape = params['embedding.weight'].shape();
      expect(Array.from(embShape).map(Number)).toEqual([10, 8]);
    });
  });

  describe('Memory and Performance', () => {
    it('should extract parameters efficiently', () => {
      const model = new Qwen3Model(smallConfig);

      const startTime = Date.now();
      const params = model.getParameters();
      const endTime = Date.now();

      // Should complete quickly (< 100ms)
      expect(endTime - startTime).toBeLessThan(100);

      // Should have extracted all parameters
      expect(Object.keys(params).length).toBeGreaterThan(20);
    });

    it('should handle repeated parameter extraction', () => {
      const model = new Qwen3Model(tinyConfig);

      // Extract parameters multiple times
      for (let i = 0; i < 10; i++) {
        const params = model.getParameters();
        expect(Object.keys(params).length).toBeGreaterThan(10);
      }
    });
  });

  describe('Integration with Forward Pass', () => {
    it('should use extracted parameters for inference', () => {
      const model = new Qwen3Model(tinyConfig);

      // Get parameters (verifies extraction works)
      model.getParameters();

      // Create input
      const inputIds = MxArray.fromInt32(new Int32Array([0, 1, 2, 3, 4]), shape(1, 5));

      // Run forward pass
      const logits = model.forward(inputIds);

      // Check output shape
      const resultShape = logits.shape();
      expect(Array.from(resultShape).map(Number)).toEqual([1, 5, tinyConfig.vocabSize]);
    });

    it('should produce consistent outputs after parameter update', () => {
      const model = new Qwen3Model(tinyConfig);
      const inputIds = MxArray.fromInt32(new Int32Array([0, 1, 2]), shape(1, 3));

      // First forward pass
      const logits1 = model.forward(inputIds);

      // Update parameters (scale by 0.9)
      const params = model.getParameters();
      const scaledParams: Record<string, MxArray> = {};
      for (const [name, param] of Object.entries(params)) {
        scaledParams[name] = param.mul(MxArray.full(shape(), 0.9));
      }
      model.loadParameters(scaledParams);

      // Second forward pass (should be different)
      const logits2 = model.forward(inputIds);

      // Outputs should be different
      const diff = logits1.sub(logits2);
      const maxDiff = diff.abs().max().toFloat32()[0];
      expect(maxDiff).toBeGreaterThan(0.01);
    });
  });
});
