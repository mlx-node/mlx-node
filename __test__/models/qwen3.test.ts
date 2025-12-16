import { describe, it, expect } from 'vitest';
import { QWEN3_CONFIGS, getQwen3Config } from '@mlx-node/lm';
import { MxArray, Qwen3Model } from '@mlx-node/core';
import { shape } from '../test-utils';

describe.sequential('Qwen3 Model', () => {
  describe('Model Configuration', () => {
    it('should have correct default configurations', () => {
      expect(QWEN3_CONFIGS['qwen3-0.6b']).toBeDefined();
      expect(QWEN3_CONFIGS['qwen3-1.8b']).toBeDefined();
      expect(QWEN3_CONFIGS['qwen3-7b']).toBeDefined();

      const config = QWEN3_CONFIGS['qwen3-0.6b'];
      expect(config.hiddenSize).toBe(1024);
      expect(config.numLayers).toBe(28);
      expect(config.numHeads).toBe(16);
      expect(config.numKvHeads).toBe(8); // GQA with 2:1 ratio
      expect(config.useQkNorm).toBe(true); // Qwen3 always uses QK normalization
    });
  });

  describe('Model Instantiation', () => {
    it('should create model from string config', () => {
      const model = new Qwen3Model(QWEN3_CONFIGS['qwen3-0.6b']);
      expect(model).toBeDefined();
      expect(model.getConfig().hiddenSize).toBe(1024);
    });

    it('should create model from custom config', () => {
      const customConfig = {
        vocabSize: 1000,
        hiddenSize: 256,
        numLayers: 4,
        numHeads: 4,
        numKvHeads: 2,
        headDim: 64, // hiddenSize / numHeads = 256 / 4 = 64
        intermediateSize: 1024,
        rmsNormEps: 1e-6,
        ropeTheta: 10000.0,
        maxPositionEmbeddings: 512,
        useQkNorm: true,
        tieWordEmbeddings: false,
        padTokenId: 0,
        eosTokenId: 1,
        bosTokenId: 0,
      };

      const model = new Qwen3Model(customConfig);
      expect(model).toBeDefined();
      expect(model.getConfig().hiddenSize).toBe(256);
    });

    it('should throw error for unknown config', () => {
      expect(() => getQwen3Config('unknown-model')).toThrow();
    });
  });

  describe('Forward Pass', () => {
    it('should perform forward pass without cache', () => {
      const model = new Qwen3Model(QWEN3_CONFIGS['qwen3-0.6b']);

      // Create dummy input
      const batchSize = 2;
      const seqLen = 10;
      const inputIds = MxArray.randint(shape(batchSize, seqLen), 0, model.getConfig().vocabSize);

      // Forward pass
      const logits = model.forward(inputIds);

      // Check output shape
      expect(logits).toBeDefined();
      // Shape should be (batch, seq_len, vocab_size)
      // Note: We'd need a shape comparison method here
      // expectedShape = shape(batchSize, seqLen, model.getConfig().vocabSize);
    });

    it('should perform forward pass with KV cache', () => {
      const model = new Qwen3Model(QWEN3_CONFIGS['qwen3-0.6b']);

      // Initialize KV caches
      model.initKvCaches();

      // Create dummy input
      const batchSize = 1;
      const seqLen = 5;
      const inputIds = MxArray.randint(shape(batchSize, seqLen), 0, model.getConfig().vocabSize);

      // First forward pass with cache (processes full prompt)
      const logits1 = model.forwardWithCache(inputIds, true);
      expect(logits1).toBeDefined();
      const logits1Shape = logits1.shape();
      expect(Number(logits1Shape[0])).toBe(batchSize);
      expect(Number(logits1Shape[1])).toBe(seqLen);

      // Second forward pass with cache (single new token)
      const newTokens = MxArray.randint(shape(batchSize, 1), 0, model.getConfig().vocabSize);
      const logits2 = model.forwardWithCache(newTokens, true);
      expect(logits2).toBeDefined();
      const logits2Shape = logits2.shape();
      expect(Number(logits2Shape[0])).toBe(batchSize);
      expect(Number(logits2Shape[1])).toBe(1); // Single token output

      // Reset caches
      model.resetKvCaches();
    });
  });

  describe('Text Generation', () => {
    it('should generate text from prompt (requires model files)', async () => {
      // Note: This test requires a real pretrained model with tokenizer
      // It will be skipped if the model is not available
      const modelPath = process.env.QWEN3_MODEL_PATH;

      if (!modelPath) {
        console.log('  ⏭️  Skipping text generation test (set QWEN3_MODEL_PATH to enable)');
        return;
      }

      // Load model with tokenizer
      const model = await Qwen3Model.loadPretrained(modelPath);

      // Generate text using the new message-based API
      const messages = [{ role: 'user', content: 'Hello, how are you?' }];
      const result = await model.generate(messages, {
        maxNewTokens: 20,
        temperature: 0.8,
        topK: 50,
        topP: 0.95,
      });

      // Verify result structure
      expect(result).toBeDefined();
      expect(result.text).toBeDefined();
      expect(typeof result.text).toBe('string');
      expect(result.text.length).toBeGreaterThan(0);

      // These are also available for GRPO training
      expect(result.tokens).toBeDefined();
      expect(result.logprobs).toBeDefined();
      expect(result.finishReason).toBeDefined();
      expect(['eos', 'length']).toContain(result.finishReason);
      expect(result.numTokens).toBeGreaterThanOrEqual(0);
      expect(result.numTokens).toBeLessThanOrEqual(20);
    });
  });

  describe('Model Components', () => {
    it('should provide access to model parameters via getParameters()', () => {
      // After Rust migration, use getParameters() to access all model weights
      const model = new Qwen3Model(QWEN3_CONFIGS['qwen3-0.6b']);
      const params = model.getParameters();

      expect(params).toBeDefined();
      expect(typeof params).toBe('object');

      // Should have key model parameters
      expect(params['lm_head.weight']).toBeDefined();
      expect(params['final_norm.weight']).toBeDefined();

      // Should have first layer attention parameters
      expect(params['layers.0.self_attn.q_proj.weight']).toBeDefined();
      expect(params['layers.0.self_attn.k_proj.weight']).toBeDefined();
      expect(params['layers.0.self_attn.v_proj.weight']).toBeDefined();
      expect(params['layers.0.self_attn.o_proj.weight']).toBeDefined();

      // Parameters should be MxArrays with proper shapes
      const lmHeadWeight = params['lm_head.weight'];
      expect(lmHeadWeight).toBeDefined();
      const shape = lmHeadWeight.shape();
      expect(shape.length).toBe(2); // [vocab_size, hidden_size]
    });
  });

  describe('Qwen3 Specific Features', () => {
    it('should have correct attention configuration for Qwen3', () => {
      const model = new Qwen3Model(QWEN3_CONFIGS['qwen3-0.6b']);
      const config = model.getConfig();

      // Qwen3 always uses QK normalization (core architectural feature)
      expect(config.useQkNorm).toBe(true);
    });

    it('should have correct GQA configuration', () => {
      const model = new Qwen3Model(QWEN3_CONFIGS['qwen3-0.6b']);
      const config = model.getConfig();

      // Qwen3-0.6b uses 16 query heads and 8 KV heads (2:1 ratio)
      expect(config.numHeads).toBe(16);
      expect(config.numKvHeads).toBe(8);
      expect(config.numHeads % config.numKvHeads).toBe(0);
    });

    it('should have high RoPE theta for long context', () => {
      const model = new Qwen3Model(QWEN3_CONFIGS['qwen3-0.6b']);
      const config = model.getConfig();

      // Qwen3 uses a much higher RoPE theta for long context
      expect(config.ropeTheta).toBe(1000000.0);
      expect(config.maxPositionEmbeddings).toBeGreaterThanOrEqual(40960);
    });
  });
});

describe.sequential('GRPO Integration', () => {
  it('should compute loss for training', () => {
    const model = new Qwen3Model(QWEN3_CONFIGS['qwen3-0.6b']);

    // Create dummy batch
    const batchSize = 2;
    const seqLen = 10;
    const inputIds = MxArray.randint(shape(batchSize, seqLen), 0, model.getConfig().vocabSize);
    const labels = MxArray.randint(shape(batchSize, seqLen), 0, model.getConfig().vocabSize);

    // Compute loss
    const loss = model.computeLoss(inputIds, labels);
    expect(loss).toBeDefined();
    // Loss should be a scalar
  });
});
