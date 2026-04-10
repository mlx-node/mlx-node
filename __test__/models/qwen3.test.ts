import { Qwen3Model } from '@mlx-node/core';
import { QWEN3_CONFIGS, getQwen3Config } from '@mlx-node/lm';
import { describe, it, expect } from 'vite-plus/test';

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
      expect(config.ropeTheta).toBe(1000000.0);
      expect(config.maxPositionEmbeddings).toBeGreaterThanOrEqual(40960);
    });

    it('should throw error for unknown config', () => {
      expect(() => getQwen3Config('unknown-model')).toThrow();
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
      const model = await Qwen3Model.load(modelPath);

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
      expect(['stop', 'length']).toContain(result.finishReason);
      expect(result.numTokens).toBeGreaterThanOrEqual(0);
      expect(result.numTokens).toBeLessThanOrEqual(20);
    });
  });
});
