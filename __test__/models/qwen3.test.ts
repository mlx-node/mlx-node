import fs from 'node:fs';
import path from 'node:path';

import { Qwen3Model } from '@mlx-node/core';
import { QWEN3_CONFIGS, getQwen3Config } from '@mlx-node/lm';
import { describe, it, expect, beforeAll } from 'vite-plus/test';

describe('Qwen3 Model', () => {
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
    // Same checkpoint resolution as the sibling real-weights tests
    // (`qwen3-generate-batch*.test.ts`, `qwen3-chunked-prefill.test.ts`):
    // `QWEN3_MODEL_PATH` first, then the converted checkpoint the `test` CI
    // job produces at `.cache/models/qwen3-0.6b-mlx-bf16`. Skips when neither
    // exists. Loaded once for every test in this block.
    let model: Qwen3Model | null = null;

    beforeAll(async () => {
      const envModelPath = process.env.QWEN3_MODEL_PATH;
      const defaultModelPath = path.join(process.cwd(), '.cache/models/qwen3-0.6b-mlx-bf16');

      let modelPath: string | null = null;
      if (envModelPath && fs.existsSync(envModelPath)) {
        modelPath = envModelPath;
      } else if (fs.existsSync(defaultModelPath)) {
        modelPath = defaultModelPath;
      } else {
        console.log('  ⏭️  Skipping text generation tests (no model found)');
        console.log(`     Set QWEN3_MODEL_PATH or place model at: ${defaultModelPath}`);
        return;
      }

      model = await Qwen3Model.load(modelPath);
    }, 60_000);

    it('should generate text from prompt (requires model files)', async () => {
      if (!model) return;

      // Generate text using the message-based API
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

    it('should reject a nonpositive maxNewTokens budget (parity with Qwen3.5)', async () => {
      // The public generate() API rejects a nonpositive budget (Err)
      // instead of panicking the model thread on Vec::with_capacity(-1 as
      // usize) == usize::MAX. Requires a real model since the guard runs
      // inside generate_sync on the model thread (post-load).
      if (!model) return;

      const messages = [{ role: 'user', content: 'Hello' }];

      // Both 0 and a negative budget must reject (not panic, not resolve).
      await expect(model.generate(messages, { maxNewTokens: 0 })).rejects.toThrow(/max_new_tokens must be > 0/);
      await expect(model.generate(messages, { maxNewTokens: -1 })).rejects.toThrow(/max_new_tokens must be > 0/);
    });
  });
});
