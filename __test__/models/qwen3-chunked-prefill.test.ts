import { describe, it, expect, beforeAll } from 'vite-plus/test';
import { Qwen3Model, type ChatMessage } from '@mlx-node/core';
import fs from 'node:fs';
import path from 'node:path';

/**
 * Tests for chunked prefill functionality.
 *
 * Chunked prefill processes long prompts in chunks rather than all at once,
 * which improves memory efficiency and enables async pipelining.
 *
 * Note: The chunked prefill feature is implemented in the internal Rust methods
 * (generateForTrainingSync, generateBatchForTrainingSync) which are called by
 * the public async APIs (generate, generateBatch). Since the sync methods are
 * not directly exposed to JavaScript, we test through the async APIs.
 */
describe('Qwen3Model - Chunked Prefill', () => {
  describe('Integration Tests (with pretrained model)', () => {
    let model: Qwen3Model | null = null;

    beforeAll(async () => {
      const envModelPath = process.env.QWEN3_MODEL_PATH;
      const defaultModelPath = path.join(process.cwd(), '.cache/models/qwen3-0.6b-mlx-bf16');

      let modelPath: string | null = null;

      if (envModelPath && fs.existsSync(envModelPath)) {
        modelPath = envModelPath;
        console.log(`  Loading model from QWEN3_MODEL_PATH: ${modelPath}`);
      } else if (fs.existsSync(defaultModelPath)) {
        modelPath = defaultModelPath;
        console.log(`  Loading model from default path: ${modelPath}`);
      } else {
        console.log('  Skipping integration tests (no pretrained model found)');
        return;
      }

      model = await Qwen3Model.loadPretrained(modelPath);
    }, 60000); // 60s timeout for model loading

    it('should generate successfully with default prefill_step_size', async () => {
      if (!model) {
        console.log('  Skipping (no model loaded)');
        return;
      }

      const messages: ChatMessage[] = [{ role: 'user', content: 'What is 2+2?' }];

      // Generate with default settings (prefill_step_size = 2048)
      const result = await model.generate(messages, {
        maxNewTokens: 5,
        temperature: 0.0,
        prefillStepSize: 2048, // Explicit default
      });

      expect(result.numTokens).toBeGreaterThan(0);
      console.log(`  Generated ${result.numTokens} tokens: "${result.text.trim()}"`);
    });

    it('should generate successfully with chunking disabled', async () => {
      if (!model) {
        console.log('  Skipping (no model loaded)');
        return;
      }

      const messages: ChatMessage[] = [{ role: 'user', content: 'What is 2+2?' }];

      // Generate without chunking (prefill_step_size = 0)
      const result = await model.generate(messages, {
        maxNewTokens: 5,
        temperature: 0.0,
        prefillStepSize: 0, // Disable chunking
      });

      expect(result.numTokens).toBeGreaterThan(0);
      console.log(`  Generated ${result.numTokens} tokens: "${result.text.trim()}"`);
    });

    it('should produce identical output with chunked vs non-chunked prefill', async () => {
      if (!model) {
        console.log('  Skipping (no model loaded)');
        return;
      }

      const messages: ChatMessage[] = [{ role: 'user', content: 'What is the capital of France? Answer in one word.' }];

      // Generate with default chunking (2048)
      const result1 = await model.generate(messages, {
        maxNewTokens: 5,
        temperature: 0.0,
        prefillStepSize: 2048,
      });

      // Generate without chunking
      const result2 = await model.generate(messages, {
        maxNewTokens: 5,
        temperature: 0.0,
        prefillStepSize: 0,
      });

      // With deterministic sampling (temp=0), results should be identical
      // Note: Both will use single-pass prefill since prompt < 2048 tokens
      const tokens1 = result1.tokens.toUint32();
      const tokens2 = result2.tokens.toUint32();

      expect(tokens1.length).toBe(tokens2.length);
      for (let i = 0; i < tokens1.length; i++) {
        expect(tokens1[i]).toBe(tokens2[i]);
      }

      console.log(`  Both produced ${tokens1.length} identical tokens`);
    });

    it('should produce identical output with various chunk sizes', async () => {
      if (!model) {
        console.log('  Skipping (no model loaded)');
        return;
      }

      // Create a longer prompt to test chunking
      const longContent = Array(100).fill('Hello world.').join(' ');
      const messages: ChatMessage[] = [{ role: 'user', content: longContent }];

      const chunkSizes = [64, 128, 256, 512, 1024, 2048];
      const results: Uint32Array[] = [];

      for (const chunkSize of chunkSizes) {
        const result = await model.generate(messages, {
          maxNewTokens: 3,
          temperature: 0.0,
          prefillStepSize: chunkSize,
        });
        results.push(result.tokens.toUint32());
        console.log(`  Chunk size ${chunkSize}: generated ${result.numTokens} tokens`);
      }

      // All results should be identical with deterministic sampling
      const baseline = results[0];
      for (let i = 1; i < results.length; i++) {
        expect(results[i].length).toBe(baseline.length);
        for (let j = 0; j < baseline.length; j++) {
          expect(results[i][j]).toBe(baseline[j]);
        }
      }

      console.log(`  All ${chunkSizes.length} chunk sizes produced identical output`);
    });

    it('should handle very small chunk sizes (stress test)', async () => {
      if (!model) {
        console.log('  Skipping (no model loaded)');
        return;
      }

      const messages: ChatMessage[] = [{ role: 'user', content: Array(20).fill('Testing small chunks.').join(' ') }];

      // Very small chunk size forces many iterations
      const result = await model.generate(messages, {
        maxNewTokens: 3,
        temperature: 0.0,
        prefillStepSize: 32,
      });

      expect(result.numTokens).toBeGreaterThan(0);
      console.log(`  Small chunks (32): generated ${result.numTokens} tokens`);
    });

    it('should work with batch generation and various chunk sizes', async () => {
      if (!model) {
        console.log('  Skipping (no model loaded)');
        return;
      }

      const prompts: ChatMessage[][] = [
        [{ role: 'user', content: 'Count to 3' }],
        [{ role: 'user', content: Array(30).fill('Hello').join(' ') }],
        [{ role: 'user', content: 'What is the capital of France?' }],
      ];

      // Test batch generation with small chunk size
      const result = await model.generateBatch(prompts, 2, {
        maxNewTokens: 3,
        temperature: 0.5,
        prefillStepSize: 64,
      });

      // 3 prompts * 2 completions = 6 total
      expect(result.tokens.length).toBe(6);
      expect(result.logprobs.length).toBe(6);
      console.log(`  Batch with chunking: ${result.tokens.length} completions`);
    });

    it('should maintain generation quality with chunked prefill', async () => {
      if (!model) {
        console.log('  Skipping (no model loaded)');
        return;
      }

      const messages: ChatMessage[] = [{ role: 'user', content: 'What is 2+2? Answer:' }];

      // Generate with small chunks (stress test)
      const result = await model.generate(messages, {
        maxNewTokens: 10,
        temperature: 0.0,
        prefillStepSize: 8, // Very small chunks
      });

      console.log(`  Response: "${result.text.trim()}"`);

      // Basic sanity check: should generate some output
      // The actual content may vary (could be thinking, direct answer, etc.)
      expect(result.numTokens).toBeGreaterThan(0);
      expect(result.text.length).toBeGreaterThan(0);
    });
  });
});
