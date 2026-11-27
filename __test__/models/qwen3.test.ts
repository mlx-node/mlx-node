import fs from 'node:fs';
import path from 'node:path';

import { GrpoTrainingEngine, Qwen3Model, type ChatMessage } from '@mlx-node/core';
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

  // Shared pretrained Qwen3-0.6B checkpoint loaded once for Text Generation,
  // Chunked Prefill, generateBatch, and Parallel Batch Generation suites.
  // Resolution order: `QWEN3_MODEL_PATH` first, then the converted checkpoint
  // the `test` CI job produces at `.cache/models/qwen3-0.6b-mlx-bf16`. Skips
  // when neither exists.
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
      console.log('  ⏭️  Skipping Qwen3 pretrained model tests (no model found)');
      console.log(`     Set QWEN3_MODEL_PATH or place model at: ${defaultModelPath}`);
      return;
    }

    model = await Qwen3Model.load(modelPath);
  }, 60_000);

  describe('Text Generation', () => {
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

  /**
   * Tests for chunked prefill functionality.
   *
   * Chunked prefill processes long prompts in chunks rather than all at once,
   * which improves memory efficiency and enables async pipelining.
   */
  describe('Chunked Prefill', () => {
    it('should generate successfully with default prefill_step_size', async () => {
      if (!model) return;

      const messages: ChatMessage[] = [{ role: 'user', content: 'What is 2+2?' }];

      const result = await model.generate(messages, {
        maxNewTokens: 5,
        temperature: 0.0,
        prefillStepSize: 2048,
      });

      expect(result.numTokens).toBeGreaterThan(0);
    });

    it('should generate successfully with chunking disabled', async () => {
      if (!model) return;

      const messages: ChatMessage[] = [{ role: 'user', content: 'What is 2+2?' }];

      const result = await model.generate(messages, {
        maxNewTokens: 5,
        temperature: 0.0,
        prefillStepSize: 0,
      });

      expect(result.numTokens).toBeGreaterThan(0);
    });

    it('should produce identical output with chunked vs non-chunked prefill', async () => {
      if (!model) return;

      const messages: ChatMessage[] = [{ role: 'user', content: 'What is the capital of France? Answer in one word.' }];

      const result1 = await model.generate(messages, {
        maxNewTokens: 5,
        temperature: 0.0,
        prefillStepSize: 2048,
      });

      const result2 = await model.generate(messages, {
        maxNewTokens: 5,
        temperature: 0.0,
        prefillStepSize: 0,
      });

      const tokens1 = result1.tokens.toUint32();
      const tokens2 = result2.tokens.toUint32();

      expect(tokens1.length).toBe(tokens2.length);
      for (let i = 0; i < tokens1.length; i++) {
        expect(tokens1[i]).toBe(tokens2[i]);
      }
    });

    it('should produce identical output with various chunk sizes', async () => {
      if (!model) return;

      const longContent = Array(100).fill('Hello world.').join(' ');
      const messages: ChatMessage[] = [{ role: 'user', content: longContent }];

      const chunkSizes = [64, 256, 2048];
      const results: Uint32Array[] = [];

      for (const chunkSize of chunkSizes) {
        const result = await model.generate(messages, {
          maxNewTokens: 3,
          temperature: 0.0,
          prefillStepSize: chunkSize,
        });
        results.push(result.tokens.toUint32());
      }

      const baseline = results[0];
      for (let i = 1; i < results.length; i++) {
        expect(results[i].length).toBe(baseline.length);
        for (let j = 0; j < baseline.length; j++) {
          expect(results[i][j]).toBe(baseline[j]);
        }
      }
    });

    it('should handle very small chunk sizes (stress test)', async () => {
      if (!model) return;

      const messages: ChatMessage[] = [{ role: 'user', content: Array(20).fill('Testing small chunks.').join(' ') }];

      const result = await model.generate(messages, {
        maxNewTokens: 3,
        temperature: 0.0,
        prefillStepSize: 32,
      });

      expect(result.numTokens).toBeGreaterThan(0);
    });

    it('should work with batch generation and various chunk sizes', async () => {
      if (!model) return;

      const prompts: ChatMessage[][] = [
        [{ role: 'user', content: 'Count to 3' }],
        [{ role: 'user', content: Array(30).fill('Hello').join(' ') }],
        [{ role: 'user', content: 'What is the capital of France?' }],
      ];

      const result = await model.generateBatch(prompts, 2, {
        maxNewTokens: 3,
        temperature: 0.5,
        prefillStepSize: 64,
      });

      expect(result.tokens.length).toBe(6);
      expect(result.logprobs.length).toBe(6);
    });

    it('should maintain generation quality with chunked prefill', async () => {
      if (!model) return;

      const messages: ChatMessage[] = [{ role: 'user', content: 'What is 2+2? Answer:' }];

      const result = await model.generate(messages, {
        maxNewTokens: 10,
        temperature: 0.0,
        prefillStepSize: 8,
      });

      expect(result.numTokens).toBeGreaterThan(0);
      expect(result.text.length).toBeGreaterThan(0);
    });
  });

  describe('generateBatch', () => {
    it('should generate multiple completions for multiple prompts', async () => {
      if (!model) return;

      const prompts: ChatMessage[][] = [
        [{ role: 'user', content: 'Count to 3' }],
        [{ role: 'user', content: 'Say hello' }],
      ];

      const groupSize = 2;
      const config = {
        maxNewTokens: 10,
        temperature: 0.8,
        returnLogprobs: true,
      };

      const result = await model.generateBatch(prompts, groupSize, config);

      expect(result.numPrompts).toBe(2);
      expect(result.groupSize).toBe(2);
      expect(result.tokens.length).toBe(4);
      expect(result.logprobs.length).toBe(4);
      expect(result.texts.length).toBe(4);
      expect(result.finishReasons.length).toBe(2);
      expect(result.finishReasons[0].length).toBe(2);
      expect(result.finishReasons[1].length).toBe(2);
      expect(result.tokenCounts.length).toBe(2);
      expect(result.tokenCounts[0].length).toBe(2);
      expect(result.tokenCounts[1].length).toBe(2);

      result.texts.forEach((text) => {
        expect(typeof text).toBe('string');
        expect(text.length).toBeGreaterThan(0);
      });

      result.tokens.forEach((tokenArray) => {
        const shape = tokenArray.shape();
        expect(shape.length).toBe(1);
        expect(shape[0]).toBeGreaterThan(0);
        expect(shape[0]).toBeLessThanOrEqual(10);
      });

      result.tokens.forEach((tokenArray, i) => {
        const tokenShape = tokenArray.shape();
        const logprobShape = result.logprobs[i].shape();
        expect(logprobShape).toEqual(tokenShape);
      });
    });

    it('should handle single prompt with multiple completions', async () => {
      if (!model) return;

      const prompts: ChatMessage[][] = [[{ role: 'user', content: 'Write a number' }]];

      const result = await model.generateBatch(prompts, 3, {
        maxNewTokens: 5,
        temperature: 0.9,
      });

      expect(result.numPrompts).toBe(1);
      expect(result.groupSize).toBe(3);
      expect(result.tokens.length).toBe(3);
      expect(result.texts.length).toBe(3);
      expect(result.finishReasons.length).toBe(1);
      expect(result.finishReasons[0].length).toBe(3);
    });

    it('should handle multiple prompts with single completion each', async () => {
      if (!model) return;

      const prompts: ChatMessage[][] = [
        [{ role: 'user', content: 'Say A' }],
        [{ role: 'user', content: 'Say B' }],
        [{ role: 'user', content: 'Say C' }],
      ];

      const result = await model.generateBatch(prompts, 1, {
        maxNewTokens: 5,
        temperature: 0.7,
      });

      expect(result.numPrompts).toBe(3);
      expect(result.groupSize).toBe(1);
      expect(result.tokens.length).toBe(3);
      expect(result.texts.length).toBe(3);
      expect(result.finishReasons.length).toBe(3);
      expect(result.finishReasons[0].length).toBe(1);
    });

    it('should produce different completions with temperature > 0', async () => {
      if (!model) return;

      const prompts: ChatMessage[][] = [[{ role: 'user', content: 'Count to 5' }]];

      const result = await model.generateBatch(prompts, 3, {
        maxNewTokens: 10,
        temperature: 0.9,
      });

      expect(result.texts.length).toBe(3);
      result.texts.forEach((text) => {
        expect(typeof text).toBe('string');
        expect(text.length).toBeGreaterThan(0);
      });
    });

    it('should respect maxNewTokens limit', async () => {
      if (!model) return;

      const prompts: ChatMessage[][] = [[{ role: 'user', content: 'Count to 100' }]];
      const maxTokens = 5;

      const result = await model.generateBatch(prompts, 2, {
        maxNewTokens: maxTokens,
        temperature: 0.8,
      });

      result.tokenCounts[0].forEach((count) => {
        expect(count).toBeLessThanOrEqual(maxTokens);
      });
    });

    it('should handle empty prompt list', async () => {
      if (!model) return;

      const result = await model.generateBatch([], 2);

      expect(result.numPrompts).toBe(0);
      expect(result.tokens.length).toBe(0);
      expect(result.texts.length).toBe(0);
    });
  });

  /**
   * Tests for true parallel batch generation using the batched FFI kernel.
   *
   * Each test creates its own GrpoTrainingEngine and resets it when done
   * to release the model thread's training state for the next test.
   */
  describe('GRPOTrainer - Parallel Batch Generation', () => {
    it('should generate completions using parallel batch generation', async () => {
      if (!model) return;

      const engine = new GrpoTrainingEngine(model, {
        groupSize: 2,
        maxCompletionLength: 10,
        temperature: 0.8,
        useParallelBatchGeneration: true,
      });

      try {
        const prompts: ChatMessage[][] = [
          [{ role: 'user', content: 'Count to 3' }],
          [{ role: 'user', content: 'Say hello' }],
        ];

        const result = await engine.generateBatchForTraining(prompts);

        expect(result.completionTexts.length).toBe(4);
        expect(result.completionLengths.length).toBe(4);
        expect(result.finishReasons.length).toBe(4);

        result.completionTexts.forEach((text) => {
          expect(typeof text).toBe('string');
          expect(text.length).toBeGreaterThan(0);
        });

        result.completionLengths.forEach((len) => {
          expect(len).toBeGreaterThan(0);
          expect(len).toBeLessThanOrEqual(10);
        });
      } finally {
        engine.reset();
      }
    });

    it('should handle variable-length prompts with left-padding', async () => {
      if (!model) return;

      const engine = new GrpoTrainingEngine(model, {
        groupSize: 2,
        maxCompletionLength: 8,
        temperature: 0.7,
        useParallelBatchGeneration: true,
      });

      try {
        const prompts: ChatMessage[][] = [
          [{ role: 'user', content: 'Hi' }],
          [{ role: 'user', content: 'Please count from one to five' }],
          [{ role: 'user', content: 'A' }],
        ];

        const result = await engine.generateBatchForTraining(prompts);

        expect(result.completionTexts.length).toBe(6);
        expect(result.completionLengths.length).toBe(6);

        result.completionTexts.forEach((text) => {
          expect(typeof text).toBe('string');
          expect(text.length).toBeGreaterThan(0);
        });
      } finally {
        engine.reset();
      }
    });

    it('should produce similar results to sequential generation', async () => {
      if (!model) return;

      const commonConfig = {
        groupSize: 1,
        maxCompletionLength: 5,
        temperature: 0.1,
      };

      const parallelEngine = new GrpoTrainingEngine(model, {
        ...commonConfig,
        useParallelBatchGeneration: true,
      });

      try {
        const prompts: ChatMessage[][] = [[{ role: 'user', content: '1+1=' }]];
        const parallelResult = await parallelEngine.generateBatchForTraining(prompts);

        expect(parallelResult.completionTexts.length).toBe(1);
        expect(parallelResult.completionTexts[0].length).toBeGreaterThan(0);
      } finally {
        parallelEngine.reset();
      }

      const sequentialEngine = new GrpoTrainingEngine(model, {
        ...commonConfig,
        useParallelBatchGeneration: false,
      });

      try {
        const prompts: ChatMessage[][] = [[{ role: 'user', content: '1+1=' }]];
        const sequentialResult = await sequentialEngine.generateBatchForTraining(prompts);

        expect(sequentialResult.completionTexts.length).toBe(1);
        expect(sequentialResult.completionTexts[0].length).toBeGreaterThan(0);
      } finally {
        sequentialEngine.reset();
      }
    });

    it('should respect maxCompletionLength limit', async () => {
      if (!model) return;

      const maxTokens = 5;
      const engine = new GrpoTrainingEngine(model, {
        groupSize: 2,
        maxCompletionLength: maxTokens,
        temperature: 0.8,
        useParallelBatchGeneration: true,
      });

      try {
        const prompts: ChatMessage[][] = [[{ role: 'user', content: 'Count to 100' }]];
        const result = await engine.generateBatchForTraining(prompts);

        result.completionLengths.forEach((len) => {
          expect(len).toBeLessThanOrEqual(maxTokens);
        });
      } finally {
        engine.reset();
      }
    });

    it('should include valid logprobs', async () => {
      if (!model) return;

      const engine = new GrpoTrainingEngine(model, {
        groupSize: 1,
        maxCompletionLength: 5,
        temperature: 0.8,
        useParallelBatchGeneration: true,
      });

      try {
        const prompts: ChatMessage[][] = [[{ role: 'user', content: 'Say hello' }]];
        const result = await engine.generateBatchForTraining(prompts);

        expect(result.completionLogprobs.length).toBeGreaterThan(0);

        result.completionLogprobs.forEach((lp) => {
          expect(isNaN(lp)).toBe(false);
          expect(lp).toBeLessThanOrEqual(0);
          expect(lp).toBeGreaterThan(-100);
        });
      } finally {
        engine.reset();
      }
    });
  });
});
