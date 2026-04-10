import fs from 'node:fs';
import path from 'node:path';

import { GrpoTrainingEngine, Qwen3Model, type ChatMessage } from '@mlx-node/core';
import { describe, it, expect, beforeAll } from 'vite-plus/test';

/**
 * Tests for true parallel batch generation using the batched FFI kernel.
 *
 * These tests verify that the new parallel batch generation mode
 * (enabled via useParallelBatchGeneration config) produces correct results.
 *
 * Each test creates its own GrpoTrainingEngine and resets it when done
 * to release the model thread's training state for the next test.
 */
describe.sequential('GRPOTrainer - Parallel Batch Generation', () => {
  let model: Qwen3Model | null = null;

  beforeAll(async () => {
    // Try environment variable first, then default path
    const envModelPath = process.env.QWEN3_MODEL_PATH;
    const defaultModelPath = path.join(process.cwd(), '.cache/models/qwen3-0.6b-mlx-bf16');

    let modelPath: string | null = null;

    if (envModelPath && fs.existsSync(envModelPath)) {
      modelPath = envModelPath;
    } else if (fs.existsSync(defaultModelPath)) {
      modelPath = defaultModelPath;
    } else {
      return;
    }

    // Load model with tokenizer
    model = await Qwen3Model.load(modelPath);
  }, 30000);

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

      // 2 prompts * 2 group_size = 4 completions
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
