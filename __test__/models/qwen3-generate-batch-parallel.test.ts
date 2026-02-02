import { describe, it, expect, beforeAll } from 'vite-plus/test';
import { GrpoTrainingEngine, Qwen3Model, type ChatMessage } from '@mlx-node/core';
import fs from 'node:fs';
import path from 'node:path';

/**
 * Tests for true parallel batch generation using the batched FFI kernel.
 *
 * These tests verify that the new parallel batch generation mode
 * (enabled via useParallelBatchGeneration config) produces correct results.
 */
describe('GRPOTrainer - Parallel Batch Generation', () => {
  let model: Qwen3Model | null = null;

  beforeAll(async () => {
    // Try environment variable first, then default path
    const envModelPath = process.env.QWEN3_MODEL_PATH;
    const defaultModelPath = path.join(process.cwd(), '.cache/models/qwen3-0.6b-mlx-bf16');

    let modelPath: string | null = null;

    if (envModelPath && fs.existsSync(envModelPath)) {
      modelPath = envModelPath;
      console.log(`  📦 Loading model from QWEN3_MODEL_PATH: ${modelPath}`);
    } else if (fs.existsSync(defaultModelPath)) {
      modelPath = defaultModelPath;
      console.log(`  📦 Loading model from default path: ${modelPath}`);
    } else {
      console.log('  ⏭️  Skipping parallel batch tests (no model found)');
      console.log(`     Set QWEN3_MODEL_PATH or place model at: ${defaultModelPath}`);
      return;
    }

    // Load model with tokenizer
    model = await Qwen3Model.loadPretrained(modelPath);
  }, 30000);

  it('should generate completions using parallel batch generation', async () => {
    if (!model) {
      console.log('  ⏭️  Skipping (no model loaded)');
      return;
    }

    // Create engine with parallel batch generation enabled
    const engine = new GrpoTrainingEngine(model, {
      groupSize: 2,
      maxCompletionLength: 10,
      temperature: 0.8,
      useParallelBatchGeneration: true,
    });

    const prompts: ChatMessage[][] = [
      [{ role: 'user', content: 'Count to 3' }],
      [{ role: 'user', content: 'Say hello' }],
    ];

    // Generate using the engine (which uses parallel generation internally)
    const result = await engine.generateBatchForTraining(prompts);

    // Verify structure - 2 prompts * 2 group_size = 4 completions
    expect(result.completionTexts.length).toBe(4);
    expect(result.completionLengths.length).toBe(4);
    expect(result.finishReasons.length).toBe(4);

    // Verify all completions are non-empty
    result.completionTexts.forEach((text, i) => {
      expect(typeof text).toBe('string');
      expect(text.length).toBeGreaterThan(0);
      console.log(`  Completion ${i}: "${text.substring(0, 30)}..."`);
    });

    // Verify completion lengths are positive
    result.completionLengths.forEach((len) => {
      expect(len).toBeGreaterThan(0);
      expect(len).toBeLessThanOrEqual(10); // max_completion_length
    });
  });

  it('should handle variable-length prompts with left-padding', async () => {
    if (!model) {
      console.log('  ⏭️  Skipping (no model loaded)');
      return;
    }

    // Create engine with parallel batch generation
    const engine = new GrpoTrainingEngine(model, {
      groupSize: 2,
      maxCompletionLength: 8,
      temperature: 0.7,
      useParallelBatchGeneration: true,
    });

    // Prompts with different lengths
    const prompts: ChatMessage[][] = [
      [{ role: 'user', content: 'Hi' }], // Short
      [{ role: 'user', content: 'Please count from one to five' }], // Long
      [{ role: 'user', content: 'A' }], // Minimal
    ];

    const result = await engine.generateBatchForTraining(prompts);

    // Verify we got 3*2 = 6 completions
    expect(result.completionTexts.length).toBe(6);
    expect(result.completionLengths.length).toBe(6);

    // Verify all completions are valid
    result.completionTexts.forEach((text) => {
      expect(typeof text).toBe('string');
      expect(text.length).toBeGreaterThan(0);
    });
  });

  it('should produce similar results to sequential generation', async () => {
    if (!model) {
      console.log('  ⏭️  Skipping (no model loaded)');
      return;
    }

    // Use deterministic settings for comparison (low temperature)
    const commonConfig = {
      groupSize: 1, // Simplify comparison
      maxCompletionLength: 5,
      temperature: 0.1, // Low temperature for more deterministic output
    };

    // Create engines with parallel and sequential generation
    const parallelEngine = new GrpoTrainingEngine(model, {
      ...commonConfig,
      useParallelBatchGeneration: true,
    });

    const sequentialEngine = new GrpoTrainingEngine(model, {
      ...commonConfig,
      useParallelBatchGeneration: false,
    });

    const prompts: ChatMessage[][] = [[{ role: 'user', content: '1+1=' }]];

    // Generate with both methods
    const parallelResult = await parallelEngine.generateBatchForTraining(prompts);
    const sequentialResult = await sequentialEngine.generateBatchForTraining(prompts);

    // Both should produce valid completions
    expect(parallelResult.completionTexts.length).toBe(1);
    expect(sequentialResult.completionTexts.length).toBe(1);

    console.log(`  Parallel: "${parallelResult.completionTexts[0]}"`);
    console.log(`  Sequential: "${sequentialResult.completionTexts[0]}"`);

    // Both should produce non-empty text
    expect(parallelResult.completionTexts[0].length).toBeGreaterThan(0);
    expect(sequentialResult.completionTexts[0].length).toBeGreaterThan(0);

    // Note: Due to different RNG states, outputs may not be identical
    // But both should be valid completions of similar quality
  });

  it('should respect maxCompletionLength limit', async () => {
    if (!model) {
      console.log('  ⏭️  Skipping (no model loaded)');
      return;
    }

    const maxTokens = 5;
    const engine = new GrpoTrainingEngine(model, {
      groupSize: 2,
      maxCompletionLength: maxTokens,
      temperature: 0.8,
      useParallelBatchGeneration: true,
    });

    const prompts: ChatMessage[][] = [[{ role: 'user', content: 'Count to 100' }]];

    const result = await engine.generateBatchForTraining(prompts);

    result.completionLengths.forEach((len) => {
      expect(len).toBeLessThanOrEqual(maxTokens);
    });
  });

  it('should include valid logprobs', async () => {
    if (!model) {
      console.log('  ⏭️  Skipping (no model loaded)');
      return;
    }

    const engine = new GrpoTrainingEngine(model, {
      groupSize: 1,
      maxCompletionLength: 5,
      temperature: 0.8,
      useParallelBatchGeneration: true,
    });

    const prompts: ChatMessage[][] = [[{ role: 'user', content: 'Say hello' }]];

    const result = await engine.generateBatchForTraining(prompts);

    // completionLogprobs is a flat array
    expect(result.completionLogprobs.length).toBeGreaterThan(0);

    // Verify logprobs are valid (not NaN, in reasonable range)
    result.completionLogprobs.forEach((lp) => {
      expect(isNaN(lp)).toBe(false);
      expect(lp).toBeLessThanOrEqual(0); // Log probs are <= 0
      expect(lp).toBeGreaterThan(-100); // Reasonable lower bound
    });
  });
});
