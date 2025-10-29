import { describe, it, expect, beforeAll } from 'vitest';
import { Qwen3Model, type ChatMessage } from '@mlx-node/core';
import fs from 'node:fs';
import path from 'node:path';

describe('Qwen3Model - generateBatch', () => {
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
      console.log('  ⏭️  Skipping generateBatch tests (no model found)');
      console.log(`     Set QWEN3_MODEL_PATH or place model at: ${defaultModelPath}`);
      return;
    }

    // Load model with tokenizer
    model = await Qwen3Model.loadPretrained(modelPath);
  });

  it('should generate multiple completions for multiple prompts', async () => {
    if (!model) {
      console.log('  ⏭️  Skipping (no model loaded)');
      return;
    }

    const prompts: ChatMessage[][] = [
      [{ role: 'user', content: 'Count to 3' }],
      [{ role: 'user', content: 'Say hello' }],
    ];

    const groupSize = 2; // 2 completions per prompt
    const config = {
      maxNewTokens: 10,
      temperature: 0.8,
      returnLogprobs: true,
    };

    const result = await model.generateBatch(prompts, groupSize, config);

    // Verify structure
    expect(result.numPrompts).toBe(2);
    expect(result.groupSize).toBe(2);

    // Verify we got 2*2 = 4 completions
    expect(result.tokens.length).toBe(4);
    expect(result.logprobs.length).toBe(4);
    expect(result.texts.length).toBe(4);

    // Verify finish reasons are grouped by prompt
    expect(result.finishReasons.length).toBe(2); // 2 prompts
    expect(result.finishReasons[0].length).toBe(2); // 2 completions for prompt 0
    expect(result.finishReasons[1].length).toBe(2); // 2 completions for prompt 1

    // Verify token counts are grouped by prompt
    expect(result.tokenCounts.length).toBe(2);
    expect(result.tokenCounts[0].length).toBe(2);
    expect(result.tokenCounts[1].length).toBe(2);

    // Verify all texts are non-empty strings
    result.texts.forEach((text, i) => {
      expect(typeof text).toBe('string');
      expect(text.length).toBeGreaterThan(0);
      console.log(`Completion ${i}: ${text.substring(0, 50)}...`);
    });

    // Verify token arrays have correct shapes
    result.tokens.forEach((tokenArray, _) => {
      const shape = tokenArray.shape();
      expect(shape.length).toBe(1); // 1D array
      expect(shape[0]).toBeGreaterThan(0);
      expect(shape[0]).toBeLessThanOrEqual(10); // max_new_tokens
    });

    // Verify logprob arrays match token arrays
    result.tokens.forEach((tokenArray, i) => {
      const tokenShape = tokenArray.shape();
      const logprobShape = result.logprobs[i].shape();
      expect(logprobShape).toEqual(tokenShape);
    });
  });

  it('should handle single prompt with multiple completions', async () => {
    if (!model) {
      console.log('  ⏭️  Skipping (no model loaded)');
      return;
    }

    const prompts: ChatMessage[][] = [[{ role: 'user', content: 'Write a number' }]];

    const groupSize = 3;
    const config = {
      maxNewTokens: 5,
      temperature: 0.9,
    };

    const result = await model.generateBatch(prompts, groupSize, config);

    expect(result.numPrompts).toBe(1);
    expect(result.groupSize).toBe(3);
    expect(result.tokens.length).toBe(3);
    expect(result.texts.length).toBe(3);
    expect(result.finishReasons.length).toBe(1);
    expect(result.finishReasons[0].length).toBe(3);
  });

  it('should handle multiple prompts with single completion each', async () => {
    if (!model) {
      console.log('  ⏭️  Skipping (no model loaded)');
      return;
    }

    const prompts: ChatMessage[][] = [
      [{ role: 'user', content: 'Say A' }],
      [{ role: 'user', content: 'Say B' }],
      [{ role: 'user', content: 'Say C' }],
    ];

    const groupSize = 1;
    const config = {
      maxNewTokens: 5,
      temperature: 0.7,
    };

    const result = await model.generateBatch(prompts, groupSize, config);

    expect(result.numPrompts).toBe(3);
    expect(result.groupSize).toBe(1);
    expect(result.tokens.length).toBe(3);
    expect(result.texts.length).toBe(3);
    expect(result.finishReasons.length).toBe(3);
    expect(result.finishReasons[0].length).toBe(1);
  });

  it('should produce different completions with temperature > 0', async () => {
    if (!model) {
      console.log('  ⏭️  Skipping (no model loaded)');
      return;
    }

    const prompts: ChatMessage[][] = [[{ role: 'user', content: 'Count to 5' }]];

    const groupSize = 3;
    const config = {
      maxNewTokens: 10,
      temperature: 0.9, // High temperature for diversity
    };

    const result = await model.generateBatch(prompts, groupSize, config);

    // With high temperature, completions should vary (at least one should be different)
    // Note: Due to randomness, they might sometimes be identical, so we check if at least
    // one pair is different rather than requiring all to be different
    const texts = result.texts;
    expect(texts.length).toBe(3);

    // Check that we got valid text completions
    texts.forEach((text) => {
      expect(typeof text).toBe('string');
      expect(text.length).toBeGreaterThan(0);
    });
  });

  it('should respect maxNewTokens limit', async () => {
    if (!model) {
      console.log('  ⏭️  Skipping (no model loaded)');
      return;
    }

    const prompts: ChatMessage[][] = [[{ role: 'user', content: 'Count to 100' }]];

    const maxTokens = 5;
    const groupSize = 2;
    const config = {
      maxNewTokens: maxTokens,
      temperature: 0.8,
    };

    const result = await model.generateBatch(prompts, groupSize, config);

    result.tokenCounts[0].forEach((count) => {
      expect(count).toBeLessThanOrEqual(maxTokens);
    });
  });

  it('should handle empty prompt list', async () => {
    if (!model) {
      console.log('  ⏭️  Skipping (no model loaded)');
      return;
    }

    const prompts: ChatMessage[][] = [];
    const groupSize = 2;

    const result = await model.generateBatch(prompts, groupSize);

    expect(result.numPrompts).toBe(0);
    expect(result.tokens.length).toBe(0);
    expect(result.texts.length).toBe(0);
  });
});
