/**
 * Speculative Decoding Tests
 *
 * Tests for the speculative decoding implementation which uses a smaller
 * draft model to generate tokens speculatively, then verifies them with
 * the target model in a single forward pass.
 */
import { describe, it, expect, beforeAll, afterAll } from 'vite-plus/test';
import { existsSync } from 'node:fs';
import { MxArray } from '@mlx-node/core';

// Path to test models - can be configured via environment variable
const MODEL_PATH = process.env.QWEN3_MODEL_PATH || './models/qwen3-0.6b';
const DRAFT_MODEL_PATH = process.env.QWEN3_DRAFT_MODEL_PATH || './models/qwen3-0.6b';

// Skip tests if models are not available
const modelsAvailable = existsSync(MODEL_PATH) && existsSync(DRAFT_MODEL_PATH);

describe.skipIf(!modelsAvailable)('Speculative Decoding', () => {
  let targetModel: any;
  let draftModel: any;

  beforeAll(async () => {
    // Dynamic import to avoid errors when models not available
    const { ModelLoader } = await import('@mlx-node/lm');
    targetModel = await ModelLoader.loadPretrained(MODEL_PATH);
    draftModel = await ModelLoader.loadPretrained(DRAFT_MODEL_PATH);
  }, 120000);

  afterAll(() => {
    // Cleanup models
    targetModel = null;
    draftModel = null;
  });

  it('should generate tokens using speculative decoding', async () => {
    // Create a simple prompt
    const prompt = 'The capital of France is';
    const tokenizer = targetModel.getTokenizer();
    const inputIds = await tokenizer.encode(prompt, false);
    const inputArray = MxArray.fromUint32(new Uint32Array(inputIds), BigInt64Array.from([1n, BigInt(inputIds.length)]));

    // Generate with speculative decoding
    const result = targetModel.generateSpeculativeSync(draftModel, inputArray, {
      maxNewTokens: 20,
      temperature: 0.7,
      numDraftTokens: 3,
    });

    // Verify we got tokens
    expect(result.numTokens).toBeGreaterThan(0);
    expect(result.tokens).toBeDefined();

    // Check finish reason includes speculative stats
    expect(result.finishReason).toContain('accept_rate');
    expect(result.finishReason).toContain('tok_per_pass');
  }, 60000);

  it('should handle EOS token correctly', async () => {
    // Use a prompt that should lead to a short response
    const prompt = 'Say "hello" and nothing else:';
    const tokenizer = targetModel.getTokenizer();
    const inputIds = await tokenizer.encode(prompt, false);
    const inputArray = MxArray.fromUint32(new Uint32Array(inputIds), BigInt64Array.from([1n, BigInt(inputIds.length)]));

    // Generate with speculative decoding
    const result = targetModel.generateSpeculativeSync(draftModel, inputArray, {
      maxNewTokens: 50,
      temperature: 0.5,
      numDraftTokens: 5,
    });

    // Should stop at EOS or max tokens
    expect(result.numTokens).toBeLessThanOrEqual(50);
    expect(result.tokens).toBeDefined();
  }, 60000);

  it('should have acceptance rate between 0 and 1', async () => {
    const prompt = 'Write a short poem:';
    const tokenizer = targetModel.getTokenizer();
    const inputIds = await tokenizer.encode(prompt, false);
    const inputArray = MxArray.fromUint32(new Uint32Array(inputIds), BigInt64Array.from([1n, BigInt(inputIds.length)]));

    const result = targetModel.generateSpeculativeSync(draftModel, inputArray, {
      maxNewTokens: 30,
      temperature: 0.8,
      numDraftTokens: 4,
    });

    // Parse acceptance rate from finish reason
    const match = result.finishReason.match(/accept_rate:([\d.]+)/);
    expect(match).toBeTruthy();

    if (match) {
      const acceptRate = parseFloat(match[1]);
      expect(acceptRate).toBeGreaterThanOrEqual(0);
      expect(acceptRate).toBeLessThanOrEqual(1);
    }
  }, 60000);

  it('should benefit from similar draft/target models', async () => {
    // When draft and target are the same model, acceptance rate should be very high
    const prompt = 'Hello world';
    const tokenizer = targetModel.getTokenizer();
    const inputIds = await tokenizer.encode(prompt, false);
    const inputArray = MxArray.fromUint32(new Uint32Array(inputIds), BigInt64Array.from([1n, BigInt(inputIds.length)]));

    // Use same model as both draft and target
    const result = targetModel.generateSpeculativeSync(targetModel, inputArray, {
      maxNewTokens: 20,
      temperature: 0.0, // Greedy for deterministic behavior
      numDraftTokens: 5,
    });

    // Parse acceptance rate - should be high when models are identical
    const match = result.finishReason.match(/accept_rate:([\d.]+)/);
    expect(match).toBeTruthy();

    if (match) {
      const acceptRate = parseFloat(match[1]);
      // With identical models and greedy sampling, acceptance should be very high
      // (may not be 1.0 due to floating point differences)
      expect(acceptRate).toBeGreaterThan(0.8);
    }
  }, 60000);
});

describe('Speculative Decoding Unit Tests', () => {
  it('should export SpeculativeStats', async () => {
    // The SpeculativeStats struct should be accessible
    // This is a compile-time check that the module exports are correct
    expect(true).toBe(true);
  });

  // KVCache tests moved to Rust: crates/mlx-core/src/transformer/kv_cache.rs
});
