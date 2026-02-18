import { describe, it, expect } from 'vite-plus/test';
import { Qwen35Model, MxArray } from '@mlx-node/core';
import { getQwen35Config } from '@mlx-node/lm';
import { shape } from '../test-utils';

describe.sequential('Qwen3.5 Forward Pass', () => {
  it('should run forward pass on small config', () => {
    const config = getQwen35Config('qwen3.5-0.6b');
    const model = new Qwen35Model(config);

    // Create dummy input: [batch=1, seq_len=4]
    const input = MxArray.fromInt32(new Int32Array([1, 2, 3, 4]), shape(1, 4));
    const logits = model.forward(input);

    // Should be [1, 4, vocab_size]
    const s = logits.shape();
    expect(s.length).toBe(3);
    expect(Number(s[0])).toBe(1);
    expect(Number(s[1])).toBe(4);
    expect(Number(s[2])).toBe(config.vocabSize);
  });

  it('should run forward with cache (incremental)', () => {
    const config = getQwen35Config('qwen3.5-0.6b');
    const model = new Qwen35Model(config);

    // Initialize caches
    model.initCaches();

    // Prefill
    const prompt = MxArray.fromInt32(new Int32Array([1, 2, 3]), shape(1, 3));
    const logits1 = model.forwardWithCache(prompt);
    const s1 = logits1.shape();
    expect(s1.length).toBe(3);
    expect(Number(s1[0])).toBe(1);
    expect(Number(s1[1])).toBe(3);

    // Decode step (single token)
    const next = MxArray.fromInt32(new Int32Array([4]), shape(1, 1));
    const logits2 = model.forwardWithCache(next);
    const s2 = logits2.shape();
    expect(Number(s2[1])).toBe(1);

    // Cleanup
    model.resetCaches();
  });

  it('should run cache init and reset', () => {
    const config = getQwen35Config('qwen3.5-0.6b');
    const model = new Qwen35Model(config);

    model.initCaches();
    model.resetCaches();

    // Should be able to run after reset
    const input = MxArray.fromInt32(new Int32Array([1]), shape(1, 1));
    const logits = model.forward(input);
    expect(logits).toBeDefined();
  });
});
