import { describe, test, expect, beforeAll } from 'vitest';
import { getBridge, runTest } from './test-bridge';

beforeAll(() => getBridge(), 60_000);

describe('Attention & SDPA', () => {
  test('SDPA causal where+softmax [1,8,4,4]', async () => {
    const r = await runTest('SDPA causal where+softmax [1,8,4,4]');
    expect(r.passed, r.error).toBe(true);
  });

  test('SDPA causal mode correctness', async () => {
    const r = await runTest('SDPA causal mode correctness');
    expect(r.passed, r.error).toBe(true);
  });

  test('SDPA causal GQA (Hq=8 Hkv=2)', async () => {
    const r = await runTest('SDPA causal GQA (Hq=8 Hkv=2)');
    expect(r.passed, r.error).toBe(true);
  });

  test('RoPE bf16 model dims', async () => {
    const r = await runTest('RoPE bf16 model dims');
    expect(r.passed, r.error).toBe(true);
  });

  test('QK norm + RoPE chain bf16', async () => {
    const r = await runTest('QK norm + RoPE chain bf16');
    expect(r.passed, r.error).toBe(true);
  });

  test('SDPA additive mask bf16', async () => {
    const r = await runTest('SDPA additive mask bf16');
    expect(r.passed, r.error).toBe(true);
  });

  test('SDPA decode GQA bf16', async () => {
    const r = await runTest('SDPA decode GQA bf16');
    expect(r.passed, r.error).toBe(true);
  });

  test('SDPA tile Tq=2 D=64 (f32)', async () => {
    const r = await runTest('SDPA tile Tq=2 D=64 (f32)');
    expect(r.passed, r.error).toBe(true);
  });

  test('SDPA tile Tq=8 D=128 causal GQA (f32)', async () => {
    const r = await runTest('SDPA tile Tq=8 D=128 causal GQA (f32)');
    expect(r.passed, r.error).toBe(true);
  });

  test('SDPA tile Tq=32 D=128 additive mask (f32)', async () => {
    const r = await runTest('SDPA tile Tq=32 D=128 additive mask (f32)');
    expect(r.passed, r.error).toBe(true);
  });

  test('SDPA tile Tq=33 D=128 causal partial-last-tile (f32)', async () => {
    const r = await runTest('SDPA tile Tq=33 D=128 causal partial-last-tile (f32)');
    expect(r.passed, r.error).toBe(true);
  });

  test('SDPA tile Tq=128 D=128 L=4096 causal (f32)', async () => {
    const r = await runTest('SDPA tile Tq=128 D=128 L=4096 causal (f32)');
    expect(r.passed, r.error).toBe(true);
  });

  test('SDPA tile D=256 Tq=8 causal GQA (f32, Qwen3.5-0.8B shape)', async () => {
    const r = await runTest('SDPA tile D=256 Tq=8 causal GQA (f32, Qwen3.5-0.8B shape)');
    expect(r.passed, r.error).toBe(true);
  });

  test('SDPA tile D=256 simple (no causal, no GQA)', async () => {
    const r = await runTest('SDPA tile D=256 simple (no causal, no GQA)');
    expect(r.passed, r.error).toBe(true);
  });

  test('SDPA tile D=256 causal (no GQA)', async () => {
    const r = await runTest('SDPA tile D=256 causal (no GQA)');
    expect(r.passed, r.error).toBe(true);
  });

  test('SDPA tile D=256 GQA (no causal)', async () => {
    const r = await runTest('SDPA tile D=256 GQA (no causal)');
    expect(r.passed, r.error).toBe(true);
  });

  test('SDPA tile D=256 minimal causal+GQA (H=4 Hkv=2)', async () => {
    const r = await runTest('SDPA tile D=256 minimal causal+GQA (H=4 Hkv=2)');
    expect(r.passed, r.error).toBe(true);
  });

  test('SDPA vector D=256 Tq=1 GQA (f32, Qwen3.5-0.8B decode shape)', async () => {
    const r = await runTest('SDPA vector D=256 Tq=1 GQA (f32, Qwen3.5-0.8B decode shape)');
    expect(r.passed, r.error).toBe(true);
  });

  test('SDPA vector D=256 simple (no GQA, Tq=1)', async () => {
    const r = await runTest('SDPA vector D=256 simple (no GQA, Tq=1)');
    expect(r.passed, r.error).toBe(true);
  });

  test('full attention layer bf16 (Q/K/V proj+QK norm+RoPE+GQA+SDPA+gate+Oproj)', async () => {
    const r = await runTest('full attention layer bf16 (Q/K/V proj+QK norm+RoPE+GQA+SDPA+gate+Oproj)');
    expect(r.passed, r.error).toBe(true);
  });

  test('attention decoder layer bf16', async () => {
    const r = await runTest('attention decoder layer bf16');
    expect(r.passed, r.error).toBe(true);
  });

  test('attention gate sigmoid gating bf16', async () => {
    const r = await runTest('attention gate sigmoid gating bf16');
    expect(r.passed, r.error).toBe(true);
  });

  test('SDPA fallback GQA correctness (Tq=8 D=128 Hq=8 Hkv=2)', async () => {
    const r = await runTest('SDPA fallback GQA correctness (Tq=8 D=128 Hq=8 Hkv=2)');
    expect(r.passed, r.error).toBe(true);
  });
});
