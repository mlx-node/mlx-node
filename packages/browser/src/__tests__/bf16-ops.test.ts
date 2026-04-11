import { describe, test, expect, beforeAll } from 'vitest';
import { getBridge, runTest } from './test-bridge';

beforeAll(() => getBridge(), 60_000);

describe('BF16 Operations', () => {
  test('bfloat16 add', async () => {
    const r = await runTest('bfloat16 add');
    expect(r.passed, r.error).toBe(true);
  });

  test('bfloat16 matmul', async () => {
    const r = await runTest('bfloat16 matmul');
    expect(r.passed, r.error).toBe(true);
  });

  test('bf16 double astype roundtrip', async () => {
    const r = await runTest('bf16 double astype roundtrip');
    expect(r.passed, r.error).toBe(true);
  });

  test('bf16 add then read as f32', async () => {
    const r = await runTest('bf16 add then read as f32');
    expect(r.passed, r.error).toBe(true);
  });

  test('bf16 add N=64 via f32 readback', async () => {
    const r = await runTest('bf16 add N=64 via f32 readback');
    expect(r.passed, r.error).toBe(true);
  });

  test('bf16 direct toFloat32 N=64', async () => {
    const r = await runTest('bf16 direct toFloat32 N=64');
    expect(r.passed, r.error).toBe(true);
  });

  test('bf16 roundtrip sizes', async () => {
    const r = await runTest('bf16 roundtrip sizes');
    expect(r.passed, r.error).toBe(true);
  });

  test('bf16 astype large readback', async () => {
    const r = await runTest('bf16 astype large readback');
    expect(r.passed, r.error).toBe(true);
  });

  test('bf16 matmul large (model-like)', async () => {
    const r = await runTest('bf16 matmul large (model-like)');
    expect(r.passed, r.error).toBe(true);
  });

  test('bf16 sum axis', async () => {
    const r = await runTest('bf16 sum axis');
    expect(r.passed, r.error).toBe(true);
  });

  test('bf16 sum then mul scalar (mean decomposed)', async () => {
    const r = await runTest('bf16 sum then mul scalar (mean decomposed)');
    expect(r.passed, r.error).toBe(true);
  });

  test('bf16 mean axis', async () => {
    const r = await runTest('bf16 mean axis');
    expect(r.passed, r.error).toBe(true);
  });

  test('bf16 rmsnorm chain', async () => {
    const r = await runTest('bf16 rmsnorm chain');
    expect(r.passed, r.error).toBe(true);
  });

  test('bf16 softmax', async () => {
    const r = await runTest('bf16 softmax');
    expect(r.passed, r.error).toBe(true);
  });

  test('bf16 large reduce (vocab argmax)', async () => {
    const r = await runTest('bf16 large reduce (vocab argmax)');
    expect(r.passed, r.error).toBe(true);
  });

  test('bf16 chained linear layers', async () => {
    const r = await runTest('bf16 chained linear layers');
    expect(r.passed, r.error).toBe(true);
  });

  test('squeeze basic', async () => {
    const r = await runTest('squeeze basic');
    expect(r.passed, r.error).toBe(true);
  });

  test('squeeze on bf16', async () => {
    const r = await runTest('squeeze on bf16');
    expect(r.passed, r.error).toBe(true);
  });

  test('squeeze large bf16', async () => {
    const r = await runTest('squeeze large bf16');
    expect(r.passed, r.error).toBe(true);
  });

  test('bf16 reshape', async () => {
    const r = await runTest('bf16 reshape');
    expect(r.passed, r.error).toBe(true);
  });

  test('bf16 matmul then squeeze', async () => {
    const r = await runTest('bf16 matmul then squeeze');
    expect(r.passed, r.error).toBe(true);
  });

  test('bf16 large copy (vocab-sized)', async () => {
    const r = await runTest('bf16 large copy (vocab-sized)');
    expect(r.passed, r.error).toBe(true);
  });

  test('bf16 multi-eval stress (model pattern)', async () => {
    const r = await runTest('bf16 multi-eval stress (model pattern)');
    expect(r.passed, r.error).toBe(true);
  });

  test('bf16 repeated eval cycles (24 layers x 3 passes)', async () => {
    const r = await runTest('bf16 repeated eval cycles (24 layers x 3 passes)');
    expect(r.passed, r.error).toBe(true);
  });

  test('bf16 argmax vocab-sized logits', async () => {
    const r = await runTest('bf16 argmax vocab-sized logits');
    expect(r.passed, r.error).toBe(true);
  });

  test('bf16 embed -> matmul(lm_head) -> argmax roundtrip', async () => {
    const r = await runTest('bf16 embed -> matmul(lm_head) -> argmax roundtrip');
    expect(r.passed, r.error).toBe(true);
  });

  test('bf16 RMSNorm -> matmul -> argmax chain', async () => {
    const r = await runTest('bf16 RMSNorm -> matmul -> argmax chain');
    expect(r.passed, r.error).toBe(true);
  });

  test('bf16 residual add 24-layer accumulation', async () => {
    const r = await runTest('bf16 residual add 24-layer accumulation');
    expect(r.passed, r.error).toBe(true);
  });
});
