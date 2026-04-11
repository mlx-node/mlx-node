import { describe, test, expect, beforeAll } from 'vitest';
import { getBridge, runTest } from './test-bridge';

beforeAll(() => getBridge(), 60_000);

describe('Packed BF16 Parity', () => {
  test('packed bf16 gemv K=896 N=2048 transposed', async () => {
    const r = await runTest('packed bf16 gemv K=896 N=2048 transposed');
    expect(r.passed, r.error).toBe(true);
  });

  test('packed bf16 gemv K=128 N=1 transposed (below-threshold fallback)', async () => {
    const r = await runTest('packed bf16 gemv K=128 N=1 transposed (below-threshold fallback)');
    expect(r.passed, r.error).toBe(true);
  });

  test('packed bf16 gemv K=896 N=7 transposed odd-N', async () => {
    const r = await runTest('packed bf16 gemv K=896 N=7 transposed odd-N');
    expect(r.passed, r.error).toBe(true);
  });

  test('packed bf16 byte-concat parity (axis=0, bf16)', async () => {
    const r = await runTest('packed bf16 byte-concat parity (axis=0, bf16)');
    expect(r.passed, r.error).toBe(true);
  });

  test('packed bf16 gemv K=1024 N=8192 transposed (in_proj_qkvz scale)', async () => {
    const r = await runTest('packed bf16 gemv K=1024 N=8192 transposed (in_proj_qkvz scale)');
    expect(r.passed, r.error).toBe(true);
  });

  test('packed bf16 rmsnorm D=4096', async () => {
    const r = await runTest('packed bf16 rmsnorm D=4096');
    expect(r.passed, r.error).toBe(true);
  });

  test('packed bf16 rmsnorm D=1024 production norm threshold (256)', async () => {
    const r = await runTest('packed bf16 rmsnorm D=1024 production norm threshold (256)');
    expect(r.passed, r.error).toBe(true);
  });

  test('packed bf16 rmsnorm D=4098 even-tail (buffer pad check)', async () => {
    const r = await runTest('packed bf16 rmsnorm D=4098 even-tail (buffer pad check)');
    expect(r.passed, r.error).toBe(true);
  });

  test('packed bf16 gemv K=896 N=2048 transposed offset-sliced', async () => {
    const r = await runTest('packed bf16 gemv K=896 N=2048 transposed offset-sliced');
    expect(r.passed, r.error).toBe(true);
  });
});
