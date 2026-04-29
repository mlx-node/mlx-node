import { describe, test, expect, beforeAll } from 'vitest';

import { getBridge, runTest } from './test-bridge';

beforeAll(() => getBridge(), 60_000);

describe('Basic Operations', () => {
  test('create float32', async () => {
    const r = await runTest('create float32');
    expect(r.passed, r.error).toBe(true);
  });

  test('zeros+eval', async () => {
    const r = await runTest('zeros+eval');
    expect(r.passed, r.error).toBe(true);
  });

  test('add', async () => {
    const r = await runTest('add');
    expect(r.passed, r.error).toBe(true);
  });

  test('mul', async () => {
    const r = await runTest('mul');
    expect(r.passed, r.error).toBe(true);
  });

  test('abs', async () => {
    const r = await runTest('abs');
    expect(r.passed, r.error).toBe(true);
  });

  test('exp', async () => {
    const r = await runTest('exp');
    expect(r.passed, r.error).toBe(true);
  });

  test('sqrt', async () => {
    const r = await runTest('sqrt');
    expect(r.passed, r.error).toBe(true);
  });

  test('reshape', async () => {
    const r = await runTest('reshape');
    expect(r.passed, r.error).toBe(true);
  });

  test('sum', async () => {
    const r = await runTest('sum');
    expect(r.passed, r.error).toBe(true);
  });

  test('max', async () => {
    const r = await runTest('max');
    expect(r.passed, r.error).toBe(true);
  });

  test('matmul 2x3*3x2', async () => {
    const r = await runTest('matmul 2x3*3x2');
    expect(r.passed, r.error).toBe(true);
  });

  test('addScalar', async () => {
    const r = await runTest('addScalar');
    expect(r.passed, r.error).toBe(true);
  });

  test('negative', async () => {
    const r = await runTest('negative');
    expect(r.passed, r.error).toBe(true);
  });

  test('sum axis', async () => {
    const r = await runTest('sum axis');
    expect(r.passed, r.error).toBe(true);
  });

  test('reduce offset view', async () => {
    const r = await runTest('reduce offset view');
    expect(r.passed, r.error).toBe(true);
  });

  test('softmax offset view', async () => {
    const r = await runTest('softmax offset view');
    expect(r.passed, r.error).toBe(true);
  });

  test('logsumexp offset view', async () => {
    const r = await runTest('logsumexp offset view');
    expect(r.passed, r.error).toBe(true);
  });

  test('broadcast add', async () => {
    const r = await runTest('broadcast add');
    expect(r.passed, r.error).toBe(true);
  });

  test('logSoftmax', async () => {
    const r = await runTest('logSoftmax');
    expect(r.passed, r.error).toBe(true);
  });

  test('concatenate', async () => {
    const r = await runTest('concatenate');
    expect(r.passed, r.error).toBe(true);
  });

  test('arange', async () => {
    const r = await runTest('arange');
    expect(r.passed, r.error).toBe(true);
  });

  test('chained (a+b)*c', async () => {
    const r = await runTest('chained (a+b)*c');
    expect(r.passed, r.error).toBe(true);
  });

  test('where (float cond)', async () => {
    const r = await runTest('where (float cond)');
    expect(r.passed, r.error).toBe(true);
  });

  test('sub', async () => {
    const r = await runTest('sub');
    expect(r.passed, r.error).toBe(true);
  });

  test('div', async () => {
    const r = await runTest('div');
    expect(r.passed, r.error).toBe(true);
  });

  test('log', async () => {
    const r = await runTest('log');
    expect(r.passed, r.error).toBe(true);
  });

  test('sin', async () => {
    const r = await runTest('sin');
    expect(r.passed, r.error).toBe(true);
  });

  test('cos', async () => {
    const r = await runTest('cos');
    expect(r.passed, r.error).toBe(true);
  });

  test('tanh', async () => {
    const r = await runTest('tanh');
    expect(r.passed, r.error).toBe(true);
  });

  test('power', async () => {
    const r = await runTest('power');
    expect(r.passed, r.error).toBe(true);
  });

  test('transpose 2D', async () => {
    const r = await runTest('transpose 2D');
    expect(r.passed, r.error).toBe(true);
  });

  test('slice', async () => {
    const r = await runTest('slice');
    expect(r.passed, r.error).toBe(true);
  });

  test('argmax', async () => {
    const r = await runTest('argmax');
    expect(r.passed, r.error).toBe(true);
  });

  test('min', async () => {
    const r = await runTest('min');
    expect(r.passed, r.error).toBe(true);
  });

  test('mean', async () => {
    const r = await runTest('mean');
    expect(r.passed, r.error).toBe(true);
  });

  test('sign', async () => {
    const r = await runTest('sign');
    expect(r.passed, r.error).toBe(true);
  });

  test('square', async () => {
    const r = await runTest('square');
    expect(r.passed, r.error).toBe(true);
  });

  test('comparison (greater)', async () => {
    const r = await runTest('comparison (greater)');
    expect(r.passed, r.error).toBe(true);
  });

  test('multi-axis sum [2,3,4]', async () => {
    const r = await runTest('multi-axis sum [2,3,4]');
    expect(r.passed, r.error).toBe(true);
  });

  test('large 1D dispatch (256K elements)', async () => {
    const r = await runTest('large 1D dispatch (256K elements)');
    expect(r.passed, r.error).toBe(true);
  });

  test('mul by scalar broadcast', async () => {
    const r = await runTest('mul by scalar broadcast');
    expect(r.passed, r.error).toBe(true);
  });

  test('mul by 0-dim scalar', async () => {
    const r = await runTest('mul by 0-dim scalar');
    expect(r.passed, r.error).toBe(true);
  });

  test('sum then mul (eager eval)', async () => {
    const r = await runTest('sum then mul (eager eval)');
    expect(r.passed, r.error).toBe(true);
  });

  test('sum then mul (lazy eval)', async () => {
    const r = await runTest('sum then mul (lazy eval)');
    expect(r.passed, r.error).toBe(true);
  });

  test('mean axis=0 on 1D', async () => {
    const r = await runTest('mean axis=0 on 1D');
    expect(r.passed, r.error).toBe(true);
  });

  test('mean axis=1 on 2D', async () => {
    const r = await runTest('mean axis=1 on 2D');
    expect(r.passed, r.error).toBe(true);
  });

  test('mean axis=-1 on 2D', async () => {
    const r = await runTest('mean axis=-1 on 2D');
    expect(r.passed, r.error).toBe(true);
  });

  test('square then reduce', async () => {
    const r = await runTest('square then reduce');
    expect(r.passed, r.error).toBe(true);
  });

  test('gather (embedding lookup)', async () => {
    const r = await runTest('gather (embedding lookup)');
    expect(r.passed, r.error).toBe(true);
  });

  test('gather respects sliced index offsets', async () => {
    const r = await runTest('gather respects sliced index offsets');
    expect(r.passed, r.error).toBe(true);
  });

  test('MoE gather_sort int64 token indices', async () => {
    const r = await runTest('MoE gather_sort int64 token indices');
    expect(r.passed, r.error).toBe(true);
  });

  test('MoE routing argpartition top-k parity', async () => {
    const r = await runTest('MoE routing argpartition top-k parity');
    expect(r.passed, r.error).toBe(true);
  });

  test('quantized affine qmatmul/gather_qmm parity', async () => {
    const r = await runTest('quantized affine qmatmul/gather_qmm parity');
    expect(r.passed, r.error).toBe(true);
  });

  test('quantized affine model-shape parity', async () => {
    const r = await runTest('quantized affine model-shape parity');
    expect(r.passed, r.error).toBe(true);
  }, 180000);

  test('rmsnorm pattern', async () => {
    const r = await runTest('rmsnorm pattern');
    expect(r.passed, r.error).toBe(true);
  });

  test('large array copy (vocab-sized)', async () => {
    const r = await runTest('large array copy (vocab-sized)');
    expect(r.passed, r.error).toBe(true);
  });

  test('chained: softmax components', async () => {
    const r = await runTest('chained: softmax components');
    expect(r.passed, r.error).toBe(true);
  });

  test('greater same-shape (no broadcast)', async () => {
    const r = await runTest('greater same-shape (no broadcast)');
    expect(r.passed, r.error).toBe(true);
  });

  test('greater broadcast (scalar threshold)', async () => {
    const r = await runTest('greater broadcast (scalar threshold)');
    expect(r.passed, r.error).toBe(true);
  });

  test('where with greater (softplus pattern)', async () => {
    const r = await runTest('where with greater (softplus pattern)');
    expect(r.passed, r.error).toBe(true);
  });

  test('large random (vocab-sized)', async () => {
    const r = await runTest('large random (vocab-sized)');
    expect(r.passed, r.error).toBe(true);
  });

  test('greaterEqual on integer arrays (causal mask creation)', async () => {
    const r = await runTest('greaterEqual on integer arrays (causal mask creation)');
    expect(r.passed, r.error).toBe(true);
  });
});
