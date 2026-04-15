import { describe, test, expect, beforeAll } from 'vitest';

import { getBridge, runTest } from './test-bridge';

beforeAll(() => getBridge(), 60_000);

describe('GDN (Gated Delta Network)', () => {
  test('GDN recurrence step correctness', async () => {
    const r = await runTest('GDN recurrence step correctness');
    expect(r.passed, r.error).toBe(true);
  });

  test('GDN forward mini (conv+split+reshape)', async () => {
    const r = await runTest('GDN forward mini (conv+split+reshape)');
    expect(r.passed, r.error).toBe(true);
  });

  test('conv state pattern (GDN-like)', async () => {
    const r = await runTest('conv state pattern (GDN-like)');
    expect(r.passed, r.error).toBe(true);
  });

  test('GDN step pattern (4D broadcast mul+sum)', async () => {
    const r = await runTest('GDN step pattern (4D broadcast mul+sum)');
    expect(r.passed, r.error).toBe(true);
  });

  test('GDN step model-sized', async () => {
    const r = await runTest('GDN step model-sized');
    expect(r.passed, r.error).toBe(true);
  });

  test('broadcast [Hv] * [B,T,Hv] model-scale', async () => {
    const r = await runTest('broadcast [Hv] * [B,T,Hv] model-scale');
    expect(r.passed, r.error).toBe(true);
  });

  test('compute_g full chain model-scale', async () => {
    const r = await runTest('compute_g full chain model-scale');
    expect(r.passed, r.error).toBe(true);
  });

  test('compute_g pattern (softplus chain)', async () => {
    const r = await runTest('compute_g pattern (softplus chain)');
    expect(r.passed, r.error).toBe(true);
  });

  test('GDN recurrence: state decay + update', async () => {
    const r = await runTest('GDN recurrence: state decay + update');
    expect(r.passed, r.error).toBe(true);
  });

  test('GDN isolated recurrence (small dims)', async () => {
    const r = await runTest('GDN isolated recurrence (small dims)');
    expect(r.passed, r.error).toBe(true);
  });

  test('GDN full forward bf16 (proj+conv1d+silu+recurrence+norm+proj)', async () => {
    const r = await runTest('GDN full forward bf16 (proj+conv1d+silu+recurrence+norm+proj)');
    expect(r.passed, r.error).toBe(true);
  });

  test('GDN 5-step recurrence bf16 state', async () => {
    const r = await runTest('GDN 5-step recurrence bf16 state');
    expect(r.passed, r.error).toBe(true);
  });

  test('GDN decoder layer bf16 (norm+GDN+residual+norm+MLP+residual)', async () => {
    const r = await runTest('GDN decoder layer bf16 (norm+GDN+residual+norm+MLP+residual)');
    expect(r.passed, r.error).toBe(true);
  });

  test('GDN real dims: concat+add [1,4,6144]', async () => {
    const r = await runTest('GDN real dims: concat+add [1,4,6144]');
    expect(r.passed, r.error).toBe(true);
  });

  test('GDN in_proj matmul [1,20,1024]@[1024,8192]', async () => {
    const r = await runTest('GDN in_proj matmul [1,20,1024]@[1024,8192]');
    expect(r.passed, r.error).toBe(true);
  });
});
