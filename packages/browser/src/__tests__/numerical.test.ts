import { describe, test, expect, beforeAll } from 'vitest';

import { getBridge, runTest } from './test-bridge';

beforeAll(() => getBridge(), 60_000);

describe('Numerical Accuracy', () => {
  test('RMSNorm numerical accuracy check', async () => {
    const r = await runTest('RMSNorm numerical accuracy check');
    expect(r.passed, r.error).toBe(true);
  });

  test('softmax numerical accuracy', async () => {
    const r = await runTest('softmax numerical accuracy');
    expect(r.passed, r.error).toBe(true);
  });

  test('multi-layer chained RMSNorm+matmul accuracy', async () => {
    const r = await runTest('multi-layer chained RMSNorm+matmul accuracy');
    expect(r.passed, r.error).toBe(true);
  });

  test('softplus numerical stability', async () => {
    const r = await runTest('softplus numerical stability');
    expect(r.passed, r.error).toBe(true);
  });

  test('compute_g: exp(-A*softplus(x)) no precision loss', async () => {
    const r = await runTest('compute_g: exp(-A*softplus(x)) no precision loss');
    expect(r.passed, r.error).toBe(true);
  });
});
