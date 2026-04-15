import { describe, test, expect, beforeAll } from 'vitest';

import { getBridge, runTest } from './test-bridge';

beforeAll(() => getBridge(), 60_000);

describe('Weights & Embedding', () => {
  test('readCppWeight API works (no model loaded)', async () => {
    const r = await runTest('readCppWeight API works (no model loaded)');
    expect(r.passed, r.error).toBe(true);
  });

  test('lazy transpose vs eager transpose equivalence', async () => {
    const r = await runTest('lazy transpose vs eager transpose equivalence');
    expect(r.passed, r.error).toBe(true);
  });

  test('embedding lookup accuracy', async () => {
    const r = await runTest('embedding lookup accuracy');
    expect(r.passed, r.error).toBe(true);
  });

  test('conv1d depthwise (GDN pattern)', async () => {
    const r = await runTest('conv1d depthwise (GDN pattern)');
    expect(r.passed, r.error).toBe(true);
  });

  test('embedding lookup [248320,1024] + matmul', async () => {
    const r = await runTest('embedding lookup [248320,1024] + matmul');
    expect(r.passed, r.error).toBe(true);
  });

  test('embedding + linear + argmax', async () => {
    const r = await runTest('embedding + linear + argmax');
    expect(r.passed, r.error).toBe(true);
  });

  test('bf16 embedding lookup model dims', async () => {
    const r = await runTest('bf16 embedding lookup model dims');
    expect(r.passed, r.error).toBe(true);
  });

  test('depthwise conv1d groups=6144 kernel=4', async () => {
    const r = await runTest('depthwise conv1d groups=6144 kernel=4');
    expect(r.passed, r.error).toBe(true);
  });
});
