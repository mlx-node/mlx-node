import { describe, test, expect, beforeAll } from 'vitest';

import { getBridge, runTest } from './test-bridge';

beforeAll(() => getBridge(), 60_000);

describe('Miscellaneous', () => {
  test('categorical sampling bf16 (temp=0.7, sliced logits)', async () => {
    const r = await runTest('categorical sampling bf16 (temp=0.7, sliced logits)');
    expect(r.passed, r.error).toBe(true);
  });

  test('KV cache bf16 write/read model dims', async () => {
    const r = await runTest('KV cache bf16 write/read model dims');
    expect(r.passed, r.error).toBe(true);
  });

  test('Q projection split bf16 correctness', async () => {
    const r = await runTest('Q projection split bf16 correctness');
    expect(r.passed, r.error).toBe(true);
  });
});
