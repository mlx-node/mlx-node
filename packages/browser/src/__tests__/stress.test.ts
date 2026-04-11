import { describe, test, expect, beforeAll } from 'vitest';
import { getBridge, runTest } from './test-bridge';

beforeAll(() => getBridge(), 60_000);

describe('Stress Tests', () => {
  test('RPC stress: rapid eval cycles with readback', async () => {
    const r = await runTest('RPC stress: rapid eval cycles with readback');
    expect(r.passed, r.error).toBe(true);
  });

  test('RPC stress: concurrent buffer lifecycle', async () => {
    const r = await runTest('RPC stress: concurrent buffer lifecycle');
    expect(r.passed, r.error).toBe(true);
  });

  test('heavy stress: 500 weight arrays + deep GDN-like graph', async () => {
    const r = await runTest('heavy stress: 500 weight arrays + deep GDN-like graph');
    expect(r.passed, r.error).toBe(true);
  });

  test('stress: weight arrays only (no compute)', async () => {
    const r = await runTest('stress: weight arrays only (no compute)');
    expect(r.passed, r.error).toBe(true);
  });

  test('stress: deep lazy graph (50 ops) then eval', async () => {
    const r = await runTest('stress: deep lazy graph (50 ops) then eval');
    expect(r.passed, r.error).toBe(true);
  });
});
