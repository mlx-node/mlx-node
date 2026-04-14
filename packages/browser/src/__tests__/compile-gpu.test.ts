import { describe, test, expect, beforeAll } from 'vitest';
import { getBridge, runTest } from './test-bridge';

beforeAll(() => getBridge(), 60_000);

describe('Compile & GPU Buffers', () => {
  test('compile basic (add, cached graph)', async () => {
    const r = await runTest('compile basic (add, cached graph)');
    expect(r.passed, r.error).toBe(true);
  });

  test('compile matmul + fused ops (5 iterations)', async () => {
    const r = await runTest('compile matmul + fused ops (5 iterations)');
    expect(r.passed, r.error).toBe(true);
  });

  test('compile repeated eval (50x heap stress)', async () => {
    const r = await runTest('compile repeated eval (50x heap stress)');
    expect(r.passed, r.error).toBe(true);
  });

  // Phase 6b: SwiGLU MLP compile fast path. Synthetic dispatch-count A/B
  // test gating on (eager - compiled) >= 40 dispatches plus byte-level
  // numerical parity. See test-worker.ts for the full assertion.
  test('compile MLP dispatch delta (Phase 6b)', async () => {
    const r = await runTest('compile MLP dispatch delta (Phase 6b)');
    expect(r.passed, r.error).toBe(true);
    // Surface the actual dispatch numbers to the test runner output so the
    // commit message / phase report can quote them without a separate
    // benchmark run. The test-worker returns { eager, compiled, delta,
    // maxAbsErr, nIters } from the run() body via the new `info` channel.
    if (r.info && typeof r.info === 'object') {
      // eslint-disable-next-line no-console
      console.log('[Phase 6b] stats =', JSON.stringify(r.info));
    }
  });

  // Phase 6c: GDN pre-recurrence compile fast path. Synthetic dispatch-count
  // A/B test gating on (eager - compiled) >= 80 dispatches plus byte-level
  // numerical parity. The prefusion body fires once per linear-attention
  // layer per token, so this is the single biggest dispatch-count lever
  // in the Phase 6 plan. See test-worker.ts for the full assertion.
  test('compile GDN prefusion dispatch delta (Phase 6c)', async () => {
    const r = await runTest('compile GDN prefusion dispatch delta (Phase 6c)');
    expect(r.passed, r.error).toBe(true);
    if (r.info && typeof r.info === 'object') {
      // eslint-disable-next-line no-console
      console.log('[Phase 6c] stats =', JSON.stringify(r.info));
    }
  });

  // Phase 6d: GDN post-recurrence compile fast path. Synthetic
  // dispatch-count A/B test gating on (eager - compiled) >= 48 dispatches
  // plus byte-level numerical parity. Same mechanism as Phase 6c but on a
  // smaller element-wise tail (sigmoid + two muls between rms_norm and
  // out-proj matmul). See test-worker.ts for the full assertion.
  test('compile GDN postfusion dispatch delta (Phase 6d)', async () => {
    const r = await runTest('compile GDN postfusion dispatch delta (Phase 6d)');
    expect(r.passed, r.error).toBe(true);
    if (r.info && typeof r.info === 'object') {
      // eslint-disable-next-line no-console
      console.log('[Phase 6d] stats =', JSON.stringify(r.info));
    }
  });

  // Phase 6e: GDN decay-gate (compute_g) compile fast path. Synthetic
  // dispatch-count A/B test gating on (eager - compiled) >= 96 dispatches
  // plus byte-level numerical parity. Fuses the 9-op softplus/exp chain
  // from gated_delta.rs into one FFI call, which compile collapses into a
  // single Compiled kernel. The biggest remaining dispatch lever — ~144
  // saved per decode token across all 18 linear-attention layers. See
  // test-worker.ts for the full assertion.
  test('compile GDN compute_g dispatch delta (Phase 6e)', async () => {
    const r = await runTest('compile GDN compute_g dispatch delta (Phase 6e)');
    expect(r.passed, r.error).toBe(true);
    if (r.info && typeof r.info === 'object') {
      // eslint-disable-next-line no-console
      console.log('[Phase 6e] stats =', JSON.stringify(r.info));
    }
  });

  test('gpu buffer array: create from eval\'d buffer, matmul 3 rounds', async () => {
    const r = await runTest('gpu buffer array: create from eval\'d buffer, matmul 3 rounds');
    expect(r.passed, r.error).toBe(true);
  });

  test('transpose of GPU-backed array correctness', async () => {
    const r = await runTest('transpose of GPU-backed array correctness');
    expect(r.passed, r.error).toBe(true);
  });

  test('matmul with transposed weight (linear_proj pattern)', async () => {
    const r = await runTest('matmul with transposed weight (linear_proj pattern)');
    expect(r.passed, r.error).toBe(true);
  });

  test('repeated matmul with transposed weight (deterministic)', async () => {
    const r = await runTest('repeated matmul with transposed weight (deterministic)');
    expect(r.passed, r.error).toBe(true);
  });

  test('scalar constant correctness (array::init)', async () => {
    const r = await runTest('scalar constant correctness (array::init)');
    expect(r.passed, r.error).toBe(true);
  });

  test('matmul determinism at model scale', async () => {
    const r = await runTest('matmul determinism at model scale');
    expect(r.passed, r.error).toBe(true);
  });

  test('large matmul 64x128*128x64', async () => {
    const r = await runTest('large matmul 64x128*128x64');
    expect(r.passed, r.error).toBe(true);
  });

  test('matmul with vocab-like dim (1x1024 * 1024x2048)', async () => {
    const r = await runTest('matmul with vocab-like dim (1x1024 * 1024x2048)');
    expect(r.passed, r.error).toBe(true);
  });

  test('chained linear: x @ W1 + b then ReLU-like', async () => {
    const r = await runTest('chained linear: x @ W1 + b then ReLU-like');
    expect(r.passed, r.error).toBe(true);
  });

  test('matmul with GPU-buffer array (model weight sim)', async () => {
    const r = await runTest('matmul with GPU-buffer array (model weight sim)');
    expect(r.passed, r.error).toBe(true);
  });

  test('large matmul (1x1024 * 1024xVocab)', async () => {
    const r = await runTest('large matmul (1x1024 * 1024xVocab)');
    expect(r.passed, r.error).toBe(true);
  });

  test('argmax on vocab-sized array', async () => {
    const r = await runTest('argmax on vocab-sized array');
    expect(r.passed, r.error).toBe(true);
  });

  test('matmul broadcast batch (multi-dim batch strides, GQA pattern)', async () => {
    const r = await runTest('matmul broadcast batch (multi-dim batch strides, GQA pattern)');
    expect(r.passed, r.error).toBe(true);
  });
});
