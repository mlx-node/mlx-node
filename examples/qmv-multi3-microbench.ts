#!/usr/bin/env node
/**
 * W6.30 — MTPLX multi3 qmv4 microbench (Qwen3.5 shapes).
 *
 * Times stock `quantized_matmul` against the ported `multi3_qmv4` kernel on
 * three shapes mirroring the Qwen3.5 verify hot path:
 *   - Q projection:  K=2560, N=2560
 *   - MLP up:         K=2560, N=9728
 *   - MLP down:       K=9728, N=2560
 *
 * All shapes use group_size=64, bf16 inputs, and warmup=10 / iters=50 by
 * default. The harness reports per-shape (stock_ns, kernel_ns, ratio) plus a
 * geometric-mean ratio across shapes. Ratio > 1 means the kernel beats stock.
 *
 * Per the W6.30 decision gate (research-only — does NOT gate W6.31/W6.37
 * production integration; see plan §Phase 2b prerequisites):
 *   - geomean >= 1.4×   → AFFINE M=3 microbench PASS
 *   - 1.2× to 1.4×       → AFFINE M=3 microbench MARGINAL (research only)
 *   - < 1.2×             → AFFINE M=3 microbench FAIL (kernel stays in tree)
 *
 * Independently, any shape with max_abs_diff > PARITY_TOL (5e-2, matching
 * the Rust `parity_tests::PARITY_TOL_BF16`) fails the gate closed regardless
 * of speed: a fast-but-wrong kernel cannot be greenlit.
 *
 *   oxnode examples/qmv-multi3-microbench.ts
 */

import { DType, multi3Qmv4Microbench } from '@mlx-node/core';

interface Shape {
  label: string;
  k: number;
  n: number;
}

const SHAPES: readonly Shape[] = [
  { label: 'q_proj   (K=2560, N=2560)', k: 2560, n: 2560 },
  { label: 'mlp_up   (K=2560, N=9728)', k: 2560, n: 9728 },
  { label: 'mlp_down (K=9728, N=2560)', k: 9728, n: 2560 },
];

const GROUP_SIZE = 64;
const WARMUP = 30;
const ITERS = 200;

function geomean(values: readonly number[]): number {
  if (values.length === 0) return 0;
  let sumLog = 0;
  for (const v of values) {
    if (v <= 0) return 0;
    sumLog += Math.log(v);
  }
  return Math.exp(sumLog / values.length);
}

function formatNs(ns: number): string {
  if (ns < 1_000) return `${ns.toFixed(0)} ns`;
  if (ns < 1_000_000) return `${(ns / 1_000).toFixed(2)} µs`;
  return `${(ns / 1_000_000).toFixed(2)} ms`;
}

// Pre-gated on parity above; this only labels the speed tier on a parity-passing run.
function decision(geo: number): string {
  if (geo >= 1.4) return 'AFFINE M=3 microbench PASS — does NOT gate W6.31/W6.37; see plan §Phase 2b prerequisites';
  if (geo >= 1.2) return 'AFFINE M=3 microbench MARGINAL — research only';
  return 'AFFINE M=3 microbench FAIL — kernel stays in tree for research';
}

/**
 * Parity tolerance for the per-shape stock-vs-kernel max-abs-diff probe.
 * Matches the Rust-side `parity_tests::PARITY_TOL_BF16`. A fast-but-wrong
 * kernel must fail the gate closed regardless of speed ratio.
 */
const PARITY_TOL = 5e-2;

function main(): void {
  console.log('W6.30 multi3_qmv4 microbench');
  console.log('============================');
  console.log(`group_size=${GROUP_SIZE} dtype=bf16 warmup=${WARMUP} iters=${ITERS}`);
  console.log();
  console.log('shape                            stock          kernel         ratio   diff');
  console.log('-------------------------------- -------------- -------------- ------- ------------');

  const ratios: number[] = [];
  const parityFailures: { label: string; diff: number }[] = [];
  for (const shape of SHAPES) {
    const r = multi3Qmv4Microbench(shape.k, shape.n, GROUP_SIZE, DType.BFloat16, WARMUP, ITERS);
    ratios.push(r.ratio);
    if (r.maxAbsDiff > PARITY_TOL) {
      parityFailures.push({ label: shape.label, diff: r.maxAbsDiff });
    }
    console.log(
      `${shape.label.padEnd(32)} ${formatNs(r.stockNs).padEnd(14)} ${formatNs(r.kernelNs).padEnd(14)} ${`${r.ratio.toFixed(2)}×`.padEnd(7)} ${r.maxAbsDiff.toExponential(2).padEnd(12)}`,
    );
  }

  // Parity fails closed: a fast-but-wrong kernel must NOT pass the speed
  // gate. Bail before reporting geomean so the failure is the headline.
  if (parityFailures.length > 0) {
    console.log();
    for (const f of parityFailures) {
      console.log(
        `decision: FAIL — parity broken (shape ${f.label}: max_abs_diff=${f.diff.toExponential(2)} > tol ${PARITY_TOL.toExponential(2)})`,
      );
    }
    process.exit(1);
  }

  const geo = geomean(ratios);
  console.log();
  console.log(`geomean ratio: ${geo.toFixed(2)}×`);
  console.log(`decision:      ${decision(geo)}`);
}

main();
