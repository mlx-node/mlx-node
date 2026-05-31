#!/usr/bin/env node
/**
 * Production qmv_fast_sg4 dispatch microbench.
 *
 * This measures normal MLX `quantized_matmul` in separate child processes so
 * env-gated dispatch predicates are read cold for each variant.
 *
 *   oxnode examples/qmv-fast-sg4-microbench.ts
 */

import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import { DType, quantizedQmvMicrobench } from '@mlx-node/core';

type Shape = {
  id: string;
  label: string;
  mode: string;
  bits: number;
  groupSize: number;
  m: number;
  k: number;
  n: number;
};

type Variant = {
  id: string;
  env: Record<string, string>;
};

type ChildResult = {
  shapeId: string;
  medianNs: number;
  checksum: number;
};

const ENV_KEYS = ['MLX_MTP_SMALL_M_QMV', 'MLX_MTP_SMALL_M_QMV_BITS', 'MLX_MTP_SMALL_M_QMV_MODES'];

const VARIANTS: Variant[] = [
  { id: 'off', env: { MLX_MTP_SMALL_M_QMV: '0' } },
  {
    id: 'affine4',
    env: {
      MLX_MTP_SMALL_M_QMV: '1',
      MLX_MTP_SMALL_M_QMV_BITS: '4',
      MLX_MTP_SMALL_M_QMV_MODES: 'affine',
    },
  },
  {
    id: 'qwen-broad',
    env: {
      MLX_MTP_SMALL_M_QMV: '1',
      MLX_MTP_SMALL_M_QMV_BITS: '4,5,6,8',
      MLX_MTP_SMALL_M_QMV_MODES: 'affine,nvfp4',
    },
  },
];

const SHAPES: Shape[] = [
  {
    id: 'q_proj_affine6',
    label: 'q_proj affine6',
    mode: 'affine',
    bits: 6,
    groupSize: 64,
    m: 3,
    k: 2560,
    n: 2560,
  },
  {
    id: 'mlp_down_affine5',
    label: 'mlp_down affine5',
    mode: 'affine',
    bits: 5,
    groupSize: 64,
    m: 3,
    k: 9728,
    n: 2560,
  },
  {
    id: 'mlp_up_nvfp4',
    label: 'mlp_up nvfp4',
    mode: 'nvfp4',
    bits: 4,
    groupSize: 16,
    m: 3,
    k: 2560,
    n: 9728,
  },
  {
    id: 'mtp_affine4',
    label: 'mtp affine4',
    mode: 'affine',
    bits: 4,
    groupSize: 64,
    m: 3,
    k: 2560,
    n: 2560,
  },
];

function argValue(name: string, fallback: string): string {
  const prefix = `--${name}=`;
  const found = process.argv.find((arg) => arg.startsWith(prefix));
  return found == null ? fallback : found.slice(prefix.length);
}

const warmup = Number(argValue('warmup', '30'));
const iters = Number(argValue('iters', '200'));
if (!Number.isFinite(warmup) || warmup < 0) throw new Error('--warmup must be a non-negative number');
if (!Number.isFinite(iters) || iters <= 0) throw new Error('--iters must be a positive number');

const childIndex = process.argv.indexOf('--child');
if (childIndex !== -1) {
  const shape = JSON.parse(process.argv[childIndex + 1] ?? '') as Shape;
  const result = quantizedQmvMicrobench(
    shape.k,
    shape.n,
    shape.m,
    shape.groupSize,
    shape.bits,
    shape.mode,
    DType.BFloat16,
    warmup,
    iters,
  );
  const payload: ChildResult = {
    shapeId: shape.id,
    medianNs: result.medianNs,
    checksum: result.checksum,
  };
  console.log(JSON.stringify(payload));
  process.exit(0);
}

function formatNs(ns: number): string {
  if (ns < 1_000) return `${ns.toFixed(0)} ns`;
  if (ns < 1_000_000) return `${(ns / 1_000).toFixed(2)} us`;
  return `${(ns / 1_000_000).toFixed(2)} ms`;
}

function geomean(values: readonly number[]): number {
  if (values.length === 0) return 0;
  let sumLog = 0;
  for (const value of values) {
    if (value <= 0) return 0;
    sumLog += Math.log(value);
  }
  return Math.exp(sumLog / values.length);
}

function childEnv(variant: Variant): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = { ...process.env };
  for (const key of ENV_KEYS) delete env[key];
  for (const [key, value] of Object.entries(variant.env)) env[key] = value;
  return env;
}

function runChild(shape: Shape, variant: Variant): ChildResult {
  const script = fileURLToPath(import.meta.url);
  const child = spawnSync(
    process.env.OXNODE ?? 'oxnode',
    [script, '--child', JSON.stringify(shape), `--warmup=${warmup}`, `--iters=${iters}`],
    {
      env: childEnv(variant),
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
    },
  );
  if (child.status !== 0) {
    throw new Error(
      `child failed for ${shape.id}/${variant.id} status=${child.status}\nstdout:\n${child.stdout}\nstderr:\n${child.stderr}`,
    );
  }
  const lines = child.stdout
    .trim()
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
  const jsonLine = [...lines].reverse().find((line) => line.startsWith('{'));
  if (jsonLine == null) {
    throw new Error(`child produced no JSON for ${shape.id}/${variant.id}:\n${child.stdout}\n${child.stderr}`);
  }
  return JSON.parse(jsonLine) as ChildResult;
}

console.log('qmv_fast_sg4 production dispatch microbench');
console.log('==========================================');
console.log(`dtype=bf16 warmup=${warmup} iters=${iters}`);
console.log();
console.log('shape               mode/bits/gs   variant      median        speedup_vs_off');
console.log('------------------- -------------- ------------ ------------- --------------');

const broadRatios: number[] = [];
for (const shape of SHAPES) {
  const results = new Map<string, ChildResult>();
  for (const variant of VARIANTS) {
    results.set(variant.id, runChild(shape, variant));
  }
  const off = results.get('off');
  if (off == null) throw new Error(`missing off result for ${shape.id}`);
  for (const variant of VARIANTS) {
    const result = results.get(variant.id);
    if (result == null) throw new Error(`missing ${variant.id} result for ${shape.id}`);
    const ratio = off.medianNs / result.medianNs;
    if (variant.id === 'qwen-broad') broadRatios.push(ratio);
    console.log(
      `${shape.label.padEnd(19)} ${`${shape.mode}/${shape.bits}/gs${shape.groupSize}`.padEnd(14)} ${variant.id.padEnd(12)} ${formatNs(result.medianNs).padEnd(13)} ${`${ratio.toFixed(3)}x`.padEnd(14)}`,
    );
  }
}

const geo = geomean(broadRatios);
console.log();
console.log(`qwen-broad geomean speedup_vs_off: ${geo.toFixed(3)}x`);
console.log(
  geo >= 1.02
    ? 'decision: qwen-broad is worth end-to-end MTP A/B'
    : 'decision: qwen-broad is not proven by this microbench',
);
