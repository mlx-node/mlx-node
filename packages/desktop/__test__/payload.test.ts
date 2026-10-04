/**
 * `paged_attn.metallib` carries the K-quant, segmented SDPA and mixed-affine
 * kernels, and the addon has no JIT fallback for them. A library from before
 * either change is ~19.5 MB or ~31.8 MB, well over the 10 MiB size floor, so
 * the floor alone would sign it into the bundle and the first GGUF K-quant
 * or mixed-affine matmul would throw. The payload check runs
 * the same marker gate `yarn build:native` does, and takes the NAX expectation
 * from the `mlx.metallib` it ships with, not from the packaging host.
 */

import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import {
  BASE_KERNEL_MARKERS,
  BRIDGE_KERNEL_MARKERS,
  KQUANT_KERNEL_MARKERS,
  KQUANT_NAX_KERNEL_MARKERS,
  MLX_NAX_ONLY_MARKERS,
} from '../../core/metallib-select.js';
import { PayloadError, checkMetallibPair } from '../scripts/payload.js';

const PAGED_KERNELS = [
  'paged_attention_bfloat16_t_cache_bfloat16_t_hs128_bs16_nt256_nsl32_ps0',
  'reshape_and_cache_kv_bfloat16_t_cache_bfloat16_t',
  'copy_blocks_bfloat16_t',
];
const MLX_BASE = [...BASE_KERNEL_MARKERS];
const MLX_NAX = [...BASE_KERNEL_MARKERS, ...MLX_NAX_ONLY_MARKERS];
const PAGED_BASE = [...PAGED_KERNELS, ...KQUANT_KERNEL_MARKERS, ...BRIDGE_KERNEL_MARKERS];
const PAGED_NAX = [...PAGED_BASE, ...KQUANT_NAX_KERNEL_MARKERS];

/**
 * A complete MTLB container over the 10 MiB paged floor that names exactly
 * `kernels`: recognized header (min-OS 26.0), declared size = byte length.
 */
function library(kernels: readonly string[], magic = 'MTLB'): Buffer {
  const header = Buffer.alloc(24);
  header.write(magic, 0, 'latin1');
  header.writeUInt16LE(0x8001, 4);
  header.writeUInt16LE(2, 6);
  header.writeUInt16LE(9, 8);
  header[11] = 0x81;
  header.writeUInt16LE(26, 12);
  const names = Buffer.from(['', ...kernels].join('\0') + '\0');
  const metallib = Buffer.concat([header, names, Buffer.alloc(11 * 1024 * 1024)]);
  metallib.writeBigUInt64LE(BigInt(metallib.byteLength), 16);
  return metallib;
}

/** `metallib` cut to `bytes`, as an interrupted copy leaves it. */
const truncated = (metallib: Buffer, bytes = 10.5 * 1024 * 1024) => metallib.subarray(0, bytes);

describe('checkMetallibPair', () => {
  let dir: string;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'desktop-payload-'));
  });
  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  function pair(mlx: Buffer, paged: Buffer): [string, string] {
    const mlxPath = join(dir, 'mlx.metallib');
    const pagedPath = join(dir, 'paged_attn.metallib');
    writeFileSync(mlxPath, mlx);
    writeFileSync(pagedPath, paged);
    return [mlxPath, pagedPath];
  }

  it('rejects a paged-only library above the size floor', () => {
    for (const mlx of [MLX_BASE, MLX_NAX]) {
      const [m, p] = pair(library(mlx), library(PAGED_KERNELS));
      expect(() => checkMetallibPair(m, p)).toThrow(PayloadError);
      expect(() => checkMetallibPair(m, p)).toThrow(/missing K-quant kernel/);
    }
  });

  it('rejects a library with the K-quant kernels but not the segmented SDPA / mixed-affine ones', () => {
    for (const [mlx, kquantNax] of [
      [MLX_BASE, []],
      [MLX_NAX, KQUANT_NAX_KERNEL_MARKERS],
    ] as const) {
      const paged = library([...PAGED_KERNELS, ...KQUANT_KERNEL_MARKERS, ...kquantNax]);
      expect(() => checkMetallibPair(...pair(library(mlx), paged))).toThrow(
        /missing segmented SDPA \/ mixed-affine kernel/,
      );
    }
  });

  it('rejects either file when it is shorter than its header declares', () => {
    for (const [mlx, paged] of [
      [MLX_BASE, PAGED_BASE],
      [MLX_NAX, PAGED_NAX],
    ] as const) {
      expect(() => checkMetallibPair(...pair(library(mlx), truncated(library(paged))))).toThrow(/header declares/);
      expect(() => checkMetallibPair(...pair(truncated(library(mlx)), library(paged)))).toThrow(/header declares/);
    }
  });

  it('accepts a pair that agrees on NAX, either way', () => {
    expect(() => checkMetallibPair(...pair(library(MLX_BASE), library(PAGED_BASE)))).not.toThrow();
    expect(() => checkMetallibPair(...pair(library(MLX_NAX), library(PAGED_NAX)))).not.toThrow();
  });

  // A host probe that says "no NAX" (an older host, or a failed probe) would
  // accept this pair; the artifacts say the build has NAX.
  it('rejects a base-only paged library next to a NAX mlx.metallib', () => {
    expect(() => checkMetallibPair(...pair(library(MLX_NAX), library(PAGED_BASE)))).toThrow(
      KQUANT_NAX_KERNEL_MARKERS[0],
    );
  });

  it('rejects a NAX paged library next to an mlx.metallib without NAX', () => {
    expect(() => checkMetallibPair(...pair(library(MLX_BASE), library(PAGED_NAX)))).toThrow(
      /carries K-quant NAX kernels but the mlx\.metallib/,
    );
  });

  it('fails closed when mlx.metallib does not say whether it has NAX', () => {
    const cases: Buffer[] = [
      library(MLX_NAX, 'NOPE'),
      library(MLX_NAX_ONLY_MARKERS),
      library([...BASE_KERNEL_MARKERS, MLX_NAX_ONLY_MARKERS[0]]),
    ];
    for (const mlx of cases) {
      for (const paged of [PAGED_BASE, PAGED_NAX]) {
        expect(() => checkMetallibPair(...pair(mlx, library(paged)))).toThrow(/cannot tell whether .* NAX/);
      }
    }
  });

  const core = join(import.meta.dirname, '..', '..', 'core');
  const built: [string, string] = [join(core, 'mlx.metallib'), join(core, 'paged_attn.metallib')];
  it.skipIf(!built.every((path) => existsSync(path)))('accepts the pair from `yarn build:native`', () => {
    expect(() => checkMetallibPair(...built)).not.toThrow();
  });
});
