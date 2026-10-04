/**
 * `paged_attn.metallib` carries the K-quant kernels, and the addon has no JIT
 * fallback for them. A paged-only library from before that change is ~19.5 MB,
 * well over the 10 MiB size floor, so the floor alone would sign it into the
 * bundle and the first GGUF K-quant matmul would throw. The payload check runs
 * the same marker gate `yarn build:native` does, and takes the NAX expectation
 * from the `mlx.metallib` it ships with, not from the packaging host.
 */

import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import {
  BASE_KERNEL_MARKERS,
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
const PAGED_BASE = [...PAGED_KERNELS, ...KQUANT_KERNEL_MARKERS];
const PAGED_NAX = [...PAGED_BASE, ...KQUANT_NAX_KERNEL_MARKERS];

/** An MTLB container over the 10 MiB paged floor that names exactly `kernels`. */
function library(kernels: readonly string[], magic = 'MTLB'): Buffer {
  const names = Buffer.from([magic, ...kernels].join('\0') + '\0');
  return Buffer.concat([names, Buffer.alloc(11 * 1024 * 1024)]);
}

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
