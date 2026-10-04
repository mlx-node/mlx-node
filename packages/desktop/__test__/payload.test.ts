/**
 * `paged_attn.metallib` carries the K-quant kernels, and the addon has no JIT
 * fallback for them. A paged-only library from before that change is ~19.5 MB,
 * well over the 10 MiB size floor, so the floor alone would sign it into the
 * bundle and the first GGUF K-quant matmul would throw. The payload check runs
 * the same marker gate `yarn build:native` does.
 */

import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { KQUANT_KERNEL_MARKERS, KQUANT_NAX_KERNEL_MARKERS, detectExpectNax } from '../../core/metallib-select.js';
import { PayloadError, checkNativeArtifact } from '../scripts/payload.js';

const PAGED_KERNELS = [
  'paged_attention_bfloat16_t_cache_bfloat16_t_hs128_bs16_nt256_nsl32_ps0',
  'reshape_and_cache_kv_bfloat16_t_cache_bfloat16_t',
  'copy_blocks_bfloat16_t',
];

/** An MTLB container over the 10 MiB floor that names exactly `kernels`. */
function library(kernels: readonly string[]): Buffer {
  const names = Buffer.from(['MTLB', ...kernels].join('\0') + '\0');
  return Buffer.concat([names, Buffer.alloc(11 * 1024 * 1024)]);
}

describe('checkNativeArtifact: paged_attn.metallib', () => {
  let dir: string;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'desktop-payload-'));
  });
  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  function write(kernels: readonly string[]): string {
    const path = join(dir, 'paged_attn.metallib');
    writeFileSync(path, library(kernels));
    return path;
  }

  it('rejects a paged-only library above the size floor', () => {
    const path = write(PAGED_KERNELS);
    for (const expectNax of [false, true]) {
      expect(() => checkNativeArtifact('paged_attn.metallib', path, { expectNax })).toThrow(PayloadError);
      expect(() => checkNativeArtifact('paged_attn.metallib', path, { expectNax })).toThrow(/missing K-quant kernel/);
    }
  });

  it('requires the K-quant NAX kernels only when the build carries them', () => {
    const path = write([...PAGED_KERNELS, ...KQUANT_KERNEL_MARKERS]);
    expect(() => checkNativeArtifact('paged_attn.metallib', path, { expectNax: false })).not.toThrow();
    expect(() => checkNativeArtifact('paged_attn.metallib', path, { expectNax: true })).toThrow(
      KQUANT_NAX_KERNEL_MARKERS[0],
    );
  });

  it('accepts a library with the K-quant and K-quant NAX kernels', () => {
    const path = write([...PAGED_KERNELS, ...KQUANT_KERNEL_MARKERS, ...KQUANT_NAX_KERNEL_MARKERS]);
    expect(() => checkNativeArtifact('paged_attn.metallib', path, { expectNax: true })).not.toThrow();
  });

  it('still applies the size floor before reading the file', () => {
    const path = join(dir, 'paged_attn.metallib');
    writeFileSync(path, Buffer.from(['MTLB', ...KQUANT_KERNEL_MARKERS].join('\0')));
    expect(() => checkNativeArtifact('paged_attn.metallib', path, { expectNax: false })).toThrow(/byte floor/);
  });

  const built = join(import.meta.dirname, '..', '..', 'core', 'paged_attn.metallib');
  it.skipIf(!existsSync(built))('accepts the paged_attn.metallib from `yarn build:native`', () => {
    expect(() => checkNativeArtifact('paged_attn.metallib', built, { expectNax: detectExpectNax() })).not.toThrow();
  });
});
