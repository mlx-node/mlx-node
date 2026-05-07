import { resolveServerTuningForUsage } from '@mlx-node/server';
import { describe, expect, it } from 'vite-plus/test';

describe('server timing tuning metadata', () => {
  it('mirrors native env parser defaults', () => {
    expect(resolveServerTuningForUsage({})).toEqual({
      server_paged_prefill_chunk_size: 0,
      server_paged_prefill_eval_interval: 8,
      server_paged_decode_cache_clear_interval: 64,
    });
  });

  it('accepts valid process-level tuning values', () => {
    expect(
      resolveServerTuningForUsage({
        MLX_PAGED_PREFILL_CHUNK_SIZE: '8192',
        MLX_PAGED_PREFILL_EVAL_INTERVAL: '16',
        MLX_PAGED_DECODE_CACHE_CLEAR_INTERVAL: '128',
      }),
    ).toEqual({
      server_paged_prefill_chunk_size: 8192,
      server_paged_prefill_eval_interval: 16,
      server_paged_decode_cache_clear_interval: 128,
    });
  });

  it('falls back for invalid values using native-compatible bounds', () => {
    expect(
      resolveServerTuningForUsage({
        MLX_PAGED_PREFILL_CHUNK_SIZE: '-1',
        MLX_PAGED_PREFILL_EVAL_INTERVAL: '0',
        MLX_PAGED_DECODE_CACHE_CLEAR_INTERVAL: 'abc',
      }),
    ).toEqual({
      server_paged_prefill_chunk_size: 0,
      server_paged_prefill_eval_interval: 8,
      server_paged_decode_cache_clear_interval: 64,
    });
  });
});
