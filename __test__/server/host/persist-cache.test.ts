import { describe, expect, it } from 'vite-plus/test';

import { COLD_TIER_RESTORE_FAMILIES, COLD_TIER_STATE_FAMILIES } from '../../../packages/lm/src/cold-tier.js';
import { MAX_YARN_CONTEXT_TOKENS } from '../../../packages/server/src/host/max-context.js';
import {
  COLD_CAPTURE_BLOCKS_ENV,
  COLD_CAPTURE_BUDGET_MS_ENV,
  PAGED_BLOCK_TOKENS,
  PERSIST_PAGED_CACHE_ENV,
  parseCaptureEnv,
  parsePersistEnv,
  PERSIST_PAGED_CACHE_DEFAULT,
  persistPagedCacheFlag,
  resolveColdCapture,
  resolvePersistPagedCache,
  WHOLE_PROMPT_CAPTURE_BLOCKS,
  WHOLE_PROMPT_CAPTURE_BUDGET_MS,
} from '../../../packages/server/src/host/persist-cache.js';

describe('the paged KV disk cache switch', () => {
  it('reads the env as `parse_bool_env` in cold_tier.rs', () => {
    for (const on of ['1', 'true', ' ON ', 'Yes']) expect(parsePersistEnv(on)).toBe(true);
    for (const off of ['0', 'false', 'Off', ' no']) expect(parsePersistEnv(off)).toBe(false);
    for (const none of [undefined, '', 'maybe', '2']) expect(parsePersistEnv(none)).toBeUndefined();
  });

  it('resolves flag > MLX_PERSIST_PAGED_CACHE > the entry default (on)', () => {
    expect(PERSIST_PAGED_CACHE_DEFAULT).toEqual({ byDefault: true });
    expect(resolvePersistPagedCache(PERSIST_PAGED_CACHE_DEFAULT, {})).toBe(true);
    expect(resolvePersistPagedCache(PERSIST_PAGED_CACHE_DEFAULT, { MLX_PERSIST_PAGED_CACHE: '0' })).toBe(false);
    expect(resolvePersistPagedCache(PERSIST_PAGED_CACHE_DEFAULT, { MLX_PERSIST_PAGED_CACHE: 'junk' })).toBe(true);
    expect(resolvePersistPagedCache({ flag: false, byDefault: true }, { MLX_PERSIST_PAGED_CACHE: '1' })).toBe(false);
    expect(resolvePersistPagedCache({ flag: true, byDefault: true }, { MLX_PERSIST_PAGED_CACHE: '0' })).toBe(true);
    expect(resolvePersistPagedCache({ byDefault: false }, {})).toBe(false);
  });

  it('names the serve flag that carries a resolved switch', () => {
    expect(persistPagedCacheFlag(true)).toBe('--persist-cache');
    expect(persistPagedCacheFlag(false)).toBe('--no-persist-cache');
  });
});

describe('whole-prompt capture limits', () => {
  it('the block cap covers the largest window a load serves', () => {
    expect(WHOLE_PROMPT_CAPTURE_BLOCKS).toBe(63_125);
    expect(WHOLE_PROMPT_CAPTURE_BLOCKS * PAGED_BLOCK_TOKENS).toBeGreaterThanOrEqual(MAX_YARN_CONTEXT_TOKENS);
    expect((WHOLE_PROMPT_CAPTURE_BLOCKS - 1) * PAGED_BLOCK_TOKENS).toBeLessThan(MAX_YARN_CONTEXT_TOKENS);
    expect(WHOLE_PROMPT_CAPTURE_BUDGET_MS).toBe(10_000);
  });
});

describe('parseCaptureEnv — `raw.trim().parse::<u64>()`', () => {
  it('reads what native reads', () => {
    expect(parseCaptureEnv('4096')).toBe(4096);
    expect(parseCaptureEnv(' \t4096\n')).toBe(4096);
    expect(parseCaptureEnv('+7')).toBe(7);
    expect(parseCaptureEnv('0')).toBe(0);
    expect(parseCaptureEnv('000123')).toBe(123);
    expect(parseCaptureEnv(' 　 9 \u0085')).toBe(9);
    expect(parseCaptureEnv('18446744073709551615')).toBe(2 ** 64);
  });

  it('answers nothing where native falls back to its default', () => {
    for (const raw of [
      undefined,
      '',
      '   ',
      '+',
      '-1',
      '1e4',
      '10ms',
      '1_000',
      '0x10',
      '4.0',
      '18446744073709551616',
      '﻿7',
    ]) {
      expect(parseCaptureEnv(raw)).toBeUndefined();
    }
  });
});

describe('resolveColdCapture', () => {
  it('writes only the variables the env does not set', () => {
    expect(resolveColdCapture(true, true, {})).toEqual({
      budget: { blocksPerTurn: 63_125, budgetMs: 10_000 },
      writes: { [COLD_CAPTURE_BLOCKS_ENV]: '63125', [COLD_CAPTURE_BUDGET_MS_ENV]: '10000' },
    });
    expect(resolveColdCapture(true, true, { [COLD_CAPTURE_BUDGET_MS_ENV]: '500' })).toEqual({
      budget: { blocksPerTurn: 63_125, budgetMs: 500 },
      writes: { [COLD_CAPTURE_BLOCKS_ENV]: '63125' },
    });
  });

  it('keeps the native defaults when the entry does not ask, and writes nothing when the cache is off', () => {
    expect(resolveColdCapture(false, true, {})).toEqual({
      budget: { blocksPerTurn: 128, budgetMs: 250 },
      writes: {},
    });
    expect(resolveColdCapture(true, false, {})).toEqual({ budget: null, writes: {} });
  });
});

describe('Flash Next stored states on disk', () => {
  // Every host asks with `byDefault: true`; a library load asks nothing and
  // leaves the env unset.
  const host = PERSIST_PAGED_CACHE_DEFAULT;

  // Its states and their chains live in the model's own tier, not the
  // process cold tier.
  it('qwen4_exp keeps its states in its own tier', () => {
    expect(COLD_TIER_RESTORE_FAMILIES.has('qwen4_exp')).toBe(false);
    expect(COLD_TIER_STATE_FAMILIES.has('qwen4_exp')).toBe(true);
  });

  it('is on by default for every host', () => {
    expect(resolvePersistPagedCache(host, {})).toBe(true);
  });

  it('stays off when the user turns it off', () => {
    expect(resolvePersistPagedCache({ ...host, flag: false }, {})).toBe(false);
    expect(resolvePersistPagedCache(host, { [PERSIST_PAGED_CACHE_ENV]: '0' })).toBe(false);
    expect(resolvePersistPagedCache({ byDefault: false }, {})).toBe(false);
  });
});
