import { describe, expect, it } from 'vite-plus/test';

import { MAX_YARN_CONTEXT_TOKENS } from '../../../packages/server/src/host/max-context.js';
import {
  COLD_CAPTURE_BLOCKS_ENV,
  COLD_CAPTURE_BUDGET_MS_ENV,
  PAGED_BLOCK_TOKENS,
  parseCaptureEnv,
  resolveColdCapture,
  WHOLE_PROMPT_CAPTURE_BLOCKS,
  WHOLE_PROMPT_CAPTURE_BUDGET_MS,
} from '../../../packages/server/src/host/persist-cache.js';

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
  const asked = { byDefault: true, wholePrompt: true };

  it('writes only the variables the env does not set', () => {
    expect(resolveColdCapture(asked, true, {})).toEqual({
      budget: { blocksPerTurn: 63_125, budgetMs: 10_000 },
      writes: { [COLD_CAPTURE_BLOCKS_ENV]: '63125', [COLD_CAPTURE_BUDGET_MS_ENV]: '10000' },
    });
    expect(resolveColdCapture(asked, true, { [COLD_CAPTURE_BUDGET_MS_ENV]: '500' })).toEqual({
      budget: { blocksPerTurn: 63_125, budgetMs: 500 },
      writes: { [COLD_CAPTURE_BLOCKS_ENV]: '63125' },
    });
  });

  it('keeps the native defaults when the entry does not ask, and writes nothing when the cache is off', () => {
    expect(resolveColdCapture({ byDefault: false, wholePrompt: false }, true, {})).toEqual({
      budget: { blocksPerTurn: 128, budgetMs: 250 },
      writes: {},
    });
    expect(resolveColdCapture(asked, false, {})).toEqual({ budget: null, writes: {} });
  });
});
