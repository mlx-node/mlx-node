import { describe, expect, it } from 'vite-plus/test';

import {
  ggufSplitPartNames,
  isBonsaiPq2GgufName,
  isGgufCompanionName,
  isQwen35XlGgufName,
  parseGgufFileName,
} from '../src/gguf-name.js';

describe('parseGgufFileName', () => {
  it('reads a llama.cpp split name, keeping the folder in the stem', () => {
    expect(parseGgufFileName('UD-Q4_K_XL/Flash-UD-Q4_K_XL-00002-of-00004.gguf')).toEqual({
      stem: 'UD-Q4_K_XL/Flash-UD-Q4_K_XL',
      ordinal: 2,
      shards: 4,
    });
  });

  it('accepts any case in the extension and the -of- separator', () => {
    expect(parseGgufFileName('M-00001-OF-00002.GGUF')).toEqual({ stem: 'M', ordinal: 1, shards: 2 });
  });

  it('treats a name without a full split suffix as a single file', () => {
    for (const path of [
      'M-Q4_K_M.gguf',
      'M-0001-of-00002.gguf',
      'M-00001-of-0002.gguf',
      'M-0000a-of-00002.gguf',
      'M00001-of-00002.gguf',
    ]) {
      expect(parseGgufFileName(path)).toEqual({ stem: path.slice(0, -'.gguf'.length), ordinal: 1, shards: 1 });
    }
  });

  it('allows an empty stem, as the native shard resolver does', () => {
    expect(parseGgufFileName('-00001-of-00002.gguf')).toEqual({ stem: '', ordinal: 1, shards: 2 });
  });

  it('rejects non-GGUF paths', () => {
    expect(parseGgufFileName('model.safetensors')).toBeUndefined();
    expect(parseGgufFileName('.gguf')).toBeUndefined();
  });
});

describe('ggufSplitPartNames', () => {
  const set = ['M-00001-of-00003.gguf', 'M-00002-of-00003.gguf', 'M-00003-of-00003.gguf'];

  it('names every part of a complete set in part order, and ignores other files', () => {
    const folder = [...set].reverse().concat(['M-00001-of-00002.gguf', 'mmproj-F16.gguf', 'm-00002-of-00003.txt']);
    expect(ggufSplitPartNames('M-00001-of-00003.gguf', folder)).toEqual(set);
    expect(ggufSplitPartNames('M-00001-of-00002.GGUF', ['M-00001-of-00002.GGUF', 'M-00002-of-00002.Gguf'])).toEqual([
      'M-00001-of-00002.GGUF',
      'M-00002-of-00002.Gguf',
    ]);
  });

  it('names the file alone when it is not a split set', () => {
    expect(ggufSplitPartNames('M-Q4_K_M.gguf', [])).toEqual(['M-Q4_K_M.gguf']);
    expect(ggufSplitPartNames('M-00001-of-00001.gguf', [])).toEqual(['M-00001-of-00001.gguf']);
  });

  it('gives nothing for a set the loader cannot open', () => {
    expect(ggufSplitPartNames('M-00001-of-00003.gguf', ['M-00001-of-00003.gguf'])).toBeUndefined();
    expect(ggufSplitPartNames('M-00001-of-00003.gguf', set.slice(0, 2))).toBeUndefined();
    expect(ggufSplitPartNames('M-00001-of-00003.gguf', [...set, 'M-00002-of-00003.GGUF'])).toBeUndefined();
    expect(ggufSplitPartNames('M-00002-of-00003.gguf', set)).toBeUndefined();
    expect(
      ggufSplitPartNames('M-00001-OF-00002.gguf', ['M-00001-OF-00002.gguf', 'M-00002-OF-00002.gguf']),
    ).toBeUndefined();
    expect(
      ggufSplitPartNames('M-00001-OF-00002.gguf', ['M-00001-OF-00002.gguf', 'M-00002-of-00002.gguf']),
    ).toBeUndefined();
    expect(ggufSplitPartNames('M-00001-of-00002.gguf', ['M-00001-of-00002.gguf', 'm-00002-of-00002.gguf'])).toBe(
      undefined,
    );
    expect(ggufSplitPartNames('M-00001-of-01025.gguf', [])).toBeUndefined();
    expect(ggufSplitPartNames('model.safetensors', [])).toBeUndefined();
  });
});

describe('isGgufCompanionName', () => {
  it('matches a companion word as a whole token, in any case', () => {
    for (const name of [
      'mmproj-F16.gguf',
      'mtp-Q8_0.gguf',
      'Model-IMATRIX.gguf',
      'x.dflash.gguf',
      'draft_q4.gguf',
      'mtp',
    ]) {
      expect(isGgufCompanionName(name), name).toBe(true);
    }
  });

  it('ignores a companion word inside a longer token', () => {
    for (const name of ['drafty-Q4_K_M.gguf', 'xmtp.gguf', 'MTP/Model-Q4_K_XL.gguf', 'mmprojector.gguf']) {
      expect(isGgufCompanionName(name), name).toBe(false);
    }
  });
});

describe('isQwen35XlGgufName', () => {
  it('admits the Unsloth Dynamic K and IQ targets after a separator, in any case', () => {
    for (const name of [
      'Qwen3.8-27B-UD-Q4_K_XL.gguf',
      'Q4_K_XL.gguf',
      'm.q12_k_m.GGUF',
      'M_IQ2_XXS.gguf',
      'M-IQ2_XS.gguf',
      'M-IQ4_NL.gguf',
      'M-iq1_s.gguf',
    ]) {
      expect(isQwen35XlGgufName(name), name).toBe(true);
    }
  });

  it('refuses other variants, missing separators, split names, and companions', () => {
    for (const name of [
      'M-Q4_0.gguf',
      'M-Q4_K_S.gguf',
      'M-Q_K_M.gguf',
      'MQ4_K_XL.gguf',
      'M-IQ4_K_M.gguf',
      'M-IQ5_XS.gguf',
      'M-Q4_K_XL-00001-of-00002.gguf',
      'mmproj-Q4_K_XL.gguf',
      'M-Q4_K_XL.safetensors',
    ]) {
      expect(isQwen35XlGgufName(name), name).toBe(false);
    }
  });
});

describe('isBonsaiPq2GgufName', () => {
  it('matches the exact Bonsai file name in any case', () => {
    expect(isBonsaiPq2GgufName('Ternary-Bonsai-2-27B-PQ2_0.gguf')).toBe(true);
    expect(isBonsaiPq2GgufName('ternary-bonsai-2-27b-pq2_0.gguf.part')).toBe(false);
    expect(isBonsaiPq2GgufName('x-ternary-bonsai-2-27b-pq2_0.gguf')).toBe(false);
  });
});
