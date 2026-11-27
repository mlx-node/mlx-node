import { describe, expect, it } from 'vite-plus/test';

import {
  ggufSplitPartNames,
  isBonsaiPq2GgufName,
  isGgufCompanionName,
  isQwen35XlGgufName,
  parseGgufFileName,
  pickGgufProjector,
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
      'dspark-Inkling-2-Mini-Q8_0.gguf',
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

describe('pickGgufProjector', () => {
  const pick = (architecture: string, names: readonly string[], weight = 'M-UD-Q4_K_XL.gguf') =>
    pickGgufProjector(architecture, weight, [weight, ...names])?.name;

  it('takes BF16, then F16, then F32 beside a Qwen3.5 weight', () => {
    expect(pick('qwen35', ['mmproj-F32.gguf', 'mmproj-F16.gguf', 'mmproj-BF16.gguf'])).toBe('mmproj-BF16.gguf');
    expect(pick('qwen35', ['mmproj-F32.gguf', 'mmproj-F16.gguf'])).toBe('mmproj-F16.gguf');
  });

  it('takes F32 before F16 beside an Inkling2 weight', () => {
    for (const architecture of ['inkling2', 'inkling2_loop']) {
      expect(pick(architecture, ['mmproj-F16.gguf', 'mmproj-F32.gguf']), architecture).toBe('mmproj-F32.gguf');
    }
  });

  it('takes mmproj-<weight> first beside a Gemma4 weight', () => {
    const weight = 'gemma-4-26B-A4B-it-Q4_0.gguf';
    expect(pick('gemma4', ['mmproj-BF16.gguf', `mmproj-${weight}`], weight)).toBe(`mmproj-${weight}`);
  });

  it('takes no projector that names none of the precisions', () => {
    expect(pickGgufProjector('qwen35', 'M-UD-Q4_K_XL.gguf', ['mmproj-model-Q8_0.gguf'])).toEqual({
      name: undefined,
      tied: [],
    });
  });

  it('takes a lone projector with no precision word beside an Inkling2 weight, and ranks it last', () => {
    expect(pick('inkling2', ['mmproj-model.gguf'])).toBe('mmproj-model.gguf');
    expect(pick('inkling2', ['mmproj-model.gguf', 'mmproj-Q8_0.gguf'])).toBe('mmproj-Q8_0.gguf');
    expect(pickGgufProjector('inkling2', 'M.gguf', ['mmproj-b.gguf', 'mmproj-a.gguf'])).toEqual({
      name: undefined,
      tied: ['mmproj-a.gguf', 'mmproj-b.gguf'],
    });
  });

  it('takes none of two projectors of one precision, and names both', () => {
    expect(
      pickGgufProjector('qwen35', 'M.gguf', ['mmproj-b-BF16.gguf', 'mmproj-a-bf16.gguf', 'mmproj-F16.gguf']),
    ).toEqual({ name: undefined, tied: ['mmproj-a-bf16.gguf', 'mmproj-b-BF16.gguf'] });
  });

  it('answers nothing for an architecture whose loader takes no projector', () => {
    for (const architecture of ['qwen35moe', 'clef', 'qwen4exp', 'hy_v4_flash', 'deepseek41', 'muse-glimmer']) {
      expect(pickGgufProjector(architecture, 'M.gguf', ['mmproj-BF16.gguf']), architecture).toBeUndefined();
    }
  });
});
