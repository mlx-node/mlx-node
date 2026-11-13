import { describe, expect, it } from 'vite-plus/test';

import {
  configQuantLabel,
  folderVariant,
  ggufModelId,
  ggufNames,
  ggufVariant,
  settleVariants,
  splitNameTail,
  type VariantMember,
} from '../src/model-variant.js';
import { MODEL_TABLE, type ModelRow } from './fixtures/model-table.js';

/** What discovery records for one row of the table, before the pass over the whole scan. */
function member(model: ModelRow): VariantMember {
  const parts = model.path.split('/');
  if (!model.path.toLowerCase().endsWith('.gguf')) {
    return { name: parts[0]!, format: 'mlx', modelType: model.modelType, ...folderVariant(parts[0]!, model.config) };
  }
  return gguf(model.path, model.modelType);
}

function settled(models: readonly ModelRow[]): VariantMember[] {
  const members = models.map(member);
  settleVariants(members);
  return members;
}

function mlx(name: string, modelType: string, config?: unknown): VariantMember {
  return { name, format: 'mlx', modelType, ...folderVariant(name, config) };
}

/** A GGUF as discovery names it when no other model asks for its id. */
function gguf(path: string, modelType: string): VariantMember {
  const parts = path.split('/');
  const repo = parts.length > 1 ? parts[0] : undefined;
  const variantFolder = parts.length > 2 ? parts[1] : undefined;
  const names = ggufNames(parts[parts.length - 1]!, repo, variantFolder);
  return {
    name: names.id,
    format: 'gguf',
    modelType,
    ...ggufVariant(names.stem, repo, variantFolder, names.id),
  };
}

describe('the rule on a models folder', () => {
  const members = settled(MODEL_TABLE);

  it('gives every model its id, base, variant and format', () => {
    expect(members.map(({ name, base, variant, format }) => ({ id: name, base, variant, format }))).toEqual(
      MODEL_TABLE.map((model) => model.expected),
    );
  });

  it('holds 77 models under 39 bases, 19 of them with two or more builds', () => {
    const bases = new Map<string, number>();
    for (const m of members) bases.set(m.base, (bases.get(m.base) ?? 0) + 1);
    expect(members).toHaveLength(77);
    expect(bases.size).toBe(39);
    expect([...bases.values()].filter((n) => n > 1)).toHaveLength(19);
    expect(Math.max(...bases.values())).toBe(10);
  });

  it('keeps the laws: unique (base, variant), one model type and one spelling per base', () => {
    expect(new Set(members.map((m) => `${m.base}\0${m.variant}`)).size).toBe(members.length);
    const types = new Map<string, Set<string>>();
    const spellings = new Map<string, Set<string>>();
    for (const m of members) {
      types.set(m.base, (types.get(m.base) ?? new Set()).add(m.modelType));
      spellings.set(m.base.toLowerCase(), (spellings.get(m.base.toLowerCase()) ?? new Set()).add(m.base));
    }
    expect([...types.values()].filter((t) => t.size > 1)).toEqual([]);
    expect([...spellings.values()].filter((s) => s.size > 1)).toEqual([]);
    expect(members.every((m) => m.base !== '' && m.variant !== '')).toBe(true);
  });

  it('changes only split ids: a split GGUF loses its shard suffix', () => {
    const before = MODEL_TABLE.map((model) => {
      const file = model.path.split('/').pop()!;
      return file.toLowerCase().endsWith('.gguf') ? file.slice(0, -'.gguf'.length) : model.path.split('/')[0]!;
    });
    expect(before.flatMap((id, i) => (id === members[i]!.name ? [] : [`${id} -> ${members[i]!.name}`]))).toEqual([
      'Corvid-70B-UD-Q3_K_XL-00001-of-00003 -> Corvid-70B-UD-Q3_K_XL',
      'Q6_K-00001-of-00002 -> rook-pack-Rook-2B-Q6_K',
    ]);
  });

  it('labels each tagged quantized folder by its config as its name does', () => {
    const tagged = MODEL_TABLE.filter((model) => {
      if (model.config === undefined || configQuantLabel(model.config) === undefined) return false;
      const tag = splitNameTail(model.path).tag;
      return tag !== undefined && tag !== 'BF16' && tag !== 'F16' && tag !== 'F32';
    });
    expect(tagged).toHaveLength(14);
    for (const model of tagged) expect(configQuantLabel(model.config)).toBe(splitNameTail(model.path).tag);
  });
});

describe('the name tail', () => {
  it.each([
    ['UD-Q4_K_XL', { base: '', tag: 'UD-Q4_K_XL', qualifiers: [], saysGguf: false }],
    [
      'Qwen3.8-Flash-Next-UD-Q4_K_XL',
      { base: 'Qwen3.8-Flash-Next', tag: 'UD-Q4_K_XL', qualifiers: [], saysGguf: false },
    ],
    ['heron-6b-unsloth-mxfp4-mlx', { base: 'heron-6b', tag: 'MXFP4', qualifiers: ['Unsloth'], saysGguf: false }],
    ['kestrel-5b-nvfp4-mtp', { base: 'kestrel-5b', tag: 'NVFP4', qualifiers: [], saysGguf: false }],
    ['nimbus-6b-it-qat-q4_0-gguf', { base: 'nimbus-6b-it-qat', tag: 'Q4_0', qualifiers: [], saysGguf: true }],
    ['tarn-22b-dflash2-gguf', { base: 'tarn-22b-dflash2', tag: undefined, qualifiers: [], saysGguf: true }],
    ['dune-1b-thinking-mlx', { base: 'dune-1b-thinking', tag: undefined, qualifiers: [], saysGguf: false }],
    ['wren-3b-it-8bit', { base: 'wren-3b-it', tag: '8-bit', qualifiers: [], saysGguf: false }],
    ['m-fp8_e4m3', { base: 'm', tag: 'FP8', qualifiers: [], saysGguf: false }],
    ['m-Q8', { base: 'm', tag: 'Q8', qualifiers: [], saysGguf: false }],
    ['m-Q4_K_M-Q8_0', { base: 'm-Q4_K_M', tag: 'Q8_0', qualifiers: [], saysGguf: false }],
    ['m-ud-mlx-Q4_K_M', { base: 'm-ud', tag: 'Q4_K_M', qualifiers: [], saysGguf: false }],
    ['Lumen-12B-KMix-9GB', { base: 'Lumen-12B-KMix-9GB', tag: undefined, qualifiers: [], saysGguf: false }],
  ])('%s', (name, tail) => {
    expect(splitNameTail(name)).toEqual(tail);
  });
});

describe('the config label', () => {
  it('takes the fewest-bit class that is not affine', () => {
    const nvfp4 = { quantization: { bits: 4, mode: 'nvfp4', 'l.0': { bits: 8, group_size: 64, mode: 'affine' } } };
    expect(configQuantLabel(nvfp4)).toBe('NVFP4');
    const mxfp = {
      quantization: {
        bits: 3,
        group_size: 64,
        mode: 'affine',
        a: { bits: 8, mode: 'mxfp8' },
        b: { bits: 4, mode: 'mxfp4' },
      },
    };
    expect(configQuantLabel(mxfp)).toBe('MXFP4');
    expect(configQuantLabel({ quantization: { bits: 4, mode: 'nvfp4', a: { bits: 8, mode: 'fp8_e4m3' } } })).toBe(
      'NVFP4',
    );
    expect(configQuantLabel({ quantization: { bits: 8, mode: 'fp8_e4m3' } })).toBe('FP8');
  });

  it('gives one bit count or the range when every class is affine', () => {
    expect(configQuantLabel({ quantization: { bits: 4, group_size: 64 } })).toBe('4-bit');
    expect(configQuantLabel({ quantization: { bits: 3, a: { bits: 6 }, b: { bits: 4, mode: 'affine' } } })).toBe(
      '3–6-bit',
    );
  });

  it('reads a foreign quantization_config only for the gemma and float-quantized methods', () => {
    expect(configQuantLabel({ quantization_config: { quant_method: 'gemma', num_bits: 4 } })).toBe('4-bit');
    expect(
      configQuantLabel({ quantization_config: { quant_method: 'compressed-tensors', format: 'float-quantized' } }),
    ).toBe('FP8');
    expect(configQuantLabel({ quantization_config: { quant_method: 'modelopt' } })).toBeUndefined();
    expect(configQuantLabel({ torch_dtype: 'bfloat16' })).toBeUndefined();
    expect(configQuantLabel(undefined)).toBeUndefined();
  });
});

describe('a GGUF', () => {
  it('keeps the file name of a single file as its id; a split set drops the suffix and a tag-only stem gets a folder', () => {
    expect(ggufNames('Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf', 'corvid-70b-gguf', 'UD-Q4_K_XL')).toEqual({
      id: 'Qwen3.8-Flash-Next-UD-Q4_K_XL',
      split: true,
      stem: 'Qwen3.8-Flash-Next-UD-Q4_K_XL',
    });
    expect(ggufNames('UD-Q4_K_XL-00001-of-00002.gguf', 'brook-gguf', undefined).id).toBe('brook-UD-Q4_K_XL');
    expect(ggufNames('UD-Q4_K_XL-00001-of-00002.gguf', 'brook-gguf', 'UD-Q4_K_XL').id).toBe('brook-UD-Q4_K_XL');
    expect(ggufNames('UD-Q4_K_XL-00001-of-00002.gguf', 'gguf', undefined).id).toBe('gguf-UD-Q4_K_XL');
    expect(ggufNames('UD-Q4_K_XL-00001-of-00002.gguf', undefined, undefined).id).toBe('UD-Q4_K_XL');
    expect(ggufNames('UD-Q4_K_XL.gguf', 'brook-gguf', undefined)).toEqual({
      id: 'UD-Q4_K_XL',
      split: false,
      stem: 'UD-Q4_K_XL',
    });
    expect(ggufNames('-00001-of-00002.gguf', 'r', undefined)).toEqual({
      id: '-00001-of-00002',
      split: true,
      stem: '-00001-of-00002',
    });
    expect(ggufModelId('UD-Q4_K_XL', 'brook-gguf', undefined)).toBe('brook-UD-Q4_K_XL');
  });

  it('takes the model of a tag-only file from a variant folder that names one, never from the repo alone', () => {
    const eight = gguf('tern-pack/Tern4-8B/Q4_K_M-00001-of-00002.gguf', 'lfm2');
    const fourteen = gguf('tern-pack/Tern4-14B/Q4_K_M-00001-of-00002.gguf', 'lfm2');
    expect([eight, fourteen].map(({ name, base, variant }) => ({ name, base, variant }))).toEqual([
      { name: 'tern-pack-Tern4-8B-Q4_K_M', base: 'tern-pack-Tern4-8B', variant: 'Q4_K_M' },
      { name: 'tern-pack-Tern4-14B-Q4_K_M', base: 'tern-pack-Tern4-14B', variant: 'Q4_K_M' },
    ]);
    expect(gguf('tern-pack/Tern4-8B/Q4_K_M.gguf', 'lfm2')).toMatchObject({
      name: 'Q4_K_M',
      base: 'tern-pack-Tern4-8B',
    });
    expect(gguf('tern4-gguf/Tern4-8B-GGUF/Q4_K_M-00001-of-00002.gguf', 'lfm2')).toMatchObject({
      name: 'Tern4-8B-Q4_K_M',
      base: 'Tern4-8B',
    });
    expect(gguf('gguf/Tern4-8B/Q4_K_M-00001-of-00002.gguf', 'lfm2')).toMatchObject({
      name: 'Tern4-8B-Q4_K_M',
      base: 'Tern4-8B',
    });
    expect(gguf('gguf/UD-Q4_K_XL/UD-Q4_K_XL-00001-of-00002.gguf', 'lfm2')).toMatchObject({
      name: 'gguf-UD-Q4_K_XL',
      base: 'gguf-UD-Q4_K_XL',
      variant: 'UD-Q4_K_XL',
    });
  });

  it('is its own base when its name holds a word the folder does not', () => {
    expect(ggufVariant('Tarn-22B-Instruct-Q4_K_M', 'tarn-22b-gguf', undefined, 'id')).toEqual({
      base: 'Tarn-22B-Instruct',
      variant: 'Q4_K_M',
    });
  });

  it('takes the folder base only when every word the folder adds is an active-parameter word', () => {
    const baseIn = (repo: string): string => ggufVariant('Foo-21B-Q4_K_M', repo, undefined, 'id').base;
    expect(ggufVariant('Brisk-1.5-18B-Q4_K_M', 'brisk-1.5-18b-a2b-gguf', undefined, 'id')).toEqual({
      base: 'brisk-1.5-18b-a2b',
      variant: 'Q4_K_M',
    });
    expect(baseIn('Foo-21B-A2B-GGUF')).toBe('Foo-21B-A2B');
    expect(baseIn('Foo-21B-a22b-GGUF')).toBe('Foo-21B-a22b');
    expect(baseIn('Foo-21B-A3.5B-GGUF')).toBe('Foo-21B-A3.5B');
    for (const repo of [
      'Foo-21B-Instruct-GGUF',
      'Foo-21B-A2B-Instruct-GGUF',
      'Foo-21B-v2-GGUF',
      'Foo-21B-1.5-GGUF',
      'Foo-21B-AB-GGUF',
      'Foo-21B-A.5B-GGUF',
      'Foo-21B-A3.B-GGUF',
      'Foo-21B-A2B5-GGUF',
      'Foo-21B--A2B-GGUF',
    ]) {
      expect(baseIn(repo)).toBe('id');
    }
    expect(ggufVariant('Tern4-Q4_K_M', 'Tern4-8B-GGUF', undefined, 'Tern4-Q4_K_M')).toEqual({
      base: 'Tern4-Q4_K_M',
      variant: 'Q4_K_M',
    });
  });

  it('is its own base when its folders name a model and none is its name or the start of it', () => {
    const baseOf = (stem: string, repo: string, sub?: string): string => ggufVariant(stem, repo, sub, 'id').base;
    expect(baseOf('model-Q4_K_M', 'Alpha-7B-GGUF')).toBe('id');
    expect(baseOf('ggml-model-Q8_0', 'Beta-9B-GGUF')).toBe('id');
    expect(baseOf('model-Q4_K_M', 'pack', 'Alpha')).toBe('id');
    expect(baseOf('Alpha-Q4_K_M', 'pack', 'Alpha')).toBe('Alpha');
    expect(baseOf('Alpha-7B-Q4_K_M', 'pack', 'alpha')).toBe('Alpha-7B');
    expect(baseOf('Tarn2.1-22B-Q4_K_M', 'Tarn2.1-GGUF')).toBe('Tarn2.1-22B');
    expect(baseOf('Brook-UD-Q4_K_XL', 'brook', 'UD-Q4_K_XL')).toBe('Brook');
    expect(baseOf('Tarn2.1-22B-Q4_K_M', 'Tarn2.1-2-GGUF')).toBe('id');
    expect(baseOf('Tarn2.1-22B-Q4_K_M', 'GGUF', 'Q4_K_M')).toBe('Tarn2.1-22B');
  });

  it('is its own base when it is named only by its tag and no folder names a model', () => {
    expect(ggufVariant('Q4_K_M', undefined, undefined, 'Q4_K_M').base).toBe('Q4_K_M');
    expect(ggufVariant('Q4_K_M', 'GGUF', undefined, 'GGUF-Q4_K_M').base).toBe('GGUF-Q4_K_M');
    expect(ggufVariant('Q8_0', 'gguf-hf', 'Q8_0', 'gguf-hf-Q8_0')).toEqual({ base: 'gguf-hf-Q8_0', variant: 'Q8_0' });
  });

  it('takes the variant folder as the label when its own name has none, else GGUF', () => {
    expect(ggufVariant('Model', 'repo', 'UD-Q4_K_XL', 'Model')).toEqual({ base: 'Model', variant: 'UD-Q4_K_XL' });
    expect(ggufVariant('Model', 'repo', 'weights', 'Model')).toEqual({ base: 'Model', variant: 'GGUF' });
    expect(ggufVariant('direct', undefined, undefined, 'direct')).toEqual({ base: 'direct', variant: 'GGUF' });
  });

  it('drops mtp from a file label', () => {
    expect(ggufVariant('Velda-40B-A4B-MTP-UD-Q4_K_XL', 'velda-40b-a4b-mtp-gguf', undefined, 'id')).toEqual({
      base: 'Velda-40B-A4B',
      variant: 'UD-Q4_K_XL',
    });
    expect(ggufVariant('Velda-40B-A4B-UD-Q4_K_XL-mtp', 'r', undefined, 'id').variant).toBe('UD-Q4_K_XL');
  });
});

describe('an MLX folder', () => {
  it('is its own base when its name says gguf', () => {
    expect(folderVariant('oriole-8b-gguf', { torch_dtype: 'bfloat16' })).toEqual({
      base: 'oriole-8b-gguf',
      variant: 'BF16',
    });
  });

  it('lets a quantization block beat a dtype word in the name, and drops mtp', () => {
    const block = { quantization: { bits: 4, group_size: 128, a: { bits: 2 } } };
    expect(folderVariant('marl-1b-lite-qat-mlx-bf16', block)).toEqual({ base: 'marl-1b-lite-qat', variant: '2–4-bit' });
    expect(folderVariant('velda-40b-a4b-mxfp8-mtp', { quantization: { bits: 8, mode: 'mxfp8' } })).toEqual({
      base: 'velda-40b-a4b',
      variant: 'MXFP8',
    });
  });

  it('keeps qat, it, instruct, thinking, assistant, dflash2 and draft in the base', () => {
    expect(folderVariant('marl-1b-lite-qat', undefined).base).toBe('marl-1b-lite-qat');
    expect(folderVariant('wren-3b-it', undefined).base).toBe('wren-3b-it');
    expect(folderVariant('wren-3b-instruct-4bit', undefined).base).toBe('wren-3b-instruct');
    expect(folderVariant('dune-1b-thinking-mlx', undefined).base).toBe('dune-1b-thinking');
    expect(folderVariant('dune-1b-assistant', undefined).base).toBe('dune-1b-assistant');
    expect(folderVariant('tarn-22b-dflash2', undefined).base).toBe('tarn-22b-dflash2');
    expect(folderVariant('tarn-22b-draft-bf16', undefined).base).toBe('tarn-22b-draft');
  });

  it('falls back to the dtype, then to MLX', () => {
    expect(folderVariant('m', { torch_dtype: 'float16' }).variant).toBe('F16');
    expect(folderVariant('m', { text_config: { dtype: 'float32' } }).variant).toBe('F32');
    expect(folderVariant('m', undefined).variant).toBe('MLX');
    expect(folderVariant('linden-11B-unsloth', undefined)).toEqual({ base: 'linden-11B', variant: 'MLX Unsloth' });
  });
});

describe('the pass over the whole scan', () => {
  it('does not group a key with two model types', () => {
    const members = [mlx('x-bf16', 'qwen3'), gguf('x-gguf/x-Q4_K_M.gguf', 'gemma4')];
    settleVariants(members);
    expect(members.map(({ name, base, variant }) => ({ name, base, variant }))).toEqual([
      { name: 'x-bf16', base: 'x-bf16', variant: 'MLX' },
      { name: 'x-Q4_K_M', base: 'x-Q4_K_M', variant: 'GGUF' },
    ]);
  });

  it('forms the groups again after a split until no base holds two model types', () => {
    const members = [
      mlx('foo-bf16', 'qwen3', { torch_dtype: 'bfloat16' }),
      gguf('foo-Q4_K_M.gguf', 'lfm2'),
      gguf('foo-bf16-Q8_0.gguf', 'lfm2'),
    ];
    settleVariants(members);
    expect(members.map(({ name, base, variant }) => ({ name, base, variant }))).toEqual([
      { name: 'foo-bf16', base: 'foo-bf16', variant: 'MLX' },
      { name: 'foo-Q4_K_M', base: 'foo-Q4_K_M', variant: 'GGUF' },
      { name: 'foo-bf16-Q8_0', base: 'foo-bf16-Q8_0', variant: 'GGUF' },
    ]);
  });

  it('gives every member the spelling that sorts first', () => {
    const members = [
      mlx('velda-40b-a4b-bf16', 'deepseek_v41'),
      gguf('Velda-40B-A4B-GGUF/Velda-40B-A4B-UD-Q4_K_XL.gguf', 'deepseek_v41'),
    ];
    settleVariants(members);
    expect(members.map((m) => m.base)).toEqual(['Velda-40B-A4B', 'Velda-40B-A4B']);
  });

  it('lets the folder that is the base keep a tied label; the others carry their id', () => {
    const config = { torch_dtype: 'bfloat16' };
    const members = [mlx('Finch-1B', 'qwen3', config), mlx('Finch-1B-mlx', 'qwen3', config)];
    members.push(mlx('finch-1b-mlx-bf16', 'qwen3', config));
    settleVariants(members);
    expect(members.map((m) => m.variant)).toEqual(['BF16', 'BF16 (Finch-1B-mlx)', 'BF16 (finch-1b-mlx-bf16)']);
    const none = [mlx('a-mlx', 'qwen3', config), mlx('a-hf', 'qwen3', config)];
    settleVariants(none);
    expect(none.map((m) => m.variant)).toEqual(['BF16 (a-mlx)', 'BF16 (a-hf)']);
  });
});
