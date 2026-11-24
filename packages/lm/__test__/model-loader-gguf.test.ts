/// <reference types="node" />

import { existsSync } from 'node:fs';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  DeepseekV41Model,
  detectModelType,
  HyV4FlashModel,
  Inkling2Model,
  loadModel,
  MuseGlimmerModel,
  Qwen35Model,
  Qwen35MoeModel,
  Qwen4ExpModel,
  supportsAutomaticDFlash2,
} from '@mlx-node/lm';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

const cleanups: Array<() => Promise<void>> = [];

afterEach(async () => {
  vi.restoreAllMocks();
  while (cleanups.length > 0) await cleanups.pop()!();
});

function u32(value: number): Buffer {
  const buffer = Buffer.alloc(4);
  buffer.writeUInt32LE(value);
  return buffer;
}

function u64(value: number): Buffer {
  const buffer = Buffer.alloc(8);
  buffer.writeBigUInt64LE(BigInt(value));
  return buffer;
}

function ggufString(value: string): Buffer {
  const bytes = Buffer.from(value, 'utf8');
  return Buffer.concat([u64(bytes.length), bytes]);
}

function minimalGguf(architecture: string): Buffer {
  return Buffer.concat([
    Buffer.from('GGUF'),
    u32(3), // GGUF version
    u64(0), // tensor count
    u64(1), // metadata count
    ggufString('general.architecture'),
    u32(8), // GGUF_TYPE_STRING
    ggufString(architecture),
  ]);
}

async function writeStandaloneGguf(architecture: string): Promise<{ root: string; modelPath: string }> {
  const root = await mkdtemp(join(tmpdir(), 'mlx-model-loader-gguf-'));
  cleanups.push(() => rm(root, { recursive: true, force: true }));
  const modelPath = join(root, 'model.gguf');
  await writeFile(modelPath, minimalGguf(architecture));
  return { root, modelPath };
}

describe('standalone GGUF model detection', () => {
  it('passes an explicit Qwen4 auxiliary checkpoint to the native loader', async () => {
    const { modelPath } = await writeStandaloneGguf('qwen4exp');
    const expected = {} as Qwen4ExpModel;
    const loader = vi.spyOn(Qwen4ExpModel, 'load').mockResolvedValue(expected);
    expect(await loadModel(modelPath, { auxiliaryModelPath: '/matching/hf', autoLoadDraft: false })).toBe(expected);
    expect(loader).toHaveBeenCalledWith(modelPath, { auxiliaryModelPath: '/matching/hf' });
  });

  // Mutation: drop `draftModelPath` from `qwen4ExpLoadOptions` → red.
  // Mutation: drop `acceptsDraftModel` from the qwen4_exp row → red (rejected).
  it('passes a Flash Next draft to the native loader; a GGUF with no draft beside it passes none', async () => {
    const { modelPath } = await writeStandaloneGguf('qwen4exp');
    const loader = vi.spyOn(Qwen4ExpModel, 'load').mockResolvedValue({} as Qwen4ExpModel);
    await loadModel(modelPath, { draftModelPath: '/flash-next-draft' });
    expect(loader).toHaveBeenLastCalledWith(modelPath, { draftModelPath: '/flash-next-draft' });
    await loadModel(modelPath);
    expect(loader).toHaveBeenLastCalledWith(modelPath, null);
  });

  // Mutation: discover a Flash Next draft in `findDFlash2Draft` → red.
  it('attaches no Flash Next draft found beside the GGUF; the option attaches it', async () => {
    const { root, modelPath } = await writeStandaloneGguf('qwen4exp');
    const draft = join(root, 'draft');
    await mkdir(draft);
    await writeFile(
      join(draft, 'config.json'),
      JSON.stringify({ architectures: ['DFlash2DraftModel'], hidden_size: 2560, num_target_layers: 48 }),
    );
    await writeFile(join(draft, 'model.safetensors'), 'weights');
    const loader = vi.spyOn(Qwen4ExpModel, 'load').mockResolvedValue({} as Qwen4ExpModel);
    await loadModel(modelPath);
    expect(loader, 'no option: no draft').toHaveBeenLastCalledWith(modelPath, null);
    await loadModel(modelPath, { draftModelPath: draft });
    expect(loader, 'the option attaches it').toHaveBeenLastCalledWith(modelPath, { draftModelPath: draft });
  });

  it('rejects Qwen4 auxiliary options for other model families', async () => {
    const { modelPath } = await writeStandaloneGguf('qwen35');
    await expect(loadModel(modelPath, { auxiliaryModelPath: '/matching/hf' })).rejects.toThrow(
      'only supported by qwen4_exp',
    );
  });

  it('passes kvFormat to the dense Qwen3.5 loader alongside the draft path', async () => {
    const { modelPath } = await writeStandaloneGguf('qwen35');
    const loaded = {} as Qwen35Model;
    const loadSpy = vi.spyOn(Qwen35Model, 'load').mockResolvedValue(loaded);
    await expect(loadModel(modelPath, { kvFormat: 'int8', autoLoadDraft: false })).resolves.toBe(loaded);
    expect(loadSpy).toHaveBeenLastCalledWith(modelPath, { draftModelPath: undefined, kvFormat: 'int8' });
    await loadModel(modelPath, { kvFormat: 'bf16', draftModelPath: '/draft' });
    expect(loadSpy).toHaveBeenLastCalledWith(modelPath, { draftModelPath: '/draft', kvFormat: 'bf16' });
  });

  // Mutation: drop `activations` from `qwen35LoadOptions` → red.
  it('passes activations to the dense Qwen3.5 loader; rejects it for gemma4', async () => {
    const { modelPath } = await writeStandaloneGguf('qwen35');
    const loadSpy = vi.spyOn(Qwen35Model, 'load').mockResolvedValue({} as Qwen35Model);
    await loadModel(modelPath, { activations: 'auto', autoLoadDraft: false });
    expect(loadSpy).toHaveBeenLastCalledWith(modelPath, expect.objectContaining({ activations: 'auto' }));
    const gemma = await writeStandaloneGguf('gemma4');
    await expect(loadModel(gemma.modelPath, { activations: 'auto' })).rejects.toThrow(
      'only supported by qwen3_5, qwen3_5_moe and inkling2',
    );
  });

  // Mutation: drop `kvFormat` from `qwen4ExpLoadOptions` → red.
  it('passes kvFormat to the Flash Next loader; rejects it for gemma4', async () => {
    const flashNext = await writeStandaloneGguf('qwen4exp');
    const loadSpy = vi.spyOn(Qwen4ExpModel, 'load').mockResolvedValue({} as Qwen4ExpModel);
    await loadModel(flashNext.modelPath, { kvFormat: 'int8' });
    expect(loadSpy).toHaveBeenLastCalledWith(flashNext.modelPath, { kvFormat: 'int8' });
    const gemma = await writeStandaloneGguf('gemma4');
    await expect(loadModel(gemma.modelPath, { kvFormat: 'int8' })).rejects.toThrow(
      'only supported by qwen3_5, qwen3_5_moe, qwen4_exp and inkling2',
    );
  });

  // Mutation: drop `maxMemoryBytes` from `qwen35LoadOptions` → red. Mutation:
  // drop `maxMemoryBytes` from `qwen4ExpLoadOptions` → red. Mutation: drop the
  // `acceptsMaxMemory` check → red (deepseek_v41 accepts it).
  it('passes maxMemoryBytes to the Qwen3.5 and Flash Next loaders and rejects it elsewhere', async () => {
    const dense = await writeStandaloneGguf('qwen35');
    const denseSpy = vi.spyOn(Qwen35Model, 'load').mockResolvedValue({} as Qwen35Model);
    await loadModel(dense.modelPath, { maxMemoryBytes: 30_064_771_072, autoLoadDraft: false });
    expect(denseSpy).toHaveBeenLastCalledWith(dense.modelPath, { maxMemoryBytes: 30_064_771_072 });
    const moe = await writeStandaloneGguf('qwen35moe');
    const moeSpy = vi.spyOn(Qwen35MoeModel, 'load').mockResolvedValue({} as Qwen35MoeModel);
    await loadModel(moe.modelPath, { maxMemoryBytes: 1_024, autoLoadDraft: false });
    expect(moeSpy).toHaveBeenLastCalledWith(moe.modelPath, { maxMemoryBytes: 1_024 });
    const flashNext = await writeStandaloneGguf('qwen4exp');
    const flashNextSpy = vi.spyOn(Qwen4ExpModel, 'load').mockResolvedValue({} as Qwen4ExpModel);
    await loadModel(flashNext.modelPath, { maxMemoryBytes: 2_048, autoLoadDraft: false });
    expect(flashNextSpy).toHaveBeenLastCalledWith(flashNext.modelPath, { maxMemoryBytes: 2_048 });
    await loadModel(flashNext.modelPath, { auxiliaryModelPath: '/matching/hf', maxMemoryBytes: 4_096 });
    expect(flashNextSpy).toHaveBeenLastCalledWith(flashNext.modelPath, {
      auxiliaryModelPath: '/matching/hf',
      maxMemoryBytes: 4_096,
    });
    const other = await writeStandaloneGguf('deepseek41');
    await expect(loadModel(other.modelPath, { maxMemoryBytes: 1_024 })).rejects.toThrow(
      'maxMemoryBytes (Metal memory budget) is only supported by qwen3_5, qwen3_5_moe, qwen4_exp, hy_v4_flash and inkling2',
    );
  });

  // Mutation: drop `maxContextTokens` from `qwen35LoadOptions`' null test → red
  // (a lone count never reaches the loader). Mutation: drop the
  // `acceptsMaxContext` check → red (deepseek_v41 accepts it).
  it('passes maxContextTokens to the Qwen3.5 and Flash Next loaders and rejects it elsewhere', async () => {
    const dense = await writeStandaloneGguf('qwen35');
    const denseSpy = vi.spyOn(Qwen35Model, 'load').mockResolvedValue({} as Qwen35Model);
    await loadModel(dense.modelPath, { maxContextTokens: 102_400, autoLoadDraft: false });
    expect(denseSpy).toHaveBeenLastCalledWith(dense.modelPath, { maxContextTokens: 102_400 });
    const moe = await writeStandaloneGguf('qwen35moe');
    const moeSpy = vi.spyOn(Qwen35MoeModel, 'load').mockResolvedValue({} as Qwen35MoeModel);
    await loadModel(moe.modelPath, { maxContextTokens: 409_600, autoLoadDraft: false });
    expect(moeSpy).toHaveBeenLastCalledWith(moe.modelPath, { maxContextTokens: 409_600 });
    const flashNext = await writeStandaloneGguf('qwen4exp');
    const flashNextSpy = vi.spyOn(Qwen4ExpModel, 'load').mockResolvedValue({} as Qwen4ExpModel);
    await loadModel(flashNext.modelPath, { maxContextTokens: 102_400, autoLoadDraft: false });
    expect(flashNextSpy).toHaveBeenLastCalledWith(flashNext.modelPath, { maxContextTokens: 102_400 });
    const other = await writeStandaloneGguf('deepseek41');
    await expect(loadModel(other.modelPath, { maxContextTokens: 102_400 })).rejects.toThrow(
      'maxContextTokens (context token limit) is only supported by qwen3_5, qwen3_5_moe, qwen4_exp, hy_v4_flash and inkling2',
    );
  });

  // Mutation: leave `cacheDisk` / `cacheDir` out of `qwen4ExpLoadOptions`' null
  // test → red (a lone option never reaches the loader).
  it('passes cacheDisk and cacheDir alone to the Flash Next loader and rejects them for gemma4', async () => {
    const flashNext = await writeStandaloneGguf('qwen4exp');
    const loadSpy = vi.spyOn(Qwen4ExpModel, 'load').mockResolvedValue({} as Qwen4ExpModel);
    await loadModel(flashNext.modelPath, { cacheDisk: '0', autoLoadDraft: false });
    expect(loadSpy).toHaveBeenLastCalledWith(flashNext.modelPath, { cacheDisk: '0' });
    await loadModel(flashNext.modelPath, { cacheDir: '/var/cache/mlx', autoLoadDraft: false });
    expect(loadSpy).toHaveBeenLastCalledWith(flashNext.modelPath, { cacheDir: '/var/cache/mlx' });
    const gemma = await writeStandaloneGguf('gemma4');
    await expect(loadModel(gemma.modelPath, { cacheDisk: '0' })).rejects.toThrow(
      'cacheDisk and cacheDir (the SSD state tier) are only supported by qwen3_5, qwen3_5_moe, qwen4_exp, hy_v4_flash and inkling2',
    );
  });

  it('passes a MoE companion and cache precision to the native loader', async () => {
    const { modelPath } = await writeStandaloneGguf('qwen35moe');
    const loaded = {} as Qwen35MoeModel;
    const loadSpy = vi.spyOn(Qwen35MoeModel, 'load').mockResolvedValue(loaded);
    await expect(loadModel(modelPath, { draftModelPath: '/moe-draft', kvFormat: 'int8' })).resolves.toBe(loaded);
    expect(loadSpy).toHaveBeenLastCalledWith(modelPath, { draftModelPath: '/moe-draft', kvFormat: 'int8' });
    await loadModel(modelPath, { autoLoadDraft: false });
    expect(loadSpy).toHaveBeenLastCalledWith(modelPath, null);
  });

  it('detects DeepSeek-V4.1 GGUF and loads it by path alone', async () => {
    const { modelPath } = await writeStandaloneGguf('deepseek41');
    expect(await detectModelType(modelPath)).toBe('deepseek_v41');
    const loaded = {} as DeepseekV41Model;
    const loadSpy = vi.spyOn(DeepseekV41Model, 'load').mockResolvedValue(loaded);
    await expect(loadModel(modelPath)).resolves.toBe(loaded);
    expect(loadSpy).toHaveBeenLastCalledWith(modelPath);
    await expect(loadModel(modelPath, { draftModelPath: '/draft' })).rejects.toThrow('only supported by gemma4');
  });

  // Mutation: drop `maxContextTokens` from `hyV4FlashLoadOptions` → red.
  // Mutation: drop `acceptsMaxContext` from the hy_v4_flash row → red.
  it('detects Hy4-flash GGUF and passes it maxContextTokens alone', async () => {
    const { modelPath } = await writeStandaloneGguf('hy_v4_flash');
    expect(await detectModelType(modelPath)).toBe('hy_v4_flash');
    const loaded = {} as HyV4FlashModel;
    const loadSpy = vi.spyOn(HyV4FlashModel, 'load').mockResolvedValue(loaded);
    await expect(loadModel(modelPath)).resolves.toBe(loaded);
    expect(loadSpy).toHaveBeenLastCalledWith(modelPath, null);
    await loadModel(modelPath, { maxContextTokens: 131_072 });
    expect(loadSpy).toHaveBeenLastCalledWith(modelPath, { maxContextTokens: 131_072 });
  });

  // Mutation: leave any of the three out of `hyV4FlashLoadOptions` (its null
  // test or its object) → red. Mutation: drop `acceptsMaxMemory` or
  // `acceptsCacheDisk` from the hy_v4_flash row → red (the load is refused).
  it.each([
    [{ maxMemoryBytes: 30_064_771_072 }],
    [{ cacheDisk: '0' }],
    [{ cacheDir: '/var/cache/mlx' }],
    [{ maxMemoryBytes: 1_024, maxContextTokens: 131_072, cacheDisk: 'auto', cacheDir: '/var/cache/mlx' }],
  ])('passes %j to the Hy4-flash loader', async (options) => {
    const { modelPath } = await writeStandaloneGguf('hy_v4_flash');
    const loadSpy = vi.spyOn(HyV4FlashModel, 'load').mockResolvedValue({} as HyV4FlashModel);
    await loadModel(modelPath, options);
    expect(loadSpy).toHaveBeenLastCalledWith(modelPath, options);
  });

  it.each([
    [{ kvFormat: 'int8' as const }, 'kvFormat (K/V cache format) is only supported by'],
    [{ decodeShare: 0.5 }, 'decodeShare (decode time owed per unit of prefill time) is only supported by'],
    [{ draftModelPath: '/draft' }, 'draftModelPath (speculative-decoding draft) is only supported by'],
    [{ auxiliaryModelPath: '/matching/hf' }, 'auxiliaryModelPath is only supported by qwen4_exp'],
  ])('rejects %j for Hy4-flash before the native load', async (options, message) => {
    const { modelPath } = await writeStandaloneGguf('hy_v4_flash');
    const loadSpy = vi.spyOn(HyV4FlashModel, 'load').mockResolvedValue({} as HyV4FlashModel);
    await expect(loadModel(modelPath, options)).rejects.toThrow(message);
    expect(loadSpy).not.toHaveBeenCalled();
  });

  it('detects Qwen3.8-Flash-Next as its own architecture', async () => {
    const { modelPath } = await writeStandaloneGguf('qwen4exp');
    expect(await detectModelType(modelPath)).toBe('qwen4_exp');
  });

  it('loads a standalone Qwen target without attaching a companion above its directory', async () => {
    const root = await mkdtemp(join(tmpdir(), 'mlx-standalone-companion-'));
    cleanups.push(() => rm(root, { recursive: true, force: true }));
    const store = join(root, 'models');
    await mkdir(store);
    const modelPath = join(store, 'Qwen3.8-27B-UD-Q4_K_XL.gguf');
    await writeFile(modelPath, minimalGguf('qwen35'));
    const writeDraft = async (parent: string): Promise<string> => {
      const draft = join(parent, 'qwen3.8-27b-dflash2');
      await mkdir(draft);
      await writeFile(join(draft, 'config.json'), JSON.stringify({ architectures: ['DFlash2DraftModel'] }));
      await writeFile(join(draft, 'model.safetensors'), 'draft weights');
      return draft;
    };
    await writeDraft(root);
    const loaded = {} as Qwen35Model;
    const loadSpy = vi.spyOn(Qwen35Model, 'load').mockResolvedValue(loaded);
    await expect(loadModel(modelPath)).resolves.toBe(loaded);
    expect(loadSpy).toHaveBeenLastCalledWith(modelPath, null);

    const inside = await writeDraft(store);
    await loadModel(modelPath);
    expect(loadSpy).toHaveBeenLastCalledWith(
      modelPath,
      supportsAutomaticDFlash2('qwen3_5') ? { draftModelPath: inside } : null,
    );
    await rm(inside, { recursive: true });
    await loadModel(modelPath);
    expect(loadSpy).toHaveBeenLastCalledWith(modelPath, null);
  });

  it('maps the qwen35 header to qwen3_5 without config.json', async () => {
    const { modelPath } = await writeStandaloneGguf('qwen35');
    await expect(detectModelType(modelPath)).resolves.toBe('qwen3_5');
  });

  it('maps the sparse qwen35moe header to qwen3_5_moe and passes the file to its loader', async () => {
    const { modelPath } = await writeStandaloneGguf('qwen35moe');
    await expect(detectModelType(modelPath)).resolves.toBe('qwen3_5_moe');
    const loaded = {} as Qwen35MoeModel;
    const loadSpy = vi.spyOn(Qwen35MoeModel, 'load').mockResolvedValue(loaded);
    await expect(loadModel(modelPath)).resolves.toBe(loaded);
    expect(loadSpy).toHaveBeenCalledWith(modelPath, null);
  });

  it('recognizes a Gemma4 GGUF file without guessing from its filename', async () => {
    const { modelPath } = await writeStandaloneGguf('gemma4');
    await expect(detectModelType(modelPath)).resolves.toBe('gemma4');
  });

  it('detects Muse from its GGUF header and passes the exact file to its loader', async () => {
    const { modelPath } = await writeStandaloneGguf('muse-glimmer');
    await expect(detectModelType(modelPath)).resolves.toBe('muse_glimmer');
    const loaded = {} as MuseGlimmerModel;
    const loadSpy = vi.spyOn(MuseGlimmerModel, 'load').mockResolvedValue(loaded);
    await expect(loadModel(modelPath)).resolves.toBe(loaded);
    expect(loadSpy).toHaveBeenCalledWith(modelPath);
  });

  it('keeps an existing sibling config.json authoritative', async () => {
    const { root, modelPath } = await writeStandaloneGguf('qwen35');
    await writeFile(join(root, 'config.json'), JSON.stringify({ model_type: 'qwen3' }), 'utf8');
    await expect(detectModelType(modelPath)).resolves.toBe('qwen3');
  });

  it('rejects an unsupported standalone architecture instead of guessing a family', async () => {
    const { modelPath } = await writeStandaloneGguf('llama');
    await expect(detectModelType(modelPath)).rejects.toThrow('Unsupported GGUF architecture "llama"');
  });

  it.each(['inkling2', 'inkling2_loop'] as const)(
    'detects %s GGUF, keeps auto-discovery soft, honors autoLoadDraft:false, and forwards explicit draftModelPath',
    async (arch) => {
      const { root, modelPath } = await writeStandaloneGguf(arch);
      await expect(detectModelType(modelPath)).resolves.toBe('inkling2');

      const dsparkPath = join(root, 'dspark-Inkling-2-Mini-Q8_0.gguf');
      await writeFile(
        dsparkPath,
        Buffer.concat([
          Buffer.from('GGUF'),
          u32(3),
          u64(0),
          u64(4),
          ggufString('general.architecture'),
          u32(8),
          ggufString('dflash'),
          ggufString('dflash.dspark.markov_rank'),
          u32(4),
          u32(256),
          ggufString('tokenizer.ggml.pre'),
          u32(8),
          ggufString('inkling2'),
          ggufString('dflash.target_layers'),
          u32(9),
          u32(4),
          u64(4),
          u32(5),
          u32(23),
          u32(41),
          u32(48),
        ]),
      );

      const loaded = {} as Inkling2Model;
      const loadSpy = vi.spyOn(Inkling2Model, 'load').mockResolvedValue(loaded);
      // Automatic mode delegates DSpark discovery to Rust (soft fallback on bad drafts)
      // instead of elevating the candidate to an explicit `draftModelPath`.
      await expect(loadModel(modelPath)).resolves.toBe(loaded);
      expect(loadSpy).toHaveBeenLastCalledWith(
        modelPath,
        supportsAutomaticDFlash2('inkling2') ? null : { autoLoadDraft: false },
      );

      // Explicit `draftModelPath` is forwarded verbatim for strict native validation.
      await loadModel(modelPath, { draftModelPath: dsparkPath });
      expect(loadSpy).toHaveBeenLastCalledWith(modelPath, { draftModelPath: dsparkPath });

      // `autoDraftModelPath` flows through to Inkling2Model.load without becoming explicit `draftModelPath`.
      const externalDflash2 = join(root, 'external-inkling-2-mini-dflash2');
      await loadModel(modelPath, { autoDraftModelPath: externalDflash2 });
      expect(loadSpy).toHaveBeenLastCalledWith(
        modelPath,
        supportsAutomaticDFlash2('inkling2')
          ? { autoDraftModelPath: externalDflash2 }
          : { autoDraftModelPath: externalDflash2, autoLoadDraft: false },
      );

      // Explicit `draftModelPath` stays distinct from `autoDraftModelPath` when both are supplied.
      await loadModel(modelPath, { draftModelPath: dsparkPath, autoDraftModelPath: externalDflash2 });
      expect(loadSpy).toHaveBeenLastCalledWith(modelPath, {
        draftModelPath: dsparkPath,
        autoDraftModelPath: externalDflash2,
      });

      // `autoLoadDraft: false` propagates to Rust so native discovery stays disabled.
      await loadModel(modelPath, {
        autoLoadDraft: false,
        kvFormat: 'bf16',
        maxMemoryBytes: 32_000_000_000,
        maxContextTokens: 131_072,
        decodeShare: 0.5,
        activations: 'auto',
        cacheDisk: '16G',
        cacheDir: '/tmp/inkling-cache',
      });
      expect(loadSpy).toHaveBeenLastCalledWith(modelPath, {
        autoLoadDraft: false,
        kvFormat: 'bf16',
        maxMemoryBytes: 32_000_000_000,
        maxContextTokens: 131_072,
        decodeShare: 0.5,
        activations: 'auto',
        cacheDisk: '16G',
        cacheDir: '/tmp/inkling-cache',
      });
    },
  );

  it('detects ~/.mlx-node/models/inkling-2-mini-gguf as inkling2 and delegates auto-draft cleanly', async () => {
    const modelDir = join(homedir(), '.mlx-node', 'models', 'inkling-2-mini-gguf');
    const modelPath = join(modelDir, 'Inkling-2-Mini-UD-Q4_K_XL.gguf');
    const expectedDraft = join(modelDir, 'dspark-Inkling-2-Mini-Q8_0.gguf');
    if (!existsSync(modelPath) || !existsSync(expectedDraft)) return;

    await expect(detectModelType(modelPath)).resolves.toBe('inkling2');
    const loadSpy = vi.spyOn(Inkling2Model, 'load').mockResolvedValue({} as Inkling2Model);
    await loadModel(modelPath);
    expect(loadSpy).toHaveBeenLastCalledWith(
      modelPath,
      supportsAutomaticDFlash2('inkling2') ? null : { autoLoadDraft: false },
    );
  });
});
