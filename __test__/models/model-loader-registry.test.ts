import { mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { detectModelType, Gemma4Model, loadModel, loadSession, type ModelType } from '@mlx-node/lm';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

const RAW_MODEL_TYPE_ALIASES = {
  harrier: ['harrier'],
  gemma4: ['gemma4', 'gemma4_text', 'gemma4_unified'],
  qwen3: ['qwen3'],
  qwen3_5: ['qwen3_5'],
  qwen3_5_moe: ['qwen3_5_moe'],
  lfm2: ['lfm2'],
  lfm2_moe: ['lfm2_moe'],
  internvl_chat: ['internvl_chat'],
  'qianfan-ocr': ['qianfan-ocr'],
} as const satisfies Record<ModelType, readonly string[]>;

const UNIFIED_GEMMA_ARCHITECTURE = 'Gemma4UnifiedForConditionalGeneration';

const UNIFIED_GEMMA_COMPATIBILITY_CASES = [
  ['missing model_type', { architectures: [UNIFIED_GEMMA_ARCHITECTURE] }],
  ['null model_type', { model_type: null, architectures: [UNIFIED_GEMMA_ARCHITECTURE] }],
  ['explicit qwen3', { model_type: 'qwen3', architectures: [UNIFIED_GEMMA_ARCHITECTURE] }],
  ['other supported type', { model_type: 'lfm2', architectures: [UNIFIED_GEMMA_ARCHITECTURE] }],
  ['unknown type', { model_type: 'llama', architectures: [UNIFIED_GEMMA_ARCHITECTURE] }],
  ['non-string type', { model_type: 42, architectures: [UNIFIED_GEMMA_ARCHITECTURE] }],
  ['exact string architecture', { architectures: UNIFIED_GEMMA_ARCHITECTURE }],
  ['competing Harrier probe', { model_type: 'qwen3', architectures: ['Qwen3Model', UNIFIED_GEMMA_ARCHITECTURE] }],
] as const satisfies readonly (readonly [string, Readonly<Record<string, unknown>>])[];

describe.sequential('declarative model loader registry', () => {
  let tempDir: string;

  beforeEach(async () => {
    tempDir = join(tmpdir(), `model-loader-registry-${Date.now()}`);
    await mkdir(tempDir, { recursive: true });
  });

  afterEach(async () => {
    vi.restoreAllMocks();
    await rm(tempDir, { recursive: true, force: true });
  });

  const writeConfig = async (config: Record<string, unknown>): Promise<void> => {
    await writeFile(join(tempDir, 'config.json'), JSON.stringify(config));
  };

  it('keeps raw model_type aliases unique and detects aliases for every public ModelType', async () => {
    const entries = Object.entries(RAW_MODEL_TYPE_ALIASES) as [ModelType, readonly string[]][];
    const aliases = entries.flatMap(([, rawModelTypes]) => rawModelTypes);
    expect(new Set(aliases).size).toBe(aliases.length);

    for (const [expectedModelType, rawModelTypes] of entries) {
      for (const rawModelType of rawModelTypes) {
        await writeConfig({ model_type: rawModelType });
        await expect(detectModelType(tempDir)).resolves.toBe(expectedModelType);
      }
    }
  });

  it.each(UNIFIED_GEMMA_COMPATIBILITY_CASES)(
    'gives the unified Gemma architecture precedence for %s',
    async (_label, config) => {
      await writeConfig({ ...config });
      await expect(detectModelType(tempDir)).resolves.toBe('gemma4');
    },
  );

  it('gives the Harrier architecture probe precedence over the qwen3 alias', async () => {
    await writeConfig({ model_type: 'qwen3', architectures: ['Qwen3Model'] });
    await expect(detectModelType(tempDir)).resolves.toBe('harrier');

    await writeConfig({ model_type: 'qwen3', architectures: ['Qwen3Model', 'Qwen3ForCausalLM'] });
    await expect(detectModelType(tempDir)).resolves.toBe('qwen3');

    await writeConfig({ architectures: 'Qwen3Model' });
    await expect(detectModelType(tempDir)).resolves.toBe('harrier');
  });

  it('fails closed for an unknown model_type', async () => {
    await writeConfig({ model_type: 'llama' });
    await expect(detectModelType(tempDir)).rejects.toThrow(`Unsupported model_type "llama" in ${tempDir}/config.json`);
  });

  it('uses the declarative qwen3 default when model_type and a recognized architecture are absent', async () => {
    await writeConfig({});
    await expect(detectModelType(tempDir)).resolves.toBe('qwen3');

    await writeConfig({ model_type: null });
    await expect(detectModelType(tempDir)).resolves.toBe('qwen3');

    await writeConfig({ architectures: ['LlamaForCausalLM'] });
    await expect(detectModelType(tempDir)).resolves.toBe('qwen3');

    await writeConfig({ architectures: `not-${UNIFIED_GEMMA_ARCHITECTURE}` });
    await expect(detectModelType(tempDir)).resolves.toBe('qwen3');

    await writeConfig({ architectures: 'not-Qwen3Model' });
    await expect(detectModelType(tempDir)).resolves.toBe('qwen3');
  });

  it('applies Harrier architecture refinement to missing and null model_type defaults', async () => {
    await writeConfig({ architectures: ['Qwen3Model'] });
    await expect(detectModelType(tempDir)).resolves.toBe('harrier');

    await writeConfig({ model_type: null, architectures: ['Qwen3Model'] });
    await expect(detectModelType(tempDir)).resolves.toBe('harrier');
  });

  it('fails closed for an explicit non-string model_type without the authoritative Gemma architecture', async () => {
    await writeConfig({ model_type: 42, architectures: ['Qwen3Model'] });
    await expect(detectModelType(tempDir)).rejects.toThrow(`Unsupported model_type "42" in ${tempDir}/config.json`);
  });

  it.each([
    ['null root', 'null'],
    ['numeric root', '42'],
  ] as const)('fails closed when the config.json root is not an object (%s)', async (_label, rawJson) => {
    await writeFile(join(tempDir, 'config.json'), rawJson);
    await expect(detectModelType(tempDir)).rejects.toThrow(
      `Malformed config.json in ${tempDir}: root must be a JSON object`,
    );
  });

  it.each([
    ['numeric architectures', { architectures: 42 }],
    ['object architectures', { architectures: {} }],
  ] as const satisfies readonly (readonly [string, Readonly<Record<string, unknown>>])[])(
    'fails closed for %s instead of coercing to the qwen3 default',
    async (_label, config) => {
      await writeConfig({ ...config });
      await expect(detectModelType(tempDir)).rejects.toThrow(
        `Malformed config.json in ${tempDir}: "architectures" must be an array or a string`,
      );
    },
  );

  it('keeps the blessed null architectures leniency after malformed-config validation', async () => {
    await writeConfig({ model_type: 'lfm2', architectures: null });
    await expect(detectModelType(tempDir)).resolves.toBe('lfm2');
  });

  it.each(['internvl_chat', 'qianfan-ocr'] as const)(
    'preserves the %s loadSession rejection before loading native weights',
    async (modelType) => {
      await writeConfig({ model_type: modelType });
      await expect(loadSession(tempDir)).rejects.toThrow(
        'loadSession: Qianfan-OCR / InternVL session support lives in @mlx-node/vlm. Import QianfanOCRModel from @mlx-node/vlm and construct ChatSession(model) directly.',
      );
    },
  );

  it('preserves the Harrier loadSession rejection before loading native weights', async () => {
    await writeConfig({ model_type: 'harrier' });
    await expect(loadSession(tempDir)).rejects.toThrow(
      'loadSession: embedding models (Harrier) cannot be wrapped in a ChatSession',
    );
  });

  it('preserves Gemma-only draft option validation', async () => {
    await writeConfig({ model_type: 'qwen3' });
    await expect(loadModel(tempDir, { draftModelPath: '/tmp/draft' })).rejects.toThrow(
      `draftModelPath (speculative-decoding draft) is only supported by gemma4 models; ${tempDir} has model_type "qwen3"`,
    );
  });

  it('forwards draftModelPath through the Gemma family descriptor', async () => {
    await writeConfig({ model_type: 'gemma4_text' });
    const loadedModel = { model: 'gemma4' };
    const loadSpy = vi.spyOn(Gemma4Model, 'load').mockResolvedValue(loadedModel as never);

    await expect(loadModel(tempDir, { draftModelPath: '/tmp/draft' })).resolves.toBe(loadedModel);
    expect(loadSpy).toHaveBeenCalledOnce();
    expect(loadSpy).toHaveBeenCalledWith(tempDir, { draftModelPath: '/tmp/draft' });
  });
});
