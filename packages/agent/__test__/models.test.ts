import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import type { LoadableModel } from '@mlx-node/lm';
import { afterAll, beforeAll, describe, expect, it, vi } from 'vite-plus/test';

import { MlxModelHost } from '../src/provider/model-host.js';
import { discoverMlxModels, type MlxModelInfo } from '../src/provider/models.js';

let modelsDir: string;
let infos: MlxModelInfo[];

async function writeModelDir(name: string, config: unknown): Promise<void> {
  const dir = join(modelsDir, name);
  await mkdir(dir, { recursive: true });
  await writeFile(join(dir, 'config.json'), JSON.stringify(config));
}

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
    u32(3),
    u64(0),
    u64(1),
    ggufString('general.architecture'),
    u32(8),
    ggufString(architecture),
  ]);
}

beforeAll(async () => {
  modelsDir = await mkdtemp(join(tmpdir(), 'mlx-agent-models-'));

  // qwen3_5 with the REAL nesting: max_position_embeddings under text_config.
  await writeModelDir('alpha-qwen35', {
    model_type: 'qwen3_5',
    text_config: { max_position_embeddings: 32768 },
    vision_config: { model_type: 'qwen3_5', hidden_size: 1152 },
  });
  // gemma4 with a root max_position_embeddings; the bogus text_config value
  // pins the read priority (root wins).
  await writeModelDir('beta-gemma', {
    model_type: 'gemma4_text',
    max_position_embeddings: 8192,
    text_config: { max_position_embeddings: 999999 },
    vision_config: { model_type: 'gemma4_vision', hidden_size: 768 },
  });
  // No max_position_embeddings anywhere → documented family fallback.
  await writeModelDir('gamma-fallback', { model_type: 'qwen3_5' });
  // lfm2_moe is loadable via the agent-local MoE launch preset
  // (LFM2.5-8B-A1B) — it MUST be discovered, not skipped.
  await writeModelDir('lfm-moe', { model_type: 'lfm2_moe' });
  // A malformed marker must stay fail-closed even for a multimodal family.
  await writeModelDir('zeta-malformed-vision', {
    model_type: 'qwen3_5_moe',
    vision_config: [],
  });
  await writeModelDir('zulu-qwen35-moe-vision', {
    model_type: 'qwen3_5_moe',
    vision_config: { model_type: 'qwen3_5', hidden_size: 1152 },
  });
  await writeModelDir('zz-gemma-unified-vision', {
    model_type: 'gemma4_text',
    unified_vision_config: { model_type: 'gemma4_unified_vision', hidden_size: 768 },
  });
  // `unified_vision_config` is a Gemma-only marker; Qwen must not infer image
  // support from it.
  await writeModelDir('zzz-qwen-unified-only', {
    model_type: 'qwen3_5',
    unified_vision_config: { model_type: 'qwen3_5', hidden_size: 1152 },
  });

  // All of the below must be skipped silently:
  await mkdir(join(modelsDir, 'no-config-dir'), { recursive: true }); // no config.json
  await writeModelDir('harrier-embed', { model_type: 'harrier' }); // non-generative
  await writeFile(join(modelsDir, 'notes.txt'), 'not a model dir'); // plain file

  infos = await discoverMlxModels(modelsDir);
});

afterAll(async () => {
  await rm(modelsDir, { recursive: true, force: true });
});

describe('discoverMlxModels', () => {
  it('returns only chat-capable model dirs, sorted by name', () => {
    expect(infos.map((m) => m.discovered.name)).toEqual([
      'alpha-qwen35',
      'beta-gemma',
      'gamma-fallback',
      'lfm-moe',
      'zeta-malformed-vision',
      'zulu-qwen35-moe-vision',
      'zz-gemma-unified-vision',
      'zzz-qwen-unified-only',
    ]);
  });

  it('detects the model type and records the full path', () => {
    const [qwen, gemma] = infos;
    expect(qwen!.discovered).toEqual({
      name: 'alpha-qwen35',
      path: join(modelsDir, 'alpha-qwen35'),
      modelType: 'qwen3_5',
    });
    expect(gemma!.discovered.modelType).toBe('gemma4');
  });

  it('builds a pi entry with dir name as id and name and zero cost', () => {
    for (const info of infos) {
      expect(info.piModel.id).toBe(info.discovered.name);
      expect(info.piModel.name).toBe(info.discovered.name);
      expect(info.piModel.cost).toEqual({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0 });
    }
  });

  it('advertises images for Gemma and Qwen checkpoints with a valid vision_config', () => {
    expect(infos[0]!.piModel.input).toEqual(['text', 'image']);
    expect(infos[1]!.piModel.input).toEqual(['text', 'image']);
    expect(infos[5]!.piModel.input).toEqual(['text', 'image']);
  });

  it('accepts unified_vision_config only for Gemma', () => {
    expect(infos[6]!.piModel.input).toEqual(['text', 'image']);
    expect(infos[7]!.piModel.input).toEqual(['text']);
  });

  it('stays text-only without a valid multimodal vision_config', () => {
    expect(infos[2]!.piModel.input).toEqual(['text']);
    expect(infos[3]!.piModel.input).toEqual(['text']);
    expect(infos[4]!.piModel.input).toEqual(['text']);
  });

  it('flags reasoning for Qwen3.5, Gemma4, and LFM2 MoE', () => {
    const [qwen, gemma, fallback, moe] = infos;
    expect(qwen!.piModel.reasoning).toBe(true);
    expect(gemma!.piModel.reasoning).toBe(true);
    expect(fallback!.piModel.reasoning).toBe(true);
    expect(moe!.piModel.reasoning).toBe(true);
  });

  it('exposes only Gemma4 distinct thinking modes', () => {
    expect(infos[1]!.piModel.thinkingLevelMap).toEqual({
      minimal: 'minimal',
      low: null,
      medium: null,
      high: 'high',
    });
    expect(infos[0]!.piModel.thinkingLevelMap).toBeUndefined();
  });

  it('discovers lfm2_moe with lfm2-family traits and the first-class MoE preset', () => {
    const moe = infos[3]!;
    expect(moe.discovered).toEqual({
      name: 'lfm-moe',
      path: join(modelsDir, 'lfm-moe'),
      modelType: 'lfm2_moe',
    });
    expect(moe.piModel.reasoning).toBe(true);
    expect(moe.piModel.contextWindow).toBe(128000); // LFM2.5 family fallback window
    expect(moe.piModel.maxTokens).toBe(8192); // agent-local lfm2_moe preset maxOutputTokens
  });

  it('reads contextWindow from text_config.max_position_embeddings (qwen3_5 nesting)', () => {
    expect(infos[0]!.piModel.contextWindow).toBe(32768);
  });

  it('prefers the root max_position_embeddings over text_config', () => {
    expect(infos[1]!.piModel.contextWindow).toBe(8192);
  });

  it('falls back to the documented family default when config carries no window', () => {
    expect(infos[2]!.piModel.contextWindow).toBe(262144);
  });

  it('sources maxTokens from the family launch preset', () => {
    const [qwen, gemma, fallback] = infos;
    expect(qwen!.piModel.maxTokens).toBe(81920);
    expect(gemma!.piModel.maxTokens).toBe(16384);
    expect(fallback!.piModel.maxTokens).toBe(81920);
  });

  it('returns an empty list for an unreadable models dir', async () => {
    expect(await discoverMlxModels(join(modelsDir, 'does-not-exist'))).toEqual([]);
  });

  it('discovers Qwen3.8 variants without persisting automatic draft paths or exposing a draft model', async () => {
    const root = await mkdtemp(join(tmpdir(), 'mlx-shared-dflash-'));
    try {
      const target = join(root, 'qwen3.8-27b-mxfp4-mlx');
      const draft = join(root, 'qwen3.8-27b-dflash2');
      await mkdir(target);
      await mkdir(draft);
      await writeFile(join(target, 'config.json'), JSON.stringify({ model_type: 'qwen3_5' }));
      await writeFile(
        join(draft, 'config.json'),
        JSON.stringify({ model_type: 'qwen3', architectures: ['DFlash2DraftModel'] }),
      );
      await writeFile(join(draft, 'model.safetensors'), 'weights');
      await writeFile(join(root, 'Qwen3.8-27B-UD-Q4_K_XL.gguf'), minimalGguf('qwen35'));
      const models = await discoverMlxModels(root);
      expect(models).toHaveLength(2);
      expect(models.every((model) => model.discovered.draftModelPath === undefined)).toBe(true);
      expect(models.some((model) => model.discovered.path === draft)).toBe(false);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it.each(['safetensors', 'top-level GGUF', 'nested GGUF', 'embedded draft'] as const)(
    'rechecks the %s companion after discovery and on each model swap',
    async (layout) => {
      const root = await mkdtemp(join(tmpdir(), 'mlx-agent-draft-lifecycle-'));
      const repo = join(root, 'qwen3.8-27b');
      const filename = 'Qwen3.8-27B-UD-Q4_K_XL.gguf';
      const target = layout === 'safetensors' ? repo : join(layout === 'top-level GGUF' ? root : repo, filename);
      const draft = layout === 'embedded draft' ? join(repo, 'draft') : join(root, 'qwen3.8-27b-dflash2');
      try {
        if (layout !== 'top-level GGUF') {
          await mkdir(repo);
          await writeFile(join(repo, 'config.json'), JSON.stringify({ model_type: 'qwen3_5' }));
        }
        if (layout === 'safetensors') {
          await writeFile(join(repo, 'model.safetensors'), 'target weights');
        } else {
          await writeFile(target, minimalGguf('qwen35'));
        }
        await mkdir(draft);
        await writeFile(
          join(draft, 'config.json'),
          JSON.stringify({ model_type: 'qwen3', architectures: ['DFlash2DraftModel'] }),
        );
        await writeFile(join(draft, 'model.safetensors'), 'draft weights');
        // Use the real startup discovery record: hand-built records would miss
        // the stale automatic path that previously became authoritative here.
        const models = await discoverMlxModels(root);
        expect(models).toHaveLength(1);
        const model = models[0]!.discovered;
        const loader = vi.fn(async (path: string) => ({ fakeModelFor: path }) as unknown as LoadableModel);
        const host = new MlxModelHost(
          [...models.map((info) => info.discovered), { name: 'other', path: '/models/other', modelType: 'qwen3' }],
          { loadModelFn: loader, resolveModelPathFn: async (entry) => `/paged/${entry.name}` },
        );
        const loadTarget = () => host.runWithResident(model.name, async () => undefined);
        const swapAway = () => host.runWithResident('other', async () => undefined);

        // A partial companion before the first load must not prevent target use.
        await rm(join(draft, 'model.safetensors'));
        await loadTarget();
        expect(loader).toHaveBeenLastCalledWith(`/paged/${model.name}`);

        // Restoring it is picked up from the original target, not the overlay.
        await writeFile(join(draft, 'model.safetensors'), 'draft weights');
        await swapAway();
        await loadTarget();
        expect(loader).toHaveBeenLastCalledWith(`/paged/${model.name}`, { draftModelPath: draft });

        // Deleting a previously loaded companion must not poison a later swap.
        await swapAway();
        await rm(draft, { recursive: true });
        await loadTarget();
        expect(loader).toHaveBeenLastCalledWith(`/paged/${model.name}`);
      } finally {
        await rm(root, { recursive: true, force: true });
      }
    },
  );

  it('discovers every nested Q<number>_K_XL target by its direct GGUF path', async () => {
    const root = await mkdtemp(join(tmpdir(), 'mlx-agent-xl-gguf-'));
    try {
      const repo = join(root, 'qwen38-gguf');
      await mkdir(repo, { recursive: true });
      await writeFile(
        join(repo, 'config.json'),
        JSON.stringify({ model_type: 'qwen3_5', text_config: { max_position_embeddings: 65536 } }),
      );
      const draft = join(repo, 'draft');
      await mkdir(draft, { recursive: true });
      await writeFile(
        join(draft, 'config.json'),
        JSON.stringify({
          model_type: 'qwen3',
          architectures: ['DFlash2DraftModel'],
          dflash_config: { block_size: 8 },
        }),
      );
      await writeFile(join(draft, 'model.safetensors'), 'draft weights');
      await Promise.all([
        writeFile(join(repo, 'Qwen3.8-27B-UD-Q3_K_XL.gguf'), 'q3'),
        writeFile(join(repo, 'Qwen3.8-27B-UD-Q4_K_XL.gguf'), 'q4'),
        writeFile(join(repo, 'Qwen3.8-27B-Q4_K_M.gguf'), 'ordinary variant'),
        writeFile(join(repo, 'imatrix_unsloth.gguf'), 'imatrix'),
        writeFile(join(repo, 'mmproj-Q4_K_XL.gguf'), 'mmproj'),
        writeFile(join(repo, 'dflash-Q4_K_XL.gguf'), 'draft'),
      ]);

      const discovered = await discoverMlxModels(root);
      expect(discovered.map((model) => model.discovered)).toEqual([
        {
          name: 'Qwen3.8-27B-UD-Q3_K_XL',
          path: join(repo, 'Qwen3.8-27B-UD-Q3_K_XL.gguf'),
          modelType: 'qwen3_5',
        },
        {
          name: 'Qwen3.8-27B-UD-Q4_K_XL',
          path: join(repo, 'Qwen3.8-27B-UD-Q4_K_XL.gguf'),
          modelType: 'qwen3_5',
        },
      ]);
      expect(discovered.map((model) => model.piModel.contextWindow)).toEqual([65536, 65536]);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it('discovers Gemma4 QAT and K-quant targets while excluding their media projectors', async () => {
    const root = await mkdtemp(join(tmpdir(), 'mlx-agent-gemma-gguf-'));
    try {
      const repo = join(root, 'gemma4-gguf');
      await mkdir(repo);
      await writeFile(
        join(repo, 'config.json'),
        JSON.stringify({
          model_type: 'gemma4_unified',
          text_config: { max_position_embeddings: 262144 },
          vision_config: { model_type: 'gemma4_unified_vision' },
        }),
      );
      for (const name of ['gemma-4-12b-it-qat-q4_0.gguf', 'gemma-4-12b-Q6_K.gguf']) {
        await writeFile(join(repo, name), minimalGguf('gemma4'));
      }
      await writeFile(join(repo, 'mmproj-gemma-4-12b-it-qat-q4_0.gguf'), minimalGguf('clip'));
      await writeFile(join(root, 'gemma-4-Q4_K_M.gguf'), minimalGguf('gemma4'));
      await writeFile(join(root, 'config.json'), JSON.stringify({ model_type: 'gemma4' }));
      await writeFile(join(root, 'tokenizer.json'), '{}');
      await writeFile(join(repo, 'tokenizer.json'), '{}');
      const found = await discoverMlxModels(root);
      expect(found.map((entry) => entry.discovered.name)).toEqual([
        'gemma-4-12b-Q6_K',
        'gemma-4-12b-it-qat-q4_0',
        'gemma-4-Q4_K_M',
      ]);
      expect(found.every((entry) => entry.discovered.modelType === 'gemma4')).toBe(true);
      expect(found[0].piModel.input).toEqual(['text', 'image']);
      expect(found[0].piModel.contextWindow).toBe(262144);

      // A converted checkpoint remains selectable even when its source XL
      // GGUF is retained beside the SafeTensors weights.
      await writeFile(join(repo, 'model.safetensors'), '');
      await writeFile(join(repo, 'gemma-4-Q4_K_XL.gguf'), minimalGguf('gemma4'));
      const converted = await discoverMlxModels(root);
      expect(converted.map((entry) => entry.discovered.name)).toEqual(['gemma-4-Q4_K_M', 'gemma4-gguf']);
      expect(converted[1].discovered.path).toBe(repo);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it('lists all Muse GGUF variants by filename and ignores renamed companions and broken files', async () => {
    const root = await mkdtemp(join(tmpdir(), 'mlx-agent-muse-gguf-'));
    try {
      const repo = join(root, 'muse-glimmer-30b-gguf');
      await mkdir(repo);
      await writeFile(
        join(repo, 'config.json'),
        JSON.stringify({ model_type: 'muse_glimmer', vision_config: { hidden_size: 1536 } }),
      );
      await writeFile(join(repo, 'tokenizer.json'), '{}');
      const names = [
        'Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf',
        'Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf',
        'muse-glimmer-30B-kquant-17gb.gguf',
        'muse-glimmer-30B-kquant-dynamic.gguf',
      ];
      for (const name of names) await writeFile(join(repo, name), minimalGguf('muse-glimmer'));
      for (const [name, arch] of [
        ['dflash-kquant.gguf', 'dflash'],
        ['mmproj-kquant.gguf', 'clip'],
        ['renamed-companion.gguf', 'dflash'],
        ['renamed-projector.gguf', 'clip'],
        ['unrelated-Q4_K_XL.gguf', 'llama'],
      ])
        await writeFile(join(repo, name), minimalGguf(arch));
      await writeFile(join(repo, 'broken.gguf'), 'incomplete download');
      const found = await discoverMlxModels(root);
      expect(found.map((entry) => entry.discovered)).toEqual(
        names.map((name) => ({
          name: name.slice(0, -5),
          path: join(repo, name),
          modelType: 'muse_glimmer',
        })),
      );
      expect(found.every((entry) => entry.piModel.input.join() === 'text')).toBe(true);

      // Retaining the downloaded XL alongside an imported model must not
      // hide the usable SafeTensors directory or switch it to another variant.
      await writeFile(join(repo, 'model.safetensors'), 'weights');
      expect((await discoverMlxModels(root)).map((entry) => entry.discovered.path)).toEqual([repo]);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it.each(
    ['gemma4', 'muse_glimmer'].flatMap((family) => ['top-level', 'nested'].map((layout) => ({ family, layout }))),
  )('requires sibling config and tokenizer files for $layout $family GGUFs', async ({ family, layout }) => {
    const root = await mkdtemp(join(tmpdir(), 'mlx-agent-gemma-assets-'));
    try {
      const repo = layout === 'top-level' ? root : join(root, 'gemma-gguf');
      await mkdir(repo, { recursive: true });
      const gguf = join(repo, 'gemma-4-Q4_K_M.gguf');
      await writeFile(gguf, minimalGguf(family === 'muse_glimmer' ? 'muse-glimmer' : 'gemma4'));
      const configPath = join(repo, 'config.json');
      const tokenizerPath = join(repo, 'tokenizer.json');
      const config = JSON.stringify({ model_type: family });

      expect(await discoverMlxModels(root)).toEqual([]);
      await writeFile(configPath, config);
      expect(await discoverMlxModels(root)).toEqual([]);
      await rm(configPath);
      await writeFile(tokenizerPath, '{}');
      expect(await discoverMlxModels(root)).toEqual([]);
      await writeFile(configPath, config);
      expect((await discoverMlxModels(root)).map((entry) => entry.discovered.path)).toEqual([gguf]);

      if (family === 'muse_glimmer') {
        await writeFile(join(repo, 'renamed-companion.gguf'), minimalGguf('dflash'));
        expect((await discoverMlxModels(root)).map((entry) => entry.discovered.path)).toEqual([gguf]);
      }

      // A directory with the required name is not a usable tokenizer file.
      await rm(tokenizerPath);
      await mkdir(tokenizerPath);
      expect(await discoverMlxModels(root)).toEqual([]);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it('discovers a standalone top-level XL GGUF from its qwen35 header', async () => {
    const root = await mkdtemp(join(tmpdir(), 'mlx-agent-standalone-gguf-'));
    try {
      const gguf = join(root, 'Qwen3.8-27B-UD-Q6_K_XL.gguf');
      await writeFile(gguf, minimalGguf('qwen35'));

      const [discovered] = await discoverMlxModels(root);
      expect(discovered?.discovered).toEqual({
        name: 'Qwen3.8-27B-UD-Q6_K_XL',
        path: gguf,
        modelType: 'qwen3_5',
      });
      expect(discovered?.piModel.contextWindow).toBe(262144);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it('does not attach a companion outside the store when loading a discovered top-level GGUF', async () => {
    const root = await mkdtemp(join(tmpdir(), 'mlx-agent-companion-boundary-'));
    try {
      const store = join(root, 'models');
      await mkdir(store);
      const gguf = join(store, 'Qwen3.8-27B-UD-Q4_K_XL.gguf');
      await writeFile(gguf, minimalGguf('qwen35'));
      const outside = join(root, 'qwen3.8-27b-dflash2');
      await mkdir(outside);
      await writeFile(join(outside, 'config.json'), JSON.stringify({ architectures: ['DFlash2DraftModel'] }));
      await writeFile(join(outside, 'model.safetensors'), 'draft weights');
      const models = await discoverMlxModels(store);
      expect(models).toHaveLength(1);
      const loader = vi.fn(async () => ({}) as LoadableModel);
      const host = new MlxModelHost(
        models.map((model) => model.discovered),
        { loadModelFn: loader },
      );
      await host.runWithResident(models[0]!.discovered.name, async () => undefined);
      expect(loader).toHaveBeenCalledExactlyOnceWith(gguf);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it('does not advertise DFlash2 companions or XL files from unsupported direct-GGUF families', async () => {
    const root = await mkdtemp(join(tmpdir(), 'mlx-agent-nontarget-gguf-'));
    try {
      const draft = join(root, 'qwen38-dflash2');
      await mkdir(draft, { recursive: true });
      await writeFile(
        join(draft, 'config.json'),
        JSON.stringify({ model_type: 'qwen3', architectures: ['DFlash2DraftModel'] }),
      );

      const qwen3 = join(root, 'qwen3-gguf');
      await mkdir(qwen3, { recursive: true });
      await writeFile(join(qwen3, 'config.json'), JSON.stringify({ model_type: 'qwen3' }));
      await writeFile(join(qwen3, 'Qwen3-8B-UD-Q5_K_XL.gguf'), 'unsupported direct qwen3');

      const ordinary = join(root, 'qwen38-q4km-gguf');
      await mkdir(ordinary, { recursive: true });
      await writeFile(join(ordinary, 'config.json'), JSON.stringify({ model_type: 'qwen3_5' }));
      await writeFile(join(ordinary, 'Qwen3.8-27B-Q4_K_M.gguf'), 'unsupported direct variant');

      expect(await discoverMlxModels(root)).toEqual([]);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it('keeps a converted SafeTensors model discoverable when it retains an imatrix GGUF', async () => {
    const root = await mkdtemp(join(tmpdir(), 'mlx-agent-converted-imatrix-'));
    try {
      const converted = join(root, 'qwen38-mlx');
      await mkdir(converted, { recursive: true });
      await writeFile(join(converted, 'config.json'), JSON.stringify({ model_type: 'qwen3_5' }));
      await writeFile(join(converted, 'model.safetensors'), 'weights');
      await writeFile(join(converted, 'imatrix_unsloth.gguf'), 'calibration');

      expect((await discoverMlxModels(root)).map((model) => model.discovered.name)).toEqual(['qwen38-mlx']);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it('assigns colliding XL model IDs in sorted directory order', async () => {
    const root = await mkdtemp(join(tmpdir(), 'mlx-agent-colliding-xl-'));
    try {
      // Create zeta first so filesystem insertion order disagrees with the
      // stable lexical order discovery must use for persisted model IDs.
      const zeta = join(root, 'zeta-repo');
      const alpha = join(root, 'alpha-repo');
      for (const repo of [zeta, alpha]) {
        await mkdir(repo, { recursive: true });
        await writeFile(join(repo, 'config.json'), JSON.stringify({ model_type: 'qwen3_5' }));
        await writeFile(join(repo, 'Shared-Qwen3.8-UD-Q4_K_XL.gguf'), 'target');
      }

      expect((await discoverMlxModels(root)).map((model) => model.discovered)).toEqual([
        {
          name: 'Shared-Qwen3.8-UD-Q4_K_XL',
          path: join(alpha, 'Shared-Qwen3.8-UD-Q4_K_XL.gguf'),
          modelType: 'qwen3_5',
        },
        {
          name: 'zeta-repo-Shared-Qwen3.8-UD-Q4_K_XL',
          path: join(zeta, 'Shared-Qwen3.8-UD-Q4_K_XL.gguf'),
          modelType: 'qwen3_5',
        },
      ]);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });
});
