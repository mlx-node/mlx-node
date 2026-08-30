import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterAll, beforeAll, describe, expect, it } from 'vite-plus/test';

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
          draftModelPath: draft,
        },
        {
          name: 'Qwen3.8-27B-UD-Q4_K_XL',
          path: join(repo, 'Qwen3.8-27B-UD-Q4_K_XL.gguf'),
          modelType: 'qwen3_5',
          draftModelPath: draft,
        },
      ]);
      expect(discovered.map((model) => model.piModel.contextWindow)).toEqual([65536, 65536]);
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
