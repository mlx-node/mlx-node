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

beforeAll(async () => {
  modelsDir = await mkdtemp(join(tmpdir(), 'mlx-agent-models-'));

  // qwen3_5 with the REAL nesting: max_position_embeddings under text_config.
  await writeModelDir('alpha-qwen35', {
    model_type: 'qwen3_5',
    text_config: { max_position_embeddings: 32768 },
  });
  // gemma4 with a root max_position_embeddings; the bogus text_config value
  // pins the read priority (root wins).
  await writeModelDir('beta-gemma', {
    model_type: 'gemma4_text',
    max_position_embeddings: 8192,
    text_config: { max_position_embeddings: 999999 },
  });
  // No max_position_embeddings anywhere → documented family fallback.
  await writeModelDir('gamma-fallback', { model_type: 'qwen3_5' });
  // lfm2_moe is loadable via the agent-local MoE launch preset
  // (LFM2.5-8B-A1B) — it MUST be discovered, not skipped.
  await writeModelDir('lfm-moe', { model_type: 'lfm2_moe' });

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
    expect(infos.map((m) => m.discovered.name)).toEqual(['alpha-qwen35', 'beta-gemma', 'gamma-fallback', 'lfm-moe']);
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

  it('builds a pi entry with dir name as id and name, text input, and zero cost', () => {
    for (const info of infos) {
      expect(info.piModel.id).toBe(info.discovered.name);
      expect(info.piModel.name).toBe(info.discovered.name);
      expect(info.piModel.input).toEqual(['text']);
      expect(info.piModel.cost).toEqual({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0 });
    }
  });

  it('flags reasoning per family: qwen3_5 and lfm2_moe yes, gemma4 no', () => {
    const [qwen, gemma, fallback, moe] = infos;
    expect(qwen!.piModel.reasoning).toBe(true);
    expect(gemma!.piModel.reasoning).toBe(false);
    expect(fallback!.piModel.reasoning).toBe(true);
    expect(moe!.piModel.reasoning).toBe(true);
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
    expect(qwen!.piModel.maxTokens).toBe(81920); // LAUNCH_PRESETS.qwen3_5.maxOutputTokens
    expect(gemma!.piModel.maxTokens).toBe(16384); // LAUNCH_PRESETS.gemma4.maxOutputTokens
    expect(fallback!.piModel.maxTokens).toBe(81920);
  });

  it('returns an empty list for an unreadable models dir', async () => {
    expect(await discoverMlxModels(join(modelsDir, 'does-not-exist'))).toEqual([]);
  });
});
