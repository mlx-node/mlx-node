import { mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { detectModelType } from '@mlx-node/lm';
import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

/**
 * `detectModelType` must normalize the Gemma 4 text-family `model_type`
 * variants to the single `gemma4` registry key. The 31B/E2B/26B checkpoints
 * report `gemma4_text`; the unified 12B checkpoint reports `gemma4_unified`
 * (with `architectures: ["Gemma4UnifiedForConditionalGeneration"]`). Both
 * resolve to the one shared `gemma4` Rust decoder + `Gemma4Model` registry
 * row — there is no separate unified registry entry.
 */
describe.sequential('Gemma4 unified model detection', () => {
  let tempDir: string;

  beforeEach(async () => {
    tempDir = join(tmpdir(), `gemma4-unified-test-${Date.now()}`);
    await mkdir(tempDir, { recursive: true });
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  it('normalizes gemma4_text to gemma4', async () => {
    await writeFile(
      join(tempDir, 'config.json'),
      JSON.stringify({
        model_type: 'gemma4_text',
        text_config: { hidden_size: 3840 },
      }),
    );

    const modelType = await detectModelType(tempDir);
    expect(modelType).toBe('gemma4');
  });

  it('normalizes gemma4_unified to gemma4', async () => {
    await writeFile(
      join(tempDir, 'config.json'),
      JSON.stringify({
        model_type: 'gemma4_unified',
        architectures: ['Gemma4UnifiedForConditionalGeneration'],
        tie_word_embeddings: true,
        text_config: { model_type: 'gemma4_unified_text', hidden_size: 3840 },
        vision_config: { model_type: 'gemma4_unified_vision' },
        audio_config: { model_type: 'gemma4_unified_audio' },
      }),
    );

    const modelType = await detectModelType(tempDir);
    expect(modelType).toBe('gemma4');
  });

  it('routes architecture-only unified checkpoints (no model_type) to gemma4', async () => {
    // Detection first selects the declarative nullish-model_type => qwen3 base,
    // then the recognized architecture refines it to `gemma4`. This mirrors
    // gemma4/persistence.rs is_unified in the native loader.
    await writeFile(
      join(tempDir, 'config.json'),
      JSON.stringify({
        architectures: ['Gemma4UnifiedForConditionalGeneration'],
        tie_word_embeddings: true,
        text_config: { model_type: 'gemma4_unified_text', hidden_size: 3840 },
        vision_config: { model_type: 'gemma4_unified_vision' },
        audio_config: { model_type: 'gemma4_unified_audio' },
      }),
    );

    const modelType = await detectModelType(tempDir);
    expect(modelType).toBe('gemma4');
  });

  it('passes through plain gemma4 unchanged', async () => {
    await writeFile(
      join(tempDir, 'config.json'),
      JSON.stringify({
        model_type: 'gemma4',
        text_config: { hidden_size: 3840 },
      }),
    );

    const modelType = await detectModelType(tempDir);
    expect(modelType).toBe('gemma4');
  });
});
