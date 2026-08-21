import { mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { detectModelType } from '@mlx-node/lm';
import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

/**
 * `detectModelType` must route Nemotron 3.5 Lightning ("nemotron_h")
 * checkpoints to the single `nemotron_h` registry key, both via the raw
 * `model_type` and via the `NemotronHForCausalLM` architecture probe
 * (mirroring the native loader's config validation in
 * crates/mlx-core/src/models/nemotron_h/config.rs).
 */
describe.sequential('Nemotron H model detection', () => {
  let tempDir: string;

  beforeEach(async () => {
    tempDir = join(tmpdir(), `nemotron-h-test-${Date.now()}`);
    await mkdir(tempDir, { recursive: true });
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  it('detects the nemotron_h model_type', async () => {
    await writeFile(
      join(tempDir, 'config.json'),
      JSON.stringify({
        model_type: 'nemotron_h',
        architectures: ['NemotronHForCausalLM'],
        hidden_size: 2048,
      }),
    );

    const modelType = await detectModelType(tempDir);
    expect(modelType).toBe('nemotron_h');
  });

  it('routes architecture-only checkpoints (no model_type) to nemotron_h', async () => {
    await writeFile(
      join(tempDir, 'config.json'),
      JSON.stringify({
        architectures: ['NemotronHForCausalLM'],
      }),
    );

    const modelType = await detectModelType(tempDir);
    expect(modelType).toBe('nemotron_h');
  });

  it('gives the NemotronHForCausalLM architecture probe precedence over an unknown model_type', async () => {
    await writeFile(
      join(tempDir, 'config.json'),
      JSON.stringify({
        model_type: 'llama',
        architectures: ['NemotronHForCausalLM'],
      }),
    );

    const modelType = await detectModelType(tempDir);
    expect(modelType).toBe('nemotron_h');
  });

  it('fails closed for an unknown model_type without the NemotronH architecture', async () => {
    await writeFile(
      join(tempDir, 'config.json'),
      JSON.stringify({
        model_type: 'llama',
        architectures: ['LlamaForCausalLM'],
      }),
    );

    await expect(detectModelType(tempDir)).rejects.toThrow(`Unsupported model_type "llama" in ${tempDir}/config.json`);
  });
});
