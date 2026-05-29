import { mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { detectModelType } from '@mlx-node/lm';
import { describe, it, expect, beforeEach, afterEach } from 'vite-plus/test';

describe.sequential('LFM2 MoE Model Detection', () => {
  let tempDir: string;

  beforeEach(async () => {
    tempDir = join(tmpdir(), `lfm2-moe-test-${Date.now()}`);
    await mkdir(tempDir, { recursive: true });
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  it('should detect lfm2_moe model_type', async () => {
    await writeFile(
      join(tempDir, 'config.json'),
      JSON.stringify({
        model_type: 'lfm2_moe',
        hidden_size: 2048,
      }),
    );

    const modelType = await detectModelType(tempDir);
    expect(modelType).toBe('lfm2_moe');
  });

  it('lfm2_moe is in SUPPORTED_MODEL_TYPES (detectModelType does not throw)', async () => {
    await writeFile(
      join(tempDir, 'config.json'),
      JSON.stringify({
        model_type: 'lfm2_moe',
        hidden_size: 2048,
      }),
    );

    // If lfm2_moe were absent from SUPPORTED_MODEL_TYPES, detectModelType would
    // throw 'Unsupported model_type "lfm2_moe"'. This test verifies it does not.
    await expect(detectModelType(tempDir)).resolves.toBe('lfm2_moe');
  });
});
