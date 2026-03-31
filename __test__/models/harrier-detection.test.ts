import { mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { detectModelType } from '@mlx-node/lm';
import { describe, it, expect, beforeEach, afterEach } from 'vite-plus/test';

describe.sequential('Harrier Model Detection', () => {
  let tempDir: string;

  beforeEach(async () => {
    tempDir = join(tmpdir(), `harrier-test-${Date.now()}`);
    await mkdir(tempDir, { recursive: true });
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  it('should detect Qwen3Model architecture as harrier', async () => {
    await writeFile(
      join(tempDir, 'config.json'),
      JSON.stringify({
        model_type: 'qwen3',
        architectures: ['Qwen3Model'],
        hidden_size: 1024,
      }),
    );

    const modelType = await detectModelType(tempDir);
    expect(modelType).toBe('harrier');
  });

  it('should detect Qwen3ForCausalLM as qwen3', async () => {
    await writeFile(
      join(tempDir, 'config.json'),
      JSON.stringify({
        model_type: 'qwen3',
        architectures: ['Qwen3ForCausalLM'],
        hidden_size: 1024,
      }),
    );

    const modelType = await detectModelType(tempDir);
    expect(modelType).toBe('qwen3');
  });

  it('should default to qwen3 when architectures is absent', async () => {
    await writeFile(
      join(tempDir, 'config.json'),
      JSON.stringify({
        model_type: 'qwen3',
        hidden_size: 1024,
      }),
    );

    const modelType = await detectModelType(tempDir);
    expect(modelType).toBe('qwen3');
  });

  it('should detect qwen3_5 model type directly', async () => {
    await writeFile(
      join(tempDir, 'config.json'),
      JSON.stringify({
        model_type: 'qwen3_5',
        hidden_size: 4096,
      }),
    );

    const modelType = await detectModelType(tempDir);
    expect(modelType).toBe('qwen3_5');
  });

  it('should throw on unsupported model_type', async () => {
    await writeFile(
      join(tempDir, 'config.json'),
      JSON.stringify({
        model_type: 'llama',
        hidden_size: 4096,
      }),
    );

    await expect(detectModelType(tempDir)).rejects.toThrow('Unsupported model_type');
  });
});
