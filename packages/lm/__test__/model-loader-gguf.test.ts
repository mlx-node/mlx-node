import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { detectModelType } from '@mlx-node/lm';
import { afterEach, describe, expect, it } from 'vite-plus/test';

const cleanups: Array<() => Promise<void>> = [];

afterEach(async () => {
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
  it('maps the qwen35 header to qwen3_5 without config.json', async () => {
    const { modelPath } = await writeStandaloneGguf('qwen35');
    await expect(detectModelType(modelPath)).resolves.toBe('qwen3_5');
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
});
