import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { detectModelType, loadModel, Qwen35Model } from '@mlx-node/lm';
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
    expect(loadSpy).toHaveBeenLastCalledWith(modelPath, { draftModelPath: inside });
    await rm(inside, { recursive: true });
    await loadModel(modelPath);
    expect(loadSpy).toHaveBeenLastCalledWith(modelPath, null);
  });

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
