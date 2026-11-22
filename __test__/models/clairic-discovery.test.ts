import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, describe, expect, it } from 'vite-plus/test';

import { detectModelType } from '../../packages/lm/src/model-detection.js';
import { discoverLocalChatModels } from '../../packages/lm/src/model-discovery.js';

const dirs: string[] = [];

afterEach(async () => {
  await Promise.all(dirs.splice(0).map((d) => rm(d, { recursive: true, force: true })));
});

async function makeQwen35Dir(root: string, name: string, config: Record<string, unknown>): Promise<string> {
  const dir = join(root, name);
  await mkdir(dir, { recursive: true });
  await writeFile(join(dir, 'config.json'), JSON.stringify(config));
  await writeFile(join(dir, 'model.safetensors'), 'fixture');
  return dir;
}

describe('Clairic model detection and discovery', () => {
  it('recognizes complete Clairic checkpoints, keeps them out of chat discovery, and exposes vision + 65536 context cap', async () => {
    const root = await mkdtemp(join(tmpdir(), 'clairic-detection-'));
    dirs.push(root);
    const dir = await makeQwen35Dir(root, 'clairic-flash', {
      model_type: 'qwen3_5',
      text_config: { hidden_size: 4096, max_position_embeddings: 262144 },
      vision_config: { hidden_size: 1152 },
    });

    expect(await detectModelType(dir)).toBe('qwen3_5');

    await writeFile(join(dir, 'clair_head_config.json'), '{}');
    await writeFile(join(dir, 'clair_head.safetensors'), 'fixture');
    expect(await detectModelType(dir)).toBe('clairic');

    // Excluded from default chat discovery
    expect(await discoverLocalChatModels(root)).toEqual([]);

    // Included with includeDecisions: true, preserving vision_config and capping context at 65536
    expect(await discoverLocalChatModels(root, { includeDecisions: true })).toMatchObject([
      {
        name: 'clairic-flash',
        modelType: 'clairic',
        supportsImages: true,
        contextWindow: 65536,
      },
    ]);
  });

  it('respects smaller trained context windows and absent vision_config', async () => {
    const root = await mkdtemp(join(tmpdir(), 'clairic-small-ctx-'));
    dirs.push(root);
    const dir = await makeQwen35Dir(root, 'clairic-text-only', {
      model_type: 'qwen3_5',
      text_config: { hidden_size: 4096, max_position_embeddings: 32768 },
    });
    await writeFile(join(dir, 'clair_head_config.json'), '{}');
    await writeFile(join(dir, 'clair_head.safetensors'), 'fixture');

    expect(await discoverLocalChatModels(root, { includeDecisions: true })).toMatchObject([
      {
        name: 'clairic-text-only',
        modelType: 'clairic',
        supportsImages: false,
        contextWindow: 32768,
      },
    ]);
  });

  it('fails closed on incomplete Clairic head assets or non-regular files', async () => {
    const root = await mkdtemp(join(tmpdir(), 'clairic-incomplete-'));
    dirs.push(root);
    const dir = await makeQwen35Dir(root, 'broken', {
      model_type: 'qwen3_5',
      text_config: { hidden_size: 4096 },
    });

    // Directory instead of regular file
    await mkdir(join(dir, 'clair_head_config.json'));
    await expect(detectModelType(dir)).rejects.toThrow('must be a regular file');
    await rm(join(dir, 'clair_head_config.json'), { recursive: true });

    // Only clair_head_config.json present
    await writeFile(join(dir, 'clair_head_config.json'), '{}');
    await expect(detectModelType(dir)).rejects.toThrow(
      'Incomplete Clairic checkpoint: both clair_head_config.json and clair_head.safetensors are required',
    );
    await rm(join(dir, 'clair_head_config.json'));

    // Only clair_head.safetensors present
    await writeFile(join(dir, 'clair_head.safetensors'), 'fixture');
    await expect(detectModelType(dir)).rejects.toThrow(
      'Incomplete Clairic checkpoint: both clair_head_config.json and clair_head.safetensors are required',
    );
  });

  it('fails closed when both CLEF and Clairic head assets exist in the same checkpoint', async () => {
    const root = await mkdtemp(join(tmpdir(), 'clairic-ambiguous-'));
    dirs.push(root);
    const dir = await makeQwen35Dir(root, 'ambiguous', {
      model_type: 'qwen3_5',
      text_config: { hidden_size: 4096 },
    });

    await writeFile(join(dir, 'joint_head_config.json'), '{}');
    await writeFile(join(dir, 'joint_head.safetensors'), 'fixture');
    await writeFile(join(dir, 'clair_head_config.json'), '{}');
    await writeFile(join(dir, 'clair_head.safetensors'), 'fixture');

    await expect(detectModelType(dir)).rejects.toThrow('Ambiguous checkpoint: both CLEF and Clairic head assets found');
  });

  it('promotes a Qwen3.5 GGUF checkpoint to clairic when sibling clair_head assets are present', async () => {
    const root = await mkdtemp(join(tmpdir(), 'clairic-gguf-detect-'));
    dirs.push(root);
    const ggufPath = join(root, 'Clairic-Flash-UD-Q4_K_XL.gguf');

    // Minimal valid GGUF v3 header with general.architecture = "qwen35"
    const archKey = Buffer.from('general.architecture', 'utf8');
    const archVal = Buffer.from('qwen35', 'utf8');
    const header = Buffer.alloc(4 + 4 + 8 + 8 + 8 + archKey.length + 4 + 8 + archVal.length);
    let off = 0;
    header.write('GGUF', off, 'ascii');
    off += 4;
    header.writeUInt32LE(3, off); // version 3
    off += 4;
    header.writeBigUInt64LE(0n, off); // tensor_count = 0
    off += 8;
    header.writeBigUInt64LE(1n, off); // kv_count = 1
    off += 8;
    header.writeBigUInt64LE(BigInt(archKey.length), off);
    off += 8;
    archKey.copy(header, off);
    off += archKey.length;
    header.writeUInt32LE(8, off); // GGUF_TYPE_STRING = 8
    off += 4;
    header.writeBigUInt64LE(BigInt(archVal.length), off);
    off += 8;
    archVal.copy(header, off);

    await writeFile(ggufPath, header);

    // Plain Qwen3.5 GGUF without clair_head stays qwen3_5
    expect(await detectModelType(ggufPath)).toBe('qwen3_5');

    // Incomplete Clairic sidecar fails closed
    await writeFile(join(root, 'clair_head_config.json'), '{}');
    await expect(detectModelType(ggufPath)).rejects.toThrow(
      'Incomplete Clairic checkpoint: both clair_head_config.json and clair_head.safetensors are required',
    );

    // Complete Clairic sidecar promotes GGUF to clairic
    await writeFile(join(root, 'clair_head.safetensors'), 'fixture');
    expect(await detectModelType(ggufPath)).toBe('clairic');

    // Excluded from default chat discovery, discovered with includeDecisions: true
    expect(await discoverLocalChatModels(root)).toEqual([]);
    expect(await discoverLocalChatModels(root, { includeDecisions: true })).toMatchObject([
      {
        path: ggufPath,
        modelType: 'clairic',
        format: 'gguf',
        contextWindow: 65536,
        supportsImages: false,
      },
    ]);

    // Sibling mmproj-*.gguf enables supportsImages for GGUF Clairic checkpoints
    await writeFile(join(root, 'mmproj-BF16.gguf'), 'fixture');
    expect(await discoverLocalChatModels(root, { includeDecisions: true })).toMatchObject([
      {
        path: ggufPath,
        modelType: 'clairic',
        format: 'gguf',
        contextWindow: 65536,
        supportsImages: true,
      },
    ]);
  });
});
