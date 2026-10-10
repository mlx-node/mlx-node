import { mkdtemp, writeFile, rm, mkdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, expect, it } from 'vite-plus/test';

import { detectModelType } from '../../packages/lm/src/model-detection.js';
import { discoverLocalChatModels } from '../../packages/lm/src/model-discovery.js';
const dirs: string[] = [];
afterEach(async () => {
  await Promise.all(dirs.splice(0).map((d) => rm(d, { recursive: true, force: true })));
});
it('recognizes CLEF assets, fails closed on an incomplete head, and separates chat discovery', async () => {
  const root = await mkdtemp(join(tmpdir(), 'clef-detection-'));
  dirs.push(root);
  const dir = join(root, 'flash');
  await mkdir(dir);
  await writeFile(
    join(dir, 'config.json'),
    JSON.stringify({ model_type: 'qwen3_5', text_config: { hidden_size: 4096, max_position_embeddings: 262144 } }),
  );
  await writeFile(join(dir, 'model.safetensors'), 'fixture');
  expect(await detectModelType(dir)).toBe('qwen3_5');
  await mkdir(join(dir, 'joint_head_config.json'));
  await expect(detectModelType(dir)).rejects.toThrow('must be a regular file');
  await rm(join(dir, 'joint_head_config.json'), { recursive: true });
  await writeFile(join(dir, 'joint_head_config.json'), '{}');
  await expect(detectModelType(dir)).rejects.toThrow('Incomplete CLEF');
  await writeFile(join(dir, 'joint_head.safetensors'), 'fixture');
  expect(await detectModelType(dir)).toBe('clef');
  expect(await discoverLocalChatModels(root)).toEqual([]);
  expect(await discoverLocalChatModels(root, { includeDecisions: true })).toMatchObject([
    { name: 'flash', modelType: 'clef', supportsImages: false, contextWindow: 16384 },
  ]);
});
