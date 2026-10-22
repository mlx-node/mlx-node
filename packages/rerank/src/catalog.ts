import { readdir, readFile, stat } from 'node:fs/promises';
import { join } from 'node:path';

import { rerankerFamily } from './families.js';
import type { RerankerFamily } from './types.js';
export { rerankerFamilies, rerankerFamily } from './families.js';
export const rerankCapabilities = Object.freeze({ rerank: true, text: true, streaming: false });
export interface DiscoveredReranker {
  name: string;
  path: string;
  modelType: RerankerFamily;
  kind: 'rerank';
}
export function isRerankerConfig(config: unknown): boolean {
  return rerankerFamily(config) !== undefined;
}
export async function discoverRerankers(
  root: string,
  onFailure?: (error: unknown) => void,
): Promise<DiscoveredReranker[]> {
  let entries;
  try {
    entries = await readdir(root, { withFileTypes: true });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') onFailure?.(error);
    return [];
  }
  const found: DiscoveredReranker[] = [];
  for (const entry of entries) {
    if (!entry.isDirectory() && !entry.isSymbolicLink()) continue;
    const path = join(root, entry.name);
    try {
      const config = JSON.parse(await readFile(join(path, 'config.json'), 'utf8'));
      const family = rerankerFamily(config);
      if (!family) continue;
      const files = await Promise.all(['tokenizer.json', 'tokenizer_config.json'].map((f) => stat(join(path, f))));
      if (files.some((f) => !f.isFile())) throw new Error(`Invalid reranker tokenizer files: ${path}`);
      const weights = await Promise.all(
        ['model.safetensors', 'weights.safetensors', 'model.safetensors.index.json'].map(async (f) => {
          try {
            return (await stat(join(path, f))).isFile();
          } catch (error) {
            if ((error as NodeJS.ErrnoException).code === 'ENOENT') return false;
            throw error;
          }
        }),
      );
      if (!weights.some(Boolean)) throw new Error(`Missing reranker checkpoint: ${path}`);
      found.push({ name: entry.name, path, modelType: family, kind: 'rerank' });
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') onFailure?.(error);
    }
  }
  return found.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
}
