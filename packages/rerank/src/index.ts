import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

import { JinaRerankerModel, XlmRobertaRerankerModel } from '@mlx-node/core';

import { rerankerFamily } from './catalog.js';
import { RerankError, rerankError, validateRerankerPolicy } from './protocol.js';
import { wrapReranker } from './runtime.js';
import type { LoadRerankerOptions, RerankerModel } from './types.js';
export type * from './types.js';
export { RerankError } from './protocol.js';
export async function loadReranker(modelPath: string, options: LoadRerankerOptions = {}): Promise<RerankerModel> {
  try {
    const config = JSON.parse(await readFile(join(modelPath, 'config.json'), 'utf8'));
    const family = rerankerFamily(config);
    if (!family) throw new RerankError('invalid_model', 'Unsupported or ambiguous reranker architecture', 500);
    const policy = validateRerankerPolicy(options.policy, family);
    const loaders = { jina_reranker: JinaRerankerModel, xlm_roberta_reranker: XlmRobertaRerankerModel };
    return wrapReranker(await loaders[family].load(modelPath, JSON.stringify(policy)));
  } catch (error) {
    throw rerankError(error);
  }
}
