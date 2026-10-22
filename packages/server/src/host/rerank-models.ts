import { rerankCapabilities, type DiscoveredReranker } from '@mlx-node/rerank/catalog';
import type { LoadRerankerOptions, RerankerModel } from '@mlx-node/rerank/types';

import type { ModelRegistry } from '../registry.js';
import type { CapabilityAdapter } from './capability-models.js';
export async function loadRerankModel(
  entry: DiscoveredReranker,
  options?: LoadRerankerOptions,
): Promise<RerankerModel> {
  const { loadReranker } = await import('@mlx-node/rerank');
  return loadReranker(entry.path, options);
}
export function rerankCapability(
  entries: DiscoveredReranker[],
  registry: ModelRegistry,
  loader: (entry: DiscoveredReranker) => Promise<RerankerModel>,
): CapabilityAdapter {
  const created = Math.floor(Date.now() / 1000);
  return {
    kind: 'rerank',
    listModels: () =>
      entries.map((e) => ({
        id: e.name,
        object: 'model',
        created,
        owned_by: 'mlx-node',
        modality: 'text',
        task: 'rerank',
        model_family: e.modelType,
        capabilities: rerankCapabilities,
      })),
    hasResident: (name) => !!registry.rerank.get(name),
    disposeAll: () => registry.rerank.disposeAll(),
    async resolveModel(name) {
      if (registry.rerank.get(name)) return;
      const entry = entries.find((e) => e.name === name);
      if (!entry) throw new Error(`Unknown reranker ${name}`);
      await registry.rerank.disposeAll();
      const model = await loader(entry);
      try {
        if (model.info.family !== entry.modelType) throw new Error('Loader returned an incompatible reranker');
        registry.rerank.register(name, model);
      } catch (error) {
        await model.dispose();
        throw error;
      }
    },
  };
}
