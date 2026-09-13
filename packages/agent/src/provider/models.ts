/**
 * Local model discovery for the mlx pi provider.
 *
 * Uses the same native-free inventory as the inference host and setup UI,
 * pairing every discovered checkpoint
 * with a pi `ProviderModelConfig` entry ready for
 * `pi.registerProvider('mlx', { models })`.
 *
 * `contextWindow` starts as the checkpoint's trained window, read from the model dir's
 * `config.json` `max_position_embeddings` (root first, then the
 * `text_config` nesting used by qwen3_5 / qwen3_5_moe / gemma4 unified
 * checkpoints). Once a Qwen or Muse-Glimmer model loads, the provider narrows
 * this shared model metadata to the physical paged-cache window so pi's later
 * auto-compaction thresholds match reality. When both config fields are absent
 * the per-family fallback documented on `FamilyTraits` (`@mlx-node/lm`
 * family-data) applies.
 */

import type { ProviderModelConfig } from '@earendil-works/pi-coding-agent';
import { discoverLocalChatModels } from '@mlx-node/lm/model-discovery';

import type { DiscoveredModelLike } from '../types.js';

/** A discovered local checkpoint paired with its pi provider model entry. */
export interface MlxModelInfo {
  discovered: DiscoveredModelLike;
  piModel: ProviderModelConfig;
}

/** Pair the shared local inventory with pi provider metadata, without loading weights. */
export async function discoverMlxModels(modelsDir: string): Promise<MlxModelInfo[]> {
  return (await discoverLocalChatModels(modelsDir)).map(
    ({ name, path, modelType, preset, traits, supportsImages, contextWindow }) => ({
      discovered: { name, path, modelType },
      piModel: {
        id: name,
        name,
        reasoning: traits.reasoning,
        // pi types are agent-only; keep the shared structural map assignable.
        thinkingLevelMap: traits.thinkingLevelMap satisfies ProviderModelConfig['thinkingLevelMap'],
        input: supportsImages ? ['text', 'image'] : ['text'],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow,
        maxTokens: preset.maxOutputTokens,
      },
    }),
  );
}
