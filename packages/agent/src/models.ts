/**
 * Local model discovery for the agent's model inventory.
 *
 * Uses the same native-free inventory as the inference host and setup UI,
 * pairing every discovered checkpoint with an {@link AgentModelConfig} entry
 * describing its chat capabilities.
 *
 * `contextWindow` starts as the checkpoint's trained window, read from the model dir's
 * `config.json` `max_position_embeddings` (root first, then the
 * `text_config` nesting used by qwen3_5 / qwen3_5_moe / gemma4 unified
 * checkpoints). When both config fields are absent the per-family fallback
 * documented on `FamilyTraits` (`@mlx-node/lm` family-data) applies.
 */

import type { FamilyThinkingLevelMap } from '@mlx-node/lm';
import { discoverLocalChatModels } from '@mlx-node/lm/model-discovery';

import type { DiscoveredModelLike } from './types.js';

/** Chat-model metadata paired with a discovered checkpoint (local, zero cost). */
export interface AgentModelConfig {
  id: string;
  name: string;
  type: 'chat';
  reasoning: boolean;
  thinkingLevelMap?: FamilyThinkingLevelMap;
  input: ('text' | 'image')[];
  cost: { input: number; output: number; cacheRead: number; cacheWrite: number };
  contextWindow: number;
  maxTokens: number;
}

/** A discovered local checkpoint paired with its agent model entry. */
export interface MlxModelInfo {
  discovered: DiscoveredModelLike;
  agentModel: AgentModelConfig;
}

/** Pair the shared local inventory with agent model metadata, without loading weights. */
export async function discoverMlxModels(modelsDir: string): Promise<MlxModelInfo[]> {
  return (await discoverLocalChatModels(modelsDir)).map(
    ({ name, path, modelType, preset, traits, supportsImages, contextWindow }) => ({
      discovered: { name, path, modelType },
      agentModel: {
        id: name,
        name,
        type: 'chat',
        reasoning: traits.reasoning,
        thinkingLevelMap: traits.thinkingLevelMap,
        input: supportsImages ? ['text', 'image'] : ['text'],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow,
        maxTokens: preset.maxOutputTokens,
      },
    }),
  );
}
