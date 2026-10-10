/** Discover locally-downloaded generative models under a given directory. */

import type { LaunchPreset, ModelType } from '@mlx-node/lm/family-data';
import { discoverLocalChatModels, type DiscoveryScanOptions } from '@mlx-node/lm/model-discovery';

/** A locally-downloaded model paired with its sampling preset. */
export interface DiscoveredModel {
  name: string;
  path: string;
  modelType: ModelType;
  preset: LaunchPreset;
  contextWindow: number;
  supportsImages: boolean;
}

/**
 * Use the same checkpoint IDs and paths as the agent and setup UI, including
 * supported GGUF files and their quant variants. No weights are loaded here.
 */
export async function discoverModels(dir: string, opts?: DiscoveryScanOptions): Promise<DiscoveredModel[]> {
  return (await discoverLocalChatModels(dir, { ...opts, includeDecisions: true })).map(
    ({ name, path, modelType, preset, contextWindow, supportsImages }) => ({
      name,
      path,
      modelType,
      preset,
      contextWindow,
      supportsImages,
    }),
  );
}
