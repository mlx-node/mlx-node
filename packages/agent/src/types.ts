import type { ModelType } from '@mlx-node/lm';

/** Structural mirror of the CLI's DiscoveredModel — avoids a cli↔agent dependency cycle. */
export interface DiscoveredModelLike {
  name: string;
  path: string;
  modelType: ModelType;
}
