import type { Lfm2Config } from '@mlx-node/core';

export const LFM2_CONFIGS: { [key: string]: Lfm2Config } = {
  'lfm2.5-1.2b-thinking': {
    vocabSize: 65536,
    hiddenSize: 2048,
    numHiddenLayers: 16,
    numAttentionHeads: 32,
    numKeyValueHeads: 8,
    maxPositionEmbeddings: 128000,
    normEps: 1e-5,
    convBias: false,
    convLCache: 3,
    blockDim: 2048,
    blockFfDim: 12288,
    blockMultipleOf: 256,
    blockFfnDimMultiplier: 1.0,
    blockAutoAdjustFfDim: true,
    ropeTheta: 1000000.0,
    layerTypes: [
      'conv',
      'conv',
      'full_attention',
      'conv',
      'conv',
      'full_attention',
      'conv',
      'conv',
      'full_attention',
      'conv',
      'full_attention',
      'conv',
      'full_attention',
      'conv',
      'full_attention',
      'conv',
    ],
    tieEmbedding: true,
    eosTokenId: 7,
    bosTokenId: 1,
    padTokenId: 0,
  },
};

export function getLfm2Config(name: string): Lfm2Config {
  const config = LFM2_CONFIGS[name];
  if (!config) {
    throw new Error(`Unknown LFM2 config: ${name}. Available: ${Object.keys(LFM2_CONFIGS).join(', ')}`);
  }
  return config;
}
