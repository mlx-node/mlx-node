/**
 * PaddleOCR-VL Model Configurations
 *
 * Based on the PaddleOCR-VL-1.5 model architecture.
 */

export interface VisionConfig {
  modelType: 'paddleocr_vl';
  hiddenSize: number;
  intermediateSize: number;
  numHiddenLayers: number;
  numAttentionHeads: number;
  numChannels: number;
  imageSize: number;
  patchSize: number;
  hiddenAct: string;
  layerNormEps: number;
  attentionDropout: number;
  spatialMergeSize: number;
}

export interface TextConfig {
  modelType: 'paddleocr_vl';
  hiddenSize: number;
  numHiddenLayers: number;
  intermediateSize: number;
  numAttentionHeads: number;
  rmsNormEps: number;
  vocabSize: number;
  numKeyValueHeads: number;
  maxPositionEmbeddings: number;
  ropeTheta: number;
  ropeTraditional: boolean;
  useBias: boolean;
  headDim: number;
  mropeSection: [number, number, number];
}

export interface PaddleOCRVLConfig {
  visionConfig: VisionConfig;
  textConfig: TextConfig;
  modelType: 'paddleocr_vl';
  ignoreIndex: number;
  imageTokenId: number;
  videoTokenId: number;
  visionStartTokenId: number;
  visionEndTokenId: number;
  eosTokenId: number;
}

/**
 * PaddleOCR-VL-1.5 default configuration
 */
const PADDLEOCR_VL_1_5_CONFIG: PaddleOCRVLConfig = {
  modelType: 'paddleocr_vl',
  ignoreIndex: -100,
  imageTokenId: 100295,
  videoTokenId: 100296,
  visionStartTokenId: 101305,
  visionEndTokenId: 101306,
  eosTokenId: 2,
  visionConfig: {
    modelType: 'paddleocr_vl',
    hiddenSize: 1152,
    intermediateSize: 4304,
    numHiddenLayers: 27,
    numAttentionHeads: 16,
    numChannels: 3,
    imageSize: 384,
    patchSize: 14,
    hiddenAct: 'gelu_pytorch_tanh',
    layerNormEps: 1e-6,
    attentionDropout: 0.0,
    spatialMergeSize: 2,
  },
  textConfig: {
    modelType: 'paddleocr_vl',
    hiddenSize: 1024,
    numHiddenLayers: 18,
    intermediateSize: 3072,
    numAttentionHeads: 16,
    rmsNormEps: 1e-5,
    vocabSize: 103424,
    numKeyValueHeads: 2,
    maxPositionEmbeddings: 131072,
    ropeTheta: 500000.0,
    ropeTraditional: false,
    useBias: false,
    headDim: 128,
    mropeSection: [16, 24, 24],
  },
};

/**
 * Available PaddleOCR-VL configurations
 */
export const PADDLEOCR_VL_CONFIGS = {
  'paddleocr-vl-1.5': PADDLEOCR_VL_1_5_CONFIG,
} as const;
