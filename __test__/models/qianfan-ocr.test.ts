import { QianfanOCRModel, createQianfanOcrConfig } from '@mlx-node/core';
/**
 * Qianfan-OCR NAPI Binding Smoke Tests
 *
 * Minimal tests to verify NAPI bindings work correctly.
 * Full unit tests are in Rust: crates/mlx-core/src/models/qianfan_ocr/
 */
import { describe, expect, it } from 'vite-plus/test';

describe('Qianfan-OCR NAPI Bindings', () => {
  it('should create config via factory function', () => {
    const config = createQianfanOcrConfig();
    expect(config.modelType).toBe('internvl_chat');
    expect(config.imgContextTokenId).toBe(151671);
    expect(config.imgStartTokenId).toBe(151669);
    expect(config.imgEndTokenId).toBe(151670);
    expect(config.eosTokenId).toBe(151645);
  });

  it('should have correct vision config defaults', () => {
    const config = createQianfanOcrConfig();
    expect(config.visionConfig.hiddenSize).toBe(1024);
    expect(config.visionConfig.intermediateSize).toBe(4096);
    expect(config.visionConfig.numHiddenLayers).toBe(24);
    expect(config.visionConfig.numAttentionHeads).toBe(16);
    expect(config.visionConfig.imageSize).toBe(448);
    expect(config.visionConfig.patchSize).toBe(14);
    expect(config.visionConfig.qkvBias).toBe(true);
  });

  it('should have correct LLM config defaults', () => {
    const config = createQianfanOcrConfig();
    expect(config.llmConfig.hiddenSize).toBe(2560);
    expect(config.llmConfig.numHiddenLayers).toBe(36);
    expect(config.llmConfig.numAttentionHeads).toBe(32);
    expect(config.llmConfig.numKeyValueHeads).toBe(8);
    expect(config.llmConfig.headDim).toBe(128);
    expect(config.llmConfig.vocabSize).toBe(153678);
    expect(config.llmConfig.useQkNorm).toBe(true);
  });

  it('should have correct image settings defaults', () => {
    const config = createQianfanOcrConfig();
    expect(config.selectLayer).toBe(-1);
    expect(config.psVersion).toBe('v2');
    expect(config.downsampleRatio).toBe(0.5);
    expect(config.dynamicImageSize).toBe(true);
    expect(config.useThumbnail).toBe(true);
    expect(config.maxDynamicPatch).toBe(12);
    expect(config.minDynamicPatch).toBe(1);
  });

  it('should construct model from config', () => {
    const config = createQianfanOcrConfig();
    const model = new QianfanOCRModel(config);
    expect(model.isInitialized).toBe(false);
  });
});
