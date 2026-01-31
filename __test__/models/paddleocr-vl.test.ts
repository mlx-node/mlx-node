/**
 * PaddleOCR-VL Model Tests
 *
 * Tests for the PaddleOCR-VL vision-language model.
 *
 * Note: Internal component tests (ERNIELanguageModel, PaddleOCRDecoderLayer, MultimodalRoPE)
 * have been migrated to Rust in crates/mlx-core/src/models/paddleocr_vl/language.rs
 *
 * Smart resize and ImageProcessor target size tests are in:
 * crates/mlx-core/src/models/paddleocr_vl/processing.rs
 */
import { describe, expect, it } from 'vite-plus/test';

// Public imports from @mlx-node/vlm
import { VLModel, ImageProcessor, type ModelConfig, createPaddleocrVlConfig } from '@mlx-node/vlm';

describe('PaddleOCR-VL Model', () => {
  describe('Config', () => {
    it('should create default config', () => {
      const config = createPaddleocrVlConfig();

      // Check vision config
      expect(config.visionConfig.hiddenSize).toBe(1152);
      expect(config.visionConfig.numHiddenLayers).toBe(27);
      expect(config.visionConfig.numAttentionHeads).toBe(16);
      expect(config.visionConfig.patchSize).toBe(14);

      // Check text config
      expect(config.textConfig.hiddenSize).toBe(1024);
      expect(config.textConfig.numHiddenLayers).toBe(18);
      expect(config.textConfig.numAttentionHeads).toBe(16);

      // Check special tokens
      expect(config.imageTokenId).toBe(100295);
      expect(config.eosTokenId).toBe(2);
    });

    it('should have correct model type', () => {
      const config = createPaddleocrVlConfig();
      expect(config.modelType).toBe('paddleocr_vl');
    });

    it('should have correct mRoPE section', () => {
      const config = createPaddleocrVlConfig();
      // mRoPE section should sum to head_dim/2 * 2 = head_dim
      // [16, 24, 24] * 2 = [32, 48, 48] -> total = 128 = head_dim
      expect(config.textConfig.mropeSection).toEqual([16, 24, 24]);
      const total = config.textConfig.mropeSection.reduce((a, b) => a + b * 2, 0);
      expect(total).toBe(config.textConfig.headDim);
    });
  });

  describe('Model Creation', () => {
    it('should create model from config', () => {
      const config: ModelConfig = createPaddleocrVlConfig();
      const model = new VLModel(config);

      expect(model.isInitialized).toBe(false);
    });

    it('should report initialization status correctly', () => {
      const config: ModelConfig = createPaddleocrVlConfig();
      const model = new VLModel(config);

      expect(model.isInitialized).toBe(false);
      expect(model.config.imageTokenId).toBe(100295);
      expect(model.config.modelType).toBe('paddleocr_vl');
    });
  });

  // Note: Multimodal RoPE tests have been migrated to Rust in language.rs:
  // - test_mrope_section_sum
  // - test_mrope_forward_output_shapes
  // - test_mrope_invalid_section_length
  // - test_mrope_getter
  // - test_apply_mrope_output_shapes

  // Note: Smart Resize tests are in Rust in processing.rs:
  // - test_smart_resize_normal
  // - test_smart_resize_too_small
  // - test_smart_resize_too_large
  // - test_smart_resize_aspect_ratio
  // - test_smart_resize_divisibility

  describe('Image Processor', () => {
    it('should create processor with default config', () => {
      const processor = new ImageProcessor();
      expect(processor.resizeFactor).toBe(28); // 14 * 2
    });

    it('should create processor with custom config', () => {
      const processor = new ImageProcessor({
        patchSize: 16,
        mergeSize: 2,
        minPixels: 100000,
        maxPixels: 3000000,
        temporalPatchSize: 1,
        imageMean: [0.48, 0.48, 0.48],
        imageStd: [0.22, 0.22, 0.22],
        doRescale: true,
        doNormalize: true,
      });
      expect(processor.resizeFactor).toBe(32); // 16 * 2
    });

    // Note: getTargetSize test migrated to Rust in processing.rs:
    // - test_image_processor
  });

  // Note: ERNIE Language Model tests have been migrated to Rust in language.rs:
  // - test_ernie_language_model_creation
  // - test_ernie_language_model_add_layers
  // - test_ernie_language_model_kv_cache_management
  // - test_ernie_language_model_get_embeddings
  // - test_ernie_language_model_position_state
  // - test_decoder_layer_creation
  // - test_attention_creation
});
